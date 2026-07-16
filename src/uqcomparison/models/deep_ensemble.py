"""Gaussian Deep Ensemble regression following Lakshminarayanan et al. (2017)."""

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - depends on optional runtime installation
    raise ImportError(
        "Install the project dependencies to use neural models: pip install -e ."
    ) from exc


class GaussianRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_layers=(50,), variance_floor: float = 1e-6):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_layers:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(width, 2)
        self.variance_floor = variance_floor

    def forward(self, x):
        output = self.output(self.backbone(x))
        mean = output[:, 0]
        variance = F.softplus(output[:, 1]) + self.variance_floor
        return mean, variance


def gaussian_nll_loss(y, mean, variance):
    return 0.5 * torch.mean(torch.log(variance) + (y - mean) ** 2 / variance)


@dataclass
class DeepEnsembleConfig:
    members: int = 5
    hidden_layers: tuple[int, ...] = (50,)
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    variance_floor: float = 1e-6
    adversarial_training: bool = False
    adversarial_fraction: float = 0.01


class DeepEnsembleRegressor:
    def __init__(self, input_dim: int, config: DeepEnsembleConfig | None = None, seed: int = 0):
        self.input_dim = input_dim
        self.config = config or DeepEnsembleConfig()
        self.seed = seed
        self.members: list[GaussianRegressor] = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
        n = len(x_tensor)
        input_range = x_tensor.max(dim=0).values - x_tensor.min(dim=0).values
        adversarial_epsilon = self.config.adversarial_fraction * input_range
        self.members = []

        for member_index in range(self.config.members):
            member_seed = self.seed + member_index
            torch.manual_seed(member_seed)
            model = GaussianRegressor(
                self.input_dim, self.config.hidden_layers, self.config.variance_floor
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            generator = torch.Generator().manual_seed(member_seed)
            for _ in range(self.config.epochs):
                order = torch.randperm(n, generator=generator)
                for start in range(0, n, self.config.batch_size):
                    batch = order[start : start + self.config.batch_size]
                    batch_x = x_tensor[batch]
                    batch_y = y_tensor[batch]
                    mean, variance = model(batch_x)
                    clean_loss = gaussian_nll_loss(batch_y, mean, variance)
                    loss = clean_loss
                    if self.config.adversarial_training:
                        differentiable_x = batch_x.detach().clone().requires_grad_(True)
                        input_mean, input_variance = model(differentiable_x)
                        input_loss = gaussian_nll_loss(
                            batch_y, input_mean, input_variance
                        )
                        gradient_x = torch.autograd.grad(input_loss, differentiable_x)[0]
                        adversarial_x = (
                            batch_x + adversarial_epsilon * gradient_x.sign()
                        ).detach()
                        adversarial_mean, adversarial_variance = model(adversarial_x)
                        loss = clean_loss + gaussian_nll_loss(
                            batch_y, adversarial_mean, adversarial_variance
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            self.members.append(model.eval())
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, _, _, total_sigma = self.predict_components(x)
        return mean, total_sigma

    @torch.no_grad()
    def predict_components(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ensemble mean plus aleatoric, epistemic, and total std."""
        if not self.members:
            raise RuntimeError("fit must be called before predict")
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        predictions = [model(x_tensor) for model in self.members]
        means = torch.stack([item[0] for item in predictions])
        variances = torch.stack([item[1] for item in predictions])
        ensemble_mean = means.mean(dim=0)
        aleatoric_variance = variances.mean(dim=0)
        epistemic_variance = means.square().mean(dim=0) - ensemble_mean.square()
        epistemic_variance = epistemic_variance.clamp_min(0.0)
        ensemble_variance = (aleatoric_variance + epistemic_variance).clamp_min(
            self.config.variance_floor
        )
        return (
            ensemble_mean.numpy(),
            torch.sqrt(aleatoric_variance).numpy(),
            torch.sqrt(epistemic_variance).numpy(),
            torch.sqrt(ensemble_variance).numpy(),
        )
