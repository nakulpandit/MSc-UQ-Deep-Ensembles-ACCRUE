"""Differentiable ACCRUE loss and variance network from Camporeale & Carè (2021)."""

from dataclasses import dataclass
import math

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install the project dependencies to use neural models: pip install -e ."
    ) from exc


def gaussian_crps_values(errors, sigma):
    z = errors / sigma
    return sigma * (
        z * torch.erf(z / math.sqrt(2.0))
        + math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * z.square())
        - 1.0 / math.sqrt(math.pi)
    )


def reliability_score(errors, sigma):
    """Analytic Gaussian reliability score (ACCRUE Eq. 8)."""
    eta = errors / (math.sqrt(2.0) * sigma)
    eta, _ = torch.sort(eta.reshape(-1))
    n = eta.numel()
    ranks = torch.arange(1, n + 1, dtype=eta.dtype, device=eta.device)
    terms = (
        eta * (torch.erf(eta) + 1.0) / n
        - eta * (2.0 * ranks - 1.0) / (n * n)
        + torch.exp(-eta.square()) / (math.sqrt(math.pi) * n)
    )
    return terms.sum() - 0.5 * math.sqrt(2.0 / math.pi)


def accrue_weight(errors) -> float:
    """Paper heuristic beta using the attainable scales of CRPS and RS."""
    errors = errors.detach().reshape(-1)
    crps_min = math.sqrt(math.log(4.0)) / 2.0 * torch.mean(torch.abs(errors))
    # Practical approximation stated after Eq. 13 in the paper.
    rs_min = 0.5 * math.sqrt(2.0 / math.pi)
    return float(rs_min / (crps_min + rs_min + 1e-12))


def accrue_loss(errors, sigma, beta: float | None = None):
    if beta is None:
        beta = accrue_weight(errors)
    crps = gaussian_crps_values(errors, sigma).mean()
    rs = reliability_score(errors, sigma)
    return beta * crps + (1.0 - beta) * rs, crps, rs


class SymmetricSaturatingLinear(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-1.0, max=1.0)


class VarianceNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers=(50, 10), sigma_floor: float = 1e-6):
        super().__init__()
        if len(hidden_layers) != 2:
            raise ValueError("paper-style ACCRUE expects two hidden layers")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            SymmetricSaturatingLinear(),
            nn.Linear(hidden_layers[1], 1),
        )
        self.sigma_floor = sigma_floor

    def forward(self, x):
        log_sigma = self.layers(x).reshape(-1)
        return torch.exp(log_sigma).clamp_min(self.sigma_floor)


@dataclass
class AccrueConfig:
    hidden_layers: tuple[int, int] = (50, 10)
    restarts: int = 5
    max_epochs: int = 100
    lbfgs_iterations_per_epoch: int = 5
    patience: int = 10
    learning_rate: float = 0.5
    sigma_floor: float = 1e-6
    min_delta: float = 1e-8


class AccrueVarianceRegressor:
    """Fit sigma(x) from inputs and fixed-oracle residuals."""

    def __init__(self, input_dim: int, config: AccrueConfig | None = None, seed: int = 0):
        self.input_dim = input_dim
        self.config = config or AccrueConfig()
        self.seed = seed
        self.model: VarianceNetwork | None = None
        self.best_validation_score_: float | None = None
        self.best_restart_: int | None = None
        self.epochs_trained_: int | None = None
        self.restart_history_: list[dict[str, float | int]] = []

    def fit(
        self,
        x_train: np.ndarray,
        residual_train: np.ndarray,
        x_validation: np.ndarray | None = None,
        residual_validation: np.ndarray | None = None,
    ):
        x = torch.as_tensor(x_train, dtype=torch.float64)
        errors = torch.as_tensor(residual_train, dtype=torch.float64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x_train must be a two-dimensional array with input_dim columns")
        if x.shape[0] != errors.shape[0] or x.shape[0] < 2:
            raise ValueError("x_train and residual_train must have equal length of at least two")

        has_validation = x_validation is not None and residual_validation is not None
        if (x_validation is None) != (residual_validation is None):
            raise ValueError("provide both validation inputs and residuals, or neither")
        if has_validation:
            x_val = torch.as_tensor(x_validation, dtype=torch.float64)
            e_val = torch.as_tensor(residual_validation, dtype=torch.float64).reshape(-1)
            if x_val.ndim != 2 or x_val.shape[1] != self.input_dim:
                raise ValueError("x_validation has the wrong shape")
            if x_val.shape[0] != e_val.shape[0] or x_val.shape[0] < 2:
                raise ValueError("validation inputs and residuals must have equal length")
            validation_beta = accrue_weight(e_val)

        training_beta = accrue_weight(errors)
        best_state = None
        best_score = float("inf")
        best_restart = None
        best_epochs = None
        self.restart_history_ = []

        for restart in range(self.config.restarts):
            torch.manual_seed(self.seed + restart)
            model = VarianceNetwork(
                self.input_dim, self.config.hidden_layers, self.config.sigma_floor
            ).double()
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=self.config.learning_rate,
                max_iter=self.config.lbfgs_iterations_per_epoch,
                tolerance_grad=1e-10,
                tolerance_change=1e-12,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()
                sigma = model(x)
                loss, _, _ = accrue_loss(errors, sigma, training_beta)
                loss.backward()
                return loss

            restart_best_score = float("inf")
            restart_best_state = None
            restart_best_epoch = None
            epochs_without_improvement = 0
            epochs_completed = 0
            for epoch in range(self.config.max_epochs):
                try:
                    optimizer.step(closure)
                except RuntimeError:
                    break
                epochs_completed = epoch + 1
                with torch.no_grad():
                    if has_validation:
                        score_tensor = accrue_loss(e_val, model(x_val), validation_beta)[0]
                    else:
                        score_tensor = accrue_loss(errors, model(x), training_beta)[0]
                    score = float(score_tensor)

                if not math.isfinite(score):
                    break
                if score < restart_best_score - self.config.min_delta:
                    restart_best_score = score
                    restart_best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                    restart_best_epoch = epochs_completed
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.config.patience:
                        break

            self.restart_history_.append(
                {
                    "restart": restart,
                    "validation_score": restart_best_score,
                    "epochs": epochs_completed,
                }
            )
            if restart_best_state is not None and restart_best_score < best_score:
                best_score = restart_best_score
                best_state = restart_best_state
                best_restart = restart
                best_epochs = restart_best_epoch

        if best_state is None:
            raise RuntimeError("all ACCRUE neural restarts failed to produce a finite model")

        self.model = VarianceNetwork(
            self.input_dim, self.config.hidden_layers, self.config.sigma_floor
        ).double()
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_validation_score_ = best_score
        self.best_restart_ = best_restart
        self.epochs_trained_ = best_epochs
        return self

    @torch.no_grad()
    def predict_sigma(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit must be called before predict_sigma")
        x_tensor = torch.as_tensor(x, dtype=torch.float64)
        return self.model(x_tensor).cpu().numpy()
