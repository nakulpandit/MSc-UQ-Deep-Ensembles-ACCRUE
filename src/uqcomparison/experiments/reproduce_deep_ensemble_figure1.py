"""Reproduce the Deep Ensembles paper's Figure 1 cubic regression task."""

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uqcomparison.metrics import gaussian_nll, rmse


@dataclass(frozen=True)
class Figure1Mode:
    runs: int
    members: int
    epochs: int
    learning_rate: float
    adversarial_training: bool
    evaluation_samples: int


MODES = {
    "debug": Figure1Mode(1, 1, 2, 0.1, True, 600),
    "development": Figure1Mode(5, 2, 10, 0.1, True, 2_000),
    "final": Figure1Mode(20, 5, 40, 0.1, True, 6_000),
    "long_diagnostic": Figure1Mode(1, 5, 2_000, 0.001, True, 6_000),
}


def default_output_directory(mode: str) -> Path:
    if mode == "final":
        return Path("results/paper_reproductions/deep_ensemble_figure1")
    return Path(f"results/development/deep_ensemble_figure1_{mode}")


def generate_training_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Twenty observations from y=x^3+Normal(0, 3^2), as in Section 3.2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4.0, 4.0, size=(20, 1))
    y = x[:, 0] ** 3 + rng.normal(0.0, 3.0, size=20)
    return x, y


def generate_evaluation_data(seed: int, count: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6.0, 6.0, size=(count, 1))
    truth = x[:, 0] ** 3
    y = truth + rng.normal(0.0, 3.0, size=count)
    return x, y, truth


def _masked_metrics(prefix, mask, y, truth, mean, aleatoric, epistemic, total):
    error = np.abs(y[mask] - mean[mask])
    result = {
        f"rmse_{prefix}": rmse(y[mask], mean[mask]),
        f"curve_rmse_{prefix}": rmse(truth[mask], mean[mask]),
        f"nll_{prefix}": gaussian_nll(y[mask], mean[mask], total[mask]),
        f"coverage_1sigma_{prefix}": float(np.mean(error <= total[mask])),
        f"coverage_2sigma_{prefix}": float(np.mean(error <= 2.0 * total[mask])),
        f"coverage_3sigma_{prefix}": float(np.mean(error <= 3.0 * total[mask])),
        f"mean_aleatoric_sigma_{prefix}": float(np.mean(aleatoric[mask])),
        f"mean_epistemic_sigma_{prefix}": float(np.mean(epistemic[mask])),
        f"mean_total_sigma_{prefix}": float(np.mean(total[mask])),
    }
    if prefix == "in_domain":
        result["sigma_mae_in_domain"] = float(np.mean(np.abs(total[mask] - 3.0)))
        result["sigma_rmse_in_domain"] = rmse(np.full(mask.sum(), 3.0), total[mask])
    return result


def _plot_figure(x_train, y_train, prediction_frame, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    x = prediction_frame["x"].to_numpy()
    mean = prediction_frame["predicted_mean"].to_numpy()
    total = prediction_frame["total_sigma"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, prediction_frame["true_mean"], color="tab:blue", label="True $x^3$")
    ax.scatter(x_train[:, 0], y_train, color="tab:red", s=28, label="20 observations")
    ax.plot(x, mean, color="black", label="Ensemble mean")
    ax.fill_between(
        x,
        mean - 3.0 * total,
        mean + 3.0 * total,
        color="0.75",
        alpha=0.75,
        label="Mean +/- 3 total sigma",
    )
    ax.axvline(-4.0, color="0.5", linestyle=":", linewidth=1)
    ax.axvline(4.0, color="0.5", linestyle=":", linewidth=1)
    ax.set(
        xlabel="x",
        ylabel="y",
        xlim=(-6.0, 6.0),
        title="Deep Ensemble Figure 1 paper configuration",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _aggregate(frame: pd.DataFrame) -> dict:
    columns = [
        column
        for column in frame.columns
        if column not in {"run", "seed"} and np.issubdtype(frame[column].dtype, np.number)
    ]
    return {
        "metrics_mean": frame[columns].mean().to_dict(),
        "metrics_std": frame[columns].std(ddof=0).to_dict(),
        "metrics_median": frame[columns].median().to_dict(),
        "metrics_iqr": (
            frame[columns].quantile(0.75) - frame[columns].quantile(0.25)
        ).to_dict(),
    }


def run(mode_name: str, output_dir: Path, seed: int = 2026, runs_override=None):
    from uqcomparison.models.deep_ensemble import DeepEnsembleConfig, DeepEnsembleRegressor

    mode = MODES[mode_name]
    runs = mode.runs if runs_override is None else runs_override
    if runs < 1:
        raise ValueError("runs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    first_prediction_frame = None
    first_training_data = None

    for run_index in range(runs):
        run_seed = seed + run_index
        x_train, y_train = generate_training_data(run_seed)
        x_test, y_test, truth = generate_evaluation_data(
            run_seed + 100_000, mode.evaluation_samples
        )
        config = DeepEnsembleConfig(
            members=mode.members,
            hidden_layers=(100,),
            epochs=mode.epochs,
            batch_size=100,
            learning_rate=mode.learning_rate,
            adversarial_training=mode.adversarial_training,
            adversarial_fraction=0.01,
        )
        start = perf_counter()
        model = DeepEnsembleRegressor(1, config=config, seed=run_seed).fit(x_train, y_train)
        training_seconds = perf_counter() - start
        inference_start = perf_counter()
        mean, aleatoric, epistemic, total = model.predict_components(x_test)
        inference_ms = 1_000.0 * (perf_counter() - inference_start) / len(x_test)
        inside = np.abs(x_test[:, 0]) <= 4.0
        outside = ~inside
        row = {
            "run": run_index,
            "seed": run_seed,
            "training_seconds": training_seconds,
            "inference_ms_per_sample": inference_ms,
        }
        row.update(
            _masked_metrics(
                "overall",
                np.ones(len(x_test), dtype=bool),
                y_test,
                truth,
                mean,
                aleatoric,
                epistemic,
                total,
            )
        )
        row.update(
            _masked_metrics(
                "in_domain", inside, y_test, truth, mean, aleatoric, epistemic, total
            )
        )
        row.update(
            _masked_metrics(
                "out_of_domain", outside, y_test, truth, mean, aleatoric, epistemic, total
            )
        )
        rows.append(row)

        if run_index == 0:
            grid = np.linspace(-6.0, 6.0, 800).reshape(-1, 1)
            grid_mean, grid_aleatoric, grid_epistemic, grid_total = (
                model.predict_components(grid)
            )
            first_prediction_frame = pd.DataFrame(
                {
                    "x": grid[:, 0],
                    "true_mean": grid[:, 0] ** 3,
                    "predicted_mean": grid_mean,
                    "aleatoric_sigma": grid_aleatoric,
                    "epistemic_sigma": grid_epistemic,
                    "total_sigma": grid_total,
                }
            )
            first_training_data = (x_train, y_train)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "per_run_metrics.csv", index=False)
    first_prediction_frame.to_csv(output_dir / "figure1_predictions.csv", index=False)
    local_figure = output_dir / "deep_ensemble_figure1_paper_config.png"
    _plot_figure(*first_training_data, first_prediction_frame, local_figure)
    canonical_figure = Path("results/figures/deep_ensemble_figure1_paper_config.png")
    canonical_figure.parent.mkdir(parents=True, exist_ok=True)
    if local_figure.resolve() != canonical_figure.resolve():
        shutil.copyfile(local_figure, canonical_figure)

    summary = {
        "method": "deep_ensemble",
        "experiment": "figure1_cubic_regression",
        "mode": mode_name,
        "runs": runs,
        "base_seed": seed,
        "paper_protocol": {
            "training_samples": 20,
            "data": "x~Uniform(-4,4); y=x^3+Normal(0,3^2)",
            "hidden_layers": [100],
            "activation": "ReLU",
            "members": mode.members,
            "epochs": mode.epochs,
            "batch_size": 100,
            "optimizer": "Adam",
            "learning_rate": mode.learning_rate,
            "adversarial_training": mode.adversarial_training,
            "fgsm_fraction_of_input_range": 0.01,
        },
        "implementation_notes": {
            "paper_run_count_specified": False,
            "evaluation_domain": [-6.0, 6.0],
            "in_domain": [-4.0, 4.0],
            "long_training_is_diagnostic_only": mode_name == "long_diagnostic",
        },
        "output_files": [
            "per_run_metrics.csv",
            "summary.json",
            "figure1_predictions.csv",
            "deep_ensemble_figure1_paper_config.png",
        ],
        "mode_configuration": asdict(mode),
        **_aggregate(results),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), default="debug")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or default_output_directory(args.mode)
    print(json.dumps(run(args.mode, output_dir, args.seed, args.runs), indent=2))


if __name__ == "__main__":
    main()
