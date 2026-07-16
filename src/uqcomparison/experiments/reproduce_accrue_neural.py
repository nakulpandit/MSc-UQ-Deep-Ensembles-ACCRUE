"""Paper-style neural ACCRUE reproduction on G, Y, W, and 5D."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from uqcomparison.data import generate_synthetic, synthetic_truth
from uqcomparison.experiments.reproduce_accrue_toys import fit_gp_oracle, paper_split
from uqcomparison.metrics import calibration_error, gaussian_crps, gaussian_nll, rmse


@dataclass(frozen=True)
class NeuralRunMode:
    runs: int
    restarts: int
    max_epochs: int
    lbfgs_iterations_per_epoch: int
    one_dimensional_samples: int
    five_dimensional_samples: int


RUN_MODES = {
    "debug": NeuralRunMode(1, 1, 3, 2, 100, 500),
    "development": NeuralRunMode(5, 2, 20, 3, 100, 2_000),
    "final": NeuralRunMode(100, 5, 100, 5, 100, 10_000),
}


def resolve_mode(name: str) -> NeuralRunMode:
    try:
        return RUN_MODES[name]
    except KeyError as exc:
        raise ValueError(f"unknown mode {name!r}") from exc


def default_output_directory(dataset: str, mode: str) -> Path:
    if mode == "final":
        return Path("results") / "paper_reproductions" / f"accrue_neural_{dataset}"
    return Path("results") / "development" / f"accrue_neural_{dataset}_{mode}"


def _metric_columns():
    return [
        "mean_rmse",
        "sigma_rmse",
        "nll",
        "crps",
        "calibration_error",
        "mean_predicted_sigma",
        "training_seconds",
        "inference_ms_per_sample",
        "best_validation_score",
        "best_restart",
        "epochs_trained",
    ]


def _aggregate(frame: pd.DataFrame) -> dict:
    columns = _metric_columns()
    return {
        "metrics_mean": frame[columns].mean().to_dict(),
        "metrics_std": frame[columns].std(ddof=0).to_dict(),
        "metrics_median": frame[columns].median().to_dict(),
        "metrics_iqr": (
            frame[columns].quantile(0.75) - frame[columns].quantile(0.25)
        ).to_dict(),
    }


def _plot_reliability(pit_values: np.ndarray, dataset: str, output_dir: Path):
    expected = np.sort(pit_values)
    observed = np.arange(1, expected.size + 1) / expected.size
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.4", label="Ideal")
    ax.plot(expected, observed, color="tab:blue", label="Neural ACCRUE")
    ax.set(
        xlabel="Expected cumulative probability",
        ylabel="Observed cumulative probability",
        title=f"{dataset.upper()} neural ACCRUE reliability",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=180)
    plt.close(fig)


def _plot_one_dimensional_recovery(
    grid: np.ndarray,
    true_sigma: np.ndarray,
    predictions: np.ndarray,
    dataset: str,
    output_dir: Path,
):
    mean = predictions.mean(axis=0)
    standard_deviation = predictions.std(axis=0)
    pd.DataFrame(
        {
            "x": grid[:, 0],
            "true_sigma": true_sigma,
            "predicted_sigma_mean": mean,
            "predicted_sigma_std": standard_deviation,
        }
    ).to_csv(output_dir / "sigma_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid[:, 0], true_sigma, color="tab:red", label="True sigma(x)")
    ax.plot(grid[:, 0], mean, color="black", label="Mean neural estimate")
    ax.fill_between(
        grid[:, 0], mean - standard_deviation, mean + standard_deviation,
        color="0.75", alpha=0.7, label="plus/minus 1 run SD",
    )
    ax.fill_between(
        grid[:, 0], mean - 2 * standard_deviation, mean + 2 * standard_deviation,
        color="0.9", alpha=0.7, label="plus/minus 2 run SD",
    )
    ax.set(
        xlabel="x",
        ylabel="standard deviation",
        title=f"Neural ACCRUE {dataset.upper()} recovery",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sigma_recovery.png", dpi=180)
    plt.close(fig)


def _plot_five_dimensional_recovery(
    true_sigma: np.ndarray,
    predicted_sigma: np.ndarray,
    output_dir: Path,
    seed: int,
):
    rng = np.random.default_rng(seed)
    count = min(50_000, true_sigma.size)
    selected = rng.choice(true_sigma.size, size=count, replace=False)
    pairs = pd.DataFrame(
        {
            "true_sigma": true_sigma[selected],
            "predicted_sigma": predicted_sigma[selected],
        }
    )
    pairs.to_csv(output_dir / "sigma_pairs_sample.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot = ax.hexbin(
        pairs["true_sigma"], pairs["predicted_sigma"], gridsize=45,
        mincnt=1, cmap="viridis",
    )
    limits = [
        min(pairs["true_sigma"].min(), pairs["predicted_sigma"].min()),
        max(pairs["true_sigma"].max(), pairs["predicted_sigma"].max()),
    ]
    ax.plot(limits, limits, "--", color="tab:red", label="Perfect recovery")
    ax.set(
        xlabel="True sigma",
        ylabel="Predicted sigma",
        title="Neural ACCRUE 5D uncertainty recovery",
        xlim=limits,
        ylim=limits,
    )
    ax.legend()
    fig.colorbar(plot, ax=ax, label="Test-point count")
    fig.tight_layout()
    fig.savefig(output_dir / "sigma_scatter.png", dpi=180)
    plt.close(fig)


def run(
    dataset: str,
    mode_name: str,
    output_dir: Path,
    seed: int = 2026,
    runs_override: int | None = None,
):
    from uqcomparison.models.accrue import AccrueConfig, AccrueVarianceRegressor

    dataset = dataset.lower()
    mode = resolve_mode(mode_name)
    runs = runs_override if runs_override is not None else mode.runs
    if runs < 1:
        raise ValueError("runs must be positive")
    sample_size = (
        mode.five_dimensional_samples if dataset == "5d" else mode.one_dimensional_samples
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    is_five_dimensional = dataset == "5d"
    if not is_five_dimensional:
        upper = np.pi if dataset == "w" else 1.0
        grid = np.linspace(0.0, upper, 400).reshape(-1, 1)
        _, true_grid_sigma = synthetic_truth(dataset, grid)
        grid_predictions = []
    pooled_true_sigma = []
    pooled_predicted_sigma = []
    pooled_pit = []
    rows = []

    for run_index in range(runs):
        run_seed = seed + run_index
        sample = generate_synthetic(dataset, n_samples=sample_size, seed=run_seed)
        train, validation, test = paper_split(len(sample.y), run_seed)

        if is_five_dimensional:
            oracle_mean = sample.mean.copy()
        else:
            oracle = fit_gp_oracle(sample.x[train], sample.y[train], run_seed)
            oracle_mean = oracle.predict(sample.x)
        residuals = sample.y - oracle_mean

        config = AccrueConfig(
            restarts=mode.restarts,
            max_epochs=mode.max_epochs,
            lbfgs_iterations_per_epoch=mode.lbfgs_iterations_per_epoch,
            patience=10,
        )
        estimator = AccrueVarianceRegressor(sample.x.shape[1], config, seed=run_seed)
        training_start = perf_counter()
        estimator.fit(
            sample.x[train], residuals[train], sample.x[validation], residuals[validation]
        )
        training_seconds = perf_counter() - training_start

        inference_start = perf_counter()
        predicted_sigma = estimator.predict_sigma(sample.x[test])
        inference_seconds = perf_counter() - inference_start
        pooled_true_sigma.extend(sample.sigma[test])
        pooled_predicted_sigma.extend(predicted_sigma)
        pooled_pit.extend(ndtr(residuals[test] / predicted_sigma))

        if not is_five_dimensional:
            grid_predictions.append(estimator.predict_sigma(grid))
        rows.append(
            {
                "run": run_index,
                "seed": run_seed,
                "sample_size": sample_size,
                "train_size": len(train),
                "validation_size": len(validation),
                "test_size": len(test),
                "mean_rmse": rmse(sample.y[test], oracle_mean[test]),
                "sigma_rmse": rmse(sample.sigma[test], predicted_sigma),
                "nll": gaussian_nll(sample.y[test], oracle_mean[test], predicted_sigma),
                "crps": gaussian_crps(sample.y[test], oracle_mean[test], predicted_sigma),
                "calibration_error": calibration_error(
                    sample.y[test], oracle_mean[test], predicted_sigma
                ),
                "mean_predicted_sigma": float(np.mean(predicted_sigma)),
                "training_seconds": training_seconds,
                "inference_ms_per_sample": 1_000 * inference_seconds / len(test),
                "best_validation_score": estimator.best_validation_score_,
                "best_restart": estimator.best_restart_,
                "epochs_trained": estimator.epochs_trained_,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "per_run_metrics.csv", index=False)
    plot_files = ["reliability_diagram.png"]
    if is_five_dimensional:
        _plot_five_dimensional_recovery(
            np.asarray(pooled_true_sigma), np.asarray(pooled_predicted_sigma), output_dir, seed
        )
        plot_files.extend(["sigma_scatter.png", "sigma_pairs_sample.csv"])
    else:
        _plot_one_dimensional_recovery(
            grid, true_grid_sigma, np.asarray(grid_predictions), dataset, output_dir
        )
        plot_files.extend(["sigma_recovery.png", "sigma_curve.csv"])
    _plot_reliability(np.asarray(pooled_pit), dataset, output_dir)

    summary = {
        "method": "neural_accrue",
        "dataset": dataset,
        "mode": mode_name,
        "runs": runs,
        "base_seed": seed,
        "sample_size": sample_size,
        "split": {"train": 0.33, "validation": 0.33, "test": 0.34},
        "mean_oracle": "exact_zero" if is_five_dimensional else "homoskedastic_gp",
        "architecture": {"hidden_layers": [50, 10], "output": "log_sigma"},
        "optimizer": {
            "name": "LBFGS",
            "restarts": mode.restarts,
            "max_epochs": mode.max_epochs,
            "iterations_per_epoch": mode.lbfgs_iterations_per_epoch,
            "validation_patience": 10,
        },
        "output_files": ["per_run_metrics.csv", "summary.json", *plot_files],
        **_aggregate(results),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["g", "y", "w", "5d"], required=True)
    parser.add_argument("--mode", choices=list(RUN_MODES), default="debug")
    parser.add_argument("--runs", type=int, help="override the mode's run count")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or default_output_directory(args.dataset, args.mode)
    summary = run(args.dataset, args.mode, output_dir, args.seed, args.runs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
