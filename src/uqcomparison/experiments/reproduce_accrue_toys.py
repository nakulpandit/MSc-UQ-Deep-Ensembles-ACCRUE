"""Reproduce ACCRUE's one-dimensional synthetic experiments and Figure 4 style plot."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from uqcomparison.data import generate_synthetic, synthetic_truth
from uqcomparison.metrics import calibration_error, gaussian_crps, gaussian_nll, rmse
from uqcomparison.models.accrue_polynomial import PolynomialAccrueRegressor


def paper_split(n_samples: int, seed: int):
    """Create the paper's 33%/33%/34% train/validation/test partition."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_train = int(np.floor(0.33 * n_samples))
    n_validation = int(np.floor(0.33 * n_samples))
    return (
        indices[:n_train],
        indices[n_train : n_train + n_validation],
        indices[n_train + n_validation :],
    )


def fit_gp_oracle(x_train, y_train, seed: int):
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(0.2, (1e-2, 1e2)) + WhiteKernel(
        0.5, (1e-4, 1e1)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=seed,
    ).fit(x_train, y_train)


def reliability_curve(pit: np.ndarray):
    pit = np.sort(np.asarray(pit))
    return pit, np.arange(1, pit.size + 1) / pit.size


def run(dataset: str, runs: int, seed: int, output_dir: Path):
    if dataset not in {"g", "y", "w"}:
        raise ValueError("the polynomial reproduction supports g, y, or w")
    output_dir.mkdir(parents=True, exist_ok=True)
    upper = np.pi if dataset == "w" else 1.0
    grid = np.linspace(0.0, upper, 400).reshape(-1, 1)
    _, true_grid_sigma = synthetic_truth(dataset, grid)

    rows = []
    grid_predictions = []
    pooled_pit = []
    for run_index in range(runs):
        run_seed = seed + run_index
        sample = generate_synthetic(dataset, seed=run_seed)
        train, validation, test = paper_split(len(sample.y), run_seed)
        oracle = fit_gp_oracle(sample.x[train], sample.y[train], run_seed)
        predicted_mean = oracle.predict(sample.x)
        residuals = sample.y - predicted_mean

        estimator = PolynomialAccrueRegressor().fit(
            sample.x[train], residuals[train], domain=(0.0, upper)
        )
        test_sigma = estimator.predict_sigma(sample.x[test])
        grid_predictions.append(estimator.predict_sigma(grid))
        pooled_pit.extend(ndtr(residuals[test] / test_sigma))
        rows.append(
            {
                "run": run_index,
                "seed": run_seed,
                "train_size": len(train),
                "validation_size": len(validation),
                "test_size": len(test),
                "polynomial_order": estimator.order_,
                "mean_rmse": rmse(sample.y[test], predicted_mean[test]),
                "sigma_rmse": rmse(sample.sigma[test], test_sigma),
                "nll": gaussian_nll(sample.y[test], predicted_mean[test], test_sigma),
                "crps": gaussian_crps(sample.y[test], predicted_mean[test], test_sigma),
                "calibration_error": calibration_error(
                    sample.y[test], predicted_mean[test], test_sigma
                ),
                "mean_predicted_sigma": float(np.mean(test_sigma)),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "per_run_metrics.csv", index=False)
    metric_columns = [
        "polynomial_order",
        "mean_rmse",
        "sigma_rmse",
        "nll",
        "crps",
        "calibration_error",
        "mean_predicted_sigma",
    ]
    summary = {
        "dataset": dataset,
        "runs": runs,
        "base_seed": seed,
        "protocol": "ACCRUE paper 33/33/34 split; GP mean oracle; Algorithm 1 polynomial sigma",
        "metrics_mean": results[metric_columns].mean().to_dict(),
        "metrics_std": results[metric_columns].std().to_dict(),
        "metrics_median": results[metric_columns].median().to_dict(),
        "metrics_iqr": (
            results[metric_columns].quantile(0.75)
            - results[metric_columns].quantile(0.25)
        ).to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    predictions = np.asarray(grid_predictions)
    predicted_mean = predictions.mean(axis=0)
    predicted_std = predictions.std(axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid[:, 0], true_grid_sigma, color="tab:red", label="True sigma(x)")
    ax.plot(grid[:, 0], predicted_mean, color="black", label="Mean estimated sigma(x)")
    ax.fill_between(
        grid[:, 0], predicted_mean - predicted_std, predicted_mean + predicted_std,
        color="0.75", alpha=0.7, label="plus/minus 1 run SD"
    )
    ax.fill_between(
        grid[:, 0], predicted_mean - 2 * predicted_std, predicted_mean + 2 * predicted_std,
        color="0.9", alpha=0.7, label="plus/minus 2 run SD"
    )
    ax.set(xlabel="x", ylabel="standard deviation", title=f"ACCRUE {dataset.upper()} recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sigma_recovery.png", dpi=180)
    plt.close(fig)

    expected, observed = reliability_curve(np.asarray(pooled_pit))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.4", label="Ideal")
    ax.plot(expected, observed, color="tab:blue", label="ACCRUE")
    ax.set(
        xlabel="Expected cumulative probability",
        ylabel="Observed cumulative probability",
        title=f"{dataset.upper()} reliability diagram",
        xlim=(0, 1), ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["g", "y", "w"], default="g")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        Path("results") / "paper_reproductions" / f"accrue_polynomial_{args.dataset}"
    )
    summary = run(args.dataset, args.runs, args.seed, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
