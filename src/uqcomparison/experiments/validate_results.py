"""Validate a saved ACCRUE paper-reproduction result directory."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "per_run_metrics.csv",
    "summary.json",
    "sigma_recovery.png",
    "reliability_diagram.png",
)

REQUIRED_COLUMNS = (
    "run",
    "seed",
    "train_size",
    "validation_size",
    "test_size",
    "polynomial_order",
    "mean_rmse",
    "sigma_rmse",
    "nll",
    "crps",
    "calibration_error",
    "mean_predicted_sigma",
)


def validate_output_directory(
    path: Path,
    expected_dataset: str | None = None,
    expected_runs: int | None = None,
) -> dict:
    path = Path(path)
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"missing result files: {', '.join(missing)}")

    summary = json.loads((path / "summary.json").read_text())
    metrics = pd.read_csv(path / "per_run_metrics.csv")
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in metrics.columns]
    if missing_columns:
        raise ValueError(f"missing metric columns: {', '.join(missing_columns)}")

    dataset = str(summary.get("dataset", "")).lower()
    runs = int(summary.get("runs", -1))
    if expected_dataset is not None and dataset != expected_dataset.lower():
        raise ValueError(f"expected dataset {expected_dataset!r}, found {dataset!r}")
    if expected_runs is not None and runs != expected_runs:
        raise ValueError(f"expected {expected_runs} runs, found {runs}")
    if len(metrics) != runs:
        raise ValueError(f"summary records {runs} runs but CSV contains {len(metrics)} rows")
    if metrics["run"].nunique() != runs:
        raise ValueError("run identifiers are not unique")

    numeric = metrics[list(REQUIRED_COLUMNS)].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("metrics contain NaN or infinite values")
    if np.any(metrics[["train_size", "validation_size", "test_size"]] <= 0):
        raise ValueError("data split sizes must be positive")
    if np.any(metrics[["sigma_rmse", "crps", "calibration_error", "mean_predicted_sigma"]] < 0):
        raise ValueError("nonnegative uncertainty metrics contain a negative value")

    for image_name in ("sigma_recovery.png", "reliability_diagram.png"):
        if (path / image_name).stat().st_size < 1_000:
            raise ValueError(f"{image_name} is unexpectedly small")

    return {
        "status": "valid",
        "path": str(path),
        "dataset": dataset,
        "runs": runs,
        "files": list(REQUIRED_FILES),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--dataset", choices=["g", "y", "w"])
    parser.add_argument("--runs", type=int)
    args = parser.parse_args()
    report = validate_output_directory(args.path, args.dataset, args.runs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
