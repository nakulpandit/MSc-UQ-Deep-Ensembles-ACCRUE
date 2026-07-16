"""Validate saved Deep Ensemble Figure 1 reproduction results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "per_run_metrics.csv",
    "summary.json",
    "figure1_predictions.csv",
    "deep_ensemble_figure1_paper_config.png",
)

METRIC_STEMS = (
    "rmse",
    "curve_rmse",
    "nll",
    "coverage_1sigma",
    "coverage_2sigma",
    "coverage_3sigma",
    "mean_aleatoric_sigma",
    "mean_epistemic_sigma",
    "mean_total_sigma",
)


def required_columns():
    columns = ["run", "seed", "training_seconds", "inference_ms_per_sample"]
    for domain in ("overall", "in_domain", "out_of_domain"):
        columns.extend(f"{stem}_{domain}" for stem in METRIC_STEMS)
    columns.extend(["sigma_mae_in_domain", "sigma_rmse_in_domain"])
    return columns


def validate(path: Path, expected_mode=None, expected_runs=None) -> dict:
    path = Path(path)
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"missing result files: {', '.join(missing)}")
    summary = json.loads((path / "summary.json").read_text())
    metrics = pd.read_csv(path / "per_run_metrics.csv")
    if summary.get("method") != "deep_ensemble":
        raise ValueError("summary method is not deep_ensemble")
    if summary.get("experiment") != "figure1_cubic_regression":
        raise ValueError("summary experiment is not figure1_cubic_regression")
    mode = str(summary.get("mode", ""))
    runs = int(summary.get("runs", -1))
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"expected mode {expected_mode!r}, found {mode!r}")
    if expected_runs is not None and runs != expected_runs:
        raise ValueError(f"expected {expected_runs} runs, found {runs}")
    if len(metrics) != runs or metrics["run"].nunique() != runs:
        raise ValueError("result row count or run identifiers do not match summary")
    missing_columns = [name for name in required_columns() if name not in metrics.columns]
    if missing_columns:
        raise ValueError(f"missing metric columns: {', '.join(missing_columns)}")
    numeric = metrics[required_columns()].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("metrics contain NaN or infinite values")
    coverage_columns = [name for name in metrics if name.startswith("coverage_")]
    if np.any(metrics[coverage_columns] < 0) or np.any(metrics[coverage_columns] > 1):
        raise ValueError("coverage metrics must be between zero and one")
    nonnegative = [
        name
        for name in required_columns()
        if name not in {"run", "seed"} and not name.startswith("nll_")
    ]
    if np.any(metrics[nonnegative] < 0):
        raise ValueError("nonnegative metrics contain a negative value")
    if (path / "deep_ensemble_figure1_paper_config.png").stat().st_size < 1_000:
        raise ValueError("Figure 1 image is unexpectedly small")
    return {
        "status": "valid",
        "kind": "deep_ensemble_figure1",
        "path": str(path),
        "mode": mode,
        "runs": runs,
        "files": list(REQUIRED_FILES),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--mode", choices=["debug", "development", "final", "long_diagnostic"])
    parser.add_argument("--runs", type=int)
    args = parser.parse_args()
    print(json.dumps(validate(args.path, args.mode, args.runs), indent=2))


if __name__ == "__main__":
    main()
