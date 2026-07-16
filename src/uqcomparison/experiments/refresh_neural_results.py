"""Upgrade saved neural ACCRUE results without repeating model training."""

import argparse
import json
from pathlib import Path

import pandas as pd

from uqcomparison.experiments.reproduce_accrue_neural import (
    _aggregate,
    _plot_column_normalized_density,
)


def refresh(path: Path) -> dict:
    path = Path(path)
    metrics_path = path / "per_run_metrics.csv"
    summary_path = path / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise ValueError("missing per_run_metrics.csv or summary.json")

    metrics = pd.read_csv(metrics_path)
    if "residual_rmse" not in metrics.columns:
        if "mean_rmse" not in metrics.columns:
            raise ValueError("metrics contain neither residual_rmse nor legacy mean_rmse")
        metrics = metrics.rename(columns={"mean_rmse": "residual_rmse"})
        metrics.to_csv(metrics_path, index=False)

    summary = json.loads(summary_path.read_text())
    if summary.get("method") != "neural_accrue":
        raise ValueError("summary method is not neural_accrue")
    summary.update(_aggregate(metrics))
    summary["metric_notes"] = {
        "residual_rmse": (
            "RMSE between observed targets and the supplied mean oracle; this is not "
            "mean-function estimation error."
        )
    }

    if str(summary.get("dataset", "")).lower() == "5d":
        pairs_path = path / "sigma_pairs_sample.csv"
        if not pairs_path.is_file():
            raise ValueError("5D results are missing sigma_pairs_sample.csv")
        _plot_column_normalized_density(pd.read_csv(pairs_path), path)

    output_files = list(summary.get("output_files", []))
    if (path / "sigma_density.png").is_file() and "sigma_density.png" not in output_files:
        insert_at = (
            output_files.index("sigma_scatter.png")
            if "sigma_scatter.png" in output_files
            else len(output_files)
        )
        output_files.insert(insert_at, "sigma_density.png")
    summary["output_files"] = output_files
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    summary = refresh(args.path)
    print(json.dumps({
        "status": "refreshed",
        "path": str(args.path),
        "dataset": summary.get("dataset"),
        "runs": summary.get("runs"),
    }, indent=2))


if __name__ == "__main__":
    main()
