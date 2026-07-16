import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from uqcomparison.experiments.refresh_neural_results import refresh


class TestRefreshNeuralResults(unittest.TestCase):
    def test_legacy_5d_results_are_upgraded_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            metrics = pd.DataFrame(
                {
                    "mean_rmse": [0.6, 0.7],
                    "sigma_rmse": [0.1, 0.2],
                    "nll": [0.5, 0.6],
                    "crps": [0.3, 0.4],
                    "calibration_error": [0.01, 0.02],
                    "mean_predicted_sigma": [0.5, 0.6],
                    "training_seconds": [1.0, 1.1],
                    "inference_ms_per_sample": [0.01, 0.02],
                    "best_validation_score": [0.2, 0.3],
                    "best_restart": [0, 1],
                    "epochs_trained": [10, 12],
                }
            )
            metrics.to_csv(path / "per_run_metrics.csv", index=False)
            pd.DataFrame(
                {
                    "true_sigma": [0.1, 0.2, 0.3, 0.4],
                    "predicted_sigma": [0.12, 0.18, 0.35, 0.38],
                }
            ).to_csv(path / "sigma_pairs_sample.csv", index=False)
            (path / "summary.json").write_text(
                json.dumps(
                    {
                        "method": "neural_accrue",
                        "dataset": "5d",
                        "runs": 2,
                        "output_files": [
                            "per_run_metrics.csv",
                            "summary.json",
                            "sigma_scatter.png",
                            "sigma_pairs_sample.csv",
                        ],
                    }
                )
            )

            summary = refresh(path)

            upgraded = pd.read_csv(path / "per_run_metrics.csv")
            self.assertIn("residual_rmse", upgraded.columns)
            self.assertNotIn("mean_rmse", upgraded.columns)
            self.assertTrue((path / "sigma_density.png").is_file())
            self.assertIn("sigma_density.png", summary["output_files"])
            self.assertIn("residual_rmse", summary["metrics_mean"])


if __name__ == "__main__":
    unittest.main()
