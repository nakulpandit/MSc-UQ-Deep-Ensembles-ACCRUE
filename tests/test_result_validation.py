import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from uqcomparison.experiments.validate_results import (
    validate_neural_output_directory,
    validate_output_directory,
)


class TestResultValidation(unittest.TestCase):
    def test_valid_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            frame = pd.DataFrame(
                [
                    {
                        "run": 0,
                        "seed": 2026,
                        "train_size": 33,
                        "validation_size": 33,
                        "test_size": 34,
                        "polynomial_order": 2,
                        "mean_rmse": 0.8,
                        "sigma_rmse": 0.2,
                        "nll": 1.2,
                        "crps": 0.4,
                        "calibration_error": 0.1,
                        "mean_predicted_sigma": 0.7,
                    }
                ]
            )
            frame.to_csv(path / "per_run_metrics.csv", index=False)
            (path / "summary.json").write_text(json.dumps({"dataset": "g", "runs": 1}))
            (path / "sigma_recovery.png").write_bytes(b"0" * 1_001)
            (path / "reliability_diagram.png").write_bytes(b"0" * 1_001)

            report = validate_output_directory(path, expected_dataset="g", expected_runs=1)
            self.assertEqual(report["status"], "valid")

    def test_missing_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "missing result files"):
                validate_output_directory(Path(temporary))

    def test_valid_neural_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            frame = pd.DataFrame(
                [
                    {
                        "run": 0,
                        "seed": 2026,
                        "sample_size": 100,
                        "train_size": 33,
                        "validation_size": 33,
                        "test_size": 34,
                        "residual_rmse": 0.8,
                        "sigma_rmse": 0.2,
                        "nll": 1.2,
                        "crps": 0.4,
                        "calibration_error": 0.1,
                        "mean_predicted_sigma": 0.7,
                        "training_seconds": 1.0,
                        "inference_ms_per_sample": 0.1,
                        "best_validation_score": 0.2,
                        "best_restart": 0,
                        "epochs_trained": 3,
                    }
                ]
            )
            frame.to_csv(path / "per_run_metrics.csv", index=False)
            summary = {
                "method": "neural_accrue",
                "dataset": "g",
                "mode": "debug",
                "runs": 1,
            }
            (path / "summary.json").write_text(json.dumps(summary))
            (path / "sigma_curve.csv").write_text("x,true_sigma\n0,1\n")
            (path / "sigma_recovery.png").write_bytes(b"0" * 1_001)
            (path / "reliability_diagram.png").write_bytes(b"0" * 1_001)

            report = validate_neural_output_directory(
                path, expected_dataset="g", expected_runs=1, expected_mode="debug"
            )
            self.assertEqual(report["kind"], "neural")


if __name__ == "__main__":
    unittest.main()
