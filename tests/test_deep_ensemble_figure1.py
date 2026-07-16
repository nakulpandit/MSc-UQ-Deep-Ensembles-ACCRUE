import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from uqcomparison.experiments.reproduce_deep_ensemble_figure1 import (
    MODES,
    default_output_directory,
    generate_training_data,
)
from uqcomparison.experiments.validate_deep_ensemble import required_columns, validate


class TestDeepEnsembleFigure1(unittest.TestCase):
    def test_final_mode_is_the_primary_paper_configuration(self):
        mode = MODES["final"]
        self.assertEqual(mode.members, 5)
        self.assertEqual(mode.epochs, 40)
        self.assertEqual(mode.learning_rate, 0.1)
        self.assertTrue(mode.adversarial_training)

    def test_long_training_remains_a_separate_diagnostic(self):
        mode = MODES["long_diagnostic"]
        self.assertEqual(mode.epochs, 2_000)
        self.assertEqual(mode.learning_rate, 0.001)
        self.assertNotEqual(
            default_output_directory("long_diagnostic"),
            default_output_directory("final"),
        )

    def test_training_generator_is_reproducible(self):
        first_x, first_y = generate_training_data(2026)
        second_x, second_y = generate_training_data(2026)
        self.assertEqual(first_x.shape, (20, 1))
        self.assertTrue((first_x == second_x).all())
        self.assertTrue((first_y == second_y).all())

    def test_saved_result_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            row = {name: 0.1 for name in required_columns()}
            row.update({"run": 0, "seed": 2026})
            pd.DataFrame([row]).to_csv(path / "per_run_metrics.csv", index=False)
            pd.DataFrame(
                {
                    "x": [-1.0, 1.0],
                    "predicted_mean": [-1.0, 1.0],
                    "total_sigma": [3.0, 3.0],
                }
            ).to_csv(path / "figure1_predictions.csv", index=False)
            (path / "deep_ensemble_figure1_paper_config.png").write_bytes(b"0" * 1_001)
            (path / "summary.json").write_text(
                json.dumps(
                    {
                        "method": "deep_ensemble",
                        "experiment": "figure1_cubic_regression",
                        "mode": "debug",
                        "runs": 1,
                    }
                )
            )
            report = validate(path, expected_mode="debug", expected_runs=1)
            self.assertEqual(report["status"], "valid")


if __name__ == "__main__":
    unittest.main()
