import unittest
from pathlib import Path

from uqcomparison.experiments.reproduce_accrue_neural import (
    default_output_directory,
    resolve_mode,
)


class TestNeuralRunnerConfiguration(unittest.TestCase):
    def test_final_mode_matches_paper_scale(self):
        mode = resolve_mode("final")
        self.assertEqual(mode.runs, 100)
        self.assertEqual(mode.restarts, 5)
        self.assertEqual(mode.one_dimensional_samples, 100)
        self.assertEqual(mode.five_dimensional_samples, 10_000)

    def test_debug_outputs_are_not_final_outputs(self):
        path = default_output_directory("g", "debug")
        self.assertEqual(path, Path("results/development/accrue_neural_g_debug"))

    def test_final_outputs_use_paper_reproduction_folder(self):
        path = default_output_directory("5d", "final")
        self.assertEqual(path, Path("results/paper_reproductions/accrue_neural_5d"))


if __name__ == "__main__":
    unittest.main()
