import unittest

import numpy as np

from uqcomparison.metrics import (
    calibration_error,
    gaussian_crps,
    gaussian_nll,
    interval_coverage,
    mae,
    rmse,
)


class TestProbabilisticMetrics(unittest.TestCase):
    def test_point_metrics_are_zero_for_exact_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(rmse(y, y), 0.0)
        self.assertEqual(mae(y, y), 0.0)

    def test_standard_normal_nll_at_mean(self):
        value = gaussian_nll([0.0], [0.0], [1.0])
        self.assertAlmostEqual(value, 0.5 * np.log(2.0 * np.pi))

    def test_crps_at_mean(self):
        value = gaussian_crps([0.0], [0.0], [1.0])
        expected = np.sqrt(2.0 / np.pi) - 1.0 / np.sqrt(np.pi)
        self.assertAlmostEqual(value, expected)

    def test_invalid_sigma_is_rejected(self):
        with self.assertRaises(ValueError):
            gaussian_nll([0.0], [0.0], [0.0])

    def test_interval_coverage_bounds(self):
        coverage = interval_coverage([0.0, 10.0], [0.0, 0.0], [1.0, 1.0], nominal=0.9)
        self.assertEqual(coverage, 0.5)

    def test_calibration_error_is_bounded(self):
        rng = np.random.default_rng(4)
        y = rng.normal(size=1000)
        value = calibration_error(y, np.zeros_like(y), np.ones_like(y))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
