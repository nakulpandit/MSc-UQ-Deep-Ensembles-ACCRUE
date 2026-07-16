import unittest

import numpy as np

from uqcomparison.data import generate_synthetic
from uqcomparison.metrics.accrue import accrue_score, reliability_score
from uqcomparison.models.accrue_polynomial import PolynomialAccrueRegressor


class TestAccrue(unittest.TestCase):
    def test_score_is_finite(self):
        errors = np.array([-1.0, -0.2, 0.1, 0.8])
        sigma = np.ones_like(errors)
        score, crps, reliability = accrue_score(errors, sigma)
        self.assertTrue(np.all(np.isfinite([score, crps, reliability])))
        self.assertAlmostEqual(reliability, reliability_score(errors, sigma))

    def test_polynomial_estimator_is_positive(self):
        sample = generate_synthetic("g", n_samples=80, seed=9)
        residuals = sample.y - sample.mean
        model = PolynomialAccrueRegressor(max_order=3, max_iterations=300).fit(
            sample.x, residuals
        )
        prediction = model.predict_sigma(np.linspace(0.0, 1.0, 50))
        self.assertTrue(np.all(prediction > 0.0))
        self.assertLessEqual(model.order_, 3)


if __name__ == "__main__":
    unittest.main()
