import importlib.util
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this runtime")
class TestNeuralAccrue(unittest.TestCase):
    def test_torch_score_matches_numpy_score(self):
        import torch

        from uqcomparison.metrics.accrue import accrue_score
        from uqcomparison.models.accrue import accrue_loss

        errors = np.array([-0.8, -0.2, 0.1, 0.7], dtype=float)
        sigma = np.array([0.6, 0.7, 0.8, 0.9], dtype=float)
        expected = accrue_score(errors, sigma)
        actual = accrue_loss(
            torch.as_tensor(errors, dtype=torch.float64),
            torch.as_tensor(sigma, dtype=torch.float64),
        )
        np.testing.assert_allclose(
            [float(value) for value in actual], expected, rtol=1e-10, atol=1e-10
        )

    def test_small_neural_fit_produces_positive_sigma(self):
        from uqcomparison.data import generate_synthetic
        from uqcomparison.models.accrue import AccrueConfig, AccrueVarianceRegressor

        sample = generate_synthetic("g", n_samples=40, seed=14)
        residuals = sample.y - sample.mean
        config = AccrueConfig(
            restarts=1,
            max_epochs=2,
            lbfgs_iterations_per_epoch=1,
            patience=1,
        )
        model = AccrueVarianceRegressor(1, config=config, seed=14).fit(
            sample.x[:25], residuals[:25], sample.x[25:], residuals[25:]
        )
        prediction = model.predict_sigma(sample.x)
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(prediction > 0.0))
        self.assertIsNotNone(model.best_validation_score_)


if __name__ == "__main__":
    unittest.main()
