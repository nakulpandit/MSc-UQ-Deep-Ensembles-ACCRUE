import unittest

import numpy as np

try:
    from uqcomparison.models.deep_ensemble import DeepEnsembleConfig, DeepEnsembleRegressor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this runtime")
class TestDeepEnsembleModel(unittest.TestCase):
    def test_prediction_components_are_positive_and_consistent(self):
        x = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
        y = x[:, 0] ** 3
        config = DeepEnsembleConfig(
            members=2,
            hidden_layers=(8,),
            epochs=2,
            batch_size=12,
            adversarial_training=True,
        )
        model = DeepEnsembleRegressor(1, config=config, seed=4).fit(x, y)
        mean, aleatoric, epistemic, total = model.predict_components(x)
        self.assertEqual(mean.shape, (12,))
        self.assertTrue(np.all(aleatoric > 0))
        self.assertTrue(np.all(epistemic >= 0))
        self.assertTrue(np.all(total > 0))
        np.testing.assert_allclose(
            total**2, aleatoric**2 + epistemic**2, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
