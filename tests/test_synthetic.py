import unittest

import numpy as np

from uqcomparison.data import generate_synthetic


class TestSyntheticData(unittest.TestCase):
    def test_all_paper_generators_have_positive_sigma(self):
        for name in ("g", "y", "w", "5d"):
            sample = generate_synthetic(name, n_samples=25, seed=7)
            self.assertTrue(np.all(sample.sigma > 0))
            self.assertEqual(sample.x.shape[0], 25)

    def test_generation_is_reproducible(self):
        first = generate_synthetic("g", seed=12)
        second = generate_synthetic("g", seed=12)
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.y, second.y)

    def test_default_sizes_match_paper(self):
        self.assertEqual(len(generate_synthetic("g", seed=1).y), 100)
        self.assertEqual(len(generate_synthetic("5d", seed=1).y), 10_000)


if __name__ == "__main__":
    unittest.main()
