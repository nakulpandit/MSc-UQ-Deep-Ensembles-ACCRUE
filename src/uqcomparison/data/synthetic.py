"""Synthetic heteroscedastic regression datasets from the ACCRUE paper."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticSample:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    mean: NDArray[np.float64]
    sigma: NDArray[np.float64]


def _functions(name: str, x: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    key = name.lower()
    if key == "g":
        mean = 2.0 * np.sin(2.0 * np.pi * x[:, 0])
        sigma = 0.5 * x[:, 0] + 0.5
    elif key == "y":
        x0 = x[:, 0]
        mean = 2.0 * (np.exp(-30.0 * (x0 - 0.25) ** 2) + np.sin(np.pi * x0**2)) - 2.0
        sigma = np.exp(np.sin(2.0 * np.pi * x0)) / 3.0
    elif key == "w":
        x0 = x[:, 0]
        mean = np.sin(2.5 * x0) * np.sin(1.5 * x0)
        sigma = 0.01 + 0.25 * (1.0 - np.sin(2.5 * x0)) ** 2
    elif key in {"5d", "five_d"}:
        mean = np.zeros(x.shape[0], dtype=np.float64)
        sigma = 0.45 * (np.cos(np.pi + 5.0 * np.sum(x, axis=1)) + 1.2)
    else:
        raise ValueError(f"Unknown synthetic dataset {name!r}; choose g, y, w, or 5d")
    return mean, sigma


def generate_synthetic(
    name: str,
    n_samples: int | None = None,
    seed: int = 0,
) -> SyntheticSample:
    """Generate one independent paper-style synthetic sample."""
    key = name.lower()
    dimensions = 5 if key in {"5d", "five_d"} else 1
    if n_samples is None:
        n_samples = 10_000 if dimensions == 5 else 100
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    upper = np.pi if key == "w" else 1.0
    x = rng.uniform(0.0, upper, size=(n_samples, dimensions))
    mean, sigma = _functions(key, x)
    y = rng.normal(mean, sigma)
    return SyntheticSample(x=x, y=y, mean=mean, sigma=sigma)
