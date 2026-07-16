"""NumPy implementation of the ACCRUE scoring equations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf


def gaussian_crps_values(
    errors: ArrayLike, sigma: ArrayLike
) -> NDArray[np.float64]:
    errors, sigma = np.broadcast_arrays(
        np.asarray(errors, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
    )
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be positive")
    z = errors / sigma
    return sigma * (
        z * erf(z / np.sqrt(2.0))
        + np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z**2)
        - 1.0 / np.sqrt(np.pi)
    )


def reliability_score(errors: ArrayLike, sigma: ArrayLike) -> float:
    """Analytic Gaussian reliability score from ACCRUE Eq. (8)."""
    errors, sigma = np.broadcast_arrays(
        np.asarray(errors, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
    )
    if errors.size == 0:
        raise ValueError("at least one error is required")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be positive")

    eta = np.sort((errors / (np.sqrt(2.0) * sigma)).reshape(-1))
    n = eta.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    terms = (
        eta * (erf(eta) + 1.0) / n
        - eta * (2.0 * ranks - 1.0) / n**2
        + np.exp(-eta**2) / (np.sqrt(np.pi) * n)
    )
    return float(np.sum(terms) - 0.5 * np.sqrt(2.0 / np.pi))


def accrue_weight(errors: ArrayLike) -> float:
    """Heuristic beta from ACCRUE Eqs. (12)-(13), using its practical RS scale."""
    errors = np.asarray(errors, dtype=np.float64)
    crps_min = np.sqrt(np.log(4.0)) / 2.0 * np.mean(np.abs(errors))
    rs_min = 0.5 * np.sqrt(2.0 / np.pi)
    return float(rs_min / (crps_min + rs_min + np.finfo(float).eps))


def accrue_score(
    errors: ArrayLike,
    sigma: ArrayLike,
    beta: float | None = None,
) -> tuple[float, float, float]:
    """Return ACCRUE, mean CRPS and reliability score."""
    if beta is None:
        beta = accrue_weight(errors)
    crps = float(np.mean(gaussian_crps_values(errors, sigma)))
    reliability = reliability_score(errors, sigma)
    return beta * crps + (1.0 - beta) * reliability, crps, reliability
