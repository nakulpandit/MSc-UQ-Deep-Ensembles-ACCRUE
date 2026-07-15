"""Common Gaussian regression metrics used across both methods."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf, ndtr, ndtri


def _arrays(y: ArrayLike, mean: ArrayLike, sigma: ArrayLike):
    y_arr, mean_arr, sigma_arr = np.broadcast_arrays(
        np.asarray(y, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    if np.any(sigma_arr <= 0) or not np.all(np.isfinite(sigma_arr)):
        raise ValueError("sigma must contain finite positive values")
    return y_arr, mean_arr, sigma_arr


def rmse(y: ArrayLike, mean: ArrayLike) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(mean)) ** 2)))


def mae(y: ArrayLike, mean: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(mean))))


def gaussian_nll(y: ArrayLike, mean: ArrayLike, sigma: ArrayLike) -> float:
    y_arr, mean_arr, sigma_arr = _arrays(y, mean, sigma)
    z = (y_arr - mean_arr) / sigma_arr
    return float(np.mean(np.log(sigma_arr) + 0.5 * z**2 + 0.5 * np.log(2.0 * np.pi)))


def gaussian_crps(y: ArrayLike, mean: ArrayLike, sigma: ArrayLike) -> float:
    """Mean analytic CRPS for univariate Gaussian forecasts (ACCRUE Eq. 3)."""
    y_arr, mean_arr, sigma_arr = _arrays(y, mean, sigma)
    error = y_arr - mean_arr
    z = error / sigma_arr
    values = sigma_arr * (
        z * erf(z / np.sqrt(2.0))
        + np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z**2)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(values))


def calibration_error(y: ArrayLike, mean: ArrayLike, sigma: ArrayLike) -> float:
    """Maximum PIT reliability deviation; ACCRUE reports 100 times this value."""
    y_arr, mean_arr, sigma_arr = _arrays(y, mean, sigma)
    pit = np.sort(ndtr((y_arr - mean_arr) / sigma_arr))
    n = pit.size
    if n == 0:
        raise ValueError("at least one observation is required")
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(upper - pit)), np.max(np.abs(pit - lower))))


def interval_coverage(
    y: ArrayLike,
    mean: ArrayLike,
    sigma: ArrayLike,
    nominal: float = 0.9,
) -> float:
    if not 0.0 < nominal < 1.0:
        raise ValueError("nominal must lie between zero and one")
    y_arr, mean_arr, sigma_arr = _arrays(y, mean, sigma)
    quantile = ndtri(0.5 + nominal / 2.0)
    return float(np.mean(np.abs(y_arr - mean_arr) <= quantile * sigma_arr))
