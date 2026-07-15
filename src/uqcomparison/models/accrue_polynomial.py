"""Algorithm 1 polynomial ACCRUE estimator for one-dimensional toy data."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from uqcomparison.metrics.accrue import accrue_score, accrue_weight


@dataclass
class PolynomialFitStep:
    order: int
    score: float
    success: bool


@dataclass
class PolynomialAccrueRegressor:
    """Estimate sigma(x) as a polynomial of order at most ten.

    This follows Algorithm 1 of Camporeale & Care (2021): begin with a
    constant equal to the residual standard deviation, increase polynomial
    order one at a time, warm-start the next order, and stop when successive
    ACCRUE scores differ by less than ``tolerance``.
    """

    max_order: int = 10
    tolerance: float = 1e-2
    sigma_floor: float = 1e-6
    max_iterations: int = 2_000
    coefficients_: NDArray[np.float64] | None = field(default=None, init=False)
    beta_: float | None = field(default=None, init=False)
    history_: list[PolynomialFitStep] = field(default_factory=list, init=False)

    @staticmethod
    def _as_1d(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        if x.ndim == 1:
            return x
        raise ValueError("Polynomial ACCRUE supports exactly one input dimension")

    def fit(
        self,
        x: NDArray[np.float64],
        residuals: NDArray[np.float64],
        domain: tuple[float, float] | None = None,
    ):
        x = self._as_1d(x)
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        if x.size != residuals.size or x.size < 2:
            raise ValueError("x and residuals must have the same length of at least two")

        initial_sigma = max(float(np.std(residuals, ddof=1)), self.sigma_floor)
        coefficients = np.array([initial_sigma], dtype=np.float64)
        self.beta_ = accrue_weight(residuals)
        self.history_ = []
        if domain is None:
            domain = (float(np.min(x)), float(np.max(x)))
        if domain[0] >= domain[1]:
            raise ValueError("domain must be an increasing (lower, upper) pair")
        positivity_grid = np.linspace(domain[0], domain[1], 256)
        previous_score = np.inf
        best_score = np.inf
        best_coefficients = coefficients.copy()

        for order in range(1, self.max_order + 1):
            start = np.pad(coefficients, (0, 1))

            def objective(theta):
                sigma = np.polynomial.polynomial.polyval(x, theta)
                domain_sigma = np.polynomial.polynomial.polyval(positivity_grid, theta)
                if (
                    np.any(domain_sigma <= self.sigma_floor)
                    or not np.all(np.isfinite(domain_sigma))
                ):
                    violation = np.minimum(domain_sigma - self.sigma_floor, 0.0)
                    return 1e6 + 1e6 * float(np.sum(violation**2))
                return accrue_score(residuals, sigma, self.beta_)[0]

            result = minimize(
                objective,
                start,
                method="BFGS",
                options={"maxiter": self.max_iterations, "gtol": 1e-8},
            )
            score = float(objective(result.x))
            self.history_.append(PolynomialFitStep(order, score, bool(result.success)))

            if score < best_score:
                best_score = score
                best_coefficients = result.x.copy()
            coefficients = result.x.copy()
            improvement = abs(previous_score - score)
            if np.isfinite(previous_score) and improvement <= self.tolerance:
                break
            previous_score = score

        self.coefficients_ = best_coefficients
        return self

    @property
    def order_(self) -> int:
        if self.coefficients_ is None:
            raise RuntimeError("fit must be called before reading order_")
        return len(self.coefficients_) - 1

    def predict_sigma(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coefficients_ is None:
            raise RuntimeError("fit must be called before predict_sigma")
        values = np.polynomial.polynomial.polyval(self._as_1d(x), self.coefficients_)
        return np.maximum(values, self.sigma_floor)
