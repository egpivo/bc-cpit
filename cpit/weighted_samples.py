"""Predictive CDF: F_adj(y|x) and F_tilde(y|x) = C_hat(F_adj(y|x)); output weighted samples w_j."""

from typing import Callable

import numpy as np

from .bc.apply import _apply_bias_correction
from .bc.params import AffineParams, XAffineParamsGAM


def empirical_cdf_at_y(y_samples: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    """Fraction of y_samples <= y for each y in y_query. y_samples (m,), y_query scalar or (k,) -> (k,)."""
    y_samples = np.asarray(y_samples).ravel()
    y_query = np.asarray(y_query)
    if y_query.ndim == 0:
        return np.clip(np.mean(y_samples <= y_query), 0.0, 1.0)
    out = (y_samples.reshape(-1, 1) <= y_query.reshape(1, -1)).mean(axis=0)
    return np.clip(out, 0.0, 1.0)


def f_adj_from_samples(
    y_samples: np.ndarray,
    params: AffineParams | XAffineParamsGAM,
    y_query: np.ndarray,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """F_adj(y|x): empirical CDF of bias-corrected samples. Pass x when params is XAffineParamsGAM."""
    y_adj = _apply_bias_correction(y_samples, params, x)
    return empirical_cdf_at_y(y_adj, y_query)


def f_tilde_from_samples(
    y_samples: np.ndarray,
    params: AffineParams | XAffineParamsGAM,
    c_hat_fn: Callable[[np.ndarray], np.ndarray],
    y_query: np.ndarray,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """F_tilde(y|x) = C_hat(F_adj(y|x))."""
    u_adj = f_adj_from_samples(y_samples, params, y_query, x)
    return c_hat_fn(u_adj)


def weighted_samples_from_cdf(
    y_grid: np.ndarray,
    f_values: np.ndarray,
    *,
    left_prepend: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_grid, weights) where weights = increments of f_values.

    left_prepend sets the value prepended before differencing (e.g. Äˆ(0)=0 for Eq. 17).
    """
    y_grid = np.asarray(y_grid).ravel()
    f_values = np.asarray(f_values).ravel()
    if len(y_grid) != len(f_values):
        raise ValueError("y_grid and f_values must have same length")
    prepend = left_prepend if left_prepend is not None else 0.0
    w = np.diff(f_values, prepend=float(prepend))
    return y_grid.copy(), w


def get_weighted_samples_at_x(
    y_samples_raw: np.ndarray,
    params: AffineParams | XAffineParamsGAM,
    c_hat_fn: Callable[[np.ndarray], np.ndarray],
    y_grid: np.ndarray | None = None,
    x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pts, weights) for the calibrated predictive distribution at one x.

    **Fast path (y_grid=None, default):** canonical Eq. (17) implementation.
    Sorts y_adj â†’ y^(1)â‰¤â€¦â‰¤y^(m); uses F_adj(y^(j)) = j/m exactly, so
    w_j = Äˆ(j/m) âˆ’ Äˆ((jâˆ’1)/m) with Äˆ(0) := 0. Weights sum to Äˆ(1) âˆ’ Äˆ(0) = 1.

    **Generic path (y_grid provided):** evaluates CDF increments Äˆ(F_adj(y_grid_j))
    on a custom grid â€” for plotting or diagnostics, not for Eq. (17) inference weights.
    If y_grid[-1] < max(y_adj), weights will sum to less than 1 (intentional).
    """
    y_samples_raw = np.asarray(y_samples_raw).ravel()
    y_adj = _apply_bias_correction(y_samples_raw, params, x)

    if y_grid is None:
        y_sorted = np.sort(y_adj)
        m = y_sorted.size
        if m == 0:
            raise ValueError("y_samples_raw must be non-empty.")
        u = np.arange(1, m + 1, dtype=float) / float(m)  # j/m
        f_tilde = np.asarray(c_hat_fn(u), dtype=float).ravel()
        return weighted_samples_from_cdf(y_sorted, f_tilde, left_prepend=0.0)

    y_grid = np.asarray(y_grid).ravel()
    f_tilde = f_tilde_from_samples(y_samples_raw, params, c_hat_fn, y_grid, x)
    return weighted_samples_from_cdf(
        y_grid, np.asarray(f_tilde).ravel(), left_prepend=0.0
    )
