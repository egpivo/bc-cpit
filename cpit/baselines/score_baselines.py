"""Score-based split-conformal baselines: SR (scaled residual), CDF-rank, eCRPS sublevel."""

from typing import Tuple

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split-conformal finite-sample quantile with "higher" interpolation:
      q = Quantile_{ceil((n+1)(1-alpha))/n}(scores)
    """
    s = np.asarray(scores).ravel()
    n = s.size
    if n == 0:
        raise ValueError("scores must be non-empty")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def sr_split_conformal_interval(
    y_calibration_samples: np.ndarray,
    y_calibration_observed: np.ndarray,
    y_test_samples: np.ndarray,
    alpha: float,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Scaled residual baseline:
      s_i = |y_i - mu_i| / sigma_i
      q = conformal_quantile(s_i)
      C(x) = [mu_test - q*sigma_test, mu_test + q*sigma_test]
    """
    y_cal_s = np.asarray(y_calibration_samples)
    y_cal_o = np.asarray(y_calibration_observed).ravel()
    if y_cal_s.ndim != 2:
        raise ValueError("y_calibration_samples must have shape (n_cal, m)")
    if y_cal_s.shape[0] != y_cal_o.size:
        raise ValueError("Calibration samples/observed size mismatch")

    mu_cal = np.mean(y_cal_s, axis=1)
    sigma_cal = np.std(y_cal_s, axis=1, ddof=1)
    sigma_cal = np.maximum(sigma_cal, eps)
    scores = np.abs(y_cal_o - mu_cal) / sigma_cal
    q = _conformal_quantile(scores, alpha)

    y_test = np.asarray(y_test_samples).ravel()
    mu_test = float(np.mean(y_test))
    sigma_test = float(np.std(y_test, ddof=1))
    sigma_test = max(sigma_test, eps)
    return float(mu_test - q * sigma_test), float(mu_test + q * sigma_test)


def empirical_crps_score(
    y_samples: np.ndarray,
    y_value: float,
) -> float:
    """
    Empirical CRPS(y) for uniform-weighted samples:
      mean_j |Y_j - y| - 0.5 * mean_{j,k} |Y_j - Y_k|

    Uses the sorting identity for the dispersion term:
      0.5 * mean_{j,k} |Y_j - Y_k| = (1/m^2) sum_j (2j - m - 1) Y_(j)
    giving O(m log m) instead of O(m^2).
    """
    y = np.sort(np.asarray(y_samples).ravel())
    m = y.size
    term1 = float(np.mean(np.abs(y - y_value)))
    j = np.arange(1, m + 1, dtype=float)
    term2 = float(np.dot(2 * j - m - 1, y) / (m * m))
    return term1 - term2


def _empirical_crps_batch(y_samples: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """CRPS(y_samples, y_values[k]) for each k. y_samples (m,), y_values (k,) -> (k,)."""
    y = np.sort(np.asarray(y_samples).ravel())
    m = y.size
    j = np.arange(1, m + 1, dtype=float)
    dispersion = np.dot(2 * j - m - 1, y) / (m * m)
    # term1 for each y_value: mean |y_j - y_val|
    term1 = np.mean(
        np.abs(y[np.newaxis, :] - np.asarray(y_values)[:, np.newaxis]), axis=1
    )
    return term1 - dispersion


def _empirical_crps_rows(y_samples: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
    """CRPS per row: y_samples (n, m), y_observed (n,) -> scores (n,)."""
    y = np.sort(np.asarray(y_samples), axis=1)  # (n, m)
    m = y.shape[1]
    j = np.arange(1, m + 1, dtype=float)
    dispersion = y @ (2 * j - m - 1) / (m * m)  # (n,)
    term1 = np.mean(np.abs(y - np.asarray(y_observed)[:, np.newaxis]), axis=1)
    return term1 - dispersion


def crps_split_conformal_interval(
    y_calibration_samples: np.ndarray,
    y_calibration_observed: np.ndarray,
    y_test_samples: np.ndarray,
    alpha: float,
    n_grid: int = 800,
    max_expand_steps: int = 6,
) -> Tuple[float, float]:
    """
    CRPS score baseline:
      s_i = CRPS_i(y_i)
      q = conformal_quantile(s_i)
      C(x) = {y: CRPS_test(y) <= q}, approximated on a dense grid.
    """
    y_cal_s = np.asarray(y_calibration_samples)
    y_cal_o = np.asarray(y_calibration_observed).ravel()
    if y_cal_s.ndim != 2:
        raise ValueError("y_calibration_samples must have shape (n_cal, m)")
    if y_cal_s.shape[0] != y_cal_o.size:
        raise ValueError("Calibration samples/observed size mismatch")

    cal_scores = _empirical_crps_rows(y_cal_s, y_cal_o)
    q = _conformal_quantile(cal_scores, alpha)

    y_test = np.asarray(y_test_samples).ravel()
    y_min = float(np.min(y_test))
    y_max = float(np.max(y_test))
    spread = max(1e-8, y_max - y_min)

    pad = 0.25 * spread
    low, high = y_min - pad, y_max + pad

    # Expand search grid when the sublevel set touches boundaries.
    for _ in range(max_expand_steps + 1):
        grid = np.linspace(low, high, n_grid)
        vals = _empirical_crps_batch(y_test, grid)
        mask = vals <= q
        if not np.any(mask):
            y_star = float(grid[np.argmin(vals)])
            return y_star, y_star
        idx = np.where(mask)[0]
        touches_left = idx[0] == 0
        touches_right = idx[-1] == (n_grid - 1)
        if not (touches_left or touches_right):
            return float(grid[idx[0]]), float(grid[idx[-1]])
        width = high - low
        if touches_left:
            low -= width
        if touches_right:
            high += width

    # Fallback after max expansion: keep finite endpoints on final grid.
    return float(grid[idx[0]]), float(grid[idx[-1]])


def sr_split_conformal_intervals_batch(
    y_calibration_samples: np.ndarray,
    y_calibration_observed: np.ndarray,
    y_test_samples: np.ndarray,
    alpha: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Batch SR baseline: one conformal quantile from calibration, applied to all test points. Returns (n_test, 2)."""
    y_cal_s = np.asarray(y_calibration_samples)
    y_cal_o = np.asarray(y_calibration_observed).ravel()
    mu_cal = np.mean(y_cal_s, axis=1)
    sigma_cal = np.maximum(np.std(y_cal_s, axis=1, ddof=1), eps)
    scores = np.abs(y_cal_o - mu_cal) / sigma_cal
    q = _conformal_quantile(scores, alpha)

    y_test = np.asarray(y_test_samples)
    mu_test = np.mean(y_test, axis=1)
    sigma_test = np.maximum(np.std(y_test, axis=1, ddof=1), eps)
    lo = mu_test - q * sigma_test
    hi = mu_test + q * sigma_test
    return np.stack([lo, hi], axis=1)


def crps_split_conformal_intervals_batch(
    y_calibration_samples: np.ndarray,
    y_calibration_observed: np.ndarray,
    y_test_samples: np.ndarray,
    alpha: float,
    n_grid: int = 800,
    max_expand_steps: int = 6,
) -> np.ndarray:
    """Batch eCRPS baseline: compute cal scores once, apply per test point. Returns (n_test, 2)."""
    y_cal_s = np.asarray(y_calibration_samples)
    y_cal_o = np.asarray(y_calibration_observed).ravel()
    cal_scores = _empirical_crps_rows(y_cal_s, y_cal_o)
    q = _conformal_quantile(cal_scores, alpha)

    y_test = np.asarray(y_test_samples)
    n_test = y_test.shape[0]
    intervals = np.zeros((n_test, 2))
    for i in range(n_test):
        row = y_test[i].ravel()
        y_min, y_max = float(np.min(row)), float(np.max(row))
        spread = max(1e-8, y_max - y_min)
        pad = 0.25 * spread
        low, high = y_min - pad, y_max + pad
        for _ in range(max_expand_steps + 1):
            grid = np.linspace(low, high, n_grid)
            vals = _empirical_crps_batch(row, grid)
            mask = vals <= q
            if not np.any(mask):
                y_star = float(grid[np.argmin(vals)])
                intervals[i] = [y_star, y_star]
                break
            idx = np.where(mask)[0]
            touches_left = idx[0] == 0
            touches_right = idx[-1] == (n_grid - 1)
            if not (touches_left or touches_right):
                intervals[i] = [float(grid[idx[0]]), float(grid[idx[-1]])]
                break
            width = high - low
            if touches_left:
                low -= width
            if touches_right:
                high += width
        else:
            intervals[i] = [float(grid[idx[0]]), float(grid[idx[-1]])]
    return intervals
