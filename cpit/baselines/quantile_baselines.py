"""
Quantile-based baselines: raw quantile interval (QR0/QR), CQR from samples.
CQR: Conformalized Quantile Regression — radius = conformal quantile of calibration
residuals; interval = [q_lo - radius, q_hi + radius] per test point.
"""

from typing import Tuple

import numpy as np

from cpit.bc import AffineParams, apply_affine


def _conformal_quantile_level(n: int, alpha: float) -> float:
    """Level for (1-alpha) coverage: ceil((n+1)(1-alpha))/n."""
    if n == 0:
        raise ValueError("n must be positive")
    return min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)


def quantile_interval_from_samples(
    y_samples: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    """
    Central (1-alpha) interval from raw sample quantiles (no calibration).
    QR0-style: use empirical quantiles of y_samples.
    """
    y = np.asarray(y_samples).ravel()
    low = np.quantile(y, alpha / 2)
    high = np.quantile(y, 1.0 - alpha / 2)
    return float(low), float(high)


def cqr_radius_from_calibration(
    y_calibration_samples: np.ndarray,
    y_calibration_observed: np.ndarray,
    alpha: float,
    params: AffineParams | None = None,
) -> float:
    """
    CQR radius from calibration: residual_i = max(q_lo_i - y_i, y_i - q_hi_i),
    radius = conformal (1-alpha) quantile of residuals (method="higher").
    If params is not None, apply affine to samples before taking quantiles.
    Negative radius is allowed (§4.1): a negative q shrinks an over-conservative base interval.
    """
    y_cal_s = np.asarray(y_calibration_samples)
    y_cal_o = np.asarray(y_calibration_observed).ravel()
    if y_cal_s.ndim == 1:
        y_cal_s = y_cal_s.reshape(1, -1)
    n_cal = y_cal_s.shape[0]
    if params is not None:
        y_cal_s = np.array([apply_affine(y_cal_s[i], params) for i in range(n_cal)])
    low_cal = np.quantile(y_cal_s, alpha / 2, axis=1)
    high_cal = np.quantile(y_cal_s, 1.0 - alpha / 2, axis=1)
    residuals = np.maximum(low_cal - y_cal_o, y_cal_o - high_cal)
    level = _conformal_quantile_level(n_cal, alpha)
    return float(np.quantile(residuals, level, method="higher"))


def cqr_intervals_batch(
    y_calibration_samples: np.ndarray,
    y_calibration_observed: np.ndarray,
    y_test_samples: np.ndarray,
    alpha: float,
    params: AffineParams | None = None,
) -> tuple[np.ndarray, float]:
    """
    CQR for many test points: one radius from calibration; per-test interval
    [q_lo - radius, q_hi + radius] using (optionally affine-corrected) quantiles.
    Returns (intervals (n_test, 2), radius). Negative radius allowed (§4.1).
    """
    radius = cqr_radius_from_calibration(
        y_calibration_samples,
        y_calibration_observed,
        alpha,
        params=params,
    )
    y_test_s = np.asarray(y_test_samples)
    if y_test_s.ndim == 1:
        y_test_s = y_test_s.reshape(1, -1)
    n_test = y_test_s.shape[0]
    if params is not None:
        y_test_s = np.array([apply_affine(y_test_s[i], params) for i in range(n_test)])
    low_test = np.quantile(y_test_s, alpha / 2, axis=1)
    high_test = np.quantile(y_test_s, 1.0 - alpha / 2, axis=1)
    # When radius < 0 (base over-covers), low - r > high + r; sort so [min, max]
    endpoints = np.stack([low_test - radius, high_test + radius], axis=1)
    intervals = np.sort(endpoints, axis=1)
    return intervals, radius
