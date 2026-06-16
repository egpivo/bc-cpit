"""Inference API: quantile, central interval, HDR interval, calibrated resampling (draft Â§2.3â€“2.5)."""

from typing import Callable

import numpy as np
from scipy.stats import norm as _norm

from .bc.apply import _apply_bias_correction
from .bc.params import AffineParams, XAffineParamsGAM
from .weighted_samples import f_tilde_from_samples, get_weighted_samples_at_x

_SMOOTH_GRID_PTS = 512  # resolution of the y-grid used when inverting the smoothed CDF


def _smooth_bandwidth(y: np.ndarray) -> float:
    """Normal-reference bandwidth for CDF smoothing (Eq. 30): Ï„ = 4^(1/3) * ÏƒÌ‚ * m^(-1/3)."""
    m = y.size
    s = float(np.std(y, ddof=1)) if m > 1 else 0.0
    return (4.0 ** (1.0 / 3.0)) * s * (m ** (-1.0 / 3.0))


def _require_positive_tau(tau: float, context: str) -> None:
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(
            f"{context} requires positive bandwidth tau; adjusted samples have zero or invalid standard deviation."
        )


def _smoothed_cdf(
    y_sorted: np.ndarray, w: np.ndarray, tau: float, y_grid: np.ndarray
) -> np.ndarray:
    """Eq. (29): FÌƒ(y) = Î£_j w_j Î¦((y - Y^(j)) / Ï„) evaluated on y_grid."""
    z = (y_grid[np.newaxis, :] - y_sorted[:, np.newaxis]) / tau  # (m, k)
    return (w[:, np.newaxis] * _norm.cdf(z)).sum(axis=0)  # (k,)


def quantile_from_weighted_samples(
    y: np.ndarray,
    w: np.ndarray,
    q: float | np.ndarray,
    *,
    smooth: bool = False,
) -> np.ndarray:
    """Weighted quantile Q(q) = inf{y : F(y) >= q}.

    smooth=True: uses smoothed CDF Â§2.5 Eq. (29)â€“(30), Ï„ = 4^(1/3) ÏƒÌ‚ m^(âˆ’1/3).
    """
    y, w = np.asarray(y).ravel(), np.asarray(w).ravel()
    q = np.asarray(q)
    q_flat = np.clip(np.atleast_1d(q).astype(float), 0.0, 1.0)

    if smooth:
        idx = np.argsort(y)
        y_s, w_s = y[idx], w[idx]
        tau = _smooth_bandwidth(y_s)
        _require_positive_tau(tau, "smooth CPIT quantile")
        margin = 4.0 * tau
        y_grid = np.linspace(y_s[0] - margin, y_s[-1] + margin, _SMOOTH_GRID_PTS)
        cdf_grid = _smoothed_cdf(y_s, w_s, tau, y_grid)
        out = np.interp(q_flat, cdf_grid, y_grid)
    else:
        idx = np.argsort(y)
        y_s, w_s = y[idx], w[idx]
        cdf = np.cumsum(w_s)
        j = np.searchsorted(cdf, q_flat, side="left")
        j = np.clip(j, 0, y_s.size - 1)
        out = y_s[j]

    return out.reshape(np.shape(q))


def quantile_from_weighted_samples_batched(
    y: np.ndarray,
    w: np.ndarray,
    q: float,
    *,
    smooth: bool = False,
) -> np.ndarray:
    """Row-wise weighted quantiles. y and w are (n, m); scalar q; returns (n,).

    smooth=True: uses smoothed CDF Eq. (29)â€“(30) per row.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.ndim == 1:
        return np.atleast_1d(quantile_from_weighted_samples(y, w, q, smooth=smooth))
    if y.ndim != 2:
        raise ValueError("y must be 1d or 2d.")
    n, m = y.shape
    if w.shape != (n, m):
        raise ValueError("w must match y with shape (n, m).")
    if smooth:
        return np.array(
            [
                float(quantile_from_weighted_samples(y[i], w[i], q, smooth=True))
                for i in range(n)
            ]
        )
    qv = float(np.clip(float(q), 0.0, 1.0))
    ww = w / w.sum(axis=1, keepdims=True)
    idx = np.argsort(y, axis=1)
    y_s = np.take_along_axis(y, idx, axis=1)
    w_s = np.take_along_axis(ww, idx, axis=1)
    cdf = np.cumsum(w_s, axis=1)
    j = np.sum(cdf < qv, axis=1)
    j = np.clip(j, 0, m - 1)
    return y_s[np.arange(n), j]


def central_interval_from_weighted_samples(
    y: np.ndarray,
    w: np.ndarray,
    alpha: float,
    *,
    smooth: bool = False,
) -> tuple[float, float]:
    """Central (1-alpha) interval: [Q(alpha/2), Q(1-alpha/2)]. smooth=True uses Eq. (29)â€“(30)."""
    low = float(quantile_from_weighted_samples(y, w, alpha / 2, smooth=smooth))
    high = float(quantile_from_weighted_samples(y, w, 1.0 - alpha / 2, smooth=smooth))
    return low, high


def hdr_interval_from_weighted_samples(
    y: np.ndarray,
    w: np.ndarray,
    alpha: float,
    *,
    smooth: bool = False,
) -> tuple[float, float]:
    """Highest Density Region (1-alpha) interval (paper Â§2.4, Eq. 22).

    smooth=False: discrete sliding-window HDR on sorted weighted order statistics.
    smooth=True: shortest connected interval under smoothed density Â§2.5 Eq. (29)â€“(30).
    """
    y, w = np.asarray(y).ravel(), np.asarray(w).ravel()
    w = w / w.sum()

    if not smooth:
        idx = np.argsort(y)
        y_s, w_s = y[idx], w[idx]
        m = len(y_s)
        W = np.concatenate([[0.0], np.cumsum(w_s)])  # W[j] = sum of w_s[0..j-1]
        target = 1.0 - alpha
        best_width = np.inf
        best_lo, best_hi = float(y_s[0]), float(y_s[-1])
        l = 0
        for k in range(m):
            if l < k:
                l = k
            while l < m - 1 and W[l + 1] - W[k] < target:
                l += 1
            if W[l + 1] - W[k] >= target:
                width = y_s[l] - y_s[k]
                if width < best_width:
                    best_width = width
                    best_lo, best_hi = float(y_s[k]), float(y_s[l])
            else:
                break  # total remaining weight < target; no valid window for k or beyond
        return best_lo, best_hi

    # smooth=True: Â§2.5 Eq.(29)/(30)
    m = len(y)
    if m <= 1:
        raise ValueError("smooth HDR requires at least 2 samples.")
    sigma_hat = float(np.std(y, ddof=1))
    tau = (4.0 ** (1.0 / 3.0)) * sigma_hat * (m ** (-1.0 / 3.0))
    _require_positive_tau(tau, "smooth HDR")
    margin = 4.0 * tau
    y_grid = np.linspace(
        float(y.min()) - margin, float(y.max()) + margin, _SMOOTH_GRID_PTS
    )
    dens = (
        np.sum(
            w[:, np.newaxis]
            * _norm.pdf((y_grid[np.newaxis, :] - y[:, np.newaxis]) / tau),
            axis=0,
        )
        / tau
    )
    dy = float(y_grid[1] - y_grid[0])

    F = np.concatenate([[0.0], np.cumsum(dens) * dy])
    target = 1.0 - alpha
    n_grid = len(y_grid)
    best_width = np.inf
    best_lo = best_hi = np.nan
    l = 0
    for k in range(n_grid):
        if l < k:
            l = k
        while l < n_grid - 1 and F[l + 1] - F[k] < target:
            l += 1
        if F[l + 1] - F[k] >= target:
            width = float(y_grid[l] - y_grid[k])
            if width < best_width:
                best_width = width
                best_lo, best_hi = float(y_grid[k]), float(y_grid[l])
        else:
            break
    if not np.isfinite(best_width):
        total_mass = float(F[-1])
        raise RuntimeError(
            "smooth HDR grid does not contain enough smoothed probability mass "
            f"to reach target {target:.6g}; integrated grid mass is {total_mass:.6g}."
        )
    return best_lo, best_hi


def quantile_from_f_tilde(
    y_samples_raw: np.ndarray,
    params: AffineParams | XAffineParamsGAM,
    c_hat_fn: Callable[[np.ndarray], np.ndarray],
    q: float | np.ndarray,
    y_min: float | None = None,
    y_max: float | None = None,
    n_grid: int = 500,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """Solve F_tilde(y) = q via grid + linear interpolation at one x."""
    y_samples_raw = np.asarray(y_samples_raw).ravel()
    y_adj = _apply_bias_correction(y_samples_raw, params, x)
    if y_min is None:
        y_min = float(np.min(y_adj)) - 1e-6
    if y_max is None:
        y_max = float(np.max(y_adj)) + 1e-6
    y_grid = np.linspace(y_min, y_max, n_grid)
    f_tilde = f_tilde_from_samples(y_samples_raw, params, c_hat_fn, y_grid, x)
    q = np.asarray(q)
    out = np.interp(np.atleast_1d(q), f_tilde, y_grid)
    return out.reshape(np.shape(q))


def central_interval_from_f_tilde(
    y_samples_raw: np.ndarray,
    params: AffineParams | XAffineParamsGAM,
    c_hat_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    x: np.ndarray | None = None,
) -> tuple[float, float]:
    """(1âˆ’Î±) central interval Â§2.3 Eq. (15)â€“(17) via weighted samples."""
    y_w, w_w = get_weighted_samples_at_x(
        np.asarray(y_samples_raw).ravel(), params, c_hat_fn, y_grid=None, x=x
    )
    return central_interval_from_weighted_samples(y_w, w_w, alpha)


def _c_hat_inv_exact(u_cal_sorted: np.ndarray, p: float) -> float:
    """Exact generalized inverse of Äˆ: inf{t : Äˆ(t) â‰¥ p}."""
    n = len(u_cal_sorted)
    if p <= 0.0:
        return 0.0
    k = int(np.ceil(p * (n + 1))) - 1
    if k <= 0:
        return 0.0
    return float(u_cal_sorted[min(k - 1, n - 1)])


def pit_inverted_interval(
    y_adj_sorted: np.ndarray,
    c_hat_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    *,
    smooth: bool = False,
    u_cal: np.ndarray | None = None,
) -> tuple[float, float]:
    """PIT-inverted (1-Î±) central interval Â§2.4.

    smooth=False: H_x is the empirical step CDF; equivalent to weighted step-CDF quantile.
    smooth=True: H_x is the smoothed CDF Â§2.5 Eq. (29); requires u_cal (sorted calibration PITs).
    """
    y_adj_sorted = np.sort(np.asarray(y_adj_sorted).ravel())
    m = y_adj_sorted.size

    if not smooth:
        j = np.arange(1, m + 1, dtype=float)
        c_at_j = np.asarray(c_hat_fn(j / m), dtype=float).ravel()
        w_cpit = np.diff(c_at_j, prepend=0.0)  # w_j = Äˆ(j/m) - Äˆ((j-1)/m), Äˆ(0)=0
        return central_interval_from_weighted_samples(
            y_adj_sorted, w_cpit, alpha, smooth=False
        )

    if u_cal is None:
        raise ValueError(
            "u_cal (sorted calibration PITs) must be provided when smooth=True."
        )
    u_cal_sorted = np.sort(np.asarray(u_cal).ravel())
    u_lo = _c_hat_inv_exact(u_cal_sorted, alpha / 2)
    u_hi = _c_hat_inv_exact(u_cal_sorted, 1.0 - alpha / 2)
    w_uniform = np.ones(m) / m
    y_lo = float(
        quantile_from_weighted_samples(y_adj_sorted, w_uniform, u_lo, smooth=True)
    )
    y_hi = float(
        quantile_from_weighted_samples(y_adj_sorted, w_uniform, u_hi, smooth=True)
    )
    return y_lo, y_hi


def calibrated_resample(
    y: np.ndarray,
    w: np.ndarray,
    size: int,
    seed: int | None = None,
) -> np.ndarray:
    """Resample with replacement using weights w; return size samples."""
    y, w = np.asarray(y).ravel(), np.asarray(w).ravel()
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=size, replace=True, p=w)
    return y[idx]
