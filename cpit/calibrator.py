"""Conformal calibrator Ĉ (draft §2.2 Eq. (14)–(15))."""

from typing import Callable

import numpy as np

from .pit import randomized_pit


def build_c_hat(
    u_cal: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build boundary-modified calibrator Ĉ(t) per Eq. (15):
        Ĉ(t) = Ĉ⁺(t) for t ∈ (0,1],  0 for t = 0.
    Use this for predictive CDF weights w_j.
    """
    u_cal = np.sort(np.asarray(u_cal).ravel())
    n_cal = len(u_cal)

    def c_hat(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = np.searchsorted(u_cal, t, side="right")  # count u_i <= t
        result = (1.0 + out) / (n_cal + 1.0)
        return np.where(t <= 0.0, 0.0, result)  # Ĉ(0) = 0 per Eq. (15)

    return c_hat


def build_c_hat_plus(u_cal: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build rank calibrator Ĉ⁺(t) per Eq. (14):
        Ĉ⁺(t) = (1 + Σ_{i∈I_cal} 1{u_i ≤ t}) / (|I_cal| + 1),  0 ≤ t ≤ 1.
    No boundary modification at t=0. Use this for diagnostic PIT values u_cpit_*.
    """
    u_cal = np.sort(np.asarray(u_cal).ravel())
    n_cal = len(u_cal)

    def c_hat_plus(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = np.searchsorted(u_cal, t, side="right")
        return (1.0 + out) / (n_cal + 1.0)

    return c_hat_plus


def fit_conformal_calibrator(
    cal_samples_adj: np.ndarray,
    y_cal: np.ndarray,
    pit_seed: int | None = None,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Fit conformal calibrator on the calibration split (Eq. 12 + 15). Returns (u_cal, c_hat_fn)."""
    cal_samples_adj = np.asarray(cal_samples_adj)
    y_cal = np.asarray(y_cal).ravel()
    if cal_samples_adj.ndim != 2 or cal_samples_adj.shape[0] != y_cal.size:
        raise ValueError("cal_samples_adj must be (n_cal, m) and y_cal (n_cal,).")
    rng = np.random.default_rng(pit_seed)
    u_cal = np.array(
        [
            randomized_pit(cal_samples_adj[i], float(y_cal[i]), rng=rng)
            for i in range(y_cal.size)
        ]
    )
    c_hat_fn = build_c_hat(u_cal)
    return u_cal, c_hat_fn


def build_c_hat_inv(
    c_hat_fn: Callable[[np.ndarray], np.ndarray],
    n_grid: int = 10000,
) -> Callable[[np.ndarray], np.ndarray]:
    """Approximate Ĉ^{-1} via uniform grid + linear interpolation. For diagnostics only."""
    t_grid = np.linspace(0.0, 1.0, n_grid)
    c_values = np.asarray(c_hat_fn(t_grid)).ravel()
    c_values = np.clip(c_values, 0.0, 1.0)

    def c_hat_inv(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p)
        return np.interp(np.atleast_1d(p).astype(float), c_values, t_grid).reshape(
            p.shape
        )

    return c_hat_inv


def c_hat_monotone_and_bounds(
    u_cal: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], float, float]:
    """Return (C_hat, lo, hi) where lo = 1/(n+1) and hi = 1.

    Ĉ(0) = 0 per Eq. (15). For t > 0, Ĉ(t) ∈ {1/(n+1), …, 1}.
    lo = 1/(n+1) is the first positive step (not the minimum — Ĉ(0)=0 is lower).
    """
    c_hat_fn = build_c_hat(u_cal)
    n = len(np.asarray(u_cal).ravel())
    return c_hat_fn, 1.0 / (n + 1), 1.0
