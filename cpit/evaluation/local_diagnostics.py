"""
Local diagnostics: CvM, QErr, and coverage within covariate bins (§5.4).
"""

from typing import Callable

import numpy as np

from .metrics import (
    coverage,
    cramer_von_mises,
    quantile_calibration_error_from_sample_matrix,
    quantile_calibration_error_from_weighted_matrix,
)


def equal_mass_bin_indices(X: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Assign each row to a bin (0 .. n_bins-1) so bins have approximately equal mass.
    X: (n,) or (n, 1). Returns (n,) integer array.
    """
    x_flat = np.asarray(X).ravel()
    q_edges = np.percentile(x_flat, np.linspace(0, 100, n_bins + 1))
    q_edges[-1] += 1e-9
    return np.digitize(x_flat, q_edges[1:-1])


def local_diagnostics(
    X: np.ndarray,
    y_test: np.ndarray,
    u_by_method: dict[str, np.ndarray],
    interval_fn_by_method: dict[str, Callable[[float], np.ndarray]],
    alpha_diag: float,
    n_bins: int = 5,
    qerr_sample_by_method: dict[str, np.ndarray] | None = None,
    qerr_weighted_by_method: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    tau_grid: np.ndarray | None = None,
) -> list[dict]:
    """
    Per-bin diagnostics: CvM of PIT, coverage, and QErr for each method.

    Parameters
    ----------
    qerr_sample_by_method : dict name -> (n, m) sample matrix for QErr via empirical quantiles
    qerr_weighted_by_method : dict name -> (Yw (n,m), Ww (n,m)) for QErr via weighted quantiles
    tau_grid : τ levels for QErr (default: 0.05..0.95 step 0.05)
    """
    X = np.asarray(X)
    y_test = np.asarray(y_test).ravel()
    n = len(y_test)
    bin_idx = equal_mass_bin_indices(X, n_bins)
    if tau_grid is None:
        tau_grid = np.arange(0.05, 1.0, 0.05)
    qerr_sample_by_method = qerr_sample_by_method or {}
    qerr_weighted_by_method = qerr_weighted_by_method or {}

    x_flat = np.asarray(X).ravel()
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        row: dict = {
            "bin": b + 1,
            "n": int(mask.sum()),
            "x_min": float(x_flat[mask].min()),
            "x_max": float(x_flat[mask].max()),
        }
        for name, u in u_by_method.items():
            u = np.asarray(u).ravel()
            if len(u) != n:
                continue
            row[f"CvM_{name}"] = cramer_von_mises(u[mask])
        for name, fn in interval_fn_by_method.items():
            iv = np.asarray(fn(alpha_diag))
            if iv.shape[0] != n:
                continue
            row[f"Cov_{name}"] = coverage(iv[mask], y_test[mask])
        for name, samples in qerr_sample_by_method.items():
            if samples.shape[0] != n:
                continue
            row[f"QErr_{name}"] = quantile_calibration_error_from_sample_matrix(
                samples[mask], y_test[mask], tau_grid
            )
        for name, (yw, ww) in qerr_weighted_by_method.items():
            if yw.shape[0] != n:
                continue
            row[f"QErr_{name}"] = quantile_calibration_error_from_weighted_matrix(
                yw[mask], ww[mask], y_test[mask], tau_grid
            )
        rows.append(row)
    return rows
