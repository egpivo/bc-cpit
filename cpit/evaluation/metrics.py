"""
Evaluation metrics: coverage, interval length, CRPS, PIT-KS, CvM, QErr.
"""

from typing import Callable

import numpy as np

from cpit.inference import quantile_from_weighted_samples_batched


def coverage(
    intervals: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    intervals: (n, 2) with (low, high) per row; y_true: (n,).
    Return fraction of indices i with low <= y_true[i] <= high.
    """
    intervals = np.asarray(intervals)
    y_true = np.asarray(y_true).ravel()
    low, high = intervals[:, 0], intervals[:, 1]
    return float(np.mean((y_true >= low) & (y_true <= high)))


def mean_interval_length(intervals: np.ndarray) -> float:
    """intervals: (n, 2). Return mean(high - low)."""
    intervals = np.asarray(intervals)
    return float(np.mean(intervals[:, 1] - intervals[:, 0]))


def crps_from_weighted_samples(
    y: np.ndarray,
    w: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    CRPS for weighted empirical CDF vs observation. Per-row: y (m,), w (m,), y_true scalar.
    Or broadcast: y (n,m), w (n,m), y_true (n,) -> (n,).
    """
    y = np.asarray(y)
    w = np.asarray(w)
    y_true = np.asarray(y_true)
    if y.ndim == 1:
        y = y.reshape(1, -1)
        w = w.reshape(1, -1)
        y_true = np.atleast_1d(y_true)
    n = y.shape[0]
    ww = w / w.sum(axis=1, keepdims=True)
    o = np.asarray(y_true).ravel()
    if o.size == 1:
        o = np.full(n, o[0], dtype=float)
    elif o.size != n:
        raise ValueError("y_true must be length n or a scalar when y is (n, m).")
    term1 = np.sum(ww * np.abs(y - o[:, np.newaxis]), axis=1)
    # Sort each row and reorder weights accordingly
    sort_idx = np.argsort(y, axis=1)
    y_sorted = np.take_along_axis(y, sort_idx, axis=1)
    w_sorted = np.take_along_axis(ww, sort_idx, axis=1)
    W_cum = np.cumsum(w_sorted, axis=1)
    term2 = np.sum(w_sorted * y_sorted * (2 * W_cum - w_sorted - 1), axis=1)
    return term1 - term2


def pit_ks_statistic(u: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic of PIT values u vs Uniform(0,1).
    KS = max_t | empirical_cdf(u)(t) - t |.
    """
    u = np.asarray(u).ravel()
    u = np.sort(u)
    n = len(u)
    # t: [0, u[0], ..., u[n-1], 1] length n+2
    t = np.concatenate([[0], u, [1]])
    # ecdf at those points: 0, 1/n, 2/n, ..., 1, 1
    ecdf = np.concatenate([[0], np.arange(1, n + 1) / n, [1]])
    return float(np.max(np.abs(ecdf - t)))


def cramer_von_mises(u: np.ndarray) -> float:
    """
    Cramér–von Mises statistic: distance of PIT values from Unif(0,1).
    ω² = 1/(12n) + Σ (U_(i) − (2i−1)/(2n))² for sorted U_(i).
    """
    u = np.sort(np.asarray(u).ravel())
    n = len(u)
    if n == 0:
        return np.nan
    i = np.arange(1, n + 1, dtype=float)
    return float(1.0 / (12 * n) + np.sum((u - (2 * i - 1) / (2 * n)) ** 2))


def intervals_from_sample_matrix(
    samples: np.ndarray,
    alpha: float,
    *,
    smooth: bool = False,
) -> np.ndarray:
    """
    Central (1−α) intervals from an (n, m) matrix of i.i.d.-style samples per row.

    With ``smooth=False`` (default): uses ``np.quantile`` (empirical order statistics).
    With ``smooth=True``: uses the Section 2.4 kernel-smoothed CDF (Eq. 19–20) with
    uniform weights w_j = 1/m; beneficial when m is small and tail quantiles are unreliable.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be 2d (n, m).")
    if smooth:
        n, m = samples.shape
        w_uniform = np.full(m, 1.0 / m)
        lo = quantile_from_weighted_samples_batched(
            samples, np.tile(w_uniform, (n, 1)), alpha / 2.0, smooth=True
        )
        hi = quantile_from_weighted_samples_batched(
            samples, np.tile(w_uniform, (n, 1)), 1.0 - alpha / 2.0, smooth=True
        )
    else:
        lo = np.quantile(samples, alpha / 2.0, axis=1)
        hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)
    return np.stack([lo, hi], axis=1)


def quantile_calibration_error_from_sample_matrix(
    samples: np.ndarray,
    y_test: np.ndarray,
    tau_grid: np.ndarray,
) -> float:
    """
    QErr with empirical row quantiles from ``samples`` (n, m), vectorized over rows per τ.
    """
    samples = np.asarray(samples, dtype=float)
    y_test = np.asarray(y_test).ravel()
    if samples.ndim != 2 or samples.shape[0] != len(y_test):
        raise ValueError("samples must be (n, m) with n = len(y_test).")
    n = len(y_test)
    if n == 0:
        return np.nan
    errs = []
    for tau in np.asarray(tau_grid).ravel():
        q_vals = np.quantile(samples, float(tau), axis=1)
        emp = np.mean(y_test <= q_vals)
        errs.append(np.abs(emp - float(tau)))
    return float(np.mean(errs))


def quantile_calibration_error_from_weighted_matrix(
    y_weighted: np.ndarray,
    w: np.ndarray,
    y_test: np.ndarray,
    tau_grid: np.ndarray,
) -> float:
    """QErr using ``quantile_from_weighted_samples_batched`` per τ."""
    y_test = np.asarray(y_test).ravel()
    n = len(y_test)
    if n == 0:
        return np.nan
    y_weighted = np.asarray(y_weighted, dtype=float)
    w = np.asarray(w, dtype=float)
    if y_weighted.shape != w.shape or y_weighted.shape[0] != n:
        raise ValueError("y_weighted and w must be (n, m) with n = len(y_test).")
    errs = []
    for tau in np.asarray(tau_grid).ravel():
        q_vals = quantile_from_weighted_samples_batched(y_weighted, w, float(tau))
        emp = np.mean(y_test <= q_vals)
        errs.append(np.abs(emp - float(tau)))
    return float(np.mean(errs))


def quantile_calibration_error(
    quantile_fn: Callable[[int, float], float],
    y_test: np.ndarray,
    tau_grid: np.ndarray,
) -> float:
    """
    QErr = (1/|T|) Σ_τ | (1/n) Σ_i 1{y_i ≤ Q_i(τ)} − τ |.
    quantile_fn(i, tau) returns the tau-quantile at test point i.
    y_test: (n,) observed responses; tau_grid: quantile levels in (0,1).
    """
    y_test = np.asarray(y_test).ravel()
    n = len(y_test)
    if n == 0:
        return np.nan
    errs = []
    for tau in tau_grid:
        q_vals = np.array([quantile_fn(i, float(tau)) for i in range(n)])
        emp = np.mean(y_test <= q_vals)
        errs.append(np.abs(emp - tau))
    return float(np.mean(errs))


def intervals_from_quantile_fn(
    quantile_fn: Callable[[int, float], float],
    n: int,
    alpha: float,
) -> np.ndarray:
    """
    Central (1−α) intervals from a per-point quantile function.
    quantile_fn(i, tau) returns the tau-quantile at point i.
    Returns (n, 2) with [low, high] per row.
    """
    low = np.array([quantile_fn(i, alpha / 2) for i in range(n)])
    high = np.array([quantile_fn(i, 1.0 - alpha / 2) for i in range(n)])
    return np.stack([low, high], axis=1)


def spread_skill_ratio(
    samples: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Spread-skill ratio: mean(ensemble_spread) / RMSE.

    spread_i = std(samples[i]), skill = sqrt(mean((mean_i - y_i)^2)).
    SSR = 1 indicates perfect reliability of ensemble dispersion.
    """
    samples = np.asarray(samples, dtype=float)
    y_true = np.asarray(y_true).ravel()
    spread = np.std(samples, axis=1, ddof=1).mean()
    mu = np.mean(samples, axis=1)
    rmse = np.sqrt(np.mean((mu - y_true) ** 2))
    return float(spread / max(rmse, 1e-12))


def brier_score(
    prob_event: np.ndarray,
    observed_event: np.ndarray,
) -> float:
    """Brier score: mean((p_i - o_i)^2) where o_i in {0, 1}.

    Lower is better; 0 = perfect.
    """
    p = np.asarray(prob_event).ravel()
    o = np.asarray(observed_event).ravel()
    return float(np.mean((p - o) ** 2))


def brier_score_from_samples(
    samples: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    event: str = "above",
) -> float:
    """Brier score for P(Y > threshold) (event='above') or P(Y < threshold) (event='below')."""
    samples = np.asarray(samples, dtype=float)
    y_true = np.asarray(y_true).ravel()
    if event == "above":
        prob = np.mean(samples > threshold, axis=1)
        obs = (y_true > threshold).astype(float)
    else:
        prob = np.mean(samples < threshold, axis=1)
        obs = (y_true < threshold).astype(float)
    return brier_score(prob, obs)


def reliability_diagram_data(
    prob_event: np.ndarray,
    observed_event: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute binned data for a reliability diagram.

    Returns (bin_centers, observed_freq, bin_counts).
    """
    p = np.asarray(prob_event).ravel()
    o = np.asarray(observed_event).ravel()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    obs_freq = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = (p >= bin_edges[b]) & (p < bin_edges[b + 1])
        if b == n_bins - 1:
            mask |= p == bin_edges[b + 1]
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            obs_freq[b] = o[mask].mean()
    return centers, obs_freq, counts


def evaluate_interval_methods(
    method_list: list[tuple[str, Callable[[float], np.ndarray]]],
    y_test: np.ndarray,
    alphas: list[float] | np.ndarray,
) -> tuple[dict[str, dict[float, float]], dict[str, dict[float, float]]]:
    """
    For each (name, fn) in method_list, fn(alpha) returns (n, 2) intervals.
    Returns (results_cov, results_len): each dict[name][alpha] = value.
    """
    y_test = np.asarray(y_test).ravel()
    results_cov: dict[str, dict[float, float]] = {}
    results_len: dict[str, dict[float, float]] = {}
    for alpha in alphas:
        for name, fn in method_list:
            iv = np.asarray(fn(float(alpha)))
            results_cov.setdefault(name, {})[float(alpha)] = coverage(iv, y_test)
            results_len.setdefault(name, {})[float(alpha)] = mean_interval_length(iv)
    return results_cov, results_len
