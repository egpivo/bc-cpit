"""
CPIT pipeline: data split -> bias correction -> PIT calibration -> predictive CDF -> inference.
Single entry point to run the full flow and get F_tilde, intervals, etc.
"""

from typing import Callable, NamedTuple

import numpy as np

from . import (
    build_c_hat,
    central_interval_from_f_tilde,
    get_weighted_samples_at_x,
    quantile_from_f_tilde,
    randomized_pit,
)
from .bc import AffineParams, apply_affine, fit_global_affine
from .data_splitter import SplitIndices, apply_split, split_four_way


class PipelineState(NamedTuple):
    split: SplitIndices
    params: AffineParams
    c_hat_fn: Callable[[np.ndarray], np.ndarray]
    u_cal: np.ndarray  # PIT on calibration (after bias correction)


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    generator: Callable[[np.ndarray, int], np.ndarray],
    *,
    n_samples_bias: int = 500,
    n_samples_cal: int = 500,
    n_samples_test: int = 500,
    train_frac: float = 0.4,
    bias_frac: float = 0.2,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
    location_only: bool = False,
    seed: int | None = 42,
    pit_seed: int | None = None,
) -> PipelineState:
    """
    Run full CPIT pipeline:
    1. Split data
    2. Fit global affine correction y_adj = a + b*y using bias split
    3. Compute randomized PIT on calibration split from adjusted samples
    4. Build C_hat from calibration PIT values
    Returns state with split, params, c_hat_fn, and u_cal for diagnostics.
    """
    n = len(y)
    split = split_four_way(
        n,
        train_frac=train_frac,
        bias_frac=bias_frac,
        cal_frac=cal_frac,
        test_frac=test_frac,
        seed=seed,
    )
    (X_train, y_train), (X_bias, y_bias), (X_cal, y_cal), (X_test, y_test) = (
        apply_split(X, y, split)
    )

    # Bias: y_bias_samples = generator at bias-support X only (row i = samples at X_bias[i]).
    y_bias_samples = np.zeros((len(X_bias), n_samples_bias), dtype=float)
    for i in range(len(X_bias)):
        y_bias_samples[i] = np.asarray(generator(X_bias[i], n_samples_bias)).ravel()
    params = fit_global_affine(y_bias, y_bias_samples, location_only=location_only)

    # Calibration: randomized PIT with tie handling on adjusted samples.
    rng = np.random.default_rng(pit_seed)
    u_cal_list = []
    for i in range(len(X_cal)):
        samples = np.asarray(generator(X_cal[i], n_samples_cal)).ravel()
        samples_adj = apply_affine(samples, params)
        u_cal_list.append(randomized_pit(samples_adj, y_cal[i], rng=rng))
    u_cal = np.array(u_cal_list)
    c_hat_fn = build_c_hat(u_cal)

    return PipelineState(split=split, params=params, c_hat_fn=c_hat_fn, u_cal=u_cal)


def predict_interval(
    state: PipelineState,
    X: np.ndarray,
    y_samples: np.ndarray,
    alpha: float = 0.1,
    n_grid: int = 500,
) -> np.ndarray:
    """
    At test X with pre-drawn y_samples (m,), return (low, high) for (1-alpha) central interval.
    For batch X, y_samples should be (n, m); returns (n, 2).
    """
    single = y_samples.ndim == 1
    if single:
        y_samples = np.asarray(y_samples).reshape(1, -1)
    n = y_samples.shape[0]
    out = np.zeros((n, 2))
    for i in range(n):
        low, high = central_interval_from_f_tilde(
            y_samples[i],
            state.params,
            state.c_hat_fn,
            alpha,
        )
        out[i] = [low, high]
    if single:
        return out[0]
    return out


def predict_quantile(
    state: PipelineState,
    y_samples: np.ndarray,
    q: float | np.ndarray,
    n_grid: int = 500,
) -> np.ndarray:
    """Quantile(s) from F_tilde at one x (y_samples (m,))."""
    return quantile_from_f_tilde(
        y_samples,
        state.params,
        state.c_hat_fn,
        q,
        n_grid=n_grid,
    )


def get_calibrated_weighted_samples(
    state: PipelineState,
    y_samples_raw: np.ndarray,
    y_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """At one x, (y_j, w_j) from F_tilde."""
    return get_weighted_samples_at_x(
        y_samples_raw,
        state.params,
        state.c_hat_fn,
        y_grid,
    )
