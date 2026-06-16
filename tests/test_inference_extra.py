"""Extra tests for cpit.inference — covers hdr, smooth, pit_inverted_interval, batched."""

import numpy as np
import pytest

from cpit.bc.params import AffineParams
from cpit.calibrator import build_c_hat
from cpit.inference import (
    calibrated_resample,
    central_interval_from_f_tilde,
    central_interval_from_weighted_samples,
    hdr_interval_from_weighted_samples,
    pit_inverted_interval,
    quantile_from_f_tilde,
    quantile_from_weighted_samples,
    quantile_from_weighted_samples_batched,
)


@pytest.fixture(scope="module")
def c_hat_fn():
    rng = np.random.default_rng(0)
    u_cal = rng.uniform(0, 1, 200)
    return build_c_hat(u_cal)


def _uniform_ws(n=100, lo=0.0, hi=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(lo, hi, n)
    w = np.ones(n) / n
    return y, w


def test_hdr_discrete_ordering():
    y, w = _uniform_ws(50)
    lo, hi = hdr_interval_from_weighted_samples(y, w, alpha=0.1)
    assert lo <= hi


def test_hdr_smooth_ordering():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 100)
    w = np.ones(100) / 100
    lo, hi = hdr_interval_from_weighted_samples(y, w, alpha=0.1, smooth=True)
    assert lo <= hi


def test_hdr_smooth_too_few_samples():
    with pytest.raises(ValueError, match="at least 2"):
        hdr_interval_from_weighted_samples(
            np.array([1.0]), np.array([1.0]), alpha=0.1, smooth=True
        )


def test_pit_inverted_interval_smooth_no_u_cal():
    y = np.array([0.0, 0.5, 1.0])
    c_fn = lambda t: t
    with pytest.raises(ValueError, match="u_cal"):
        pit_inverted_interval(y, c_fn, alpha=0.1, smooth=True)


def test_pit_inverted_interval_discrete(c_hat_fn):
    rng = np.random.default_rng(2)
    y_adj = np.sort(rng.normal(0, 1, 50))
    lo, hi = pit_inverted_interval(y_adj, c_hat_fn, alpha=0.1, smooth=False)
    assert lo <= hi


def test_pit_inverted_interval_smooth(c_hat_fn):
    rng = np.random.default_rng(3)
    y_adj = np.sort(rng.normal(0, 1, 100))
    u_cal = rng.uniform(0, 1, 200)
    lo, hi = pit_inverted_interval(y_adj, c_hat_fn, alpha=0.1, smooth=True, u_cal=u_cal)
    assert lo <= hi


def test_quantile_from_weighted_samples_smooth():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 80)
    w = np.ones(80) / 80
    q = quantile_from_weighted_samples(y, w, 0.5, smooth=True)
    assert np.isfinite(q)


def test_quantile_from_weighted_samples_scalar_q():
    y, w = _uniform_ws(50)
    q = quantile_from_weighted_samples(y, w, 0.5)
    assert np.isfinite(q)


def test_quantile_from_weighted_samples_array_q():
    y, w = _uniform_ws(50)
    qs = quantile_from_weighted_samples(y, w, np.array([0.1, 0.5, 0.9]))
    assert qs.shape == (3,)
    assert qs[0] <= qs[1] <= qs[2]


def test_quantile_batched_1d():
    y, w = _uniform_ws(50)
    q = quantile_from_weighted_samples_batched(y, w, 0.5)
    assert q.shape == (1,)


def test_quantile_batched_2d():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, (10, 50))
    w = np.ones((10, 50)) / 50
    q = quantile_from_weighted_samples_batched(y, w, 0.5)
    assert q.shape == (10,)


def test_quantile_batched_2d_smooth():
    rng = np.random.default_rng(6)
    y = rng.normal(0, 1, (5, 50))
    w = np.ones((5, 50)) / 50
    q = quantile_from_weighted_samples_batched(y, w, 0.5, smooth=True)
    assert q.shape == (5,)


def test_quantile_batched_wrong_ndim():
    with pytest.raises(ValueError, match="1d or 2d"):
        quantile_from_weighted_samples_batched(
            np.ones((2, 3, 4)), np.ones((2, 3, 4)), 0.5
        )


def test_quantile_batched_shape_mismatch():
    with pytest.raises(ValueError, match="shape \\(n, m\\)"):
        quantile_from_weighted_samples_batched(np.ones((5, 10)), np.ones((4, 10)), 0.5)


def test_central_interval_smooth():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, 60)
    w = np.ones(60) / 60
    lo, hi = central_interval_from_weighted_samples(y, w, alpha=0.1, smooth=True)
    assert lo <= hi


def test_calibrated_resample():
    rng = np.random.default_rng(8)
    y = rng.normal(0, 1, 50)
    w = np.ones(50) / 50
    samples = calibrated_resample(y, w, size=100, seed=0)
    assert samples.shape == (100,)
    assert set(samples).issubset(set(y))


def test_quantile_from_f_tilde_monotone(c_hat_fn):
    rng = np.random.default_rng(9)
    params = AffineParams(a=0.0, b=1.0)
    y_raw = rng.normal(0, 1, 100)
    qs = quantile_from_f_tilde(y_raw, params, c_hat_fn, np.array([0.1, 0.5, 0.9]))
    assert qs[0] <= qs[1] <= qs[2]


def test_central_interval_from_f_tilde(c_hat_fn):
    rng = np.random.default_rng(10)
    params = AffineParams(a=0.0, b=1.0)
    y_raw = rng.normal(0, 1, 100)
    lo, hi = central_interval_from_f_tilde(y_raw, params, c_hat_fn, alpha=0.1)
    assert lo <= hi
