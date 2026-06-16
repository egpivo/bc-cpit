import numpy as np
import pytest

from cpit import (
    build_c_hat,
    calibrated_resample,
    central_interval_from_weighted_samples,
    hdr_interval_from_weighted_samples,
    quantile_from_f_tilde,
    quantile_from_weighted_samples,
    quantile_from_weighted_samples_batched,
)
from cpit.bc import AffineParams


def test_quantile_from_weighted_samples_batched_matches_rows():
    rng = np.random.default_rng(0)
    n, m = 30, 12
    y = rng.normal(size=(n, m))
    w = rng.uniform(size=(n, m))
    w /= w.sum(axis=1, keepdims=True)
    for qv in (0.1, 0.5, 0.9):
        bat = quantile_from_weighted_samples_batched(y, w, qv)
        slow = np.array(
            [quantile_from_weighted_samples(y[i], w[i], qv) for i in range(n)]
        )
        np.testing.assert_array_almost_equal(bat, slow.ravel(), decimal=10)


def test_quantile_from_weighted_samples():
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.5, 0.3])
    q = quantile_from_weighted_samples(y, w, 0.5)
    assert 1.5 <= q <= 2.5
    q0 = quantile_from_weighted_samples(y, w, [0.0, 0.5, 1.0])
    np.testing.assert_array_almost_equal(q0[0], 1.0)
    np.testing.assert_array_almost_equal(q0[2], 3.0)


def test_central_interval_from_weighted_samples():
    y = np.linspace(0, 10, 100)
    w = np.ones(100) / 100
    low, high = central_interval_from_weighted_samples(y, w, 0.1)
    assert low < high
    assert 0.4 < low < 0.6  # ~0.5
    assert 9.4 < high < 9.6  # ~9.5


def test_quantile_vs_weighted_consistency():
    # F_tilde from many samples ~ weighted quantile
    np.random.seed(1)
    y_samples = np.random.randn(500)
    params = AffineParams(1.0, 0.0)
    u_cal = np.random.uniform(0, 1, 200)
    c_hat_fn = build_c_hat(u_cal, seed=2)
    q_direct = quantile_from_f_tilde(y_samples, params, c_hat_fn, 0.5)
    from cpit import get_weighted_samples_at_x

    y_pts, w_pts = get_weighted_samples_at_x(y_samples, params, c_hat_fn)
    q_w = quantile_from_weighted_samples(y_pts, w_pts, 0.5)
    # Should be close (same distribution)
    assert abs(q_direct - q_w) < 0.5


def test_calibrated_resample():
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([0.0, 1.0, 0.0])  # all mass at 2
    out = calibrated_resample(y, w, 10, seed=0)
    np.testing.assert_array_almost_equal(out, np.full(10, 2.0))


def test_smooth_intervals_reject_zero_bandwidth():
    y = np.ones(5)
    w = np.ones(5) / 5

    with pytest.raises(ValueError, match="positive bandwidth tau"):
        quantile_from_weighted_samples(y, w, 0.5, smooth=True)
    with pytest.raises(ValueError, match="positive bandwidth tau"):
        hdr_interval_from_weighted_samples(y, w, 0.1, smooth=True)
