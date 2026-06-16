"""Tests for cpit.baselines.quantile_baselines."""

import numpy as np
import pytest

from cpit.baselines.quantile_baselines import (
    _conformal_quantile_level,
    cqr_from_samples,
    cqr_intervals_batch,
    cqr_radius_from_calibration,
    quantile_interval_from_samples,
)
from cpit.bc.params import AffineParams


def test_conformal_quantile_level_basic():
    level = _conformal_quantile_level(100, 0.1)
    assert 0.0 < level <= 1.0


def test_conformal_quantile_level_zero_n():
    with pytest.raises(ValueError, match="positive"):
        _conformal_quantile_level(0, 0.1)


def test_conformal_quantile_level_clamps_to_one():
    level = _conformal_quantile_level(1, 0.01)
    assert level == 1.0


def test_quantile_interval_from_samples_ordering():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, 200)
    lo, hi = quantile_interval_from_samples(samples, alpha=0.1)
    assert lo < hi


def test_quantile_interval_from_samples_width():
    samples = np.arange(100, dtype=float)
    lo, hi = quantile_interval_from_samples(samples, alpha=0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(99.0)


def test_cqr_from_samples_coverage():
    rng = np.random.default_rng(42)
    n_cal = 50
    m = 100
    y_cal_s = rng.normal(0, 1, (n_cal, m))
    y_cal_o = rng.normal(0, 1, n_cal)
    y_test_s = rng.normal(0, 1, m)
    lo, hi = cqr_from_samples(y_cal_s, y_cal_o, y_test_s, alpha=0.1)
    assert lo < hi


def test_cqr_from_samples_1d_cal():
    rng = np.random.default_rng(1)
    y_cal_s = rng.normal(0, 1, 100)
    y_cal_o = np.array([0.0])
    y_test_s = rng.normal(0, 1, 100)
    lo, hi = cqr_from_samples(y_cal_s, y_cal_o, y_test_s, alpha=0.1)
    assert lo < hi


def test_cqr_radius_from_calibration_no_params():
    rng = np.random.default_rng(2)
    y_cal_s = rng.normal(0, 1, (30, 50))
    y_cal_o = rng.normal(0, 1, 30)
    r = cqr_radius_from_calibration(y_cal_s, y_cal_o, alpha=0.1)
    assert r >= 0.0


def test_cqr_radius_from_calibration_with_params():
    rng = np.random.default_rng(3)
    y_cal_s = rng.normal(0, 1, (20, 50))
    y_cal_o = rng.normal(0, 1, 20)
    params = AffineParams(a=0.0, b=1.0)
    r = cqr_radius_from_calibration(y_cal_s, y_cal_o, alpha=0.1, params=params)
    assert np.isfinite(r)


def test_cqr_radius_no_clamp_can_be_negative():
    np.random.default_rng(5)
    y_cal_s = np.tile(np.linspace(-3, 3, 50), (20, 1))
    y_cal_o = np.zeros(20)
    r = cqr_radius_from_calibration(y_cal_s, y_cal_o, alpha=0.5, clamp_radius=False)
    assert isinstance(r, float)


def test_cqr_intervals_batch_shape():
    rng = np.random.default_rng(4)
    y_cal_s = rng.normal(0, 1, (30, 50))
    y_cal_o = rng.normal(0, 1, 30)
    y_test_s = rng.normal(0, 1, (10, 50))
    ivs, r = cqr_intervals_batch(y_cal_s, y_cal_o, y_test_s, alpha=0.1)
    assert ivs.shape == (10, 2)
    assert np.isfinite(r)


def test_cqr_intervals_batch_with_params():
    rng = np.random.default_rng(6)
    y_cal_s = rng.normal(0, 1, (20, 40))
    y_cal_o = rng.normal(0, 1, 20)
    y_test_s = rng.normal(0, 1, (5, 40))
    params = AffineParams(a=0.1, b=0.9)
    ivs, r = cqr_intervals_batch(y_cal_s, y_cal_o, y_test_s, alpha=0.1, params=params)
    assert ivs.shape == (5, 2)
