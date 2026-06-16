"""Tests for cpit.baselines.score_baselines."""

import numpy as np
import pytest

from cpit.baselines.score_baselines import (
    _conformal_quantile,
    _empirical_crps_batch,
    _empirical_crps_rows,
    cdf_rank_split_conformal_interval,
    crps_split_conformal_interval,
    crps_split_conformal_intervals_batch,
    empirical_crps_score,
    sr_split_conformal_interval,
    sr_split_conformal_intervals_batch,
)


@pytest.fixture
def cal_data():
    rng = np.random.default_rng(42)
    n_cal, m = 40, 80
    y_cal_s = rng.normal(0, 1, (n_cal, m))
    y_cal_o = rng.normal(0, 1, n_cal)
    return y_cal_s, y_cal_o


@pytest.fixture
def test_samples():
    rng = np.random.default_rng(99)
    return rng.normal(0, 1, 80)


def test_conformal_quantile_basic():
    scores = np.array([0.1, 0.5, 0.9, 0.3])
    q = _conformal_quantile(scores, alpha=0.1)
    assert np.isfinite(q)


def test_conformal_quantile_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _conformal_quantile(np.array([]), alpha=0.1)


def test_sr_interval_ordering(cal_data, test_samples):
    y_cal_s, y_cal_o = cal_data
    lo, hi = sr_split_conformal_interval(y_cal_s, y_cal_o, test_samples, alpha=0.1)
    assert lo <= hi


def test_sr_interval_wrong_ndim(test_samples):
    with pytest.raises(ValueError, match="shape"):
        sr_split_conformal_interval(np.ones(10), np.ones(10), test_samples, alpha=0.1)


def test_sr_interval_size_mismatch(test_samples):
    with pytest.raises(ValueError, match="mismatch"):
        sr_split_conformal_interval(
            np.ones((5, 10)), np.ones(6), test_samples, alpha=0.1
        )


def test_cdf_rank_interval_ordering(cal_data, test_samples):
    y_cal_s, y_cal_o = cal_data
    lo, hi = cdf_rank_split_conformal_interval(
        y_cal_s, y_cal_o, test_samples, alpha=0.1
    )
    assert lo <= hi


def test_cdf_rank_interval_wrong_ndim(test_samples):
    with pytest.raises(ValueError, match="shape"):
        cdf_rank_split_conformal_interval(
            np.ones(10), np.ones(10), test_samples, alpha=0.1
        )


def test_cdf_rank_interval_size_mismatch(test_samples):
    with pytest.raises(ValueError, match="mismatch"):
        cdf_rank_split_conformal_interval(
            np.ones((5, 10)), np.ones(6), test_samples, alpha=0.1
        )


def test_empirical_crps_score_non_negative():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 50)
    score = empirical_crps_score(y, 0.0)
    assert score >= 0.0


def test_empirical_crps_score_perfect():
    y = np.full(100, 1.0)
    score = empirical_crps_score(y, 1.0)
    assert score == pytest.approx(0.0, abs=1e-10)


def test_empirical_crps_batch_shape():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 50)
    vals = np.array([-1.0, 0.0, 1.0])
    out = _empirical_crps_batch(y, vals)
    assert out.shape == (3,)
    assert np.all(out >= 0.0)


def test_empirical_crps_rows_shape():
    rng = np.random.default_rng(3)
    y_s = rng.normal(0, 1, (10, 50))
    y_o = rng.normal(0, 1, 10)
    out = _empirical_crps_rows(y_s, y_o)
    assert out.shape == (10,)
    assert np.all(out >= 0.0)


def test_crps_interval_ordering(cal_data, test_samples):
    y_cal_s, y_cal_o = cal_data
    lo, hi = crps_split_conformal_interval(y_cal_s, y_cal_o, test_samples, alpha=0.1)
    assert lo <= hi


def test_crps_interval_wrong_ndim(test_samples):
    with pytest.raises(ValueError, match="shape"):
        crps_split_conformal_interval(np.ones(10), np.ones(10), test_samples, alpha=0.1)


def test_crps_interval_size_mismatch(test_samples):
    with pytest.raises(ValueError, match="mismatch"):
        crps_split_conformal_interval(
            np.ones((5, 10)), np.ones(6), test_samples, alpha=0.1
        )


def test_sr_batch_shape(cal_data):
    y_cal_s, y_cal_o = cal_data
    rng = np.random.default_rng(4)
    y_test_s = rng.normal(0, 1, (8, 80))
    ivs = sr_split_conformal_intervals_batch(y_cal_s, y_cal_o, y_test_s, alpha=0.1)
    assert ivs.shape == (8, 2)
    assert np.all(ivs[:, 0] <= ivs[:, 1])


def test_crps_batch_shape(cal_data):
    y_cal_s, y_cal_o = cal_data
    rng = np.random.default_rng(5)
    y_test_s = rng.normal(0, 1, (5, 80))
    ivs = crps_split_conformal_intervals_batch(y_cal_s, y_cal_o, y_test_s, alpha=0.1)
    assert ivs.shape == (5, 2)
    assert np.all(ivs[:, 0] <= ivs[:, 1])
