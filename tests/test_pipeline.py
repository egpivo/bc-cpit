"""Tests for cpit.pipeline."""

import numpy as np
import pytest

from cpit.pipeline import (
    PipelineState,
    get_calibrated_weighted_samples,
    predict_interval,
    predict_quantile,
    run_pipeline,
)


def _make_generator(a=1.0, b=0.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)

    def gen(x, n):
        return a * x + b + rng.normal(0, noise, size=n)

    return gen


@pytest.fixture(scope="module")
def pipeline_state():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.uniform(0, 1, (n, 1))
    y = X[:, 0] * 2 + rng.normal(0, 0.3, n)
    gen = _make_generator(a=2.0, noise=0.3)
    return run_pipeline(X, y, gen, n_samples_bias=50, n_samples_cal=50, seed=42)


def test_run_pipeline_returns_state(pipeline_state):
    assert isinstance(pipeline_state, PipelineState)


def test_pipeline_u_cal_in_unit_interval(pipeline_state):
    u = pipeline_state.u_cal
    assert u.ndim == 1
    assert np.all(u >= 0.0) and np.all(u <= 1.0)


def test_pipeline_c_hat_monotone(pipeline_state):
    t = np.linspace(0, 1, 50)
    vals = pipeline_state.c_hat_fn(t)
    assert np.all(np.diff(vals) >= -1e-12)


def test_predict_interval_single(pipeline_state):
    rng = np.random.default_rng(1)
    samples = rng.normal(1.0, 0.3, 100)
    result = predict_interval(pipeline_state, None, samples, alpha=0.1)
    assert result.shape == (2,)
    assert result[0] < result[1]


def test_predict_interval_batch(pipeline_state):
    rng = np.random.default_rng(2)
    samples = rng.normal(1.0, 0.3, (5, 100))
    result = predict_interval(pipeline_state, None, samples, alpha=0.1)
    assert result.shape == (5, 2)
    assert np.all(result[:, 0] < result[:, 1])


def test_predict_quantile(pipeline_state):
    rng = np.random.default_rng(3)
    samples = rng.normal(1.0, 0.3, 100)
    q = predict_quantile(pipeline_state, samples, 0.5)
    assert np.isfinite(q)


def test_predict_quantile_array(pipeline_state):
    rng = np.random.default_rng(4)
    samples = rng.normal(1.0, 0.3, 100)
    q = predict_quantile(pipeline_state, samples, np.array([0.1, 0.5, 0.9]))
    assert q.shape == (3,)
    assert q[0] <= q[1] <= q[2]


def test_get_calibrated_weighted_samples(pipeline_state):
    rng = np.random.default_rng(5)
    samples = rng.normal(1.0, 0.3, 100)
    y_w, w_w = get_calibrated_weighted_samples(pipeline_state, samples)
    assert y_w.shape == w_w.shape
    assert np.isclose(w_w.sum(), 1.0, atol=1e-9)


def test_run_pipeline_location_only(tmp_path):
    rng = np.random.default_rng(7)
    n = 100
    X = rng.uniform(0, 1, (n, 1))
    y = X[:, 0] + rng.normal(0, 0.2, n)
    gen = _make_generator(a=1.0, noise=0.2, seed=7)
    state = run_pipeline(
        X, y, gen, n_samples_bias=30, n_samples_cal=30, location_only=True, seed=7
    )
    assert state.params.b == pytest.approx(1.0)
