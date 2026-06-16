"""Extra tests for cpit.evaluation.metrics."""

import numpy as np
import pytest

from cpit.evaluation.metrics import (
    brier_score,
    brier_score_from_samples,
    coverage,
    cramer_von_mises,
    crps_from_weighted_samples,
    evaluate_interval_methods,
    intervals_from_quantile_fn,
    intervals_from_sample_matrix,
    mean_interval_length,
    pit_ks_statistic,
    quantile_calibration_error,
    quantile_calibration_error_from_sample_matrix,
    quantile_calibration_error_from_weighted_matrix,
    reliability_diagram_data,
    spread_skill_ratio,
)


def test_coverage_perfect():
    ivs = np.array([[0.0, 2.0], [0.0, 2.0]])
    y = np.array([1.0, 1.0])
    assert coverage(ivs, y) == 1.0


def test_coverage_none():
    ivs = np.array([[3.0, 4.0]])
    y = np.array([1.0])
    assert coverage(ivs, y) == 0.0


def test_mean_interval_length():
    ivs = np.array([[0.0, 2.0], [1.0, 4.0]])
    assert mean_interval_length(ivs) == pytest.approx(2.5)


def test_crps_scalar():
    y = np.array([0.0, 1.0, 2.0])
    w = np.ones(3) / 3
    c = crps_from_weighted_samples(y, w, 1.0)
    assert c.shape == (1,)
    assert c[0] >= 0.0


def test_crps_batch():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, (5, 20))
    w = np.ones((5, 20)) / 20
    y_true = rng.normal(0, 1, 5)
    c = crps_from_weighted_samples(y, w, y_true)
    assert c.shape == (5,)
    assert np.all(c >= 0.0)


def test_crps_wrong_y_true_size():
    y = np.ones((3, 10))
    w = np.ones((3, 10)) / 10
    with pytest.raises(ValueError, match="length n"):
        crps_from_weighted_samples(y, w, np.array([1.0, 2.0]))


def test_pit_ks_uniform():
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 1, 1000)
    ks = pit_ks_statistic(u)
    assert ks < 0.05


def test_cramer_von_mises_uniform():
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 1, 1000)
    cvm = cramer_von_mises(u)
    assert 0.0 < cvm < 0.5


def test_cramer_von_mises_empty():
    assert np.isnan(cramer_von_mises(np.array([])))


def test_intervals_from_sample_matrix_shape():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, (10, 100))
    ivs = intervals_from_sample_matrix(s, alpha=0.1)
    assert ivs.shape == (10, 2)
    assert np.all(ivs[:, 0] <= ivs[:, 1])


def test_intervals_from_sample_matrix_smooth():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, (5, 50))
    ivs = intervals_from_sample_matrix(s, alpha=0.1, smooth=True)
    assert ivs.shape == (5, 2)


def test_intervals_from_sample_matrix_1d_raises():
    with pytest.raises(ValueError, match="2d"):
        intervals_from_sample_matrix(np.ones(10), alpha=0.1)


def test_quantile_calibration_error_from_sample_matrix():
    rng = np.random.default_rng(2)
    s = rng.normal(0, 1, (50, 100))
    y = rng.normal(0, 1, 50)
    tau_grid = np.linspace(0.1, 0.9, 9)
    err = quantile_calibration_error_from_sample_matrix(s, y, tau_grid)
    assert 0.0 <= err <= 1.0


def test_quantile_calibration_error_from_sample_matrix_empty():
    err = quantile_calibration_error_from_sample_matrix(
        np.zeros((0, 10)), np.zeros(0), np.array([0.5])
    )
    assert np.isnan(err)


def test_quantile_calibration_error_from_sample_matrix_shape_error():
    with pytest.raises(ValueError):
        quantile_calibration_error_from_sample_matrix(
            np.ones((5, 10)), np.ones(6), np.array([0.5])
        )


def test_quantile_calibration_error_from_weighted_matrix():
    rng = np.random.default_rng(3)
    n, m = 30, 50
    y_w = rng.normal(0, 1, (n, m))
    w = np.ones((n, m)) / m
    y_true = rng.normal(0, 1, n)
    tau_grid = np.array([0.25, 0.5, 0.75])
    err = quantile_calibration_error_from_weighted_matrix(y_w, w, y_true, tau_grid)
    assert 0.0 <= err <= 1.0


def test_quantile_calibration_error_from_weighted_matrix_empty():
    err = quantile_calibration_error_from_weighted_matrix(
        np.zeros((0, 5)), np.zeros((0, 5)), np.zeros(0), np.array([0.5])
    )
    assert np.isnan(err)


def test_quantile_calibration_error_from_weighted_matrix_shape_error():
    with pytest.raises(ValueError):
        quantile_calibration_error_from_weighted_matrix(
            np.ones((5, 10)), np.ones((4, 10)), np.ones(5), np.array([0.5])
        )


def test_quantile_calibration_error_fn():
    rng = np.random.default_rng(4)
    y_test = rng.normal(0, 1, 20)

    def q_fn(i, tau):
        return float(np.quantile(y_test, tau))

    err = quantile_calibration_error(q_fn, y_test, np.array([0.1, 0.5, 0.9]))
    assert 0.0 <= err <= 1.0


def test_quantile_calibration_error_fn_empty():
    err = quantile_calibration_error(lambda i, t: 0.0, np.array([]), np.array([0.5]))
    assert np.isnan(err)


def test_intervals_from_quantile_fn_shape():
    def q_fn(i, tau):
        return tau * 2.0 - 1.0

    ivs = intervals_from_quantile_fn(q_fn, n=5, alpha=0.1)
    assert ivs.shape == (5, 2)
    assert np.all(ivs[:, 0] < ivs[:, 1])


def test_spread_skill_ratio():
    rng = np.random.default_rng(5)
    samples = rng.normal(0, 1, (50, 20))
    y_true = rng.normal(0, 1, 50)
    ssr = spread_skill_ratio(samples, y_true)
    assert ssr > 0.0


def test_brier_score_perfect():
    p = np.array([1.0, 0.0])
    o = np.array([1.0, 0.0])
    assert brier_score(p, o) == pytest.approx(0.0)


def test_brier_score_worst():
    p = np.array([1.0, 0.0])
    o = np.array([0.0, 1.0])
    assert brier_score(p, o) == pytest.approx(1.0)


def test_brier_score_from_samples_above():
    rng = np.random.default_rng(6)
    samples = rng.normal(0, 1, (20, 100))
    y_true = rng.normal(0, 1, 20)
    bs = brier_score_from_samples(samples, y_true, threshold=0.0, event="above")
    assert 0.0 <= bs <= 1.0


def test_brier_score_from_samples_below():
    rng = np.random.default_rng(7)
    samples = rng.normal(0, 1, (20, 100))
    y_true = rng.normal(0, 1, 20)
    bs = brier_score_from_samples(samples, y_true, threshold=0.0, event="below")
    assert 0.0 <= bs <= 1.0


def test_reliability_diagram_data_shape():
    rng = np.random.default_rng(8)
    p = rng.uniform(0, 1, 100)
    o = (rng.uniform(0, 1, 100) > 0.5).astype(float)
    centers, freq, counts = reliability_diagram_data(p, o, n_bins=5)
    assert centers.shape == (5,)
    assert freq.shape == (5,)
    assert counts.shape == (5,)
    assert counts.sum() == 100


def test_reliability_diagram_last_bin_includes_one():
    p = np.array([1.0])
    o = np.array([1.0])
    _, _, counts = reliability_diagram_data(p, o, n_bins=5)
    assert counts.sum() == 1


def test_evaluate_interval_methods():
    rng = np.random.default_rng(9)
    y_test = rng.normal(0, 1, 20)

    def make_fn():
        def fn(alpha):
            return np.column_stack([y_test - 1.0, y_test + 1.0])

        return fn

    methods = [("method_a", make_fn()), ("method_b", make_fn())]
    cov, length = evaluate_interval_methods(methods, y_test, alphas=[0.1, 0.2])
    assert set(cov.keys()) == {"method_a", "method_b"}
    assert 0.1 in cov["method_a"]
    assert 0.2 in cov["method_a"]
