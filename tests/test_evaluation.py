"""
Tests for evaluation metrics: coverage, mean_interval_length, pit_ks_statistic,
crps_from_weighted_samples, cramer_von_mises, quantile_calibration_error,
intervals_from_quantile_fn, evaluate_interval_methods, local_diagnostics.
"""

import numpy as np
import pytest

from cpit.evaluation import (
    coverage,
    cramer_von_mises,
    crps_from_weighted_samples,
    equal_mass_bin_indices,
    evaluate_interval_methods,
    intervals_from_quantile_fn,
    intervals_from_sample_matrix,
    local_diagnostics,
    mean_interval_length,
    pit_ks_statistic,
    quantile_calibration_error,
    quantile_calibration_error_from_sample_matrix,
)


def test_coverage_all_hit():
    intervals = np.array([[0.0, 10.0], [1.0, 5.0], [-1.0, 1.0]])
    y_true = np.array([5.0, 3.0, 0.0])
    assert coverage(intervals, y_true) == 1.0


def test_coverage_none_hit():
    intervals = np.array([[0.0, 1.0], [0.0, 1.0]])
    y_true = np.array([2.0, -1.0])
    assert coverage(intervals, y_true) == 0.0


def test_coverage_half():
    intervals = np.array([[0.0, 2.0], [0.0, 2.0], [10.0, 20.0]])
    y_true = np.array([1.0, 1.0, 25.0])  # two in [0,2], one outside [10,20]
    assert coverage(intervals, y_true) == pytest.approx(2.0 / 3)


def test_mean_interval_length():
    intervals = np.array([[0.0, 2.0], [1.0, 5.0], [-1.0, 1.0]])
    assert mean_interval_length(intervals) == pytest.approx((2 + 4 + 2) / 3)


def test_pit_ks_uniform():
    u = np.linspace(0.01, 0.99, 100)
    ks = pit_ks_statistic(u)
    assert ks < 0.1


def test_pit_ks_degenerate():
    u = np.ones(50) * 0.5
    ks = pit_ks_statistic(u)
    assert ks >= 0.5


def test_crps_from_weighted_samples_single():
    y = np.array([0.0, 1.0, 2.0])
    w = np.array([0.0, 1.0, 0.0])
    crps = crps_from_weighted_samples(y, w, np.array([1.0]))
    assert crps.shape == (1,)
    np.testing.assert_almost_equal(crps[0], 0.0)


def test_crps_from_weighted_samples_batch():
    n = 5
    y = np.random.randn(n, 20)
    w = np.ones((n, 20)) / 20
    y_true = np.random.randn(n)
    crps = crps_from_weighted_samples(y, w, y_true)
    assert crps.shape == (n,)
    assert np.all(np.isfinite(crps))
    assert np.all(crps >= 0)
    slow = np.array(
        [crps_from_weighted_samples(y[i], w[i], y_true[i]).item() for i in range(n)]
    )
    np.testing.assert_array_almost_equal(crps, slow, decimal=10)


def test_intervals_from_sample_matrix_matches_rowwise():
    rng = np.random.default_rng(3)
    s = rng.normal(size=(40, 25))
    alpha = 0.1
    iv = intervals_from_sample_matrix(s, alpha)
    lo_e = np.array([np.quantile(s[i], alpha / 2) for i in range(40)])
    hi_e = np.array([np.quantile(s[i], 1 - alpha / 2) for i in range(40)])
    np.testing.assert_array_almost_equal(iv[:, 0], lo_e)
    np.testing.assert_array_almost_equal(iv[:, 1], hi_e)


def test_quantile_calibration_error_from_sample_matrix_matches():
    rng = np.random.default_rng(4)
    n, m = 50, 20
    s = rng.normal(size=(n, m))
    y_test = rng.normal(size=n)
    taus = np.linspace(0.1, 0.9, 5)

    def qfn(i: int, tau: float) -> float:
        return float(np.quantile(s[i], tau))

    q1 = quantile_calibration_error(qfn, y_test, taus)
    q2 = quantile_calibration_error_from_sample_matrix(s, y_test, taus)
    assert q1 == pytest.approx(q2, rel=0, abs=1e-9)


# --- cramer_von_mises ---


def test_cramer_von_mises_empty():
    assert np.isnan(cramer_von_mises(np.array([])))


def test_cramer_von_mises_uniform():
    """Uniform-like PIT should give relatively small CvM."""
    rng = np.random.default_rng(42)
    u = np.sort(rng.uniform(0, 1, size=500))
    cvm = cramer_von_mises(u)
    assert cvm < 0.2


def test_cramer_von_mises_degenerate():
    """All PIT at 0.5: CvM should be large."""
    u = np.ones(100) * 0.5
    cvm = cramer_von_mises(u)
    # (u_i - (2i-1)/(2n))Â² with u_i=0.5, i=1..n
    n = 100
    i = np.arange(1, n + 1, dtype=float)
    expected = 1.0 / (12 * n) + np.sum((0.5 - (2 * i - 1) / (2 * n)) ** 2)
    assert cvm == pytest.approx(expected)
    assert cvm > 0.2


def test_cramer_von_mises_single():
    """Single value: Ï‰Â² = 1/12 + (u - 0.5)Â²."""
    assert cramer_von_mises(np.array([0.5])) == pytest.approx(1.0 / 12)
    assert cramer_von_mises(np.array([0.0])) == pytest.approx(1.0 / 12 + 0.25)


# --- quantile_calibration_error ---


def test_quantile_calibration_error_perfect():
    """If Q_i(Ï„) = true Ï„-quantile, empirical proportion â‰ˆ Ï„ so QErr â‰ˆ 0."""
    n = 200
    tau_grid = np.array([0.25, 0.5, 0.75])
    y_test = np.random.randn(n)

    def quantile_fn(i: int, tau: float) -> float:
        # Perfect calibration: return a value so that exactly tau fraction are below
        sorted_y = np.sort(y_test)
        idx = int(np.round((n - 1) * tau))
        return float(sorted_y[idx])

    qerr = quantile_calibration_error(quantile_fn, y_test, tau_grid)
    assert qerr >= 0
    assert qerr < 0.15  # should be small for this proxy of perfect


def test_quantile_calibration_error_constant_high():
    """Constant quantile always above y: empirical proportion = 1, so |1 - Ï„|."""
    y_test = np.array([0.0, 1.0, 2.0])
    tau_grid = np.array([0.5])

    def quantile_fn(i: int, tau: float) -> float:
        return 10.0  # always above all y

    qerr = quantile_calibration_error(quantile_fn, y_test, tau_grid)
    assert qerr == pytest.approx(0.5)  # |1 - 0.5| = 0.5


def test_quantile_calibration_error_constant_low():
    """Constant quantile always below y: empirical proportion = 0."""
    y_test = np.array([0.0, 1.0, 2.0])
    tau_grid = np.array([0.5])

    def quantile_fn(i: int, tau: float) -> float:
        return -10.0

    qerr = quantile_calibration_error(quantile_fn, y_test, tau_grid)
    assert qerr == pytest.approx(0.5)  # |0 - 0.5| = 0.5


def test_quantile_calibration_error_empty_y():
    """Empty y_test should return nan."""

    def quantile_fn(i: int, tau: float) -> float:
        return 0.0

    qerr = quantile_calibration_error(quantile_fn, np.array([]), np.array([0.5]))
    assert np.isnan(qerr)


# --- intervals_from_quantile_fn ---


def test_intervals_from_quantile_fn():
    """Central (1-Î±) interval from quantile function."""
    n = 4
    alpha = 0.10

    def quantile_fn(i: int, tau: float) -> float:
        # return constant so low = quantile_fn(i, 0.05), high = quantile_fn(i, 0.95)
        return float(i) + tau * 10

    iv = intervals_from_quantile_fn(quantile_fn, n, alpha)
    assert iv.shape == (n, 2)
    np.testing.assert_array_almost_equal(
        iv[:, 0], [0.5, 1.5, 2.5, 3.5]
    )  # alpha/2 = 0.05
    np.testing.assert_array_almost_equal(iv[:, 1], [9.5, 10.5, 11.5, 12.5])  # 0.95


# --- evaluate_interval_methods ---


def test_evaluate_interval_methods():
    """Coverage and mean length for two methods and two alphas."""
    n = 20
    y_test = np.random.randn(n)
    alphas = [0.10, 0.20]

    def wide(alpha: float):
        return np.stack([y_test - 5, y_test + 5], axis=1)

    def narrow(alpha: float):
        return np.stack([y_test - 0.5, y_test + 0.5], axis=1)

    method_list = [("wide", wide), ("narrow", narrow)]
    results_cov, results_len = evaluate_interval_methods(method_list, y_test, alphas)
    assert results_cov["wide"][0.10] == results_cov["wide"][0.20] == 1.0
    assert results_len["wide"][0.10] == pytest.approx(10.0)
    assert results_len["narrow"][0.10] == pytest.approx(1.0)
    assert results_cov["narrow"][0.10] <= 1.0


# --- equal_mass_bin_indices ---


def test_equal_mass_bin_indices():
    """Bins have roughly equal counts."""
    X = np.linspace(0, 1, 100)
    bin_idx = equal_mass_bin_indices(X, 5)
    assert bin_idx.shape == (100,)
    assert set(bin_idx) <= {0, 1, 2, 3, 4}
    counts = np.bincount(bin_idx, minlength=5)
    assert np.all(counts >= 15)  # roughly 20 per bin


# --- local_diagnostics ---


def test_local_diagnostics():
    """Returns one row per bin with CvM and coverage keys."""
    n = 100
    X = np.linspace(0, 1, n)
    y_test = np.random.randn(n)
    u_bc = np.random.uniform(0, 1, n)
    u_cpit = np.random.uniform(0, 1, n)

    def iv_bc(alpha: float):
        return np.stack([y_test - 1, y_test + 1], axis=1)

    def iv_cpit(alpha: float):
        return np.stack([y_test - 2, y_test + 2], axis=1)

    rows = local_diagnostics(
        X,
        y_test,
        u_by_method={"BC": u_bc, "CPIT": u_cpit},
        interval_fn_by_method={"BC": iv_bc, "CPIT": iv_cpit},
        alpha_diag=0.10,
        n_bins=5,
    )
    assert len(rows) == 5
    for row in rows:
        assert "bin" in row and "n" in row
        assert "CvM_BC" in row and "CvM_CPIT" in row
        assert "Cov_BC" in row and "Cov_CPIT" in row
        assert row["Cov_BC"] == 1.0 and row["Cov_CPIT"] == 1.0
