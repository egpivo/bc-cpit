"""
Tests for predictive CDF: empirical_cdf_at_y, weighted_samples_from_cdf, get_weighted_samples_at_x.
"""

import numpy as np

from cpit import (
    build_c_hat,
    central_interval_from_weighted_samples,
    empirical_cdf_at_y,
    f_adj_from_samples,
    f_tilde_from_samples,
    get_weighted_samples_at_x,
    pit_inverted_interval,
    weighted_samples_from_cdf,
)
from cpit.bc import AffineParams


def test_empirical_cdf_at_y_scalar():
    y_samples = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert empirical_cdf_at_y(y_samples, 0.0) == 0.2
    assert empirical_cdf_at_y(y_samples, 0.5) == 0.6
    assert empirical_cdf_at_y(y_samples, 1.0) == 1.0
    assert empirical_cdf_at_y(y_samples, -0.1) == 0.0
    assert empirical_cdf_at_y(y_samples, 1.1) == 1.0


def test_empirical_cdf_at_y_vectorized():
    y_samples = np.array([0.0, 0.5, 1.0])
    y_query = np.array([0.0, 0.25, 0.5, 1.0])
    out = empirical_cdf_at_y(y_samples, y_query)
    np.testing.assert_array_almost_equal(out, [1.0 / 3, 1.0 / 3, 2.0 / 3, 1.0])


def test_weighted_samples_from_cdf():
    y_grid = np.array([0.0, 1.0, 2.0, 3.0])
    f_values = np.array([0.0, 0.25, 0.75, 1.0])
    y_out, w_out = weighted_samples_from_cdf(y_grid, f_values)
    np.testing.assert_array_almost_equal(y_out, y_grid)
    assert w_out.shape == (4,)
    np.testing.assert_almost_equal(w_out.sum(), 1.0)
    assert np.all(w_out >= 0)


def test_weighted_samples_from_cdf_with_prepend():
    y_grid = np.array([1.0, 2.0, 3.0])
    f_values = np.array([0.2, 0.5, 1.0])
    y_out, w_out = weighted_samples_from_cdf(y_grid, f_values, left_prepend=0.0)
    np.testing.assert_almost_equal(w_out.sum(), 1.0)
    assert np.all(w_out >= 0)


def test_get_weighted_samples_at_x_global_affine():
    """get_weighted_samples_at_x with global AffineParams returns (y_sorted, w) with sum(w)=1."""
    rng = np.random.default_rng(11)
    m = 50
    y_samples = rng.normal(0, 1, m)
    params = AffineParams(a=0.0, b=1.0)
    u_cal = np.sort(rng.uniform(0, 1, 30))
    c_hat_fn = build_c_hat(u_cal)
    y_w, w_w = get_weighted_samples_at_x(y_samples, params, c_hat_fn)
    assert len(y_w) == len(w_w)
    np.testing.assert_almost_equal(w_w.sum(), 1.0)
    assert np.all(w_w >= 0)
    assert np.all(np.diff(y_w) >= 0)  # sorted


def test_f_adj_from_samples_global():
    y_samples = np.array([1.0, 2.0, 3.0])
    params = AffineParams(a=0.0, b=1.0)
    f = f_adj_from_samples(y_samples, params, np.array([1.5]))
    np.testing.assert_almost_equal(f, 1.0 / 3)


def test_f_tilde_from_samples():
    y_samples = np.array([0.0, 0.5, 1.0])
    params = AffineParams(0.0, 1.0)
    u_cal = np.array([0.2, 0.5, 0.8])
    c_hat_fn = build_c_hat(u_cal)
    f = f_tilde_from_samples(y_samples, params, c_hat_fn, np.array([0.5]))
    assert 0 <= float(np.asarray(f).ravel()[0]) <= 1


def test_eq11_weights_match_c_hat_increments_in_unique_case():
    """
    For unique sorted y_samples, empirical F_adj(y^(j)) hits exactly j/m.
    With Äˆ(0)=0 (Eq. 14), weights are exactly:
      w_j = Äˆ(j/m) âˆ’ Äˆ((jâˆ’1)/m)
    and sum to Äˆ(1) âˆ’ Äˆ(0) = 1 âˆ’ 0 = 1 without any renormalization.

    This test verifies those weight equalities and that a step-style quantile
    from those weights matches the theoretical atom chosen by Äˆ.
    """
    m = 10
    y_samples = np.linspace(-1.0, 1.0, m)  # unique & sorted
    params = AffineParams(a=0.0, b=1.0)

    rng = np.random.default_rng(0)
    n_cal = 50
    u_cal = rng.uniform(1e-3, 1 - 1e-3, size=n_cal)
    c_hat_fn = build_c_hat(u_cal)

    y_w, w_w = get_weighted_samples_at_x(y_samples, params, c_hat_fn)
    np.testing.assert_array_equal(y_w, np.sort(y_samples))

    # Äˆ(0) = 0 per Eq. (14).
    c0 = float(np.asarray(c_hat_fn(np.array([0.0]))).ravel()[0])
    assert c0 == 0.0

    j = np.arange(1, m + 1, dtype=float)
    c_full = np.asarray(c_hat_fn(j / m)).ravel()
    c_prev = np.asarray(c_hat_fn((j - 1.0) / m)).ravel()

    # Weights are the raw increments; they sum to 1 without renormalization.
    w_expected = c_full - c_prev
    np.testing.assert_allclose(w_w, w_expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(w_expected.sum(), 1.0, atol=1e-12)

    # Step-style quantile: smallest j such that cumsum(w) >= q.
    c_cum = np.cumsum(w_expected)
    q = np.array([0.05, 0.20, 0.60])

    for qk in q:
        j_theory = int(np.searchsorted(c_full, qk, side="left")) + 1  # 1..m
        y_theory = y_samples[j_theory - 1]

        j_weighted = int(np.searchsorted(c_cum, qk, side="left")) + 1
        y_weighted = y_samples[j_weighted - 1]

        assert y_weighted == y_theory


# â”€â”€ pit_inverted_interval tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_pit_inverted_interval_equals_weighted_step_quantile():
    """smooth=False: pit_inverted_interval must be algebraically identical to
    central_interval_from_weighted_samples with CPIT weights."""
    rng = np.random.default_rng(7)
    m = 20
    y_sorted = np.sort(rng.normal(0, 1, m))
    u_cal = rng.uniform(0.05, 0.95, 100)
    c_hat_fn = build_c_hat(u_cal)

    j = np.arange(1, m + 1, dtype=float)
    c_at_j = np.asarray(c_hat_fn(j / m), dtype=float).ravel()
    w_cpit = np.diff(c_at_j, prepend=0.0)

    for alpha in [0.02, 0.05, 0.10, 0.20]:
        ref = central_interval_from_weighted_samples(
            y_sorted, w_cpit, alpha, smooth=False
        )
        got = pit_inverted_interval(y_sorted, c_hat_fn, alpha, smooth=False)
        assert (
            got[0] == ref[0] and got[1] == ref[1]
        ), f"alpha={alpha}: pit_inverted={got} != weighted_step={ref}"


def test_pit_inverted_interval_boundary_ties():
    """When u_cal has ties, the exact step inversion should still return valid bounds."""
    u_cal = np.array([0.2, 0.2, 0.5, 0.8], dtype=float)
    c_hat_fn = build_c_hat(u_cal)
    y_sorted = np.linspace(0.0, 1.0, 10)

    lo, hi = pit_inverted_interval(y_sorted, c_hat_fn, 0.10, smooth=False)
    assert lo <= hi
    assert y_sorted[0] <= lo <= y_sorted[-1]
    assert y_sorted[0] <= hi <= y_sorted[-1]


def test_pit_inverted_interval_smooth_requires_u_cal():
    """smooth=True without u_cal must raise ValueError."""
    rng = np.random.default_rng(0)
    y_sorted = np.sort(rng.normal(0, 1, 20))
    c_hat_fn = build_c_hat(rng.uniform(0, 1, 50))
    try:
        pit_inverted_interval(y_sorted, c_hat_fn, 0.10, smooth=True)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pit_inverted_interval_smooth_returns_valid_interval():
    """smooth=True must return a finite valid interval (lo <= hi).

    The widening property relative to smooth=False is not generally guaranteed
    because smooth=True uses Äˆ^{-1} from u_cal (different quantile levels than
    the CPIT-weighted step quantile), so endpoint ordering is not monotone.
    We check structural correctness instead.
    """
    rng = np.random.default_rng(42)
    m = 30
    y_sorted = np.sort(rng.normal(5, 2, m))
    u_cal = rng.uniform(0.05, 0.95, 150)
    c_hat_fn = build_c_hat(u_cal)

    lo_smo, hi_smo = pit_inverted_interval(
        y_sorted, c_hat_fn, 0.02, smooth=True, u_cal=u_cal
    )
    assert np.isfinite(lo_smo) and np.isfinite(hi_smo)
    assert lo_smo <= hi_smo
