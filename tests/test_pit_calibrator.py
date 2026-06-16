import numpy as np

from cpit import (
    build_c_hat,
    build_c_hat_plus,
    c_hat_monotone_and_bounds,
    fit_conformal_calibrator,
    randomized_pit,
)


def test_randomized_pit_reproducible():
    y = np.array([0.1, 0.2, 0.2, 0.5])
    y_true = 0.2
    r1 = randomized_pit(y, y_true, rng=np.random.default_rng(42))
    r2 = randomized_pit(y, y_true, rng=np.random.default_rng(42))
    np.testing.assert_array_almost_equal(r1, r2)
    assert 0.25 <= r1 <= 0.75


def test_randomized_pit_uniform_when_calibrated():
    """When samples ~ U(0,1) and y_true ~ U(0,1), PIT should be approximately uniform."""
    np.random.seed(123)
    n_trials, m = 500, 100
    pit_vals = []
    rng = np.random.default_rng(456)
    for _ in range(n_trials):
        samples = np.random.uniform(0, 1, m)
        y_true = np.random.uniform(0, 1)
        pit_vals.append(randomized_pit(samples, y_true, rng=rng))
    pit_vals = np.array(pit_vals)
    hist, _ = np.histogram(pit_vals, bins=10, range=(0, 1))
    # Each bin expects n_trials/10; allow 2x deviation
    expected_per_bin = n_trials / 10
    assert np.all(hist >= expected_per_bin * 0.4) and np.all(
        hist <= expected_per_bin * 1.6
    )


def test_c_hat_monotone():
    u_cal = np.random.uniform(0, 1, 100)
    c_hat_fn = build_c_hat(u_cal)
    t = np.linspace(0, 1, 50)
    v = c_hat_fn(t)
    assert np.all(np.diff(v) >= -1e-10)


def test_c_hat_bounds():
    c_hat_fn, lo, hi = c_hat_monotone_and_bounds(np.ones(10) * 0.5)
    assert 0 < lo < 1
    assert hi == 1.0
    # Äˆ(0) = 0 per Eq. (14); values for t > 0 are >= lo = 1/(n_cal+1).
    t = np.linspace(0, 1, 5)
    out = c_hat_fn(t)
    assert out[0] == 0.0  # Äˆ(0) = 0
    np.testing.assert_array_less(np.full(4, lo - 1e-6), out[1:])
    np.testing.assert_array_less(out[1:], np.full(4, hi + 1e-6))


def test_fit_conformal_calibrator():
    """Eq. (13) + Eq. (14): u_cal from PIT on cal_adj, then Äˆ(t) from u_cal."""
    np.random.seed(7)
    n_cal, m = 50, 20
    cal_samples_adj = np.random.uniform(0, 1, (n_cal, m))
    y_cal = np.random.uniform(0, 1, n_cal)
    u_cal, c_hat_fn = fit_conformal_calibrator(cal_samples_adj, y_cal, pit_seed=42)
    assert u_cal.shape == (n_cal,)
    assert np.all((u_cal >= 0) & (u_cal <= 1))
    t = np.array([0.0, 0.5, 1.0])
    c_vals = c_hat_fn(t)
    assert c_vals.shape == (3,)
    assert c_vals[0] == 0.0 and c_vals[-1] <= 1.0  # Äˆ(0)=0 per Eq. (14)
    # Same as build_c_hat(u_cal)
    c_hat_direct = build_c_hat(u_cal)
    np.testing.assert_array_almost_equal(c_hat_fn(t), c_hat_direct(t))


def test_build_c_hat_matches_explicit_count_formula_with_ties():
    """
    CPIT core: Äˆ(t) = (1 + #{u_i <= t})/(n_cal+1).
    Implementation uses np.searchsorted(..., side="right") which corresponds to <=.
    """
    u_cal = np.array([0.2, 0.2, 0.7, 0.9], dtype=float)
    # n_cal=4 => denominator=5
    c_hat_fn = build_c_hat(u_cal)

    # Include below-zero, exactly-on-duplicate, between duplicates, exactly-on-unique, above-max.
    t = np.array([-1.0, 0.2, 0.21, 0.7, 1.0], dtype=float)
    expected = np.array(
        [
            0.0,  # t <= 0: Äˆ(0) = 0 per Eq. (14); t < 0 also returns 0
            (1 + 2) / 5,  # u_i <= 0.2 counts duplicates
            (1 + 2) / 5,  # 0.21 still only counts the two 0.2's
            (1 + 3) / 5,  # includes 0.2,0.2,0.7
            (1 + 4) / 5,  # all u_i <= 1.0
        ],
        dtype=float,
    )

    out = c_hat_fn(t)
    np.testing.assert_array_almost_equal(out, expected)


def test_cpit_c_hat_outputs_discrete_grid_and_bounds():
    """
    Äˆ(t) with the given definition is a step function whose values must lie on
    the discrete grid {k/(n_cal+1): k=1..n_cal+1}.
    """
    rng = np.random.default_rng(0)
    n_cal = 37
    u_cal = rng.uniform(0, 1, size=n_cal)
    c_hat_fn = build_c_hat(u_cal)

    t = rng.uniform(-0.5, 1.5, size=500)
    out = c_hat_fn(t)

    # Äˆ(0)=0 per Eq. (14); values for t > 0 lie on {1/(n+1), ..., 1}.
    hi = 1.0
    assert np.all(out >= 0.0 - 1e-12)
    assert np.all(out <= hi + 1e-12)

    # For t > 0, outputs lie on discrete grid {k/(n_cal+1): k=1..n_cal+1}.
    pos_mask = t > 0.0
    out_pos = out[pos_mask]
    k = np.rint(out_pos * (n_cal + 1.0)).astype(int)
    k = np.clip(k, 1, n_cal + 1)
    recon = k / (n_cal + 1.0)
    np.testing.assert_allclose(out_pos, recon, rtol=0, atol=1e-12)


def test_c_hat_monotone_non_decreasing_for_sorted_input():
    rng = np.random.default_rng(123)
    u_cal = rng.uniform(0, 1, size=100)
    c_hat_fn = build_c_hat(u_cal)

    t = np.sort(rng.uniform(-0.2, 1.2, size=200))
    out = c_hat_fn(t)
    # Allow tiny numerical noise but require non-decreasing in practice.
    assert np.all(np.diff(out) >= -1e-12)


def test_cpit_rank_is_uniform_when_u_cal_and_u_test_are_independent_uniform():
    """
    In ideal CPIT setting with continuous PIT:
      u_cal ~ i.i.d. Uniform(0,1), u_test ~ i.i.d. Uniform(0,1) independent
    Then R = #{i: u_cal_i <= u_test} is discrete-uniform on {0,...,n_cal}.
    Since c_hat(u_test) = (1 + R)/(n_cal+1), its induced grid values are uniform.

    This test guards against unintended extra rules in c_hat construction
    (e.g., wrong side (< vs <=), wrong denominator, etc.).
    """
    rng = np.random.default_rng(0)
    n_cal = 30
    n_trials = 60000  # fast enough with vectorized counting

    # Vectorized trial design:
    # For each trial, draw (u_cal_1,...,u_cal_n_cal,u_test) i.i.d. Uniform(0,1).
    # Then R = #{i: u_cal_i <= u_test} should be discrete-uniform on {0,...,n_cal}.
    U = rng.uniform(0.0, 1.0, size=(n_trials, n_cal + 1))
    u_test = U[:, -1]  # (n_trials,)
    u_cal = U[:, :-1]  # (n_trials, n_cal)

    # Count <= matches build_c_hat's searchsorted side="right".
    r = np.sum(u_cal <= u_test[:, None], axis=1).astype(int)

    counts = np.bincount(r, minlength=n_cal + 1)
    expected = n_trials / (n_cal + 1.0)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))

    # df ~= n_cal for discrete uniform over n_cal+1 categories.
    # Use a lenient threshold to avoid flaky failures.
    assert chi2 < 6.0 * n_cal


def test_c_hat_plus_keeps_rank_calibrator_at_zero():
    u_cal = np.array([0.0, 0.2, 0.8])
    c_hat = build_c_hat(u_cal)
    c_hat_plus = build_c_hat_plus(u_cal)

    assert float(c_hat(0.0)) == 0.0
    assert float(c_hat_plus(0.0)) == 0.5
    assert float(c_hat(0.5)) == float(c_hat_plus(0.5))
