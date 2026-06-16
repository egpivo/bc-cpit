"""
Tests for global affine (draft Â§2.1.1 Eq. (3)) and x-dependent GAM (draft Â§2.1.2 Eq. (6)â€“(10)).
"""

import numpy as np

import cpit.bc as bc
from cpit.bc import (
    AffineParams,
    XAffineParamsGAM,
    apply_affine,
    apply_affine_x,
    fit_global_affine,
    fit_x_dependent_affine_gam,
)


def test_global_affine_weighted_quasi_likelihood():
    """Draft Â§2.1.1 Eq. (3): (Ã¢0, bÌ‚0) by profiling a0(b0) and 1D search on b0 > 0."""
    rng = np.random.default_rng(46)
    n, m = 60, 25
    mu_true = np.linspace(0, 2, n)
    y_s = mu_true[:, None] + rng.standard_normal((n, m))
    y_obs = mu_true + 0.5 * rng.standard_normal(n)
    params = fit_global_affine(y_obs, y_s)
    mu_hat = np.mean(y_s, axis=1)
    sigma2 = np.var(y_s, axis=1, ddof=1)
    sigma2_safe = np.maximum(sigma2, 1e-12)
    inv_var = 1.0 / sigma2_safe
    b_expected = bc._profile_b0_global(y_obs, mu_hat, inv_var, b_lo=1e-6, b_hi=1e6)
    sw = float(np.sum(inv_var))
    a_expected = float(np.sum(inv_var * (y_obs - b_expected * mu_hat)) / sw)
    np.testing.assert_almost_equal(params.a, a_expected, decimal=8)
    np.testing.assert_almost_equal(params.b, b_expected, decimal=8)


def test_global_affine_location_only():
    """location_only: b=1 and a follows weighted Eq (3) formula."""
    rng = np.random.default_rng(45)
    n, m = 30, 10
    mu_true = rng.normal(0, 1, n)
    y_s = mu_true[:, None] + rng.standard_normal((n, m))
    mu_hat = np.mean(y_s, axis=1)
    y_obs = mu_hat + 0.3 * rng.standard_normal(n)
    params = fit_global_affine(y_obs, y_s, location_only=True)
    assert params.b == 1.0
    residuals = y_obs - mu_hat
    sigma2 = np.var(y_s, axis=1, ddof=1)
    sigma2_safe = np.maximum(sigma2, 1e-12)
    w = 1.0 / sigma2_safe
    a_expected = float(np.sum(w * residuals) / np.sum(w))
    assert abs(params.a - a_expected) < 1e-10


def test_apply_affine():
    """Draft Eq. (4): y_adj = Ã¢0 + bÌ‚0 y. For a=1, b=2, y_raw=[0,1,-1] â†’ [1,3,-1]."""
    params = AffineParams(a=1.0, b=2.0)
    y_raw = np.array([0.0, 1.0, -1.0])
    out = apply_affine(y_raw, params)
    np.testing.assert_array_almost_equal(out, [1.0, 3.0, -1.0])


def test_global_affine_scale_no_clamp():
    """Global scale fit should still produce a positive finite b without extra clamp."""
    rng = np.random.default_rng(7)
    n, m = 25, 10
    mu_true = np.linspace(-1, 1, n)
    y_s = mu_true[:, None] + 0.01 * rng.standard_normal((n, m))
    y_obs = mu_true + 5.0 * rng.standard_normal(n)
    params = fit_global_affine(y_obs, y_s, b_min=0.8, b_max=1.2)
    assert np.isfinite(params.b)
    assert params.b > 0.0


def test_apply_affine_x_global_fallback():
    """apply_affine_x with AffineParams delegates to Eq. (4): y_adj = Ã¢0 + bÌ‚0 y."""
    params = AffineParams(a=-4.0, b=3.0)
    y_raw = np.array([1.0, 2.0])
    out = apply_affine_x(y_raw, np.array([[0.5]]), params)
    np.testing.assert_array_almost_equal(out, [-1.0, 2.0])


def test_x_dependent_affine_gam_shape_and_apply():
    """GAM a(x): fit_x_dependent_affine_gam returns XAffineParamsGAM; apply_affine_x works."""
    rng = np.random.default_rng(123)
    n, m = 60, 15
    x_bias = rng.uniform(0, 1, (n, 1))
    mu_hat = 2 * x_bias.ravel() + np.sin(4 * np.pi * x_bias.ravel())
    sigma_hat = np.full(n, 0.5)
    y_s = mu_hat[:, None] + sigma_hat[:, None] * rng.standard_normal((n, m))
    y_obs = mu_hat + 0.3 * rng.standard_normal(n)
    params = fit_x_dependent_affine_gam(x_bias, y_obs, y_s)
    assert isinstance(params, XAffineParamsGAM)
    for x in [np.array([[0.0]]), np.array([[0.5]]), np.array([[1.0]])]:
        y_raw = np.array([1.0])
        out = apply_affine_x(y_raw, x, params)
        assert np.isfinite(out).all()
        assert out.size == 1


def test_x_dependent_affine_gam_fixed_lam():
    """gam_lam large: fixed penalty fit runs and apply_affine_x is finite."""
    rng = np.random.default_rng(301)
    n, m = 50, 12
    x_bias = rng.uniform(0, 1, (n, 1))
    mu_hat = 2 * x_bias.ravel()
    sigma_hat = np.full(n, 0.5)
    y_s = mu_hat[:, None] + sigma_hat[:, None] * rng.standard_normal((n, m))
    y_obs = mu_hat + 0.3 * rng.standard_normal(n)
    params = fit_x_dependent_affine_gam(x_bias, y_obs, y_s, gam_lam=1e6)
    assert isinstance(params, XAffineParamsGAM)
    out = apply_affine_x(np.array([0.0]), np.array([[0.4]]), params)
    assert np.isfinite(out).all()


def test_x_dependent_affine_gam_max_iter_and_tol_controls():
    """Alternating updates should produce finite output under strict iteration controls."""
    rng = np.random.default_rng(124)
    n, m = 60, 15
    x_bias = rng.uniform(0, 1, (n, 1))
    mu_hat = 2 * x_bias.ravel() + np.sin(4 * np.pi * x_bias.ravel())
    sigma_hat = np.full(n, 0.5)
    y_s = mu_hat[:, None] + sigma_hat[:, None] * rng.standard_normal((n, m))
    y_obs = mu_hat + 0.3 * rng.standard_normal(n)
    params = fit_x_dependent_affine_gam(x_bias, y_obs, y_s, max_iter=2, tol=1e-8)
    y_raw = np.array([0.25, -0.25, 0.5])
    out = apply_affine_x(y_raw, np.array([[0.4]]), params)
    assert np.isfinite(out).all()


def test_x_dependent_affine_gam_lambda_trend_edof():
    """Larger lambda should shrink effective DoF for both location and scale GAMs."""
    rng = np.random.default_rng(125)
    n, m = 80, 20
    x_bias = rng.uniform(0, 1, (n, 1))
    mu_hat = np.sin(4 * np.pi * x_bias.ravel()) + 0.5 * x_bias.ravel()
    sigma_hat = 0.2 + 0.1 * x_bias.ravel()
    y_s = mu_hat[:, None] + sigma_hat[:, None] * rng.standard_normal((n, m))
    y_obs = mu_hat + 0.35 * rng.standard_normal(n)
    p_lo = fit_x_dependent_affine_gam(x_bias, y_obs, y_s, gam_lam=0.05)
    p_hi = fit_x_dependent_affine_gam(x_bias, y_obs, y_s, gam_lam=1e6)
    assert float(p_hi.gam_a.statistics_["edof"]) < float(p_lo.gam_a.statistics_["edof"])
    assert float(p_hi.gam_q.statistics_["edof"]) < float(p_lo.gam_q.statistics_["edof"])


def test_x_dependent_affine_gam_params_structure():
    """XAffineParamsGAM has exactly (gam_a, gam_q, a0, b0, b_min, b_max) â€” no log_q_sigma2."""
    from pygam import GammaGAM

    rng = np.random.default_rng(55)
    n, m = 60, 15
    x_bias = rng.uniform(0, 1, (n, 1))
    mu_hat = 1.5 * x_bias.ravel()
    sigma_hat = np.full(n, 0.4)
    y_s = mu_hat[:, None] + sigma_hat[:, None] * rng.standard_normal((n, m))
    y_obs = mu_hat + 0.3 * rng.standard_normal(n)
    params = fit_x_dependent_affine_gam(x_bias, y_obs, y_s)
    assert isinstance(params, XAffineParamsGAM)
    assert set(params._fields) == {"gam_a", "gam_q", "a0", "b0", "b_min", "b_max"}
    assert "log_q_sigma2" not in params._fields
    assert isinstance(params.gam_q, GammaGAM)


def test_x_dependent_affine_gam_b_equals_sqrt_eq_q():
    """Eq. (10): bÌ‚(x) = sqrt(E[q|x]).  _a_b_at_x_gam must agree with gam_q.predict."""
    rng = np.random.default_rng(88)
    n, m = 120, 20
    x_bias = rng.uniform(0, 1, (n, 1))
    mu_hat = x_bias.ravel()
    sigma_hat = 0.3 + 0.2 * x_bias.ravel()
    y_s = mu_hat[:, None] + sigma_hat[:, None] * rng.standard_normal((n, m))
    y_obs = mu_hat + 0.5 * sigma_hat * rng.standard_normal(n)
    params = fit_x_dependent_affine_gam(x_bias, y_obs, y_s, max_iter=4, gam_lam=1.0)
    x_test = np.array([[0.5]])
    _, b = bc._a_b_at_x_gam(x_test, params)
    b_from_gam = float(np.sqrt(max(params.gam_q.predict(x_test).ravel()[0], 1e-12)))
    assert abs(b - b_from_gam) < 1e-10


def test_x_dependent_affine_gam_design1_scale_near_oracle():
    """Draft(6) Design1 sanity: fitted b(x) should stay near oracle scale ratio (~1.3)."""
    rng = np.random.default_rng(777)
    n, m = 400, 100
    x_bias = rng.uniform(0, 1, (n, 1))
    mu_true = np.sin(2 * np.pi * x_bias.ravel())
    sigma_true = 0.15
    sigma_gen = sigma_true / 1.3
    mu_gen = (mu_true - 0.25) / 1.3
    y_s = mu_gen[:, None] + sigma_gen * rng.standard_normal((n, m))
    y_obs = mu_true + sigma_true * rng.standard_normal(n)
    params = fit_x_dependent_affine_gam(x_bias, y_obs, y_s, gam_lam=1e-2, max_iter=8)
    x_eval = np.linspace(0, 1, 200)
    b_eval = np.array(
        [
            apply_affine_x(np.array([0.0, 1.0]), np.array([[x]]), params)[1]
            - apply_affine_x(np.array([0.0, 1.0]), np.array([[x]]), params)[0]
            for x in x_eval
        ]
    )
    assert np.isfinite(b_eval).all()
    assert float(np.mean(b_eval)) > 1.0
    assert float(np.mean(b_eval)) < 1.6
