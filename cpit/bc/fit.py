"""Fit bias-correction parameters (draft §2.1)."""

from typing import TYPE_CHECKING

import numpy as np

from .params import AffineParams, XAffineParamsGAM

if TYPE_CHECKING:
    pass


# ── Global affine (§2.1.1) ───────────────────────────────────────────────────


def _global_quasi_negloglik(
    b0: float,
    y_obs: np.ndarray,
    mu_hat: np.ndarray,
    inv_var: np.ndarray,
) -> float:
    """Eq. (3): Σ_i [ 2 log b0 + (y_i - a0 - b0 μ_i)^2 / (b0^2 σ_i^2) ] with a0 profiled out."""
    sw = float(np.sum(inv_var))
    if sw <= 0:
        return np.inf
    a0 = float(np.sum(inv_var * (y_obs - b0 * mu_hat)) / sw)
    resid = y_obs - a0 - b0 * mu_hat
    return float(np.sum(2.0 * np.log(b0) + inv_var * (resid**2) / (b0**2)))


def _profile_b0_global(
    y_obs: np.ndarray,
    mu_hat: np.ndarray,
    inv_var: np.ndarray,
    *,
    b_lo: float,
    b_hi: float,
    objective_gap_tol: float = 1e-6,
    local_probe_rel: float = 1e-3,
) -> float:
    """Minimize quasi-log-likelihood over b0 > 0 on [b_lo, b_hi]."""
    if not (np.isfinite(b_lo) and np.isfinite(b_hi) and b_lo < b_hi):
        raise ValueError("Need finite b_lo < b_hi for b0 search.")
    from scipy.optimize import minimize_scalar

    def _objective(b: float) -> float:
        return _global_quasi_negloglik(float(b), y_obs, mu_hat, inv_var)

    res = minimize_scalar(
        _objective,
        bounds=(b_lo, b_hi),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not res.success or not np.isfinite(res.x) or not np.isfinite(res.fun):
        raise RuntimeError(f"Global scale search failed: {res.message}")

    log_x = np.log(float(res.x))
    local_span = float(max(local_probe_rel, 1e-6))
    offsets = np.linspace(-local_span, local_span, 7)
    local_candidates = np.exp(log_x + offsets)
    local_candidates = local_candidates[
        (local_candidates >= b_lo) & (local_candidates <= b_hi)
    ]
    local_candidates = np.unique(np.concatenate([local_candidates, [float(res.x)]]))
    local_best = min(_objective(float(b)) for b in local_candidates)
    objective_gap = float(res.fun - local_best)
    if objective_gap > objective_gap_tol:
        raise RuntimeError(
            f"Global scale search did not converge tightly enough: objective gap {objective_gap:.3e} "
            f"> tol {objective_gap_tol:.3e}."
        )
    return float(res.x)


def fit_global_affine(
    y_bias_observed: np.ndarray,
    y_bias_samples: np.ndarray,
    *,
    location_only: bool = False,
    b_min: float | None = None,
    b_max: float | None = None,
    objective_gap_tol: float = 1e-6,
) -> AffineParams:
    """Global affine §2.1.1 Eq. (3): estimate (â0, b̂0) on the bias split, apply via Eq. (4).

    location_only=True fixes b0=1 and solves for â0 only.
    b_min/b_max are accepted for API compatibility but unused in the global search.
    """
    y_obs = np.asarray(y_bias_observed).ravel()
    y_s = np.asarray(y_bias_samples)
    if y_s.ndim == 1:
        y_s = y_s.reshape(1, -1)
    if y_s.shape[0] != y_obs.size:
        raise ValueError(
            "y_bias_samples must have shape (n_bias, m) with n_bias = len(y_bias_observed)."
        )
    if y_s.shape[1] < 2 and not location_only:
        raise ValueError(
            "Need at least 2 generator samples per bias point for scale correction."
        )

    mu_hat = np.mean(y_s, axis=1)
    sigma2 = np.var(y_s, axis=1, ddof=1)
    inv_var = 1.0 / np.maximum(sigma2, 1e-12)
    sw = float(np.sum(inv_var))

    if location_only:
        b = 1.0
        a = float(np.sum(inv_var * (y_obs - b * mu_hat)) / sw)
        return AffineParams(a=a, b=b)

    b_hat = _profile_b0_global(
        y_obs, mu_hat, inv_var, b_lo=1e-6, b_hi=1e6, objective_gap_tol=objective_gap_tol
    )
    a_hat = float(np.sum(inv_var * (y_obs - b_hat * mu_hat)) / sw)
    return AffineParams(a=a_hat, b=b_hat)


# ── x-dependent GAM (§2.1.2) ────────────────────────────────────────────────


def _gam_terms_ps(
    n_features: int, gam_n_splines: int | None, cyclic_cols: list[int] | None = None
):
    """P-spline terms for each feature; cyclic_cols use basis='cp' (day-of-year)."""
    from pygam import s

    cyclic_set = set(cyclic_cols or [])
    kw: dict[str, int] = {}
    if gam_n_splines is not None:
        kw["n_splines"] = int(gam_n_splines)

    def _one_term(j: int):
        if j in cyclic_set:
            return s(j, basis="cp", **kw)
        return s(j, **kw)

    terms = _one_term(0)
    for j in range(1, n_features):
        terms += _one_term(j)
    return terms


def fit_x_dependent_affine_gam(
    x_bias: np.ndarray,
    y_bias_observed: np.ndarray,
    y_bias_samples: np.ndarray,
    *,
    gam_n_splines: int | None = None,
    gam_lam: float | None = None,
    b_min: float | None = None,
    b_max: float | None = None,
    max_iter: int = 10,
    tol: float = 1e-4,
    min_iter: int = 3,
    require_loss_stable: bool = True,
    loss_step_rtol: float = 1e-5,
    loss_step_atol: float = 0.005,
    min_loss_stable_iters: int = 2,
    cyclic_cols: list[int] | None = None,
) -> XAffineParamsGAM:
    """Fit x-dependent affine §2.1.2 alternating updates (Eq. 6–10).

    Implements the draft literally: no clamp on b(x), no clip on q_i, no ε² floor.
    b_min/b_max default to None (no clamp).

    Init: m^(0)(x) = â0 + b̂0 μ̂(x), h^(0)(x) = 2 log b̂0 from global Eq. (3).
    Each cycle:
      Eq. (8): weighted LinearGAM for m(·), fit_intercept=False, w_i = 1/(σ̂²_i exp(h_i)).
      Eq. (9)–(10): q_i = (y_i − m(x_i))² / σ̂²_i; GammaGAM (log link) on q_i; b(x)=sqrt(E[q|x]).

    Stops after ≥ min_iter cycles when δ ≤ tol and loss stable for min_loss_stable_iters steps.
    """
    from pygam import GammaGAM, LinearGAM

    x = np.asarray(x_bias)
    y_obs = np.asarray(y_bias_observed).ravel()
    y_s = np.asarray(y_bias_samples)
    if y_s.ndim == 1:
        y_s = y_s.reshape(1, -1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] != y_obs.size or y_s.shape[0] != y_obs.size:
        raise ValueError(
            "x_bias, y_bias_observed, y_bias_samples must have matching n_bias."
        )

    mu_hat = np.mean(y_s, axis=1)
    std_hat = np.std(y_s, axis=1, ddof=1)

    n_features = x.shape[1]
    terms_a = _gam_terms_ps(n_features, gam_n_splines, cyclic_cols=cyclic_cols)
    terms_b = _gam_terms_ps(n_features, gam_n_splines, cyclic_cols=cyclic_cols)

    lam_kw: dict[str, float] = {}
    if gam_lam is not None:
        if float(gam_lam) <= 0:
            raise ValueError("gam_lam must be positive when given.")
        lam_kw["lam"] = float(gam_lam)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if int(min_iter) < 1:
        raise ValueError("min_iter must be >= 1")
    if loss_step_rtol < 0 or loss_step_atol < 0:
        raise ValueError("loss_step_rtol and loss_step_atol must be non-negative")
    if int(min_loss_stable_iters) < 1:
        raise ValueError("min_loss_stable_iters must be >= 1")

    def _fit_gam(
        gam: "LinearGAM | GammaGAM",
        xx: np.ndarray,
        yy: np.ndarray,
        *,
        weights: np.ndarray | None = None,
    ) -> None:
        if gam_lam is None:
            gam.gridsearch(xx, yy, weights=weights, progress=False, objective="GCV")
        else:
            gam.fit(xx, yy, weights=weights)

    gp = fit_global_affine(y_obs, y_s, b_min=b_min, b_max=b_max)
    global_base_m = gp.a + gp.b * mu_hat

    m_curr = global_base_m.copy()
    log_b_curr = np.full_like(y_obs, float(np.log(gp.b)), dtype=np.float64)

    gam_a: LinearGAM | None = None
    gam_q: "GammaGAM | None" = None
    loss_prev: float | None = None
    loss_stable_streak = 0

    for _iter in range(int(max_iter)):
        # Eq. (8): w_i = 1 / (σ̂_i² exp(h(x_i))),  h = 2 log b
        w_loc = 1.0 / (std_hat**2 * np.exp(2.0 * log_b_curr))
        # fit_intercept=False: global base a₀+b₀μ̂ is subtracted from the
        # response, so the GAM fits only the smooth deviation Σ fm,l{φl(x)}.
        # At large λ, smooth terms→0 and m(x)→global_base_m (global correction).
        gam_a = LinearGAM(terms=terms_a, fit_intercept=False, **lam_kw)
        _fit_gam(gam_a, x, y_obs - global_base_m, weights=w_loc)
        m_next = global_base_m + gam_a.predict(x).ravel()

        # Eq. (9): q_i = (y_i - m(x_i))² / σ̂²_i  (literal — no clip, no ε²)
        q = (y_obs - m_next) ** 2 / std_hat**2

        # Eq. (9)–(10): Gamma GAM (log link) on q_i.
        # fit_intercept=False is equivalent to the draft's parameterization
        # h(x) = 2 log b₀ + Σ fh,l{φl(x)}: the P-spline second-difference penalty
        # has a null space containing constants, so the constant 2 log b₀ is
        # absorbed into the spline fit without being penalized — identical to
        # having an explicit unpenalized intercept. At large λ the spline
        # converges to a linear function whose intercept ≈ log(b₀²), so b(x)→b₀.
        gam_h = GammaGAM(terms=terms_b, fit_intercept=False, **lam_kw)
        _fit_gam(gam_h, x, q)
        log_b_next = 0.5 * np.log(gam_h.predict(x).ravel())
        gam_q = gam_h

        delta = max(
            float(np.max(np.abs(m_next - m_curr))),
            float(np.max(np.abs(log_b_next - log_b_curr))),
        )
        core_loss = float(
            np.sum(
                2.0 * log_b_next
                + (y_obs - m_next) ** 2 / (std_hat**2 * np.exp(2.0 * log_b_next))
            )
        )
        if loss_prev is None:
            loss_step_ok = False
        else:
            loss_step_ok = abs(core_loss - loss_prev) <= (
                float(loss_step_atol) + float(loss_step_rtol) * max(abs(core_loss), 1.0)
            )
        loss_stable_streak = loss_stable_streak + 1 if loss_step_ok else 0

        m_curr = m_next
        log_b_curr = log_b_next
        loss_prev = core_loss

        done_param = delta <= tol
        loss_stable = loss_stable_streak >= int(min_loss_stable_iters)
        if require_loss_stable:
            if (_iter + 1) >= int(min_iter) and done_param and loss_stable:
                break
        else:
            if done_param:
                break

    if gam_a is None or gam_q is None:
        raise RuntimeError("alternating GAM fit produced no models (check max_iter).")
    return XAffineParamsGAM(
        gam_a=gam_a,
        gam_q=gam_q,
        a0=float(gp.a),
        b0=float(gp.b),
        b_min=b_min,
        b_max=b_max,
    )
