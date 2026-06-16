"""Apply bias-correction transforms (draft §2.1 Eq. (4)–(5))."""

import numpy as np

from .params import AffineParams, XAffineParamsGAM


def apply_affine(y_raw: np.ndarray, params: AffineParams) -> np.ndarray:
    """Eq. (4): y_adj = a0 + b0 * y_raw."""
    y_raw = np.asarray(y_raw)
    return params.a + params.b * y_raw


def _clamp_b(b: float, b_min: float | None, b_max: float | None) -> float:
    if b_min is not None:
        b = max(b, b_min)
    if b_max is not None:
        b = min(b, b_max)
    return b


def _a_b_at_x_gam(x: np.ndarray, params: XAffineParamsGAM) -> tuple[float, float]:
    """Return (smooth_dev_m, b(x)) from GAM params (draft §2.1.2, Eq. (6)/(9)–(10)).

    b(x) = sqrt(E[q|x]) from gam_q (Eq. 10). GammaGAM log-link ensures E[q|x] > 0,
    so no floor is needed. b_min/b_max clamp is opt-in (None by default).
    """
    x = np.asarray(x)
    if x.ndim == 0:
        x = np.array([[float(x)]])
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    smooth_dev = float(params.gam_a.predict(x).ravel()[0])
    b = float(np.sqrt(params.gam_q.predict(x).ravel()[0]))
    b = _clamp_b(b, getattr(params, "b_min", None), getattr(params, "b_max", None))
    return smooth_dev, b


def apply_affine_x(
    y_raw: np.ndarray,
    x: np.ndarray,
    params: AffineParams | XAffineParamsGAM,
) -> np.ndarray:
    """Apply bias correction at covariate x.

    AffineParams: Eq. (4), y_adj = a0 + b0 * y_raw.
    XAffineParamsGAM: Eq. (5)–(10), y_adj = m(x) + b(x) * (y_raw - μ̂).
    """
    if isinstance(params, AffineParams):
        return apply_affine(y_raw, params)
    y_raw = np.asarray(y_raw)
    mu_hat = float(np.mean(y_raw))
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        x_arr = np.array([[float(x_arr)]])
    elif x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    smooth_dev, b = _a_b_at_x_gam(x_arr, params)
    m_x = params.a0 + params.b0 * mu_hat + smooth_dev
    return m_x + b * (y_raw - mu_hat)


def _apply_bias_correction(
    y_samples: np.ndarray,
    params: AffineParams | XAffineParamsGAM,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """Apply affine correction; x is required for XAffineParamsGAM."""
    if isinstance(params, XAffineParamsGAM):
        if x is None:
            raise ValueError(
                "x must be provided when using x-dependent affine parameters."
            )
        return apply_affine_x(y_samples, x, params)
    return apply_affine(y_samples, params)
