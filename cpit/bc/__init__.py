"""Bias correction subpackage (draft §2.1)."""

from .apply import (
    _a_b_at_x_gam,
    _apply_bias_correction,
    _clamp_b,
    apply_affine,
    apply_affine_x,
)
from .fit import (
    _gam_terms_ps,
    _profile_b0_global,
    fit_global_affine,
    fit_x_dependent_affine_gam,
)
from .params import AffineParams, XAffineParamsGAM

__all__ = [
    "AffineParams",
    "XAffineParamsGAM",
    "fit_global_affine",
    "fit_x_dependent_affine_gam",
    "apply_affine",
    "apply_affine_x",
    "_apply_bias_correction",
    "_a_b_at_x_gam",
    "_clamp_b",
    "_gam_terms_ps",
    "_profile_b0_global",
]
