"""cpit: bias-corrected conformal PIT calibration for sample-based generators."""

from .calibrator import (
    build_c_hat,
    build_c_hat_inv,
    build_c_hat_plus,
    c_hat_monotone_and_bounds,
    fit_conformal_calibrator,
)
from .inference import (
    calibrated_resample,
    central_interval_from_f_tilde,
    central_interval_from_weighted_samples,
    hdr_interval_from_weighted_samples,
    pit_inverted_interval,
    quantile_from_f_tilde,
    quantile_from_weighted_samples,
    quantile_from_weighted_samples_batched,
)
from .pit import randomized_pit, randomized_pit_batch
from .weighted_samples import (
    empirical_cdf_at_y,
    f_adj_from_samples,
    f_tilde_from_samples,
    get_weighted_samples_at_x,
    weighted_samples_from_cdf,
)

__all__ = [
    "randomized_pit",
    "randomized_pit_batch",
    "build_c_hat",
    "build_c_hat_plus",
    "fit_conformal_calibrator",
    "build_c_hat_inv",
    "c_hat_monotone_and_bounds",
    "empirical_cdf_at_y",
    "f_adj_from_samples",
    "f_tilde_from_samples",
    "weighted_samples_from_cdf",
    "get_weighted_samples_at_x",
    "quantile_from_weighted_samples",
    "quantile_from_weighted_samples_batched",
    "central_interval_from_weighted_samples",
    "hdr_interval_from_weighted_samples",
    "quantile_from_f_tilde",
    "central_interval_from_f_tilde",
    "pit_inverted_interval",
    "calibrated_resample",
]
