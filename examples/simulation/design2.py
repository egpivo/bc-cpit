"""
Design 2 (draft §5.1): x-dependent location-scale distortion.

True DGP: Y = sin(2πX) + σ(X)ε, σ(x) = 0.05 + 0.25x
Generator: Ŷ|x ~ N(0.85 sin(2πx) + 0.15x − 0.05, (0.08 + 0.10(1−x))²)
"""
from .dgp import DESIGN2
from .runner import run_replicate, ALPHAS

N_TRAIN, N_BIAS, N_CAL, N_TEST = 0, 1000, 1000, 5000
M = 100
PIT_SEED = 43


def run_design2_replicate(seed: int, **kwargs) -> dict:
    kwargs.setdefault("n_train", N_TRAIN)
    kwargs.setdefault("n_bias", N_BIAS)
    kwargs.setdefault("n_cal", N_CAL)
    kwargs.setdefault("n_test", N_TEST)
    kwargs.setdefault("m", M)
    kwargs.setdefault("pit_seed", PIT_SEED)
    return run_replicate(DESIGN2, seed, **kwargs)
