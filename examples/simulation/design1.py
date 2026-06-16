"""
Design 1 (draft §5.1): Global location-scale distortion.

True DGP: Y|x ~ N(sin(2πx), 0.15²)
Generator: Ŷ|x ~ N((sin(2πx) − 0.25)/1.3, (0.15/1.3)²)
"""
from .dgp import DESIGN1
from .runner import run_replicate, ALPHAS

N_TRAIN, N_BIAS, N_CAL, N_TEST = 0, 1000, 1000, 5000
M = 100
PIT_SEED = 43


def run_design1_replicate(seed: int, **kwargs) -> dict:
    kwargs.setdefault("n_train", N_TRAIN)
    kwargs.setdefault("n_bias", N_BIAS)
    kwargs.setdefault("n_cal", N_CAL)
    kwargs.setdefault("n_test", N_TEST)
    kwargs.setdefault("m", M)
    kwargs.setdefault("pit_seed", PIT_SEED)
    return run_replicate(DESIGN1, seed, **kwargs)
