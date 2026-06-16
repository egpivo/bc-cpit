"""
Design 3 (draft §5.1): Shape misspecification (skewed).

True DGP: Y = sin(2πX) + σ(X)(η−1), σ(x) = 0.10+0.20x, η ~ Exp(1)
Generator: Ŷ|x ~ N(sin(2πx), σ²(x)) — correct mean/var, wrong shape

Draft §5.4: additionally report upper-tail calibration at τ ∈ {0.96..0.99},
reported separately from the standard τ grid so QErr is not mixed.
"""
import numpy as np

from .dgp import DESIGN3
from .runner import run_replicate, ALPHAS

N_TRAIN, N_BIAS, N_CAL, N_TEST = 0, 1000, 1000, 5000
M = 100
PIT_SEED = 43

TAU_GRID_EXTRA = np.array([0.96, 0.97, 0.98, 0.99])


def run_design3_replicate(seed: int, **kwargs) -> dict:
    """Run one Design 3 replicate; upper-tail τ reported separately."""
    kwargs.setdefault("n_train", N_TRAIN)
    kwargs.setdefault("n_bias", N_BIAS)
    kwargs.setdefault("n_cal", N_CAL)
    kwargs.setdefault("n_test", N_TEST)
    kwargs.setdefault("m", M)
    kwargs.setdefault("pit_seed", PIT_SEED)
    kwargs.setdefault("tau_grid_extra", TAU_GRID_EXTRA)
    return run_replicate(DESIGN3, seed, **kwargs)
