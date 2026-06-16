"""
Four-way data split: train / bias / calibration / test.
Reproducible via fixed seed.
"""

from typing import NamedTuple

import numpy as np


class SplitIndices(NamedTuple):
    """Indices for the four splits."""

    train: np.ndarray
    bias: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


def split_four_way(
    n: int,
    *,
    train_frac: float = 0.4,
    bias_frac: float = 0.2,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int | None = None,
) -> SplitIndices:
    """
    Split indices [0, n-1] into train / bias / calibration / test.
    Fractions should sum to 1.0 (or will be normalized).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.intp)
    rng.shuffle(idx)

    total = train_frac + bias_frac + cal_frac + test_frac
    if total <= 0:
        raise ValueError("At least one fraction must be positive.")
    t, b, c, te = (
        train_frac / total,
        bias_frac / total,
        cal_frac / total,
        test_frac / total,
    )

    i1 = int(n * t)
    i2 = i1 + int(n * b)
    i3 = i2 + int(n * c)

    return SplitIndices(
        train=idx[:i1],
        bias=idx[i1:i2],
        calibration=idx[i2:i3],
        test=idx[i3:],
    )


def apply_split(
    X: np.ndarray,
    y: np.ndarray,
    split: SplitIndices,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """
    Apply SplitIndices to (X, y). Returns ((X_train, y_train), (X_bias, y_bias), (X_cal, y_cal), (X_test, y_test)).
    """
    return (
        (X[split.train], y[split.train]),
        (X[split.bias], y[split.bias]),
        (X[split.calibration], y[split.calibration]),
        (X[split.test], y[split.test]),
    )
