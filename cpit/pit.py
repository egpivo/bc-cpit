"""Randomized PIT (draft §2.2)."""

import numpy as np


def randomized_pit(
    y_samples_adj: np.ndarray,
    y_true: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomized PIT: u = (N + V*(N*+1)) / (m+1), Eq. (9)."""
    y = np.asarray(y_samples_adj).ravel()
    m = y.size
    if m == 0:
        raise ValueError("y_samples_adj must be non-empty.")
    if rng is None:
        rng = np.random.default_rng()
    a = np.sum(y < y_true)
    b = np.sum(y == y_true)
    v = rng.uniform(0.0, 1.0)
    return (a + v * (b + 1)) / float(m + 1)


def randomized_pit_batch(
    y_samples_adj: np.ndarray,
    y_true: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Batch randomized PIT: y_samples_adj (n, m), y_true (n,) -> u (n,)."""
    y_s = np.asarray(y_samples_adj)
    y_t = np.asarray(y_true).ravel()
    if y_s.ndim != 2:
        raise ValueError("y_samples_adj must have shape (n, m).")
    if y_s.shape[0] != y_t.size:
        raise ValueError("y_samples_adj and y_true must share first dimension n.")
    rng = np.random.default_rng(seed)
    out = np.zeros(y_t.size, dtype=float)
    for i in range(y_t.size):
        out[i] = randomized_pit(y_s[i], float(y_t[i]), rng=rng)
    return out
