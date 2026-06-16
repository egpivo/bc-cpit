"""
PIT histogram: bin counts for calibration diagnostics.
"""

import numpy as np


def pit_histogram_counts(u: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    u: PIT values in [0,1]. Return counts in bins [0,1/n_bins), [1/n_bins, 2/n_bins), ..., [1-1/n_bins, 1].
    """
    u = np.asarray(u).ravel()
    u = np.clip(u, 0.0, 1.0)
    edges = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(u, bins=edges)
    return counts
