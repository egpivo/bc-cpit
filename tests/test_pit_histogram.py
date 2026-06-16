"""Tests for cpit.evaluation.pit_histogram."""

import numpy as np

from cpit.evaluation.pit_histogram import pit_histogram_counts


def test_counts_sum_to_n():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 1, 100)
    counts = pit_histogram_counts(u, n_bins=10)
    assert counts.sum() == 100


def test_counts_shape():
    u = np.linspace(0, 1, 50)
    counts = pit_histogram_counts(u, n_bins=5)
    assert counts.shape == (5,)


def test_counts_clips_out_of_range():
    u = np.array([-0.1, 0.5, 1.1])
    counts = pit_histogram_counts(u, n_bins=2)
    assert counts.sum() == 3


def test_uniform_u_roughly_flat():
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 1, 10000)
    counts = pit_histogram_counts(u, n_bins=10)
    assert counts.min() > 800
    assert counts.max() < 1200
