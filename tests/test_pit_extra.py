"""Extra tests for cpit.pit to reach coverage on all branches."""

import numpy as np
import pytest

from cpit.pit import randomized_pit, randomized_pit_batch


def test_randomized_pit_below():
    rng = np.random.default_rng(0)
    samples = np.array([1.0, 2.0, 3.0])
    u = randomized_pit(samples, 0.0, rng=rng)
    assert u == pytest.approx(0.0)


def test_randomized_pit_above():
    rng = np.random.default_rng(0)
    samples = np.array([1.0, 2.0, 3.0])
    u = randomized_pit(samples, 4.0, rng=rng)
    assert u == pytest.approx(1.0)


def test_randomized_pit_ties():
    rng = np.random.default_rng(0)
    samples = np.array([2.0, 2.0, 2.0])
    u = randomized_pit(samples, 2.0, rng=rng)
    assert 0.0 <= u <= 1.0


def test_randomized_pit_no_rng():
    samples = np.array([1.0, 2.0, 3.0])
    u = randomized_pit(samples, 1.5)
    assert 0.0 <= u <= 1.0


def test_randomized_pit_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        randomized_pit(np.array([]), 1.0)


def test_randomized_pit_batch_shape():
    rng = np.random.default_rng(1)
    y_s = rng.normal(0, 1, (10, 50))
    y_t = rng.normal(0, 1, 10)
    u = randomized_pit_batch(y_s, y_t, seed=0)
    assert u.shape == (10,)
    assert np.all(u >= 0.0) and np.all(u <= 1.0)


def test_randomized_pit_batch_wrong_ndim():
    with pytest.raises(ValueError, match="shape \\(n, m\\)"):
        randomized_pit_batch(np.ones(10), np.ones(10))


def test_randomized_pit_batch_mismatch():
    with pytest.raises(ValueError, match="first dimension"):
        randomized_pit_batch(np.ones((5, 10)), np.ones(6))
