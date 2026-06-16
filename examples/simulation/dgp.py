"""
Data-Generating Processes for simulation designs (draft §5.1).

Each DGP defines:
  - generate_y(X, rng) → Y array  (true response)
  - sample_generator(x, m, rng) → m samples from the misspecified generator
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class DGP:
    """A data-generating process for simulation studies."""
    name: str
    generate_y: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    sample_generator: Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def _mu_sin(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * np.asarray(x))


# ── Design 1: Global location-scale distortion ──────────────────────────────
# Y = sin(2πX) + ε,  ε ~ N(0, 0.15²)
# Generator: Ŷ|x ~ N((sin(2πx)−0.25)/1.3, (0.15/1.3)²)

def _design1_generate_y(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _mu_sin(X).ravel() + 0.15 * rng.standard_normal(len(X))


def _design1_sample_generator(x: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    x = np.atleast_1d(x)
    mean = ((_mu_sin(x) - 0.25) / 1.3).ravel()
    std = 0.15 / 1.3
    return mean + std * rng.standard_normal(m)


DESIGN1 = DGP(
    name="Design 1: global location-scale distortion",
    generate_y=_design1_generate_y,
    sample_generator=_design1_sample_generator,
)


# ── Design 2: x-dependent location-scale distortion ─────────────────────────
# Y = sin(2πX) + σ(X)ε,  σ(x) = 0.05+0.25x,  ε ~ N(0,1)
# Generator: Ŷ|x ~ N(0.85·sin(2πx)+0.15x−0.05, (0.08+0.10(1−x))²)

def _design2_generate_y(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(X)
    sigma = 0.05 + 0.25 * X.ravel()
    return _mu_sin(X).ravel() + sigma * rng.standard_normal(len(X))


def _design2_sample_generator(x: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    x = np.atleast_1d(x)
    mean = (0.85 * _mu_sin(x) + 0.15 * x - 0.05).ravel()
    std = (0.08 + 0.10 * (1 - np.asarray(x))).ravel()
    return mean + std * rng.standard_normal(m)


DESIGN2 = DGP(
    name="Design 2: x-dependent location-scale distortion",
    generate_y=_design2_generate_y,
    sample_generator=_design2_sample_generator,
)


# ── Design 3: Shape misspecification (skewed) ────────────────────────────────
# Y = sin(2πX) + σ(X)(η−1),  σ(x) = 0.10+0.20x,  η ~ Exp(1)
# Generator: Ŷ|x ~ N(sin(2πx), σ²(x))  — matches mean/var, wrong shape

def _design3_generate_y(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(X)
    sigma = 0.10 + 0.20 * X.ravel()
    eta = rng.exponential(1.0, size=len(X))
    return _mu_sin(X).ravel() + sigma * (eta - 1)


def _design3_sample_generator(x: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    x = np.atleast_1d(x)
    mean = _mu_sin(x).ravel()
    std = (0.10 + 0.20 * np.asarray(x)).ravel()
    return mean + std * rng.standard_normal(m)


DESIGN3 = DGP(
    name="Design 3: shape misspecification (skewed)",
    generate_y=_design3_generate_y,
    sample_generator=_design3_sample_generator,
)
