"""Parameter containers for bias correction (draft §2.1)."""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pygam import GammaGAM, LinearGAM


class AffineParams(NamedTuple):
    """Global affine §2.1.1 Eq (2)–(4): y_adj = a + b * y_raw, b > 0."""

    a: float
    b: float


class XAffineParamsGAM(NamedTuple):
    """x-dependent affine §2.1.2 Eq (5)–(10).

    m(x) = a0 + b0 μ̂(x) + gam_a.predict(x)   (Eq. 6/8)
    b(x) = sqrt(gam_q.predict(x))               (Eq. 9–10)
    """

    gam_a: "LinearGAM"
    gam_q: "GammaGAM"
    a0: float = 0.0
    b0: float = 1.0
    b_min: float | None = None
    b_max: float | None = None
