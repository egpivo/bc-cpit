"""Golden-value integration tests for Design 1 (seed=42).

These tests run one full simulation replicate and check that the key metrics
(CvM, QErr, CRPS) for all methods match the reference values recorded before
the src/ refactor. They serve as regression anchors: if any core logic changes
unintentionally during restructuring, at least one of these will fail.

Reference values were produced by:
    run_replicate(DESIGN1, seed=42, mode='all')
on commit 6333066 (before refactor).

Tolerance: 1e-6 relative (float64 arithmetic is deterministic for fixed seeds).
"""

import pytest

pytestmark = pytest.mark.integration  # skip with: pytest -m "not integration"


@pytest.fixture(scope="module")
def design1_result():
    from examples.simulation.dgp import DESIGN1
    from examples.simulation.runner import run_replicate

    return run_replicate(DESIGN1, seed=42, mode="all")


# â”€â”€ reference values (seed=42, recorded pre-refactor) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GOLDEN = {
    "cvm_raw": 479.18120067,
    "cvm_bc": 1.51744667,
    "cvm_bc_gam": 1.29282867,
    "cvm_cpit": 0.90163551,
    "cvm_cpit_gam": 0.79174412,
    "qerr_raw": 0.27260000,
    "qerr_bc": 0.01683158,
    "qerr_bc_gam": 0.01568421,
    "qerr_cpit": 0.01283158,
    "qerr_cpit_gam": 0.01068421,
    "crps_raw": 0.19094947,
    "crps_bc": 0.08568954,
    "crps_bc_gam": 0.08574408,
    "crps_cpit": 0.08567935,
    "crps_cpit_gam": 0.08574240,
}

_RTOL = 1e-5  # relative tolerance; small float differences across numpy versions


@pytest.mark.parametrize("key,expected", list(GOLDEN.items()))
def test_golden_metric(design1_result, key, expected):
    actual = design1_result[key]
    assert abs(actual - expected) <= _RTOL * max(abs(expected), 1e-12), (
        f"{key}: got {actual:.8f}, expected {expected:.8f} "
        f"(diff={abs(actual-expected):.2e})"
    )
