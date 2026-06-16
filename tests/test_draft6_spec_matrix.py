"""Draft(6) equation-level test matrix guard.

Task 1 intentionally adds a stable matrix of required test IDs.
Subsequent tasks implement concrete assertions for each ID.
"""

REQUIRED_TEST_IDS = [
    "T-EQ3-A",
    "T-EQ3-B",
    "T-EQ4-APPLY",
    "T-EQ8-LOC",
    "T-EQ9-10-SCALE",
    "T-ALT-CONV",
    "T-LAMBDA-TREND",
    "T-PIT-TIE",
]


def test_draft6_required_test_ids_are_unique():
    assert len(REQUIRED_TEST_IDS) == len(set(REQUIRED_TEST_IDS))


def test_draft6_required_test_ids_stable():
    assert REQUIRED_TEST_IDS == [
        "T-EQ3-A",
        "T-EQ3-B",
        "T-EQ4-APPLY",
        "T-EQ8-LOC",
        "T-EQ9-10-SCALE",
        "T-ALT-CONV",
        "T-LAMBDA-TREND",
        "T-PIT-TIE",
    ]
