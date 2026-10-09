"""Tests for the backend-neutral linear state classifier."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.measurement.classifiers import StateClassifierLinear


def test_predict_maps_line_decisions_and_rejects_unmapped_patterns() -> None:
    """Linear decisions should use the configured state and rejection mapping."""
    classifier = StateClassifierLinear(
        lines=((1.0, 0.0, 0.0), (1.0, 0.0, -1.0)),
        state_map={(True, True): 0, (False, False): 1},
    )

    labels = classifier.predict(np.array([-1.0 + 0.0j, 0.5 + 0.0j, 2.0 + 0.0j]))

    assert labels.tolist() == [0, -1, 1]
    assert classifier.n_states == 2


def test_classify_decisions_preserves_leading_shape() -> None:
    """Decision arrays should retain all axes except the line-decision axis."""
    classifier = StateClassifierLinear(
        lines=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        state_map={(False, False): 0, (True, True): 1},
        rejection_state=-2,
    )
    decisions = np.array(
        [
            [[False, False], [True, False]],
            [[True, True], [False, True]],
        ]
    )

    labels = classifier.classify_decisions(decisions)

    assert labels.tolist() == [[0, -2], [1, -2]]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"lines": (), "state_map": {(False,): 0}}, ValueError, "At least one"),
        (
            {"lines": ((0.0, 0.0, 1.0),), "state_map": {(False,): 0}},
            ValueError,
            "must not both be zero",
        ),
        (
            {"lines": ((1.0, 0.0, 0.0),), "state_map": {(False, True): 0}},
            ValueError,
            "one decision per line",
        ),
        (
            {"lines": ((1.0, 0.0, 0.0),), "state_map": {(False,): 1}},
            ValueError,
            "contiguous integers",
        ),
        (
            {
                "lines": ((1.0, 0.0, 0.0),),
                "state_map": {(False,): 0},
                "rejection_state": 1,
            },
            ValueError,
            "must be negative",
        ),
    ],
)
def test_rejects_invalid_linear_classifier_definitions(
    kwargs: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    """Invalid lines and state correspondence maps should fail at construction."""
    with pytest.raises(error_type, match=match):
        StateClassifierLinear(**kwargs)  # type: ignore[arg-type]
