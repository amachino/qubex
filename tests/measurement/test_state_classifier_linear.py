"""Tests for the backend-neutral linear state classifier."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import qubex.visualization as viz
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


def test_predict_applies_scale_and_phase() -> None:
    """Linear prediction should use the same normalization as other classifiers."""
    scale = 2.0
    phase = np.pi / 2
    normalized_samples = np.array([-1.0 + 0.0j, 0.5 + 0.0j, 2.0 + 0.0j])
    input_samples = normalized_samples * np.exp(1j * phase) / scale
    classifier = StateClassifierLinear(
        lines=((1.0, 0.0, 0.0), (1.0, 0.0, -1.0)),
        state_map={(True, True): 0, (False, False): 1},
        scale=scale,
        phase=phase,
    )

    labels = classifier.predict(input_samples)

    assert labels.tolist() == [0, -1, 1]
    np.testing.assert_allclose(
        classifier.input_lines,
        ((0.0, 1.0, 0.0), (0.0, 1.0, -0.5)),
        atol=1e-15,
    )


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


def test_classify_with_plot_shows_data_and_linear_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """classify(plot=True) should show classified I/Q data and configured lines."""
    classifier = StateClassifierLinear(
        lines=((1.0, 0.0, 0.0), (1.0, 0.0, -1.0)),
        state_map={(True, True): 0, (False, False): 1},
    )
    data = np.array([-1.0 + 0.0j, 0.5 + 0.0j, 2.0 + 0.0j])
    figure = object()
    captured: dict[str, object] = {}

    def fake_make_linear_classification_figure(**kwargs: object) -> object:
        captured.update(kwargs)
        return figure

    def fake_show_figure(shown_figure: object, *, filename: str) -> None:
        captured["shown_figure"] = shown_figure
        captured["filename"] = filename

    monkeypatch.setattr(
        viz,
        "make_linear_classification_figure",
        fake_make_linear_classification_figure,
    )
    monkeypatch.setattr(viz, "show_figure", fake_show_figure)

    counts = classifier.classify("Q00", data, plot=True)

    assert counts == {0: 1, 1: 1}
    assert captured["target"] == "Q00"
    assert np.array_equal(cast(np.ndarray, captured["data"]), data)
    assert np.array_equal(
        cast(np.ndarray, captured["labels"]),
        np.array([0, -1, 1]),
    )
    assert captured["lines"] == classifier.input_lines
    assert captured["shown_figure"] is figure
    assert captured["filename"] == "state_classification_Q00"


def test_linear_classification_figure_contains_shots_and_boundaries() -> None:
    """The linear-classification figure should retain both I/Q shots and lines."""
    figure = viz.make_linear_classification_figure(
        target="Q00",
        data=np.array([-1.0 - 0.5j, 0.5 + 0.0j, 2.0 + 0.5j]),
        labels=np.array([0, -1, 1]),
        lines=((1.0, 0.0, 0.0), (0.0, 1.0, -0.25)),
    )

    traces = cast(list[dict[str, Any]], figure.to_dict()["data"])
    marker_traces = [trace for trace in traces if trace.get("mode") == "markers"]
    line_traces = [trace for trace in traces if trace.get("mode") == "lines"]

    assert sum(len(trace["x"]) for trace in marker_traces) == 3
    assert len(line_traces) == 2


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
        (
            {
                "lines": ((1.0, 0.0, 0.0),),
                "state_map": {(False,): 0},
                "scale": 0.0,
            },
            ValueError,
            "scale must be a positive finite value",
        ),
        (
            {
                "lines": ((1.0, 0.0, 0.0),),
                "state_map": {(False,): 0},
                "phase": np.inf,
            },
            ValueError,
            "phase must be finite",
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
