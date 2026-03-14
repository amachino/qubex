"""Tests for measurement visualization helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest
from qxpulse import Blank, Gaussian, PulseSchedule

import qubex.visualization as viz
from qubex.measurement.classifiers import StateClassifierGMM, StateClassifierKMeans
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.visualization.schedule_visualizer import (
    make_measurement_schedule_figure,
    make_sequencer_timeline_figure,
    plot_measurement_schedule,
    plot_sequencer_timeline,
)


def _make_kmeans_classifier() -> StateClassifierKMeans:
    """Return a minimal k-means classifier for plotting tests."""
    return StateClassifierKMeans(
        dataset={
            0: np.array([0.0 + 0.0j]),
            1: np.array([1.0 + 0.0j]),
        },
        model=cast(
            Any,
            SimpleNamespace(
                cluster_centers_=np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                    ]
                )
            ),
        ),
        label_map={0: 0, 1: 1},
        confusion_matrix=np.eye(2),
        scale=1.0,
        phase=0.0,
        created_at="",
    )


def _make_gmm_classifier() -> StateClassifierGMM:
    """Return a minimal GMM classifier for plotting tests."""
    return StateClassifierGMM(
        dataset={
            0: np.array([0.0 + 0.0j]),
            1: np.array([1.0 + 0.0j]),
        },
        model=cast(
            Any,
            SimpleNamespace(
                means_=np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                    ]
                ),
                covariances_=np.array([0.25, 0.5]),
                weights_=np.array([0.5, 0.5]),
            ),
        ),
        label_map={0: 0, 1: 1},
        confusion_matrix=np.eye(2),
        scale=1.0,
        phase=0.0,
        created_at="",
    )


def test_make_classification_figure_adds_points_centers_and_sigma_bands() -> None:
    """Given classified IQ data, when making a classification figure, then the expected traces are present."""
    data = np.array(
        [
            0.0 + 0.0j,
            0.1 + 0.1j,
            1.0 + 0.0j,
            1.1 + 0.1j,
        ]
    )
    labels = np.array([0, 0, 1, 1])

    figure = viz.make_classification_figure(
        target="Q00",
        data=data,
        labels=labels,
        centers={0: 0.0 + 0.0j, 1: 1.0 + 0.0j},
        stddevs={0: 0.2, 1: 0.3},
    )

    assert isinstance(figure, go.Figure)
    traces = tuple(cast(Any, figure.data))
    assert len(traces) == 6
    assert figure.layout.title.text == "State classification : Q00"
    trace_names = {str(trace.name) for trace in traces}
    assert {"|0⟩", "|1⟩", "|0⟩ ± 2σ", "|1⟩ ± 2σ"} <= trace_names


@pytest.mark.parametrize(
    ("classifier_factory", "expects_stddevs"),
    [
        (_make_kmeans_classifier, False),
        (_make_gmm_classifier, True),
    ],
    ids=["kmeans", "gmm"],
)
def test_classifier_plot_uses_classification_figure_helper(
    monkeypatch: pytest.MonkeyPatch,
    classifier_factory: Callable[[], StateClassifierGMM | StateClassifierKMeans],
    expects_stddevs: bool,
) -> None:
    """Given a state classifier, when plotting, then it delegates figure construction to the visualization helper."""
    classifier = classifier_factory()
    data = np.array([0.0 + 0.0j, 1.0 + 0.0j])
    labels = np.array([0, 1])
    figure = go.Figure()
    captured: dict[str, object] = {}

    def fake_make_classification_figure(
        *,
        target: str,
        data: np.ndarray,
        labels: np.ndarray,
        centers: dict[int, complex],
        stddevs: dict[int, float] | None = None,
        n_samples: int = 1000,
    ) -> go.Figure:
        captured["target"] = target
        captured["data"] = data
        captured["labels"] = labels
        captured["centers"] = centers
        captured["stddevs"] = stddevs
        captured["n_samples"] = n_samples
        return figure

    def fake_show_figure(
        shown_figure: go.Figure,
        *,
        filename: str,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        captured["shown_figure"] = shown_figure
        captured["filename"] = filename
        captured["width"] = width
        captured["height"] = height

    monkeypatch.setattr(
        viz, "make_classification_figure", fake_make_classification_figure
    )
    monkeypatch.setattr(viz, "show_figure", fake_show_figure)

    classifier.plot("Q00", data, labels)

    assert captured["target"] == "Q00"
    assert np.array_equal(cast(np.ndarray, captured["data"]), data)
    assert np.array_equal(cast(np.ndarray, captured["labels"]), labels)
    assert captured["centers"] == classifier.centers
    assert captured["shown_figure"] is figure
    assert captured["filename"] == "state_classification_Q00"
    if expects_stddevs:
        assert captured["stddevs"] == classifier.stddevs
    else:
        assert captured["stddevs"] is None


def test_make_measurement_schedule_figure_adds_capture_overlay() -> None:
    """Given capture windows, when making schedule figure, then capture overlays are present."""
    with PulseSchedule(["Q00", "Q01", "RQ00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(duration=4.0, sampling_period=0.4))
        pulse_schedule.add(
            "Q00",
            Gaussian(
                duration=8.0,
                amplitude=0.5,
                sigma=1.6,
                zero_bounds=False,
                sampling_period=0.4,
            ),
        )
        pulse_schedule.add("Q01", Blank(duration=4.0, sampling_period=0.4))
        pulse_schedule.add(
            "Q01",
            Gaussian(
                duration=8.0,
                amplitude=0.25,
                sigma=1.6,
                zero_bounds=False,
                sampling_period=0.4,
            ),
        )
        pulse_schedule.add("RQ00", Blank(duration=4.0, sampling_period=0.4))
        pulse_schedule.add(
            "RQ00",
            Gaussian(
                duration=8.0,
                amplitude=0.3,
                sigma=1.6,
                zero_bounds=False,
                sampling_period=0.4,
            ),
        )

    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=["RQ00"],
                    start_time=4.0,
                    duration=8.0,
                )
            ]
        ),
    )

    figure = make_measurement_schedule_figure(schedule)

    assert isinstance(figure, go.Figure)
    figure_dict = figure.to_dict()
    shapes = [
        shape
        for shape in figure_dict.get("layout", {}).get("shapes", [])
        if isinstance(shape, dict)
    ]
    capture_shapes = [
        shape
        for shape in shapes
        if shape.get("yref") == "y5 domain"
        and shape.get("x0") == 4.0
        and shape.get("x1") == 12.0
    ]
    assert len(capture_shapes) == 1
    trace_names = {
        str(trace.get("name"))
        for trace in figure_dict.get("data", [])
        if isinstance(trace, dict)
    }
    assert "Blank" in trace_names
    assert "Capture" in trace_names


def test_make_measurement_schedule_figure_hides_workaround_capture_by_default() -> None:
    """Given workaround capture, when making schedule figure, then workaround band is excluded by default."""
    with PulseSchedule(["RQ00"]) as pulse_schedule:
        pulse_schedule.add("RQ00", Blank(duration=4.0, sampling_period=0.4))
        pulse_schedule.add(
            "RQ00",
            Gaussian(
                duration=8.0,
                amplitude=0.3,
                sigma=1.6,
                zero_bounds=False,
                sampling_period=0.4,
            ),
        )
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=["RQ00"],
                    start_time=0.0,
                    duration=4.0,
                    is_workaround=True,
                ),
                Capture(
                    channels=["RQ00"],
                    start_time=4.0,
                    duration=8.0,
                ),
            ]
        ),
    )

    figure = make_measurement_schedule_figure(schedule)
    figure_dict = figure.to_dict()
    shapes = [
        shape
        for shape in figure_dict.get("layout", {}).get("shapes", [])
        if isinstance(shape, dict)
    ]
    capture_shapes = [
        shape
        for shape in shapes
        if shape.get("yref") == "y domain"
        and shape.get("y0") == 0.0
        and shape.get("y1") == 0.2
    ]

    assert len(capture_shapes) == 1
    assert capture_shapes[0]["x0"] == 0.0
    assert capture_shapes[0]["x1"] == 8.0


def test_make_measurement_schedule_figure_can_show_workaround_capture() -> None:
    """Given workaround capture, when hide flag is false, then workaround band is rendered."""
    with PulseSchedule(["RQ00"]) as pulse_schedule:
        pulse_schedule.add("RQ00", Blank(duration=4.0, sampling_period=0.4))
        pulse_schedule.add(
            "RQ00",
            Gaussian(
                duration=8.0,
                amplitude=0.3,
                sigma=1.6,
                zero_bounds=False,
                sampling_period=0.4,
            ),
        )
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=["RQ00"],
                    start_time=0.0,
                    duration=4.0,
                    is_workaround=True,
                ),
                Capture(
                    channels=["RQ00"],
                    start_time=4.0,
                    duration=8.0,
                ),
            ]
        ),
    )

    figure = make_measurement_schedule_figure(
        schedule,
        hide_workaround_capture=False,
    )
    figure_dict = figure.to_dict()
    shapes = [
        shape
        for shape in figure_dict.get("layout", {}).get("shapes", [])
        if isinstance(shape, dict)
    ]
    capture_shapes = [
        shape
        for shape in shapes
        if shape.get("yref") == "y domain"
        and shape.get("y0") == 0.0
        and shape.get("y1") == 0.2
    ]
    capture_ranges = {(shape["x0"], shape["x1"]) for shape in capture_shapes}

    assert capture_ranges == {(0.0, 4.0), (4.0, 12.0)}


def test_make_measurement_schedule_figure_hides_workaround_blank_by_default() -> None:
    """Given workaround capture and leading blank, when making schedule figure, then workaround blank is excluded by default."""
    with PulseSchedule(["RQ00"]) as pulse_schedule:
        pulse_schedule.add("RQ00", Blank(duration=4.0, sampling_period=0.4))
        pulse_schedule.add(
            "RQ00",
            Gaussian(
                duration=8.0,
                amplitude=0.3,
                sigma=1.6,
                zero_bounds=False,
                sampling_period=0.4,
            ),
        )
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=["RQ00"],
                    start_time=0.0,
                    duration=4.0,
                    is_workaround=True,
                ),
                Capture(
                    channels=["RQ00"],
                    start_time=4.0,
                    duration=8.0,
                ),
            ]
        ),
    )

    figure = make_measurement_schedule_figure(schedule)
    figure_dict = figure.to_dict()
    shapes = [
        shape
        for shape in figure_dict.get("layout", {}).get("shapes", [])
        if isinstance(shape, dict)
    ]
    blank_shapes = [
        shape
        for shape in shapes
        if shape.get("yref") == "y domain"
        and shape.get("y0") == 0.0
        and shape.get("y1") == 1.0
    ]

    assert blank_shapes == []
    trace_names = {
        str(trace.get("name"))
        for trace in figure_dict.get("data", [])
        if isinstance(trace, dict)
    }
    assert "Blank" not in trace_names


@dataclass(frozen=True)
class _Waveform:
    sampling_period_ns: float
    iq_array: np.ndarray


@dataclass(frozen=True)
class _Event:
    waveform_name: str
    start_offset_ns: float
    gain: float
    phase_offset_deg: float


@dataclass(frozen=True)
class _CaptureWindow:
    name: str
    start_offset_ns: float
    length_ns: float


def test_make_sequencer_timeline_figure_renders_event_and_capture_lanes() -> None:
    """Given sequencer data, when making timeline figure, then both event and capture traces are rendered."""
    sequencer = SimpleNamespace(
        _waveform_library={
            "wf0": _Waveform(
                sampling_period_ns=0.4,
                iq_array=np.array([0.5 + 0.0j, 0.2 + 0.1j], dtype=np.complex128),
            )
        },
        _alias_to_events={
            "ctrl-00": [
                _Event(
                    waveform_name="wf0",
                    start_offset_ns=4.0,
                    gain=0.8,
                    phase_offset_deg=30.0,
                )
            ]
        },
        _alias_to_capwin={
            "readout-00": [
                _CaptureWindow(
                    name="cap0",
                    start_offset_ns=4.0,
                    length_ns=8.0,
                )
            ]
        },
    )

    figure = make_sequencer_timeline_figure(sequencer)

    assert isinstance(figure, go.Figure)
    figure_dict = figure.to_dict()
    assert len(figure_dict.get("data", [])) == 2
    layout = figure_dict.get("layout", {})
    margin = layout.get("margin", {})
    yaxis = layout.get("yaxis", {})
    assert margin.get("l", 0) >= 120
    assert yaxis.get("automargin") is True
    trace_names = {
        str(trace.get("name"))
        for trace in figure_dict.get("data", [])
        if isinstance(trace, dict)
    }
    assert trace_names == {
        "Waveform Event",
        "Capture Window",
    }


def test_make_sequencer_timeline_figure_raises_on_empty_timeline() -> None:
    """Given empty sequencer data, when making timeline figure, then ValueError is raised."""
    sequencer = SimpleNamespace(
        _waveform_library={},
        _alias_to_events={},
        _alias_to_capwin={},
    )

    with pytest.raises(ValueError, match="timeline is empty"):
        make_sequencer_timeline_figure(sequencer)


def test_plot_measurement_schedule_returns_none(monkeypatch) -> None:
    """Given schedule plot API, when plotting, then None is returned."""
    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(duration=4.0, sampling_period=0.4))
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )

    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)
    result = plot_measurement_schedule(schedule)
    assert result is None


def test_measurement_schedule_plot_delegates_to_schedule_visualizer(
    monkeypatch,
) -> None:
    """Given MeasurementSchedule.plot, when called, then schedule visualizer receives forwarded options."""
    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(duration=4.0, sampling_period=0.4))
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )
    called: dict[str, object] = {}

    def _plot_measurement_schedule(
        plot_schedule: MeasurementSchedule,
        *,
        show_physical_pulse: bool,
        hide_workaround_capture: bool,
        title: str,
        width: int,
        n_samples: int | None,
        divide_by_two_pi: bool,
        line_shape: str,
        template: str,
    ) -> None:
        called["schedule"] = plot_schedule
        called["show_physical_pulse"] = show_physical_pulse
        called["hide_workaround_capture"] = hide_workaround_capture
        called["title"] = title
        called["width"] = width
        called["n_samples"] = n_samples
        called["divide_by_two_pi"] = divide_by_two_pi
        called["line_shape"] = line_shape
        called["template"] = template

    monkeypatch.setattr(
        "qubex.visualization.schedule_visualizer.plot_measurement_schedule",
        _plot_measurement_schedule,
    )

    result = schedule.plot(
        show_physical_pulse=True,
        hide_workaround_capture=False,
        title="My Schedule",
        width=1234,
        n_samples=16,
        divide_by_two_pi=True,
        line_shape="linear",
        template="plotly_white",
    )

    assert result is None
    assert called["schedule"] is schedule
    assert called["show_physical_pulse"] is True
    assert called["hide_workaround_capture"] is False
    assert called["title"] == "My Schedule"
    assert called["width"] == 1234
    assert called["n_samples"] == 16
    assert called["divide_by_two_pi"] is True
    assert called["line_shape"] == "linear"
    assert called["template"] == "plotly_white"


def test_plot_sequencer_timeline_returns_none(monkeypatch) -> None:
    """Given timeline plot API, when plotting, then None is returned."""
    sequencer = SimpleNamespace(
        _waveform_library={
            "wf0": _Waveform(
                sampling_period_ns=0.4,
                iq_array=np.array([0.5 + 0.0j, 0.2 + 0.1j], dtype=np.complex128),
            )
        },
        _alias_to_events={
            "ctrl-00": [
                _Event(
                    waveform_name="wf0",
                    start_offset_ns=4.0,
                    gain=0.8,
                    phase_offset_deg=30.0,
                )
            ]
        },
        _alias_to_capwin={"readout-00": []},
    )

    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)
    result = plot_sequencer_timeline(sequencer)
    assert result is None
