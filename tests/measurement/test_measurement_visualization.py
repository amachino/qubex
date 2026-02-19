"""Tests for measurement visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest
from qxpulse import Blank, Gaussian, PulseSchedule

from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.visualization.schedule_visualizer import (
    make_measurement_schedule_figure,
    make_sequencer_timeline_figure,
    plot_measurement_schedule,
    plot_sequencer_timeline,
)


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
    shapes = figure_dict.get("layout", {}).get("shapes", [])
    assert len(shapes) >= 1
    assert {shape.get("yref") for shape in shapes if isinstance(shape, dict)} == {
        "y5 domain"
    }
    trace_names = {
        str(trace.get("name"))
        for trace in figure_dict.get("data", [])
        if isinstance(trace, dict)
    }
    assert "Capture" in trace_names


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
