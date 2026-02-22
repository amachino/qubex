"""Tests for Quel3 sequencer compiler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from qubex.backend.quel3.managers.sequencer_compiler import Quel3SequencerCompiler
from qubex.measurement.adapters import (
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3TargetTimeline,
    Quel3WaveformDefinition,
    Quel3WaveformEvent,
)


@dataclass(frozen=True)
class _RegisteredWaveform:
    sampling_period_ns: float | None
    values: np.ndarray


@dataclass(frozen=True)
class _Event:
    instrument_alias: str
    waveform_name: str
    start_offset_ns: float
    gain: float
    phase_offset_deg: float


@dataclass(frozen=True)
class _CaptureWindow:
    instrument_alias: str
    window_name: str
    start_offset_ns: float
    length_ns: float


class _RecordingSequencer:
    def __init__(self, default_sampling_period_ns: float) -> None:
        self.default_sampling_period_ns = default_sampling_period_ns
        self.registered_waveforms: dict[str, _RegisteredWaveform] = {}
        self.events: list[_Event] = []
        self.capture_windows: list[_CaptureWindow] = []

    def register_waveform(
        self,
        name: str,
        waveform: npt.ArrayLike,
        sampling_period_ns: float | None = None,
    ) -> None:
        self.registered_waveforms[name] = _RegisteredWaveform(
            sampling_period_ns=sampling_period_ns,
            values=np.asarray(waveform, dtype=np.complex128),
        )

    def add_event(
        self,
        instrument_alias: str,
        waveform_name: str,
        start_offset_ns: float,
        gain: float = 1.0,
        phase_offset_deg: float = 0.0,
    ) -> None:
        self.events.append(
            _Event(
                instrument_alias=instrument_alias,
                waveform_name=waveform_name,
                start_offset_ns=start_offset_ns,
                gain=gain,
                phase_offset_deg=phase_offset_deg,
            )
        )

    def add_capture_window(
        self,
        instrument_alias: str,
        window_name: str,
        start_offset_ns: float,
        length_ns: float,
    ) -> None:
        self.capture_windows.append(
            _CaptureWindow(
                instrument_alias=instrument_alias,
                window_name=window_name,
                start_offset_ns=start_offset_ns,
                length_ns=length_ns,
            )
        )


def _make_payload(
    *,
    waveform_library: dict[str, Quel3WaveformDefinition],
    timelines: dict[str, Quel3TargetTimeline],
) -> Quel3ExecutionPayload:
    return Quel3ExecutionPayload(
        waveform_library=waveform_library,
        timelines=timelines,
        instrument_aliases={target: f"alias-{target}" for target in timelines},
        output_target_labels={target: target for target in timelines},
        interval_ns=100.0,
        repeats=16,
        mode="avg",
        dsp_demodulation=True,
        enable_sum=False,
        enable_classification=False,
        line_param0=(1.0, 0.0, 0.0),
        line_param1=(0.0, 1.0, 0.0),
    )


def test_compiler_registers_waveforms_and_forwards_events() -> None:
    """Given payload library/events, when compiling, then waveforms and events are forwarded."""
    waveform_name = "wf_shared_0000"
    waveform_values = np.array([1.0 + 0.0j, 0.3 + 0.2j], dtype=np.complex128)
    timeline = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        events=(
            Quel3WaveformEvent(
                waveform_name=waveform_name,
                start_offset_ns=12.0,
                gain=0.5,
                phase_offset_deg=90.0,
            ),
        ),
        capture_windows=(
            Quel3CaptureWindow(name="capture_0", start_offset_ns=20.0, length_ns=8.0),
        ),
        length_ns=100.0,
    )
    payload = _make_payload(
        waveform_library={
            waveform_name: Quel3WaveformDefinition(
                waveform=waveform_values,
                sampling_period_ns=0.4,
            )
        },
        timelines={"RQ00": timeline},
    )

    compiler = Quel3SequencerCompiler()
    sequencer = compiler.compile(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
    )

    assert set(sequencer.registered_waveforms.keys()) == {waveform_name}
    registered = sequencer.registered_waveforms[waveform_name]
    assert registered.sampling_period_ns == pytest.approx(0.4)
    assert np.array_equal(registered.values, waveform_values)
    assert sequencer.events == [
        _Event(
            instrument_alias="alias-RQ00",
            waveform_name=waveform_name,
            start_offset_ns=12.0,
            gain=0.5,
            phase_offset_deg=90.0,
        )
    ]
    assert sequencer.capture_windows == [
        _CaptureWindow(
            instrument_alias="alias-RQ00",
            window_name="RQ00:capture_0",
            start_offset_ns=20.0,
            length_ns=8.0,
        )
    ]


def test_compiler_reuses_payload_waveform_across_targets() -> None:
    """Given shared waveform name in payload, when compiling, then both targets reference one registered waveform."""
    waveform_name = "wf_shared_0000"
    waveform_values = np.array([1.0 + 0.0j], dtype=np.complex128)
    timeline_a = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        events=(Quel3WaveformEvent(waveform_name=waveform_name, start_offset_ns=4.0),),
        capture_windows=(),
        length_ns=10.0,
    )
    timeline_b = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        events=(
            Quel3WaveformEvent(
                waveform_name=waveform_name,
                start_offset_ns=8.0,
                gain=0.7,
                phase_offset_deg=30.0,
            ),
        ),
        capture_windows=(),
        length_ns=10.0,
    )
    payload = _make_payload(
        waveform_library={
            waveform_name: Quel3WaveformDefinition(
                waveform=waveform_values,
                sampling_period_ns=0.4,
            )
        },
        timelines={"RQ00": timeline_a, "RQ01": timeline_b},
    )

    compiler = Quel3SequencerCompiler()
    sequencer = compiler.compile(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
    )

    assert len(sequencer.registered_waveforms) == 1
    assert [event.waveform_name for event in sequencer.events] == [
        waveform_name,
        waveform_name,
    ]


def test_compiler_rejects_event_with_unknown_waveform_name() -> None:
    """Given unknown waveform name, when compiling, then ValueError is raised."""
    payload = _make_payload(
        waveform_library={
            "wf_known": Quel3WaveformDefinition(
                waveform=np.array([1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            )
        },
        timelines={
            "RQ00": Quel3TargetTimeline(
                sampling_period_ns=0.4,
                events=(
                    Quel3WaveformEvent(
                        waveform_name="wf_unknown",
                        start_offset_ns=0.0,
                    ),
                ),
                capture_windows=(),
                length_ns=1.0,
            )
        },
    )

    compiler = Quel3SequencerCompiler()
    with pytest.raises(ValueError, match="Unknown waveform name"):
        compiler.compile(
            payload=payload,
            sequencer_factory=_RecordingSequencer,
            default_sampling_period_ns=0.4,
        )
