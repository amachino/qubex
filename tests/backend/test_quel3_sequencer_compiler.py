"""Tests for Quel3 sequencer compiler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qubex.backend.quel3.quel3_sequencer_compiler import Quel3SequencerCompiler
from qubex.measurement.adapters import (
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3TargetTimeline,
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


def _make_payload(timelines: dict[str, Quel3TargetTimeline]) -> Quel3ExecutionPayload:
    return Quel3ExecutionPayload(
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


def test_compiler_reuses_waveform_for_scale_phase_variants() -> None:
    """Given shape-equivalent waveforms, when compiling, then registration is shared."""
    base = np.array([0.0 + 0.0j, 0.2 + 0.4j, 0.5 + 1.0j, 0.0 + 0.0j])
    scalar = 0.6 * np.exp(1j * np.deg2rad(30.0))
    timeline_a = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        events=(
            Quel3WaveformEvent(
                start_offset_ns=0.0,
                waveform=base.astype(np.complex128),
                sampling_period_ns=0.4,
            ),
        ),
        capture_windows=(),
        length_ns=1.6,
    )
    timeline_b = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        events=(
            Quel3WaveformEvent(
                start_offset_ns=0.0,
                waveform=(base * scalar).astype(np.complex128),
                sampling_period_ns=0.4,
            ),
        ),
        capture_windows=(),
        length_ns=1.6,
    )
    payload = _make_payload({"RQ00": timeline_a, "RQ01": timeline_b})

    compiler = Quel3SequencerCompiler()
    sequencer = compiler.compile(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
    )

    assert len(sequencer.registered_waveforms) == 1
    assert len(sequencer.events) == 2
    first_event = sequencer.events[0]
    second_event = sequencer.events[1]
    assert first_event.waveform_name == second_event.waveform_name
    assert first_event.start_offset_ns == 0.4
    assert second_event.start_offset_ns == 0.4

    registered = sequencer.registered_waveforms[first_event.waveform_name].values
    first_scale = first_event.gain * np.exp(
        1j * np.deg2rad(first_event.phase_offset_deg)
    )
    second_scale = second_event.gain * np.exp(
        1j * np.deg2rad(second_event.phase_offset_deg)
    )
    assert np.allclose(registered * first_scale, base[1:3])
    assert np.allclose(registered * second_scale, (base * scalar)[1:3])


def test_compiler_skips_blank_and_splits_non_blank_segments() -> None:
    """Given sparse waveform, when compiling, then only non-blank segments become events."""
    timeline = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        events=(
            Quel3WaveformEvent(
                start_offset_ns=0.0,
                waveform=np.array(
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.5j],
                    dtype=np.complex128,
                ),
                sampling_period_ns=0.4,
            ),
        ),
        capture_windows=(
            Quel3CaptureWindow(name="capture_0", start_offset_ns=0.8, length_ns=0.4),
        ),
        length_ns=2.0,
    )
    payload = _make_payload({"RQ00": timeline})

    compiler = Quel3SequencerCompiler()
    sequencer = compiler.compile(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
    )

    assert len(sequencer.registered_waveforms) == 1
    assert [event.start_offset_ns for event in sequencer.events] == [0.4, 1.6]
    assert sequencer.capture_windows == [
        _CaptureWindow(
            instrument_alias="alias-RQ00",
            window_name="RQ00:capture_0",
            start_offset_ns=0.8,
            length_ns=0.4,
        )
    ]


def test_compiler_handles_all_blank_waveform() -> None:
    """Given blank waveform, when compiling, then no waveform or event is registered."""
    timeline = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        events=(
            Quel3WaveformEvent(
                start_offset_ns=0.0,
                waveform=np.zeros(8, dtype=np.complex128),
                sampling_period_ns=0.4,
            ),
        ),
        capture_windows=(
            Quel3CaptureWindow(name="capture_0", start_offset_ns=0.0, length_ns=0.8),
        ),
        length_ns=3.2,
    )
    payload = _make_payload({"RQ00": timeline})

    compiler = Quel3SequencerCompiler()
    sequencer = compiler.compile(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
    )

    assert sequencer.registered_waveforms == {}
    assert sequencer.events == []
    assert len(sequencer.capture_windows) == 1
