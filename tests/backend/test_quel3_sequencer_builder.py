"""Tests for QuEL-3 sequencer builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from qubex.backend.quel3 import Quel3CaptureMode
from qubex.backend.quel3.builders.sequencer_builder import Quel3SequencerBuilder
from qubex.measurement.adapters import (
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
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


@dataclass(frozen=True)
class _Binding:
    alias: str
    sampling_period_fs: int
    step_samples: int


class _RecordingSequencer:
    def __init__(
        self,
        default_sampling_period_ns: float,
        enforce_sample_grid: bool = True,
        iter_blank_ns: float = 2_000,
    ) -> None:
        self.default_sampling_period_ns = default_sampling_period_ns
        self.enforce_sample_grid = enforce_sample_grid
        self.iter_blank_ns = iter_blank_ns
        self.registered_waveforms: dict[str, _RegisteredWaveform] = {}
        self.events: list[_Event] = []
        self.capture_windows: list[_CaptureWindow] = []
        self.bindings: list[_Binding] = []
        self.iterations: int = 1
        self.extended_by_ns: float = 0.0

    def bind(
        self,
        alias: str,
        sampling_period_fs: int,
        step_samples: int,
    ) -> None:
        self.bindings.append(
            _Binding(
                alias=alias,
                sampling_period_fs=sampling_period_fs,
                step_samples=step_samples,
            )
        )

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

    def set_iterations(self, iterations: int) -> None:
        self.iterations = iterations

    def extend_length_ns(self, additional_ns: float) -> None:
        self.extended_by_ns += additional_ns

    def get_aligned_length_fs(self, post_blank_fs: int = 0) -> int:
        return post_blank_fs

    def export_set_fixed_timeline_directive(
        self,
        instrument_alias: str,
    ) -> object:
        del instrument_alias
        return object()


def _make_payload(
    *,
    waveform_library: dict[str, Quel3Waveform],
    fixed_timelines: dict[str, Quel3FixedTimeline],
    n_iterations: int = 16,
    shot_interval_ns: float = 0.0,
) -> Quel3ExecutionPayload:
    return Quel3ExecutionPayload(
        waveform_library=waveform_library,
        fixed_timelines=fixed_timelines,
        n_iterations=n_iterations,
        shot_interval_ns=shot_interval_ns,
        capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
    )


def test_builder_registers_waveforms_and_forwards_events() -> None:
    """Given payload library/events, when building, waveforms and events are forwarded."""
    waveform_name = "wf_shared_0000"
    waveform_values = np.array([1.0 + 0.0j, 0.3 + 0.2j], dtype=np.complex128)
    timeline = Quel3FixedTimeline(
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
            waveform_name: Quel3Waveform(
                iq_array=waveform_values,
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={"alias-RQ00": timeline},
    )

    builder = Quel3SequencerBuilder()
    sequencer = builder.build(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
        alias_bindings={"alias-RQ00": (400_000, 64)},
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
            phase_offset_deg=-90.0,
        )
    ]
    assert sequencer.capture_windows == [
        _CaptureWindow(
            instrument_alias="alias-RQ00",
            window_name="capture_0",
            start_offset_ns=20.0,
            length_ns=8.0,
        )
    ]
    assert sequencer.bindings == [
        _Binding(alias="alias-RQ00", sampling_period_fs=400_000, step_samples=64)
    ]
    assert sequencer.extended_by_ns == pytest.approx(0.0)
    assert sequencer.iter_blank_ns == pytest.approx(0.0)
    assert sequencer.iterations == 16


def test_builder_aligns_timeline_items_to_alias_sampling_grid() -> None:
    """Given off-grid resolved timeline items, builder should align them to the alias grid."""
    waveform_name = "wf_shared_0000"
    timeline = Quel3FixedTimeline(
        events=(
            Quel3WaveformEvent(
                waveform_name=waveform_name,
                start_offset_ns=2484.4,
            ),
        ),
        capture_windows=(
            Quel3CaptureWindow(
                name="capture_0",
                start_offset_ns=2484.4,
                length_ns=0.4,
            ),
        ),
        length_ns=2484.8,
    )
    payload = _make_payload(
        waveform_library={
            waveform_name: Quel3Waveform(
                iq_array=np.array([1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.8,
            )
        },
        fixed_timelines={"alias-RQ00": timeline},
    )

    builder = Quel3SequencerBuilder()
    sequencer = builder.build(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
        alias_bindings={"alias-RQ00": (800_000, 64)},
    )

    assert sequencer.events[0].start_offset_ns == pytest.approx(2484.8)
    assert sequencer.capture_windows[0].start_offset_ns == pytest.approx(2484.8)
    assert sequencer.capture_windows[0].length_ns == pytest.approx(0.8)


def test_builder_accepts_near_grid_timing_roundoff() -> None:
    """Given sub-millisample timing roundoff, builder should keep the nearest grid time."""
    waveform_name = "wf_shared_0000"
    sample_roundoff_ns = 0.8 * 5e-4
    timeline = Quel3FixedTimeline(
        events=(
            Quel3WaveformEvent(
                waveform_name=waveform_name,
                start_offset_ns=2484.8 + sample_roundoff_ns,
            ),
        ),
        capture_windows=(
            Quel3CaptureWindow(
                name="capture_0",
                start_offset_ns=2484.8 + sample_roundoff_ns,
                length_ns=0.8 + sample_roundoff_ns,
            ),
        ),
        length_ns=2485.6,
    )
    payload = _make_payload(
        waveform_library={
            waveform_name: Quel3Waveform(
                iq_array=np.array([1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.8,
            )
        },
        fixed_timelines={"alias-RQ00": timeline},
    )

    builder = Quel3SequencerBuilder()
    sequencer = builder.build(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
        alias_bindings={"alias-RQ00": (800_000, 64)},
    )

    assert sequencer.events[0].start_offset_ns == pytest.approx(2484.8)
    assert sequencer.capture_windows[0].start_offset_ns == pytest.approx(2484.8)
    assert sequencer.capture_windows[0].length_ns == pytest.approx(0.8)


def test_builder_reuses_payload_waveform_across_targets() -> None:
    """Given shared waveform name in payload, when building, both targets reuse one registered waveform."""
    waveform_name = "wf_shared_0000"
    waveform_values = np.array([1.0 + 0.0j], dtype=np.complex128)
    timeline_a = Quel3FixedTimeline(
        events=(Quel3WaveformEvent(waveform_name=waveform_name, start_offset_ns=4.0),),
        capture_windows=(),
        length_ns=10.0,
    )
    timeline_b = Quel3FixedTimeline(
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
            waveform_name: Quel3Waveform(
                iq_array=waveform_values,
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={"alias-RQ00": timeline_a, "alias-RQ01": timeline_b},
    )

    builder = Quel3SequencerBuilder()
    sequencer = builder.build(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
        alias_bindings={
            "alias-RQ00": (400_000, 64),
            "alias-RQ01": (400_000, 64),
        },
    )

    assert len(sequencer.registered_waveforms) == 1
    assert [event.waveform_name for event in sequencer.events] == [
        waveform_name,
        waveform_name,
    ]


def test_builder_rejects_event_with_unknown_waveform_name() -> None:
    """Given unknown waveform name, when building, ValueError is raised."""
    payload = _make_payload(
        waveform_library={
            "wf_known": Quel3Waveform(
                iq_array=np.array([1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={
            "alias-RQ00": Quel3FixedTimeline(
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

    builder = Quel3SequencerBuilder()
    with pytest.raises(ValueError, match="Unknown waveform name"):
        builder.build(
            payload=payload,
            sequencer_factory=_RecordingSequencer,
            default_sampling_period_ns=0.4,
            alias_bindings={"alias-RQ00": (400_000, 64)},
        )


def test_builder_rejects_missing_alias_binding() -> None:
    """Given missing alias binding, when building, ValueError is raised."""
    payload = _make_payload(
        waveform_library={
            "wf_known": Quel3Waveform(
                iq_array=np.array([1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={
            "alias-RQ00": Quel3FixedTimeline(
                events=(
                    Quel3WaveformEvent(
                        waveform_name="wf_known",
                        start_offset_ns=0.0,
                    ),
                ),
                capture_windows=(),
                length_ns=1.0,
            )
        },
    )

    builder = Quel3SequencerBuilder()
    with pytest.raises(ValueError, match="Missing sequencer binding"):
        builder.build(
            payload=payload,
            sequencer_factory=_RecordingSequencer,
            default_sampling_period_ns=0.4,
            alias_bindings={},
        )


def test_builder_passes_shot_interval_as_iteration_blank() -> None:
    """Given a shot interval, builder passes its aligned value as the iteration blank."""
    payload = _make_payload(
        waveform_library={
            "wf_known": Quel3Waveform(
                iq_array=np.array([1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={
            "alias-RQ00": Quel3FixedTimeline(
                events=(
                    Quel3WaveformEvent(
                        waveform_name="wf_known",
                        start_offset_ns=0.0,
                    ),
                ),
                capture_windows=(),
                length_ns=10.0,
            )
        },
        shot_interval_ns=2048.0,
    )

    builder = Quel3SequencerBuilder()
    sequencer = builder.build(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
        alias_bindings={"alias-RQ00": (400_000, 64)},
    )

    assert sequencer.iter_blank_ns == pytest.approx(2048.0)
    assert sequencer.extended_by_ns == pytest.approx(0.0)


def test_builder_applies_minimum_iteration_blank_floor() -> None:
    """Given a tiny shot interval, builder passes the aligned minimum iteration blank."""
    payload = _make_payload(
        waveform_library={
            "wf_known": Quel3Waveform(
                iq_array=np.array([1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={
            "alias-RQ00": Quel3FixedTimeline(
                events=(
                    Quel3WaveformEvent(
                        waveform_name="wf_known",
                        start_offset_ns=0.0,
                    ),
                ),
                capture_windows=(),
                length_ns=10.0,
            )
        },
        shot_interval_ns=1.0,
    )

    builder = Quel3SequencerBuilder()
    sequencer = builder.build(
        payload=payload,
        sequencer_factory=_RecordingSequencer,
        default_sampling_period_ns=0.4,
        alias_bindings={"alias-RQ00": (400_000, 64)},
    )

    assert sequencer.iter_blank_ns == pytest.approx(1024.0)
    assert sequencer.extended_by_ns == pytest.approx(0.0)
