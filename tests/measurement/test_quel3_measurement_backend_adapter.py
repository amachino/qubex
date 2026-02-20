"""Tests for Quel3 measurement backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from qubex.measurement.adapters import (
    Quel3ExecutionPayload,
    Quel3MeasurementBackendAdapter,
)
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.models import (
    DspConfig,
    FrequencyConfig,
    MeasurementConfig,
    MeasurementSchedule,
)
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule


@dataclass
class _FakeWaveform:
    values: np.ndarray
    sampling_period: float = 0.4

    @property
    def duration(self) -> float:
        return len(self.values) * self.sampling_period


@dataclass
class _FakeSequence:
    waveforms: list[_FakeWaveform]

    def get_flattened_waveforms(
        self, apply_frame_shifts: bool = True
    ) -> list[_FakeWaveform]:
        del apply_frame_shifts
        return self.waveforms


@dataclass
class _FakePulseSchedule:
    duration: float
    sequences: dict[str, _FakeSequence]
    valid: bool = True

    @property
    def labels(self) -> list[str]:
        return list(self.sequences.keys())

    def is_valid(self) -> bool:
        return self.valid

    def get_sequence(self, label: str, *, copy: bool = False) -> _FakeSequence:
        del copy
        return self.sequences[label]

    def get_sampled_sequences(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        del copy
        raise AssertionError("Quel3 adapter must not call get_sampled_sequences().")


def _make_config() -> MeasurementConfig:
    return MeasurementConfig(
        mode="avg",
        shots=16,
        interval=100.0,
        frequency=FrequencyConfig(frequencies={}),
        dsp=DspConfig(
            enable_dsp_demodulation=True,
            enable_dsp_sum=False,
            enable_dsp_classification=False,
            line_param0=(1.0, 0.0, 0.0),
            line_param1=(0.0, 1.0, 0.0),
        ),
    )


def test_quel3_adapter_accepts_relaxed_schedule() -> None:
    """Given relaxed schedule, when validating, then no error is raised."""
    target = "RQ00"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=12.4,
            sequences={
                target: _FakeSequence(
                    waveforms=[_FakeWaveform(np.array([0.1 + 0.0j] * 31))]
                )
            },
        ),
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=[target],
                    start_time=2.3,
                    duration=3.2,
                ),
            ]
        ),
    )

    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=type("_BC", (), {"DEFAULT_SAMPLING_PERIOD": 0.4})(),
        experiment_system=object(),  # type: ignore[arg-type]
    )

    adapter.validate_schedule(schedule)


def test_quel3_adapter_rejects_capture_outside_pulse_duration() -> None:
    """Given out-of-range capture, when validating, then ValueError is raised."""
    target = "RQ00"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=10.0,
            sequences={
                target: _FakeSequence(
                    waveforms=[_FakeWaveform(np.array([0.1 + 0.0j] * 25))]
                )
            },
        ),
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=[target],
                    start_time=9.0,
                    duration=2.0,
                ),
            ]
        ),
    )
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=type("_BC", (), {"DEFAULT_SAMPLING_PERIOD": 0.4})(),
        experiment_system=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="exceeds pulse schedule duration"):
        adapter.validate_schedule(schedule)


def test_quel3_adapter_builds_fixed_timeline_payload() -> None:
    """Given schedule and config, when building request, then payload contains per-target timeline and captures."""
    target = "RQ00"
    waveform = np.array([0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                target: _FakeSequence(
                    waveforms=[_FakeWaveform(waveform, sampling_period=0.4)]
                )
            },
        ),
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=[target],
                    start_time=0.4,
                    duration=0.4,
                ),
            ]
        ),
    )
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=type("_BC", (), {"DEFAULT_SAMPLING_PERIOD": 0.4})(),
        experiment_system=cast(
            Any,
            type(
                "_ES",
                (),
                {"get_awg_frequency": staticmethod(lambda _: 100_000_000.0)},
            )(),
        ),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
    )

    request = adapter.build_execution_request(schedule=schedule, config=_make_config())

    payload = request.payload
    assert isinstance(payload, Quel3ExecutionPayload)
    assert payload.interval_ns == 102
    assert payload.repeats == 16
    assert payload.mode == "avg"
    assert payload.instrument_aliases == {target: target}
    assert payload.output_target_labels == {target: "Q00"}
    assert target in payload.timelines
    timeline = payload.timelines[target]
    assert timeline.sampling_period_ns == 0.4
    assert len(timeline.events) == 1
    assert timeline.events[0].start_offset_ns == pytest.approx(0.0)
    assert timeline.events[0].sampling_period_ns == pytest.approx(0.4)
    assert np.array_equal(timeline.events[0].waveform, waveform)
    assert timeline.length_ns == pytest.approx(1.2)
    assert timeline.modulation_frequency_hz == pytest.approx(100_000_000.0)
    assert len(timeline.capture_windows) == 1
    assert timeline.capture_windows[0].name == "capture_0"
    assert timeline.capture_windows[0].start_offset_ns == pytest.approx(0.4)
    assert timeline.capture_windows[0].length_ns == pytest.approx(0.4)


def test_quel3_adapter_uses_backend_alias_resolver_hook() -> None:
    """Given alias resolver hook, when building request, then payload includes resolved aliases."""
    target = "RQ00"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                target: _FakeSequence(
                    waveforms=[
                        _FakeWaveform(
                            np.array([0.0 + 0.0j], dtype=np.complex128),
                            sampling_period=0.4,
                        )
                    ]
                )
            },
        ),
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=[target],
                    start_time=0.4,
                    duration=0.4,
                ),
            ]
        ),
    )
    backend_controller = type(
        "_BC",
        (),
        {
            "DEFAULT_SAMPLING_PERIOD": 0.4,
            "resolve_instrument_alias": staticmethod(lambda value: f"alias-{value}"),
        },
    )()
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=backend_controller,
        experiment_system=cast(
            Any,
            type(
                "_ES",
                (),
                {"get_awg_frequency": staticmethod(lambda _: 100_000_000.0)},
            )(),
        ),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
    )

    request = adapter.build_execution_request(schedule=schedule, config=_make_config())

    payload = request.payload
    assert isinstance(payload, Quel3ExecutionPayload)
    assert payload.instrument_aliases == {target: "alias-RQ00"}
    assert payload.output_target_labels == {target: "Q00"}


def test_quel3_adapter_uses_registry_for_output_target_labels() -> None:
    """Given target registry, when building payload, then output labels use registry mapping."""
    target = "raw-readout-target"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                target: _FakeSequence(
                    waveforms=[
                        _FakeWaveform(
                            np.array([0.0 + 0.0j], dtype=np.complex128),
                            sampling_period=0.4,
                        )
                    ]
                )
            },
        ),
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=[target],
                    start_time=0.4,
                    duration=0.4,
                ),
            ]
        ),
    )

    class _TargetRegistry:
        @staticmethod
        def measurement_output_label(label: str) -> str:
            return "Q17" if label == target else label

    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=type("_BC", (), {"DEFAULT_SAMPLING_PERIOD": 0.4})(),
        experiment_system=cast(
            Any,
            type(
                "_ES",
                (),
                {
                    "get_awg_frequency": staticmethod(lambda _: 100_000_000.0),
                    "target_registry": _TargetRegistry(),
                },
            )(),
        ),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
    )

    request = adapter.build_execution_request(schedule=schedule, config=_make_config())

    payload = request.payload
    assert isinstance(payload, Quel3ExecutionPayload)
    assert payload.output_target_labels == {target: "Q17"}
