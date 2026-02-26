"""Tests for Quel3 measurement backend adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

from qubex.backend.quel3 import (
    Quel3BackendController,
    Quel3BackendExecutionResult,
)
from qubex.measurement.adapters import (
    Quel3ExecutionPayload,
    Quel3MeasurementBackendAdapter,
)
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.models import (
    MeasurementConfig,
    MeasurementResult,
    MeasurementSchedule,
)
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.pulse import Arbitrary, PulseArray
from qubex.system import TargetRegistry
from qubex.typing import MeasurementMode


@dataclass
class _FakePulseSchedule:
    duration: float
    sequences: dict[str, PulseArray]
    valid: bool = True

    @property
    def labels(self) -> list[str]:
        return list(self.sequences.keys())

    def is_valid(self) -> bool:
        return self.valid

    def get_sequence(self, label: str, *, copy: bool = False) -> PulseArray:
        del copy
        return self.sequences[label]

    def get_sampled_sequences(self) -> dict[str, np.ndarray]:
        raise AssertionError("Quel3 adapter must not call get_sampled_sequences().")


@dataclass
class _FakeExperimentSystem:
    target_registry: Any = field(default_factory=TargetRegistry)
    awg_frequency: float = 100_000_000.0

    def get_awg_frequency(self, _: str) -> float:
        return self.awg_frequency

    def resolve_qubit_label(self, label: str) -> str:
        resolver = getattr(self.target_registry, "resolve_qubit_label", None)
        if not callable(resolver):
            raise ValueError(  # noqa: TRY004
                f"Qubit label could not be resolved from `{label}`."
            )
        try:
            return str(resolver(label, allow_legacy=True))
        except TypeError:
            return str(resolver(label))


def _make_backend_controller() -> Quel3BackendController:
    return Quel3BackendController()


def _make_config(
    *,
    mode: MeasurementMode = "avg",
    shots: int = 16,
) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=shots,
        shot_interval_ns=100.0,
        shot_averaging=(mode == "avg"),
        time_integration=False,
        state_classification=False,
    )


def _pulse_array(
    *,
    values: np.ndarray,
    sampling_period: float = 0.4,
    scale: float | None = None,
    phase: float | None = None,
) -> PulseArray:
    return PulseArray(
        [
            Arbitrary(
                values=values,
                sampling_period=sampling_period,
                scale=scale,
                phase=phase,
            )
        ]
    )


def test_quel3_adapter_accepts_relaxed_schedule() -> None:
    """Given relaxed schedule, when validating, then no error is raised."""
    target = "RQ00"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=12.4,
            sequences={target: _pulse_array(values=np.array([0.1 + 0.0j] * 31))},
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
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
    )

    result = adapter.validate_schedule(schedule)
    assert result is None


def test_quel3_adapter_rejects_capture_outside_pulse_duration() -> None:
    """Given out-of-range capture, when validating, then ValueError is raised."""
    target = "RQ00"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=10.0,
            sequences={target: _pulse_array(values=np.array([0.1 + 0.0j] * 25))},
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
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
    )

    with pytest.raises(ValueError, match="exceeds pulse schedule duration"):
        adapter.validate_schedule(schedule)


def test_quel3_adapter_builds_fixed_timeline_payload() -> None:
    """Given schedule and config, when building request, then payload contains per-alias timeline and captures."""
    target = "RQ00"
    alias = "alias-RQ00"
    shape = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                target: _pulse_array(values=shape, sampling_period=0.4, scale=0.5)
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
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
        instrument_alias_map={target: alias},
    )

    request = adapter.build_execution_request(schedule=schedule, config=_make_config())

    payload = request.payload
    assert isinstance(payload, Quel3ExecutionPayload)
    assert payload.interval_ns == 102
    assert payload.repeats == 16
    assert payload.mode == "avg"
    assert alias in payload.fixed_timelines
    timeline = payload.fixed_timelines[alias]
    assert len(timeline.events) == 1
    event = timeline.events[0]
    assert event.start_offset_ns == pytest.approx(0.0)
    assert event.gain == pytest.approx(0.5)
    assert event.phase_offset_deg == pytest.approx(0.0)
    assert event.waveform_name in payload.waveform_library
    waveform_def = payload.waveform_library[event.waveform_name]
    assert waveform_def.sampling_period_ns == pytest.approx(0.4)
    assert np.array_equal(
        waveform_def.iq_array,
        shape,
    )
    assert timeline.length_ns == pytest.approx(1.2)
    assert len(timeline.capture_windows) == 1
    assert timeline.capture_windows[0].name == f"{alias}:0"
    assert timeline.capture_windows[0].start_offset_ns == pytest.approx(0.4)
    assert timeline.capture_windows[0].length_ns == pytest.approx(0.4)


def test_quel3_adapter_keeps_zero_regions_inside_one_waveform_event() -> None:
    """Given zeros inside one pulse, when building payload, then adapter keeps one event."""
    target = "RQ00"
    alias = "alias-RQ00"
    waveform = np.array(
        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j],
        dtype=np.complex128,
    )
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=2.0,
            sequences={target: _pulse_array(values=waveform, sampling_period=0.4)},
        ),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
        instrument_alias_map={target: alias},
    )

    request = adapter.build_execution_request(schedule=schedule, config=_make_config())

    payload = request.payload
    assert isinstance(payload, Quel3ExecutionPayload)
    timeline = payload.fixed_timelines[alias]
    assert len(timeline.events) == 1
    event = timeline.events[0]
    assert event.start_offset_ns == pytest.approx(0.0)
    assert event.gain == pytest.approx(1.0)
    assert event.phase_offset_deg == pytest.approx(0.0)
    waveform_def = payload.waveform_library[event.waveform_name]
    assert np.array_equal(
        waveform_def.iq_array,
        np.array(
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j],
            dtype=np.complex128,
        ),
    )


def test_quel3_adapter_uses_adapter_alias_map() -> None:
    """Given adapter alias map, when building request, then fixed timeline keys use aliases."""
    target = "RQ00"
    alias = "alias-RQ00"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                target: _pulse_array(
                    values=np.array([0.0 + 0.0j], dtype=np.complex128),
                    sampling_period=0.4,
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
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
        instrument_alias_map={target: alias},
    )

    request = adapter.build_execution_request(schedule=schedule, config=_make_config())

    payload = request.payload
    assert isinstance(payload, Quel3ExecutionPayload)
    assert set(payload.fixed_timelines.keys()) == {alias}


def test_quel3_adapter_rejects_multiple_targets_for_same_alias() -> None:
    """Given duplicated alias mapping, when building payload, then ValueError is raised."""
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                "RQ00": _pulse_array(
                    values=np.array([0.0 + 0.0j], dtype=np.complex128),
                    sampling_period=0.4,
                ),
                "RQ01": _pulse_array(
                    values=np.array([0.0 + 0.0j], dtype=np.complex128),
                    sampling_period=0.4,
                ),
            },
        ),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
        instrument_alias_map={"RQ00": "alias-shared", "RQ01": "alias-shared"},
    )

    with pytest.raises(ValueError, match="Multiple targets mapped to one"):
        adapter.build_execution_request(schedule=schedule, config=_make_config())


def test_quel3_adapter_uses_registry_for_result_target_labels() -> None:
    """Given target registry, when building measurement result, then backend alias labels are converted to registry output labels."""
    target = "raw-readout-target"
    alias = "alias-raw"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                target: _pulse_array(
                    values=np.array([0.0 + 0.0j], dtype=np.complex128),
                    sampling_period=0.4,
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
        backend_controller=_make_backend_controller(),
        experiment_system=cast(
            Any,
            _FakeExperimentSystem(target_registry=_TargetRegistry()),
        ),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
        instrument_alias_map={target: alias},
    )

    _ = adapter.build_execution_request(schedule=schedule, config=_make_config())
    backend_result = Quel3BackendExecutionResult(
        mode="avg",
        data={alias: [np.array([1.0 + 0.0j], dtype=np.complex128)]},
    )
    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(),
        device_config={},
        sampling_period_ns=0.4,
    )

    assert set(result.data.keys()) == {"Q17"}


def test_quel3_adapter_reuses_shared_shape_with_scale_and_phase() -> None:
    """Given scale/phase variants, when building payload, then events reuse one waveform and keep scale/phase."""
    target_a = "RQ00"
    target_b = "RQ01"
    base = np.array([1.0 + 0.5j, 0.5 + 0.25j], dtype=np.complex128)
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=0.8,
            sequences={
                target_a: _pulse_array(values=base, sampling_period=0.4),
                target_b: _pulse_array(
                    values=base,
                    sampling_period=0.4,
                    scale=0.6,
                    phase=np.deg2rad(30.0),
                ),
            },
        ),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
        instrument_alias_map={target_a: "alias-RQ00", target_b: "alias-RQ01"},
    )

    request = adapter.build_execution_request(schedule=schedule, config=_make_config())

    payload = request.payload
    assert isinstance(payload, Quel3ExecutionPayload)
    assert len(payload.waveform_library) == 1
    event_a = payload.fixed_timelines["alias-RQ00"].events[0]
    event_b = payload.fixed_timelines["alias-RQ01"].events[0]
    assert event_a.waveform_name == event_b.waveform_name
    assert event_a.gain == pytest.approx(1.0)
    assert event_a.phase_offset_deg == pytest.approx(0.0)
    assert event_b.gain == pytest.approx(0.6)
    assert event_b.phase_offset_deg == pytest.approx(30.0)


def test_quel3_adapter_requires_explicit_alias_mapping_for_execution_payload() -> None:
    """Given missing alias mapping, when building payload, then a configuration error is raised."""
    target = "RQ00"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                target: _pulse_array(
                    values=np.array([0.0 + 0.0j], dtype=np.complex128),
                    sampling_period=0.4,
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
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
    )

    with pytest.raises(ValueError, match="Instrument alias is not configured"):
        adapter.build_execution_request(schedule=schedule, config=_make_config())


def test_quel3_adapter_build_measurement_result_rejects_measurement_result() -> None:
    """Given canonical result input, when adapter builds result, then it raises TypeError."""
    unexpected = MeasurementResult(
        data={"Q00": [np.array([1.0 + 0.0j], dtype=np.complex128)]},
        measurement_config=_make_config(mode="avg"),
        device_config={"kind": "quel3"},
        sampling_period_ns=0.4,
    )
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
    )

    with pytest.raises(TypeError, match="Quel3BackendExecutionResult"):
        adapter.build_measurement_result(
            backend_result=cast(Any, unexpected),
            measurement_config=_make_config(),
            device_config={"kind": "quel3"},
            sampling_period_ns=0.4,
        )


def test_quel3_adapter_build_measurement_result_converts_backend_result() -> None:
    """Given QuEL-3 backend result, when adapter builds result, then it converts to canonical model."""
    target = "RQ00"
    alias = "alias-RQ00"
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=_FakePulseSchedule(
            duration=1.2,
            sequences={
                target: _pulse_array(
                    values=np.array([0.0 + 0.0j], dtype=np.complex128),
                    sampling_period=0.4,
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
    config = _make_config()
    backend_result = Quel3BackendExecutionResult(
        mode="avg",
        data={alias: [np.array([2.0 + 0.0j], dtype=np.complex128)]},
        sampling_period_ns=0.4,
    )
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
        instrument_alias_map={target: alias},
    )
    _ = adapter.build_execution_request(schedule=schedule, config=_make_config())

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=config,
        device_config={"unused": True},
        sampling_period_ns=1.0,
    )

    assert isinstance(result, MeasurementResult)
    assert result.measurement_config.shot_averaging is True
    assert result.device_config == {}
    assert result.measurement_config == config
    assert result.sampling_period_ns == pytest.approx(0.4)
    assert np.array_equal(
        result.data["Q00"][0],
        np.array([2.0 + 0.0j], dtype=np.complex128),
    )


def test_quel3_adapter_build_measurement_result_rejects_noncanonical_type() -> None:
    """Given non-canonical backend result, when adapter builds result, then it raises TypeError."""
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=_make_backend_controller(),
        experiment_system=cast(Any, _FakeExperimentSystem()),
        constraint_profile=MeasurementConstraintProfile.quel3(0.4),
    )

    with pytest.raises(TypeError, match="Quel3BackendExecutionResult"):
        adapter.build_measurement_result(
            backend_result=cast(Any, {"iq_result": {}}),
            measurement_config=_make_config(),
            device_config={},
            sampling_period_ns=0.4,
        )
