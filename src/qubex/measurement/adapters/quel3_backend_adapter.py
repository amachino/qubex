"""QuEL-3 measurement backend adapter implementation."""

from __future__ import annotations

import math

import numpy as np
from qxpulse import Blank, Pulse, PulseArray

from qubex.backend import (
    BackendExecutionRequest,
    ExperimentSystem,
)
from qubex.backend.quel3 import (
    Quel3BackendController,
    Quel3BackendExecutionResult,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
    Quel3WaveformEvent,
)
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule


class Quel3MeasurementBackendAdapter:
    """Measurement backend adapter for QuEL-3 fixed-timeline execution."""

    def __init__(
        self,
        *,
        backend_controller: Quel3BackendController,
        experiment_system: ExperimentSystem,
        constraint_profile: MeasurementConstraintProfile | None = None,
    ) -> None:
        self._backend_controller = backend_controller
        self._experiment_system = experiment_system
        self._output_target_labels_by_alias: dict[str, str] = {}
        if constraint_profile is None:
            constraint_profile = MeasurementConstraintProfile.quel3(
                sampling_period_ns=backend_controller.sampling_period
            )
        self._constraint_profile = constraint_profile

    def validate_schedule(self, schedule: MeasurementSchedule) -> None:
        """Validate schedule with relaxed Quel3 constraints."""
        pulse_schedule = schedule.pulse_schedule
        if not pulse_schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")
        channel_captures = schedule.capture_schedule.channels
        if len(channel_captures) == 0:
            raise ValueError("Capture schedule must not be empty.")
        for channel, captures in channel_captures.items():
            for capture in captures:
                if capture.start_time < 0:
                    raise ValueError(
                        f"Capture start time must be non-negative: {channel}."
                    )
                if capture.duration <= 0:
                    raise ValueError(f"Capture duration must be positive: {channel}.")
                if capture.start_time + capture.duration > pulse_schedule.duration:
                    raise ValueError(
                        f"Capture exceeds pulse schedule duration: {channel}."
                    )

    def build_execution_request(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> BackendExecutionRequest:
        """Build backend execution request as Quel3 fixed-timeline payload."""
        pulse_schedule = schedule.pulse_schedule
        channel_captures = schedule.capture_schedule.channels
        waveform_library: dict[str, Quel3Waveform] = {}
        waveform_name_by_shape_key: dict[str, str] = {}
        waveform_index = 0
        fixed_timelines: dict[str, Quel3FixedTimeline] = {}
        output_target_labels_by_alias: dict[str, str] = {}
        target_registry = self._experiment_system.target_registry
        alias_map = self._backend_controller.instrument_alias_map

        for target in pulse_schedule.labels:
            alias = str(alias_map.get(target, "")).strip()
            if len(alias) == 0:
                raise ValueError(
                    f"Instrument alias is not configured for target `{target}`."
                )
            if alias in fixed_timelines:
                raise ValueError(
                    f"Multiple targets mapped to one instrument alias are not supported: `{alias}`."
                )

            sequence = pulse_schedule.get_sequence(target, copy=False)
            events, waveform_index = self._create_waveform_events(
                sequence=sequence,
                waveform_name_by_shape_key=waveform_name_by_shape_key,
                waveform_library=waveform_library,
                waveform_index=waveform_index,
            )
            captures = sorted(
                channel_captures.get(target, []), key=lambda c: c.start_time
            )
            capture_windows = tuple(
                Quel3CaptureWindow(
                    name=f"{alias}:{index}",
                    start_offset_ns=capture.start_time,
                    length_ns=capture.duration,
                )
                for index, capture in enumerate(captures)
            )
            fixed_timelines[alias] = Quel3FixedTimeline(
                events=events,
                capture_windows=capture_windows,
                length_ns=pulse_schedule.duration,
            )
            try:
                output_target_labels_by_alias[alias] = str(
                    self._experiment_system.resolve_qubit_label(target)
                )
            except ValueError:
                output_target_labels_by_alias[alias] = str(
                    target_registry.measurement_output_label(target)
                )

        self._output_target_labels_by_alias = output_target_labels_by_alias
        interval_ns = math.ceil(pulse_schedule.duration + config.interval)
        payload = Quel3ExecutionPayload(
            waveform_library=waveform_library,
            fixed_timelines=fixed_timelines,
            interval_ns=interval_ns,
            repeats=config.shots,
            mode=config.mode,
        )
        return BackendExecutionRequest(payload=payload)

    def build_measurement_result(
        self,
        *,
        backend_result: Quel3BackendExecutionResult,
        measurement_config: MeasurementConfig,
        device_config: dict,
        sampling_period_ns: float | None,
    ) -> MeasurementResult:
        """Build canonical result from QuEL-3 backend result payload."""
        _ = device_config
        if not isinstance(backend_result, Quel3BackendExecutionResult):
            raise TypeError("QuEL-3 backend must return `Quel3BackendExecutionResult`.")
        converted_data: dict[str, list[np.ndarray]] = {}
        for alias, values in backend_result.data.items():
            output_target = self._output_target_labels_by_alias.get(alias, alias)
            converted_data.setdefault(output_target, []).extend(values)
        return MeasurementResult(
            mode=backend_result.mode,
            data=converted_data,
            device_config={},
            measurement_config=measurement_config.to_dict(),
            sampling_period_ns=(
                backend_result.sampling_period_ns
                if backend_result.sampling_period_ns is not None
                else sampling_period_ns
            ),
        )

    @classmethod
    def _create_waveform_events(
        cls,
        *,
        sequence: PulseArray,
        waveform_name_by_shape_key: dict[str, str],
        waveform_library: dict[str, Quel3Waveform],
        waveform_index: int,
    ) -> tuple[tuple[Quel3WaveformEvent, ...], int]:
        """Create sparse waveform events and shared waveform library entries."""
        events: list[Quel3WaveformEvent] = []
        current_offset_ns = 0.0
        for waveform in sequence.get_flattened_waveforms(apply_frame_shifts=True):
            duration_ns = waveform.duration
            if isinstance(waveform, Blank):
                current_offset_ns += duration_ns
                continue

            if isinstance(waveform, Pulse):
                sampling_period_ns = waveform.sampling_period
                scale = waveform.scale
                shape = np.asarray(waveform.shape_values, dtype=np.complex128)
                shape_hash = waveform.shape_hash
                waveform_name = waveform_name_by_shape_key.get(shape_hash)
                if waveform_name is None:
                    waveform_name = f"waveform_{waveform_index:04d}"
                    waveform_index += 1
                    waveform_library[waveform_name] = Quel3Waveform(
                        iq_array=shape,
                        sampling_period_ns=sampling_period_ns,
                    )
                    waveform_name_by_shape_key[shape_hash] = waveform_name
                events.append(
                    Quel3WaveformEvent(
                        waveform_name=waveform_name,
                        start_offset_ns=current_offset_ns,
                        gain=scale,
                        phase_offset_deg=math.degrees(waveform.phase),
                    )
                )
                current_offset_ns += duration_ns
                continue

            raise TypeError(
                f"Unsupported waveform type in PulseArray: {type(waveform).__name__}."
            )
        return tuple(events), waveform_index
