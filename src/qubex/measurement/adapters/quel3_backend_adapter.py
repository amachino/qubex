"""QuEL-3 measurement backend adapter implementation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
from qxpulse import Blank, Pulse, PulseArray

from qubex.backend import (
    BackendExecutionRequest,
)
from qubex.backend.quel3 import (
    Quel3BackendController,
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
    Quel3WaveformEvent,
)
from qubex.backend.quel3.quel3_backend_constants import READOUT_SAMPLING_PERIOD_NS
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.models.capture_data import CaptureData
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions
from qubex.system import ExperimentSystem
from qubex.system.target_type import TargetType


def _as_read_only_array(data: object) -> np.ndarray:
    """Return read-only NumPy array view for capture payloads."""
    array = np.asarray(data).view()
    array.setflags(write=False)
    return array


class Quel3MeasurementBackendAdapter:
    """Measurement backend adapter for QuEL-3 fixed-timeline execution."""

    def __init__(
        self,
        *,
        backend_controller: Quel3BackendController,
        experiment_system: ExperimentSystem,
        constraint_profile: MeasurementConstraintProfile | None = None,
        instrument_alias_map: Mapping[str, str] | None = None,
    ) -> None:
        self._backend_controller = backend_controller
        self._experiment_system = experiment_system
        self._instrument_alias_map = dict(instrument_alias_map or {})
        self._output_target_labels_by_target: dict[str, str] = {}
        self._capture_targets_by_alias: dict[str, list[str]] = {}
        if constraint_profile is None:
            constraint_profile = MeasurementConstraintProfile.quel3(
                sampling_period_ns=backend_controller.sampling_period_ns
            )
        self._constraint_profile = constraint_profile

    @property
    def instrument_alias_map(self) -> Mapping[str, str]:
        """Return configured target-to-instrument alias mapping."""
        return self._instrument_alias_map

    def set_instrument_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Replace full target-to-alias mapping in adapter layer."""
        self._instrument_alias_map = dict(alias_map)

    def update_instrument_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Update target-to-alias mapping entries in adapter layer."""
        self._instrument_alias_map.update(alias_map)

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
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> BackendExecutionRequest:
        """Build backend execution request as Quel3 fixed-timeline payload."""
        _ = quel1_options
        pulse_schedule = schedule.pulse_schedule
        channel_captures = schedule.capture_schedule.channels
        waveform_library: dict[str, Quel3Waveform] = {}
        waveform_name_by_shape_key: dict[tuple[str, int], str] = {}
        waveform_index = 0
        fixed_timelines: dict[str, Quel3FixedTimeline] = {}
        output_target_labels_by_target: dict[str, str] = {}
        instrument_bindings: dict[str, str] = {}
        target_registry = self._experiment_system.target_registry
        alias_map = self._instrument_alias_map

        for target in pulse_schedule.labels:
            configured_alias = str(alias_map.get(target, "")).strip()
            if len(configured_alias) == 0:
                raise ValueError(
                    "Missing QuEL-3 instrument alias mapping for "
                    f"target `{target}`. Configure instruments before measurement."
                )
            instrument_bindings[target] = f"alias:{configured_alias}"

            sequence = pulse_schedule.get_sequence(target, copy=False)
            target_type = self._experiment_system.get_target(target).type
            events, waveform_index = self._create_waveform_events(
                target_is_read=(target_type is TargetType.READ),
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
                    name=f"{target}:{index}",
                    start_offset_ns=capture.start_time,
                    length_ns=capture.duration,
                )
                for index, capture in enumerate(captures)
            )
            fixed_timelines[target] = Quel3FixedTimeline(
                events=events,
                capture_windows=capture_windows,
                length_ns=pulse_schedule.duration,
            )
            try:
                output_target_labels_by_target[target] = str(
                    self._experiment_system.resolve_qubit_label(target)
                )
            except ValueError:
                output_target_labels_by_target[target] = str(
                    target_registry.measurement_output_label(target)
                )

        self._output_target_labels_by_target = output_target_labels_by_target
        interval_ns = math.ceil(pulse_schedule.duration + config.shot_interval)
        capture_mode = self._resolve_capture_mode(config)
        payload = Quel3ExecutionPayload(
            waveform_library=waveform_library,
            fixed_timelines=fixed_timelines,
            interval_ns=interval_ns,
            repeats=config.n_shots,
            capture_mode=capture_mode,
            instrument_bindings=instrument_bindings,
        )
        self._capture_targets_by_alias = self._build_capture_targets_by_alias(payload)
        return BackendExecutionRequest(payload=payload)

    def build_measurement_result(
        self,
        *,
        backend_result: Quel3BackendExecutionResult,
        measurement_config: MeasurementConfig,
        device_config: dict,
        sampling_period: float,
    ) -> MeasurementResult:
        """Build canonical result from QuEL-3 backend result payload."""
        _ = device_config
        if not isinstance(backend_result, Quel3BackendExecutionResult):
            raise TypeError("QuEL-3 backend must return `Quel3BackendExecutionResult`.")
        backend_sampling_period = backend_result.config.get("sampling_period_ns")
        if backend_sampling_period is None:
            resolved_sampling_period = sampling_period
        elif isinstance(backend_sampling_period, (int, float)):
            resolved_sampling_period = backend_sampling_period
        else:
            raise TypeError(
                "QuEL-3 backend result config `sampling_period_ns` must be numeric."
            )
        converted_data: dict[str, list[CaptureData]] = {}
        for alias, values in backend_result.data.items():
            capture_targets = self._capture_targets_by_alias.get(alias)
            if capture_targets is None:
                output_target = self._output_target_labels_by_target.get(alias, alias)
                converted_data.setdefault(output_target, []).extend(
                    [
                        CaptureData.from_primary_data(
                            target=output_target,
                            data=_as_read_only_array(value),
                            config=measurement_config,
                            sampling_period=resolved_sampling_period,
                        )
                        for value in values
                    ]
                )
                continue
            if len(capture_targets) != len(values):
                raise ValueError(
                    f"Capture target count mismatch for alias `{alias}`: "
                    f"targets={len(capture_targets)} values={len(values)}."
                )
            for capture_target, capture_value in zip(
                capture_targets, values, strict=True
            ):
                output_target = self._output_target_labels_by_target.get(
                    capture_target,
                    capture_target,
                )
                converted_data.setdefault(output_target, []).append(
                    CaptureData.from_primary_data(
                        target=output_target,
                        data=_as_read_only_array(capture_value),
                        config=measurement_config,
                        sampling_period=resolved_sampling_period,
                    )
                )
        return MeasurementResult(
            data=converted_data,
            device_config={},
            measurement_config=measurement_config,
        )

    @staticmethod
    def _resolve_capture_mode(config: MeasurementConfig) -> Quel3CaptureMode:
        """Resolve quelware-compatible capture mode from measurement config."""
        if config.shot_averaging:
            if config.time_integration:
                return Quel3CaptureMode.AVERAGED_VALUE
            return Quel3CaptureMode.AVERAGED_WAVEFORM
        if config.time_integration:
            return Quel3CaptureMode.VALUES_PER_ITER
        return Quel3CaptureMode.RAW_WAVEFORMS

    @staticmethod
    def _build_capture_targets_by_alias(
        payload: Quel3ExecutionPayload,
    ) -> dict[str, list[str]]:
        """Build alias-to-target capture mapping from the request payload."""
        alias_to_entries: dict[str, list[tuple[float, float, int, str]]] = defaultdict(
            list
        )
        sequence_index = 0
        for target, timeline in payload.fixed_timelines.items():
            binding = payload.instrument_bindings.get(target, f"alias:{target}")
            if binding.startswith("alias:"):
                alias = binding.removeprefix("alias:").strip() or target
            else:
                # Port-based bindings are resolved in backend runtime, so keep
                # per-target mapping keys in adapter-side bookkeeping.
                alias = target
            for _event in timeline.events:
                sequence_index += 1
            for window in timeline.capture_windows:
                alias_to_entries[alias].append(
                    (
                        window.start_offset_ns,
                        window.length_ns,
                        sequence_index,
                        target,
                    )
                )
                sequence_index += 1
        return {
            alias: [
                target
                for _start, _length, _order, target in sorted(
                    entries,
                    key=lambda item: (item[0], item[1], item[2]),
                )
            ]
            for alias, entries in alias_to_entries.items()
        }

    @classmethod
    def _create_waveform_events(
        cls,
        *,
        target_is_read: bool,
        sequence: PulseArray,
        waveform_name_by_shape_key: dict[tuple[str, int], str],
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
                if shape.size == 0:
                    current_offset_ns += duration_ns
                    continue
                shape, sampling_period_ns = cls._normalize_waveform_for_target(
                    target_is_read=target_is_read,
                    shape=shape,
                    sampling_period_ns=sampling_period_ns,
                )
                shape_key = (
                    waveform.shape_hash,
                    round(float(sampling_period_ns) * 1e6),
                )
                waveform_name = waveform_name_by_shape_key.get(shape_key)
                if waveform_name is None:
                    waveform_name = f"waveform_{waveform_index:04d}"
                    waveform_index += 1
                    waveform_library[waveform_name] = Quel3Waveform(
                        iq_array=shape,
                        sampling_period_ns=sampling_period_ns,
                    )
                    waveform_name_by_shape_key[shape_key] = waveform_name
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

    @staticmethod
    def _normalize_waveform_for_target(
        *,
        target_is_read: bool,
        shape: np.ndarray,
        sampling_period_ns: float,
    ) -> tuple[np.ndarray, float]:
        """
        Normalize waveform sampling periods for QuEL-3 target classes.

        This is a temporary QuEL-3 workaround while Qubex still carries one
        backend-level sampling period instead of per-channel `dt`.
        Control waveforms stay on the shared QuEL-3 control grid (0.4 ns).
        Readout waveforms are normalized here to the readout grid
        (`READOUT_SAMPLING_PERIOD_NS`)
        before registration so mixed control/readout schedules can still be
        executed through the current single-`dt` stack.
        """
        if not target_is_read:
            return shape, sampling_period_ns

        readout_sampling_period_ns = READOUT_SAMPLING_PERIOD_NS
        if np.isclose(sampling_period_ns, readout_sampling_period_ns):
            return shape, sampling_period_ns

        ratio = readout_sampling_period_ns / sampling_period_ns
        rounded_ratio = round(ratio)
        if rounded_ratio <= 0 or not np.isclose(ratio, rounded_ratio):
            raise ValueError(
                "Readout waveform sampling period must divide the QuEL-3 readout "
                "sampling period exactly: "
                f"sampling_period_ns={sampling_period_ns}."
            )
        remainder = shape.size % rounded_ratio
        if remainder != 0:
            pad_width = rounded_ratio - remainder
            shape = np.pad(shape, (0, pad_width), mode="edge")
        reshaped = shape.reshape(-1, rounded_ratio)
        return reshaped.mean(axis=1), readout_sampling_period_ns
