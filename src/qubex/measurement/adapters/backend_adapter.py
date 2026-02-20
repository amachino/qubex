"""Backend-adapter layer for measurement schedule execution."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from qxpulse import Blank

from qubex.backend import (
    BackendExecutionRequest,
    ExperimentSystem,
    TargetRegistry,
)
from qubex.backend.quel1 import (
    Quel1BackendController,
    Quel1ExecutionPayload,
)
from qubex.backend.quel3 import (
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3TargetTimeline,
    Quel3WaveformDefinition,
    Quel3WaveformEvent,
)
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.models.measure_result import MeasureMode
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule


class MeasurementBackendAdapter(Protocol):
    """Protocol for converting measurement requests into backend requests."""

    def validate_schedule(self, schedule: MeasurementSchedule) -> None:
        """Validate backend-specific constraints for a measurement schedule."""
        ...

    def build_execution_request(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> BackendExecutionRequest:
        """Build backend execution request from measurement schedule/config."""
        ...


@dataclass(frozen=True)
class _WaveformSegment:
    """One contiguous non-blank waveform segment."""

    start_index: int
    values: np.ndarray


class Quel3MeasurementBackendAdapter:
    """Relaxed backend adapter that builds Quel3 fixed-timeline payloads."""

    def __init__(
        self,
        *,
        backend_controller: Any,
        experiment_system: ExperimentSystem,
        constraint_profile: MeasurementConstraintProfile | None = None,
    ) -> None:
        self._backend_controller = backend_controller
        self._experiment_system = experiment_system
        if constraint_profile is None:
            sampling_period = getattr(
                backend_controller, "DEFAULT_SAMPLING_PERIOD", None
            )
            if not isinstance(sampling_period, (float, int)):
                raise ValueError(
                    "Quel3MeasurementBackendAdapter requires a relaxed constraint profile or backend DEFAULT_SAMPLING_PERIOD."
                )
            constraint_profile = MeasurementConstraintProfile.quel3(
                sampling_period_ns=float(sampling_period)
            )
        self._constraint_profile = constraint_profile

    @property
    def sampling_period(self) -> float:
        """Return sampling period (ns)."""
        return self.constraint_profile.sampling_period_ns

    @property
    def constraint_profile(self) -> MeasurementConstraintProfile:
        """Return backend measurement constraints."""
        return self._constraint_profile

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
        waveform_library: dict[str, Quel3WaveformDefinition] = {}
        waveform_name_by_shape_key: dict[str, str] = {}
        waveform_index = 0
        timelines: dict[str, Quel3TargetTimeline] = {}
        instrument_aliases: dict[str, str] = {}
        output_target_labels: dict[str, str] = {}
        alias_resolver = getattr(
            self._backend_controller,
            "resolve_instrument_alias",
            None,
        )
        target_registry = getattr(self._experiment_system, "target_registry", None)
        fallback_registry = TargetRegistry()

        def _resolve_qubit_label(target: str) -> str:
            resolve_qubit_label = getattr(
                self._experiment_system,
                "resolve_qubit_label",
                None,
            )
            if callable(resolve_qubit_label):
                return str(resolve_qubit_label(target))

            if target_registry is not None and hasattr(
                target_registry,
                "resolve_qubit_label",
            ):
                resolver = target_registry.resolve_qubit_label
                try:
                    return str(resolver(target, allow_legacy=True))
                except TypeError:
                    return str(resolver(target))

            return fallback_registry.resolve_qubit_label(target, allow_legacy=True)

        for target in pulse_schedule.labels:
            sequence = pulse_schedule.get_sequence(target, copy=False)
            events, waveform_index = self._create_quel3_events(
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
                    name=f"capture_{index}",
                    start_offset_ns=float(capture.start_time),
                    length_ns=float(capture.duration),
                )
                for index, capture in enumerate(captures)
            )
            modulation_frequency_hz: float | None
            try:
                modulation_frequency_hz = float(
                    self._experiment_system.get_awg_frequency(target)
                )
            except Exception:
                modulation_frequency_hz = None
            timelines[target] = Quel3TargetTimeline(
                sampling_period_ns=self.sampling_period,
                events=events,
                capture_windows=capture_windows,
                length_ns=float(pulse_schedule.duration),
                modulation_frequency_hz=modulation_frequency_hz,
            )
            if callable(alias_resolver):
                instrument_aliases[target] = str(alias_resolver(target))
            else:
                instrument_aliases[target] = target
            if target_registry is not None and hasattr(
                target_registry, "measurement_output_label"
            ):
                output_target_labels[target] = str(
                    target_registry.measurement_output_label(target)
                )
            else:
                try:
                    output_target_labels[target] = _resolve_qubit_label(target)
                except ValueError:
                    output_target_labels[target] = target
        interval_ns = math.ceil(float(pulse_schedule.duration + config.interval))
        payload = Quel3ExecutionPayload(
            waveform_library=waveform_library,
            timelines=timelines,
            instrument_aliases=instrument_aliases,
            output_target_labels=output_target_labels,
            interval_ns=interval_ns,
            repeats=config.shots,
            mode=config.mode,
            dsp_demodulation=config.dsp.enable_dsp_demodulation,
            enable_sum=config.dsp.enable_dsp_sum,
            enable_classification=config.dsp.enable_dsp_classification,
            line_param0=config.dsp.line_param0,
            line_param1=config.dsp.line_param1,
        )
        return BackendExecutionRequest(payload=payload)

    @classmethod
    def _create_quel3_events(
        cls,
        *,
        sequence: Any,
        waveform_name_by_shape_key: dict[str, str],
        waveform_library: dict[str, Quel3WaveformDefinition],
        waveform_index: int,
    ) -> tuple[tuple[Quel3WaveformEvent, ...], int]:
        """Create sparse waveform events and shared waveform library entries."""
        events: list[Quel3WaveformEvent] = []
        current_offset_ns = 0.0
        for waveform in sequence.get_flattened_waveforms(apply_frame_shifts=True):
            duration_ns = float(waveform.duration)
            if isinstance(waveform, Blank):
                current_offset_ns += duration_ns
                continue
            sampled = np.asarray(waveform.values, dtype=np.complex128)
            sampling_period_ns = float(waveform.sampling_period)
            for segment in cls._iter_non_blank_segments(sampled):
                shape, gain, phase_offset_deg = cls._factor_shape(segment.values)
                shape_key = cls._shape_key(
                    shape=shape,
                    sampling_period_ns=sampling_period_ns,
                )
                waveform_name = waveform_name_by_shape_key.get(shape_key)
                if waveform_name is None:
                    waveform_name = f"wf_shared_{waveform_index:04d}"
                    waveform_index += 1
                    waveform_library[waveform_name] = Quel3WaveformDefinition(
                        waveform=shape,
                        sampling_period_ns=sampling_period_ns,
                    )
                    waveform_name_by_shape_key[shape_key] = waveform_name

                events.append(
                    Quel3WaveformEvent(
                        waveform_name=waveform_name,
                        start_offset_ns=(
                            current_offset_ns
                            + float(segment.start_index) * sampling_period_ns
                        ),
                        gain=gain,
                        phase_offset_deg=phase_offset_deg,
                    )
                )
            current_offset_ns += duration_ns
        return tuple(events), waveform_index

    @classmethod
    def _iter_non_blank_segments(
        cls,
        waveform: np.ndarray,
    ) -> Iterator[_WaveformSegment]:
        """Yield contiguous non-blank waveform segments."""
        non_blank_indices = np.flatnonzero(np.abs(waveform) > cls._amplitude_epsilon())
        if non_blank_indices.size == 0:
            return

        start = int(non_blank_indices[0])
        previous = start
        for index in non_blank_indices[1:]:
            index_int = int(index)
            if index_int == previous + 1:
                previous = index_int
                continue
            yield _WaveformSegment(
                start_index=start,
                values=waveform[start : previous + 1],
            )
            start = index_int
            previous = index_int

        yield _WaveformSegment(
            start_index=start,
            values=waveform[start : previous + 1],
        )

    @classmethod
    def _factor_shape(
        cls,
        values: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Factor one segment into normalized shape and complex scalar."""
        amplitudes = np.abs(values)
        peak_index = int(np.argmax(amplitudes))
        peak_value = values[peak_index]
        gain = float(amplitudes[peak_index])
        if gain <= cls._amplitude_epsilon():
            raise ValueError("Non-blank segment peak amplitude must be positive.")
        shape = np.asarray(values / peak_value, dtype=np.complex128)
        phase_offset_deg = float(np.rad2deg(np.angle(peak_value)))
        return shape, gain, phase_offset_deg

    @classmethod
    def _shape_key(
        cls,
        *,
        shape: np.ndarray,
        sampling_period_ns: float,
    ) -> str:
        """Create deterministic deduplication key for one normalized shape."""
        quantized_real = np.rint(shape.real / cls._shape_quantization()).astype(
            np.int64
        )
        quantized_imag = np.rint(shape.imag / cls._shape_quantization()).astype(
            np.int64
        )
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(np.asarray(shape.size, dtype=np.int64).tobytes())
        hasher.update(np.asarray(sampling_period_ns, dtype=np.float64).tobytes())
        hasher.update(quantized_real.tobytes())
        hasher.update(quantized_imag.tobytes())
        return hasher.hexdigest()

    @staticmethod
    def _amplitude_epsilon() -> float:
        """Return amplitude threshold for blank detection."""
        return 1e-12

    @staticmethod
    def _shape_quantization() -> float:
        """Return quantization step used for shape deduplication."""
        return 1e-9


class Quel1MeasurementBackendAdapter:
    """QuEL-1 specific adapter from measurement models to backend request."""

    def __init__(
        self,
        *,
        backend_controller: Quel1BackendController,
        experiment_system: ExperimentSystem,
        constraint_profile: MeasurementConstraintProfile | None = None,
    ) -> None:
        self._backend_controller = backend_controller
        self._experiment_system = experiment_system
        if constraint_profile is None:
            constraint_profile = MeasurementConstraintProfile.quel1()
        self._constraint_profile = constraint_profile

    @property
    def sampling_period(self) -> float:
        """Return sampling period (ns)."""
        return self.constraint_profile.sampling_period_ns

    @property
    def constraint_profile(self) -> MeasurementConstraintProfile:
        """Return backend measurement constraints."""
        return self._constraint_profile

    def validate_schedule(self, schedule: MeasurementSchedule) -> None:
        """Validate QuEL-1 specific pulse/capture constraints."""
        profile = self.constraint_profile
        block_duration = profile.block_duration_ns
        word_duration = profile.word_duration_ns
        pulse_schedule = schedule.pulse_schedule
        if not pulse_schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")
        if (
            profile.enforce_block_alignment
            and block_duration is not None
            and not self._is_multiple(pulse_schedule.duration, block_duration)
        ):
            raise ValueError(
                f"Pulse sequence duration must be a multiple of {block_duration} ns."
            )

        channel_captures = schedule.capture_schedule.channels
        if len(channel_captures) == 0:
            raise ValueError("Capture schedule must not be empty.")

        readout_ranges = pulse_schedule.get_pulse_ranges(list(channel_captures.keys()))
        for channel, captures in channel_captures.items():
            sorted_captures = sorted(captures, key=lambda c: c.start_time)
            if not sorted_captures:
                raise ValueError(f"No capture windows for channel {channel}.")

            first_capture = sorted_captures[0]
            if (
                profile.enforce_block_alignment
                and block_duration is not None
                and not self._is_multiple(first_capture.start_time, block_duration)
            ):
                raise ValueError(
                    "The first capture start time must be a multiple of "
                    f"{block_duration} ns."
                )
            if (
                profile.enforce_word_alignment
                and word_duration is not None
                and not self._is_multiple(first_capture.duration, word_duration)
            ):
                raise ValueError(
                    f"Capture duration must be a multiple of {word_duration} ns."
                )
            workaround_duration = profile.workaround_capture_duration_ns
            if profile.require_workaround_capture and not np.isclose(
                first_capture.duration, workaround_duration
            ):
                raise ValueError(
                    "The first capture must be the workaround capture with duration "
                    f"{workaround_duration} ns."
                )

            for capture in sorted_captures:
                if (
                    profile.enforce_word_alignment
                    and word_duration is not None
                    and not self._is_multiple(capture.start_time, word_duration)
                ):
                    raise ValueError(
                        f"Capture start time must be a multiple of {word_duration} ns."
                    )
                if (
                    profile.enforce_word_alignment
                    and word_duration is not None
                    and not self._is_multiple(capture.duration, word_duration)
                ):
                    raise ValueError(
                        f"Capture duration must be a multiple of {word_duration} ns."
                    )

            ranges = readout_ranges.get(channel, [])
            expected_capture_count = len(ranges) + (
                1 if profile.require_workaround_capture else 0
            )
            if len(sorted_captures) != expected_capture_count:
                raise ValueError(
                    f"Capture schedule mismatch for {channel}: expected {expected_capture_count} captures."
                )
            offset = 1 if profile.require_workaround_capture else 0
            for capture, rng in zip(sorted_captures[offset:], ranges, strict=True):
                expected_start = rng.start * self.sampling_period
                expected_duration = len(rng) * self.sampling_period
                if not np.isclose(capture.start_time, expected_start):
                    raise ValueError(
                        f"Capture start mismatch for {channel}: {capture.start_time} != {expected_start}."
                    )
                if not np.isclose(capture.duration, expected_duration):
                    raise ValueError(
                        f"Capture duration mismatch for {channel}: {capture.duration} != {expected_duration}."
                    )

        for captures in channel_captures.values():
            sorted_captures = sorted(captures, key=lambda c: c.start_time)
            for idx in range(len(sorted_captures) - 1):
                current = sorted_captures[idx]
                nxt = sorted_captures[idx + 1]
                gap = nxt.start_time - (current.start_time + current.duration)
                if (
                    profile.enforce_capture_spacing
                    and word_duration is not None
                    and gap < word_duration
                ):
                    raise ValueError(
                        f"Capture post-blank must be at least {word_duration} ns."
                    )
                if (
                    profile.enforce_capture_spacing
                    and word_duration is not None
                    and not self._is_multiple(gap, word_duration)
                ):
                    raise ValueError(
                        f"Capture post-blank must be a multiple of {word_duration} ns."
                    )

            last = sorted_captures[-1]
            tail_blank = pulse_schedule.duration - (last.start_time + last.duration)
            if tail_blank < 0:
                raise ValueError("Capture schedule exceeds pulse schedule duration.")
            if (
                profile.enforce_capture_spacing
                and word_duration is not None
                and not self._is_multiple(tail_blank, word_duration)
            ):
                raise ValueError(
                    f"Final post-blank must be a multiple of {word_duration} ns."
                )

    def build_execution_request(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> BackendExecutionRequest:
        """Build a QuEL backend execution request from measurement inputs."""
        profile = self.constraint_profile
        block_duration = profile.block_duration_ns
        measure_mode = MeasureMode(config.mode)
        base_duration = schedule.pulse_schedule.duration
        if profile.enforce_block_alignment and block_duration is not None:
            interval = int(
                math.ceil((base_duration + config.interval) / block_duration)
                * block_duration
            )
            # Compatibility guard:
            # legacy measurement flow added one extra block margin on the
            # waveform-path for interval<=0, which avoided negative trailing
            # chunk blanks after converter-side packing/alignment adjustments.
            # Remove this workaround once qubecalib compatibility is no longer
            # required in the QuEL-1 measurement path.
            if config.interval <= 0:
                minimum_interval = int(
                    math.ceil((base_duration + block_duration) / block_duration)
                    * block_duration
                )
                interval = max(interval, minimum_interval)
        else:
            interval = math.ceil(base_duration + config.interval)
        gen_sampled_sequence, cap_sampled_sequence = self._create_sampled_sequences(
            schedule=schedule
        )
        targets = list(
            dict.fromkeys([*gen_sampled_sequence.keys(), *cap_sampled_sequence.keys()])
        )
        resource_map = self._backend_controller.get_resource_map(targets)

        payload = Quel1ExecutionPayload(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map=resource_map,
            interval=interval,
            repeats=config.shots,
            integral_mode=measure_mode.integral_mode,
            dsp_demodulation=config.dsp.enable_dsp_demodulation,
            enable_sum=config.dsp.enable_dsp_sum,
            enable_classification=config.dsp.enable_dsp_classification,
            line_param0=config.dsp.line_param0,
            line_param1=config.dsp.line_param1,
        )
        return BackendExecutionRequest(
            payload=payload,
        )

    def _create_sampled_sequences(
        self,
        *,
        schedule: MeasurementSchedule,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create sampled sequences from measurement schedule."""
        pulse_schedule = schedule.pulse_schedule
        capture_schedule = schedule.capture_schedule
        capture_delays = self._experiment_system.control_params.capture_delay_word

        readout_targets = list(capture_schedule.channels.keys())
        readout_ranges = pulse_schedule.get_pulse_ranges(readout_targets)

        capture_delay_sample: dict[str, int] = {}
        word_length = self.constraint_profile.word_length_samples
        if word_length is None:
            raise ValueError(
                "word_length_samples is required for backend execution request."
            )
        for target in readout_targets:
            target_registry = getattr(self._experiment_system, "target_registry", None)
            fallback_registry = TargetRegistry()
            resolve_qubit_label = getattr(
                self._experiment_system,
                "resolve_qubit_label",
                None,
            )
            if callable(resolve_qubit_label):
                qubit_label = str(resolve_qubit_label(target))
            elif target_registry is not None and hasattr(
                target_registry,
                "resolve_qubit_label",
            ):
                resolver = target_registry.resolve_qubit_label
                try:
                    qubit_label = str(resolver(target, allow_legacy=True))
                except TypeError:
                    qubit_label = str(resolver(target))
            else:
                qubit_label = fallback_registry.resolve_qubit_label(
                    target,
                    allow_legacy=True,
                )
            mux = self._experiment_system.get_mux_by_qubit(qubit_label)
            capture_delay_word = capture_delays.get(mux.index, 0)
            capture_delay_sample[target] = capture_delay_word * word_length

        sampled_sequences = pulse_schedule.get_sampled_sequences(copy=False)
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            seq = sampled_sequences[target]
            omega = 2 * np.pi * self._experiment_system.get_diff_frequency(target)
            delay = capture_delay_sample[target]
            for rng in ranges:
                offset = (rng.start + delay) * self.sampling_period
                seq[rng] *= np.exp(1j * omega * offset)

        gen_sequences: dict[str, Any] = {}
        for target, waveform in sampled_sequences.items():
            if self._experiment_system.get_target(target).sideband != "L":
                waveform = np.conj(waveform)
            gen_sequences[target] = (
                self._backend_controller.create_gen_sampled_sequence(
                    target_name=target,
                    modulation_frequency=self._experiment_system.get_awg_frequency(
                        target
                    ),
                    real=np.real(waveform),
                    imag=np.imag(waveform),
                )
            )

        cap_sequences: dict[str, Any] = {}
        for target, captures in capture_schedule.channels.items():
            sorted_captures = sorted(captures, key=lambda c: c.start_time)
            if not sorted_captures:
                continue

            capture_slots: list[tuple[int, int]] = []
            for idx, current_capture in enumerate(sorted_captures):
                current_start = round(current_capture.start_time / self.sampling_period)
                capture_range_length = round(
                    current_capture.duration / self.sampling_period
                )
                if idx + 1 < len(sorted_captures):
                    next_start = round(
                        sorted_captures[idx + 1].start_time / self.sampling_period
                    )
                    post_blank_length = next_start - (
                        current_start + capture_range_length
                    )
                else:
                    post_blank_length = pulse_schedule.length - (
                        current_start + capture_range_length
                    )
                capture_slots.append((capture_range_length, post_blank_length))

            cap_sequences[target] = (
                self._backend_controller.create_cap_sampled_sequence(
                    target_name=target,
                    modulation_frequency=self._experiment_system.get_awg_frequency(
                        target
                    ),
                    capture_delay=capture_delay_sample[target],
                    capture_slots=capture_slots,
                )
            )

        return gen_sequences, cap_sequences

    @staticmethod
    def _is_multiple(
        value: float,
        base: float,
        *,
        atol: float = 1e-9,
    ) -> bool:
        """Return True if value is a near-integer multiple of base."""
        if base == 0:
            return False
        quotient = value / base
        return math.isclose(quotient, round(quotient), abs_tol=atol)
