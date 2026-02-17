"""Backend-adapter layer for measurement schedule execution."""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from qubex.backend import (
    BackendExecutionRequest,
    ExperimentSystem,
    Target,
)
from qubex.backend.quel1 import (
    Quel1BackendController,
    Quel1ExecutionPayload,
)
from qubex.measurement.models.measure_result import MeasureMode
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule

from .measurement_constraint_profile import MeasurementConstraintProfile


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
            constraint_profile = MeasurementConstraintProfile.strict_quel1()
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
        if profile.enforce_block_alignment and block_duration is not None:
            interval = int(
                math.ceil(
                    (schedule.pulse_schedule.duration + config.interval)
                    / block_duration
                )
                * block_duration
            )
        else:
            interval = math.ceil(schedule.pulse_schedule.duration + config.interval)
        gen_sampled_sequence, cap_sampled_sequence = self._create_sampled_sequences(
            schedule=schedule
        )
        targets = list(
            dict.fromkeys([*gen_sampled_sequence.keys(), *cap_sampled_sequence.keys()])
        )
        resource_map = self._backend_controller.get_resource_map(targets)

        sequencer = self._backend_controller.create_quel1_sequencer(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map=resource_map,
            interval=interval,
        )
        payload = Quel1ExecutionPayload(
            sequencer=sequencer,
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
            mux = self._experiment_system.get_mux_by_qubit(Target.qubit_label(target))
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
