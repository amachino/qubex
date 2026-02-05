"""Backend-adapter layer for measurement schedule execution."""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from qubex.backend import (
    BLOCK_DURATION,
    EXTRA_SUM_SECTION_LENGTH,
    SAMPLING_PERIOD,
    WORD_DURATION,
    WORD_LENGTH,
    BackendExecutionRequest,
    DeviceController,
    ExperimentSystem,
    QuelExecutionPayload,
    Target,
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


class QuelMeasurementBackendAdapter:
    """QuEL-specific adapter from measurement models to backend request."""

    def __init__(
        self,
        *,
        device_controller: DeviceController,
        experiment_system: ExperimentSystem,
    ) -> None:
        self._device_controller = device_controller
        self._experiment_system = experiment_system

    def validate_schedule(self, schedule: MeasurementSchedule) -> None:
        """Validate QuEL-specific pulse/capture constraints."""
        pulse_schedule = schedule.pulse_schedule
        if not pulse_schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")
        if not self._is_multiple(pulse_schedule.duration, BLOCK_DURATION):
            raise ValueError(
                f"Pulse sequence duration must be a multiple of {BLOCK_DURATION} ns."
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
            if not self._is_multiple(first_capture.start_time, BLOCK_DURATION):
                raise ValueError(
                    "The first capture start time must be a multiple of "
                    f"{BLOCK_DURATION} ns."
                )
            if not self._is_multiple(first_capture.duration, WORD_DURATION):
                raise ValueError(
                    f"Capture duration must be a multiple of {WORD_DURATION} ns."
                )
            workaround_duration = EXTRA_SUM_SECTION_LENGTH * SAMPLING_PERIOD
            if not np.isclose(first_capture.duration, workaround_duration):
                raise ValueError(
                    "The first capture must be the workaround capture with duration "
                    f"{workaround_duration} ns."
                )

            for capture in sorted_captures:
                if not self._is_multiple(capture.start_time, WORD_DURATION):
                    raise ValueError(
                        f"Capture start time must be a multiple of {WORD_DURATION} ns."
                    )
                if not self._is_multiple(capture.duration, WORD_DURATION):
                    raise ValueError(
                        f"Capture duration must be a multiple of {WORD_DURATION} ns."
                    )

            ranges = readout_ranges.get(channel, [])
            if len(sorted_captures) != len(ranges) + 1:
                raise ValueError(
                    f"Capture schedule mismatch for {channel}: expected {len(ranges) + 1} captures."
                )
            for capture, rng in zip(sorted_captures[1:], ranges, strict=True):
                expected_start = rng.start * SAMPLING_PERIOD
                expected_duration = len(rng) * SAMPLING_PERIOD
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
                if gap < WORD_DURATION:
                    raise ValueError(
                        f"Capture post-blank must be at least {WORD_DURATION} ns."
                    )
                if not self._is_multiple(gap, WORD_DURATION):
                    raise ValueError(
                        f"Capture post-blank must be a multiple of {WORD_DURATION} ns."
                    )

            last = sorted_captures[-1]
            tail_blank = pulse_schedule.duration - (last.start_time + last.duration)
            if tail_blank < 0:
                raise ValueError("Capture schedule exceeds pulse schedule duration.")
            if not self._is_multiple(tail_blank, WORD_DURATION):
                raise ValueError(
                    f"Final post-blank must be a multiple of {WORD_DURATION} ns."
                )

    def build_execution_request(
        self, *, schedule: MeasurementSchedule, config: MeasurementConfig
    ) -> BackendExecutionRequest:
        """Build a QuEL backend execution request from measurement inputs."""
        measure_mode = MeasureMode(config.mode)
        interval = (
            math.ceil(
                (schedule.pulse_schedule.duration + config.interval) / BLOCK_DURATION
            )
            * BLOCK_DURATION
        )
        gen_sampled_sequence, cap_sampled_sequence = self._create_sampled_sequences(
            schedule=schedule
        )
        targets = list(gen_sampled_sequence.keys() | cap_sampled_sequence.keys())
        resource_map = self._device_controller.get_resource_map(targets)

        from qubex.backend.sequencer_mod import SequencerMod

        sequencer = SequencerMod(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map=resource_map,  # type: ignore[arg-type]
            interval=interval,
            sysdb=self._device_controller.qubecalib.sysdb,
            driver=self._device_controller.quel1system,
        )
        payload = QuelExecutionPayload(
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
        try:
            from qubecalib import neopulse as pls
        except ImportError as exc:
            raise ModuleNotFoundError(name="qubecalib") from exc

        pulse_schedule = schedule.pulse_schedule
        capture_schedule = schedule.capture_schedule
        capture_delays = self._experiment_system.control_params.capture_delay_word

        readout_targets = list(capture_schedule.channels.keys())
        readout_ranges = pulse_schedule.get_pulse_ranges(readout_targets)

        capture_delay_sample: dict[str, int] = {}
        for target in readout_targets:
            mux = self._experiment_system.get_mux_by_qubit(Target.qubit_label(target))
            capture_delay_word = capture_delays.get(mux.index, 0)
            capture_delay_sample[target] = capture_delay_word * WORD_LENGTH

        sampled_sequences = pulse_schedule.get_sampled_sequences(copy=False)
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            seq = sampled_sequences[target]
            omega = 2 * np.pi * self._experiment_system.get_diff_frequency(target)
            delay = capture_delay_sample[target]
            for rng in ranges:
                offset = (rng.start + delay) * SAMPLING_PERIOD
                seq[rng] *= np.exp(1j * omega * offset)

        gen_sequences: dict[str, Any] = {}
        for target, waveform in sampled_sequences.items():
            if self._experiment_system.get_target(target).sideband != "L":
                waveform = np.conj(waveform)
            gen_sequences[target] = pls.GenSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
                modulation_frequency=self._experiment_system.get_awg_frequency(target),
                sub_sequences=[
                    pls.GenSampledSubSequence(
                        real=np.real(waveform),
                        imag=np.imag(waveform),
                        repeats=1,
                        post_blank=None,
                        original_post_blank=None,
                    )
                ],
            )

        cap_sequences: dict[str, Any] = {}
        for target, captures in capture_schedule.channels.items():
            sorted_captures = sorted(captures, key=lambda c: c.start_time)
            if not sorted_captures:
                continue

            cap_sub_sequence = pls.CapSampledSubSequence(
                capture_slots=[],
                repeats=None,
                prev_blank=capture_delay_sample[target],
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
            )

            for idx, current_capture in enumerate(sorted_captures):
                current_start = round(current_capture.start_time / SAMPLING_PERIOD)
                capture_range_length = round(current_capture.duration / SAMPLING_PERIOD)
                if idx + 1 < len(sorted_captures):
                    next_start = round(
                        sorted_captures[idx + 1].start_time / SAMPLING_PERIOD
                    )
                    post_blank_length = next_start - (
                        current_start + capture_range_length
                    )
                else:
                    post_blank_length = pulse_schedule.length - (
                        current_start + capture_range_length
                    )

                cap_sub_sequence.capture_slots.append(
                    pls.CaptureSlots(
                        duration=capture_range_length,
                        post_blank=post_blank_length,
                        original_duration=None,  # type: ignore[arg-type]
                        original_post_blank=None,  # type: ignore[arg-type]
                    )
                )

            cap_sequences[target] = pls.CapSampledSequence(
                target_name=target,
                repeats=None,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
                modulation_frequency=self._experiment_system.get_awg_frequency(target),
                sub_sequences=[cap_sub_sequence],
            )

        return gen_sequences, cap_sequences

    @staticmethod
    def _is_multiple(value: float, base: float, *, atol: float = 1e-9) -> bool:
        """Return True if value is a near-integer multiple of base."""
        if base == 0:
            return False
        quotient = value / base
        return math.isclose(quotient, round(quotient), abs_tol=atol)
