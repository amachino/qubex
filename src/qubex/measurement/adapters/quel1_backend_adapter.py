"""QuEL-1 measurement backend adapter implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from qubex.backend import (
    BackendExecutionRequest,
)
from qubex.backend.quel1 import (
    Quel1BackendController,
    Quel1BackendExecutionResult,
    Quel1ExecutionPayload,
)
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.models.capture_data import CaptureData
from qubex.measurement.models.measure_result import MeasureMode
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions
from qubex.system import ExperimentSystem, TargetRegistry

if TYPE_CHECKING:
    import numpy.typing as npt

    from qubex.backend.quel1.compat.qubecalib_protocols import (
        CapSampledSequenceProtocol,
        GenSampledSequenceProtocol,
    )


def _as_read_only_array(data: object) -> np.ndarray:
    """Return read-only NumPy array view for capture payloads."""
    array = np.asarray(data).view()
    array.setflags(write=False)
    return array


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

    def validate_schedule(self, schedule: MeasurementSchedule) -> None:
        """Validate QuEL-1 specific pulse/capture constraints."""
        profile = self._constraint_profile
        block_duration = profile.block_duration_ns
        word_duration = profile.word_duration_ns
        sampling_period = profile.sampling_period_ns
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

        range_targets = self._resolve_pulse_range_targets(
            pulse_schedule=pulse_schedule,
            capture_targets=list(channel_captures),
        )
        readout_ranges = pulse_schedule.get_pulse_ranges(range_targets)
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

            if self._is_entire_schedule_capture(
                captures=sorted_captures,
                schedule_duration=pulse_schedule.duration,
            ):
                continue

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
                expected_start = rng.start * sampling_period
                expected_duration = len(rng) * sampling_period
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
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> BackendExecutionRequest:
        """Build a QuEL backend execution request from measurement inputs."""
        profile = self._constraint_profile
        block_duration = profile.block_duration_ns
        measure_mode = MeasureMode.AVG if config.shot_averaging else MeasureMode.SINGLE
        base_duration = schedule.pulse_schedule.duration
        if profile.enforce_block_alignment and block_duration is not None:
            interval_ns = int(
                math.ceil((base_duration + config.shot_interval) / block_duration)
                * block_duration
            )
            # Compatibility guard:
            # Keep one extra block for interval<=0 because skew.yaml wait
            # (and port_wait) is applied later as leading waveform padding in
            # qubecalib; without this margin, converter-side packing/alignment
            # can make trailing chunk blank words negative.
            # Remove this workaround once qubecalib compatibility is no longer
            # required in the QuEL-1 measurement path.
            if config.shot_interval <= 0:
                minimum_interval = int(
                    math.ceil((base_duration + block_duration) / block_duration)
                    * block_duration
                )
                interval_ns = max(interval_ns, minimum_interval)
        else:
            interval_ns = math.ceil(base_duration + config.shot_interval)
        gen_sampled_sequence, cap_sampled_sequence = self._create_sampled_sequences(
            schedule=schedule
        )
        targets = list(
            dict.fromkeys([*gen_sampled_sequence.keys(), *cap_sampled_sequence.keys()])
        )
        resource_lookup_target_by_target = {
            target: self._resolve_resource_lookup_target(target) for target in targets
        }
        resource_lookup_targets = list(
            dict.fromkeys(resource_lookup_target_by_target.values())
        )
        resource_map_by_lookup_target = self._backend_controller.get_resource_map(
            resource_lookup_targets
        )
        resource_map = {
            target: resource_map_by_lookup_target[lookup_target]
            for target, lookup_target in resource_lookup_target_by_target.items()
        }
        dsp_demodulation = (
            True
            if quel1_options is None or quel1_options.demodulation is None
            else quel1_options.demodulation
        )

        payload = Quel1ExecutionPayload(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map=resource_map,
            interval_ns=interval_ns,
            repeats=config.n_shots,
            integral_mode=measure_mode.integral_mode,
            dsp_demodulation=dsp_demodulation,
            enable_sum=config.time_integration,
            enable_classification=config.state_classification,
            line_param0=(
                None
                if quel1_options is None
                else quel1_options.classification_line_param0
            ),
            line_param1=(
                None
                if quel1_options is None
                else quel1_options.classification_line_param1
            ),
        )
        return BackendExecutionRequest(
            payload=payload,
        )

    def build_measurement_result(
        self,
        *,
        backend_result: object,
        measurement_config: MeasurementConfig,
        device_config: dict,
        sampling_period: float,
    ) -> MeasurementResult:
        """Build canonical result from a QuEL-1 backend result payload."""
        if not isinstance(backend_result, Quel1BackendExecutionResult):
            raise TypeError(
                "QuEL-1 adapter expects backend_result to be `Quel1BackendExecutionResult`."
            )

        shot_averaging = measurement_config.shot_averaging
        skip_extra_capture = self._constraint_profile.require_workaround_capture
        norm_factor = 2 ** (-32)  # normalization factor for 32-bit data
        target_registry = getattr(self._experiment_system, "target_registry", None)

        iq_data: dict[str, list[npt.ArrayLike]] = {}
        for target, iqs in sorted(backend_result.data.items()):
            sideband = "U"
            try:
                sideband_candidate = self._experiment_system.get_target(target).sideband
                if sideband_candidate in ("U", "L"):
                    sideband = sideband_candidate
            except KeyError:
                sideband = "U"
            if sideband == "L":
                iq_data[target] = [np.conjugate(iq) for iq in iqs]
            else:
                iq_data[target] = iqs

        measure_data: dict[str, list[CaptureData]] = {}
        if not shot_averaging:
            for target, iqs in iq_data.items():
                if target_registry is not None and hasattr(
                    target_registry,
                    "measurement_output_label",
                ):
                    qubit = str(target_registry.measurement_output_label(target))
                elif target.startswith("R"):
                    qubit = target[1:]
                else:
                    qubit = target
                values: list[CaptureData] = []
                for index, iq in enumerate(iqs):
                    if skip_extra_capture and index == 0:
                        # skip the first extra capture
                        continue
                    values.append(
                        CaptureData.from_primary_data(
                            target=qubit,
                            data=_as_read_only_array(
                                np.asarray(iq, dtype=np.complex128) * norm_factor
                            ),
                            config=measurement_config,
                            sampling_period=sampling_period,
                        )
                    )
                measure_data[qubit] = values
        else:
            for target, iqs in iq_data.items():
                if target_registry is not None and hasattr(
                    target_registry,
                    "measurement_output_label",
                ):
                    qubit = str(target_registry.measurement_output_label(target))
                elif target.startswith("R"):
                    qubit = target[1:]
                else:
                    qubit = target
                values: list[CaptureData] = []
                for index, iq in enumerate(iqs):
                    if skip_extra_capture and index == 0:
                        # skip the first extra capture
                        continue
                    values.append(
                        CaptureData.from_primary_data(
                            target=qubit,
                            data=_as_read_only_array(
                                np.asarray(iq, dtype=np.complex128).squeeze()
                                * norm_factor
                                / measurement_config.n_shots
                            ),
                            config=measurement_config,
                            sampling_period=sampling_period,
                        )
                    )
                measure_data[qubit] = values

        return MeasurementResult(
            data=measure_data,
            device_config=device_config,
            measurement_config=measurement_config,
        )

    def _resolve_resource_lookup_target(self, target: str) -> str:
        """Resolve target name used to look up QuEL system resource-map entries."""
        try:
            _ = self._experiment_system.get_target(target)
        except AttributeError:
            pass
        except KeyError:
            pass
        else:
            return target
        try:
            _ = self._experiment_system.get_cap_target(target)
        except AttributeError:
            pass
        except KeyError:
            pass
        else:
            return target

        try:
            read_in_targets = self._experiment_system.read_in_targets
        except AttributeError:
            read_in_targets = ()

        for cap_target in read_in_targets:
            try:
                port_id = cap_target.channel.port.id
            except AttributeError:
                continue
            if port_id == target:
                return str(cap_target.label)
        return target

    def _create_sampled_sequences(
        self,
        *,
        schedule: MeasurementSchedule,
    ) -> tuple[
        dict[str, GenSampledSequenceProtocol],
        dict[str, CapSampledSequenceProtocol],
    ]:
        """Create sampled sequences from measurement schedule."""
        pulse_schedule = schedule.pulse_schedule
        capture_schedule = schedule.capture_schedule
        capture_targets = list(capture_schedule.channels.keys())
        pulse_range_targets = self._resolve_pulse_range_targets(
            pulse_schedule=pulse_schedule,
            capture_targets=capture_targets,
        )
        readout_ranges = pulse_schedule.get_pulse_ranges(pulse_range_targets)

        capture_delay_sample: dict[str, int] = {}
        word_length = self._constraint_profile.word_length_samples
        sampling_period = self._constraint_profile.sampling_period_ns
        if word_length is None:
            raise ValueError(
                "word_length_samples is required for backend execution request."
            )
        capture_is_entire_schedule: dict[str, bool] = {}
        for target in capture_targets:
            captures = sorted(
                capture_schedule.channels.get(target, []),
                key=lambda c: c.start_time,
            )
            is_entire_schedule = self._is_entire_schedule_capture(
                captures=captures,
                schedule_duration=pulse_schedule.duration,
            )
            capture_is_entire_schedule[target] = is_entire_schedule
            if is_entire_schedule:
                # Full-span capture windows are already defined in absolute
                # schedule coordinates, so capture-delay offsets are not applied.
                capture_delay_sample[target] = 0
            else:
                capture_delay_sample[target] = self._resolve_capture_delay_samples(
                    target=target,
                    word_length=word_length,
                )

        sampled_sequences = pulse_schedule.get_sampled_sequences()
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            if target not in sampled_sequences:
                continue
            seq = sampled_sequences[target]
            schedule_frequency = self._resolve_schedule_frequency(
                pulse_schedule=pulse_schedule,
                target=target,
            )
            diff_frequency = self._resolve_diff_frequency(
                target=target,
                schedule_frequency=schedule_frequency,
            )
            omega = 2 * np.pi * diff_frequency
            delay = capture_delay_sample[target]
            for rng in ranges:
                offset = (rng.start + delay) * sampling_period
                seq[rng] *= np.exp(1j * omega * offset)

        gen_sequences: dict[str, GenSampledSequenceProtocol] = {}
        for target, waveform in sampled_sequences.items():
            sideband = "U"
            try:
                sideband_candidate = self._experiment_system.get_target(target).sideband
                if sideband_candidate in ("U", "L"):
                    sideband = sideband_candidate
            except KeyError:
                sideband = "U"
            if sideband != "L":
                waveform = np.conj(waveform)
            schedule_frequency = self._resolve_schedule_frequency(
                pulse_schedule=pulse_schedule,
                target=target,
            )
            gen_sequences[target] = self._create_gen_sampled_sequence(
                target_name=target,
                modulation_frequency=self._resolve_modulation_frequency(
                    target=target,
                    schedule_frequency=schedule_frequency,
                ),
                real=np.real(waveform),
                imag=np.imag(waveform),
            )

        cap_sequences: dict[str, CapSampledSequenceProtocol] = {}
        for target, captures in capture_schedule.channels.items():
            sorted_captures = sorted(captures, key=lambda c: c.start_time)
            if not sorted_captures:
                continue
            schedule_frequency = self._resolve_schedule_frequency(
                pulse_schedule=pulse_schedule,
                target=target,
            )

            capture_slots: list[tuple[int, int]] = []
            for idx, current_capture in enumerate(sorted_captures):
                current_start = round(current_capture.start_time / sampling_period)
                capture_range_length = round(current_capture.duration / sampling_period)
                if idx + 1 < len(sorted_captures):
                    next_start = round(
                        sorted_captures[idx + 1].start_time / sampling_period
                    )
                    post_blank_length = next_start - (
                        current_start + capture_range_length
                    )
                else:
                    post_blank_length = pulse_schedule.length - (
                        current_start + capture_range_length
                    )
                capture_slots.append((capture_range_length, post_blank_length))

            cap_sequences[target] = self._create_cap_sampled_sequence(
                target_name=target,
                modulation_frequency=self._resolve_modulation_frequency(
                    target=target,
                    schedule_frequency=schedule_frequency,
                ),
                capture_delay=(
                    0
                    if capture_is_entire_schedule.get(target, False)
                    else capture_delay_sample.get(target, 0)
                ),
                capture_slots=capture_slots,
            )

        return gen_sequences, cap_sequences

    def _resolve_capture_delay_samples(
        self,
        *,
        target: str,
        word_length: int,
    ) -> int:
        """Resolve capture delay in samples for one capture target."""
        capture_delays = self._experiment_system.control_params.capture_delay_word
        target_registry = getattr(self._experiment_system, "target_registry", None)
        fallback_registry = TargetRegistry()
        resolve_qubit_label = getattr(
            self._experiment_system,
            "resolve_qubit_label",
            None,
        )

        qubit_label: str | None = None
        if callable(resolve_qubit_label):
            try:
                qubit_label = str(resolve_qubit_label(target))
            except ValueError:
                qubit_label = None
        elif target_registry is not None and hasattr(
            target_registry,
            "resolve_qubit_label",
        ):
            resolver = target_registry.resolve_qubit_label
            try:
                qubit_label = str(resolver(target, allow_legacy=True))
            except TypeError:
                try:
                    qubit_label = str(resolver(target))
                except ValueError:
                    qubit_label = None
            except ValueError:
                qubit_label = None
        else:
            try:
                qubit_label = str(
                    fallback_registry.resolve_qubit_label(
                        target,
                        allow_legacy=True,
                    )
                )
            except ValueError:
                qubit_label = None

        if qubit_label is not None:
            try:
                mux = self._experiment_system.get_mux_by_qubit(qubit_label)
            except (KeyError, ValueError):
                mux = None
            if mux is not None:
                capture_delay_word = capture_delays.get(mux.index, 0)
                return capture_delay_word * word_length

        control_system = getattr(self._experiment_system, "control_system", None)
        if control_system is not None and hasattr(control_system, "get_port_by_id"):
            try:
                port = control_system.get_port_by_id(target)
            except KeyError:
                return 0
            channels = getattr(port, "channels", ())
            if channels:
                ndelay = getattr(channels[0], "ndelay", None)
                if isinstance(ndelay, int):
                    return ndelay * word_length
        return 0

    def _resolve_modulation_frequency(
        self,
        *,
        target: str,
        schedule_frequency: float | None = None,
    ) -> float:
        """Resolve modulation frequency for generation or capture sequence."""
        if schedule_frequency is not None:
            nco_frequency = self._resolve_nco_frequency(target=target)
            if nco_frequency is None:
                return schedule_frequency
            sideband: str | None = None
            try:
                sideband = self._experiment_system.get_target(target).sideband
            except KeyError:
                sideband = None
            if sideband == "L":
                return nco_frequency - schedule_frequency
            return schedule_frequency - nco_frequency

        try:
            return float(self._experiment_system.get_awg_frequency(target))
        except (KeyError, ValueError):
            pass

        nco_frequency = self._resolve_nco_frequency(target=target)
        if nco_frequency is not None:
            return nco_frequency
        return 0.0

    def _resolve_diff_frequency(
        self,
        *,
        target: str,
        schedule_frequency: float | None = None,
    ) -> float:
        """Resolve difference frequency for readout phase compensation."""
        if schedule_frequency is not None:
            nco_frequency = self._resolve_nco_frequency(target=target)
            if nco_frequency is None:
                return schedule_frequency
            return schedule_frequency - nco_frequency

        try:
            return float(self._experiment_system.get_diff_frequency(target))
        except (KeyError, ValueError):
            return 0.0

    def _resolve_nco_frequency(
        self,
        *,
        target: str,
    ) -> float | None:
        """Resolve NCO frequency from experiment system metadata."""
        try:
            return float(self._experiment_system.get_nco_frequency(target))
        except (AttributeError, KeyError, ValueError):
            pass

        control_system = getattr(self._experiment_system, "control_system", None)
        if control_system is not None and hasattr(control_system, "get_port_by_id"):
            try:
                port = control_system.get_port_by_id(target)
            except KeyError:
                return None
            channels = getattr(port, "channels", ())
            if channels:
                fnco_freq = getattr(channels[0], "fnco_freq", None)
                if isinstance(fnco_freq, (int, float)):
                    return float(fnco_freq)
        return None

    @staticmethod
    def _resolve_schedule_frequency(
        *,
        pulse_schedule: object,
        target: str,
    ) -> float | None:
        """Resolve channel frequency metadata from pulse schedule when available."""
        get_frequency = getattr(pulse_schedule, "get_frequency", None)
        if not callable(get_frequency):
            return None
        try:
            frequency = get_frequency(target)
        except KeyError:
            return None
        if isinstance(frequency, (int, float)):
            return float(frequency)
        return None

    def _create_gen_sampled_sequence(
        self,
        *,
        target_name: str,
        real: npt.ArrayLike,
        imag: npt.ArrayLike,
        modulation_frequency: float,
    ) -> GenSampledSequenceProtocol:
        """Create one generation sampled sequence for QuEL-1 payload."""
        driver = self._backend_controller.driver
        return driver.GenSampledSequence(
            target_name=target_name,
            prev_blank=0,
            post_blank=None,
            original_prev_blank=0,
            original_post_blank=None,
            modulation_frequency=modulation_frequency,
            sub_sequences=[
                driver.GenSampledSubSequence(
                    real=real,
                    imag=imag,
                    repeats=1,
                    post_blank=None,
                    original_post_blank=None,
                )
            ],
        )

    def _create_cap_sampled_sequence(
        self,
        *,
        target_name: str,
        modulation_frequency: float,
        capture_delay: int,
        capture_slots: list[tuple[int, int]],
    ) -> CapSampledSequenceProtocol:
        """Create one capture sampled sequence for QuEL-1 payload."""
        driver = self._backend_controller.driver
        cap_sub_sequence = driver.CapSampledSubSequence(
            capture_slots=[],
            repeats=None,
            prev_blank=capture_delay,
            post_blank=None,
            original_prev_blank=0,
            original_post_blank=None,
        )
        for duration, post_blank in capture_slots:
            cap_sub_sequence.capture_slots.append(
                driver.CaptureSlots(
                    duration=duration,
                    post_blank=post_blank,
                    original_duration=None,  # type: ignore[arg-type]
                    original_post_blank=None,  # type: ignore[arg-type]
                )
            )
        return driver.CapSampledSequence(
            target_name=target_name,
            repeats=None,
            prev_blank=0,
            post_blank=None,
            original_prev_blank=0,
            original_post_blank=None,
            modulation_frequency=modulation_frequency,
            sub_sequences=[cap_sub_sequence],
        )

    def _is_entire_schedule_capture(
        self,
        *,
        captures: list,
        schedule_duration: float,
    ) -> bool:
        """Return whether captures represent one full-span capture over schedule."""
        profile = self._constraint_profile
        if profile.require_workaround_capture:
            if len(captures) != 2:
                return False
            first_capture, second_capture = captures
            capture_start = profile.extra_capture_duration_ns
            capture_duration = max(0.0, schedule_duration - capture_start)
            return bool(
                np.isclose(first_capture.start_time, 0.0)
                and np.isclose(
                    first_capture.duration,
                    profile.workaround_capture_duration_ns,
                )
                and np.isclose(second_capture.start_time, capture_start)
                and np.isclose(second_capture.duration, capture_duration)
            )
        if len(captures) != 1:
            return False
        capture = captures[0]
        return bool(
            np.isclose(capture.start_time, 0.0)
            and np.isclose(capture.duration, schedule_duration)
        )

    @staticmethod
    def _resolve_pulse_range_targets(
        *,
        pulse_schedule: object,
        capture_targets: list[str],
    ) -> list[str]:
        """Resolve capture targets that also exist in pulse-range labels."""
        pulse_labels = getattr(pulse_schedule, "labels", None)
        label_set: set[str] = set()
        if isinstance(pulse_labels, list):
            label_set = {str(label) for label in pulse_labels}
        else:
            get_pulse_ranges = getattr(pulse_schedule, "get_pulse_ranges", None)
            if callable(get_pulse_ranges):
                try:
                    ranges = get_pulse_ranges()
                except TypeError:
                    ranges = None
                if isinstance(ranges, dict):
                    label_set = {str(label) for label in ranges}
        return [target for target in capture_targets if target in label_set]

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
