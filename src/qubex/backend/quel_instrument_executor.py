from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np

from qubex.measurement.models import MeasureData, MeasureMode, MultipleMeasureResult
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.pulse import PulseSchedule
from qubex.typing import TargetMap

from .device_controller import SAMPLING_PERIOD, DeviceController, RawResult
from .experiment_system import ExperimentSystem
from .sequencer_mod import SequencerMod
from .system_manager import SystemManager

logger = logging.getLogger(__name__)

try:
    from qubecalib import Sequencer
    from qubecalib import neopulse as pls
except ImportError as e:
    logger.info(e)

WORD_LENGTH = 4  # samples
WORD_DURATION = WORD_LENGTH * SAMPLING_PERIOD  # ns
BLOCK_LENGTH = WORD_LENGTH * 16  # samples
BLOCK_DURATION = BLOCK_LENGTH * SAMPLING_PERIOD
EXTRA_SUM_SECTION_LENGTH = WORD_LENGTH * 4  # samples
EXTRA_POST_BLANK_LENGTH = WORD_LENGTH  # samples
EXTRA_CAPTURE_LENGTH = EXTRA_SUM_SECTION_LENGTH + EXTRA_POST_BLANK_LENGTH  # samples
EXTRA_CAPTURE_DURATION = EXTRA_CAPTURE_LENGTH * SAMPLING_PERIOD  # ns


class QuelInstrumentExecutor:
    def __init__(
        self,
        *,
        system_manager: SystemManager,
        device_controller: DeviceController,
        experiment_system: ExperimentSystem,
        classifiers: TargetMap,
    ) -> None:
        self._system_manager = system_manager
        self._device_controller = device_controller
        self._experiment_system = experiment_system
        self._classifiers = classifiers

    def execute(
        self,
        *,
        schedule: PulseSchedule,
        capture_schedule: CaptureSchedule,
        measure_mode: MeasureMode,
        shots: int,
        interval: float,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool = False,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> MultipleMeasureResult:
        sequencer = self._create_sequencer(
            schedule=schedule,
            capture_schedule=capture_schedule,
            interval=interval,
        )
        backend_result = self._device_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=shots,
            integral_mode=measure_mode.integral_mode,
            dsp_demodulation=enable_dsp_demodulation,
            enable_sum=enable_dsp_sum,
            enable_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
        result = self._create_multiple_measure_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=shots,
        )
        self._save_if_needed(result)
        return result

    def pad_schedule_for_capture(self, schedule: PulseSchedule) -> None:
        schedule.pad(
            total_duration=schedule.duration + EXTRA_CAPTURE_DURATION,
            pad_side="left",
        )
        sequence_duration = (
            math.ceil(schedule.duration / BLOCK_DURATION + 1) * BLOCK_DURATION
        )
        schedule.pad(
            total_duration=sequence_duration,
            pad_side="right",
        )

    def _save_if_needed(self, result: MultipleMeasureResult) -> None:
        rawdata_dir = self._system_manager.rawdata_dir
        if rawdata_dir is not None:
            result.save(data_dir=rawdata_dir)

    def _create_sequencer(
        self,
        *,
        schedule: PulseSchedule,
        capture_schedule: CaptureSchedule,
        interval: float,
    ) -> Sequencer:
        if interval is None:
            raise ValueError("Interval must be provided.")

        gen_sequences = self._create_gen_sequences(
            schedule=schedule,
            capture_schedule=capture_schedule,
        )
        cap_sequences = self._create_cap_sequences(
            schedule=schedule,
            capture_schedule=capture_schedule,
        )

        backend_interval = (
            math.ceil((schedule.duration + interval) / BLOCK_DURATION) * BLOCK_DURATION
        )

        targets = list(gen_sequences.keys() | cap_sequences.keys())
        resource_map = self._device_controller.get_resource_map(targets)

        return SequencerMod(
            gen_sampled_sequence=gen_sequences,
            cap_sampled_sequence=cap_sequences,
            resource_map=resource_map,  # type: ignore
            interval=backend_interval,
            sysdb=self._device_controller.qubecalib.sysdb,
            driver=self._device_controller.quel1system,
        )

    def _create_gen_sequences(
        self,
        *,
        schedule: PulseSchedule,
        capture_schedule: CaptureSchedule,
    ) -> dict[str, pls.GenSampledSequence]:
        sampled_sequences = schedule.get_sampled_sequences(copy=False)
        captures_by_target = self._group_captures_by_target(capture_schedule)

        for target, captures in captures_by_target.items():
            if target not in sampled_sequences:
                continue
            seq = sampled_sequences[target]
            omega = 2 * np.pi * self._experiment_system.get_diff_frequency(target)
            for capture in captures:
                start = self._time_to_samples(capture.start_time)
                duration = self._time_to_samples(capture.duration)
                if duration == 0:
                    continue
                offset = capture.start_time
                seq[start : start + duration] *= np.exp(1j * omega * offset)

        gen_sequences: dict[str, pls.GenSampledSequence] = {}
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
        return gen_sequences

    def _create_cap_sequences(
        self,
        *,
        schedule: PulseSchedule,
        capture_schedule: CaptureSchedule,
    ) -> dict[str, pls.CapSampledSequence]:
        cap_sequences: dict[str, pls.CapSampledSequence] = {}
        captures_by_target = self._group_captures_by_target(capture_schedule)

        for target, captures in captures_by_target.items():
            if not captures:
                continue

            captures = sorted(captures, key=lambda c: c.start_time)
            first_start = self._time_to_samples(captures[0].start_time)

            cap_sub_sequence = pls.CapSampledSubSequence(
                capture_slots=[],
                repeats=None,
                prev_blank=first_start,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
            )

            for idx, capture in enumerate(captures):
                start = self._time_to_samples(capture.start_time)
                duration = self._time_to_samples(capture.duration)
                if idx > 0:
                    if start % WORD_LENGTH != 0:
                        raise ValueError(
                            "Capture range should start at a multiple of 4 samples (8 ns)."
                        )
                    if duration % WORD_LENGTH != 0:
                        raise ValueError(
                            "Capture duration should be a multiple of 4 samples (8 ns)."
                        )
                if idx < len(captures) - 1:
                    next_start = self._time_to_samples(captures[idx + 1].start_time)
                    post_blank = next_start - (start + duration)
                    if idx > 0:
                        if post_blank < WORD_LENGTH:
                            raise ValueError(
                                "Readout pulses must have at least 8 ns post-blank time."
                            )
                        if post_blank % WORD_LENGTH != 0:
                            raise ValueError(
                                "Post-blank time should be a multiple of 4 samples (8 ns)."
                            )
                else:
                    post_blank = schedule.length - (start + duration)

                cap_sub_sequence.capture_slots.append(
                    pls.CaptureSlots(
                        duration=duration,
                        post_blank=post_blank,
                        original_duration=None,  # type: ignore
                        original_post_blank=None,
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
                sub_sequences=[
                    cap_sub_sequence,
                ],
            )

        return cap_sequences

    def _group_captures_by_target(
        self,
        capture_schedule: CaptureSchedule,
    ) -> dict[str, list]:
        by_target: dict[str, list] = defaultdict(list)
        for capture in capture_schedule.captures:
            for channel in capture.channels:
                by_target[channel].append(capture)
        return by_target

    @staticmethod
    def _time_to_samples(duration: float) -> int:
        frac = duration / SAMPLING_PERIOD
        N = round(frac)
        if abs(frac - N) > 1e-9:
            raise ValueError(
                f"Duration must be a multiple of the sampling period ({SAMPLING_PERIOD} ns)."
            )
        return N

    def _create_multiple_measure_result(
        self,
        *,
        backend_result: RawResult,
        measure_mode: MeasureMode,
        shots: int,
    ) -> MultipleMeasureResult:
        label_slice = slice(1, None)  # remove the resonator prefix "R"
        norm_factor = 2 ** (-32)  # normalization factor for 32-bit data

        iq_data: dict[str, list[np.ndarray]] = {}
        for target, iqs in sorted(backend_result.data.items()):
            sideband = self._experiment_system.get_target(target).sideband
            if sideband == "L":
                iq_data[target] = [np.conjugate(iq) for iq in iqs]
            else:
                iq_data[target] = iqs

        measure_data: dict[str, list[MeasureData]] = {}
        if measure_mode == MeasureMode.SINGLE:
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                measure_data[qubit] = []
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(
                        MeasureData(
                            target=qubit,
                            mode=measure_mode,
                            raw=iq * norm_factor,
                            classifier=self._classifiers.get(qubit),
                        )
                    )
        elif measure_mode == MeasureMode.AVG:
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                measure_data[qubit] = []
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(
                        MeasureData(
                            target=qubit,
                            mode=measure_mode,
                            raw=iq.squeeze() * norm_factor / shots,
                        )
                    )
        else:
            raise ValueError(f"Invalid measure mode: {measure_mode}")

        return MultipleMeasureResult(
            mode=measure_mode,
            data=measure_data,
            config=self._device_controller.box_config,
        )
