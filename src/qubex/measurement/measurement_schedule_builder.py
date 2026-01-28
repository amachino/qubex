from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from qubex.backend import SAMPLING_PERIOD, ControlParams, Mux, Target
from qubex.backend.quel_instrument_executor import (
    EXTRA_SUM_SECTION_LENGTH,
    WORD_DURATION,
    WORD_LENGTH,
)
from qubex.pulse import Blank, FlatTop, PulseArray, PulseSchedule, RampType
from qubex.typing import TargetMap

from .defaults import (
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_READOUT_RAMPTIME,
)
from .models.capture_schedule import Capture, CaptureSchedule
from .models.measurement_schedule import MeasurementSchedule


@dataclass(frozen=True)
class MeasurementScheduleDefaults:
    readout_duration: float = DEFAULT_READOUT_DURATION
    readout_ramptime: float = DEFAULT_READOUT_RAMPTIME
    readout_pre_margin: float = DEFAULT_READOUT_PRE_MARGIN
    readout_post_margin: float = DEFAULT_READOUT_POST_MARGIN


ReadoutPulseFactory = Callable[..., PulseArray]
PumpPulseFactory = Callable[..., FlatTop]


class MeasurementScheduleBuilder:
    def __init__(
        self,
        *,
        targets: TargetMap[Target],
        mux_dict: Mapping[str, Mux],
        control_params: ControlParams,
        readout_pulse_factory: ReadoutPulseFactory,
        pump_pulse_factory: PumpPulseFactory,
        defaults: MeasurementScheduleDefaults | None = None,
    ) -> None:
        self._targets = targets
        self._mux_dict = mux_dict
        self._control_params = control_params
        self._readout_pulse_factory = readout_pulse_factory
        self._pump_pulse_factory = pump_pulse_factory
        self._defaults = defaults or MeasurementScheduleDefaults()

    def get_readout_targets(self, schedule: PulseSchedule) -> list[str]:
        return [label for label in schedule.labels if self._targets[label].is_read]

    def add_readout_pulses(
        self,
        *,
        schedule: PulseSchedule,
        readout_amplitudes: dict[str, float] | None,
        readout_duration: float | None,
        readout_pre_margin: float | None,
        readout_post_margin: float | None,
        readout_ramptime: float | None,
        readout_drag_coeff: float | None,
        readout_ramp_type: RampType | None,
    ) -> None:
        if readout_amplitudes is None:
            readout_amplitudes = self._control_params.readout_amplitude
        if readout_duration is None:
            readout_duration = self._defaults.readout_duration
        if readout_pre_margin is None:
            readout_pre_margin = self._defaults.readout_pre_margin
        if readout_post_margin is None:
            readout_post_margin = self._defaults.readout_post_margin

        readout_targets = list(
            {
                Target.read_label(label)
                for label in schedule.labels
                if not self._targets[label].is_pump
            }
        )
        with schedule:
            for target in readout_targets:
                schedule.add(
                    target,
                    self._readout_pulse_factory(
                        target=target,
                        duration=readout_duration,
                        amplitude=readout_amplitudes.get(target),
                        ramptime=readout_ramptime,
                        type=readout_ramp_type,
                        drag_coeff=readout_drag_coeff,
                        pre_margin=readout_pre_margin,
                        post_margin=readout_post_margin,
                    ),
                )

    def add_pump_pulses(
        self,
        *,
        schedule: PulseSchedule,
        readout_ranges: dict[str, list[range]],
        readout_pre_margin: float | None,
        readout_ramptime: float | None,
        readout_ramp_type: RampType | None,
    ) -> None:
        if readout_pre_margin is None:
            readout_pre_margin = self._defaults.readout_pre_margin

        with schedule:
            for target, ranges in readout_ranges.items():
                if not ranges:
                    continue
                mux = self._mux_dict[Target.qubit_label(target)]
                for i in range(len(ranges)):
                    current_range = ranges[i]

                    if i == 0:
                        blank_duration = current_range.start * SAMPLING_PERIOD
                    else:
                        prev_range = ranges[i - 1]
                        blank_duration = (
                            current_range.start - prev_range.stop
                        ) * SAMPLING_PERIOD

                    blank_duration -= readout_pre_margin

                    pump_duration = (
                        current_range.stop - current_range.start
                    ) * SAMPLING_PERIOD + readout_pre_margin

                    pump_amplitude = self._control_params.get_pump_amplitude(mux.index)

                    schedule.add(
                        mux.label,
                        PulseArray(
                            [
                                Blank(blank_duration),
                                self._pump_pulse_factory(
                                    target=target,
                                    duration=pump_duration,
                                    amplitude=pump_amplitude,
                                    ramptime=readout_ramptime,
                                    type=readout_ramp_type,
                                ),
                            ]
                        ),
                    )

    def create_capture_schedule(
        self,
        *,
        schedule: PulseSchedule,
        readout_ranges: dict[str, list[range]],
        capture_delays: dict[int, int] | None = None,
    ) -> CaptureSchedule:
        if capture_delays is None:
            capture_delays = self._control_params.capture_delay_word

        captures: list[Capture] = []
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            if ranges[0].start % WORD_LENGTH != 0:
                raise ValueError(
                    "Capture range should start at a multiple of 4 samples "
                    f"({WORD_DURATION} ns)."
                )

            mux = self._mux_dict[Target.qubit_label(target)]
            capture_delay_word = capture_delays.get(mux.index)
            if capture_delay_word is None:
                capture_delay_word = 0
            delay_samples = capture_delay_word * WORD_LENGTH
            delay_time = delay_samples * SAMPLING_PERIOD

            captures.append(
                Capture(
                    channels=[target],
                    start_time=delay_time,
                    duration=EXTRA_SUM_SECTION_LENGTH * SAMPLING_PERIOD,
                )
            )

            for i, current_range in enumerate(ranges):
                capture_range_length = len(current_range)
                if current_range.start % WORD_LENGTH != 0:
                    raise ValueError(
                        "Capture range should start at a multiple of 4 samples "
                        f"({WORD_DURATION} ns)."
                    )
                if capture_range_length % WORD_LENGTH != 0:
                    raise ValueError(
                        "Capture duration should be a multiple of 4 samples "
                        f"({WORD_DURATION} ns)."
                    )

                if i < len(ranges) - 1:
                    next_range = ranges[i + 1]
                    post_blank_length = next_range.start - current_range.stop
                    if post_blank_length < WORD_LENGTH:
                        raise ValueError(
                            "Readout pulses must have at least "
                            f"{WORD_DURATION} ns post-blank time."
                        )
                    if post_blank_length % WORD_LENGTH != 0:
                        raise ValueError(
                            "Post-blank time should be a multiple of 4 samples "
                            f"({WORD_DURATION} ns)."
                        )
                else:
                    last_post_blank_length = schedule.length - current_range.stop
                    if last_post_blank_length < 0:
                        raise ValueError("Invalid capture range length.")

                captures.append(
                    Capture(
                        channels=[target],
                        start_time=(current_range.start + delay_samples)
                        * SAMPLING_PERIOD,
                        duration=capture_range_length * SAMPLING_PERIOD,
                    )
                )

        return CaptureSchedule(captures=captures)

    def build_measurement_schedule(
        self,
        *,
        schedule: PulseSchedule,
        add_last_measurement: bool,
        add_pump_pulses: bool,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        capture_delays: dict[int, int] | None = None,
    ) -> MeasurementSchedule:
        if add_last_measurement:
            self.add_readout_pulses(
                schedule=schedule,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                readout_ramptime=readout_ramptime,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
            )

        readout_targets = self.get_readout_targets(schedule)
        if not readout_targets:
            raise ValueError("No readout targets in the pulse schedule.")

        readout_ranges = schedule.get_pulse_ranges(readout_targets)
        if add_pump_pulses:
            self.add_pump_pulses(
                schedule=schedule,
                readout_ranges=readout_ranges,
                readout_pre_margin=readout_pre_margin,
                readout_ramptime=readout_ramptime,
                readout_ramp_type=readout_ramp_type,
            )

        capture_schedule = self.create_capture_schedule(
            schedule=schedule,
            readout_ranges=readout_ranges,
            capture_delays=capture_delays,
        )
        return MeasurementSchedule(
            pulse_schedule=schedule,
            capture_schedule=capture_schedule,
        )


__all__ = ["MeasurementScheduleBuilder", "MeasurementScheduleDefaults"]
