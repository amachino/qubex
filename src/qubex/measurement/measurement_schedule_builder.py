"""
Build measurement and capture schedules from pulse schedules.

This module provides utilities for augmenting a PulseSchedule with readout
and (optionally) pump pulses, and for creating a corresponding CaptureSchedule.

Timing in this module is expressed in the same units as SAMPLING_PERIOD
(i.e., sample count multiplied by SAMPLING_PERIOD yields a time value).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from qubex.backend import SAMPLING_PERIOD, ControlParams, Mux, Target
from qubex.backend.quel_device_executor import (
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
    """
    Default parameters used when building measurement schedules.

    Attributes
    ----------
    readout_duration
        Readout pulse duration.
    readout_ramptime
        Rise/fall time for ramped readout envelopes.
    readout_pre_margin
        Time margin inserted before the readout pulse.
    readout_post_margin
        Time margin inserted after the readout pulse.

    Notes
    -----
    These defaults are used only when the corresponding argument is passed as
    `None` to schedule-building methods.
    """

    readout_duration: float = DEFAULT_READOUT_DURATION
    readout_ramptime: float = DEFAULT_READOUT_RAMPTIME
    readout_pre_margin: float = DEFAULT_READOUT_PRE_MARGIN
    readout_post_margin: float = DEFAULT_READOUT_POST_MARGIN


ReadoutPulseFactory = Callable[..., PulseArray]
PumpPulseFactory = Callable[..., FlatTop]
ScheduleAdjuster = Callable[[PulseSchedule], None]


class MeasurementScheduleBuilder:
    """Builder for measurement and capture schedules."""

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
        """
        Initialize a schedule builder.

        Parameters
        ----------
        targets
            Mapping from schedule label to `Target` metadata.
            Used to identify readout and pump channels.
        mux_dict
            Mapping from qubit label to `Mux`.
            Used to map a target/qubit to its mux index for pump/capture settings.
        control_params
            Control parameters providing default amplitudes and capture delays.
        readout_pulse_factory
            Factory used to create a readout `PulseArray`.
        pump_pulse_factory
            Factory used to create a pump `FlatTop` pulse.
        defaults
            Optional override of default readout timing parameters.
        """
        self._targets = targets
        self._mux_dict = mux_dict
        self._control_params = control_params
        self._readout_pulse_factory = readout_pulse_factory
        self._pump_pulse_factory = pump_pulse_factory
        self._defaults = defaults or MeasurementScheduleDefaults()

    def get_readout_targets(self, schedule: PulseSchedule) -> list[str]:
        """
        Return schedule labels that correspond to readout targets.

        Parameters
        ----------
        schedule
            Pulse schedule whose labels are inspected.

        Returns
        -------
        list[str]
            Labels in `schedule.labels` for which the corresponding
            `Target` is a readout target.
        """
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
        """
        Add readout pulses to a pulse schedule in-place.

        Parameters
        ----------
        schedule
            Schedule to be modified.
        readout_amplitudes
            Optional per-readout-target amplitude mapping. When `None`,
            `ControlParams.readout_amplitude` is used.
        readout_duration
            Readout duration. When `None`, the builder default is used.
        readout_pre_margin
            Time margin inserted before each readout pulse. When `None`, the
            builder default is used.
        readout_post_margin
            Time margin inserted after each readout pulse. When `None`, the
            builder default is used.
        readout_ramptime
            Ramp time forwarded to the readout pulse factory.
        readout_drag_coeff
            DRAG coefficient forwarded to the readout pulse factory.
        readout_ramp_type
            Ramp type forwarded to the readout pulse factory.

        Notes
        -----
        This method only adds readout pulses for channels present in
        `schedule.labels` that are not pump targets.
        """
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
        """
        Add pump pulses aligned to readout ranges in-place.

        Parameters
        ----------
        schedule
            Schedule to be modified.
        readout_ranges
            Mapping from readout target label to a list of sample-index ranges
            where readout pulses occur.
        readout_pre_margin
            Pre-margin used to extend the pump window. When `None`, the builder
            default is used.
        readout_ramptime
            Ramp time forwarded to the pump pulse factory.
        readout_ramp_type
            Ramp type forwarded to the pump pulse factory.

        Notes
        -----
        Pump amplitude is determined by `ControlParams.get_pump_amplitude`
        using the mux index resolved from `readout_ranges` keys.
        """
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
        """
        Create a capture schedule from readout pulse ranges.

        Parameters
        ----------
        schedule
            Pulse schedule used for overall timing and for validating the last
            range length.
        readout_ranges
            Mapping from readout target label to a list of sample-index ranges
            representing readout pulses.
        capture_delays
            Optional mapping from mux index to capture delay, expressed in
            words. When `None`,
            `ControlParams.capture_delay_word` is used.

        Returns
        -------
        CaptureSchedule
            Capture schedule containing one initial extra-sum capture and one
            capture per readout range for each target.

        Raises
        ------
        ValueError
            If any capture range start or duration is not aligned to
            `WORD_LENGTH` samples,
            if readout pulses do not provide sufficient post-blank time between
            consecutive ranges, or if the last range extends beyond the schedule.
        """
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
        schedule_adjuster: ScheduleAdjuster | None = None,
    ) -> MeasurementSchedule:
        """
        Build a `MeasurementSchedule`.

        Parameters
        ----------
        schedule
            Base schedule to use and (optionally) modify.
        add_last_measurement
            If `True`, readout pulses are appended to the schedule.
        add_pump_pulses
            If `True`, pump pulses are added based on detected readout ranges.
        readout_amplitudes, readout_duration, readout_pre_margin, readout_post_margin
            Optional overrides for readout pulse construction. Any value passed
            as `None` falls back to builder defaults.
        readout_ramptime, readout_drag_coeff, readout_ramp_type
            Parameters forwarded to the readout (and/or pump) pulse factories.
        capture_delays
            Optional capture delay mapping; see `create_capture_schedule`.
        schedule_adjuster
            Optional callback to adjust the pulse schedule before extracting
            readout ranges and generating capture settings.

        Returns
        -------
        MeasurementSchedule
            Measurement schedule containing the (possibly augmented) pulse
            schedule and the derived capture schedule.

        Raises
        ------
        ValueError
            If there are no readout targets in the pulse schedule, or if capture
            schedule creation fails due to invalid alignment/spacing constraints.
        """
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

        if schedule_adjuster is not None:
            schedule_adjuster(schedule)

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
