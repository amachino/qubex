"""Builder for measurement pulse schedules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from qubex.backend import SAMPLING_PERIOD as _SAMPLING_PERIOD
from qubex.backend import ControlParams, Mux, Target
from qubex.pulse import PulseSchedule, RampType

from .measurement_defaults import (
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
)
from .measurement_pulse_factory import MeasurementPulseFactory
from .models.capture_schedule import Capture, CaptureSchedule
from .models.measurement_schedule import MeasurementSchedule

WORD_LENGTH: Final = 4  # samples
WORD_DURATION: Final = WORD_LENGTH * _SAMPLING_PERIOD  # ns
BLOCK_LENGTH: Final = WORD_LENGTH * 16  # samples
BLOCK_DURATION: Final = BLOCK_LENGTH * _SAMPLING_PERIOD  # ns

EXTRA_SUM_SECTION_LENGTH = WORD_LENGTH * 4  # samples
EXTRA_POST_BLANK_LENGTH = WORD_LENGTH  # samples
EXTRA_CAPTURE_LENGTH = EXTRA_SUM_SECTION_LENGTH + EXTRA_POST_BLANK_LENGTH  # samples
EXTRA_CAPTURE_DURATION = EXTRA_CAPTURE_LENGTH * _SAMPLING_PERIOD  # ns


class MeasurementScheduleBuilder:
    """Build hardware-ready measurement schedules for execution."""

    def __init__(
        self,
        *,
        control_params: ControlParams,
        pulse_factory: MeasurementPulseFactory,
        targets: Mapping[str, Target],
        mux_dict: Mapping[str, Mux],
    ) -> None:
        self._control_params = control_params
        self._pulse_factory = pulse_factory
        self._targets = targets
        self._mux_dict = mux_dict

    def build(
        self,
        *,
        schedule: PulseSchedule,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_drag_coeff: float | None = None,
        add_last_measurement: bool = False,
        add_pump_pulses: bool = False,
        plot: bool = False,
    ) -> MeasurementSchedule:
        """Build an execution-ready measurement schedule from user inputs."""
        if not schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")

        if readout_amplitudes is None:
            readout_amplitudes = self._control_params.readout_amplitude
        if readout_duration is None:
            readout_duration = DEFAULT_READOUT_DURATION
        if readout_pre_margin is None:
            readout_pre_margin = DEFAULT_READOUT_PRE_MARGIN
        if readout_post_margin is None:
            readout_post_margin = DEFAULT_READOUT_POST_MARGIN

        if add_last_measurement:
            sequence_duration = (
                math.ceil(schedule.duration / WORD_DURATION) * WORD_DURATION
            )
            schedule.pad(total_duration=sequence_duration, pad_side="right")

            readout_targets = list(
                {
                    Target.read_label(label)
                    for label in schedule.labels
                    if not self._targets[label].is_pump
                }
            )
            with PulseSchedule(schedule.labels + readout_targets) as ps:
                ps.call(schedule)
                ps.barrier()
                for target in readout_targets:
                    ps.add(
                        target,
                        self._pulse_factory.readout_pulse(
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
            schedule = ps
        else:
            readout_targets = [
                label for label in schedule.labels if self._targets[label].is_read
            ]

        if not readout_targets:
            raise ValueError("No readout targets in the pulse schedule.")

        schedule.pad(
            total_duration=schedule.duration + EXTRA_CAPTURE_DURATION,
            pad_side="left",
        )

        sequence_duration = (
            math.ceil(schedule.duration / BLOCK_DURATION + 1) * BLOCK_DURATION
        )
        schedule.pad(total_duration=sequence_duration, pad_side="right")

        if add_pump_pulses:
            muxes = []
            for target in readout_targets:
                qubit_label = Target.qubit_label(target)
                mux = self._mux_dict[qubit_label]
                if mux not in muxes:
                    muxes.append(mux)
            with PulseSchedule() as ps_with_pumps:
                ps_with_pumps.call(schedule)
                for mux in muxes:
                    pump_amplitude = self._control_params.get_pump_amplitude(mux.index)
                    ps_with_pumps.add(
                        mux.label,
                        self._pulse_factory.pump_pulse(
                            mux_index=mux.index,
                            duration=schedule.duration,
                            amplitude=pump_amplitude,
                            ramptime=readout_ramptime,
                            type=readout_ramp_type,
                        ),
                    )

            if not ps_with_pumps.is_valid():
                raise ValueError("Invalid pulse schedule with pump pulses.")
            schedule = ps_with_pumps

        if plot:
            schedule.plot()

        capture_schedule = self._build_capture_schedule(
            schedule=schedule,
            readout_targets=readout_targets,
        )

        return MeasurementSchedule(
            pulse_schedule=schedule,
            capture_schedule=capture_schedule,
        )

    def _build_capture_schedule(
        self,
        *,
        schedule: PulseSchedule,
        readout_targets: list[str],
    ) -> CaptureSchedule:
        """Build a capture schedule aligned to readout windows and workaround capture."""
        captures: list[Capture] = []
        readout_ranges = schedule.get_pulse_ranges(readout_targets)
        workaround_duration = EXTRA_SUM_SECTION_LENGTH * _SAMPLING_PERIOD

        for target in readout_targets:
            ranges = readout_ranges.get(target, [])
            if not ranges:
                continue

            # WORKAROUND: keep the extra first capture that aligns subsequent captures.
            captures.append(
                Capture(
                    channels=[target],
                    start_time=0.0,
                    duration=workaround_duration,
                )
            )
            captures.extend(
                [
                    Capture(
                        channels=[target],
                        start_time=rng.start * _SAMPLING_PERIOD,
                        duration=len(rng) * _SAMPLING_PERIOD,
                    )
                    for rng in ranges
                ]
            )

        return CaptureSchedule(captures=captures)
