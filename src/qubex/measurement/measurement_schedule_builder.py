"""Builder for measurement pulse schedules."""

from __future__ import annotations

import math
from collections.abc import Mapping

from qxpulse import PulseSchedule, RampType

from qubex.backend import (
    ControlParams,
    Mux,
    Target,
    TargetRegistry,
)

from .measurement_constraint_profile import MeasurementConstraintProfile
from .measurement_pulse_factory import MeasurementPulseFactory
from .models.capture_schedule import Capture, CaptureSchedule
from .models.measurement_schedule import MeasurementSchedule

_STRICT_PROFILE = MeasurementConstraintProfile.strict_quel1()
WORD_DURATION = _STRICT_PROFILE.word_duration_ns or 0.0
BLOCK_DURATION = _STRICT_PROFILE.block_duration_ns or 0.0
EXTRA_SUM_SECTION_LENGTH = _STRICT_PROFILE.extra_sum_section_length_samples
EXTRA_CAPTURE_DURATION = _STRICT_PROFILE.extra_capture_duration_ns


class MeasurementScheduleBuilder:
    """Build hardware-ready measurement schedules for execution."""

    def __init__(
        self,
        *,
        control_params: ControlParams,
        pulse_factory: MeasurementPulseFactory,
        targets: Mapping[str, Target],
        mux_dict: Mapping[str, Mux],
        target_registry: TargetRegistry | None = None,
        sampling_period: float | None = None,
        constraint_profile: MeasurementConstraintProfile | None = None,
    ) -> None:
        self._control_params = control_params
        self._pulse_factory = pulse_factory
        self._targets = targets
        self._mux_dict = mux_dict
        self._target_registry = target_registry or TargetRegistry()
        if constraint_profile is None:
            if sampling_period is None:
                constraint_profile = MeasurementConstraintProfile.strict_quel1()
            else:
                constraint_profile = MeasurementConstraintProfile.strict_quel1(
                    sampling_period_ns=sampling_period
                )
        self._constraint_profile = constraint_profile

    def _resolve_qubit_label(self, target_label: str) -> str:
        """Resolve qubit label using target registry (legacy fallback enabled)."""
        resolver = self._target_registry.resolve_qubit_label
        try:
            return str(resolver(target_label, allow_legacy=True))
        except TypeError:
            return str(resolver(target_label))

    def _resolve_read_label(self, target_label: str) -> str:
        """Resolve readout label using target registry (legacy fallback enabled)."""
        resolver = self._target_registry.resolve_read_label
        try:
            return str(resolver(target_label, allow_legacy=True))
        except TypeError:
            return str(resolver(target_label))

    @property
    def sampling_period(self) -> float:
        """Return sampling period (ns) used for capture-time conversion."""
        return self._constraint_profile.sampling_period_ns

    @property
    def constraint_profile(self) -> MeasurementConstraintProfile:
        """Return the backend constraint profile used by this builder."""
        return self._constraint_profile

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
        else:
            # Accept both qubit labels (e.g. Q00) and readout labels (e.g. RQ00).
            readout_amplitudes = {
                self._resolve_read_label(label): amplitude
                for label, amplitude in readout_amplitudes.items()
            }

        if add_last_measurement:
            sequence_duration = schedule.duration
            word_duration = self.constraint_profile.word_duration_ns
            if (
                self.constraint_profile.enforce_word_alignment
                and word_duration is not None
            ):
                sequence_duration = (
                    math.ceil(sequence_duration / word_duration) * word_duration
                )
            schedule.pad(total_duration=sequence_duration, pad_side="right")

            readout_targets = list(
                dict.fromkeys(
                    [
                        self._resolve_read_label(label)
                        for label in schedule.labels
                        if not self._targets[label].is_pump
                    ]
                )
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

        if self.constraint_profile.require_workaround_capture:
            schedule.pad(
                total_duration=schedule.duration
                + self.constraint_profile.extra_capture_duration_ns,
                pad_side="left",
            )

        block_duration = self.constraint_profile.block_duration_ns
        if (
            self.constraint_profile.enforce_block_alignment
            and block_duration is not None
        ):
            sequence_duration = (
                math.ceil(schedule.duration / block_duration) * block_duration
            )
            schedule.pad(total_duration=sequence_duration, pad_side="right")

        if add_pump_pulses:
            muxes = []
            for target in readout_targets:
                qubit_label = self._resolve_qubit_label(target)
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
        workaround_duration = self.constraint_profile.workaround_capture_duration_ns

        for target in readout_targets:
            ranges = readout_ranges.get(target, [])
            if not ranges:
                continue

            if self.constraint_profile.require_workaround_capture:
                # Keep workaround capture only for strict backends that require it.
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
                        start_time=rng.start * self.sampling_period,
                        duration=len(rng) * self.sampling_period,
                    )
                    for rng in ranges
                ]
            )

        return CaptureSchedule(captures=captures)
