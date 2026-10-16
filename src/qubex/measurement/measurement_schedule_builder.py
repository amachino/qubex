"""Builder for measurement pulse schedules."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise

from qxpulse import PulseSchedule, RampType

from qubex.system import ControlParameters, Mux, Target, TargetRegistry

from .measurement_constraint_profile import MeasurementConstraintProfile
from .measurement_pulse_factory import MeasurementPulseFactory
from .models.capture_schedule import Capture, CapturePlacement, CaptureSchedule
from .models.measurement_schedule import MeasurementSchedule

__all__ = ["CapturePlacement", "CaptureWindows", "MeasurementScheduleBuilder"]

CaptureWindow = tuple[float, float]
"""Capture window given as `(start, duration)` in ns from the schedule start."""

CaptureWindows = Mapping[str, Sequence[CaptureWindow]]
"""Explicit capture windows keyed by channel label."""


class MeasurementScheduleBuilder:
    """Build hardware-ready measurement schedules for execution."""

    def __init__(
        self,
        *,
        control_params: ControlParameters,
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
                constraint_profile = MeasurementConstraintProfile.quel1()
            else:
                constraint_profile = MeasurementConstraintProfile.quel1(
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

    @staticmethod
    def _copy_channel_frequencies(
        *,
        source: PulseSchedule,
        destination: PulseSchedule,
    ) -> None:
        """Copy channel frequency metadata from one schedule to another."""
        destination_labels = set(destination.labels)
        destination.set_frequencies(
            {
                label: frequency
                for label, frequency in source.get_frequencies().items()
                if label in destination_labels
            }
        )

    @staticmethod
    def _apply_frequency_overrides(
        *,
        schedule: PulseSchedule,
        frequencies: dict[str, float] | None,
    ) -> None:
        """Apply validated frequency overrides to schedule channels."""
        if frequencies is None:
            return

        known_labels = set(schedule.labels)
        unknown_labels = sorted(set(frequencies) - known_labels)
        if unknown_labels:
            joined = ", ".join(f"`{label}`" for label in unknown_labels)
            raise ValueError(f"Unknown frequency override target(s): {joined}.")

        resolved: dict[str, float] = {}
        for label, frequency in frequencies.items():
            value = float(frequency)
            if not math.isfinite(value):
                raise ValueError(f"Frequency override for `{label}` must be finite.")
            resolved[label] = value

        schedule.set_frequencies(resolved)

    def build(
        self,
        *,
        schedule: PulseSchedule,
        frequencies: dict[str, float] | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_drag_coeff: float | None = None,
        readout_amplification: bool | None = None,
        final_measurement: bool | None = None,
        capture_placement: CapturePlacement | None = None,
        capture_targets: list[str] | None = None,
        capture_windows: CaptureWindows | None = None,
        plot: bool | None = None,
    ) -> MeasurementSchedule:
        """Build an execution-ready measurement schedule from user inputs."""
        if readout_amplification is None:
            readout_amplification = False
        if final_measurement is None:
            final_measurement = False
        if capture_placement is None:
            capture_placement = "pulse_aligned"
        if plot is None:
            plot = False

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

        if final_measurement:
            sequence_duration = schedule.duration
            word_duration = self.constraint_profile.word_duration_ns
            if (
                self.constraint_profile.enforce_word_alignment
                and word_duration is not None
            ):
                sequence_duration = (
                    math.ceil(sequence_duration / word_duration) * word_duration
                )
            if self.constraint_profile.final_readout_guard_duration_ns > 0.0:
                # QuEL-1 capture start can drift by one block at runtime.
                # Keep one block blank before appended final readout so it
                # does not overlap trailing control even when that happens.
                sequence_duration += (
                    self.constraint_profile.final_readout_guard_duration_ns
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
                self._copy_channel_frequencies(source=schedule, destination=ps)
                ps.barrier()
                for target in readout_targets:
                    ps.add(
                        target,
                        self._pulse_factory.readout_pulse(
                            target=target,
                            duration=readout_duration,
                            amplitude=readout_amplitudes.get(target),
                            pre_margin=readout_pre_margin,
                            post_margin=readout_post_margin,
                            ramp_time=readout_ramp_time,
                            ramp_type=readout_ramp_type,
                            drag_coeff=readout_drag_coeff,
                        ),
                    )
            schedule = ps
        else:
            readout_targets = [
                label for label in schedule.labels if self._targets[label].is_read
            ]

        if capture_targets is None:
            capture_targets = list(readout_targets)

        if capture_placement == "pulse_aligned" and not readout_targets:
            raise ValueError("No readout targets in the pulse schedule.")
        if capture_placement == "entire_schedule" and not capture_targets:
            raise ValueError(
                "No capture targets specified for entire-schedule capture placement."
            )

        explicit_windows: dict[str, list[CaptureWindow]] | None = None
        if capture_placement == "explicit":
            # Resolve on the caller's time axis before any padding shifts it.
            explicit_windows = self._resolve_capture_windows(
                schedule=schedule,
                capture_windows=capture_windows,
            )

        if self.constraint_profile.require_workaround_capture:
            schedule.pad(
                total_duration=schedule.duration
                + self.constraint_profile.extra_capture_duration_ns,
                pad_side="left",
            )

        if explicit_windows is not None:
            # Explicit windows may extend past the pulses, so grow the schedule
            # before block alignment instead of asking the caller for padding.
            self._extend_schedule_for_capture_windows(
                schedule=schedule,
                capture_windows=explicit_windows,
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

        if readout_amplification:
            muxes = []
            for target in readout_targets:
                qubit_label = self._resolve_qubit_label(target)
                mux = self._mux_dict[qubit_label]
                if mux not in muxes:
                    muxes.append(mux)
            with PulseSchedule() as ps_with_pumps:
                ps_with_pumps.call(schedule)
                self._copy_channel_frequencies(
                    source=schedule,
                    destination=ps_with_pumps,
                )
                for mux in muxes:
                    pump_amplitude = self._control_params.get_pump_amplitude(mux.index)
                    ps_with_pumps.add(
                        mux.label,
                        self._pulse_factory.pump_pulse(
                            mux_index=mux.index,
                            duration=schedule.duration,
                            amplitude=pump_amplitude,
                            ramp_time=readout_ramp_time,
                            ramp_type=readout_ramp_type,
                        ),
                    )

            if not ps_with_pumps.is_valid():
                raise ValueError("Invalid pulse schedule with pump pulses.")
            schedule = ps_with_pumps

        self._apply_frequency_overrides(schedule=schedule, frequencies=frequencies)

        if plot:
            schedule.plot()

        capture_schedule = self._build_capture_schedule(
            schedule=schedule,
            readout_targets=readout_targets,
            capture_targets=capture_targets,
            capture_placement=capture_placement,
            capture_windows=explicit_windows,
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
        capture_targets: list[str],
        capture_placement: CapturePlacement,
        capture_windows: Mapping[str, Sequence[CaptureWindow]] | None = None,
    ) -> CaptureSchedule:
        """Build a capture schedule for the requested placement mode."""
        captures: list[Capture] = []
        workaround_duration = self.constraint_profile.workaround_capture_duration_ns

        if capture_placement == "explicit":
            if capture_windows is None:
                raise ValueError(
                    "`capture_windows` is required when `capture_placement` is `explicit`."
                )
            return self._build_explicit_capture_schedule(
                capture_windows=capture_windows
            )

        if capture_placement == "entire_schedule":
            full_capture_start = 0.0
            full_capture_duration = schedule.duration
            if self.constraint_profile.require_workaround_capture:
                full_capture_start = self.constraint_profile.extra_capture_duration_ns
                full_capture_duration = max(0.0, schedule.duration - full_capture_start)
            for target in capture_targets:
                if self.constraint_profile.require_workaround_capture:
                    captures.append(
                        Capture(
                            channels=[target],
                            start_time=0.0,
                            duration=workaround_duration,
                            is_workaround=True,
                        )
                    )
                if full_capture_duration > 0.0:
                    captures.append(
                        Capture(
                            channels=[target],
                            start_time=full_capture_start,
                            duration=full_capture_duration,
                        )
                    )
            return CaptureSchedule(captures=captures, placement=capture_placement)

        readout_ranges = schedule.get_pulse_ranges(readout_targets)

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
                        is_workaround=True,
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

        return CaptureSchedule(captures=captures, placement=capture_placement)

    def _build_explicit_capture_schedule(
        self,
        *,
        capture_windows: Mapping[str, Sequence[CaptureWindow]],
    ) -> CaptureSchedule:
        """Build a capture schedule from resolved caller-positioned windows."""
        require_workaround = self.constraint_profile.require_workaround_capture
        workaround_duration = self.constraint_profile.workaround_capture_duration_ns

        captures: list[Capture] = []
        for channel, windows in capture_windows.items():
            if require_workaround:
                captures.append(
                    Capture(
                        channels=[channel],
                        start_time=0.0,
                        duration=workaround_duration,
                        is_workaround=True,
                    )
                )
            for start_time, duration in windows:
                captures.append(
                    Capture(
                        channels=[channel],
                        start_time=start_time,
                        duration=duration,
                    )
                )

        return CaptureSchedule(captures=captures, placement="explicit")

    def _extend_schedule_for_capture_windows(
        self,
        *,
        schedule: PulseSchedule,
        capture_windows: Mapping[str, Sequence[CaptureWindow]],
    ) -> None:
        """Grow the schedule so every explicit capture window fits inside it."""
        required_duration = max(
            (
                start_time + duration
                for windows in capture_windows.values()
                for start_time, duration in windows
            ),
            default=0.0,
        )
        if required_duration > schedule.duration:
            schedule.pad(total_duration=required_duration, pad_side="right")

    def _resolve_capture_windows(
        self,
        *,
        schedule: PulseSchedule,
        capture_windows: CaptureWindows | None,
    ) -> dict[str, list[CaptureWindow]]:
        """
        Validate explicit capture windows and move them onto the device time axis.

        Windows are checked against the backend constraints and otherwise used
        unchanged. The only transformation is the leading pad that strict
        backends insert for the workaround capture, which the caller never sees.
        """
        if not capture_windows:
            raise ValueError(
                "`capture_windows` is required when `capture_placement` is `explicit`."
            )
        offset = (
            self.constraint_profile.extra_capture_duration_ns
            if self.constraint_profile.require_workaround_capture
            else 0.0
        )
        labels = set(schedule.labels)
        resolved: dict[str, list[CaptureWindow]] = {}
        for channel, windows in capture_windows.items():
            if channel not in labels:
                raise ValueError(
                    f"Capture channel `{channel}` is not in the pulse schedule."
                )
            if not windows:
                raise ValueError(f"`capture_windows` holds no windows for `{channel}`.")
            checked = [
                self._validate_capture_window(
                    channel=channel,
                    index=index,
                    start=start,
                    duration=duration,
                )
                for index, (start, duration) in enumerate(windows)
            ]
            ordered = sorted(enumerate(checked), key=lambda item: item[1])
            for (_, previous), (_, current) in pairwise(ordered):
                self._validate_capture_window_spacing(
                    channel=channel,
                    previous=previous,
                    current=current,
                )
            resolved[channel] = [
                (start_time + offset, duration) for _, (start_time, duration) in ordered
            ]
        return resolved

    def _validate_capture_window(
        self,
        *,
        channel: str,
        index: int,
        start: float,
        duration: float,
    ) -> CaptureWindow:
        """
        Check one explicit window against the backend word grid.

        The coarser duration grid that decimation imposes depends on a run-time
        option, so the QuEL-1 adapter checks it when the request is built.
        """
        label = f"Capture window {index} ({start}, {duration}) on `{channel}`"
        start_time = float(start)
        window_duration = float(duration)
        if not math.isfinite(start_time) or start_time < 0.0:
            raise ValueError(
                f"{label}: start must be a finite, non-negative time in ns."
            )
        if not math.isfinite(window_duration) or window_duration <= 0.0:
            raise ValueError(
                f"{label}: duration must be a finite, positive time in ns."
            )
        word_duration = self.constraint_profile.word_duration_ns
        if word_duration is not None and self.constraint_profile.enforce_word_alignment:
            if not _is_multiple(start_time, word_duration):
                raise ValueError(
                    f"{label}: start must be a multiple of {word_duration} ns "
                    "on this backend."
                )
            if not _is_multiple(window_duration, word_duration):
                raise ValueError(
                    f"{label}: duration must be a multiple of {word_duration} ns "
                    "on this backend."
                )
        return start_time, window_duration

    def _validate_capture_window_spacing(
        self,
        *,
        channel: str,
        previous: CaptureWindow,
        current: CaptureWindow,
    ) -> None:
        """Reject explicit windows that overlap or leave too little post-blank."""
        word_duration = self.constraint_profile.word_duration_ns
        if not self.constraint_profile.enforce_capture_spacing or word_duration is None:
            return
        gap = current[0] - (previous[0] + previous[1])
        if gap < word_duration:
            raise ValueError(
                f"Capture windows ({previous[0]}, {previous[1]}) and "
                f"({current[0]}, {current[1]}) on `{channel}` overlap or leave less "
                f"than {word_duration} ns of post-blank (gap = {gap} ns)."
            )


def _is_multiple(value: float, unit: float) -> bool:
    """Return whether `value` is an integer multiple of `unit` within tolerance."""
    ratio = value / unit
    return math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9)
