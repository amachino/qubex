"""Tests for constraint-profile behavior in schedule builder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from qxpulse import PulseSchedule, Rect

from qubex.backend import ControlParams, Target
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder


def _make_builder(
    *,
    profile: MeasurementConstraintProfile,
) -> MeasurementScheduleBuilder:
    return MeasurementScheduleBuilder(
        control_params=cast(
            ControlParams,
            SimpleNamespace(readout_amplitude={"RQ00": 0.1}),
        ),
        pulse_factory=cast(MeasurementPulseFactory, SimpleNamespace()),
        targets=cast(
            dict[str, Target],
            {"RQ00": SimpleNamespace(is_pump=False, is_read=True)},
        ),
        mux_dict={},
        constraint_profile=profile,
    )


def test_builder_adds_workaround_capture_for_strict_profile() -> None:
    """Given strict profile, when building schedule, then workaround capture is inserted."""
    builder = _make_builder(profile=MeasurementConstraintProfile.strict_quel1())

    with PulseSchedule(["RQ00"]) as schedule:
        schedule.add("RQ00", Rect(duration=20, amplitude=0.1))

    result = builder.build(schedule=schedule)

    assert len(result.capture_schedule.captures) == 2


def test_builder_skips_workaround_capture_for_relaxed_profile() -> None:
    """Given relaxed profile, when building schedule, then workaround capture is not inserted."""
    builder = _make_builder(profile=MeasurementConstraintProfile.relaxed(2.0))

    with PulseSchedule(["RQ00"]) as schedule:
        schedule.add("RQ00", Rect(duration=20, amplitude=0.1))

    original_duration = schedule.duration
    result = builder.build(schedule=schedule)

    assert len(result.capture_schedule.captures) == 1
    assert result.pulse_schedule.duration == original_duration
