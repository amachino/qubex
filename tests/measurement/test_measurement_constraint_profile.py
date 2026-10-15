"""Tests for constraint-profile behavior in schedule builder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from qxpulse import CrossResonance, PulseSchedule, Rect

from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.system import ControlParameters, Target


def _make_builder(
    *,
    profile: MeasurementConstraintProfile,
) -> MeasurementScheduleBuilder:
    return MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
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
    builder = _make_builder(profile=MeasurementConstraintProfile.quel1())

    with PulseSchedule(["RQ00"]) as schedule:
        schedule.add("RQ00", Rect(duration=20, amplitude=0.1))

    result = builder.build(schedule=schedule)

    assert len(result.capture_schedule.captures) == 2
    assert result.capture_schedule.captures[0].is_workaround is True
    assert result.capture_schedule.captures[1].is_workaround is False


def test_builder_skips_workaround_capture_for_relaxed_profile() -> None:
    """Given relaxed profile, when building schedule, then workaround capture is not inserted."""
    builder = _make_builder(profile=MeasurementConstraintProfile.quel3(2.0))

    with PulseSchedule(["RQ00"]) as schedule:
        schedule.add("RQ00", Rect(duration=20, amplitude=0.1))

    original_duration = schedule.duration
    result = builder.build(schedule=schedule)

    assert len(result.capture_schedule.captures) == 1
    assert result.capture_schedule.captures[0].is_workaround is False
    assert result.pulse_schedule.duration == original_duration


def test_quel1_profile_defaults_final_readout_guard_to_one_block() -> None:
    """Given QuEL-1 profile, when reading final readout guard, then the default guard is one block."""
    profile = MeasurementConstraintProfile.quel1()

    assert profile.final_readout_guard_length_samples == profile.block_length_samples
    assert profile.final_readout_guard_duration_ns == profile.block_duration_ns


def test_builder_preserves_repeated_cross_resonance_duration() -> None:
    """Sequential CR repetition points should retain the full control duration before readout."""
    sequence = CrossResonance(
        "Q00",
        "Q01",
        cr_amplitude=0.1,
        cr_duration=272,
        cr_ramptime=16,
        echo=True,
        pi_pulse=Rect(duration=40, amplitude=0.1),
    )
    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
            SimpleNamespace(readout_amplitude={"RQ00": 0.1, "RQ01": 0.1}),
        ),
        pulse_factory=cast(
            MeasurementPulseFactory,
            SimpleNamespace(readout_pulse=lambda **_: Rect(duration=16, amplitude=0.1)),
        ),
        targets=cast(
            dict[str, Target],
            {
                label: SimpleNamespace(is_pump=False, is_read=False)
                for label in sequence.labels
            },
        ),
        mux_dict={},
        constraint_profile=MeasurementConstraintProfile.quel3(2.0),
    )

    for repetitions in range(4):
        result = builder.build(
            schedule=sequence.repeated(repetitions), final_measurement=True
        )

        assert result.pulse_schedule.duration == 624 * repetitions + 16
        assert len(result.capture_schedule.captures) == 2
        assert all(
            capture.start_time == 624 * repetitions
            for capture in result.capture_schedule.captures
        )
