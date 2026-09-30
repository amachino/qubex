"""Tests for constraint-profile behavior in schedule builder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import CrossResonance, PulseSchedule, Rect

from qubex.backend.quel1.compat.driver_loader import load_quel1_driver
from qubex.measurement.adapters import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.measurement.models.measurement_config import MeasurementConfig
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


@pytest.mark.parametrize("duration", [2, 30, 32, 34, 100, 128, 160, 256, 768])
def test_entire_schedule_preserves_waveforms_with_aligned_captures(
    duration: int,
) -> None:
    """Full-span QuEL-1 captures preserve every waveform sample and reserve aligned post blanks."""
    builder = _make_builder(profile=MeasurementConstraintProfile.quel1())
    with PulseSchedule(["RQ00"]) as schedule:
        schedule.add("RQ00", Rect(duration=duration, amplitude=0.1))
    original = schedule.get_sampled_sequences()["RQ00"].copy()

    result = builder.build(schedule=schedule, capture_placement="entire_schedule")

    dummy, main = result.capture_schedule.captures
    total = result.pulse_schedule.duration
    assert total % 128 == 0
    assert dummy.start_time == 0
    assert dummy.duration == 32
    assert main.start_time == 64
    assert main.duration > 0
    assert main.duration % 32 == 0
    assert main.start_time + main.duration == total - 32
    assert main.duration >= duration
    samples = result.pulse_schedule.get_sampled_sequences()["RQ00"]
    np.testing.assert_array_equal(samples[32 : 32 + len(original)], original)
    np.testing.assert_array_equal(samples[-16:], np.zeros(16))


def test_entire_schedule_keeps_relaxed_backend_timing() -> None:
    """Full-span QuEL-3 capture retains its original duration without workaround padding."""
    builder = _make_builder(profile=MeasurementConstraintProfile.quel3(2.0))
    with PulseSchedule(["RQ00"]) as schedule:
        schedule.add("RQ00", Rect(duration=100, amplitude=0.1))

    result = builder.build(schedule=schedule, capture_placement="entire_schedule")

    assert result.pulse_schedule.duration == 100
    assert len(result.capture_schedule.captures) == 1
    assert result.capture_schedule.captures[0].start_time == 0
    assert result.capture_schedule.captures[0].duration == 100


@pytest.mark.parametrize("shot_interval", [0, 2, 8, 32, 128, 200_000])
def test_entire_schedule_compiles_with_post_blanks_without_shot_margin(
    shot_interval: int,
) -> None:
    """Full-span captures retain aligned hardware sections and zero delay for every shot margin."""
    profile = MeasurementConstraintProfile.quel1()
    builder = _make_builder(profile=profile)
    with PulseSchedule(["RQ00"]) as schedule:
        schedule.add("RQ00", Rect(duration=160, amplitude=0.1))
    measurement = builder.build(schedule=schedule, capture_placement="entire_schedule")
    driver = load_quel1_driver()
    backend = SimpleNamespace(
        driver=driver,
        get_resource_map=lambda targets: {target: [{}] for target in targets},
    )
    system = SimpleNamespace(
        control_params=SimpleNamespace(capture_delay_word={0: 19}),
        get_target=lambda _: SimpleNamespace(sideband="U"),
        get_diff_frequency=lambda _: 0.0,
        get_awg_frequency=lambda _: 0.0,
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, backend),
        experiment_system=cast(Any, system),
        constraint_profile=profile,
    )
    config = MeasurementConfig(
        n_shots=64,
        shot_interval=shot_interval,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )

    adapter.validate_schedule(measurement)
    request = adapter.build_execution_request(schedule=measurement, config=config)
    payload = cast(Any, request.payload)
    param = driver.CaptureParamTools.create(
        sequence=payload.cap_sampled_sequence["RQ00"],
        capture_delay_words=0,
        repeats=64,
        interval_samples=payload.interval_ns // 2,
    )

    assert param.capture_delay == 0
    assert len(param.sum_section_list) == 2
    assert [words for words, _ in param.sum_section_list] == [4, 20]
    assert param.sum_section_list[0][1] == 4
    assert param.sum_section_list[1][1] >= 4


def test_entire_schedule_pump_preserves_trailing_blank() -> None:
    """Generated readout amplification stops before the full-span trailing blank."""
    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
            SimpleNamespace(
                readout_amplitude={"RQ00": 0.1}, get_pump_amplitude=lambda _: 0.2
            ),
        ),
        pulse_factory=cast(
            MeasurementPulseFactory,
            SimpleNamespace(
                pump_pulse=lambda **kwargs: Rect(
                    duration=kwargs["duration"], amplitude=kwargs["amplitude"]
                )
            ),
        ),
        targets=cast(
            dict[str, Target], {"RQ00": SimpleNamespace(is_pump=False, is_read=True)}
        ),
        mux_dict=cast(Any, {"Q00": SimpleNamespace(index=0, label="MUX00")}),
        constraint_profile=MeasurementConstraintProfile.quel1(),
    )
    with PulseSchedule(["RQ00"]) as schedule:
        schedule.add("RQ00", Rect(duration=160, amplitude=0.1))

    result = builder.build(
        schedule=schedule,
        capture_placement="entire_schedule",
        readout_amplification=True,
    )

    for samples in result.pulse_schedule.get_sampled_sequences().values():
        np.testing.assert_array_equal(samples[-16:], np.zeros(16))


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
