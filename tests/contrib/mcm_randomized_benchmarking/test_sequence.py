"""Tests for MCM randomized benchmarking sequence construction."""

from __future__ import annotations

from typing import Any

import pytest

from qubex.contrib.experiment.mcm_randomized_benchmarking import mcm_rb_sequence
from qubex.pulse import Blank, FlatTop, PulseArray, Rect


def test_protocols_match_duration_for_the_same_random_sequence(
    fake_experiment: Any,
) -> None:
    """All protocols should preserve the same total duration for one seed and length."""
    schedules = {
        protocol: mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol=protocol,
            n_cliffords=2,
            seed=17,
        )
        for protocol in ("mcm-rb", "delay-rb", "mcm-rep")
    }

    assert {schedule.duration for schedule in schedules.values()} == {160.0}
    assert all(schedule.labels == ["Q0", "RQ1"] for schedule in schedules.values())

    assert (
        sum(
            isinstance(element, Rect)
            for element in schedules["mcm-rb"].get_sequence("RQ1").elements
        )
        == 2
    )
    assert all(
        isinstance(element, Blank)
        for element in schedules["delay-rb"].get_sequence("RQ1").elements
    )
    assert (
        sum(
            isinstance(element, Rect)
            for element in schedules["mcm-rep"].get_sequence("RQ1").elements
        )
        == 2
    )


def test_randomized_ancilla_uses_seeded_flips_and_parity_recovery(
    fake_experiment: Any,
) -> None:
    """Randomized ancilla should follow a seeded I/X pattern and recover its parity."""
    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        "Q1",
        protocol="mcm-rb",
        n_cliffords=5,
        seed=3,
        ancilla_mode="randomized",
    )

    ancilla_elements = schedule.get_sequence("Q1").flattened_elements
    pulse_starts: list[float] = []
    recovery = None
    elapsed = 0.0
    for element in ancilla_elements:
        if elapsed == 456.0:
            recovery = element
        if isinstance(element, Rect):
            pulse_starts.append(elapsed)
        elapsed += element.duration

    assert schedule.labels == ["Q0", "Q1", "RQ1"]
    assert schedule.duration == 480.0
    assert schedule.is_valid()
    assert pulse_starts == [8.0, 104.0, 288.0, 456.0]
    assert isinstance(recovery, Rect)


def test_randomized_ancilla_uses_blank_recovery_for_even_parity(
    fake_experiment: Any,
) -> None:
    """An even randomized flip parity should end with a duration-matched blank."""
    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        "Q1",
        protocol="mcm-rb",
        n_cliffords=5,
        seed=18,
        ancilla_mode="randomized",
    )

    ancilla_elements = schedule.get_sequence("Q1").flattened_elements
    recovery = None
    elapsed = 0.0
    for element in ancilla_elements:
        if elapsed == 456.0:
            recovery = element
        elapsed += element.duration

    assert sum(isinstance(element, Rect) for element in ancilla_elements) == 2
    assert isinstance(recovery, Blank)
    assert recovery.duration == 16.0


def test_randomized_ancilla_pattern_is_shared_by_mcm_and_delay_protocols(
    fake_experiment: Any,
) -> None:
    """MCM and delay references should share the same randomized ancilla gates."""
    schedules = {
        protocol: mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol=protocol,
            n_cliffords=5,
            seed=3,
            ancilla_mode="randomized",
        )
        for protocol in ("mcm-rb", "delay-rb")
    }

    assert schedules["mcm-rb"].duration == schedules["delay-rb"].duration
    assert (
        schedules["mcm-rb"].get_sequence("Q1").values.tolist()
        == schedules["delay-rb"].get_sequence("Q1").values.tolist()
    )


def test_randomized_ancilla_rejects_mcm_repetition(fake_experiment: Any) -> None:
    """Randomized ancilla mode should require a Clifford-bearing protocol."""
    with pytest.raises(ValueError, match="mcm-rep"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rep",
            n_cliffords=1,
            seed=3,
            ancilla_mode="randomized",
        )


def test_randomized_ancilla_validates_x180_override(fake_experiment: Any) -> None:
    """Randomized ancilla mode should validate its X180 pulse sampling grid."""
    with pytest.raises(ValueError, match="ancilla_x180"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=3,
            ancilla_mode="randomized",
            ancilla_x180=Rect(
                duration=16.0,
                amplitude=1.0,
                sampling_period=1.0,
            ),
        )


def test_randomized_ancilla_uses_x180_override(fake_experiment: Any) -> None:
    """Randomized ancilla gates and recovery should use the X180 override."""
    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        "Q1",
        protocol="mcm-rb",
        n_cliffords=1,
        seed=3,
        ancilla_mode="randomized",
        ancilla_x180=Rect(
            duration=12.0,
            amplitude=0.8,
            sampling_period=2.0,
        ),
    )

    ancilla_pulses = [
        element
        for element in schedule.get_sequence("Q1").flattened_elements
        if isinstance(element, Rect)
    ]

    assert [pulse.duration for pulse in ancilla_pulses] == [12.0, 12.0]
    assert schedule.duration == 104.0


def test_standard_ancilla_ignores_x180_override(fake_experiment: Any) -> None:
    """Standard mode should not resolve or validate an unused ancilla X180."""
    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        "Q1",
        protocol="mcm-rb",
        n_cliffords=1,
        seed=3,
        ancilla_x180=Rect(
            duration=16.0,
            amplitude=1.0,
            sampling_period=1.0,
        ),
    )

    assert schedule.labels == ["Q0", "RQ1"]
    assert schedule.duration == 80.0


def test_control_echo_places_two_pi_pulses_symmetrically_in_each_measurement(
    fake_experiment: Any,
) -> None:
    """Echoed MCM repetition should place X pulses at one and three quarters."""
    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        "Q1",
        protocol="mcm-rep",
        n_cliffords=1,
        seed=17,
        control_echo=True,
    )

    control_elements = schedule.get_sequence("Q0").flattened_elements
    echo_pulses = [
        element
        for element in control_elements
        if isinstance(element, Rect) and element.duration == 16.0
    ]
    measurement_block = control_elements[1:6]

    assert len(echo_pulses) == 2
    assert [getattr(element, "duration", None) for element in measurement_block] == [
        8.0,
        16.0,
        16.0,
        16.0,
        8.0,
    ]
    assert isinstance(measurement_block[0], Blank)
    assert isinstance(measurement_block[1], Rect)
    assert isinstance(measurement_block[2], Blank)
    assert isinstance(measurement_block[3], Rect)
    assert isinstance(measurement_block[4], Blank)


def test_control_echo_uses_quarters_of_ramp_trimmed_active_readout(
    fake_experiment: Any,
) -> None:
    """Echo pulse centers should quarter the ramp-trimmed active interval."""
    measurement = PulseArray(
        [
            Blank(duration=16.0, sampling_period=2.0),
            FlatTop(
                duration=64.0,
                amplitude=0.25,
                tau=16.0,
                sampling_period=2.0,
            ).padded(total_duration=80.0),
        ]
    )

    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        "Q1",
        protocol="mcm-rep",
        n_cliffords=1,
        seed=17,
        control_echo=True,
        measurement_waveform=measurement,
    )

    measurement_block = schedule.get_sequence("Q0").flattened_elements[1:6]

    assert [element.duration for element in measurement_block] == [
        28.0,
        16.0,
        8.0,
        16.0,
        28.0,
    ]


def test_control_echo_rejects_all_zero_measurement_waveform(
    fake_experiment: Any,
) -> None:
    """Echo construction should require an identifiable active readout interval."""
    with pytest.raises(ValueError, match="active readout interval"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            control_echo=True,
            measurement_waveform=Blank(duration=64.0, sampling_period=2.0),
        )


def test_control_echo_rejects_ramp_off_sampling_grid(fake_experiment: Any) -> None:
    """Echo construction should require a readout ramp on the sampling grid."""
    measurement = FlatTop(
        duration=64.0,
        amplitude=0.25,
        tau=3.0,
        sampling_period=2.0,
    )

    with pytest.raises(ValueError, match="ramp duration"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            control_echo=True,
            measurement_waveform=measurement,
        )


def test_control_echo_rejects_multiple_active_flat_top_pulses(
    fake_experiment: Any,
) -> None:
    """Echo construction should reject an ambiguous multi-pulse active interval."""
    measurement = PulseArray(
        [
            FlatTop(
                duration=32.0,
                amplitude=0.25,
                tau=4.0,
                sampling_period=2.0,
            ),
            FlatTop(
                duration=32.0,
                amplitude=0.25,
                tau=4.0,
                sampling_period=2.0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="at most one FlatTop"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            control_echo=True,
            measurement_waveform=measurement,
        )


def test_control_echo_rejects_measurements_shorter_than_two_pi_pulses(
    fake_experiment: Any,
) -> None:
    """Echo construction should reject a measurement window shorter than two pi pulses."""
    fake_experiment.pulse.measurement_duration = 24.0

    with pytest.raises(ValueError, match="at least twice"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            control_echo=True,
        )


def test_control_echo_rejects_asymmetric_sampling_grid(fake_experiment: Any) -> None:
    """Echo construction should reject centers off the pulse sampling grid."""
    fake_experiment.pulse.measurement_duration = 66.0

    with pytest.raises(ValueError, match="sampling grid"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            control_echo=True,
        )


def test_sequence_rejects_nested_waveform_sampling_period_mismatch(
    fake_experiment: Any,
) -> None:
    """Sequence construction should validate every physical pulse sampling grid."""
    measurement = PulseArray([Rect(duration=64.0, amplitude=0.25, sampling_period=1.0)])

    with pytest.raises(ValueError, match="sampling period"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            measurement_waveform=measurement,
        )


def test_unused_echo_override_is_ignored(fake_experiment: Any) -> None:
    """An echo override should not be validated while control echo is disabled."""
    incompatible_echo = Rect(
        duration=16.0,
        amplitude=1.0,
        sampling_period=1.0,
    )

    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        "Q1",
        protocol="mcm-rb",
        n_cliffords=1,
        seed=17,
        control_echo=False,
        echo_x180=incompatible_echo,
    )

    assert schedule.duration == 80.0


def test_boolean_clifford_count_is_rejected(fake_experiment: Any) -> None:
    """A boolean should not be accepted as a randomized Clifford count."""
    with pytest.raises(TypeError, match="nonnegative integer"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=True,
            seed=17,
        )


@pytest.mark.parametrize("protocol", ["unknown", "mcm_rb"])
def test_invalid_protocol_is_rejected(fake_experiment: Any, protocol: str) -> None:
    """Sequence construction should reject protocol names outside the public literals."""
    with pytest.raises(ValueError, match="protocol"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol=protocol,  # type: ignore[arg-type]
            n_cliffords=1,
            seed=17,
        )


def test_invalid_ancilla_mode_is_rejected(fake_experiment: Any) -> None:
    """Sequence construction should reject unknown ancilla modes."""
    with pytest.raises(ValueError, match="ancilla_mode"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            ancilla_mode="unknown",  # type: ignore[arg-type]
        )
