"""Tests for MCM randomized benchmarking sequence construction."""

from __future__ import annotations

from typing import Any

import pytest

from qubex.contrib.experiment.mcm_randomized_benchmarking import mcm_rb_sequence
from qubex.pulse import Blank, Rect


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
    """Echo construction should reject free time that cannot split into quarters."""
    fake_experiment.pulse.measurement_duration = 66.0

    with pytest.raises(ValueError, match="four equal"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            control_echo=True,
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
