"""Tests for MCM randomized benchmarking sequence construction."""

from __future__ import annotations

from typing import Any

import pytest

from qubex.contrib.experiment.mcm_randomized_benchmarking import (
    MCMRBProtocol,
    mcm_rb_sequence,
)
from qubex.pulse import (
    Blank,
    FlatTop,
    PulseArray,
    Rect,
    get_sampling_period,
    set_sampling_period,
)


@pytest.mark.parametrize("control_echo", [False, True])
def test_protocols_match_duration_for_the_same_random_sequence(
    fake_experiment: Any,
    control_echo: bool,
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
            control_echo=control_echo,
        )
        for protocol in ("mcm-rb", "delay-rb", "mcm-rep", "delay-rep")
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
    assert all(
        isinstance(element, Blank)
        for element in schedules["delay-rep"].get_sequence("RQ1").elements
    )


def test_sequence_synchronizes_the_global_pulse_sampling_period(
    fake_experiment: Any,
) -> None:
    """Sequence construction should use the experiment sampling period globally."""
    original_sampling_period = get_sampling_period()
    try:
        set_sampling_period(1.0)

        schedule = mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
        )

        assert get_sampling_period() == 2.0
        assert schedule.duration == 80.0
        assert schedule.is_valid()
        assert {
            schedule.get_sequence(label).sampling_period for label in schedule.labels
        } == {2.0}
    finally:
        set_sampling_period(original_sampling_period)


@pytest.mark.parametrize("protocol", ["mcm-rb", "delay-rb", "mcm-rep", "delay-rep"])
def test_one_ancilla_with_multiple_controls_builds_matched_schedules(
    fake_experiment: Any,
    protocol: MCMRBProtocol,
) -> None:
    """One ancilla should support simultaneous independently randomized controls."""
    schedule = mcm_rb_sequence(
        fake_experiment,
        ["Q0", "Q2"],
        "Q1",
        protocol=protocol,
        n_cliffords=2,
        seed=17,
    )

    assert schedule.labels == ["Q0", "Q2", "RQ1"]
    assert schedule.duration == 160.0
    assert schedule.is_valid()
    for control in ("Q0", "Q2"):
        control_elements = schedule.get_sequence(control).flattened_elements
        if protocol.endswith("rep"):
            assert all(isinstance(element, Blank) for element in control_elements)
        else:
            assert any(isinstance(element, Rect) for element in control_elements)


def test_multiple_controls_use_reproducible_independent_clifford_seeds(
    fake_experiment: Any,
) -> None:
    """Each simultaneous control should receive a reproducible independent seed."""
    generator = fake_experiment.benchmarking_service.clifford_generator

    mcm_rb_sequence(
        fake_experiment,
        ["Q0", "Q2"],
        "Q1",
        protocol="mcm-rb",
        n_cliffords=2,
        seed=17,
    )
    first_calls = list(generator.calls)
    generator.calls.clear()
    mcm_rb_sequence(
        fake_experiment,
        ["Q0", "Q2"],
        "Q1",
        protocol="mcm-rb",
        n_cliffords=2,
        seed=17,
    )

    assert generator.calls == first_calls
    assert len(first_calls) == 2
    assert first_calls[0][2] != first_calls[1][2]


def test_protocols_match_when_controls_have_different_clifford_durations(
    fake_experiment: Any,
) -> None:
    """Protocol matching should use the longest simultaneous control layer."""

    class SeedDependentCliffordGenerator:
        """Generate a seed-dependent physical Clifford duration."""

        @staticmethod
        def create_rb_sequences(
            n: int,
            type: str,
            seed: int | None,
        ) -> tuple[list[list[str]], list[str]]:
            assert type == "1Q"
            assert seed is not None
            clifford = ["X90"] * (seed % 5 + 1)
            return [clifford for _ in range(n)], ["X90"]

    fake_experiment.benchmarking_service.clifford_generator = (
        SeedDependentCliffordGenerator()
    )

    schedules = {
        protocol: mcm_rb_sequence(
            fake_experiment,
            ["Q0", "Q2"],
            "Q1",
            protocol=protocol,
            n_cliffords=1,
            seed=17,
        )
        for protocol in ("mcm-rb", "delay-rb", "mcm-rep", "delay-rep")
    }

    assert {schedule.duration for schedule in schedules.values()} == {112.0}
    assert all(schedule.is_valid() for schedule in schedules.values())


def test_multiple_randomized_ancillas_use_independent_recoverable_streams(
    fake_experiment: Any,
) -> None:
    """Each simultaneous ancilla should use an independent seeded I/X stream."""
    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        ["Q1", "Q2"],
        protocol="mcm-rb",
        n_cliffords=12,
        seed=3,
        ancilla_mode="randomized",
    )

    assert schedule.labels == ["Q0", "Q1", "Q2", "RQ1", "RQ2"]
    assert schedule.is_valid()
    assert (
        schedule.get_sequence("Q1").values.tolist()
        != schedule.get_sequence("Q2").values.tolist()
    )
    for ancilla in ("Q1", "Q2"):
        assert schedule.get_sequence(ancilla).duration == schedule.duration
        assert (
            sum(
                isinstance(element, Rect)
                for element in schedule.get_sequence(ancilla).flattened_elements
            )
            % 2
            == 0
        )


def test_multiple_controls_and_ancillas_preserve_duration_with_echo(
    fake_experiment: Any,
) -> None:
    """Many-to-many schedules should match duration and echo every control."""
    schedules = {
        protocol: mcm_rb_sequence(
            fake_experiment,
            ["Q0", "Q2"],
            ["Q1", "Q3"],
            protocol=protocol,
            n_cliffords=2,
            seed=17,
            control_echo=True,
            ancilla_mode="randomized",
        )
        for protocol in ("mcm-rb", "delay-rb", "mcm-rep", "delay-rep")
    }

    assert len({schedule.duration for schedule in schedules.values()}) == 1
    assert all(schedule.is_valid() for schedule in schedules.values())
    for schedule in schedules.values():
        for control in ("Q0", "Q2"):
            assert (
                sum(
                    isinstance(element, Rect) and element.duration == 16.0
                    for element in schedule.get_sequence(control).flattened_elements
                )
                >= 4
            )


def test_multiple_ancillas_require_matching_ramp_trimmed_active_windows(
    fake_experiment: Any,
) -> None:
    """Multiple ancillas should reject mismatched ramp-trimmed active windows."""
    with pytest.raises(ValueError, match="ramp-trimmed active intervals"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            ["Q1", "Q2"],
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            measurement_waveform={
                "Q1": Rect(duration=64.0, amplitude=0.25, sampling_period=2.0),
                "Q2": Rect(duration=80.0, amplitude=0.25, sampling_period=2.0),
            },
        )


def test_multiple_ancillas_reject_shifted_active_windows_with_equal_duration(
    fake_experiment: Any,
) -> None:
    """Equal readout durations should not hide shifted ancilla active windows."""
    measurement_waveforms = {
        "Q1": PulseArray(
            [
                Blank(duration=8.0, sampling_period=2.0),
                Rect(duration=64.0, amplitude=0.25, sampling_period=2.0),
                Blank(duration=24.0, sampling_period=2.0),
            ]
        ),
        "Q2": PulseArray(
            [
                Blank(duration=16.0, sampling_period=2.0),
                Rect(duration=64.0, amplitude=0.25, sampling_period=2.0),
                Blank(duration=16.0, sampling_period=2.0),
            ]
        ),
    }

    with pytest.raises(ValueError, match=r"Q1.*8\.0.*72\.0.*Q2.*16\.0.*80\.0"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            ["Q1", "Q2"],
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            measurement_waveform=measurement_waveforms,
        )


def test_matching_active_windows_allow_different_readout_slot_durations(
    fake_experiment: Any,
) -> None:
    """Matching active windows should allow different trailing readout margins."""
    original_sampling_period = get_sampling_period()
    try:
        set_sampling_period(2.0)
        common_active_pulse = FlatTop(
            duration=64.0,
            amplitude=0.25,
            tau=8.0,
            sampling_period=2.0,
        )
        measurement_waveforms = {
            "Q1": PulseArray(
                [
                    Blank(duration=8.0, sampling_period=2.0),
                    common_active_pulse,
                    Blank(duration=8.0, sampling_period=2.0),
                ]
            ),
            "Q2": PulseArray(
                [
                    Blank(duration=8.0, sampling_period=2.0),
                    common_active_pulse,
                    Blank(duration=24.0, sampling_period=2.0),
                ]
            ),
        }

        schedule = mcm_rb_sequence(
            fake_experiment,
            "Q0",
            ["Q1", "Q2"],
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            control_echo=True,
            measurement_waveform=measurement_waveforms,
        )

        assert schedule.duration == 112.0
        assert schedule.is_valid()
        assert schedule.get_sequence("RQ1").duration == 112.0
        assert schedule.get_sequence("RQ2").duration == 112.0
    finally:
        set_sampling_period(original_sampling_period)


def test_multiple_targets_require_target_keyed_waveform_overrides(
    fake_experiment: Any,
) -> None:
    """A scalar override should be rejected for a multiple-target role."""
    with pytest.raises(ValueError, match=r"x90.*mapping"):
        mcm_rb_sequence(
            fake_experiment,
            ["Q0", "Q2"],
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
            x90=Rect(duration=8.0, amplitude=0.5, sampling_period=2.0),
        )


@pytest.mark.parametrize(
    ("control", "ancilla", "message"),
    [
        ([], ["Q1"], "control.*at least one"),
        (["Q0", "RQ0"], ["Q1"], "control.*duplicate"),
        (["Q0"], ["Q0", "Q1"], "disjoint"),
    ],
)
def test_multiple_target_validation_rejects_ambiguous_groups(
    fake_experiment: Any,
    control: list[str],
    ancilla: list[str],
    message: str,
) -> None:
    """Multiple-target inputs should be nonempty, unique, and role-disjoint."""
    with pytest.raises(ValueError, match=message):
        mcm_rb_sequence(
            fake_experiment,
            control,
            ancilla,
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
        )


def test_target_validation_rejects_shared_terminal_readout_labels(
    fake_experiment: Any,
) -> None:
    """Simultaneous targets should require distinct terminal readout labels."""
    fake_experiment.experiment_system.resolve_read_label = lambda target: "RSHARED"

    with pytest.raises(ValueError, match="distinct readout labels"):
        mcm_rb_sequence(
            fake_experiment,
            "Q0",
            "Q1",
            protocol="mcm-rb",
            n_cliffords=1,
            seed=17,
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
        assert isinstance(element, (Blank, Rect))
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
        assert isinstance(element, (Blank, Rect))
        if elapsed == 456.0:
            recovery = element
        elapsed += element.duration

    assert sum(isinstance(element, Rect) for element in ancilla_elements) == 2
    assert isinstance(recovery, Blank)
    assert recovery.duration == 16.0


def test_randomized_ancilla_pattern_is_shared_by_all_protocols(
    fake_experiment: Any,
) -> None:
    """All protocols should share ancilla gates, recovery, and total duration."""
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
        for protocol in ("mcm-rb", "delay-rb", "mcm-rep", "delay-rep")
    }

    assert len({schedule.duration for schedule in schedules.values()}) == 1
    assert (
        len(
            {
                tuple(schedule.get_sequence("Q1").values.tolist())
                for schedule in schedules.values()
            }
        )
        == 1
    )
    expected_control_durations = [
        duration
        for clifford_duration in (8.0, 16.0, 8.0, 16.0, 8.0)
        for duration in (clifford_duration, 16.0, 64.0)
    ] + [16.0, 8.0]
    for protocol in ("mcm-rep", "delay-rep"):
        control_elements = schedules[protocol].get_sequence("Q0").flattened_elements
        assert all(isinstance(element, Blank) for element in control_elements)
        assert [
            element.duration
            for element in control_elements
            if isinstance(element, Blank)
        ] == expected_control_durations
    assert all(
        isinstance(element, Blank)
        for element in schedules["delay-rep"].get_sequence("RQ1").elements
    )
    assert (
        sum(
            isinstance(element, Rect)
            for element in schedules["mcm-rep"].get_sequence("RQ1").elements
        )
        == 5
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


def test_delay_repetition_keeps_control_echo_in_reference_windows(
    fake_experiment: Any,
) -> None:
    """Delay repetition should retain control echo while blanking readout pulses."""
    schedule = mcm_rb_sequence(
        fake_experiment,
        "Q0",
        "Q1",
        protocol="delay-rep",
        n_cliffords=2,
        seed=17,
        control_echo=True,
        ancilla_mode="randomized",
    )

    control_elements = schedule.get_sequence("Q0").flattened_elements

    assert (
        sum(
            isinstance(element, Rect) and element.duration == 16.0
            for element in control_elements
        )
        == 4
    )
    assert all(
        isinstance(element, Blank) for element in schedule.get_sequence("RQ1").elements
    )


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

    assert all(isinstance(element, (Blank, Rect)) for element in measurement_block)
    assert [
        element.duration
        for element in measurement_block
        if isinstance(element, (Blank, Rect))
    ] == [
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
