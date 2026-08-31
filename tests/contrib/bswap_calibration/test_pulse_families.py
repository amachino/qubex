"""Explicit bSWAP pulse-family dispatch and sampled carrier/frame contracts."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from qxpulse import FlatTop, Rect

from qubex.contrib.experiment.bswap_calibration import pulses


def recipe(**updates: Any) -> dict[str, Any]:
    """Return the I-only historical shape parameters, not a calibrated gate."""
    record = dict(
        pulse_family="RaisedCosine",
        gate_kind="bswap",
        amplitude=0.85,
        frequency_ghz=4.772115,
        duration_ns=644.0,
        ramp_ns=30.0,
        gate_start_ns=24.0,
        cancel_amplitude_ratio=0.04,
        cancel_phase_rad=0.3,
        phase_calibration=dict(
            pre_active_rad=0.0, post_active_rad=0.4, post_passive_rad=-0.2
        ),
    )
    record.update(updates)
    return record


def make(record: dict[str, Any], **kwargs: Any) -> Any:
    """Use one explicit cyclic-Rabi to angular-envelope conversion."""
    options = dict(rabi_ghz_per_amplitude=0.636, transition_frequency_ghz=4.41735)
    options.update(kwargs)
    return pulses.make_bswap_pulse(record, **options)


@pytest.mark.parametrize("duration", [60.0, 360.0, 644.0])
@pytest.mark.parametrize("gain", [0.1, 0.636, 2.0])
def test_raised_cosine_matches_historical_i_only_flat_top(
    duration: float, gain: float
) -> None:
    """Angular construction and final K scaling preserve the old I-only samples."""
    record = recipe(duration_ns=duration)
    before = deepcopy(record)
    actual = make(record, rabi_ghz_per_amplitude=gain)
    expected = FlatTop(
        duration=duration,
        amplitude=0.85,
        tau=30,
        type="RaisedCosine",
        sampling_period=2,
    )
    np.testing.assert_allclose(actual.values, expected.values, rtol=0, atol=4e-16)
    np.testing.assert_array_equal(actual.values.imag, np.zeros(int(duration / 2)))
    np.testing.assert_array_equal(actual.times, expected.times)
    assert actual.duration == duration
    assert actual.amplitude == pytest.approx(2 * np.pi * gain * 0.85)
    assert actual.scale == pytest.approx(1 / (2 * np.pi * gain))
    assert actual.correction_type is None
    assert record == before


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "window", [{"type": "hann"}, {"type": "tukey", "rise_end": 0.2, "fall_start": 0.7}]
)
def test_default_and_explicit_squad_are_bitwise_unchanged(
    explicit: bool, window: dict
) -> None:
    """The generic factory delegates SQUAD without changing a single sampled value."""
    record = recipe(ramp_ns=16, window=window, cd_strength=0.7, design_delta_scale=1.3)
    if explicit:
        record["pulse_family"] = "Squad"
    else:
        del record["pulse_family"]
    options = dict(rabi_ghz_per_amplitude=0.636, transition_frequency_ghz=4.41735)
    direct = pulses.make_squad_pulse(record, **options)
    generic = pulses.make_bswap_pulse(record, **options)
    np.testing.assert_array_equal(generic.values, direct.values)
    np.testing.assert_array_equal(generic.times, direct.times)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pulse_family", "raised_cosine"),
        ("pulse_family", None),
        ("window", {"type": "hann"}),
        ("window", {}),
        ("window", None),
        ("design_delta_scale", 1.0),
        ("delta", -0.2),
        ("correction_type", "CD"),
        ("correction_factor", 0.0),
        ("beta", 0.0),
        ("cd_strength", 1.0),
        ("cd_strength", -1.0),
        ("cd_strength", float("nan")),
    ],
)
def test_unsupported_shape_options_are_not_silently_ignored(
    key: str, value: Any
) -> None:
    """Unknown families, SQUAD window knobs and unsupported corrections fail clearly."""
    with pytest.raises(
        (ValueError, TypeError), match=r"family|RaisedCosine|cd_strength"
    ):
        make(recipe(**{key: value}))


@pytest.mark.parametrize(
    ("updates", "kwargs"),
    [
        ({"duration_ns": 59.0}, {}),
        ({"duration_ns": 645.0}, {}),
        ({"ramp_ns": 29.0}, {}),
        ({"ramp_ns": 0.0}, {}),
        ({"amplitude": 0.0}, {}),
        ({"amplitude": 1.01}, {}),
        ({"frequency_ghz": float("nan")}, {}),
        ({}, {"transition_frequency_ghz": float("nan")}),
        ({}, {"rabi_ghz_per_amplitude": 0.0}),
        ({}, {"sampling_period_ns": 0.0}),
        ({}, {"max_command": 0.84}),
    ],
)
def test_invalid_units_grid_or_headroom_fail(updates: Any, kwargs: Any) -> None:
    """The I-only family retains finite-unit, native-grid and headroom checks."""
    with pytest.raises(
        ValueError, match=r"finite|grid|positive|include|amplitude|headroom"
    ):
        make(recipe(**updates), **kwargs)


def test_i_only_shape_is_carrier_independent_even_at_zero_detuning() -> None:
    """RaisedCosine I-only has no artificial SQUAD zero-detuning singularity."""
    a = make(recipe(cd_strength=0.0))
    b = make(recipe(cd_strength=0.0, frequency_ghz=4.41735))
    np.testing.assert_array_equal(a.values, b.values)


def test_compiler_uses_explicit_family_with_existing_carrier_and_frame_transport() -> (
    None
):
    """Repeated RaisedCosine main/cancel tones retain exact logical/carrier placement."""
    bank = {
        "bswap": recipe(),
        "sqrt_bswap": recipe(
            gate_kind="sqrt_bswap",
            duration_ns=360.0,
            phase_calibration=dict(
                pre_active_rad=0.3, post_active_rad=0.2, post_passive_rad=-0.4
            ),
        ),
    }
    qubits = ("A", "P")
    references = {"A": 4.41735, "P": 5.103}
    targets = {"D": 4.77, "C": 4.769}
    kwargs: dict[str, Any] = dict(
        recipes=bank,
        qubits=qubits,
        drive_label="D",
        cancel_label="C",
        target_frequencies_ghz=targets,
        reference_frequencies_ghz=references,
        rabi_ghz_per_amplitude=0.636,
        x90={q: Rect(duration=24, amplitude=0.1) for q in qubits},
        xpi={q: Rect(duration=24, amplitude=0.2) for q in qubits},
        prepared=("0", "1"),
        basis="XY",
        backend_preamble_ns=40.0,
    )
    schedule, report = pulses.compile_campaign(
        ["BSWAP", ("VZ", 0.2, -0.4), "RAW_SQRT_BSWAP", "BSWAP"], **kwargs
    )
    events = [e for e in report["events"] if e["kind"] != "local"]
    assert len(events) == 3
    assert report["duration_ns"] == 24 + 644 + 360 + 644 + 24
    for event in events:
        base = bank[event["kind"]]
        raw = FlatTop(
            duration=base["duration_ns"],
            amplitude=base["amplitude"],
            tau=30,
            type="RaisedCosine",
            sampling_period=2,
        )
        first = round(event["start_ns"] / 2)
        last = first + raw.length
        for label, ratio, relative in [
            ("D", 1.0, 0.0),
            ("C", base["cancel_amplitude_ratio"], base["cancel_phase_rad"]),
        ]:
            actual = np.asarray(
                schedule.get_sequence(label).get_values(apply_frame_shifts=True)
            )[first:last]
            expected = (
                raw.values
                * ratio
                * np.exp(1j * (event["logical_drive_phase_rad"] + relative))
            )
            expected *= np.exp(
                -2j
                * np.pi
                * (base["frequency_ghz"] - targets[label])
                * (event["start_ns"] + 40 + raw.times)
            )
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-11)


def test_compiler_default_squad_matches_explicit_squad_bitwise() -> None:
    """Dispatch does not change existing compiler samples or reports for SQUAD."""
    record = recipe(
        ramp_ns=16, cd_strength=1.0, design_delta_scale=1.0, window={"type": "hann"}
    )
    del record["pulse_family"]
    kwargs: dict[str, Any] = dict(
        qubits=("A", "P"),
        drive_label="D",
        cancel_label="C",
        target_frequencies_ghz={"D": 4.77, "C": 4.77},
        reference_frequencies_ghz={"A": 4.41735, "P": 5.103},
        rabi_ghz_per_amplitude=0.636,
        x90={q: Rect(duration=24, amplitude=0.1) for q in ("A", "P")},
        xpi={q: Rect(duration=24, amplitude=0.2) for q in ("A", "P")},
    )
    a, ra = pulses.compile_campaign(["BSWAP"], recipes={"bswap": record}, **kwargs)
    b, rb = pulses.compile_campaign(
        ["BSWAP"], recipes={"bswap": {**record, "pulse_family": "Squad"}}, **kwargs
    )
    for label in ("A", "P", "D", "C"):
        np.testing.assert_array_equal(
            a.get_sequence(label).get_values(apply_frame_shifts=True),
            b.get_sequence(label).get_values(apply_frame_shifts=True),
        )
    assert ra == rb
