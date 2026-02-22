"""Tests for the Arbitrary pulse."""

import warnings

import numpy as np
import pytest

import qubex as qx
from qubex.pulse import Arbitrary, Pulse

dt = qx.pulse.get_sampling_period()


def test_arbitrary_inheritance():
    """Arbitrary should inherit from Pulse."""
    assert issubclass(Arbitrary, Pulse)


def test_arbitrary_instantiation_is_not_deprecated():
    """Arbitrary should instantiate without a deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pulse = Arbitrary([0.1, 0.2j])

    assert isinstance(pulse, Pulse)
    assert pulse.values == pytest.approx([0.1, 0.2j])


def test_empty_init():
    """Arbitrary should be initialized with no parameters."""
    pulse = Arbitrary()
    assert pulse.name == "Arbitrary"
    assert pulse.length == 0
    assert pulse.duration == 0.0
    assert pulse.phase == 0.0
    assert pulse.scale == 1.0
    assert pulse.detuning == 0.0
    assert pulse.values == pytest.approx([])
    assert pulse.times == pytest.approx([])


def test_empty_list():
    """Arbitrary should be initialized with an empty list."""
    pulse = Arbitrary([])
    assert pulse.name == "Arbitrary"
    assert pulse.length == 0
    assert pulse.duration == 0.0
    assert pulse.phase == 0.0
    assert pulse.scale == 1.0
    assert pulse.detuning == 0.0
    assert pulse.values == pytest.approx([])
    assert pulse.times == pytest.approx([])


def test_init():
    """Arbitrary should be initialized with valid parameters."""
    pulse = Arbitrary([0, -0.5, +0.5j])
    assert pulse.length == 3
    assert pulse.duration == 3 * dt
    assert pulse.times == pytest.approx(np.arange(3) * dt)
    assert pulse.values == pytest.approx([0, -0.5, +0.5j])
    assert pulse.real == pytest.approx([0, -0.5, 0])
    assert pulse.imag == pytest.approx([0, 0, 0.5])
    assert pulse.abs == pytest.approx([0, 0.5, 0.5])
    assert pulse.angle == pytest.approx([0, np.pi, np.pi / 2])


def test_sampling_period_default():
    """Arbitrary should use the global default sampling period when not provided."""
    pulse = Arbitrary([0, 1])
    assert pulse.sampling_period == dt
    assert pulse.duration == 2 * dt
    assert pulse.times == pytest.approx(np.arange(2) * dt)


def test_sampling_period_override():
    """Arbitrary should use the provided sampling period per instance."""
    custom_dt = dt / 2
    pulse = Arbitrary([0, 1, 2], sampling_period=custom_dt)
    assert pulse.sampling_period == custom_dt
    assert pulse.duration == 3 * custom_dt
    assert pulse.times == pytest.approx(np.arange(3) * custom_dt)


def test_copy():
    """Arbitrary should be copied."""
    pulse = Arbitrary([1, 2, 3])
    copy = pulse.copy()
    assert isinstance(copy, Pulse)
    assert copy is not pulse
    assert copy.values == pytest.approx(pulse.values)


def test_paddded():
    """Arbitrary should be padded with zeros."""
    pulse = Arbitrary([1, 1, 1])
    padded = pulse.padded(10 * dt)
    assert padded != pulse
    assert padded.values == pytest.approx([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    padded_right = pulse.padded(10 * dt, "right")
    assert padded_right.values == pytest.approx([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    padded_left = pulse.padded(10 * dt, "left")
    assert padded_left.values == pytest.approx([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])


def test_scaled():
    """Arbitrary should be scaled by a given parameter."""
    pulse = Arbitrary([0.1, 0.2, 0.3])
    scaled = pulse.scaled(2)
    assert scaled.scale == 2
    assert scaled != pulse
    assert scaled.values == pytest.approx([0.2, 0.4, 0.6])
    assert scaled.scaled(1.5).values == pytest.approx([0.3, 0.6, 0.9])


def test_detuned():
    """Arbitrary should be detuned by a given parameter."""
    pulse = Arbitrary([0.1, 0.1, 0.1])
    detuned = pulse.detuned(0.001)
    assert detuned.detuning == 0.001
    assert detuned != pulse
    assert detuned.values == pytest.approx(
        [
            0.1,
            0.1 * np.exp(-1j * 0.001 * 2 * np.pi * dt),
            0.1 * np.exp(-2j * 0.001 * 2 * np.pi * dt),
        ]
    )


def test_shifted():
    """Arbitrary should be phase shifted by a given parameter."""
    pulse = Arbitrary([1, -1, 1j])
    shifted = pulse.shifted(np.pi / 2)
    assert shifted.phase == np.pi / 2
    assert shifted != pulse
    assert shifted.values == pytest.approx([1j, -1j, -1])


def test_repeated():
    """Arbitrary should be repeated a given number of times."""
    pulse = Arbitrary([1, 2, 3])
    repeated = pulse.repeated(3)
    assert repeated != pulse
    assert repeated.values == pytest.approx([1, 2, 3, 1, 2, 3, 1, 2, 3])


def test_inverted():
    """Arbitrary should be inverted."""
    pulse = Arbitrary([1, 2, 3])
    inverted = pulse.inverted()
    assert inverted != pulse
    assert inverted.values == pytest.approx([-3, -2, -1])
    assert inverted.inverted().values == pytest.approx(pulse.values)


def test_shape_values_preserves_internal_zeros():
    """Arbitrary shape values should keep zero regions inside one pulse waveform."""
    pulse = Arbitrary(
        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        scale=2.0,
        sampling_period=0.4,
    )

    assert pulse.shape_values == pytest.approx(
        np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    )


def test_shape_values_removes_scale_and_phase():
    """Arbitrary shape values should remove scale and phase from pulse values."""
    pulse = Arbitrary([1.0 + 0.0j], scale=2.0, phase=np.pi / 2)

    assert pulse.shape_values == pytest.approx(
        np.array([1.0 + 0.0j], dtype=np.complex128)
    )


def test_shape_values_of_all_zero_pulse():
    """Arbitrary shape values should stay all-zero for all-zero pulses."""
    pulse = Arbitrary([0.0 + 0.0j, 0.0 + 0.0j])

    assert pulse.shape_values == pytest.approx(
        np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    )


def test_shape_hash_ignores_scale_and_phase():
    """Arbitrary shape hash should be identical for scale/phase-only differences."""
    values = [0.1 + 0.2j, 0.3 - 0.4j]
    pulse_a = Arbitrary(values, scale=0.5, phase=0.1)
    pulse_b = Arbitrary(values, scale=1.7, phase=-0.9)

    shape_hash_a = pulse_a.shape_hash
    shape_hash_b = pulse_b.shape_hash
    assert shape_hash_a == shape_hash_b


def test_shape_hash_depends_on_detuning():
    """Arbitrary shape hash should differ when detuning changes."""
    values = [0.1 + 0.2j, 0.3 - 0.4j]
    pulse_a = Arbitrary(values, detuning=0.0)
    pulse_b = Arbitrary(values, detuning=0.02)

    shape_hash_a = pulse_a.shape_hash
    shape_hash_b = pulse_b.shape_hash
    assert shape_hash_a != shape_hash_b
