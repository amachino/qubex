import numpy as np
import pytest

import qubex as qx
from qubex.pulse import Pulse, PulseArray, Waveform

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """PulseArray should inherit from Waveform."""
    assert issubclass(PulseArray, Waveform)


def test_empty_init():
    """PulseArray should be initialized without any parameters."""
    arr = PulseArray()
    assert arr.name == "PulseArray"
    assert arr.length == 0
    assert arr.duration == 0.0
    assert arr.phase == 0.0
    assert arr.scale == 1.0
    assert arr.detuning == 0.0
    assert arr.values == pytest.approx([])
    assert arr.times == pytest.approx([])


def test_empty_list():
    """PulseArray should be initialized with an empty list."""
    arr = PulseArray([])
    assert arr.name == "PulseArray"
    assert arr.length == 0
    assert arr.duration == 0.0
    assert arr.phase == 0.0
    assert arr.scale == 1.0
    assert arr.detuning == 0.0
    assert arr.values == pytest.approx([])
    assert arr.times == pytest.approx([])


def test_init():
    """PulseArray should be initialized with valid parameters."""
    pulse_1 = Pulse([0, -0.5, +0.5j])
    pulse_2 = Pulse([1, 2, 3])
    arr = PulseArray([pulse_1.scaled(2), pulse_2.repeated(2)])
    assert arr.length == 9
    assert arr.duration == 9 * dt
    assert arr.times == pytest.approx(np.arange(9) * dt)
    assert arr.values == pytest.approx([0, -1, 1j, 1, 2, 3, 1, 2, 3])
    assert arr.real == pytest.approx([0, -1, 0, 1, 2, 3, 1, 2, 3])
    assert arr.imag == pytest.approx([0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert arr.abs == pytest.approx([0, 1, 1, 1, 2, 3, 1, 2, 3])
    assert arr.angle == pytest.approx([0, np.pi, np.pi / 2, 0, 0, 0, 0, 0, 0])


def test_copy():
    """PulseArray should be copied."""
    arr = PulseArray([Pulse([1, 2, 3]), Pulse([4, 5, 6])])
    copy = arr.copy()
    assert isinstance(copy, PulseArray)
    assert copy is not arr
    assert copy.values == pytest.approx(arr.values)


def test_paddded():
    """PulseArray should be padded with zeros."""
    arr = PulseArray([Pulse([1, 1]), Pulse([2, 2])])
    padded = arr.padded(10 * dt)
    assert padded != arr
    assert padded.values == pytest.approx([1, 1, 2, 2, 0, 0, 0, 0, 0, 0])

    padded_right = arr.padded(10 * dt, "right")
    assert padded_right.values == pytest.approx([1, 1, 2, 2, 0, 0, 0, 0, 0, 0])

    padded_left = arr.padded(10 * dt, "left")
    assert padded_left.values == pytest.approx([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])


def test_scaled():
    """PulseArray should be scaled by a given parameter."""
    pulse = Pulse([1, 2, 3])
    arr = PulseArray([pulse, pulse.scaled(2)])
    scaled = arr.scaled(0.1)
    assert scaled != arr
    assert scaled.values == pytest.approx([0.1, 0.2, 0.3, 0.2, 0.4, 0.6])


def test_detuned():
    """PulseArray should be detuned by a given parameter."""
    arr = PulseArray([Pulse([0.1, 0.1, 0.1])])
    detuned = arr.detuned(0.001)
    assert detuned != arr
    assert detuned.values == pytest.approx(
        [
            0.1,
            0.1 * np.exp(1j * 0.001 * 2 * np.pi * dt),
            0.1 * np.exp(2j * 0.001 * 2 * np.pi * dt),
        ]
    )


def test_shifted():
    """PulseArray should be shifted by a given parameter."""
    arr = PulseArray([Pulse([1, -1, 1j])])
    shifted = arr.shifted(np.pi / 2)
    assert shifted != arr
    assert shifted.values == pytest.approx([1j, -1j, -1])


def test_repeated():
    """PulseArray should be repeated a given number of times."""
    arr = PulseArray([Pulse([1, 2, 3]), Pulse([4, 5, 6])])
    repeated = arr.repeated(2)
    assert repeated != arr
    assert repeated.values == pytest.approx([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])


def test_nested():
    """PulseArray should be nested."""
    pulse = Pulse([1, 2, 3])
    arr = PulseArray([pulse, pulse.scaled(2)])
    nested = PulseArray([arr, arr.scaled(2)])
    assert nested.values == pytest.approx([1, 2, 3, 2, 4, 6, 2, 4, 6, 4, 8, 12])
