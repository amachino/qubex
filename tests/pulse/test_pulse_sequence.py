import numpy as np
import pytest

from qubex.pulse import Pulse, PulseSequence, Waveform

dt = PulseSequence.SAMPLING_PERIOD


def test_inheritance():
    """PulseSequence should inherit from Waveform."""
    assert issubclass(PulseSequence, Waveform)


def test_empty_init():
    """PulseSequence should be initialized without any parameters."""
    seq = PulseSequence()
    assert seq.length == 0
    assert seq.duration == 0


def test_empty_list():
    """PulseSequence should be initialized with an empty list."""
    seq = PulseSequence([])
    assert seq.length == 0
    assert seq.duration == 0


def test_init():
    """PulseSequence should be initialized with valid parameters."""
    pulse_1 = Pulse([0, -0.5, +0.5j])
    pulse_2 = Pulse([1, 2, 3])
    seq = PulseSequence([pulse_1, pulse_2.repeated(2)])
    assert seq.length == 9
    assert seq.duration == 9 * dt
    assert seq.times == pytest.approx(np.arange(9) * dt)
    assert seq.values == pytest.approx([0, -0.5, 0.5j, 1, 2, 3, 1, 2, 3])
    assert seq.real == pytest.approx([0, -0.5, 0, 1, 2, 3, 1, 2, 3])
    assert seq.imag == pytest.approx([0, 0, 0.5, 0, 0, 0, 0, 0, 0])
    assert seq.abs == pytest.approx([0, 0.5, 0.5, 1, 2, 3, 1, 2, 3])
    assert seq.angle == pytest.approx([0, np.pi, np.pi / 2, 0, 0, 0, 0, 0, 0])


def test_copy():
    """PulseSequence should be copied."""
    seq = PulseSequence([Pulse([1, 2, 3]), Pulse([4, 5, 6])])
    copy = seq.copy()
    assert copy is not seq
    assert copy.values == pytest.approx(seq.values)


def test_paddded():
    """PulseSequence should be padded with zeros."""
    seq = PulseSequence([Pulse([1, 1]), Pulse([2, 2])])
    padded = seq.padded(10 * dt)
    assert padded != seq
    assert padded.values == pytest.approx([1, 1, 2, 2, 0, 0, 0, 0, 0, 0])

    padded_right = seq.padded(10 * dt, "right")
    assert padded_right.values == pytest.approx([1, 1, 2, 2, 0, 0, 0, 0, 0, 0])

    padded_left = seq.padded(10 * dt, "left")
    assert padded_left.values == pytest.approx([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])


def test_scaled():
    """PulseSequence should be scaled by a given parameter."""
    seq = PulseSequence([Pulse([1, 2, 3]), Pulse([4, 5, 6])])
    scaled = seq.scaled(0.1)
    assert scaled != seq
    assert scaled.values == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_detuned():
    """PulseSequence should be detuned by a given parameter."""
    seq = PulseSequence([Pulse([0.1, 0.1, 0.1])])
    detuned = seq.detuned(0.001)
    assert detuned != seq
    assert detuned.values == pytest.approx(
        [
            0.1,
            0.1 * np.exp(1j * 0.001 * 2 * np.pi * dt),
            0.1 * np.exp(2j * 0.001 * 2 * np.pi * dt),
        ]
    )


def test_shifted():
    """PulseSequence should be shifted by a given parameter."""
    seq = PulseSequence([Pulse([1, -1, 1j])])
    shifted = seq.shifted(np.pi / 2)
    assert shifted != seq
    assert shifted.values == pytest.approx([1j, -1j, -1])


def test_repeated():
    """PulseSequence should be repeated a given number of times."""
    seq = PulseSequence([Pulse([1, 2, 3]), Pulse([4, 5, 6])])
    repeated = seq.repeated(2)
    assert repeated != seq
    assert repeated.values == pytest.approx([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
