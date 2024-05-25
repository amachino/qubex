import numpy as np
import pytest
from qubex.pulse import Pulse, Waveform

dt = Pulse.SAMPLING_PERIOD


def test_inheritance():
    """Pulse should inherit from Waveform."""
    assert issubclass(Pulse, Waveform)


def test_empty_init():
    """Pulse should raise a TypeError if no parameters are provided."""
    with pytest.raises(TypeError):
        Pulse()  # type: ignore


def test_empty_list():
    """Pulse should be initialized with an empty list."""
    pulse = Pulse([])
    assert pulse.length == 0
    assert pulse.duration == 0


def test_init():
    """Pulse should be initialized with valid parameters."""
    pulse = Pulse([0, -0.5, +0.5j])
    assert pulse.length == 3
    assert pulse.duration == 3 * dt
    assert pulse.times == pytest.approx(np.arange(3) * dt)
    assert pulse.values == pytest.approx([0, -0.5, +0.5j])
    assert pulse.real == pytest.approx([0, -0.5, 0])
    assert pulse.imag == pytest.approx([0, 0, 0.5])
    assert pulse.abs == pytest.approx([0, 0.5, 0.5])
    assert pulse.angle == pytest.approx([0, np.pi, np.pi / 2])


def test_copy():
    """Pulse should be copied."""
    pulse = Pulse([1, 2, 3])
    copy = pulse.copy()
    assert copy is not pulse
    assert copy.values == pytest.approx(pulse.values)


def test_paddded():
    """Pulse should be padded with zeros."""
    pulse = Pulse([1, 1, 1])
    padded = pulse.padded(10 * dt)
    assert padded != pulse
    assert padded.values == pytest.approx([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    padded_right = pulse.padded(10 * dt, "right")
    assert padded_right.values == pytest.approx([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    padded_left = pulse.padded(10 * dt, "left")
    assert padded_left.values == pytest.approx([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])


def test_scaled():
    """Pulse should be scaled by a given parameter."""
    pulse = Pulse([0.1, 0.2, 0.3])
    scaled = pulse.scaled(2)
    assert scaled.values == pytest.approx([0.2, 0.4, 0.6])


def test_detuned():
    """Pulse should be detuned by a given parameter."""
    pulse = Pulse([0.1, 0.1, 0.1])
    detuned = pulse.detuned(0.001)
    assert detuned != pulse
    assert detuned.values == pytest.approx(
        [
            0.1,
            0.1 * np.exp(1j * 0.001 * 2 * np.pi * dt),
            0.1 * np.exp(2j * 0.001 * 2 * np.pi * dt),
        ]
    )


def test_shifted():
    """Pulse should be phase shifted by a given parameter."""
    pulse = Pulse([1, -1, 1j])
    shifted = pulse.shifted(np.pi / 2)
    assert shifted != pulse
    assert shifted.values == pytest.approx([1j, -1j, -1])


def test_repeated():
    """Pulse should be repeated a given number of times."""
    pulse = Pulse([1, 2, 3])
    repeated = pulse.repeated(3)
    assert repeated != pulse
    assert repeated.values == pytest.approx([1, 2, 3, 1, 2, 3, 1, 2, 3])
