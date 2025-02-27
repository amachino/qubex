import numpy as np
import pytest

from qubex.pulse import PhaseShift, Pulse, PulseArray

dt = PulseArray.SAMPLING_PERIOD


def test_empty_init():
    """PhaseShift should raise a TypeError if no parameters are provided."""
    with pytest.raises(TypeError):
        PhaseShift()  # type: ignore


def test_init():
    """PhaseShift should be initialized with valid parameters."""
    phase_shift = PhaseShift(np.pi / 2)
    assert phase_shift.theta == pytest.approx(np.pi / 2)


def test_phase_shift():
    """PhaseShift should be repeated a given number of times."""
    seq = PulseArray(
        [
            Pulse([1]),
            PhaseShift(np.pi / 2),
            Pulse([1]),
            PhaseShift(np.pi / 2),
            Pulse([1]),
            PhaseShift(np.pi),
            Pulse([1]),
            Pulse([1]),
        ],
    )
    assert seq.values == pytest.approx([1, 1j, -1, 1, 1])
    assert seq.reversed().values == pytest.approx([1, 1, -1, 1j, 1])
