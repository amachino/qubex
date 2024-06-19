import numpy as np
import pytest
from qubex.pulse import PhaseShift, Pulse, PulseSequence

dt = PulseSequence.SAMPLING_PERIOD


def test_empty_init():
    """PhaseShift should raise a TypeError if no parameters are provided."""
    with pytest.raises(TypeError):
        PulseSequence()  # type: ignore


def test_init():
    """PhaseShift should be initialized with valid parameters."""
    phase_shift = PhaseShift(np.pi / 2)
    assert phase_shift.theta == pytest.approx(np.pi / 2)


def test_phase_shift():
    """PhaseShift should be repeated a given number of times."""
    seq = PulseSequence(
        [
            Pulse([1, 2, 3]),
            PhaseShift(np.pi),
            Pulse([1, 2, 3]),
            Pulse([1, 2, 3]),
        ],
    )
    assert seq.values == pytest.approx([1, 2, 3, -1, -2, -3, -1, -2, -3])
