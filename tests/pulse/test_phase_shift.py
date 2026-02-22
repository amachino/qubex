"""Tests for the PhaseShift pulse."""

import numpy as np
import pytest

import qubex as qx
from qubex.pulse import Arbitrary, PhaseShift, PulseArray

dt = qx.pulse.get_sampling_period()


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
            Arbitrary([1]),
            PhaseShift(np.pi / 2),
            Arbitrary([1]),
            PhaseShift(np.pi / 2),
            Arbitrary([1]),
            PhaseShift(np.pi),
            Arbitrary([1]),
            Arbitrary([1]),
        ],
    )
    assert seq.values == pytest.approx([1, 1j, -1, 1, 1])


def test_nested():
    """PhaseShift should be used in a nested PulseArray."""
    seq = PulseArray(
        [
            PulseArray(
                [
                    Arbitrary([1]),
                    PhaseShift(np.pi / 2),
                    Arbitrary([1]),
                ],
            ),
            PhaseShift(np.pi / 2),
            PulseArray(
                [
                    Arbitrary([1]),
                    PhaseShift(np.pi / 2),
                    Arbitrary([1]),
                ],
            ),
            Arbitrary([1]),
        ],
    )
    assert seq.values == pytest.approx([1, 1j, -1, -1j, -1j])
