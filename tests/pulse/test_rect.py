import numpy as np
import pytest

import qubex as qx
from qubex.pulse import Pulse, Rect

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """Rect should inherit from Pulse."""
    assert issubclass(Rect, Pulse)


def test_empty_init():
    """Rect should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Rect()  # type: ignore


def test_init():
    """Rect should be initialized with valid parameters."""
    pulse = Rect(duration=5 * dt, amplitude=0.1)
    assert pulse.duration == 5 * dt
    assert (pulse.values == [0.1, 0.1, 0.1, 0.1, 0.1]).all()


def test_zero_duration():
    """Rect should be initialized with zero duration."""
    pulse = Rect(duration=0, amplitude=0.1)
    assert pulse.duration == 0
    assert (pulse.values == np.array([], dtype=np.complex128)).all()


def test_invalid_duration():
    """Rect should raise a ValueError if duration is not a multiple of the sampling period."""
    with pytest.raises(ValueError):
        Rect(duration=5 * dt + np.pi, amplitude=0.1)
