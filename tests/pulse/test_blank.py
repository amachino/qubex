import numpy as np
import pytest
from qubex.pulse import Blank, Pulse

dt = Blank.SAMPLING_PERIOD


def test_inheritance():
    """Blank should inherit from Pulse."""
    assert issubclass(Blank, Pulse)


def test_empty_init():
    """Blank should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Blank()  # type: ignore


def test_init():
    """Blank should be initialized with a duration."""
    pulse = Blank(duration=5 * dt)
    assert pulse.duration == 5 * dt
    assert (pulse.values == [0, 0, 0, 0, 0]).all()


def test_invalid_duration():
    """Blank should raise a ValueError if duration is not a multiple of the sampling period."""
    with pytest.raises(ValueError):
        Blank(duration=5 * dt + np.pi)
