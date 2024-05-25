import numpy as np
import pytest
from qubex.pulse import Gauss, Pulse

dt = Pulse.SAMPLING_PERIOD


def test_inheritance():
    """Gauss should inherit from Pulse."""
    assert issubclass(Gauss, Pulse)


def test_empty_init():
    """Gauss should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Gauss()  # type: ignore


def test_init():
    """Gauss should be initialized with valid parameters."""
    pulse = Gauss(duration=5 * dt, amplitude=1, sigma=2 * dt)
    assert pulse.duration == 5 * dt
    assert pulse.values == pytest.approx(
        [
            0.60653066 + 0.0j,
            0.8824969 + 0.0j,
            1.0 + 0.0j,
            0.8824969 + 0.0j,
            0.60653066 + 0.0j,
        ]
    )


def test_zero_duration():
    """Gauss should be initialized with zero duration."""
    pulse = Gauss(duration=0, amplitude=1, sigma=1)
    assert pulse.duration == 0
    assert (pulse.values == np.array([], dtype=np.complex128)).all()


def test_invalid_parameter():
    """Gauss should raise a ValueError if sigma is zero."""
    with pytest.raises(ValueError):
        Gauss(duration=5 * dt, amplitude=1, sigma=0)
