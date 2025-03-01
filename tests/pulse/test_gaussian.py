import pytest

import qubex as qx
from qubex.pulse import Gaussian, Pulse

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """Gaussian should inherit from Pulse."""
    assert issubclass(Gaussian, Pulse)


def test_empty_init():
    """Gaussian should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Gaussian()  # type: ignore


def test_init():
    """Gaussian should be initialized with valid parameters."""
    pulse = Gaussian(duration=5 * dt, amplitude=1, sigma=2 * dt, beta=1)
    assert pulse.name == "Gaussian"
    assert pulse.length == 5
    assert pulse.duration == 5 * dt
    assert pulse.amplitude == 1
    assert pulse.sigma == 2 * dt
    assert pulse.beta == 1
    assert pulse.values == pytest.approx(
        [
            0.60653066 + 0.15163266j,
            0.8824969 + 0.11031211j,
            1.0 + 0.0j,
            0.8824969 - 0.11031211j,
            0.60653066 - 0.15163266j,
        ]
    )


def test_zero_duration():
    """Gaussian should be initialized with zero duration."""
    pulse = Gaussian(duration=0, amplitude=1, sigma=1, beta=1)
    assert pulse.duration == 0
    assert pulse.values == pytest.approx([])


def test_invalid_parameter():
    """Gaussian should raise a ValueError if sigma is zero."""
    with pytest.raises(ValueError):
        Gaussian(duration=5 * dt, amplitude=1, sigma=0, beta=1)
