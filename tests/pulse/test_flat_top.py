import numpy as np
import pytest

import qubex as qx
from qubex.pulse import FlatTop, Pulse

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """FlatTop should inherit from Pulse."""
    assert issubclass(FlatTop, Pulse)


def test_empty_init():
    """FlatTop should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        FlatTop()  # type: ignore


def test_init():
    """FlatTop should be initialized with valid parameters."""
    pulse = FlatTop(duration=5 * dt, amplitude=1, tau=2 * dt)
    assert pulse.duration == 5 * dt
    assert pulse.values == pytest.approx(
        [
            0.14644661 + 0.0j,
            0.85355339 + 0.0j,
            1.0 + 0.0j,
            0.85355339 + 0.0j,
            0.14644661 + 0.0j,
        ]
    )


def test_zero_duration():
    """FlatTop should be initialized with zero duration."""
    pulse = FlatTop(duration=0, amplitude=1, tau=0)
    assert pulse.duration == 0
    assert (pulse.values == np.array([], dtype=np.complex128)).all()


def test_invalid_duration():
    """FlatTop should raise a ValueError if duration is not greater than `2 * tau`."""
    with pytest.raises(ValueError):
        FlatTop(duration=5 * dt, amplitude=1, tau=3 * dt)
