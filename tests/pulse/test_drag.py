import numpy as np
import pytest
from qubex.pulse import Drag, Pulse

dt = Pulse.SAMPLING_PERIOD


def test_inheritance():
    """Drag should inherit from Pulse."""
    assert issubclass(Drag, Pulse)


def test_empty_init():
    """Drag should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Drag()  # type: ignore


def test_init():
    """Drag should be initialized with valid parameters."""
    pulse = Drag(duration=5 * dt, amplitude=1, beta=1)
    assert pulse.duration == 5 * dt
    assert pulse.values == pytest.approx(
        [
            0.3040094 + 0.29528056j,
            0.80460065 + 0.18768758j,
            1.0 + 0.0j,
            0.80460065 - 0.18768758j,
            0.3040094 - 0.29528056j,
        ]
    )


def test_zero_duration():
    """Drag should be initialized with zero duration."""
    pulse = Drag(duration=0, amplitude=1, beta=1)
    assert pulse.duration == 0
    assert (pulse.values == np.array([], dtype=np.complex128)).all()
