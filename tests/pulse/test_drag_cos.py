import numpy as np
import pytest
from qubex.pulse import DragCos, Pulse

dt = Pulse.SAMPLING_PERIOD


def test_inheritance():
    """DragCos should inherit from Pulse."""
    assert issubclass(DragCos, Pulse)


def test_empty_init():
    """DragCos should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        DragCos()  # type: ignore


def test_init():
    """DragCos should be initialized with valid parameters."""
    pulse = DragCos(duration=5 * dt, amplitude=1, beta=1)
    assert pulse.duration == 5 * dt
    assert pulse.values == pytest.approx(
        [
            0.0954915 + 1.84658183e-01j,
            0.6545085 + 2.98783216e-01j,
            1.0 + 3.84734139e-17j,
            0.6545085 - 2.98783216e-01j,
            0.0954915 - 1.84658183e-01j,
        ]
    )


def test_zero_duration():
    """DragCos should be initialized with zero duration."""
    pulse = DragCos(duration=0, amplitude=1, beta=1)
    assert pulse.duration == 0
    assert (pulse.values == np.array([], dtype=np.complex128)).all()
