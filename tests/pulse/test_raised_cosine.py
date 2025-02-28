import numpy as np
import pytest

import qubex as qx
from qubex.pulse import Pulse, RaisedCosine

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """RaisedCosine should inherit from Pulse."""
    assert issubclass(RaisedCosine, Pulse)


def test_empty_init():
    """RaisedCosine should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        RaisedCosine()  # type: ignore


def test_init():
    """RaisedCosine should be initialized with valid parameters."""
    pulse = RaisedCosine(duration=5 * dt, amplitude=1, beta=1)
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
    """RaisedCosine should be initialized with zero duration."""
    pulse = RaisedCosine(duration=0, amplitude=1, beta=1)
    assert pulse.duration == 0
    assert (pulse.values == np.array([], dtype=np.complex128)).all()
