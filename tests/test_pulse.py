# pylint: disable=all
import numpy as np
import pytest


def test_rect():
    from qubex.pulse import Rect

    rect = Rect(
        duration=10,
        amplitude=0.1,
    )
    assert rect.values.shape == (5,)
    assert rect.values.dtype == np.complex128
    assert rect.values[0] == 0.1

    with pytest.raises(ValueError):
        Rect(
            duration=11,
            amplitude=0.1,
        )
