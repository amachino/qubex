from __future__ import annotations

from typing import Final

import numpy as np

from ..pulse import Pulse


class Rect(Pulse):
    """
    A class to represent a rectangular pulse.

    Parameters
    ----------
    duration : float
        Duration of the rectangular pulse in ns.
    amplitude : float
        Amplitude of the rectangular pulse.

    Examples
    --------
    >>> pulse = Rect(duration=100, amplitude=0.1)
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        **kwargs,
    ):
        self.amplitude: Final = amplitude

        N = self._number_of_samples(duration)
        real = amplitude * np.ones(N)
        imag = 0
        values = real + 1j * imag

        super().__init__(values, **kwargs)
