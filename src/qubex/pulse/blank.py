from __future__ import annotations

import numpy as np

from .pulse import Pulse


class Blank(Pulse):
    """
    A class to represent a blank pulse.

    Parameters
    ----------
    duration : float
        Duration of the blank pulse in ns.

    Examples
    --------
    >>> pulse = Blank(duration=100)
    """

    def __init__(
        self,
        duration: float,
    ):
        N = self._number_of_samples(duration)
        real = np.zeros(N, dtype=np.float64)
        imag = 0
        values = real + 1j * imag

        super().__init__(values)
