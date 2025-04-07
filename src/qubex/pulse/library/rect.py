from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

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

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
    ) -> NDArray:
        """
        Rectangular pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the rectangular pulse in ns.
        amplitude : float
            Amplitude of the rectangular pulse.

        Returns
        -------
        NDArray
            Rectangular pulse values.
        """
        t = np.asarray(t)
        return np.where(
            (t >= 0) & (t <= duration),
            amplitude,
            0,
        ).astype(np.complex128)
