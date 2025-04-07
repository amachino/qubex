from __future__ import annotations

from typing import Final

import numpy as np

from ..pulse import Pulse


class RaisedCosine(Pulse):
    """
    A class to represent a raised cosine pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    beta : float, optional
        DRAG correction coefficient. Default is 0.0.

    Examples
    --------
    >>> pulse = RaisedCosine(
    ...     duration=100,
    ...     amplitude=1.0,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        beta: float = 0.0,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.beta: Final = beta

        t = self._sampling_points(duration)

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            Omega = amplitude * (1.0 - np.cos(2 * np.pi * t / duration)) * 0.5
            dOmega = (
                2
                * np.pi
                / duration
                * amplitude
                * np.sin(2 * np.pi * t / duration)
                * 0.5
            )
            values = Omega + beta * 1j * dOmega

        super().__init__(values, **kwargs)
