from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
        DRAG correction coefficient. Default is None.

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
        beta: float | None = None,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.beta: Final = beta

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                beta=beta,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        beta: float | None = None,
    ) -> NDArray:
        """
        Raised cosine pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the pulse in ns.
        amplitude : float
            Amplitude of the pulse.
        beta : float, optional
            DRAG correction coefficient. Default is None.
        """
        t = np.asarray(t)

        if duration == 0:
            return np.zeros_like(t, dtype=np.complex128)

        Omega = amplitude * (1.0 - np.cos(2 * np.pi * t / duration)) * 0.5
        if beta is None:
            values = Omega
        else:
            dOmega = np.pi / duration * amplitude * np.sin(2 * np.pi * t / duration)
            values = Omega + beta * 1j * dOmega
        return np.where(
            (t >= 0) & (t <= duration),
            values,
            0,
        ).astype(np.complex128)
