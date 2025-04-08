from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..pulse import Pulse


class Bump(Pulse):
    """
    A class to represent a bump (smooth compactly supported) pulse.

    Parameters
    ----------
    duration : float
        Duration of the bump pulse in ns.
    amplitude : float
        Amplitude of the bump pulse.
    beta : float or None, optional
        DRAG coefficient. If None, no imaginary component is added.

    Examples
    --------
    >>> pulse = Bump(
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
        Bump pulse function using a smooth compact support function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the bump pulse in ns.
        amplitude : float
            Amplitude of the bump pulse.
        beta : float or None
            DRAG coefficient. If None, no imaginary component is added.

        Returns
        -------
        NDArray
            Bump pulse values.
        """
        t = np.asarray(t)
        mu = duration * 0.5
        u = 2 * (t - mu) / duration
        Omega = np.exp(1) * np.exp(-1 / (1 - u**2))

        dOmega = np.zeros_like(t)
        if beta is None:
            values = amplitude * Omega
        else:
            dOmega = -4 * u / duration / (1 - u**2) ** 2 * Omega
            values = amplitude * (Omega + 1j * beta * dOmega)

        return np.where(
            (t >= 0) & (t <= duration),
            values,
            0,
        ).astype(np.complex128)
