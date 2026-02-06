"""Vertical ramp pulse shape helpers."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import deprecated

from qxpulse.pulse import Pulse


@deprecated(
    "The 'VertRamp' class is deprecated and will be removed in a future release."
)
class VertRamp(Pulse):
    """
    A class to represent a vertical ramp pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.

    Examples
    --------
    >>> pulse = VertRamp(
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
        super().__init__(**kwargs)

        self.amplitude: Final = amplitude
        self.beta: Final = beta

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
            )
        self._values = np.array(values, dtype=np.complex128)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
    ) -> NDArray:
        """
        Vertical ramp pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the pulse in ns.
        amplitude : float
            Amplitude of the pulse.
        """
        t = np.asarray(t)

        if duration == 0:
            return np.zeros_like(t, dtype=np.complex128)

        # Compute only on the valid domain to avoid sqrt of negative numbers
        mask = (t >= 0) & (t <= duration)
        out = np.zeros_like(t, dtype=np.float64)
        if np.any(mask):
            u = t[mask] / duration
            # Clip for numerical stability at the boundary (e.g., u ~ 1 +/- eps)
            arg = np.clip(1.0 - u**2, 0.0, 1.0)
            out[mask] = amplitude * (1.0 - np.sqrt(arg))

        return out.astype(np.complex128)
