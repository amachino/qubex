"""Rectangular pulse shape helpers."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import override

from qxpulse.pulse import Pulse


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
        super().__init__(
            duration=duration,
            **kwargs,
        )

        self.amplitude: Final = amplitude
        self._finalize_initialization()

    @override
    def _sample_values(self) -> NDArray[np.complex128]:
        """Return sampled values for the rectangular pulse."""
        if self.length == 0:
            return np.array([], dtype=np.complex128)
        duration = self.duration
        return Rect.func(
            t=self._sampling_points(duration),
            duration=duration,
            amplitude=self.amplitude,
        )

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
