from __future__ import annotations

from typing import Final

import numpy as np

from ..pulse import Pulse


class Gaussian(Pulse):
    """
    A class to represent a Gaussian pulse.

    Parameters
    ----------
    duration : float
        Duration of the Gaussian pulse in ns.
    amplitude : float
        Amplitude of the Gaussian pulse.
    sigma : float
        Standard deviation of the Gaussian pulse in ns.
    beta : float, optional
        DRAG correction amplitude.

    Examples
    --------
    >>> pulse = Gaussian(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     sigma=10,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        sigma: float,
        beta: float = 0.0,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.sigma: Final = sigma
        self.beta: Final = beta

        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        t = self._sampling_points(duration)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = (mu - t) / (sigma**2) * real
        values = real + beta * 1j * imag

        super().__init__(values, **kwargs)
