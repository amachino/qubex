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
    sigma : float, optional
        Standard deviation of the Gaussian pulse. If None, it is set to duration / 2.
    beta : float, optional
        DRAG correction coefficient. Default is 0.0.
    zero_bounds : bool, optional
        If True, the pulse is truncated to have zero bounds.

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
        sigma: float | None = None,
        beta: float = 0.0,
        zero_bounds: bool = True,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.sigma: Final = sigma
        self.beta: Final = beta

        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        t = self._sampling_points(duration)
        mu = duration * 0.5
        if sigma is None:
            sigma = mu / 2
        offset = -np.exp(-0.5 * (mu / sigma) ** 2) if zero_bounds else 0.0
        factor = amplitude / (1 + offset)
        Omega = factor * (np.exp(-((t - mu) ** 2) / (2 * sigma**2)) + offset)
        dOmega = (
            (mu - t)
            / (sigma**2)
            * (factor * (np.exp(-((t - mu) ** 2) / (2 * sigma**2))))
        )
        values = Omega + beta * 1j * dOmega

        super().__init__(values, **kwargs)
