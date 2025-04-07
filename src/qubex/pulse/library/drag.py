from __future__ import annotations

from typing import Final, Literal

import numpy as np

from ..pulse import Pulse


class Drag(Pulse):
    """
    A class to represent a DRAG pulse.

    Parameters
    ----------
    duration : float
        Duration of the DRAG pulse in ns.
    amplitude : float
        Amplitude of the DRAG pulse.
    beta : float
        DRAG correction coefficient.
    type : Literal["gaussian", "raised_cosine"], optional
        Type of the pulse. Default is "gaussian".

    Examples
    --------
    >>> pulse = Drag(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     beta=1.0,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        beta: float,
        type: Literal["gaussian", "raised_cosine"] = "gaussian",
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.beta: Final = beta

        t = self._sampling_points(duration)

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        elif type == "gaussian":
            mu = duration * 0.5
            sigma = mu / 2
            offset = -np.exp(-0.5 * (mu / sigma) ** 2)
            factor = amplitude / (1 + offset)
            Omega = factor * (np.exp(-((t - mu) ** 2) / (2 * sigma**2)) + offset)
            dOmega = (
                (mu - t)
                / (sigma**2)
                * (factor * (np.exp(-((t - mu) ** 2) / (2 * sigma**2))))
            )
            values = Omega + beta * 1j * dOmega
        elif type == "raised_cosine":
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
        else:
            raise ValueError(f"Unknown pulse type: {type}")

        super().__init__(values, **kwargs)
