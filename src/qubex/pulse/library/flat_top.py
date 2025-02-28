from __future__ import annotations

from typing import Final

import numpy as np

from ..pulse import Pulse


class FlatTop(Pulse):
    """
    A class to represent a raised cosine flat-top pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    tau : float
        Rise and fall time of the pulse in ns.

    Examples
    --------
    >>> pulse = FlatTop(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     tau=10,
    ... )

    Notes
    -----
    flat-top period = duration - 2 * tau
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        tau: float,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.tau: Final = tau

        if duration == 0:
            super().__init__([], **kwargs)
            return

        flattime = duration - 2 * tau

        if flattime < 0:
            raise ValueError("duration must be greater than `2 * tau`.")

        t_rise = self._sampling_points(tau)
        t_flat = self._sampling_points(flattime)

        v_rise = 0.5 * amplitude * (1 - np.cos(np.pi * t_rise / tau))
        v_flat = amplitude * np.ones_like(t_flat)
        v_fall = 0.5 * amplitude * (1 + np.cos(np.pi * t_rise / tau))

        values = np.concatenate((v_rise, v_flat, v_fall)).astype(np.complex128)

        super().__init__(values, **kwargs)
