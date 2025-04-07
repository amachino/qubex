from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..pulse import Pulse
from .drag import Drag


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
    beta : float, optional
        DRAG correction coefficient. Default is 0.0.

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
        beta: float = 0.0,
        type: Literal["gaussian", "raised_cosine"] = "gaussian",
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.tau: Final = tau

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                tau=tau,
                beta=beta,
                type=type,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        tau: float,
        beta: float = 0.0,
        type: Literal["gaussian", "raised_cosine"] = "gaussian",
    ) -> NDArray:
        """
        Flat-top pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the pulse in ns.
        amplitude : float
            Amplitude of the pulse.
        tau : float
            Rise and fall time of the pulse in ns.
        beta : float, optional
            DRAG correction coefficient. Default is 0.0.
        type : Literal["gaussian", "raised_cosine"], optional
            Type of the pulse. Default is "gaussian".

        Returns
        -------
        NDArray
            Flat-top pulse values.
        """
        t = np.asarray(t)
        flattime = duration - 2 * tau

        if flattime < 0:
            raise ValueError("duration must be greater than `2 * tau`.")

        v_rise = Drag.func(
            t=t,
            duration=2 * tau,
            amplitude=amplitude,
            beta=beta,
            type=type,
        )
        v_flat = amplitude * np.ones_like(t)
        v_fall = Drag.func(
            t=t - flattime,
            duration=2 * tau,
            amplitude=amplitude,
            beta=beta,
            type=type,
        )

        return np.where(
            (t >= 0) & (t <= duration),
            np.where(
                (t >= tau) & (t <= duration - tau),
                v_flat,
                np.where(
                    (t < tau),
                    v_rise,
                    v_fall,
                ),
            ),
            0,
        ).astype(np.complex128)
