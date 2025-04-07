from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..pulse import Pulse
from .gaussian import Gaussian
from .raised_cosine import RaisedCosine
from .sintegral import Sintegral


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
        type: Literal["gaussian", "raised_cosine", "sintegral"] = "gaussian",
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.beta: Final = beta
        self.type: Final = type

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
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
        beta: float,
        type: Literal["gaussian", "raised_cosine", "sintegral"] = "gaussian",
    ) -> NDArray:
        """
        DRAG pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the DRAG pulse in ns.
        amplitude : float
            Amplitude of the DRAG pulse.
        beta : float
            DRAG correction coefficient.
        type : Literal["gaussian", "raised_cosine"], optional
            Type of the pulse. Default is "gaussian".
        """
        if type == "gaussian":
            return Gaussian.func(
                t=t,
                duration=duration,
                amplitude=amplitude,
                sigma=duration / 4,
                zero_bounds=True,
                beta=beta,
            )
        elif type == "raised_cosine":
            return RaisedCosine.func(
                t=t,
                duration=duration,
                amplitude=amplitude,
                beta=beta,
            )
        elif type == "sintegral":
            return Sintegral.func(
                t=t,
                duration=duration,
                amplitude=amplitude,
                power=2,
                beta=beta,
            )
        else:
            raise ValueError(f"Unknown pulse type: {type}")
