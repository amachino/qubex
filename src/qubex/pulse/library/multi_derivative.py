from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..pulse import Pulse
from .sintegral import MultiDerivativeSintegral


class MultiDerivative(Pulse):
    """
    A class to represent a DRAG pulse.

    Parameters
    ----------
    duration : float
        Duration of the DRAG pulse in ns.
    amplitude : float
        Amplitude of the DRAG pulse.

    betas : dict[int, float] | None
        multi-Derivative pulse correction coefficients.
    power : int
        Power of the sine integral function.

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
        betas: dict[int, float] | None = None,
        power: int = 2,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.betas: Final = betas
        self.power: Final = power

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                betas=betas,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        betas: dict[int, float] | None = None,
        power: int = 2,
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
        betas : dict[int, float] | None
            multi-Derivative pulse correction coefficients.
        power : int
            Power of the sine integral function.
        """

        return MultiDerivativeSintegral.func(
            t=t,
            duration=duration,
            amplitude=amplitude,
            power=power,
            betas=betas,
        )
