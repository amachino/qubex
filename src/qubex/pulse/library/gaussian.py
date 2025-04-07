from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    zero_bounds : bool, optional
        If True, the pulse is truncated to have zero bounds.
    beta : float, optional
        DRAG correction coefficient. Default is 0.0.

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
        zero_bounds: bool = True,
        beta: float = 0.0,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.sigma: Final = sigma
        self.zero_bounds: Final = zero_bounds
        self.beta: Final = beta

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                sigma=sigma,
                zero_bounds=zero_bounds,
                beta=beta,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        sigma: float | None = None,
        zero_bounds: bool = True,
        beta: float = 0.0,
    ) -> NDArray:
        """
        Gaussian pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the Gaussian pulse in ns.
        amplitude : float
            Amplitude of the Gaussian pulse.
        sigma : float, optional
            Standard deviation of the Gaussian pulse. If None, it is set to duration / 2.
        zero_bounds : bool, optional
            If True, the pulse is truncated to have zero bounds.
        beta : float, optional
            DRAG correction coefficient. Default is 0.0.

        Returns
        -------
        NDArray
            Gaussian pulse values.
        """
        t = np.asarray(t)
        mu = duration * 0.5
        if sigma is None:
            sigma = mu / 2
        if sigma <= 0:
            raise ValueError("Sigma must be greater than zero.")
        offset = -np.exp(-0.5 * (mu / sigma) ** 2) if zero_bounds else 0.0
        factor = amplitude / (1 + offset)
        Omega = factor * (np.exp(-((t - mu) ** 2) / (2 * sigma**2)) + offset)
        dOmega = (
            (mu - t)
            / (sigma**2)
            * (factor * (np.exp(-((t - mu) ** 2) / (2 * sigma**2))))
        )
        values = Omega + beta * 1j * dOmega
        return np.where(
            (t >= 0) & (t <= duration),
            values,
            0,
        ).astype(np.complex128)
