"""Gaussian pulse shape helpers."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import override

from qxpulse.pulse import Pulse


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
        DRAG correction coefficient. Default is None.

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
        beta: float | None = None,
        **kwargs,
    ):
        super().__init__(
            duration=duration,
            **kwargs,
        )

        self.amplitude: Final = amplitude
        self.sigma: Final = sigma
        self.zero_bounds: Final = zero_bounds
        self.beta: Final = beta

        if self.length > 0 and sigma is not None and sigma <= 0:
            raise ValueError("Sigma must be greater than zero.")

        self._finalize_initialization()

    @override
    def _sample_values(self) -> NDArray[np.complex128]:
        """Return sampled values for the Gaussian pulse."""
        if self.length == 0:
            return np.array([], dtype=np.complex128)
        duration = self.duration
        return Gaussian.func(
            t=self._sampling_points(duration),
            duration=duration,
            amplitude=self.amplitude,
            sigma=self.sigma,
            zero_bounds=self.zero_bounds,
            beta=self.beta,
        )

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        sigma: float | None = None,
        zero_bounds: bool = True,
        beta: float | None = None,
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
            DRAG correction coefficient. Default is None.

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
        if beta is None:
            values = Omega
        else:
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
