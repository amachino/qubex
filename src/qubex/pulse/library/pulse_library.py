from __future__ import annotations

import numpy as np
from typing_extensions import deprecated

from ..pulse import Pulse


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
        N = self._number_of_samples(duration)
        real = amplitude * np.ones(N)
        imag = 0
        values = real + 1j * imag

        super().__init__(values, **kwargs)


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
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        t = self._sampling_points(duration)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = (mu - t) / (sigma**2) * real
        values = real + beta * 1j * imag

        super().__init__(values, **kwargs)


@deprecated("Use `Gaussian` instead.")
class Gauss(Gaussian):
    pass


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
        DRAG correction amplitude.

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
        **kwargs,
    ):
        t = self._sampling_points(duration)
        sigma = duration * 0.5
        offset = -np.exp(-0.5)
        factor = amplitude / (1 + offset)
        real = factor * (np.exp(-((t - sigma) ** 2) / (2 * sigma**2)) + offset)
        imag = (
            (sigma - t)
            / (sigma**2)
            * (factor * (np.exp(-((t - sigma) ** 2) / (2 * sigma**2))))
        )
        values = real + beta * 1j * imag

        super().__init__(values, **kwargs)


@deprecated("Use `Gaussian` instead.")
class DragGauss(Pulse):
    """
    A class to represent a DRAG Gaussian pulse.

    Parameters
    ----------
    duration : float
        Duration of the DRAG Gaussian pulse in ns.
    amplitude : float
        Amplitude of the DRAG Gaussian pulse.
    sigma : float
        Standard deviation of the DRAG Gaussian pulse in ns.
    beta : float
        The correction amplitude.

    Examples
    --------
    >>> pulse = DragGauss(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     sigma=10,
    ...     beta=0.1,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        sigma: float,
        beta: float,
        **kwargs,
    ):
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        t = self._sampling_points(duration)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = (mu - t) / (sigma**2) * real
        values = real + beta * 1j * imag

        super().__init__(values, **kwargs)


class RaisedCosine(Pulse):
    """
    A class to represent a raised cosine pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    beta : float, optional
        DRAG correction amplitude.

    Examples
    --------
    >>> pulse = RaisedCosine(
    ...     duration=100,
    ...     amplitude=1.0,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        beta: float = 0.0,
        **kwargs,
    ):
        t = self._sampling_points(duration)

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            real = amplitude * (1.0 - np.cos(2 * np.pi * t / duration)) * 0.5
            imag = (
                2
                * np.pi
                / duration
                * amplitude
                * np.sin(2 * np.pi * t / duration)
                * 0.5
            )
            values = real + beta * 1j * imag

        super().__init__(values, **kwargs)


@deprecated("Use `RaisedCosine` instead.")
class DragCos(Pulse):
    """
    A class to represent a DRAG cosine pulse.

    Parameters
    ----------
    duration : float
        Duration of the DRAG cosine pulse in ns.
    amplitude : float
        Amplitude of the DRAG cosine pulse.
    beta : float
        The correction amplitude.

    Examples
    --------
    >>> pulse = DragCos(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     beta=0.1,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        beta: float,
        **kwargs,
    ):
        t = self._sampling_points(duration)

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            real = amplitude * (1.0 - np.cos(2 * np.pi * t / duration)) * 0.5
            imag = (
                2
                * np.pi
                / duration
                * amplitude
                * np.sin(2 * np.pi * t / duration)
                * 0.5
            )
            values = real + beta * 1j * imag

        super().__init__(values, **kwargs)


@deprecated("Will be removed.")
class TabuchiDD(Pulse):
    """
    Class representing the Tabuchi Dynamical Decoupling pulse sequence.

    Parameters
    ----------
    duration : float
        The total duration of the pulse sequence in nanoseconds.
    beta : float, optional
        Beta parameter influencing the x and y components of the pulse.
    phi : float, optional
        Phi parameter influencing the x component of the pulse.
    **kwargs
        Additional keyword arguments passed to the Pulse constructor.

    Attributes
    ----------
    vx_n_T_over_pi : list[float]
        Coefficients for the x component of the pulse.
    vy_n_T_over_pi : list[float]
        Coefficients for the y component of the pulse.
    t : np.ndarray
        Time points for the pulse sequence.
    T : int
        Total duration of the pulse sequence in nanoseconds.
    vx_n : np.ndarray
        Scaled coefficients for the x component.
    vy_n : np.ndarray
        Scaled coefficients for the y component.

    Notes
    -----
    Y. Tabuchi, M. Negoro, and M. Kitagawa, “Design method of dynamical
    decouplingsequences integrated with optimal control theory,” Phys.
    Rev. A, vol.96, p.022331, Aug. 2017.
    """

    vx_n_T_over_pi = [
        -0.7030256,
        3.3281747,
        11.390077,
        2.9375301,
        -1.8758792,
        1.7478474,
        5.6966577,
        -0.5452435,
        4.0826786,
    ]

    vy_n_T_over_pi = [
        -3.6201768,
        3.8753985,
        -1.2311919,
        -0.2998110,
        3.1170274,
        0.3956137,
        -0.3593987,
        -3.5266063,
        2.4900307,
    ]

    def __init__(
        self,
        duration: float,
        beta=0.0,
        phi=0.0,
        **kwargs,
    ):
        self.t = self._sampling_points(duration)
        self.T = duration  # [ns]
        values = np.array([])  # [MHz]
        if duration != 0:
            self.vx_n = np.array(self.vx_n_T_over_pi) * np.pi / duration
            self.vy_n = np.array(self.vy_n_T_over_pi) * np.pi / duration
            values = self._calc_values(beta, phi)
        super().__init__(values, **kwargs)

    def _calc_values(self, beta: float, phi: float) -> np.ndarray:
        error_x = beta + np.tan(phi * np.pi / 180)
        x = (1 + error_x) * np.array([self._vx(t) for t in self.t])

        error_y = beta
        y = (1 + error_y) * np.array([self._vy(t) for t in self.t])

        values = (x + 1j * y) / np.pi / 2 * 1e3
        return values

    def _vx(self, t) -> float:
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vx_n, 1)
        )

    def _vy(self, t) -> float:
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vy_n, 1)
        )
