import numpy as np
import numpy.typing as npt

from .pulse import Pulse


class Blank(Pulse):
    """
    A class to represent a blank pulse.

    Parameters
    ----------
    duration : float
        Duration of the blank pulse in ns.

    Examples
    --------
    >>> pulse = Blank(duration=100)
    """

    def __init__(
        self,
        duration: float,
    ):
        N = self._number_of_samples(duration)
        real = np.zeros(N, dtype=np.float64)
        imag = 0
        values = real + 1j * imag
        super().__init__(values)


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
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude)
        super().__init__(values)

    def _calc_values(
        self,
        duration: float,
        amplitude: float,
    ) -> npt.NDArray[np.complex128]:
        N = self._number_of_samples(duration)
        real = amplitude * np.ones(N)
        imag = 0
        values = real + 1j * imag
        return values


class FlatTop(Pulse):
    """
    A class to represent a raised cosine flat-top pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    tau : int
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
    |        ________________________
    |       /                        \
    |      /                          \
    |     /                            \
    |    /                              \
    |___                                 _______
    |   <---->                      <---->
    |     tau                        tau
    |   <-------------------------------->
    |                duration
    | 
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        tau: int,
        **kwargs,
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, tau)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: float,
        amplitude: float,
        tau: int,
    ) -> npt.NDArray[np.complex128]:
        flattime = duration - 2 * tau

        if flattime < 0:
            raise ValueError("duration must be greater than `2 * tau`.")

        t_rise = self._sampling_points(tau)
        t_flat = self._sampling_points(flattime)

        v_rise = 0.5 * amplitude * (1 - np.cos(np.pi * t_rise / tau))
        v_flat = amplitude * np.ones_like(t_flat)
        v_fall = 0.5 * amplitude * (1 + np.cos(np.pi * t_rise / tau))

        values = np.concatenate((v_rise, v_flat, v_fall)).astype(np.complex128)

        return values


class Gauss(Pulse):
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

    Examples
    --------
    >>> pulse = Gauss(duration=100, amplitude=1.0, sigma=10)
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        sigma: float,
        **kwargs,
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, sigma)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: float,
        amplitude: float,
        sigma: float,
    ) -> npt.NDArray[np.complex128]:
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        t = self._sampling_points(duration)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = 0
        values = real + 1j * imag
        return values


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
        The correction amplitude.

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
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, beta)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: float,
        amplitude: float,
        beta: float,
    ) -> npt.NDArray[np.complex128]:
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
        return values


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
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, sigma, beta)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: float,
        amplitude: float,
        sigma: float,
        beta: float,
    ) -> npt.NDArray[np.complex128]:
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        t = self._sampling_points(duration)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = (mu - t) / (sigma**2) * real
        values = real + beta * 1j * imag
        return values


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
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, beta)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: float,
        amplitude: float,
        beta: float,
    ) -> npt.NDArray[np.complex128]:
        t = self._sampling_points(duration)
        real = amplitude * (1.0 - np.cos(2 * np.pi * t / duration)) * 0.5
        imag = 2 * np.pi / duration * amplitude * np.sin(2 * np.pi * t / duration) * 0.5
        values = real + beta * 1j * imag
        return values
