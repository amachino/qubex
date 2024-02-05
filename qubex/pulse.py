"""
Provides classes to represent waveforms. A waveform is a sequence of complex
numbers representing I and Q values of a pulse. The waveform can be plotted
in the time domain as a function of time or in the polar domain as a function
of amplitude and phase.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Final, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Waveform(ABC):
    """
    An abstract base class to represent a waveform.

    Parameters
    ----------
    scale : float, optional
        Scaling factor of the waveform.
    time_offset : int, optional
        Time offset of the waveform in ns.
    phase_offset : float, optional
        Phase offset of the waveform in rad.
    """

    SAMPLING_PERIOD: Final[int] = 2  # ns

    def __init__(
        self,
        scale: float = 1.0,
        time_offset: int = 0,
        phase_offset: float = 0.0,
    ):
        self.scale = scale
        self.time_offset = time_offset
        self.phase_offset = phase_offset

    @property
    @abstractmethod
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the values of the waveform."""

    @property
    def length(self) -> int:
        """Returns the length of the waveform in samples."""
        return len(self.values)

    @property
    def duration(self) -> int:
        """Returns the duration of the waveform in ns."""
        return self.length * self.SAMPLING_PERIOD

    @property
    def times(self) -> npt.NDArray[np.int64]:
        """Returns the time array of the waveform in ns."""
        return np.arange(self.length) * self.SAMPLING_PERIOD + self.time_offset

    @property
    def real(self) -> npt.NDArray[np.float64]:
        """Returns the real part of the waveform."""
        return np.real(self.values)

    @property
    def imag(self) -> npt.NDArray[np.float64]:
        """Returns the imaginary part of the waveform."""
        return np.imag(self.values)

    @property
    def ampl(self) -> npt.NDArray[np.float64]:
        """Returns the amplitude of the waveform."""
        return np.abs(self.values)

    @property
    def phase(self) -> npt.NDArray[np.float64]:
        """Returns the phase of the waveform."""
        return np.angle(self.values)

    @abstractmethod
    def copy(self) -> "Waveform":
        """Returns a copy of the waveform."""

    @abstractmethod
    def scaled(self, scale: float) -> "Waveform":
        """Returns a copy of the waveform scaled by the given factor."""

    @abstractmethod
    def shifted(self, phase: float) -> "Waveform":
        """Returns a copy of the waveform shifted by the given phase."""

    @abstractmethod
    def repeated(self, n: int) -> "Waveform":
        """Returns a copy of the waveform repeated n times."""

    def _ns_to_samples(self, duration: int) -> int:
        """Converts a duration in ns to a length in samples."""
        if duration < 0:
            raise ValueError("Duration must be positive.")
        if duration % self.SAMPLING_PERIOD != 0:
            raise ValueError(
                f"Duration must be a multiple of the sampling period ({self.SAMPLING_PERIOD} ns)."
            )
        return duration // self.SAMPLING_PERIOD

    def plot(
        self,
        polar=False,
        savefig: Optional[str] = None,
        title="",
    ):
        """
        Plots the waveform.

        Parameters
        ----------
        polar : bool, optional
            If True, plots the waveform in the polar domain, otherwise in the
            time domain.
        savefig : str, optional
            If provided, saves the figure to the given path.
        title : str, optional
            Title of the plot.
        """
        if polar:
            self.plot_polar(
                title=title,
                savefig=savefig,
            )
        else:
            self.plot_xy(
                title=title,
                savefig=savefig,
            )

    def plot_xy(
        self,
        savefig: Optional[str] = None,
        title="",
        xlabel="Time (ns)",
        ylabel="Amplitude (arb. units)",
    ):
        """
        Plots the waveform in the time domain.

        Parameters
        ----------
        savefig : str, optional
            If provided, saves the figure to the given path.
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label of the x-axis.
        ylabel : str, optional
            Label of the y-axis.
        """
        _, ax = plt.subplots(figsize=(6, 2))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)
        real = np.append(self.real, self.real[-1])
        imag = np.append(self.imag, self.imag[-1])
        ax.step(times, real, label="I", where="post")
        ax.step(times, imag, label="Q", where="post")
        ax.legend()
        if savefig is not None:
            plt.savefig(savefig, dpi=300)
        else:
            ax.grid()
        plt.show()

    def plot_polar(
        self,
        savefig: Optional[str] = None,
        title="",
        xlabel="Time (ns)",
        ylabel="Amplitude (arb. units)",
    ):
        """
        Plots the waveform in the polar domain.

        Parameters
        ----------
        savefig : str, optional
            If provided, saves the figure to the given path.
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label of the x-axis.
        ylabel : str, optional
            Label of the y-axis.
        """
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
        fig.suptitle(title)
        ax[0].set_ylabel(ylabel)
        ax[1].set_ylabel("Phase (rad)")
        ax[1].set_xlabel(xlabel)
        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)
        ampl = np.append(self.ampl, self.ampl[-1])
        phase = np.append(self.phase, self.phase[-1])
        ax[0].step(times, ampl, where="post")
        ax[1].step(times, phase, where="post")
        if savefig is not None:
            plt.savefig(savefig, dpi=300)
        else:
            ax[0].grid()
            ax[1].grid()
        plt.show()


class Pulse(Waveform):
    """
    A class to represent a pulse.

    Parameters
    ----------
    values : ArrayLike
        Values of the pulse.
    scale : float, optional
        Scaling factor of the pulse.
    time_offset : int, optional
        Time offset of the pulse in ns.
    phase_offset : float, optional
        Phase offset of the pulse in rad.
    """

    def __init__(
        self,
        values: npt.ArrayLike,
        scale: float = 1.0,
        time_offset: int = 0,
        phase_offset: float = 0.0,
    ):
        super().__init__(scale, time_offset, phase_offset)
        self._values = np.array(values)

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the values of the pulse."""
        return self._values * self.scale * np.exp(1j * self.phase_offset)

    def copy(self) -> "Pulse":
        """Returns a copy of the pulse."""
        return deepcopy(self)

    def scaled(self, scale: float) -> "Pulse":
        """Returns a copy of the pulse scaled by the given factor."""
        new_pulse = deepcopy(self)
        new_pulse.scale *= scale
        return new_pulse

    def shifted(self, phase: float) -> "Pulse":
        """Returns a copy of the pulse shifted by the given phase."""
        new_pulse = deepcopy(self)
        new_pulse.phase_offset += phase
        return new_pulse

    def repeated(self, n: int) -> "PulseSequence":
        """Returns a pulse sequence of n copies of the pulse."""
        new_pulse = deepcopy(self)
        sequence = PulseSequence([new_pulse] * n)
        return sequence


class PulseSequence(Waveform):
    """
    A class to represent a pulse sequence.

    Parameters
    ----------
    waveforms : Sequence[Waveform], optional
        Waveforms of the pulse sequence.
    scale : float, optional
        Scaling factor of the pulse sequence.
    time_offset : int, optional
        Time offset of the pulse sequence in ns.
    phase_offset : float, optional
        Phase offset of the pulse sequence in rad.

    Examples
    --------
    >>> seq = PulseSequence([
    ...     Rect(duration=10, amplitude=1.0),
    ...     Blank(duration=10),
    ...     Gauss(duration=10, amplitude=1.0, sigma=2.0),
    ... ])
    """

    def __init__(
        self,
        waveforms: Optional[Sequence[Waveform]] = None,
        scale: float = 1.0,
        time_offset: int = 0,
        phase_offset: float = 0.0,
    ):
        super().__init__(scale, time_offset, phase_offset)
        if waveforms is None:
            waveforms = []
        self.waveforms = waveforms

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the concatenated values of the pulse sequence."""
        if len(self.waveforms) == 0:
            return np.array([])
        concat_values = np.concatenate([w.values for w in self.waveforms])
        values = concat_values * self.scale * np.exp(1j * self.phase_offset)
        return values

    def copy(self) -> "PulseSequence":
        """Returns a copy of the pulse sequence."""
        return deepcopy(self)

    def scaled(self, scale: float) -> "PulseSequence":
        """Returns a copy of the pulse sequence scaled by the given factor."""
        new_sequence = deepcopy(self)
        new_sequence.scale *= scale
        return new_sequence

    def shifted(self, phase: float) -> "PulseSequence":
        """Returns a copy of the pulse sequence shifted by the given phase."""
        new_sequence = deepcopy(self)
        new_sequence.phase_offset += phase
        return new_sequence

    def repeated(self, n: int) -> "PulseSequence":
        """Returns a copy of the pulse sequence repeated n times."""
        new_sequence = deepcopy(self)
        new_sequence.waveforms = list(new_sequence.waveforms) * n
        return new_sequence

    def reversed(self) -> "PulseSequence":
        """Returns a copy of the pulse sequence with the order of the waveforms reversed."""
        new_sequence = deepcopy(self)
        new_sequence.waveforms = list(reversed(new_sequence.waveforms))
        return new_sequence


class Blank(Pulse):
    """
    A class to represent a blank pulse.

    Parameters
    ----------
    duration : int
        Duration of the blank pulse in ns.

    Examples
    --------
    >>> pulse = Blank(duration=100)
    """

    def __init__(
        self,
        duration: int,
    ):
        length = self._ns_to_samples(duration)
        real = np.zeros(length, dtype=complex)
        imag = 0
        values = real + 1j * imag
        super().__init__(values)


class Rect(Pulse):
    """
    A class to represent a rectangular pulse.

    Parameters
    ----------
    duration : int
        Duration of the rectangular pulse in ns.
    amplitude : float
        Amplitude of the rectangular pulse.

    Examples
    --------
    >>> pulse = Rect(duration=100, amplitude=0.1)
    """

    def __init__(
        self,
        duration: int,
        amplitude: float,
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude)
        super().__init__(values)

    def _calc_values(
        self,
        duration: int,
        amplitude: float,
    ) -> npt.NDArray[np.complex128]:
        length = self._ns_to_samples(duration)
        real = amplitude * np.ones(length)
        imag = 0
        values = real + 1j * imag
        return values


class FlatTop(Pulse):
    """
    A class to represent a raised cosine flat-top pulse.

    Parameters
    ----------
    duration : int
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    tau : int
        Rise and fall time of the pulse in ns.
    
    Examples
    --------
    >>> pulse = FlatTop(
    ...     width=100,
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
        duration: int,
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
        duration: int,
        amplitude: float,
        tau: int,
    ) -> npt.NDArray[np.complex128]:
        flattime = duration - 2 * tau

        if flattime < 0:
            raise ValueError("duration must be greater than `2 * tau`.")

        length_rise = self._ns_to_samples(tau)
        length_flat = self._ns_to_samples(flattime)

        t_rise = np.linspace(0, tau, length_rise)
        t_flat = np.linspace(0, flattime, length_flat)

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
    duration : int
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
        duration: int,
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
        duration: int,
        amplitude: float,
        sigma: float,
    ) -> npt.NDArray[np.complex128]:
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
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
    duration : int
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
        duration: int,
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
        duration: int,
        amplitude: float,
        beta: float,
    ) -> npt.NDArray[np.complex128]:
        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
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
    duration : int
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
        duration: int,
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
        duration: int,
        amplitude: float,
        sigma: float,
        beta: float,
    ) -> npt.NDArray[np.complex128]:
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
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
    duration : int
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
        duration: int,
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
        duration: int,
        amplitude: float,
        beta: float,
    ) -> npt.NDArray[np.complex128]:
        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
        real = amplitude * (1.0 - np.cos(2 * np.pi * t / duration)) * 0.5
        imag = 2 * np.pi / duration * amplitude * np.sin(2 * np.pi * t / duration) * 0.5
        values = real + beta * 1j * imag
        return values
