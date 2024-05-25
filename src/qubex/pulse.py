"""
Provides classes to represent waveforms. A waveform is a sequence of complex
numbers representing I and Q values of a pulse. The waveform can be plotted
in the time domain as a function of time or in the polar domain as a function
of amplitude and phase.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Final, Literal, Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Waveform(ABC):
    """
    An abstract base class to represent a waveform.

    Parameters
    ----------
    scale : float, optional
        Scaling factor of the waveform.
    detuning : float, optional
        Detuning of the waveform in GHz.
    phase_shift : float, optional
        Phase shift of the waveform in rad.
    """

    SAMPLING_PERIOD: Final[float] = 2.0  # ns

    def __init__(
        self,
        *,
        scale: float = 1.0,
        detuning: float = 0.0,
        phase_shift: float = 0.0,
    ):
        self._scale = scale
        self._detuning = detuning
        self._phase_shift = phase_shift

    @property
    @abstractmethod
    def length(self) -> int:
        """Returns the length of the waveform in samples."""

    @property
    @abstractmethod
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the I/Q values of the waveform."""

    @property
    def duration(self) -> float:
        """Returns the duration of the waveform in ns."""
        return self.length * self.SAMPLING_PERIOD

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """Returns the time array of the waveform in ns."""
        return np.arange(self.length) * self.SAMPLING_PERIOD

    @property
    def real(self) -> npt.NDArray[np.float64]:
        """Returns the real part of the waveform."""
        return np.real(self.values)

    @property
    def imag(self) -> npt.NDArray[np.float64]:
        """Returns the imaginary part of the waveform."""
        return np.imag(self.values)

    @property
    def abs(self) -> npt.NDArray[np.float64]:
        """Returns the amplitude of the waveform."""
        return np.abs(self.values)

    @property
    def angle(self) -> npt.NDArray[np.float64]:
        """Returns the phase of the waveform."""
        return np.where(self.abs == 0, 0, np.angle(self.values))

    @abstractmethod
    def copy(self) -> "Waveform":
        """Returns a copy of the waveform."""

    @abstractmethod
    def padded(
        self, total_duration: float, pad_side: Literal["right", "left"] = "right"
    ) -> "Waveform":
        """Returns a copy of the waveform with zero padding."""

    @abstractmethod
    def scaled(self, scale: float) -> "Waveform":
        """Returns a copy of the waveform scaled by the given factor."""

    @abstractmethod
    def detuned(self, detuning: float) -> "Waveform":
        """Returns a copy of the waveform detuned by the given frequency."""

    @abstractmethod
    def shifted(self, phase: float) -> "Waveform":
        """Returns a copy of the waveform shifted by the given phase."""

    @abstractmethod
    def repeated(self, n: int) -> "Waveform":
        """Returns a copy of the waveform repeated n times."""

    def _number_of_samples(
        self,
        duration: float,
    ) -> int:
        """
        Returns the number of samples in the waveform.

        Parameters
        ----------
        duration : float
            Duration of the waveform in ns.
        """
        dt = self.SAMPLING_PERIOD
        if duration < 0:
            raise ValueError("Duration must be positive.")

        # Tolerance for floating point comparison
        tolerance = 1e-9
        frac = duration / dt
        N = round(frac)
        if abs(frac - N) > tolerance:
            raise ValueError(
                f"Duration must be a multiple of the sampling period ({dt} ns)."
            )
        return N

    def _sampling_points(
        self,
        duration: float,
    ) -> npt.NDArray[np.float64]:
        """
        Returns the sampling points of the waveform.

        Parameters
        ----------
        duration : float
            Duration of the waveform in ns.
        """
        dt = self.SAMPLING_PERIOD
        N = self._number_of_samples(duration)
        # Sampling points are at the center of each time interval
        sampling_points = np.linspace(dt / 2, duration - dt / 2, N)
        return sampling_points

    def plot(
        self,
        *,
        polar=False,
        title="",
    ):
        """
        Plots the waveform.

        Parameters
        ----------
        polar : bool, optional
            If True, the waveform is plotted with amplitude and phase.
        title : str, optional
            Title of the plot.
        """
        if polar:
            self.plot_polar(title=title)
        else:
            self.plot_xy(title=title)

    def plot_xy(
        self,
        *,
        title="",
        xlabel="Time (ns)",
        ylabel="Amplitude (arb. units)",
    ):
        """
        Plots the waveform with I/Q values.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label of the x-axis.
        ylabel : str, optional
            Label of the y-axis.
        """
        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)
        real = np.append(self.real, self.real[-1])
        imag = np.append(self.imag, self.imag[-1])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=times,
                y=real,
                mode="lines",
                name="I",
                line_shape="hv",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=imag,
                mode="lines",
                name="Q",
                line_shape="hv",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=800,
            template="plotly_white",
        )
        fig.show()

    def plot_polar(
        self,
        *,
        title="",
        xlabel="Time (ns)",
        ylabel_1="Amplitude (arb. units)",
        ylabel_2="Phase (rad)",
    ):
        """
        Plots the waveform with amplitude and phase.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label of the x-axis.
        ylabel : str, optional
            Label of the y-axis.
        """
        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)
        ampl = np.append(self.abs, self.abs[-1])
        phase = np.append(self.angle, self.angle[-1])

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(title, ""),
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=ampl,
                mode="lines",
                name="Amplitude",
                line_shape="hv",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=phase,
                mode="lines",
                name="Phase",
                line_shape="hv",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text=xlabel, row=2, col=1)
        fig.update_yaxes(title_text=ylabel_1, row=1, col=1)
        fig.update_yaxes(title_text=ylabel_2, row=2, col=1)
        fig.update_layout(
            width=800,
            template="plotly_white",
        )
        fig.show()


class Pulse(Waveform):
    """
    A class to represent a pulse.

    Parameters
    ----------
    values : ArrayLike
        I/Q values of the pulse.
    scale : float, optional
        Scaling factor of the pulse.
    detuning : float, optional
        Detuning of the pulse in GHz.
    phase_shift : float, optional
        Phase shift of the pulse in rad.
    """

    def __init__(
        self,
        values: npt.ArrayLike,
        *,
        scale: float = 1.0,
        detuning: float = 0.0,
        phase_shift: float = 0.0,
    ):
        super().__init__(
            scale=scale,
            detuning=detuning,
            phase_shift=phase_shift,
        )
        self._values = np.array(values, dtype=np.complex128)

    @property
    def length(self) -> int:
        """Returns the length of the pulse in samples."""
        return len(self._values)

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the I/Q values of the pulse."""
        return (
            self._values
            * self._scale
            * np.exp(1j * (2 * np.pi * self._detuning * self.times + self._phase_shift))
        )

    def copy(self) -> "Pulse":
        """Returns a copy of the pulse."""
        return deepcopy(self)

    def padded(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> "Pulse":
        """
        Returns a copy of the pulse with zero padding.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        N = self._number_of_samples(total_duration)
        if pad_side == "right":
            values = np.pad(self._values, (0, N - self.length), mode="constant")
        elif pad_side == "left":
            values = np.pad(self._values, (N - self.length, 0), mode="constant")
        else:
            raise ValueError("pad_side must be either 'right' or 'left'.")
        new_pulse = deepcopy(self)
        new_pulse._values = values
        return new_pulse

    def scaled(self, scale: float) -> "Pulse":
        """Returns a copy of the pulse scaled by the given factor."""
        new_pulse = deepcopy(self)
        new_pulse._scale *= scale
        return new_pulse

    def detuned(self, detuning: float) -> "Pulse":
        """Returns a copy of the pulse detuned by the given frequency."""
        new_pulse = deepcopy(self)
        new_pulse._detuning += detuning
        return new_pulse

    def shifted(self, phase: float) -> "Pulse":
        """Returns a copy of the pulse shifted by the given phase."""
        new_pulse = deepcopy(self)
        new_pulse._phase_shift += phase
        return new_pulse

    def repeated(self, n: int) -> "Pulse":
        """Returns a copy of the pulse repeated n times."""
        new_pulse = deepcopy(self)
        new_pulse._values = np.tile(self._values, n)
        return new_pulse


class PulseSequence(Waveform):
    """
    A class to represent a pulse sequence.

    Parameters
    ----------
    waveforms : Sequence[Waveform]
        Waveforms of the pulse sequence.
    scale : float, optional
        Scaling factor of the pulse sequence.
    detuning : float, optional
        Detuning of the pulse sequence in GHz.
    phase_shift : float, optional
        Phase shift of the pulse sequence in rad.

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
        waveforms: Sequence[Waveform],
        *,
        scale: float = 1.0,
        detuning: float = 0.0,
        phase_shift: float = 0.0,
    ):
        super().__init__(
            scale=scale,
            detuning=detuning,
            phase_shift=phase_shift,
        )
        self._waveforms: list[Waveform] = list(waveforms)

    @property
    def length(self) -> int:
        """Returns the total length of the pulse sequence in samples."""
        return sum([w.length for w in self._waveforms])

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the concatenated values of the pulse sequence."""
        if len(self._waveforms) == 0:
            return np.array([])
        concat_values = np.concatenate([w.values for w in self._waveforms])
        values = (
            concat_values
            * self._scale
            * np.exp(1j * (2 * np.pi * self._detuning * self.times + self._phase_shift))
        )
        return values

    def copy(self) -> "PulseSequence":
        """Returns a copy of the pulse sequence."""
        return deepcopy(self)

    def padded(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> "PulseSequence":
        """
        Returns a copy of the pulse sequence with zero padding.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse sequence in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        new_sequence = deepcopy(self)
        blank = Blank(duration=total_duration - new_sequence.duration)
        if pad_side == "right":
            new_waveforms = new_sequence._waveforms + [blank]
        elif pad_side == "left":
            new_waveforms = [blank] + new_sequence._waveforms
        else:
            raise ValueError("pad_side must be either 'right' or 'left'.")
        new_sequence._waveforms = new_waveforms
        return new_sequence

    def scaled(self, scale: float) -> "PulseSequence":
        """Returns a copy of the pulse sequence scaled by the given factor."""
        new_sequence = deepcopy(self)
        new_sequence._scale *= scale
        return new_sequence

    def detuned(self, detuning: float) -> "PulseSequence":
        """Returns a copy of the pulse sequence detuned by the given frequency."""
        new_sequence = deepcopy(self)
        new_sequence._detuning += detuning
        return new_sequence

    def shifted(self, phase: float) -> "PulseSequence":
        """Returns a copy of the pulse sequence shifted by the given phase."""
        new_sequence = deepcopy(self)
        new_sequence._phase_shift += phase
        return new_sequence

    def repeated(self, n: int) -> "PulseSequence":
        """Returns a copy of the pulse sequence repeated n times."""
        new_sequence = deepcopy(self)
        new_sequence._waveforms = list(new_sequence._waveforms) * n
        return new_sequence

    def reversed(self) -> "PulseSequence":
        """Returns a copy of the pulse sequence with the order of the waveforms reversed."""
        new_sequence = deepcopy(self)
        new_sequence._waveforms = list(reversed(new_sequence._waveforms))
        return new_sequence


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
