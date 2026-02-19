"""Waveform primitives and utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Default sampling period in ns
DEFAULT_SAMPLING_PERIOD = 2.0


class Waveform(ABC):
    """
    An abstract base class to represent a waveform.

    Parameters
    ----------
    scale : float, optional
        Scaling factor of the waveform.
    detuning : float, optional
        Detuning of the waveform in GHz.
    phase : float, optional
        Phase of the waveform in rad.
    sampling_period : float, optional
        Sampling period of the waveform in ns.
    """

    SAMPLING_PERIOD: float = DEFAULT_SAMPLING_PERIOD

    def __init__(
        self,
        *,
        scale: float | None = None,
        detuning: float | None = None,
        phase: float | None = None,
        sampling_period: float | None = None,
        **kwargs,
    ):
        if scale is None:
            scale = 1.0
        if detuning is None:
            detuning = 0.0
        if phase is None:
            phase = 0.0
        if sampling_period is None:
            sampling_period = self.SAMPLING_PERIOD
        self._scale = scale
        self._detuning = detuning
        self._phase = phase
        self._sampling_period = sampling_period

    @property
    def name(self) -> str:
        """Returns the label of the waveform."""
        return self.__class__.__name__

    @property
    def scale(self) -> float:
        """Returns the scaling factor of the waveform."""
        return self._scale

    @property
    def detuning(self) -> float:
        """Returns the detuning of the waveform in GHz."""
        return self._detuning

    @property
    def phase(self) -> float:
        """Returns the phase of the waveform in rad."""
        return self._phase

    @property
    def sampling_period(self) -> float:
        """Returns the sampling period of the waveform in ns."""
        return self._sampling_period

    @property
    @abstractmethod
    def length(self) -> int:
        """Returns the length of the waveform in samples."""

    @property
    @abstractmethod
    def values(self) -> NDArray[np.complex128]:
        """Returns the I/Q values of the waveform."""

    @property
    def duration(self) -> float:
        """Returns the duration of the waveform in ns."""
        return self.length * self.sampling_period

    @cached_property
    def cached_duration(self) -> float:
        """Return the cached duration of the waveform."""
        return self.duration

    @property
    def times(self) -> NDArray[np.float64]:
        """Return the time array of the waveform in ns."""
        return np.arange(self.length, dtype=np.float64) * self.sampling_period

    @property
    def real(self) -> NDArray[np.float64]:
        """Return the real part of the waveform."""
        return np.real(self.values)

    @property
    def imag(self) -> NDArray[np.float64]:
        """Return the imaginary part of the waveform."""
        return np.imag(self.values)

    @property
    def abs(self) -> NDArray[np.float64]:
        """Return the amplitude of the waveform."""
        return np.abs(self.values)

    @property
    def angle(self) -> NDArray[np.float64]:
        """Return the phase of the waveform."""
        return np.where(self.abs == 0, 0, np.angle(self.values))

    @abstractmethod
    def copy(self) -> Waveform:
        """Return a copy of the waveform."""

    @abstractmethod
    def padded(
        self, total_duration: float, pad_side: Literal["right", "left"] = "right"
    ) -> Waveform:
        """Return a copy of the waveform with zero padding."""

    @abstractmethod
    def scaled(self, scale: float) -> Waveform:
        """Return a copy of the waveform scaled by the given factor."""

    @abstractmethod
    def detuned(self, detuning: float) -> Waveform:
        """Return a copy of the waveform detuned by the given frequency."""

    @abstractmethod
    def shifted(self, phase: float) -> Waveform:
        """Return a copy of the waveform shifted by the given phase."""

    @abstractmethod
    def repeated(self, n: int) -> Waveform:
        """Return a copy of the waveform repeated n times."""

    @abstractmethod
    def inverted(self) -> Waveform:
        """Return a copy of the waveform with the time inverted."""

    def reset_cached_duration(self) -> None:
        """Reset the cached duration of the waveform."""
        cast(dict, self.__dict__).pop("cached_duration", None)

    def _number_of_samples(
        self,
        duration: float,
    ) -> int:
        """
        Return the number of samples in the waveform.

        Parameters
        ----------
        duration : float
            Duration of the waveform in ns.
        """
        dt = self.sampling_period
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
    ) -> NDArray[np.float64]:
        """
        Return the sampling points of the waveform.

        Parameters
        ----------
        duration : float
            Duration of the waveform in ns.
        """
        dt = self.sampling_period
        N = self._number_of_samples(duration)
        # Sampling points are at the center of each time interval
        sampling_points = np.linspace(dt / 2, duration - dt / 2, N)
        return sampling_points

    def plot(
        self,
        *,
        polar: bool = False,
        title: str | None = None,
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
        divide_by_two_pi: bool = False,
    ) -> None:
        """
        Plot the waveform.

        Parameters
        ----------
        polar : bool, optional
            If True, the waveform is plotted with amplitude and phase.
        title : str, optional
            Title of the plot.
        """
        if title is None:
            title = f"{self.name} ({self.duration} ns)"
        if polar:
            self.plot_polar(
                title=title,
                line_shape=line_shape,
            )
        else:
            self.plot_xy(
                title=title,
                line_shape=line_shape,
                divide_by_two_pi=divide_by_two_pi,
            )

    def plot_xy(
        self,
        *,
        n_samples: int | None = None,
        divide_by_two_pi: bool = False,
        title: str | None = None,
        xlabel: str = "Time (ns)",
        ylabel: str = "Amplitude (arb. units)",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
    ) -> None:
        """
        Plot the waveform with I/Q values.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label of the x-axis.
        ylabel : str, optional
            Label of the y-axis.
        """
        if self.length == 0:
            logger.warning("Waveform is empty.")
            return

        times = np.append(self.times, self.times[-1] + self.sampling_period)
        real = np.append(self.real, self.real[-1])
        imag = np.append(self.imag, self.imag[-1])

        if n_samples is not None and len(times) > n_samples:
            indices = np.linspace(0, len(times) - 1, n_samples).astype(int)
            times = times[indices]
            real = real[indices]
            imag = imag[indices]

        if divide_by_two_pi:
            real /= 2 * np.pi * 1e-3
            imag /= 2 * np.pi * 1e-3

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=times,
                y=real,
                mode="lines",
                name="X",
                line_shape=line_shape,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=imag,
                mode="lines",
                name="Y",
                line_shape=line_shape,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title="Amplitude (MHz)" if divide_by_two_pi else ylabel,
        )
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

    def plot_polar(
        self,
        *,
        title: str = "",
        xlabel: str = "Time (ns)",
        ylabel_1: str = "Amplitude (arb. units)",
        ylabel_2: str = "Phase (rad)",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
    ) -> None:
        """
        Plot the waveform with amplitude and phase.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label of the x-axis.
        ylabel : str, optional
            Label of the y-axis.
        """
        if self.length == 0:
            logger.warning("Waveform is empty.")
            return

        times = np.append(self.times, self.times[-1] + self.sampling_period)
        ampl = np.append(self.abs, self.abs[-1])
        phase = np.append(self.angle, self.angle[-1])

        fig = go.Figure()
        fig.set_subplots(
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
                line_shape=line_shape,
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
                line_shape=line_shape,
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text=xlabel, row=2, col=1)
        fig.update_yaxes(title_text=ylabel_1, row=1, col=1)
        fig.update_yaxes(title_text=ylabel_2, row=2, col=1)
        fig.update_layout(width=600)
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

    def plot_spectrum(
        self,
        *,
        title: str | None = None,
        xlabel: str = "Frequency (MHz)",
        ylabel: str = "Amplitude (arb. units)",
        zero_padding_factor: int = 100,
        frequency_sign: Literal["positive", "negative"] = "negative",
        xlim: tuple[float, float] | None = None,
    ) -> None:
        """
        Plot the spectrum of the waveform.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label of the x-axis.
        ylabel : str, optional
            Label of the y-axis.
        """
        if self.length == 0:
            logger.warning("Waveform is empty.")
            return

        if title is None:
            title = "Frequency spectrum"

        if xlim is not None:
            xlim = (xlim[0] * 1e3, xlim[1] * 1e3)

        pulse = self.padded(
            total_duration=self.duration * zero_padding_factor,
            pad_side="right",
        )

        N = pulse.length
        values = pulse.values
        fft_values = np.fft.fft(values)
        if frequency_sign == "positive":
            d = self.sampling_period
        else:
            d = -self.sampling_period
        freqs = np.fft.fftfreq(N, d=d)
        idx = np.argsort(freqs)
        freqs = freqs[idx]
        fft_values = np.abs(fft_values[idx])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freqs * 1e3,
                y=fft_values,
                mode="lines",
                name="FFT",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis_range=xlim if xlim is not None else None,
        )
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )
