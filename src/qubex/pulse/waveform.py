from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

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

    SAMPLING_PERIOD: float = 2.0  # ns

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
    def copy(self) -> Waveform:
        """Returns a copy of the waveform."""

    @abstractmethod
    def padded(
        self, total_duration: float, pad_side: Literal["right", "left"] = "right"
    ) -> Waveform:
        """Returns a copy of the waveform with zero padding."""

    @abstractmethod
    def scaled(self, scale: float) -> Waveform:
        """Returns a copy of the waveform scaled by the given factor."""

    @abstractmethod
    def detuned(self, detuning: float) -> Waveform:
        """Returns a copy of the waveform detuned by the given frequency."""

    @abstractmethod
    def shifted(self, phase: float) -> Waveform:
        """Returns a copy of the waveform shifted by the given phase."""

    @abstractmethod
    def repeated(self, n: int) -> Waveform:
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
        title=None,
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
        if title is None:
            title = f"Waveform ({self.duration} ns)"
        if polar:
            self.plot_polar(title=title)
        else:
            self.plot_xy(title=title)

    def plot_xy(
        self,
        *,
        n_max_points: int | None = None,
        devide_by_two_pi=False,
        title=None,
        xlabel="Time (ns)",
        ylabel="Amplitude (arb. units)",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
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
        if self.length == 0:
            print("Waveform is empty.")
            return

        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)
        real = np.append(self.real, self.real[-1])
        imag = np.append(self.imag, self.imag[-1])

        if n_max_points is not None and len(times) > n_max_points:
            indices = np.linspace(0, len(times) - 1, n_max_points).astype(int)
            times = times[indices]
            real = real[indices]
            imag = imag[indices]

        if devide_by_two_pi:
            real /= 2 * np.pi
            imag /= 2 * np.pi

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=times,
                y=real,
                mode="lines",
                name="I",
                line_shape=line_shape,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=imag,
                mode="lines",
                name="Q",
                line_shape=line_shape,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
        )
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "svg",
                    "scale": 3,
                },
            }
        )

    def plot_polar(
        self,
        *,
        title="",
        xlabel="Time (ns)",
        ylabel_1="Amplitude (arb. units)",
        ylabel_2="Phase (rad)",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
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
        if self.length == 0:
            print("Waveform is empty.")
            return

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
                    "format": "svg",
                    "scale": 3,
                },
            }
        )
