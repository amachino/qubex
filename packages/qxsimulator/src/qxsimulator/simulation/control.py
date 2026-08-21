"""Define piecewise-constant control signals for quantum simulations."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qxcore import Frequency
from qxpulse import Waveform
from qxvisualizer import show_figure

from qxsimulator.system.models import Object
from qxsimulator.system.models._normalization import normalize_frequency_to_ghz

logger = logging.getLogger(__name__)


class Control:
    """
    Represent a finite-duration, piecewise-constant control signal.

    Parameters
    ----------
    target : Object | str
        Driven system object or its label. When an `Object` is given and
        `frequency` is omitted, the object's frequency is used.
    waveform : Waveform | list | npt.NDArray
        One-dimensional complex I/Q amplitudes in rad/ns with shape
        `(n_segments,)`.
    durations : list | npt.NDArray | None, optional
        One-dimensional segment durations in ns with shape `(n_segments,)`.
        If omitted for a `Waveform`, its sampling period is used. Otherwise,
        `Waveform.SAMPLING_PERIOD` is used.
    frequency : float | Frequency | None, optional
        Cyclic drive frequency. Bare numbers are interpreted in GHz, and
        dimensioned frequencies are converted to GHz. A value is required when
        `target` is a string.
    final_frame_shift : float, optional
        Accumulated logical frame shift after the control in radians. This is
        metadata for result interpretation, not physical evolution. The
        default is 0.
    frame_shifts : list | npt.NDArray | None, optional
        One-dimensional accumulated logical frame shifts in radians with shape
        `(n_segments,)`. If omitted, all segment shifts are zero. This is
        metadata for result interpretation, not physical evolution.

    Raises
    ------
    ValueError
        If a string target has no frequency, segment-data lengths differ, a
        segment duration is not finite and positive, or a frame shift is not
        finite.

    Notes
    -----
    Waveform values, durations, and frame shifts are copied on construction
    and exposed as read-only arrays. The normalized frequency is stored as a
    float in GHz, and an `Object` target is stored by label. Empty waveform,
    duration, and frame-shift arrays represent a zero-duration control.
    """

    def __init__(
        self,
        target: Object | str,
        waveform: Waveform | list | npt.NDArray,
        durations: list | npt.NDArray | None = None,
        frequency: float | Frequency | None = None,
        final_frame_shift: float = 0.0,
        frame_shifts: list | npt.NDArray | None = None,
    ):
        if frequency is None:
            if isinstance(target, Object):
                frequency = target.frequency
            else:
                raise ValueError("Frequency is required for a string target.")

        if isinstance(target, Object):
            target = target.label

        self.target = target
        self.frequency = normalize_frequency_to_ghz(frequency)
        waveform_values = (
            waveform.values if isinstance(waveform, Waveform) else waveform
        )
        self._waveform = np.array(
            waveform_values,
            dtype=np.complex128,
            copy=True,
        )
        if durations is not None:
            self._durations = np.array(
                durations,
                dtype=np.float64,
                copy=True,
            )
        elif isinstance(waveform, Waveform):
            self._durations = np.full(
                len(self._waveform),
                waveform.sampling_period,
                dtype=np.float64,
            )
        else:
            self._durations = np.full(
                len(self._waveform),
                Waveform.SAMPLING_PERIOD,
                dtype=np.float64,
            )
        self._frame_shifts = np.array(
            np.zeros(len(self._waveform), dtype=np.float64)
            if frame_shifts is None
            else frame_shifts,
            dtype=np.float64,
            copy=True,
        )
        self.final_frame_shift = float(final_frame_shift)

        if len(self._waveform) != len(self._durations):
            raise ValueError("The lengths of waveform and durations do not match.")
        if len(self._waveform) != len(self._frame_shifts):
            raise ValueError("The lengths of waveform and frame_shifts do not match.")
        if not np.all(np.isfinite(self._durations)) or np.any(self._durations <= 0):
            raise ValueError("Segment durations must be finite and greater than zero.")
        if not np.all(np.isfinite(self._frame_shifts)) or not np.isfinite(
            self.final_frame_shift
        ):
            raise ValueError("Frame shifts must be finite.")

        self._waveform.flags.writeable = False
        self._durations.flags.writeable = False
        self._frame_shifts.flags.writeable = False

    @property
    def waveform(self) -> npt.NDArray[np.complex128]:
        """Return read-only segment amplitudes of shape `(n_segments,)` in rad/ns."""
        return self._waveform

    @property
    def durations(self) -> npt.NDArray[np.float64]:
        """Return read-only durations of shape `(n_segments,)` in ns."""
        return self._durations

    @property
    def frame_shifts(self) -> npt.NDArray[np.float64]:
        """Return read-only frame shifts of shape `(n_segments,)` in radians."""
        return self._frame_shifts

    @property
    def n_segments(self) -> int:
        """Return the number of waveform segments."""
        return len(self.waveform)

    @cached_property
    def duration(self) -> float:
        """Return the total control duration in ns."""
        return float(np.sum(self.durations))

    @cached_property
    def times(self) -> npt.NDArray[np.float64]:
        """Return read-only boundary times of shape `(n_segments + 1,)` in ns."""
        times = np.concatenate(([0.0], np.cumsum(self.durations)))
        times.flags.writeable = False
        return times

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Return step-plot amplitudes in rad/ns, repeating the final value."""
        if self.n_segments == 0:
            return self.waveform.copy()
        return np.append(self.waveform, self.waveform[-1])

    def get_samples(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        """
        Sample the finite-duration control using zero-order hold.

        Parameters
        ----------
        times : npt.NDArray[np.float64]
            Query times in ns.

        Returns
        -------
        npt.NDArray[np.complex128]
            Complex I/Q amplitudes in rad/ns with the same shape as `times`.

        Notes
        -----
        An internal segment boundary belongs to the segment starting at that
        boundary. The final boundary returns the last segment amplitude, while
        times outside the closed interval `[0, duration]` return zero.
        """
        query_times = np.asarray(times, dtype=np.float64)
        samples = np.zeros(query_times.shape, dtype=np.complex128)
        if self.n_segments == 0:
            return samples

        active = (query_times >= 0.0) & (query_times <= self.duration)
        indices = np.searchsorted(
            self.times[1:],
            query_times[active],
            side="right",
        )
        indices = np.minimum(indices, self.n_segments - 1)
        samples[active] = self.waveform[indices]
        return samples

    def get_frame_shifts(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Sample accumulated logical frame shifts using zero-order hold.

        Parameters
        ----------
        times : npt.NDArray[np.float64]
            Query times in ns.

        Returns
        -------
        npt.NDArray[np.float64]
            Frame shifts in radians with the same shape as `times`.

        Notes
        -----
        An internal segment boundary belongs to the segment starting at that
        boundary. Times before the control return zero. The final boundary and
        later times return `final_frame_shift` so that the terminal logical
        frame persists after the physical control ends.
        """
        query_times = np.asarray(times, dtype=np.float64)
        frame_shifts = np.zeros(query_times.shape, dtype=np.float64)

        active = (query_times >= 0.0) & (query_times < self.duration)
        if self.n_segments > 0:
            indices = np.searchsorted(
                self.times[1:],
                query_times[active],
                side="right",
            )
            frame_shifts[active] = self.frame_shifts[indices]

        frame_shifts[query_times >= self.duration] = self.final_frame_shift
        return frame_shifts

    def plot(
        self,
        times: npt.NDArray[np.float64] | None = None,
        n_samples: int | None = None,
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
    ) -> None:
        """
        Plot the control waveform as I/Q components.

        Parameters
        ----------
        times : npt.NDArray[np.float64] | None, optional
            Query times in ns. If omitted, the segment boundaries are used.
        n_samples : int | None, optional
            Non-negative maximum number of displayed points. If omitted, all
            points are displayed.
        line_shape : {"hv", "vh", "hvh", "vhv", "spline", "linear"}, optional
            Plotly line shape used to connect displayed points. The default is
            `"hv"`.

        Raises
        ------
        ValueError
            If `n_samples` is negative, or Plotly rejects `line_shape`.

        Notes
        -----
        Amplitudes are converted from angular frequency in rad/ns to cyclic
        frequency in MHz and displayed through `qxvisualizer`. An empty
        waveform emits a warning and produces no plot. This method displays an
        interactive figure and returns no figure object.
        """
        if self.n_segments == 0:
            logger.warning("Waveform is empty.")
            return

        if times is None:
            times = self.times
            real = self.values.real / (2 * np.pi * 1e-3)
            imag = self.values.imag / (2 * np.pi * 1e-3)
        else:
            samples = self.get_samples(times)
            real = samples.real / (2 * np.pi * 1e-3)
            imag = samples.imag / (2 * np.pi * 1e-3)

        if n_samples is not None and len(times) > n_samples:
            indices = np.linspace(0, len(times) - 1, n_samples).astype(int)
            times = times[indices]
            real = real[indices]
            imag = imag[indices]

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
            title="Control signal",
            xaxis_title="Time (ns)",
            yaxis_title="Amplitude (MHz)",
            template="qubex",
        )
        show_figure(fig, filename="control_signal")
