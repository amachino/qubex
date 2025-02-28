from __future__ import annotations

import logging
from copy import deepcopy
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from typing_extensions import deprecated

from ..style import COLORS
from .blank import Blank
from .phase_shift import PhaseShift, VirtualZ
from .pulse import Pulse
from .waveform import Waveform

logger = logging.getLogger(__name__)


class PulseArray(Waveform):
    """
    A class to represent an array of Pulse and PhaseShift.

    Parameters
    ----------
    elements : Sequence[Pulse | PhaseShift]
        List of pulses and phase shifts in the pulse array.
    scale : float, optional
        Scaling factor of the pulse array.
    detuning : float, optional
        Detuning of the pulse array in GHz.
    phase_shift : float, optional
        Phase shift of the pulse array in rad.

    Examples
    --------
    >>> arr = PulseArray([
    ...     Rect(duration=10, amplitude=1.0),
    ...     Blank(duration=10),
    ...     PhaseShift(theta=np.pi/2),
    ...     Gaussian(duration=10, amplitude=1.0, sigma=2.0),
    ... ])

    >>> arr = PulseArray()
    >>> arr.add(Rect(duration=10, amplitude=1.0))
    >>> arr.add(Blank(duration=10))
    >>> arr.add(PhaseShift(theta=np.pi/2))
    >>> arr.add(Gaussian(duration=10, amplitude=1.0, sigma=2.0))
    """

    def __init__(
        self,
        elements: Sequence = [],
        *,
        scale: float | None = None,
        detuning: float | None = None,
        phase: float | None = None,
    ):
        super().__init__(
            scale=scale,
            detuning=detuning,
            phase=phase,
        )
        self._elements: list[Waveform | PhaseShift] = list(elements)

    def __repr__(self) -> str:
        elements = []
        for obj in self._elements:
            if isinstance(obj, (PhaseShift, VirtualZ)):
                elements.append(f"{obj.__class__.__name__}({obj.theta:.2f})")
            else:
                elements.append(f"{obj.__class__.__name__}({obj.length})")
        return f"PulseArray([{', '.join(elements)}])"

    @property
    def elements(self) -> list[Pulse | PhaseShift]:
        """Returns the flattened list of pulses and phase shifts in the pulse array."""
        elements = []
        for obj in self._elements:
            if isinstance(obj, PulseArray):
                elements.extend(obj.elements)
            else:
                elements.append(obj)
        return elements

    @property
    def pulses(self) -> list[Pulse]:
        """Returns the list of pulses in the pulse array."""
        pulses: list[Pulse] = []
        current_phase = 0.0
        for obj in self.elements:
            if isinstance(obj, Pulse):
                pulses.append(obj.shifted(current_phase))
            elif isinstance(obj, PhaseShift):
                current_phase += obj.theta
            else:
                logger.warning(f"Unknown element type: {type(obj)}")
        return pulses

    @property
    def length(self) -> int:
        """Returns the total length of the pulse array in samples."""
        return sum([pulse.length for pulse in self.pulses])

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the concatenated values of the pulse array."""
        if len(self.pulses) == 0:
            return np.array([])

        concat_values = np.concatenate([pulse.values for pulse in self.pulses])
        values = (
            concat_values
            * self._scale
            * np.exp(1j * (2 * np.pi * self._detuning * self.times + self._phase))
        )
        return values

    @property
    def frame_shifts(self) -> npt.NDArray[np.float64]:
        """Returns the frame shifts of the pulse array."""
        phases = []
        current_phase = 0.0
        for obj in self.elements:
            if isinstance(obj, Pulse):
                phases += [current_phase] * obj.length
            elif isinstance(obj, PhaseShift):
                current_phase += obj.theta
            else:
                logger.warning(f"Unknown element type: {type(obj)}")
        return np.array(phases)

    @property
    def final_frame_shift(self) -> float:
        """Returns the final frame shift of the pulse array."""
        # NOTE: This is not the same as frame_shifts[-1]
        current_phase = 0.0
        for obj in self.elements:
            if isinstance(obj, PhaseShift):
                current_phase += obj.theta
        return current_phase

    def add(self, obj: Waveform | PhaseShift) -> None:
        """Adds the given waveform or phase shift to the pulse sequence."""
        self._elements.append(obj)

    def pad(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> None:
        """
        Adds zero padding to the pulse array.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse array in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        duration = total_duration - self.duration
        if duration < 0:
            raise ValueError(
                f"Total duration ({total_duration}) must be greater than the current duration ({self.duration})."
            )
        blank = Blank(duration)
        if pad_side == "right":
            self._elements.append(blank)
        elif pad_side == "left":
            self._elements.insert(0, blank)
        else:
            raise ValueError("pad_side must be either 'right' or 'left'.")

    def copy(self) -> PulseArray:
        """Returns a copy of the pulse array."""
        return deepcopy(self)

    def padded(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> PulseArray:
        """
        Returns a copy of the pulse array with zero padding.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse array in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        duration = total_duration - self.duration
        if duration < 0:
            raise ValueError(
                f"Total duration ({total_duration}) must be greater than the current duration ({self.duration})."
            )
        new_array = deepcopy(self)
        blank = Blank(duration=duration)
        if pad_side == "right":
            new_elements = new_array._elements + [blank]
        elif pad_side == "left":
            new_elements = [blank] + new_array._elements
        else:
            raise ValueError("pad_side must be either 'right' or 'left'.")
        new_array._elements = new_elements
        return new_array

    def scaled(self, scale: float) -> PulseArray:
        """Returns a copy of the pulse array scaled by the given factor."""
        new_array = deepcopy(self)
        new_array._scale *= scale
        return new_array

    def detuned(self, detuning: float) -> PulseArray:
        """Returns a copy of the pulse array detuned by the given frequency."""
        new_array = deepcopy(self)
        new_array._detuning += detuning
        return new_array

    def shifted(self, phase: float) -> PulseArray:
        """Returns a copy of the pulse array shifted by the given phase."""
        new_array = deepcopy(self)
        new_array._phase += phase
        return new_array

    def repeated(self, n: int) -> PulseArray:
        """Returns a copy of the pulse array repeated n times."""
        new_array = deepcopy(self)
        new_array._elements = list(new_array._elements) * n
        return new_array

    def reversed(self) -> PulseArray:
        """Returns a copy of the pulse array with the time reversed."""
        new_array = PulseArray()
        for obj in reversed(self.elements):
            if isinstance(obj, Pulse):
                new_array.add(obj.reversed())
            elif isinstance(obj, PhaseShift):
                new_array.add(PhaseShift(-obj.theta))
            else:
                logger.warning(f"Unknown element type: {type(obj)}")
        return new_array

    def added(self, obj: Waveform | PhaseShift) -> PulseArray:
        """Returns a copy of the pulse array with the given waveform or phase shift added."""
        new_array = deepcopy(self)
        new_array._elements.append(obj)
        return new_array

    def plot(
        self,
        *,
        n_samples: int | None = None,
        divide_by_two_pi: bool = False,
        title: str | None = None,
        xlabel: str = "Time (ns)",
        ylabel: str = "Amplitude (arb. unit)",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
        show_frame_shifts: bool = True,
    ):
        """
        Plots the waveform of the pulse array.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label of the x-axis.
        ylabel : str, optional
            Label of the y-axis.
        line_shape : {"hv", "vh", "hvh", "vhv", "spline", "linear"}, optional
            Determines the line shape.
        show_frame_shifts : bool, optional
            Whether to show the frame shifts.
        """
        if self.length == 0:
            print("Waveform is empty.")
            return

        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)
        real = np.append(self.real, self.real[-1])
        imag = np.append(self.imag, self.imag[-1])
        frame_shifts = np.append(self.frame_shifts, self.final_frame_shift)
        frame_shifts = (frame_shifts + np.pi) % (2 * np.pi) - np.pi

        if n_samples is not None and len(times) > n_samples:
            indices = np.linspace(0, len(times) - 1, n_samples).astype(int)
            times = times[indices]
            real = real[indices]
            imag = imag[indices]

        if divide_by_two_pi:
            real /= 2 * np.pi * 1e-3
            imag /= 2 * np.pi * 1e-3

        y_max = np.max(self.abs)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=times,
                y=real,
                mode="lines",
                name="I",
                line_shape=line_shape,
                line=dict(color=COLORS[0]),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=imag,
                mode="lines",
                name="Q",
                line_shape=line_shape,
                line=dict(color=COLORS[1]),
            ),
        )

        if show_frame_shifts:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=frame_shifts,
                    name="φ",
                    mode="lines",
                    line_shape=line_shape,
                    line=dict(color=COLORS[2], dash="dot"),
                    yaxis="y2",
                ),
            )

        fig.update_layout(
            title=title or f"Pulse Sequence ({self.duration} ns)",
            xaxis_title=xlabel,
            yaxis=dict(
                title="Amplitude (MHz)" if divide_by_two_pi else ylabel,
                range=[-y_max * 1.2, y_max * 1.2],
            ),
            yaxis2=dict(
                title="Frame shift (rad)",
                overlaying="y",
                side="right",
                range=[-np.pi * 1.2, np.pi * 1.2],
                tickvals=[-np.pi, 0, np.pi],
                ticktext=["-π", "0", "π"],
            ),
            legend=dict(
                orientation="h",
                xanchor="right",
                yanchor="bottom",
                x=1,
                y=0,
                # semi-transparent background
                bgcolor="rgba(255, 255, 255, 0.9)",
            ),
        )

        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )


@deprecated("Use `PulseArray` instead.")
class PulseSequence(PulseArray):
    pass
