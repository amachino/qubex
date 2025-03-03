from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Collection, Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..style import COLORS
from .blank import Blank
from .pulse import Waveform
from .pulse_array import PhaseShift, PulseArray


@dataclass
class Channel:
    label: str
    sequence: PulseArray = field(default_factory=PulseArray)
    offset: float = 0.0
    frequency: float | None = None
    target: str | None = None
    frame: str | None = None


class PulseSchedule:
    def __init__(
        self,
        channels: list[str] | dict[str, Channel] | None = None,
        /,
    ):
        """
        A class to represent a pulse schedule.

        Parameters
        ----------
        channels : list[str] | dict[str, Any]
            The

        Examples
        --------
        >>> from qubex.pulse import PulseSchedule, FlatTop
        >>> with PulseSchedule(["Q01", "RQ01", "Q02", "RQ02"]) as seq:
        ...     seq.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     seq.barrier()
        ...     seq.add("Q02", FlatTop(duration=100, amplitude=1, tau=10))
        ...     seq.barrier()
        ...     seq.add("RQ01", FlatTop(duration=200, amplitude=1, tau=10))
        ...     seq.add("RQ02", FlatTop(duration=200, amplitude=1, tau=10))
        >>> seq.plot()
        """
        if channels is None:
            self._channels = {}
        elif isinstance(channels, list):
            self._channels = {label: Channel(label) for label in channels}
        elif isinstance(channels, dict):
            self._channels = deepcopy(channels)
        else:
            raise ValueError("Invalid channels.")

        self._global_offset = 0.0

    def __enter__(self):
        """
        Enter the context manager.

        Examples
        --------
        >>> with PulseSchedule([...]) as seq:
        ...     seq.add(...)
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager and add a barrier to the sequence.

        The following codes are equivalent:

        >>> seq = PulseSchedule([...])
        >>> seq.add(...)
        >>> seq.barrier()

        >>> with PulseSchedule([...]) as seq:
        ...     seq.add(...)

        Note that duration of sequences might be different if context manager is not used.
        """
        self.barrier()

    @property
    def labels(self) -> list[str]:
        """
        Returns the channel labels.
        """
        return list(self._channels.keys())

    @property
    def values(self) -> dict[str, npt.NDArray[np.complex128]]:
        """
        Returns the sampled pulse sequences.
        """
        return self.get_sampled_sequences()

    @property
    def length(self) -> int:
        """
        Returns the length of the pulse schedule in samples.
        """
        if len(self._channels) == 0:
            return 0
        if not self.is_valid():
            raise ValueError("Inconsistent sequence lengths.")
        return len(next(iter(self.values.values())))

    @property
    def duration(self) -> float:
        """
        Returns the duration of the pulse schedule in ns.
        """
        return self.length * Waveform.SAMPLING_PERIOD

    def add(
        self,
        /,
        label: str,
        obj: Waveform | PhaseShift,
    ):
        """
        Add a waveform or a phase shift to the pulse schedule.

        Parameters
        ----------
        label : str
            The channel label.
        obj : Waveform | PhaseShift
            The waveform or phase shift to add.

        Examples
        --------
        >>> with PulseSchedule(["Q01"]) as seq:
        ...     seq.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        """
        self._add_channels_if_not_exist([label])

        self._channels[label].sequence.add(obj)

        if isinstance(obj, Waveform):
            self._channels[label].offset += obj.duration

    def barrier(
        self,
        labels: Collection[str] | None = None,
    ):
        """
        Add a barrier to the pulse schedule.

        Parameters
        ----------
        labels : list[str], optional
            The channel labels to add the barrier.

        Examples
        --------
        >>> with PulseSchedule(["Q01"]) as seq:
        ...     seq.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     seq.barrier()
        """
        if labels is None:
            labels = self.labels
            self._global_offset = self._max_offset()
        else:
            labels = list(labels)

        self._add_channels_if_not_exist(labels)

        for label in labels:
            diff = self._max_offset(labels) - self._channels[label].offset
            if diff > 0:
                self.add(label, Blank(duration=diff))

    def call(
        self,
        schedule: PulseSchedule,
    ):
        """
        Call another pulse schedule in the current pulse schedule.

        Parameters
        ----------
        schedule : PulseSchedule
            The pulse schedule to call.

        Examples
        --------
        >>> with PulseSchedule(["Q01", "Q02"]) as ctrl:
        ...     ctrl.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     ctrl.barrier()
        ...     ctrl.add("Q02", FlatTop(duration=100, amplitude=1, tau=10))
        >>> with PulseSchedule(["RQ01", "RQ02"]) as read:
        ...     read.add("RQ01", FlatTop(duration=200, amplitude=1, tau=10))
        ...     read.add("RQ02", FlatTop(duration=200, amplitude=1, tau=10))
        >>> with PulseSchedule(["Q01", "Q02", "RQ01", "RQ02"]) as seq:
        ...     seq.call(ctrl)
        ...     seq.barrier()
        ...     seq.call(read)
        >>> seq.plot()
        """
        if schedule == self:
            raise ValueError("Cannot call itself.")

        self.barrier(schedule.labels)
        sequences = schedule.get_sequences()
        for target, sequence in sequences.items():
            self.add(target, sequence)

    def copy(self) -> PulseSchedule:
        """
        Returns a copy of the pulse schedule.
        """
        return deepcopy(self)

    def scaled(self, scale: float) -> PulseSchedule:
        """
        Returns a scaled pulse schedule.
        """
        new_sched = deepcopy(self)
        for channel in new_sched._channels.values():
            channel.sequence._scale *= scale
        return new_sched

    def detuned(self, detuning: float) -> PulseSchedule:
        """
        Returns a detuned pulse schedule.
        """
        new_sched = deepcopy(self)
        for channel in new_sched._channels.values():
            channel.sequence._detuning += detuning
        return new_sched

    def shifted(self, phase: float) -> PulseSchedule:
        """
        Returns a shifted pulse schedule.
        """
        new_sched = deepcopy(self)
        for channel in new_sched._channels.values():
            channel.sequence._phase += phase
        return new_sched

    def repeated(self, n: int) -> PulseSchedule:
        """
        Returns a repeated pulse schedule.
        """
        new_sched = PulseSchedule()
        for _ in range(n):
            new_sched.call(self)
        return new_sched

    def reversed(self) -> PulseSchedule:
        """Returns a time-reversed pulse schedule."""
        with PulseSchedule() as new_sched:
            for label, channel in self._channels.items():
                new_sched.add(label, channel.sequence.reversed())
        return new_sched

    def plot(
        self,
        *,
        show_physical_pulse: bool = False,
        title: str = "Pulse Sequence",
        width: int = 800,
        n_samples: int = 1024,
        divide_by_two_pi: bool = False,
        time_unit: Literal["ns", "samples"] = "ns",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
    ):
        """
        Plots the pulse schedule.

        Examples
        --------
        >>> from qubex.pulse import PulseSchedule, FlatTop
        >>> with PulseSchedule(["Q01", "RQ01", "Q02", "RQ02"]) as seq:
        ...     seq.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     seq.barrier()
        ...     seq.add("Q02", FlatTop(duration=100, amplitude=1, tau=10))
        ...     seq.barrier()
        ...     seq.add("RQ01", FlatTop(duration=200, amplitude=1, tau=10))
        ...     seq.add("RQ02", FlatTop(duration=200, amplitude=1, tau=10))
        >>> seq.plot()
        """
        if self._max_offset() == 0.0:
            print("No data to plot.")
            return

        n_channels = len(self._channels)

        if n_channels == 0:
            print("No data to plot.")
            return

        sequences = self.get_sequences()

        fig = make_subplots(
            rows=n_channels,
            cols=1,
            shared_xaxes=True,
            specs=[[{"secondary_y": True}] for _ in range(n_channels)],
        )
        for i, (target, seq) in enumerate(sequences.items()):
            if time_unit == "ns":
                times = np.append(seq.times, seq.times[-1] + seq.SAMPLING_PERIOD)
            else:
                times = np.arange(seq.length + 1)

            if show_physical_pulse:
                values = seq.get_values(apply_frame_shifts=True)
            else:
                values = seq.get_values(apply_frame_shifts=False)
            real = np.real(values)
            imag = np.imag(values)
            real = np.append(real, real[-1])
            imag = np.append(imag, imag[-1])
            phase = -np.append(seq.frame_shifts, seq.final_frame_shift)
            phase = (phase + np.pi) % (2 * np.pi) - np.pi

            if len(times) > n_samples:
                times = self._downsample(times, n_samples)
                real = self._downsample(real, n_samples)
                imag = self._downsample(imag, n_samples)
                phase = self._downsample(phase, n_samples)

            if divide_by_two_pi:
                real /= 2 * np.pi * 1e-3
                imag /= 2 * np.pi * 1e-3

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=real,
                    name="I" if show_physical_pulse else "X",
                    mode="lines",
                    line_shape=line_shape,
                    line=dict(color=COLORS[0]),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=imag,
                    name="Q" if show_physical_pulse else "Y",
                    mode="lines",
                    line_shape=line_shape,
                    line=dict(color=COLORS[1]),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
                secondary_y=False,
            )
            if not show_physical_pulse:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=phase,
                        name="φ",
                        mode="lines",
                        line_shape=line_shape,
                        line=dict(color=COLORS[2], dash="dot"),
                        showlegend=(i == 0),
                    ),
                    row=i + 1,
                    col=1,
                    secondary_y=True,
                )

        fig.update_layout(
            title=title,
            height=80 * n_channels + 140,
            width=width,
        )
        fig.update_xaxes(
            row=n_channels,
            col=1,
            title_text="Time (ns)" if time_unit == "ns" else "Time (samples)",
        )
        for i, (target, seq) in enumerate(sequences.items()):
            y_max = np.max(seq.abs)
            if divide_by_two_pi:
                y_max /= 2 * np.pi * 1e-3
            fig.update_yaxes(
                row=i + 1,
                col=1,
                title_text=target,
                range=[-1.2 * y_max, 1.2 * y_max],
                secondary_y=False,
            )
            if not show_physical_pulse:
                fig.update_yaxes(
                    row=i + 1,
                    col=1,
                    range=[-np.pi * 1.2, np.pi * 1.2],
                    tickvals=[-np.pi, 0, np.pi],
                    ticktext=["-π", "0", "π"],
                    secondary_y=True,
                )
            annotations = []
            frequency = self._channels[target].frequency
            if frequency is not None:
                annotations.append(f"{frequency:.2f} GHz")
            target = self._channels[target].target
            if target is not None:
                annotations.append(f"{target}")
            fig.add_annotation(
                x=0.02,
                y=0.06,
                xref="x domain",
                yref="y domain",
                text=" → ".join(annotations),
                showarrow=False,
                row=i + 1,
                col=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
            )
        fig.show()

    def is_valid(self) -> bool:
        """
        Returns True if the pulse schedule is valid.
        """
        return len(set(ch.sequence.length for ch in self._channels.values())) == 1

    def get_sequence(
        self,
        label: str,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
    ) -> PulseArray:
        """
        Returns the pulse sequence for a specific channel.

        Parameters
        ----------
        label : str
            The channel label.
        duration : float, optional
            The duration of the sequences.
        align : {"start", "end"}, optional
            The alignment of the sequences.

        Returns
        -------
        PulseArray
            The pulse sequence for the channel.
        """
        if duration is not None:
            pad_side: Literal["right", "left"] = "right" if align == "start" else "left"
            return self._channels[label].sequence.padded(duration, pad_side)
        else:
            return self._channels[label].sequence.copy()

    def get_sequences(
        self,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
    ) -> dict[str, PulseArray]:
        """
        Returns the pulse sequences.

        Parameters
        ----------
        duration : float, optional
            The duration of the sequences.
        align : {"start", "end"}, optional
            The alignment of the sequences.

        Returns
        -------
        dict[str, PulseArray]
            The pulse sequences.
        """
        return {
            label: self.get_sequence(label, duration, align) for label in self.labels
        }

    def get_sampled_sequence(
        self,
        label: str,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
    ) -> npt.NDArray[np.complex128]:
        """
        Returns the sampled pulse sequence for a specific channel.

        Parameters
        ----------
        label : str
            The channel label.
        duration : float, optional
            The duration of the sequences.
        align : {"start", "end"}, optional
            The alignment of the sequences.

        Returns
        -------
        npt.NDArray[np.complex128]
            The sampled pulse sequence for the channel.
        """
        sequence = self.get_sequence(label, duration, align)
        return sequence.values

    def get_sampled_sequences(
        self,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
    ) -> dict[str, npt.NDArray[np.complex128]]:
        """
        Returns the sampled pulse sequences.

        Parameters
        ----------
        duration : float, optional
            The duration of the sequences.
        align : {"start", "end"}, optional
            The alignment of the sequences.

        Returns
        -------
        dict[str, npt.NDArray[np.complex128]]
            The sampled pulse sequences
        """
        return {
            label: self.get_sampled_sequence(label, duration, align)
            for label in self.labels
        }

    def get_final_frame_shift(
        self,
        label: str,
    ) -> float:
        """
        Returns the final frame shift for a specific channel.

        Parameters
        ----------
        label : str
            The channel label.

        Returns
        -------
        float
            The final frame shift.
        """
        phase = self._channels[label].sequence.final_frame_shift
        phase = (phase + np.pi) % (np.pi * 2) - np.pi
        return phase

    def get_final_frame_shifts(
        self,
    ) -> dict[str, float]:
        """
        Returns the final frame shifts.

        Returns
        -------
        dict[str, float]
            The final frame shifts.
        """
        return {label: self.get_final_frame_shift(label) for label in self.labels}

    def get_pulse_ranges(
        self,
        targets: list[str] | None = None,
    ) -> dict[str, list[range]]:
        """
        Returns the pulse ranges.

        Parameters
        ----------
        targets : list[str], optional
            The target labels.

        Returns
        -------
        dict[str, list[range]]
            The pulse ranges.
        """
        targets = targets or self.labels
        ranges: dict[str, list[range]] = {target: [] for target in targets}
        for target in targets:
            current_offset = 0
            for waveform in self._channels[target].sequence.get_waveforms():
                next_offset = current_offset + waveform.length
                if not isinstance(waveform, Blank):
                    ranges[target].append(range(current_offset, next_offset))
                current_offset = next_offset
        return ranges

    def _add_channels_if_not_exist(self, labels: list[str]):
        for label in labels:
            if label not in self.labels:
                self._channels[label] = Channel(label)
                if self._global_offset > 0:
                    self.add(label, Blank(duration=self._global_offset))

    def _max_offset(
        self,
        labels: list[str] | None = None,
    ) -> float:
        if labels is None:
            offsets = [channel.offset for channel in self._channels.values()]
        else:
            offsets = [self._channels[label].offset for label in labels]

        max_offset = max(offsets, default=0.0)

        return max_offset

    @staticmethod
    def _downsample(
        data: npt.NDArray,
        n_max_points: int,
    ) -> npt.NDArray:
        if len(data) <= n_max_points:
            return data
        indices = np.linspace(0, len(data) - 1, n_max_points).astype(int)
        return data[indices]
