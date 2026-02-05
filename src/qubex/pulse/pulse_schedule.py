from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Collection, Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing_extensions import deprecated

from ..style import COLORS
from .blank import Blank
from .pulse_array import PhaseShift, PulseArray
from .waveform import Waveform


@dataclass
class PulseChannel:
    label: str
    sequence: PulseArray = field(default_factory=PulseArray)
    frequency: float | None = None
    target: str | None = None
    frame: str | None = None


class PulseSchedule:
    def __init__(
        self,
        channels: list[str] | list[PulseChannel] | None = None,
        /,
    ):
        """
        A class to represent a pulse schedule.

        Parameters
        ----------
        channels : list[str] | list[PulseChannel], optional
            The channels of the pulse schedule.

        Examples
        --------
        >>> from qubex.pulse import PulseSchedule, FlatTop
        >>> with PulseSchedule() as ps:
        ...     ps.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     ps.barrier()
        ...     ps.add("Q02", FlatTop(duration=100, amplitude=1, tau=10))
        ...     ps.barrier()
        ...     ps.add("RQ01", FlatTop(duration=200, amplitude=1, tau=10))
        ...     ps.add("RQ02", FlatTop(duration=200, amplitude=1, tau=10))
        >>> ps.plot()
        """
        self._channels: dict[str, PulseChannel] = {}

        if channels is not None:
            if isinstance(channels, list):
                for channel in channels:
                    if isinstance(channel, str):
                        self._channels[channel] = PulseChannel(label=channel)
                    elif isinstance(channel, PulseChannel):
                        self._channels[channel.label] = deepcopy(channel)
                    else:
                        raise ValueError("Invalid channels.")
            else:
                raise ValueError("Invalid channels.")

        self._offsets = defaultdict(lambda: 0.0)
        self._global_offset = 0.0

    def __enter__(self):
        """
        Enter the context manager.

        Examples
        --------
        >>> with PulseSchedule() as ps:
        ...     ps.add(...)
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager and add a barrier to the sequence.

        The following codes are equivalent:

        >>> ps = PulseSchedule()
        >>> ps.add(...)
        >>> ps.barrier()

        >>> with PulseSchedule([...]) as ps:
        ...     ps.add(...)

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
        # NOTE:
        #   Using floor division (//) with floating point numbers can lead to an off-by-one
        #   error due to binary representation (e.g. 100 // 0.1 -> 999.0 instead of 1000.0).
        #   We therefore compute the ratio and round it to the nearest integer.
        #   This assumes that duration is always intended to be an integer multiple of
        #   Waveform.SAMPLING_PERIOD within normal floating point tolerance.
        return int(round(self.duration / Waveform.SAMPLING_PERIOD))

    @property
    def duration(self) -> float:
        """
        Returns the duration of the pulse schedule in ns.
        """
        return self._max_offset()

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
        >>> with PulseSchedule() as ps:
        ...     ps.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        """
        self._add_channels_if_not_exist([label])

        self._channels[label].sequence.add(obj)

        if isinstance(obj, Waveform):
            self._offsets[label] += obj.cached_duration

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
        >>> with PulseSchedule() as ps:
        ...     ps.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     ps.barrier()
        """
        if labels is None:
            labels = self.labels
            self._global_offset = self._max_offset()
        else:
            labels = list(labels)

        self._add_channels_if_not_exist(labels)

        for label in labels:
            diff = self._max_offset(labels) - self._offsets[label]
            if diff > 0:
                self.add(label, Blank(duration=diff))

    def call(
        self,
        schedule: PulseSchedule,
        copy: bool = False,
    ):
        """
        Call another pulse schedule in the current pulse schedule.

        Parameters
        ----------
        schedule : PulseSchedule
            The pulse schedule to call.

        Examples
        --------
        >>> with PulseSchedule() as ctrl:
        ...     ctrl.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     ctrl.barrier()
        ...     ctrl.add("Q02", FlatTop(duration=100, amplitude=1, tau=10))
        >>> with PulseSchedule() as read:
        ...     read.add("RQ01", FlatTop(duration=200, amplitude=1, tau=10))
        ...     read.add("RQ02", FlatTop(duration=200, amplitude=1, tau=10))
        >>> with PulseSchedule() as ps:
        ...     ps.call(ctrl)
        ...     ps.barrier()
        ...     ps.call(read)
        >>> ps.plot()
        """
        if schedule == self:
            raise ValueError("Cannot call itself.")

        self.barrier(schedule.labels)
        sequences = schedule.get_sequences(copy=copy)
        for label, sequence in sequences.items():
            self.add(label, sequence)

    def copy(self) -> PulseSchedule:
        """
        Returns a copy of the pulse schedule.
        """
        return deepcopy(self)

    def padded(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> PulseSchedule:
        """
        Returns a copy of the pulse schedule with zero padding.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse schedule in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        duration = total_duration - self.duration
        if duration < 0:
            raise ValueError(
                f"Total duration ({total_duration}) must be greater than the current duration ({self.duration})."
            )
        with PulseSchedule() as new_sched:
            for label, channel in self._channels.items():
                new_sched.add(label, channel.sequence.padded(total_duration, pad_side))
        return new_sched

    def pad(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> None:
        """
        Pads the pulse schedule with blank pulses.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse schedule in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        duration = total_duration - self.duration
        if duration < 0:
            raise ValueError(
                f"Total duration ({total_duration}) must be greater than the current duration ({self.duration})."
            )
        for channel in self._channels.values():
            channel.sequence.pad(total_duration, pad_side)
            self._offsets[channel.label] = total_duration
        self._global_offset = total_duration

    def scaled(self, scale: float) -> PulseSchedule:
        """
        Returns a scaled pulse schedule.
        """
        if scale == 1:
            return self
        new_sched = deepcopy(self)
        for channel in new_sched._channels.values():
            channel.sequence._scale *= scale
        return new_sched

    def detuned(self, detuning: float) -> PulseSchedule:
        """
        Returns a detuned pulse schedule.
        """
        if detuning == 0:
            return self
        new_sched = deepcopy(self)
        for channel in new_sched._channels.values():
            channel.sequence._detuning += detuning
        return new_sched

    def shifted(self, phase: float) -> PulseSchedule:
        """
        Returns a shifted pulse schedule.
        """
        if phase == 0:
            return self
        new_sched = deepcopy(self)
        for channel in new_sched._channels.values():
            channel.sequence._phase += phase
        return new_sched

    def repeated(self, n: int) -> PulseSchedule:
        """
        Returns a repeated pulse schedule.
        """
        if n == 1:
            return self
        new_sched = PulseSchedule()
        for label, channel in self._channels.items():
            new_sched.add(label, channel.sequence.repeated(n))
            new_sched._channels[label].frequency = channel.frequency
            new_sched._channels[label].target = channel.target
        return new_sched

    @deprecated(
        "The `reversed` method is deprecated, use `inverted` instead.",
    )
    def reversed(self) -> PulseSchedule:
        return self.inverted()

    def inverted(self) -> PulseSchedule:
        """Returns an inverted pulse schedule."""
        with PulseSchedule() as new_sched:
            for label, channel in self._channels.items():
                new_sched.add(label, channel.sequence.inverted())
        return new_sched

    def plot(
        self,
        *,
        show_physical_pulse: bool = False,
        title: str = "Pulse Sequence",
        width: int = 800,
        n_samples: int | None = None,
        divide_by_two_pi: bool = False,
        time_unit: Literal["ns", "samples"] = "ns",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
    ):
        """
        Plots the pulse schedule.

        Examples
        --------
        >>> with PulseSchedule() as ps:
        ...     ps.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     ps.barrier()
        ...     ps.add("Q02", FlatTop(duration=100, amplitude=1, tau=10))
        ...     ps.barrier()
        ...     ps.add("RQ01", FlatTop(duration=200, amplitude=1, tau=10))
        ...     ps.add("RQ02", FlatTop(duration=200, amplitude=1, tau=10))
        >>> ps.plot()
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
        for i, (label, seq) in enumerate(sequences.items()):
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

            if n_samples is not None and len(times) > n_samples:
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
        for i, (label, seq) in enumerate(sequences.items()):
            y_max = np.max(seq.abs)
            if divide_by_two_pi:
                y_max /= 2 * np.pi * 1e-3
            fig.update_yaxes(
                row=i + 1,
                col=1,
                title_text=label,
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
            frequency = self._channels[label].frequency
            target = self._channels[label].target
            if frequency is not None and target is not None:
                annotations = []
                annotations.append(f"{frequency:.2f} GHz")
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
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

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
        copy: bool = True,
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
        copy : bool, optional
            If True, returns a copy of the sequence.

        Returns
        -------
        PulseArray
            The pulse sequence for the channel.
        """
        sequence = self._channels[label].sequence
        if duration is not None:
            pad_side: Literal["right", "left"] = "right" if align == "start" else "left"
            if copy:
                return sequence.padded(duration, pad_side)
            else:
                sequence.pad(duration, pad_side)
                return sequence
        else:
            if copy:
                return sequence.copy()
            else:
                return sequence

    def get_sequences(
        self,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
        copy: bool = True,
    ) -> dict[str, PulseArray]:
        """
        Returns the pulse sequences.

        Parameters
        ----------
        duration : float, optional
            The duration of the sequences.
        align : {"start", "end"}, optional
            The alignment of the sequences.
        copy : bool, optional
            If True, returns a copy of the sequences.

        Returns
        -------
        dict[str, PulseArray]
            The pulse sequences.
        """
        return {
            label: self.get_sequence(
                label=label,
                duration=duration,
                align=align,
                copy=copy,
            )
            for label in self.labels
        }

    def get_sampled_sequence(
        self,
        label: str,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
        copy: bool = True,
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
        copy : bool, optional
            If True, returns a copy of the sequence.

        Returns
        -------
        npt.NDArray[np.complex128]
            The sampled pulse sequence for the channel.
        """
        sequence = self.get_sequence(
            label=label,
            duration=duration,
            align=align,
            copy=copy,
        )
        return sequence.values

    def get_sampled_sequences(
        self,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
        copy: bool = True,
    ) -> dict[str, npt.NDArray[np.complex128]]:
        """
        Returns the sampled pulse sequences.

        Parameters
        ----------
        duration : float, optional
            The duration of the sequences.
        align : {"start", "end"}, optional
            The alignment of the sequences.
        copy : bool, optional
            If True, returns a copy of the sequences.

        Returns
        -------
        dict[str, npt.NDArray[np.complex128]]
            The sampled pulse sequences
        """
        return {
            label: self.get_sampled_sequence(
                label=label,
                duration=duration,
                align=align,
                copy=copy,
            )
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
        labels: list[str] | None = None,
    ) -> dict[str, list[range]]:
        """
        Returns the pulse ranges.

        Parameters
        ----------
        labels : list[str], optional
            The channel labels.

        Returns
        -------
        dict[str, list[range]]
            The pulse ranges.
        """
        labels = labels or self.labels
        ranges: dict[str, list[range]] = {label: [] for label in labels}
        for label in labels:
            current_offset = 0
            for waveform in self._channels[label].sequence.get_flattened_waveforms(
                apply_frame_shifts=False
            ):
                next_offset = current_offset + waveform.length
                if not isinstance(waveform, Blank):
                    ranges[label].append(range(current_offset, next_offset))
                current_offset = next_offset
        return ranges

    def get_blank_ranges(
        self,
        labels: list[str] | None = None,
    ) -> dict[str, list[range]]:
        """
        Returns the blank ranges.

        Parameters
        ----------
        labels : list[str], optional
            The channel labels.

        Returns
        -------
        dict[str, list[range]]
            The blank ranges.
        """
        labels = labels or self.labels
        ranges: dict[str, list[range]] = {label: [] for label in labels}
        for label in labels:
            current_offset = 0
            for waveform in self._channels[label].sequence.get_flattened_waveforms(
                apply_frame_shifts=False
            ):
                next_offset = current_offset + waveform.length
                if isinstance(waveform, Blank):
                    ranges[label].append(range(current_offset, next_offset))
                current_offset = next_offset
        return ranges

    def get_frequency(
        self,
        label: str,
    ) -> float | None:
        """
        Returns the frequency for a specific channel.

        Parameters
        ----------
        label : str
            The channel label.

        Returns
        -------
        float | None
            The frequency.
        """
        return self._channels[label].frequency

    def get_frequencies(
        self,
    ) -> dict[str, float | None]:
        """
        Returns the frequencies.

        Returns
        -------
        dict[str, float | None]
            The frequencies.
        """
        return {label: self.get_frequency(label) for label in self.labels}

    def get_target(
        self,
        label: str,
    ) -> str | None:
        """
        Returns the target for a specific channel.

        Parameters
        ----------
        label : str
            The channel label.

        Returns
        -------
        str | None
            The target.
        """
        return self._channels[label].target

    def get_targets(
        self,
    ) -> dict[str, str | None]:
        """
        Returns the targets.

        Returns
        -------
        dict[str, str | None]
            The targets.
        """
        return {label: self.get_target(label) for label in self.labels}

    def get_frame(
        self,
        label: str,
    ) -> str | None:
        """
        Returns the frame for a specific channel.

        Parameters
        ----------
        label : str
            The channel label.

        Returns
        -------
        str | None
            The frame.
        """
        return self._channels[label].frame

    def get_frames(
        self,
    ) -> dict[str, str | None]:
        """
        Returns the frames.

        Returns
        -------
        dict[str, str | None]
            The frames.
        """
        return {label: self.get_frame(label) for label in self.labels}

    def get_offset(
        self,
        label: str,
    ) -> float:
        """
        Returns the offset for a specific channel.

        Parameters
        ----------
        label : str
            The channel label.

        Returns
        -------
        float
            The offset.
        """
        return self._offsets[label]

    def _add_channels_if_not_exist(self, labels: list[str]):
        for label in labels:
            if label not in self.labels:
                self._channels[label] = PulseChannel(label)
                if self._global_offset > 0:
                    self.add(label, Blank(duration=self._global_offset))

    def _max_offset(
        self,
        labels: list[str] | None = None,
    ) -> float:
        if labels is None:
            offsets = list(self._offsets.values())
        else:
            offsets = [self._offsets[label] for label in labels]

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
