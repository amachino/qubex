from __future__ import annotations

from copy import deepcopy
from typing import Any, Collection, Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..style import COLORS
from .blank import Blank
from .pulse import Waveform
from .pulse_array import PhaseShift, PulseArray


class PulseSchedule:
    def __init__(
        self,
        targets: list[str] | dict[str, Any],
    ):
        """
        A class to represent a pulse schedule.

        Parameters
        ----------
        targets : list[str] | dict[str, Any]
            The control targets.

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
        if isinstance(targets, dict):
            self.targets = targets
        else:
            self.targets = {
                target: {
                    "frequency": None,
                    "object": None,
                }
                for target in targets
            }

        self._channels = {target: PulseArray() for target in targets}
        self._offsets = {target: 0.0 for target in targets}

    @property
    def labels(self) -> list[str]:
        """
        Returns the target labels.
        """
        return list(self.targets.keys())

    @property
    def frequencies(self) -> dict[str, float | None]:
        """
        Returns the target frequencies of the sequences.
        """
        return {
            target: props.get("frequency") for target, props in self.targets.items()
        }

    @property
    def objects(self) -> dict[str, str | None]:
        """
        Returns the target objects of the sequences.
        """
        return {target: props.get("object") for target, props in self.targets.items()}

    @property
    def channels(self) -> dict[str, PulseArray]:
        """
        Returns the pulse channels.
        """
        return self._channels.copy()

    @property
    def sampled_sequences(self) -> dict[str, npt.NDArray[np.complex128]]:
        """
        Returns the sampled pulse sequences.
        """
        return self.get_sampled_sequences()

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
    def length(self) -> int:
        """
        Returns the length of the pulse schedule in samples.
        """
        if not self.is_valid():
            raise ValueError("Inconsistent sequence lengths.")
        return next(iter(self._channels.values())).length

    @property
    def duration(self) -> float:
        """
        Returns the duration of the pulse schedule in ns.
        """
        return self.length * Waveform.SAMPLING_PERIOD

    def add(
        self,
        target: str,
        obj: Waveform | PhaseShift,
    ):
        """
        Add a waveform or a phase shift to the pulse schedule.

        Parameters
        ----------
        target : str
            The target label.
        obj : Waveform | PhaseShift
            The waveform or phase shift to add.

        Raises
        ------
        ValueError
            If the target is not valid.

        Examples
        --------
        >>> with PulseSchedule(["Q01"]) as seq:
        ...     seq.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        """
        if target not in self.targets:
            raise ValueError(f"Invalid target: {target}")

        self._channels[target].add(obj)

        if isinstance(obj, Waveform):
            self._offsets[target] += obj.duration

    def barrier(
        self,
        targets: Collection[str] | None = None,
    ):
        """
        Add a barrier to the pulse schedule.

        Parameters
        ----------
        targets : list[str], optional
            The target labels to add the barrier to.

        Examples
        --------
        >>> with PulseSchedule(["Q01"]) as seq:
        ...     seq.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     seq.barrier()
        """
        if targets is None:
            targets = list(self.targets)
        else:
            targets = list(targets)
        for target in targets:
            self.add(
                target,
                Blank(duration=self._max_offset(targets) - self._offsets[target]),
            )

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

        for target in schedule.targets:
            if target not in self.targets:
                raise ValueError(f"The target {target} is not in the current schedule.")

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
        new = PulseSchedule(self.targets)
        for target, sequence in self._channels.items():
            new.add(target, sequence.scaled(scale))
        return new

    def detuned(self, detuning: float) -> PulseSchedule:
        """
        Returns a detuned pulse schedule.
        """
        new = PulseSchedule(self.targets)
        for target, sequence in self._channels.items():
            new.add(target, sequence.detuned(detuning))
        return new

    def shifted(self, phase: float) -> PulseSchedule:
        """
        Returns a shifted pulse schedule.
        """
        new = PulseSchedule(self.targets)
        for target, sequence in self._channels.items():
            new.add(target, sequence.shifted(phase))
        return new

    def repeated(self, n: int) -> PulseSchedule:
        """
        Returns a repeated pulse schedule.
        """
        new = PulseSchedule(self.targets)
        for _ in range(n):
            new.call(self)
        return new

    def reversed(self) -> PulseSchedule:
        """Returns a time-reversed pulse schedule."""
        new = PulseSchedule(self.targets)
        for target, sequence in self._channels.items():
            new.add(target, sequence.reversed())
        return new

    def plot(
        self,
        *,
        title: str = "Pulse Sequence",
        width: int = 800,
        n_samples: int = 1024,
        divide_by_two_pi: bool = False,
        time_unit: Literal["ns", "samples"] = "ns",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
        show_phase: bool = True,
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

        n_targets = len(self.targets)

        if n_targets == 0:
            print("No data to plot.")
            return

        sequences = self.get_sequences()

        fig = make_subplots(
            rows=n_targets,
            cols=1,
            shared_xaxes=True,
            specs=[[{"secondary_y": True}] for _ in range(n_targets)],
        )
        for i, (target, seq) in enumerate(sequences.items()):
            if time_unit == "ns":
                times = np.append(seq.times, seq.times[-1] + seq.SAMPLING_PERIOD)
            else:
                times = np.arange(seq.length + 1)
            real = np.append(seq.real, seq.real[-1])
            imag = np.append(seq.imag, seq.imag[-1])
            phase = np.append(seq.frame_shifts, seq.final_frame_shift)
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
                    name="I",
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
                    name="Q",
                    mode="lines",
                    line_shape=line_shape,
                    line=dict(color=COLORS[1]),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
                secondary_y=False,
            )
            if show_phase:
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
            height=80 * n_targets + 140,
            width=width,
        )
        fig.update_xaxes(
            row=n_targets,
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
            fig.update_yaxes(
                row=i + 1,
                col=1,
                range=[-np.pi * 1.2, np.pi * 1.2],
                tickvals=[-np.pi, 0, np.pi],
                ticktext=["-π", "0", "π"],
                secondary_y=True,
            )
            annotations = []
            if self.frequencies.get(target) is not None:
                annotations.append(f"{self.frequencies[target]:.2f} GHz")
            if self.objects.get(target) is not None:
                annotations.append(f"{self.objects[target]}")
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
        return len(set(seq.length for seq in self._channels.values())) == 1

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
        dict[str, PulseSequence]
            The pulse sequences.
        """
        if duration is not None:
            pad_side: Literal["right", "left"] = "right" if align == "start" else "left"
            return {
                target: seq.padded(duration, pad_side)
                for target, seq in self._channels.items()
            }
        else:
            return self._channels.copy()

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
        sequences = self.get_sequences(duration, align)
        return {target: sequence.values for target, sequence in sequences.items()}

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
            for waveform in self._channels[target].waveforms:
                next_offset = current_offset + waveform.length
                if not isinstance(waveform, Blank):
                    ranges[target].append(range(current_offset, next_offset))
                current_offset = next_offset
        return ranges

    def _max_offset(
        self,
        targets: list[str] | None = None,
    ) -> float:
        if targets is None:
            offsets = list(self._offsets.values())
        else:
            offsets = [self._offsets[target] for target in targets]

        return max(offsets, default=0.0)

    @staticmethod
    def _downsample(
        data: npt.NDArray,
        n_max_points: int,
    ) -> npt.NDArray:
        if len(data) <= n_max_points:
            return data
        indices = np.linspace(0, len(data) - 1, n_max_points).astype(int)
        return data[indices]
