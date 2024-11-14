from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..style import COLORS
from .pulse import Blank, Waveform
from .pulse_sequence import PhaseShift, PulseSequence


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
        >>> with PulseSchedule(["Q01", "RQ01", "Q02", "RQ02"]) as ps:
        ...     ps.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     ps.barrier()
        ...     ps.add("Q02", FlatTop(duration=100, amplitude=1, tau=10))
        ...     ps.barrier()
        ...     ps.add("RQ01", FlatTop(duration=200, amplitude=1, tau=10))
        ...     ps.add("RQ02", FlatTop(duration=200, amplitude=1, tau=10))
        >>> ps.plot()
        """
        if isinstance(targets, list):
            self.targets = {
                target: {
                    "frequency": None,
                    "object": None,
                }
                for target in targets
            }
        else:
            self.targets = targets

        self._sequences = {target: PulseSequence() for target in targets}
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
    def sequences(self) -> dict[str, PulseSequence]:
        """
        Returns the pulse sequences.
        """
        return self._sequences.copy()

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
        >>> with PulseSchedule([...]) as ps:
        ...     ps.add(...)
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager and add a barrier to the sequence.

        The following codes are equivalent:

        >>> ps = PulseSchedule([...])
        >>> ps.add(...)
        >>> ps.barrier()

        >>> with PulseSchedule([...]) as ps:
        ...     ps.add(...)

        Note that the duration of sequences might be different if the context manager is not used.
        """
        self.barrier()

    @property
    def length(self) -> int:
        """
        Returns the length of the pulse schedule in samples.
        """
        if not self.is_valid():
            raise ValueError("Inconsistent sequence lengths.")
        return next(iter(self._sequences.values())).length

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
        >>> with PulseSchedule(["Q01"]) as ps:
        ...     ps.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        """
        if target not in self.targets:
            raise ValueError(f"Invalid target: {target}")

        self._sequences[target].add(obj)

        if isinstance(obj, Waveform):
            self._offsets[target] += obj.duration

    def barrier(
        self,
        targets: list[str] | None = None,
    ):
        """
        Add a barrier to the pulse schedule.

        Parameters
        ----------
        targets : list[str], optional
            The target labels to add the barrier to.

        Examples
        --------
        >>> with PulseSchedule(["Q01"]) as ps:
        ...     ps.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ...     ps.barrier()
        """
        targets = targets or list(self.targets.keys())
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
        >>> with PulseSchedule(["Q01", "Q02", "RQ01", "RQ02"]) as ps:
        ...     ps.call(ctrl)
        ...     ps.barrier()
        ...     ps.call(read)
        >>> ps.plot()
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

    def repeated(self, n: int) -> PulseSchedule:
        """
        Returns a repeated pulse schedule.
        """
        new = PulseSchedule(self.targets)
        for _ in range(n):
            new.call(self)
        return new

    def plot(
        self,
        *,
        width: int = 800,
        n_max_points: int = 1024,
        time_unit: Literal["ns", "samples"] = "ns",
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "linear",
    ):
        """
        Plots the pulse schedule.

        Examples
        --------
        >>> from qubex.pulse import PulseSchedule, FlatTop
        >>> with PulseSchedule(["Q01", "RQ01", "Q02", "RQ02"]) as ps:
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

        n_targets = len(self.targets)

        if n_targets == 0:
            print("No data to plot.")
            return

        sequences = self.get_sequences()

        fig = make_subplots(
            rows=n_targets,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
        for i, (target, seq) in enumerate(sequences.items()):
            if time_unit == "ns":
                times = np.append(seq.times, seq.times[-1] + seq.SAMPLING_PERIOD)
            else:
                times = np.arange(seq.length + 1)
            real = np.append(seq.real, seq.real[-1])
            imag = np.append(seq.imag, seq.imag[-1])

            if len(times) > n_max_points:
                times = self._downsample(times, n_max_points)
                real = self._downsample(real, n_max_points)
                imag = self._downsample(imag, n_max_points)

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
            )
        fig.update_layout(
            title="Pulse Schedule",
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
            fig.update_yaxes(
                row=i + 1,
                col=1,
                title_text=target,
                range=[-1.1 * y_max, 1.1 * y_max],
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
                # set bg color to white
                bgcolor="rgba(255, 255, 255, 0.8)",
            )
        fig.show()

    def is_valid(self) -> bool:
        """
        Returns True if the pulse schedule is valid.
        """
        return len(set(seq.length for seq in self._sequences.values())) == 1

    def get_sequences(
        self,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
    ) -> dict[str, PulseSequence]:
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
                for target, seq in self._sequences.items()
            }
        else:
            return self._sequences.copy()

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
            for waveform in self._sequences[target].waveforms:
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
