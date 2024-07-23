from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..style import COLORS
from .pulse import Blank, Waveform
from .pulse_sequence import PhaseShift, PulseSequence


class PulseSchedule:
    """
    Examples
    --------
    with PulseSchedule(["Q01", "RQ01", "Q02", "RQ02"]) as ps:
        ps.add("Q01", FlatTop(duration=30, amplitude=1, tau=10))
        ps.barrier()
        ps.add("Q02", FlatTop(duration=100, amplitude=1, tau=10))
        ps.barrier()
        ps.add("RQ01", FlatTop(duration=200, amplitude=1, tau=10))
        ps.add("RQ02", FlatTop(duration=200, amplitude=1, tau=10))

    ps.plot()
    """

    def __init__(
        self,
        targets: list[str],
    ):
        self.targets = targets
        self._sequences = {target: PulseSequence() for target in targets}
        self._offsets = {target: 0.0 for target in targets}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.barrier()

    def get_sequences(
        self,
        duration: float | None = None,
        align: Literal["start", "end"] = "start",
    ) -> dict[str, PulseSequence]:
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
        sequences = self.get_sequences(duration, align)
        return {target: sequence.values for target, sequence in sequences.items()}

    def _max_offset(
        self,
        targets: list[str] | None = None,
    ) -> float:
        if targets is None:
            offsets = list(self._offsets.values())
        else:
            offsets = [self._offsets[target] for target in targets]

        return max(offsets, default=0.0)

    def add(
        self,
        target: str,
        obj: Waveform | PhaseShift,
    ):
        if target not in self.targets:
            raise ValueError(f"Invalid target: {target}")

        self._sequences[target].add(obj)

        if isinstance(obj, Waveform):
            self._offsets[target] += obj.duration

    def barrier(
        self,
        targets: list[str] | None = None,
    ):
        targets = targets or self.targets
        for target in targets:
            self.add(target, Blank(duration=self._max_offset() - self._offsets[target]))

    def plot(self):
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
            times = np.append(seq.times, seq.times[-1] + seq.SAMPLING_PERIOD)
            real = np.append(seq.real, seq.real[-1])
            imag = np.append(seq.imag, seq.imag[-1])
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=real,
                    name="I",
                    mode="lines",
                    line_shape="hv",
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
                    line_shape="hv",
                    line=dict(color=COLORS[1]),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )
        fig.update_layout(
            title="Pulse Schedule",
            template="plotly",
        )
        fig.update_xaxes(
            row=n_targets,
            col=1,
            title_text="Time (ns)",
        )
        for i, (target, seq) in enumerate(sequences.items()):
            y_max = np.max(seq.abs)
            fig.update_yaxes(
                row=i + 1,
                col=1,
                title_text=target,
                range=[-1.1 * y_max, 1.1 * y_max],
            )
        fig.show()
