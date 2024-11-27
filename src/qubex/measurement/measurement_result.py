from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ..analysis import plot_fft, plot_waveform, scatter_iq_data
from ..backend import SAMPLING_PERIOD

SAMPLING_PERIOD_SINGLE = SAMPLING_PERIOD
SAMPLING_PERIOD_AVG = SAMPLING_PERIOD * 4


class MeasureMode(Enum):
    SINGLE = "single"
    AVG = "avg"

    @property
    def integral_mode(self) -> str:
        return {
            MeasureMode.SINGLE: "single",
            MeasureMode.AVG: "integral",
        }[self]


@dataclass(frozen=True)
class MeasureData:
    target: str
    mode: MeasureMode
    raw: NDArray
    kerneled: NDArray
    classified: NDArray
    n_states: int | None

    @property
    def length(self) -> int:
        return len(self.raw)

    @property
    def times(self) -> NDArray[np.float64]:
        if self.mode == MeasureMode.SINGLE:
            return np.arange(self.length) * SAMPLING_PERIOD_SINGLE
        elif self.mode == MeasureMode.AVG:
            return np.arange(self.length) * SAMPLING_PERIOD_AVG
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @property
    def counts(self) -> dict[str, int]:
        if len(self.classified) == 0 or self.n_states is None:
            raise ValueError("No classification data available")
        classified_labels = self.classified
        count = np.bincount(classified_labels, minlength=self.n_states)
        state = {str(label): count[label] for label in range(len(count))}
        return state

    @property
    def probabilities(self) -> NDArray[np.float64]:
        if len(self.classified) == 0:
            raise ValueError("No classification data available")
        total = sum(self.counts.values())
        return np.array([count / total for count in self.counts.values()])

    @property
    def standard_deviations(self) -> NDArray[np.float64]:
        if len(self.classified) == 0:
            raise ValueError("No classification data available")
        return np.sqrt(
            self.probabilities * (1 - self.probabilities) / sum(self.counts.values())
        )

    def plot(self):
        if self.mode == MeasureMode.SINGLE:
            scatter_iq_data(
                data={self.target: self.kerneled},
                title=f"Readout IQ data of {self.target}",
            )
        elif self.mode == MeasureMode.AVG:
            plot_waveform(
                data=self.raw,
                sampling_period=SAMPLING_PERIOD_AVG,
                title=f"Readout waveform of {self.target}",
                xlabel="Capture time (ns)",
                ylabel="Amplitude (arb. units)",
            )

    def plot_fft(self):
        plot_fft(
            times=self.times,
            data=self.raw,
            title=f"Fourier transform of {self.target}",
            xlabel="Frequency (GHz)",
            ylabel="Amplitude (arb. units)",
        )


@dataclass(frozen=True)
class MeasureResult:
    mode: MeasureMode
    data: dict[str, MeasureData]
    config: dict

    @property
    def counts(self) -> dict[str, int]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        classified_data = np.column_stack(
            [data.classified for data in self.data.values()]
        )
        classified_labels = np.array(
            ["".join(map(str, row)) for row in classified_data]
        )
        counts = dict(Counter(classified_labels))
        counts = {key: counts[key] for key in sorted(counts.keys())}
        return counts

    def plot(self):
        if self.mode == MeasureMode.SINGLE:
            data = {qubit: data.kerneled for qubit, data in self.data.items()}
            scatter_iq_data(data=data)
        elif self.mode == MeasureMode.AVG:
            for qubit in self.data.values():
                qubit.plot()

    def plot_fft(self):
        for qubit in self.data.values():
            qubit.plot_fft()
