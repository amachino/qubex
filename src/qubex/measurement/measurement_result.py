from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Collection

import numpy as np
from numpy.typing import NDArray

from ..analysis import visualization as viz
from ..backend import SAMPLING_PERIOD

SAMPLING_PERIOD_SINGLE = SAMPLING_PERIOD
SAMPLING_PERIOD_AVG = SAMPLING_PERIOD * 4


class MeasureMode(Enum):
    SINGLE = "single"
    AVG = "avg"

    @property
    def integral_mode(self) -> str:
        if self == MeasureMode.SINGLE:
            return "single"
        elif self == MeasureMode.AVG:
            return "integral"
        else:
            raise ValueError(f"Invalid mode: {self}")


@dataclass(frozen=True)
class MeasureData:
    target: str
    mode: MeasureMode
    raw: NDArray
    kerneled: NDArray
    classified: NDArray
    n_states: int | None = None

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

    def plot(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        if self.mode == MeasureMode.SINGLE:
            return viz.scatter_iq_data(
                data={self.target: self.kerneled},
                title=f"Readout IQ data : {self.target}",
                return_figure=return_figure,
                save_image=save_image,
            )
        elif self.mode == MeasureMode.AVG:
            return viz.plot_waveform(
                data=self.raw,
                sampling_period=SAMPLING_PERIOD_AVG,
                title=f"Readout waveform : {self.target}",
                xlabel="Capture time (ns)",
                ylabel="Signal (arb. units)",
                return_figure=return_figure,
                save_image=save_image,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def plot_fft(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        return viz.plot_fft(
            x=self.times,
            y=self.raw,
            title=f"Fourier transform : {self.target}",
            xlabel="Frequency (GHz)",
            ylabel="Signal (arb. units)",
            return_figure=return_figure,
            save_image=save_image,
        )


@dataclass(frozen=True)
class MeasureResult:
    mode: MeasureMode
    data: dict[str, MeasureData]
    config: dict

    @property
    def counts(self) -> dict[str, int]:
        return self.get_counts()

    @property
    def probabilities(self) -> dict[str, float]:
        return self.get_probabilities()

    @property
    def standard_deviations(self) -> dict[str, float]:
        return self.get_standard_deviations()

    def get_counts(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, int]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        if targets is None:
            targets = self.data.keys()
        classified_data = np.column_stack(
            [self.data[target].classified for target in targets]
        )
        classified_labels = np.array(
            ["".join(map(str, row)) for row in classified_data]
        )
        counts = dict(Counter(classified_labels))
        counts = dict(sorted(counts.items()))
        return counts

    def get_probabilities(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, float]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        total = sum(self.get_counts(targets).values())
        return {key: count / total for key, count in self.get_counts(targets).items()}

    def get_standard_deviations(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, float]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        return {
            key: np.sqrt(prob * (1 - prob) / total)
            for key, prob, total in zip(
                self.get_counts(targets).keys(),
                self.get_probabilities(targets).values(),
                self.get_counts(targets).values(),
            )
        }

    def plot(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        if self.mode == MeasureMode.SINGLE:
            data = {qubit: data.kerneled for qubit, data in self.data.items()}
            return viz.scatter_iq_data(
                data=data,
                return_figure=return_figure,
                save_image=save_image,
            )
        elif self.mode == MeasureMode.AVG:
            figures = []
            for data in self.data.values():
                fig = data.plot(
                    return_figure=return_figure,
                    save_image=save_image,
                )
                figures.append(fig)
            if return_figure:
                return figures
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def plot_fft(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        figures = []
        for data in self.data.values():
            fig = data.plot_fft(
                return_figure=return_figure,
                save_image=save_image,
            )
            figures.append(fig)
        if return_figure:
            return figures


@dataclass(frozen=True)
class MultipleMeasureResult:
    mode: MeasureMode
    data: dict[str, list[MeasureData]]
    config: dict
