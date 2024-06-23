from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from numpy.typing import NDArray

from ..analysis import plot_waveform, scatter_iq_data


class MeasureMode(Enum):
    SINGLE = "single"
    AVG = "avg"

    @property
    def integral_mode(self) -> str:
        return {
            MeasureMode.SINGLE: "single",
            MeasureMode.AVG: "integral",
        }[self]


@dataclass
class MeasureData:
    target: str
    mode: MeasureMode
    raw: NDArray
    kerneled: NDArray
    classified: dict[int, int] | None

    def plot(self):
        if self.mode == MeasureMode.SINGLE:
            scatter_iq_data(
                data={self.target: self.kerneled},
                title=f"Readout IQ data of {self.target}",
            )
        elif self.mode == MeasureMode.AVG:
            plot_waveform(
                data=self.raw,
                sampling_period=8,
                title=f"Readout waveform of {self.target}",
                xlabel="Capture time (ns)",
                ylabel="Amplitude (arb. units)",
            )


@dataclass
class MeasureResult:
    mode: MeasureMode
    data: dict[str, MeasureData]
    config: dict

    def plot(self):
        if self.mode == MeasureMode.SINGLE:
            data = {qubit: data.kerneled for qubit, data in self.data.items()}
            scatter_iq_data(data=data)
        elif self.mode == MeasureMode.AVG:
            for qubit in self.data.values():
                qubit.plot()
