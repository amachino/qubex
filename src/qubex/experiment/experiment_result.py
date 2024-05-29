from __future__ import annotations

from datetime import datetime
from typing import Generic, Optional, TypeVar

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from ..fitting import RabiParam
from ..typing import TargetMap


class TargetResult:
    """
    Data class representing the result of an experiment for a target.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        data: NDArray,
    ):
        self.target = target
        self.data = data
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def plot(self, *args, **kwargs):
        raise NotImplementedError


T = TypeVar("T", bound=TargetResult)


class ExperimentResult(Generic[T]):
    """
    Data class representing the result of an experiment.

    Attributes
    ----------
    data: TargetMap[TargetResult]
        Result of the experiment.
    rabi_params: TargetMap[RabiParam]
        Parameters of the Rabi oscillation.
    """

    def __init__(
        self,
        data: TargetMap[T],
        rabi_params: TargetMap[RabiParam] | None = None,
    ):
        self.data = data
        self.rabi_params = rabi_params
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def plot(self):
        rabi_params = self.rabi_params
        for target in self.data:
            if rabi_params is None:
                self.data[target].plot()
            else:
                self.data[target].plot(rabi_params[target])


class SweepResult(TargetResult):
    """
    Data class representing the result of a sweep experiment.

    Attributes
    ----------
    target : str
        Target of the experiment.
    sweep_range : NDArray
        Sweep range of the experiment.
    sweep_value_label : str
        Label of the sweep value.
    data : NDArray
        Measured data.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        sweep_range: NDArray,
        sweep_value_label: str,
        data: NDArray,
    ):
        super().__init__(target, data)
        self.sweep_range = sweep_range
        self.sweep_value_label = sweep_value_label

    def rotated(self, param: RabiParam) -> NDArray:
        return self.data * np.exp(-1j * param.angle)

    def normalized(self, param: RabiParam) -> NDArray:
        values = self.data * np.exp(-1j * param.angle)
        values_normalized = (values.imag - param.offset) / param.amplitude
        return values_normalized

    def plot(self, param: Optional[RabiParam] = None):
        if param is None:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.sweep_range,
                    y=self.data.real,
                    mode="lines+markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.sweep_range,
                    y=self.data.imag,
                    mode="lines+markers",
                )
            )
            fig.update_layout(
                title=f"Measured value : {self.target}",
                xaxis_title=self.sweep_value_label,
                yaxis_title="Measured value",
            )
            fig.show()
        else:
            values = self.normalized(param)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.sweep_range,
                    y=values,
                    mode="lines+markers",
                    marker_color="black",
                )
            )
            fig.update_layout(
                title=f"Measured value : {self.target}",
                xaxis_title=self.sweep_value_label,
                yaxis_title="Normalized value",
            )
            fig.show()


class AmplRabiRelation(TargetResult):
    """
    The relation between the control amplitude and the Rabi rate.

    Attributes
    ----------
    target : str
        Target of the experiment.
    sweep_range : NDArray
        Sweep range of the experiment.
    data : NDArray
        Measured data.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        sweep_range: NDArray,
        data: NDArray,
    ):
        super().__init__(target, data)
        self.sweep_range = sweep_range

    def plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.sweep_range,
                y=self.data * 1e3,
                mode="lines+markers",
                marker_color="black",
            )
        )
        fig.update_layout(
            title=f"Relation between control amplitude and Rabi rate : {self.target}",
            xaxis_title="Control amplitude (arb. units)",
            yaxis_title="Rabi rate (MHz)",
        )
        fig.show()


class FreqRabiRelation(TargetResult):
    """
    The relation between the control frequency and the Rabi rate.

    Attributes
    ----------
    target : str
        Target of the experiment.
    sweep_range : NDArray
        Sweep range of the experiment.
    data : NDArray
        Measured data.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        sweep_range: NDArray,
        frequency_range: NDArray,
        data: NDArray,
    ):
        super().__init__(target, data)
        self.sweep_range = sweep_range
        self.frequency_range = frequency_range

    def plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.frequency_range,
                y=self.data * 1e3,
                mode="lines+markers",
                marker_color="black",
            )
        )
        fig.update_layout(
            title=f"Relation between control frequency and Rabi rate : {self.target}",
            xaxis_title="Control frequency (GHz)",
            yaxis_title="Rabi rate (MHz)",
        )
        fig.show()


class TimePhaseRelation(TargetResult):
    """
    The relation between the control window and the phase shift.

    Attributes
    ----------
    target : str
        Target of the experiment.
    sweep_range : NDArray
        Sweep range of the experiment.
    data : NDArray
        Measured data.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        sweep_range: NDArray,
        data: NDArray,
    ):
        super().__init__(target, data)
        self.sweep_range = sweep_range

    @property
    def phases(self) -> NDArray[np.float64]:
        return np.angle(self.data)

    @property
    def phase_diffs(self) -> NDArray[np.float64]:
        delta_phases = np.diff(self.phases)
        delta_phases[delta_phases < 0] += 2 * np.pi
        return delta_phases

    @property
    def phase_shift(self) -> float:
        """Return the average phase shift per chunk."""
        return np.mean(self.phase_diffs).astype(float)

    def plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.sweep_range,
                y=self.phases,
                mode="lines+markers",
                marker_color="black",
            )
        )
        fig.update_layout(
            title=f"Phase shift of {self.target} : {self.phase_shift:.5g} rad/chunk",
            xaxis_title="Control window (ns)",
            yaxis_title="Phase (rad)",
        )
        fig.show()
