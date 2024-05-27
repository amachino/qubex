from __future__ import annotations

from datetime import datetime
from typing import Generic, Optional, TypeVar

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .fitting import RabiParam
from .typing import TargetMap


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
                    marker=dict(symbol="circle", size=8, color="#636EFA"),
                    line=dict(width=1, color="grey", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.sweep_range,
                    y=self.data.imag,
                    mode="lines+markers",
                    marker=dict(symbol="circle", size=8, color="#EF553B"),
                    line=dict(width=1, color="grey", dash="dash"),
                )
            )
            fig.update_layout(
                title=self.target,
                xaxis_title="Sweep value",
                yaxis_title="Measured value",
                width=600,
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
                    marker=dict(symbol="circle", size=8, color="#636EFA"),
                    line=dict(width=1, color="grey", dash="dash"),
                )
            )
            fig.update_layout(
                title=self.target,
                xaxis_title="Sweep value",
                yaxis_title="Normalized value",
                width=600,
            )
            fig.show()


class AmplitudeRabiRelation(TargetResult):
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
                marker=dict(symbol="circle", size=8, color="#636EFA"),
                line=dict(width=1, color="grey", dash="dash"),
            )
        )
        fig.update_layout(
            title="Relation between control amplitude and Rabi rate",
            xaxis_title="Control amplitude (arb. unit)",
            yaxis_title="Rabi rate (MHz)",
            width=600,
        )
        fig.show()


class PhaseShiftData(TargetResult):
    """
    Data class representing the result of a phase shift experiment.

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
                marker=dict(symbol="circle", size=8, color="#636EFA"),
                line=dict(width=1, color="grey", dash="dash"),
            )
        )
        fig.update_layout(
            title=f"Phase shift of {self.target} : {self.phase_shift:.5g} rad/chunk",
            xaxis_title="Control window (ns)",
            yaxis_title="Phase (rad)",
            width=600,
        )
        fig.show()
