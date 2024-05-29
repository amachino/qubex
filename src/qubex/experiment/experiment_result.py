from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .. import fitting
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

    def fit(self, *args, **kwargs):
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

    def plot(
        self,
        *args,
        **kwargs,
    ):
        for target in self.data:
            rabi_param = self.rabi_params[target] if self.rabi_params else None
            self.data[target].plot(*args, rabi_param=rabi_param, **kwargs)

    def fit(self) -> TargetMap[Any]:
        return {target: self.data[target].fit() for target in self.data}


class RabiResult(TargetResult):
    """
    Data class representing the result of a Rabi oscillation experiment.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    time_range : NDArray
        Time range of the experiment.
    rabi_param : RabiParam
        Parameters of the Rabi oscillation.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        data: NDArray,
        time_range: NDArray,
        rabi_param: RabiParam,
    ):
        super().__init__(
            target=target,
            data=data,
        )
        self.time_range = time_range
        self.rabi_param = rabi_param

    @property
    def rotated(self) -> NDArray:
        param = self.rabi_param
        return self.data * np.exp(-1j * param.angle)

    @property
    def normalized(self) -> NDArray:
        param = self.rabi_param
        values = self.data * np.exp(-1j * param.angle)
        values_normalized = (values.imag - param.offset) / param.amplitude
        return values_normalized

    def plot(
        self,
        *,
        normalize: bool,
        **kwargs,
    ):
        if normalize:
            values = self.normalized
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.time_range,
                    y=values,
                    error_y=dict(
                        type="constant",
                        value=self.rabi_param.noise / self.rabi_param.amplitude,
                    ),
                )
            )
            fig.update_layout(
                title=f"Rabi oscillation of {self.target} : {self.rabi_param.frequency * 1e3:.2f} MHz",
                xaxis_title="Drive time (ns)",
                yaxis_title="Normalized value",
            )
            fig.show()
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.time_range,
                    y=self.data.real,
                    name="I",
                )
            )
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.time_range,
                    y=self.data.imag,
                    name="Q",
                )
            )
            fig.update_layout(
                title=f"Rabi oscillation of {self.target} : {self.rabi_param.frequency * 1e3:.2f} MHz",
                xaxis_title="Drive time (ns)",
                yaxis_title="Measured value",
            )
            fig.show()

    def fit(self) -> RabiParam:
        return fitting.fit_rabi(
            target=self.target,
            times=self.time_range,
            data=self.data,
        )


class SweepResult(TargetResult):
    """
    Data class representing the result of a sweep experiment.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    sweep_range : NDArray
        Sweep range of the experiment.
    sweep_value_label : str
        Label of the sweep value.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        data: NDArray,
        sweep_range: NDArray,
        sweep_value_label: str,
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

    def plot(
        self,
        *,
        normalize: bool,
        rabi_param: Optional[RabiParam] = None,
    ):
        if normalize:
            if rabi_param is None:
                raise ValueError("rabi_param must be provided for normalization.")
            values = self.normalized(rabi_param)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.sweep_range,
                    y=values,
                    error_y=dict(
                        type="constant",
                        value=rabi_param.noise / rabi_param.amplitude,
                    ),
                )
            )
            fig.update_layout(
                title=f"Measured value : {self.target}",
                xaxis_title=self.sweep_value_label,
                yaxis_title="Normalized value",
            )
            fig.show()
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.sweep_range,
                    y=self.data.real,
                    name="I",
                )
            )
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.sweep_range,
                    y=self.data.imag,
                    name="Q",
                )
            )
            fig.update_layout(
                title=f"Measured I/Q value : {self.target}",
                xaxis_title=self.sweep_value_label,
                yaxis_title="Measured value",
            )
            fig.show()


class AmplRabiRelation(TargetResult):
    """
    The relation between the drive amplitude and the Rabi rate.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    sweep_range : NDArray
        Sweep range of the experiment.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        data: NDArray,
        sweep_range: NDArray,
    ):
        super().__init__(target, data)
        self.sweep_range = sweep_range

    def plot(self, *args, **kwargs):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.sweep_range,
                y=self.data * 1e3,
            )
        )
        fig.update_layout(
            title=f"Relation between drive amplitude and Rabi rate : {self.target}",
            xaxis_title="Drive amplitude (arb. units)",
            yaxis_title="Rabi rate (MHz)",
        )
        fig.show()


class FreqRabiRelation(TargetResult):
    """
    The relation between the drive frequency and the Rabi rate.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    sweep_range : NDArray
        Sweep range of the experiment.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        data: NDArray,
        sweep_range: NDArray,
        frequency_range: NDArray,
    ):
        super().__init__(target, data)
        self.sweep_range = sweep_range
        self.frequency_range = frequency_range

    def plot(self, *args, **kwargs):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.frequency_range,
                y=self.data * 1e3,
            )
        )
        fig.update_layout(
            title=f"Relation between drive frequency and Rabi rate : {self.target}",
            xaxis_title="Drive frequency (GHz)",
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
    data : NDArray
        Measured data.
    sweep_range : NDArray
        Sweep range of the experiment.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        data: NDArray,
        target: str,
        sweep_range: NDArray,
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

    def plot(self, *args, **kwargs):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.sweep_range,
                y=self.phases,
            )
        )
        fig.update_layout(
            title=f"Phase shift of {self.target} : {self.phase_shift:.5g} rad/chunk",
            xaxis_title="Control window (ns)",
            yaxis_title="Phase (rad)",
        )
        fig.show()
