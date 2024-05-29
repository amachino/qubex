from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .. import fitting
from ..fitting import RabiParam
from ..typing import TargetMap


class TargetData:
    """
    Data class representing some data of a target.

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


T = TypeVar("T", bound=TargetData)


class ExperimentResult(Generic[T]):
    """
    Data class representing the result of an experiment.

    Attributes
    ----------
    data: TargetMap[TargetData]
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
            self.data[target].plot(*args, **kwargs)

    def fit(self) -> TargetMap[Any]:
        return {target: self.data[target].fit() for target in self.data}


class RabiData(TargetData):
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
    def rotated(self) -> NDArray[np.complex128]:
        param = self.rabi_param
        return self.data * np.exp(-1j * param.angle)

    @property
    def normalized(self) -> NDArray[np.float64]:
        param = self.rabi_param
        values = self.data * np.exp(-1j * param.angle)
        values_normalized = (values.imag - param.offset) / param.amplitude
        return values_normalized

    def plot(
        self,
        *,
        normalize: bool = False,
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


class SweepData(TargetData):
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
    rabi_param : RabiParam, optional
        Parameters of the Rabi oscillation.
    created_at : str
        Time when the experiment is conducted.
    """

    def __init__(
        self,
        target: str,
        data: NDArray,
        sweep_range: NDArray,
        sweep_value_label: str,
        rabi_param: RabiParam | None = None,
    ):
        super().__init__(target, data)
        self.sweep_range = sweep_range
        self.sweep_value_label = sweep_value_label
        self.rabi_param = rabi_param

    @property
    def rotated(self) -> NDArray[np.complex128]:
        param = self.rabi_param
        if param is None:
            raise ValueError("rabi_param must be provided for rotation.")
        return self.data * np.exp(-1j * param.angle)

    @property
    def normalized(self) -> NDArray[np.float64]:
        param = self.rabi_param
        if param is None:
            raise ValueError("rabi_param must be provided for rotation.")
        values = self.data * np.exp(-1j * param.angle)
        values_normalized = (values.imag - param.offset) / param.amplitude
        return values_normalized

    def plot(
        self,
        *,
        normalize: bool = False,
    ):
        if normalize:
            param = self.rabi_param
            if param is None:
                raise ValueError("rabi_param must be provided for rotation.")
            values = self.normalized
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.sweep_range,
                    y=values,
                    error_y=dict(
                        type="constant",
                        value=param.noise / param.amplitude,
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


class AmplRabiData(TargetData):
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

    def plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.sweep_range,
                y=self.data * 1e3,
            )
        )
        fig.update_layout(
            title=f"Drive amplitude and Rabi rate : {self.target}",
            xaxis_title="Drive amplitude (arb. units)",
            yaxis_title="Rabi rate (MHz)",
        )
        fig.show()


class FreqRabiData(TargetData):
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

    def plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.frequency_range,
                y=self.data * 1e3,
            )
        )
        fig.update_layout(
            title=f"Drive frequency and Rabi rate : {self.target}",
            xaxis_title="Drive frequency (GHz)",
            yaxis_title="Rabi rate (MHz)",
        )
        fig.show()


class TimePhaseData(TargetData):
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

    def plot(self):
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
