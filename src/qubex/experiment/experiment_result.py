from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .. import fitting
from ..fitting import RabiParam
from ..typing import TargetMap
from .experiment_record import ExperimentRecord


@dataclass
class TargetData:
    """
    Data class representing some data of a target.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    """

    target: str
    data: NDArray

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError


T = TypeVar("T", bound=TargetData)


@dataclass
class ExperimentResult(Generic[T]):
    """
    Data class representing the result of an experiment.

    Attributes
    ----------
    data: TargetMap[TargetData]
        Result of the experiment.
    rabi_params: TargetMap[RabiParam]
        Parameters of the Rabi oscillation.
    created_at: str
        Time when the experiment is conducted.
    """

    data: TargetMap[T]
    rabi_params: TargetMap[RabiParam] | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def plot(
        self,
        *args,
        **kwargs,
    ):
        for target in self.data:
            self.data[target].plot(*args, **kwargs)

    def fit(self) -> TargetMap[Any]:
        return {target: self.data[target].fit() for target in self.data}

    def save(
        self,
        name: str = "ExperimentResult",
        description: str = "",
    ) -> ExperimentRecord[ExperimentResult[T]]:
        return ExperimentRecord.create(
            data=self,
            name=name,
            description=description,
        )


@dataclass
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
    """

    time_range: NDArray
    rabi_param: RabiParam

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


@dataclass
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
    rabi_param : RabiParam, optional
        Parameters of the Rabi oscillation.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Title of the x-axis.
    yaxis_title : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    """

    sweep_range: NDArray
    rabi_param: RabiParam | None = None
    title: str = "Sweep result"
    xaxis_title: str = "Sweep value"
    yaxis_title: str = "Measured value"
    xaxis_type: str = "linear"
    yaxis_type: str = "linear"

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
        xaxis_type: str | None = None,
        yaxis_type: str | None = None,
    ):
        if normalize:
            param = self.rabi_param
            if param is None:
                print("rabi_param must be provided for normalization.")
                return
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
                title=f"{self.title} : {self.target}",
                xaxis_title=self.xaxis_title,
                xaxis_type=xaxis_type if xaxis_type is not None else self.xaxis_type,
                yaxis_title=self.yaxis_title,
                yaxis_type=yaxis_type if yaxis_type is not None else self.yaxis_type,
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
                title=f"{self.title} : {self.target}",
                xaxis_title=self.xaxis_title,
                xaxis_type=xaxis_type if xaxis_type is not None else self.xaxis_type,
                yaxis_title=self.yaxis_title,
                yaxis_type=yaxis_type if yaxis_type is not None else self.yaxis_type,
            )
            fig.show()


@dataclass
class T1Data(SweepData):
    """
    Data class representing the result of a T1 experiment.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    sweep_range : NDArray
        Sweep range of the experiment.
    rabi_param : RabiParam, optional
        Parameters of the Rabi oscillation.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Title of the x-axis.
    yaxis_title : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    """

    t1: float = np.nan

    @classmethod
    def new(cls, sweep_data: SweepData, t1: float) -> T1Data:
        return cls(
            target=sweep_data.target,
            data=sweep_data.data,
            sweep_range=sweep_data.sweep_range,
            rabi_param=sweep_data.rabi_param,
            title=sweep_data.title,
            xaxis_title=sweep_data.xaxis_title,
            yaxis_title=sweep_data.yaxis_title,
            xaxis_type=sweep_data.xaxis_type,
            yaxis_type=sweep_data.yaxis_type,
            t1=t1,
        )

    def fit(self) -> float:
        tau = fitting.fit_exp_decay(
            target=self.target,
            x=self.sweep_range,
            y=0.5 * (1 - self.normalized),
            title="T1",
            xaxis_title="Time (ns)",
            yaxis_title="Population",
            xaxis_type="log",
            yaxis_type="linear",
        )
        return tau


@dataclass
class T2Data(SweepData):
    """
    Data class representing the result of a T2 experiment.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    sweep_range : NDArray
        Sweep range of the experiment.
    rabi_param : RabiParam, optional
        Parameters of the Rabi oscillation.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Title of the x-axis.
    yaxis_title : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    """

    t2: float = np.nan

    @classmethod
    def new(cls, sweep_data: SweepData, t2: float) -> T2Data:
        return cls(
            target=sweep_data.target,
            data=sweep_data.data,
            sweep_range=sweep_data.sweep_range,
            rabi_param=sweep_data.rabi_param,
            title=sweep_data.title,
            xaxis_title=sweep_data.xaxis_title,
            yaxis_title=sweep_data.yaxis_title,
            xaxis_type=sweep_data.xaxis_type,
            yaxis_type=sweep_data.yaxis_type,
            t2=t2,
        )

    def fit(self) -> float:
        tau = fitting.fit_ramsey(
            target=self.target,
            x=self.sweep_range,
            y=self.normalized,
            title="T2",
            xaxis_title="Time (μs)",
            yaxis_title="Measured value",
            xaxis_type="linear",
            yaxis_type="linear",
        )
        return tau


@dataclass
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
    """

    sweep_range: NDArray

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


@dataclass
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
    frequency_range : NDArray
        Frequency range of the experiment.
    """

    sweep_range: NDArray
    frequency_range: NDArray

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

    def fit(self) -> tuple[float, float]:
        return fitting.fit_detuned_rabi(
            target=self.target,
            control_frequencies=self.frequency_range,
            rabi_frequencies=self.data,
        )


@dataclass
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
    """

    sweep_range: NDArray

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
        """Return the average phase shift per 128 ns."""
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
            title=f"Phase shift of {self.target} : {self.phase_shift:.5g} rad/128ns",
            xaxis_title="Control window (ns)",
            yaxis_title="Phase (rad)",
        )
        fig.show()


@dataclass
class AmplCalibData(TargetData):
    """
    The relation between the control amplitude and the measured value.

    Attributes
    ----------
    target : str
        Target of the experiment.
    data : NDArray
        Measured data.
    sweep_range : NDArray
        Sweep range of the experiment.
    """

    sweep_range: NDArray

    def plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.sweep_range,
                y=self.data,
            )
        )
        fig.update_layout(
            title=f"Amplitude calibration : {self.target}",
            xaxis_title="Control amplitude (arb. units)",
            yaxis_title="Measured value (arb. units)",
        )
        fig.show()

    def fit(self) -> tuple[float, float]:
        return fitting.fit_ampl_calib_data(
            target=self.target,
            amplitude=self.sweep_range,
            data=-self.data,
        )
