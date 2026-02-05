from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from ..analysis import FitResult, fitting, util
from ..analysis import visualization as viz
from ..typing import TargetMap
from .experiment_record import ExperimentRecord
from .rabi_param import RabiParam


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
    status: str = "success"
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def __repr__(self) -> str:
        data_repr = "{" + ", ".join(f"{k}:..." for k in self.data.keys()) + "}"
        return f"<ExperimentResult data={data_repr}>"

    def plot(
        self,
        *args,
        **kwargs,
    ):
        for target in self.data:
            self.data[target].plot(*args, **kwargs)

    def fit(
        self,
        *args,
        **kwargs,
    ) -> TargetMap[Any]:
        return {target: self.data[target].fit(*args, **kwargs) for target in self.data}

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
    state_centers : dict[int, complex], optional
        Centers of the states.
    """

    time_range: NDArray
    rabi_param: RabiParam
    state_centers: dict[int, complex] | None = None

    @property
    def rotated(self) -> NDArray[np.complex128]:
        angle = self.rabi_param.angle
        return util.rotate(self.data, -angle)

    @property
    def normalized(self) -> NDArray[np.float64]:
        return self.rabi_param.normalize(self.data)

    @property
    def zvalues(self) -> NDArray[np.float64]:
        if self.state_centers is None:
            raise ValueError("state_centers must be provided for zvalues.")
        p = np.array(self.data, dtype=np.complex128)
        g, e = self.state_centers[0], self.state_centers[1]
        v_ge = e - g
        v_gp = p - g
        v_gp_proj = np.real(v_gp * np.conj(v_ge)) / np.abs(v_ge)
        return 1 - 2 * np.abs(v_gp_proj) / np.abs(v_ge)

    def plot(
        self,
        *,
        normalize: bool = False,
        use_zvalue: bool = False,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
        return_figure: bool = False,
        images_dir: Path | str | None = None,
    ):
        fig = go.Figure()

        fig.update_layout(
            title=title or f"Rabi oscillation : {self.target}",
            xaxis_title=xlabel or "Drive duration (ns)",
            yaxis_title=ylabel or "Signal (arb. units)",
            width=width,
            height=height,
            template="qubex",
        )

        if use_zvalue:
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.time_range,
                    y=self.zvalues,
                )
            )
            fig.update_layout(
                yaxis_title="Z value",
                yaxis_range=[-1.2, 1.2],
            )
        elif normalize:
            values = self.normalized
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
                yaxis_title="Normalized signal",
                yaxis_range=[-1.2, 1.2],
            )
        else:
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

        fig.show(
            config=viz.get_config(
                filename=f"rabi_data_{self.target}",
                width=width,
                height=height,
            )
        )
        if images_dir is not None:
            viz.save_figure_image(
                fig,
                name=f"rabi_data_{self.target}",
                images_dir=images_dir,
                width=width,
                height=height,
            )
        if return_figure:
            return fig

    def fit(
        self,
        use_zvalue: bool = False,
        yaxis_range: tuple[float, float] | None = None,
        **kwargs,
    ) -> FitResult:
        return fitting.fit_rabi(
            target=self.target,
            times=self.time_range,
            data=self.data if not use_zvalue else self.zvalues + 0j,
            yaxis_range=yaxis_range if not use_zvalue else (-1.2, 1.2),
            **kwargs,
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
    state_centers : dict[int, complex], optional
        Centers of the states.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Title of the x-axis.
    ylabel : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    """

    sweep_range: NDArray
    rabi_param: RabiParam | None = None
    state_centers: dict[int, complex] | None = None
    title: str = "Sweep result"
    xlabel: str = "Sweep value"
    ylabel: str = "Measured signal"
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
        return param.normalize(self.data)

    @property
    def zvalues(self) -> NDArray[np.float64]:
        if self.state_centers is None:
            raise ValueError("state_centers must be provided for zvalues.")
        p = np.array(self.data, dtype=np.complex128)
        g, e = self.state_centers[0], self.state_centers[1]
        v_ge = e - g
        v_gp = p - g
        v_gp_proj = np.real(v_gp * np.conj(v_ge)) / np.abs(v_ge)
        return 1 - 2 * np.abs(v_gp_proj) / np.abs(v_ge)

    def plot(
        self,
        *,
        normalize: bool = False,
        use_zvalue: bool = False,
        title: str | None = None,
        xaxis_type: str | None = None,
        yaxis_type: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
        return_figure: bool = False,
        images_dir: Path | str | None = None,
    ):
        fig = go.Figure()

        fig.update_layout(
            title=title or f"{self.title} : {self.target}",
            xaxis_title=xlabel or self.xlabel,
            xaxis_type=xaxis_type if xaxis_type is not None else self.xaxis_type,
            yaxis_title=ylabel or self.ylabel,
            yaxis_type=yaxis_type if yaxis_type is not None else self.yaxis_type,
            width=width,
            height=height,
            template="qubex",
        )

        if use_zvalue:
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=self.sweep_range,
                    y=self.zvalues,
                )
            )
            fig.update_layout(
                yaxis_title=ylabel or "Z value",
                yaxis_range=[-1.2, 1.2],
            )
        elif normalize:
            param = self.rabi_param
            if param is None:
                print("rabi_param must be provided for normalization.")
                return  # type: ignore
            values = self.normalized
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
                yaxis_title=ylabel or "Normalized signal",
                yaxis_range=[-1.2, 1.2],
            )
        else:
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

        fig.show(
            config=viz.get_config(
                filename=f"sweep_data_{self.target}",
                width=width,
                height=height,
            )
        )
        if images_dir is not None:
            viz.save_figure_image(
                fig,
                name=f"sweep_data_{self.target}",
                images_dir=images_dir,
                width=width,
                height=height,
            )
        if return_figure:
            return fig


@dataclass
class AmplCalibData(SweepData):
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
    rabi_param : RabiParam, optional
        Parameters of the Rabi oscillation.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Title of the x-axis.
    ylabel : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    calib_value : float, optional
        Calibrated value.
    r2 : float, optional
        Coefficient of determination.
    """

    sweep_range: NDArray
    calib_value: float = np.nan
    r2: float = np.nan

    @classmethod
    def new(
        cls,
        sweep_data: SweepData,
        calib_value: float,
        r2: float,
    ) -> AmplCalibData:
        return cls(
            target=sweep_data.target,
            data=sweep_data.data,
            sweep_range=sweep_data.sweep_range,
            rabi_param=sweep_data.rabi_param,
            title=sweep_data.title,
            xlabel=sweep_data.xlabel,
            ylabel=sweep_data.ylabel,
            xaxis_type=sweep_data.xaxis_type,
            yaxis_type=sweep_data.yaxis_type,
            calib_value=calib_value,
            r2=r2,
        )

    def fit(self, **kwargs) -> FitResult:
        return fitting.fit_ampl_calib_data(
            target=self.target,
            amplitude_range=self.sweep_range,
            data=self.normalized,
            **kwargs,
        )


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
    xlabel : str, optional
        Title of the x-axis.
    ylabel : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    t1 : float, optional
        T1 time.
    t1_err : float, optional
        Error of the T1 time.
    r2 : float, optional
        Coefficient of determination
    """

    t1: float = np.nan
    t1_err: float = np.nan
    r2: float = np.nan

    @classmethod
    def new(
        cls,
        sweep_data: SweepData,
        t1: float,
        t1_err: float,
        r2: float,
    ) -> T1Data:
        return cls(
            target=sweep_data.target,
            data=sweep_data.data,
            sweep_range=sweep_data.sweep_range,
            rabi_param=sweep_data.rabi_param,
            title=sweep_data.title,
            xlabel=sweep_data.xlabel,
            ylabel=sweep_data.ylabel,
            xaxis_type=sweep_data.xaxis_type,
            yaxis_type=sweep_data.yaxis_type,
            t1=t1,
            t1_err=t1_err,
            r2=r2,
        )

    def fit(self, **kwargs) -> FitResult:
        return fitting.fit_exp_decay(
            target=self.target,
            x=self.sweep_range,
            y=0.5 * (1 - self.normalized),
            title="T1",
            xlabel="Time (μs)",
            ylabel="Population",
            xaxis_type=self.xaxis_type,
            yaxis_type=self.yaxis_type,
            **kwargs,
        )


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
    xlabel : str, optional
        Title of the x-axis.
    ylabel : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    t2 : float, optional
        T2 echo time.
    t2_err : float, optional
        Error of the T2 echo time.
    r2 : float, optional
        Coefficient of determination.
    """

    t2: float = np.nan
    t2_err: float = np.nan
    r2: float = np.nan

    @classmethod
    def new(
        cls,
        sweep_data: SweepData,
        t2: float,
        t2_err: float,
        r2: float,
    ) -> T2Data:
        return cls(
            target=sweep_data.target,
            data=sweep_data.data,
            sweep_range=sweep_data.sweep_range,
            rabi_param=sweep_data.rabi_param,
            title=sweep_data.title,
            xlabel=sweep_data.xlabel,
            ylabel=sweep_data.ylabel,
            xaxis_type=sweep_data.xaxis_type,
            yaxis_type=sweep_data.yaxis_type,
            t2=t2,
            t2_err=t2_err,
            r2=r2,
        )

    def fit(self, **kwargs) -> FitResult:
        return fitting.fit_exp_decay(
            target=self.target,
            x=self.sweep_range,
            y=0.5 * (1 + self.normalized),
            title="T2",
            xlabel="Time (μs)",
            ylabel="Population",
            xaxis_type=self.xaxis_type,
            yaxis_type=self.yaxis_type,
            **kwargs,
        )


@dataclass
class RamseyData(SweepData):
    """
    Data class representing the result of a Ramsey experiment.

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
    xlabel : str, optional
        Title of the x-axis.
    ylabel : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    t2 : float, optional
        T2* time.
    ramsey_freq : float, optional
        Frequency of the Ramsey fringes.
    bare_freq : float, optional
        Bare frequency of the qubit.
    r2 : float, optional
        Coefficient of determination.
    """

    t2: float = np.nan
    ramsey_freq: float = np.nan
    bare_freq: float = np.nan
    r2: float = np.nan

    @classmethod
    def new(
        cls,
        sweep_data: SweepData,
        t2: float,
        ramsey_freq: float,
        bare_freq: float,
        r2: float,
    ) -> RamseyData:
        return cls(
            target=sweep_data.target,
            data=sweep_data.data,
            sweep_range=sweep_data.sweep_range,
            rabi_param=sweep_data.rabi_param,
            title=sweep_data.title,
            xlabel=sweep_data.xlabel,
            ylabel=sweep_data.ylabel,
            xaxis_type=sweep_data.xaxis_type,
            yaxis_type=sweep_data.yaxis_type,
            t2=t2,
            ramsey_freq=ramsey_freq,
            bare_freq=bare_freq,
            r2=r2,
        )

    def fit(self, **kwargs) -> FitResult:
        return fitting.fit_ramsey(
            target=self.target,
            times=self.sweep_range,
            data=self.normalized,
            xaxis_type=self.xaxis_type,
            yaxis_type=self.yaxis_type,
            **kwargs,
        )


@dataclass
class RBData(SweepData):
    """
    Data class representing the result of a randomized benchmarking

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
    xlabel : str, optional
        Title of the x-axis.
    ylabel : str, optional
        Title of the y-axis.
    xaxis_type : str, optional
        Type of the x-axis.
    yaxis_type : str, optional
        Type of the y-axis.
    depolarizing_rate : float, optional
        Depolarizing rate.
    avg_gate_error : float, optional
        Average gate error.
    avg_gate_fidelity : float, optional
        Average gate fidelity.
    """

    depolarizing_rate: float = np.nan
    avg_gate_error: float = np.nan
    avg_gate_fidelity: float = np.nan

    @classmethod
    def new(
        cls,
        sweep_data: SweepData,
        depolarizing_rate: float,
        avg_gate_error: float,
        avg_gate_fidelity: float,
    ) -> RBData:
        return cls(
            target=sweep_data.target,
            data=sweep_data.data,
            sweep_range=sweep_data.sweep_range,
            rabi_param=sweep_data.rabi_param,
            title=sweep_data.title,
            xlabel=sweep_data.xlabel,
            ylabel=sweep_data.ylabel,
            xaxis_type=sweep_data.xaxis_type,
            yaxis_type=sweep_data.yaxis_type,
            depolarizing_rate=depolarizing_rate,
            avg_gate_error=avg_gate_error,
            avg_gate_fidelity=avg_gate_fidelity,
        )

    def fit(self, **kwargs) -> FitResult:
        return fitting.fit_rb(
            target=self.target,
            x=self.sweep_range,
            y=self.normalized,
            xaxis_type=self.xaxis_type,
            yaxis_type=self.yaxis_type,
            **kwargs,
        )


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
    rabi_data: list[RabiData]

    def plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.sweep_range,
                y=self.data * 1e3,
            )
        )
        fig.update_layout(
            title=f"Rabi rate vs drive amplitude : {self.target}",
            xaxis_title="Drive amplitude (arb. units)",
            yaxis_title="Rabi rate (MHz)",
        )
        fig.show()

    def fit(self, **kwargs) -> FitResult:
        return fitting.fit_linear(
            self.sweep_range,
            self.data * 1e3,  # Convert to MHz
            title=f"Rabi rate vs drive amplitude : {self.target}",
            xlabel="Drive amplitude (arb. units)",
            ylabel="Rabi rate (MHz)",
            xmin=0.0,
            ymin=0.0,
            **kwargs,
        )


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
    rabi_data: list[RabiData]

    def plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.frequency_range,
                y=self.data * 1e3,
            )
        )
        fig.update_layout(
            title=f"Rabi rate vs drive frequency : {self.target}",
            xaxis_title="Drive frequency (GHz)",
            yaxis_title="Rabi rate (MHz)",
        )
        fig.show()

    def fit(self, **kwargs) -> FitResult:
        return fitting.fit_detuned_rabi(
            target=self.target,
            control_frequencies=self.frequency_range,
            rabi_frequencies=self.data,
            **kwargs,
        )
