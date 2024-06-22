from typing import Literal

import numpy as np
import plotly.graph_objs as go
from IPython.display import display
from numpy.typing import ArrayLike, NDArray

from .typing import IQArray, TargetMap


def plot(
    y: ArrayLike,
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    **kwargs,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode=mode, **kwargs))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="qubex",
    )
    fig.show()


def plot_xy(
    *,
    x: ArrayLike,
    y: ArrayLike,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    **kwargs,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, **kwargs))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="qubex",
    )
    fig.show()


def plot_xy_square(
    *,
    x: ArrayLike,
    y: ArrayLike,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    **kwargs,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, **kwargs))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="qubex+square",
    )
    fig.show()


def plot_waveform(
    data: NDArray[np.complex128],
    sampling_period: float = 2.0,
    title: str = "Waveform",
    xlabel: str = "Time (ns)",
    ylabel: str = "Amplitude (arb. units)",
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(data)) * sampling_period,
            y=np.real(data),
            mode="lines",
            name="I",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(data)) * sampling_period,
            y=np.imag(data),
            mode="lines",
            name="Q",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    fig.show()


def scatter_iq_data(
    data: TargetMap[IQArray],
    title: str = "I/Q plane",
    xlabel: str = "In-phase (arb. units)",
    ylabel: str = "Quadrature (arb. units)",
) -> None:
    fig = go.Figure()
    for qubit, iq in data.items():
        scatter = go.Scatter(
            x=np.real(iq),
            y=np.imag(iq),
            mode="markers",
            name=qubit,
        )
        fig.add_trace(scatter)
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=500,
        height=400,
        margin=dict(l=120, r=120),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.show()


class IQPlotter:
    def __init__(self):
        self._num_scatters = None
        self._widget = go.FigureWidget()
        self._widget.update_layout(
            title="I/Q plane",
            xaxis_title="In-phase (arb. units)",
            yaxis_title="Quadrature (arb. units)",
            width=500,
            height=400,
            margin=dict(l=120, r=120),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=True,
        )

    def update(self, data: TargetMap[IQArray]):
        if self._num_scatters is None:
            display(self._widget)
            for qubit in data:
                self._widget.add_scatter(name=qubit, mode="markers")
            self._num_scatters = len(data)
        if len(data) != self._num_scatters:
            raise ValueError("Number of scatters does not match")
        for idx, qubit in enumerate(data):
            scatter: go.Scatter = self._widget.data[idx]  # type: ignore
            scatter.x = np.real(data[qubit])
            scatter.y = np.imag(data[qubit])
