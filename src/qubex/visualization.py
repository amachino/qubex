import numpy as np
import plotly.graph_objs as go
from numpy.typing import NDArray

from .typing import IQArray, TargetMap


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
    xlabel: str = "I (arb. units)",
    ylabel: str = "Q (arb. units)",
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
