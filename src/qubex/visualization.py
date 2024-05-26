import numpy as np
import plotly.graph_objs as go
from numpy.typing import NDArray


def plot_waveform(
    data: NDArray[np.complex128],
    sampling_period: int = 2,
    title: str = "",
    xlabel: str = "Time (ns)",
    ylabel: str = "Amplitude (arb. unit)",
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(data)) * sampling_period,
            y=np.real(data),
            mode="lines",
            name="Real",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(data)) * sampling_period,
            y=np.imag(data),
            mode="lines",
            name="Imag",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=600,
    )
    fig.show()


def scatter_iq_data(data: dict[str, list[complex]]):
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
        title="I/Q data",
        xaxis_title="I",
        yaxis_title="Q",
        width=500,
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.show()
