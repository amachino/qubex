"""Visualization helpers for quantum simulations."""

from __future__ import annotations

from typing import Literal

import plotly.graph_objects as go
from numpy.typing import NDArray
from qxpulse.style import COLORS

DEFAULT_TEMPLATE = "plotly"


def plot_bloch_vectors(
    times: NDArray,
    bloch_vectors: NDArray,
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str = "State evolution",
    xlabel: str = "Time (ns)",
    ylabel: str = "Expectation value",
    width: int = 600,
    height: int = 300,
    template: str = DEFAULT_TEMPLATE,
    return_figure: bool = False,
) -> go.Figure | None:
    """Plot Bloch vector trajectories over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=bloch_vectors[:, 0],
            mode=mode,
            name="〈X〉",
            line=dict(color=COLORS[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=bloch_vectors[:, 1],
            mode=mode,
            name="〈Y〉",
            line=dict(color=COLORS[1]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=bloch_vectors[:, 2],
            mode=mode,
            name="〈Z〉",
            line=dict(color=COLORS[2]),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis=dict(range=[-1.1, 1.1]),
        width=width,
        height=height,
        template=template,
    )

    if return_figure:
        return fig

    fig.show()
    return None
