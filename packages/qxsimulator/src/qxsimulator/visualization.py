"""Visualization helpers for quantum simulations."""

from __future__ import annotations

from typing import Literal

import plotly.graph_objects as go
from numpy.typing import NDArray
from qxvisualizer.figure import (
    DEFAULT_HEIGHT,
    DEFAULT_TEMPLATE,
    DEFAULT_WIDTH,
    make_figure,
    show_figure,
)
from qxvisualizer.style import COLORS


def make_bloch_vectors_figure(
    times: NDArray,
    bloch_vectors: NDArray,
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str = "State evolution",
    xlabel: str = "Time (ns)",
    ylabel: str = "Expectation value",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """Create a Bloch-vector timeline figure."""
    fig = make_figure(
        template=template,
        width=width,
        height=height,
    )
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
    )
    return fig


def plot_bloch_vectors(
    times: NDArray,
    bloch_vectors: NDArray,
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str = "State evolution",
    xlabel: str = "Time (ns)",
    ylabel: str = "Expectation value",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    template: str = DEFAULT_TEMPLATE,
) -> None:
    """Plot Bloch vector trajectories over time."""
    figure = make_bloch_vectors_figure(
        times=times,
        bloch_vectors=bloch_vectors,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        template=template,
    )
    show_figure(
        figure,
        filename="bloch_vectors",
    )
