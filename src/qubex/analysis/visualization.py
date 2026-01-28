"""Plotting helpers for analysis outputs."""

from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objs as go
import qctrlvisualizer as qcv
from numpy.typing import ArrayLike, NDArray

from qubex.style import COLORS, get_colors, get_config
from qubex.typing import IQArray

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = "./images"
DEFAULT_TEMPLATE = "qubex"


def display_bloch_sphere(bloch_vectors: NDArray[np.float64]) -> None:
    """Display Bloch sphere visualization for the given vectors."""
    qcv.display_bloch_sphere_from_bloch_vectors(bloch_vectors)


def save_figure_image(
    fig: go.Figure,
    name: str = "image",
    *,
    images_dir: Path | str = DEFAULT_IMAGES_DIR,
    format: Literal["png", "svg", "jpeg", "webp"] = "png",
    width: int | None = None,
    height: int | None = None,
    scale: int = 3,
) -> None:
    """Save a figure to an image file in the images directory."""
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    if width is None:
        width = 600
    if height is None:
        height = 300

    counter = 1
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    file_path = os.path.join(
        images_dir,
        f"{current_date}_{name}_{counter}.{format}",
    )

    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(
            images_dir,
            f"{current_date}_{name}_{counter}.{format}",
        )

    fig.write_image(
        file_path,
        format=format,
        width=width,
        height=height,
        scale=scale,
    )
    logger.info(f"Image saved to {file_path}")


def plot(
    *,
    y: ArrayLike,
    x: ArrayLike | None = None,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | list[float] | None = None,
    ylim: tuple[float, float] | list[float] | None = None,
    width: int = 600,
    height: int = 300,
    template: str = DEFAULT_TEMPLATE,
    return_figure: bool = False,
    save_image: bool = False,
    **kwargs,
) -> go.Figure | None:
    """Plot 1D or 2D array data and optionally save the figure."""
    fig = go.Figure()
    y = np.asarray(y)

    if y.ndim == 1:
        if x is None:
            x = np.arange(len(y))
        fig.add_trace(go.Scatter(x=x, y=np.real(y), mode=mode, name="Real", **kwargs))
        if np.iscomplexobj(y):
            fig.add_trace(
                go.Scatter(x=x, y=np.imag(y), mode=mode, name="Imag", **kwargs)
            )
    elif y.ndim == 2:
        if x is None:
            x = np.arange(y.shape[0])
        for i in range(y.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=x, y=np.real(y[:, i]), mode=mode, name=f"Real {i}", **kwargs
                )
            )
            if np.iscomplexobj(y):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.imag(y[:, i]),
                        mode=mode,
                        name=f"Imag {i}",
                        **kwargs,
                    )
                )
    else:
        raise ValueError("y must be 1D or 2D")

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_range=xlim,
        yaxis_range=ylim,
        width=width,
        height=height,
        template=template,
    )

    if save_image:
        save_figure_image(
            fig,
            name="plot",
            width=width,
            height=height,
        )

    if return_figure:
        return fig
    else:
        fig.show(
            config=get_config(
                filename="plot",
                width=width,
                height=height,
            )
        )


def plot_cdf(
    data: ArrayLike | Mapping,
    *,
    xlim: tuple[float, float] | None = None,
    title: str = "Cumulative distribution",
    xlabel: str = "Value",
    ylabel: str = "Cumulative probability",
    width: int = 600,
    height: int = 400,
    template: str = DEFAULT_TEMPLATE,
    return_figure: bool = False,
    save_image: bool = False,
) -> go.Figure | None:
    """Plot a cumulative distribution function for the data."""
    dataset = {}
    data_min = float("inf")
    data_max = float("-inf")

    if not isinstance(data, Mapping):
        values = np.asarray(data).astype(float)
        values = values[~np.isnan(values)]
        values.sort()
        dataset["data"] = values
        data_min = np.min(values)
        data_max = np.max(values)
    else:
        for key, values in data.items():
            values = np.asarray(values).astype(float)
            values = values[~np.isnan(values)]
            values.sort()
            dataset[key] = values
            data_min = min(data_min, np.min(values))
            data_max = max(data_max, np.max(values))

    fig = go.Figure()

    for key, values in dataset.items():
        N = len(values)
        mean_val = np.mean(values)

        if xlim is None:
            dx = (data_max - data_min) / 100
            xlim = (data_min - dx, data_max + dx)

        x = [xlim[0], *values.tolist(), xlim[1]]
        y = [0] + [(i + 1) / N for i in range(N)] + [1]

        fig.add_scatter(
            x=x,
            y=y,
            name=key,
            mode="lines",
            line=dict(
                color=COLORS[0],
                width=3,
            ),
            line_shape="hv",
        )

        fig.add_vline(
            x=mean_val,
            line_width=2,
            line_dash="dash",
            line_color="lightgrey",
            layer="below",
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis=dict(range=[0, 1]),
        width=width,
        height=height,
        template=template,
    )

    if save_image:
        save_figure_image(
            fig,
            name="plot_cdf",
            width=width,
            height=height,
        )

    if return_figure:
        return fig
    else:
        fig.show(
            config=get_config(
                filename="plot_cdf",
                width=width,
                height=height,
            )
        )


def plot_fft(
    x: NDArray,
    y: NDArray,
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "lines",
    title: str = "FFT result",
    xlabel: str = "Frequency",
    ylabel: str = "Amplitude",
    xlim: tuple[float, float] | list[float] | None = None,
    ylim: tuple[float, float] | list[float] | None = None,
    width: int = 600,
    height: int = 300,
    template: str = DEFAULT_TEMPLATE,
    return_figure: bool = False,
    save_image: bool = False,
    **kwargs,
) -> go.Figure | None:
    """Plot FFT magnitude for the provided signal."""
    N = len(x)
    dt = x[1] - x[0]
    f = np.fft.fftfreq(N, dt)[: N // 2]
    F = np.fft.fft(y)[: N // 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=np.abs(F) / N, mode=mode, **kwargs))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_range=xlim,
        yaxis_range=ylim,
        width=width,
        height=height,
        template=template,
    )

    if save_image:
        save_figure_image(
            fig,
            name="plot_fft",
            width=width,
            height=height,
        )

    if return_figure:
        return fig
    else:
        fig.show(
            config=get_config(
                filename="plot_fft",
                width=width,
                height=height,
            )
        )


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
    save_image: bool = False,
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

    if save_image:
        save_figure_image(
            fig,
            name="plot_bloch_vectors",
            width=width,
            height=height,
        )

    if return_figure:
        return fig
    else:
        fig.show(
            config=get_config(
                filename="plot_bloch_vectors",
                width=width,
                height=height,
            )
        )


def plot_waveform(
    data: NDArray,
    *,
    sampling_period: float = 2.0,
    mode: Literal["lines", "markers", "lines+markers"] = "lines",
    title: str = "Waveform",
    xlabel: str = "Time (ns)",
    ylabel: str = "Signal (arb. units)",
    width: int = 600,
    height: int = 300,
    template: str = DEFAULT_TEMPLATE,
    return_figure: bool = False,
    save_image: bool = False,
) -> go.Figure | None:
    """Plot waveform I/Q components over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(data)) * sampling_period,
            y=np.real(data),
            mode=mode,
            name="I",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(data)) * sampling_period,
            y=np.imag(data),
            mode=mode,
            name="Q",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        template=template,
    )

    if save_image:
        save_figure_image(
            fig,
            name="plot_waveform",
            width=width,
            height=height,
        )

    if return_figure:
        return fig
    else:
        fig.show(
            config=get_config(
                filename="plot_waveform",
                width=width,
                height=height,
            )
        )


def scatter_iq_data(
    data: IQArray | Mapping[str, IQArray],
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "markers",
    title: str = "I/Q plane",
    xlabel: str = "In-phase (arb. units)",
    ylabel: str = "Quadrature (arb. units)",
    width: int = 500,
    height: int = 400,
    text: Collection[str] | None = None,
    template: str = DEFAULT_TEMPLATE,
    return_figure: bool = False,
    save_image: bool = False,
) -> go.Figure | None:
    """Scatter-plot IQ data for one or more channels."""
    if not isinstance(data, Mapping):
        data = {"data": data}

    fig = go.Figure()
    colors = get_colors(alpha=0.8)
    max_val = np.max([np.max(np.abs(data[qubit])) for qubit in data])
    axis_range = [-max_val * 1.1, max_val * 1.1]
    dtick = max_val / 2
    for idx, (qubit, iq) in enumerate(data.items()):
        color = colors[idx % len(colors)]
        scatter = go.Scatter(
            x=np.real(iq),
            y=np.imag(iq),
            mode=mode,
            name=qubit,
            text=text if text is not None else qubit,
            marker=dict(
                size=4,
                color=f"rgba{color}",
            ),
        )
        fig.add_trace(scatter)
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        template=template,
        margin=dict(l=120, r=120),
        xaxis=dict(
            range=axis_range,
            dtick=dtick,
            tickformat=".2g",
            showticklabels=True,
            zeroline=True,
            zerolinecolor="black",
            showgrid=True,
        ),
        yaxis=dict(
            range=axis_range,
            scaleanchor="x",
            scaleratio=1,
            dtick=dtick,
            tickformat=".2g",
            showticklabels=True,
            zeroline=True,
            zerolinecolor="black",
            showgrid=True,
        ),
    )

    if save_image:
        save_figure_image(
            fig,
            name="plot_state_distribution",
            width=width,
            height=height,
        )

    if return_figure:
        return fig
    else:
        fig.show(
            config=get_config(
                filename="plot_state_distribution",
                width=width,
                height=height,
            )
        )
