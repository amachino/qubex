"""Plotting helpers for analysis outputs."""

from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objs as go
import qctrlvisualizer as qcv
from numpy.typing import ArrayLike, NDArray

from qubex.typing import IQArray

from .figure_factory import (
    DEFAULT_TEMPLATE,
    FIGURE_SIZE_IQ,
    FIGURE_SIZE_STANDARD,
    FIGURE_SIZE_TALL,
    IQ_AXIS_MARGIN_LEFT,
    IQ_AXIS_MARGIN_RIGHT,
    make_figure,
    show_figure,
)
from .style import COLORS, get_colors

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = "./images"


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
        width = FIGURE_SIZE_STANDARD.width
    if height is None:
        height = FIGURE_SIZE_STANDARD.height

    counter = 1
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    file_path = os.path.join(images_dir, f"{current_date}_{name}_{counter}.{format}")

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
    logger.info("Image saved to %s", file_path)


def _render_figure(
    fig: go.Figure,
    *,
    filename: str,
    width: int,
    height: int,
    save_image: bool,
    image_name: str,
) -> None:
    """Save and show a figure using shared Plotly behavior."""
    if save_image:
        save_figure_image(
            fig,
            name=image_name,
            width=width,
            height=height,
        )
    show_figure(
        fig,
        filename=filename,
        width=width,
        height=height,
    )


def make_plot_figure(
    *,
    y: ArrayLike,
    x: ArrayLike | None = None,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | list[float] | None = None,
    ylim: tuple[float, float] | list[float] | None = None,
    width: int = FIGURE_SIZE_STANDARD.width,
    height: int = FIGURE_SIZE_STANDARD.height,
    template: str = DEFAULT_TEMPLATE,
    **kwargs: Any,
) -> go.Figure:
    """Build a 1D or 2D array plot figure."""
    fig = make_figure(
        template=template,
        width=width,
        height=height,
    )
    y_data = np.asarray(y)

    if y_data.ndim == 1:
        x_data = np.arange(len(y_data)) if x is None else x
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=np.real(y_data),
                mode=mode,
                name="Real",
                **kwargs,
            )
        )
        if np.iscomplexobj(y_data):
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=np.imag(y_data),
                    mode=mode,
                    name="Imag",
                    **kwargs,
                )
            )
    elif y_data.ndim == 2:
        x_data = np.arange(y_data.shape[0]) if x is None else x
        for index in range(y_data.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=np.real(y_data[:, index]),
                    mode=mode,
                    name=f"Real {index}",
                    **kwargs,
                )
            )
            if np.iscomplexobj(y_data):
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=np.imag(y_data[:, index]),
                        mode=mode,
                        name=f"Imag {index}",
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
    )
    return fig


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
    width: int = FIGURE_SIZE_STANDARD.width,
    height: int = FIGURE_SIZE_STANDARD.height,
    template: str = DEFAULT_TEMPLATE,
    save_image: bool = False,
    **kwargs: Any,
) -> None:
    """Plot a 1D or 2D array and show the figure."""
    fig = make_plot_figure(
        y=y,
        x=x,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        width=width,
        height=height,
        template=template,
        **kwargs,
    )
    _render_figure(
        fig,
        filename="plot",
        width=width,
        height=height,
        save_image=save_image,
        image_name="plot",
    )


def make_cdf_figure(
    data: ArrayLike | Mapping,
    *,
    xlim: tuple[float, float] | None = None,
    title: str = "Cumulative distribution",
    xlabel: str = "Value",
    ylabel: str = "Cumulative probability",
    width: int = FIGURE_SIZE_TALL.width,
    height: int = FIGURE_SIZE_TALL.height,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """Build a cumulative distribution figure."""
    dataset: dict[Any, NDArray[np.float64]] = {}
    data_min = float("inf")
    data_max = float("-inf")

    if not isinstance(data, Mapping):
        values = np.asarray(data).astype(float)
        values = values[~np.isnan(values)]
        values.sort()
        dataset["data"] = values
        data_min = float(np.min(values))
        data_max = float(np.max(values))
    else:
        for key, values in data.items():
            values_array = np.asarray(values).astype(float)
            values_array = values_array[~np.isnan(values_array)]
            values_array.sort()
            dataset[key] = values_array
            data_min = min(data_min, float(np.min(values_array)))
            data_max = max(data_max, float(np.max(values_array)))

    resolved_xlim = xlim
    if resolved_xlim is None:
        dx = (data_max - data_min) / 100
        resolved_xlim = (data_min - dx, data_max + dx)

    fig = make_figure(
        template=template,
        width=width,
        height=height,
    )
    for key, values in dataset.items():
        n_points = len(values)
        mean_val = float(np.mean(values))
        x_values = [resolved_xlim[0], *values.tolist(), resolved_xlim[1]]
        y_values = [0] + [(index + 1) / n_points for index in range(n_points)] + [1]

        fig.add_scatter(
            x=x_values,
            y=y_values,
            name=str(key),
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
    )
    return fig


def plot_cdf(
    data: ArrayLike | Mapping,
    *,
    xlim: tuple[float, float] | None = None,
    title: str = "Cumulative distribution",
    xlabel: str = "Value",
    ylabel: str = "Cumulative probability",
    width: int = FIGURE_SIZE_TALL.width,
    height: int = FIGURE_SIZE_TALL.height,
    template: str = DEFAULT_TEMPLATE,
    save_image: bool = False,
) -> None:
    """Plot a cumulative distribution function and show the figure."""
    fig = make_cdf_figure(
        data,
        xlim=xlim,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        template=template,
    )
    _render_figure(
        fig,
        filename="plot_cdf",
        width=width,
        height=height,
        save_image=save_image,
        image_name="plot_cdf",
    )


def make_fft_figure(
    x: NDArray,
    y: NDArray,
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "lines",
    title: str = "FFT result",
    xlabel: str = "Frequency",
    ylabel: str = "Amplitude",
    xlim: tuple[float, float] | list[float] | None = None,
    ylim: tuple[float, float] | list[float] | None = None,
    width: int = FIGURE_SIZE_STANDARD.width,
    height: int = FIGURE_SIZE_STANDARD.height,
    template: str = DEFAULT_TEMPLATE,
    **kwargs: Any,
) -> go.Figure:
    """Build an FFT magnitude figure."""
    n_points = len(x)
    dt = x[1] - x[0]
    frequencies = np.fft.fftfreq(n_points, dt)[: n_points // 2]
    amplitudes = np.fft.fft(y)[: n_points // 2]

    fig = make_figure(
        template=template,
        width=width,
        height=height,
    )
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=np.abs(amplitudes) / n_points,
            mode=mode,
            **kwargs,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_range=xlim,
        yaxis_range=ylim,
    )
    return fig


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
    width: int = FIGURE_SIZE_STANDARD.width,
    height: int = FIGURE_SIZE_STANDARD.height,
    template: str = DEFAULT_TEMPLATE,
    save_image: bool = False,
    **kwargs: Any,
) -> None:
    """Plot FFT magnitude and show the figure."""
    fig = make_fft_figure(
        x,
        y,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        width=width,
        height=height,
        template=template,
        **kwargs,
    )
    _render_figure(
        fig,
        filename="plot_fft",
        width=width,
        height=height,
        save_image=save_image,
        image_name="plot_fft",
    )


def make_bloch_vectors_figure(
    times: NDArray,
    bloch_vectors: NDArray,
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    title: str = "State evolution",
    xlabel: str = "Time (ns)",
    ylabel: str = "Expectation value",
    width: int = FIGURE_SIZE_STANDARD.width,
    height: int = FIGURE_SIZE_STANDARD.height,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """Build a Bloch-vector trajectory figure."""
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
    width: int = FIGURE_SIZE_STANDARD.width,
    height: int = FIGURE_SIZE_STANDARD.height,
    template: str = DEFAULT_TEMPLATE,
    save_image: bool = False,
) -> None:
    """Plot Bloch vector trajectories and show the figure."""
    fig = make_bloch_vectors_figure(
        times,
        bloch_vectors,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        template=template,
    )
    _render_figure(
        fig,
        filename="plot_bloch_vectors",
        width=width,
        height=height,
        save_image=save_image,
        image_name="plot_bloch_vectors",
    )


def make_waveform_figure(
    data: NDArray,
    *,
    sampling_period: float = 2.0,
    mode: Literal["lines", "markers", "lines+markers"] = "lines",
    title: str = "Waveform",
    xlabel: str = "Time (ns)",
    ylabel: str = "Signal (arb. units)",
    width: int = FIGURE_SIZE_STANDARD.width,
    height: int = FIGURE_SIZE_STANDARD.height,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """Build an I/Q waveform figure."""
    time_axis = np.arange(len(data)) * sampling_period
    fig = make_figure(
        template=template,
        width=width,
        height=height,
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=np.real(data),
            mode=mode,
            name="I",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=np.imag(data),
            mode=mode,
            name="Q",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    return fig


def plot_waveform(
    data: NDArray,
    *,
    sampling_period: float = 2.0,
    mode: Literal["lines", "markers", "lines+markers"] = "lines",
    title: str = "Waveform",
    xlabel: str = "Time (ns)",
    ylabel: str = "Signal (arb. units)",
    width: int = FIGURE_SIZE_STANDARD.width,
    height: int = FIGURE_SIZE_STANDARD.height,
    template: str = DEFAULT_TEMPLATE,
    save_image: bool = False,
) -> None:
    """Plot waveform I/Q components and show the figure."""
    fig = make_waveform_figure(
        data,
        sampling_period=sampling_period,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        template=template,
    )
    _render_figure(
        fig,
        filename="plot_waveform",
        width=width,
        height=height,
        save_image=save_image,
        image_name="plot_waveform",
    )


def make_iq_scatter_figure(
    data: IQArray | Mapping[str, IQArray],
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "markers",
    title: str = "I/Q plane",
    xlabel: str = "In-phase (arb. units)",
    ylabel: str = "Quadrature (arb. units)",
    width: int = FIGURE_SIZE_IQ.width,
    height: int = FIGURE_SIZE_IQ.height,
    text: Collection[str] | None = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """Build an IQ scatter figure for one or more channels."""
    if not isinstance(data, Mapping):
        data = {"data": data}

    fig = make_figure(
        template=template,
        width=width,
        height=height,
    )
    colors = get_colors(alpha=0.8)
    max_val = np.max([np.max(np.abs(data[target])) for target in data])
    axis_range = [-max_val * 1.1, max_val * 1.1]
    dtick = max_val / 2
    for index, (target, iq_values) in enumerate(data.items()):
        color = colors[index % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=np.real(iq_values),
                y=np.imag(iq_values),
                mode=mode,
                name=target,
                text=text if text is not None else target,
                marker=dict(
                    size=4,
                    color=f"rgba{color}",
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        margin=dict(
            l=IQ_AXIS_MARGIN_LEFT,
            r=IQ_AXIS_MARGIN_RIGHT,
        ),
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
    return fig


def scatter_iq_data(
    data: IQArray | Mapping[str, IQArray],
    *,
    mode: Literal["lines", "markers", "lines+markers"] = "markers",
    title: str = "I/Q plane",
    xlabel: str = "In-phase (arb. units)",
    ylabel: str = "Quadrature (arb. units)",
    width: int = FIGURE_SIZE_IQ.width,
    height: int = FIGURE_SIZE_IQ.height,
    text: Collection[str] | None = None,
    template: str = DEFAULT_TEMPLATE,
    save_image: bool = False,
) -> None:
    """Scatter-plot IQ data and show the figure."""
    fig = make_iq_scatter_figure(
        data,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        text=text,
        template=template,
    )
    _render_figure(
        fig,
        filename="plot_state_distribution",
        width=width,
        height=height,
        save_image=save_image,
        image_name="plot_state_distribution",
    )
