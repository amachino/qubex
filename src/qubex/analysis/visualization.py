"""Compatibility shim for legacy `qubex.analysis.visualization` imports."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Literal

import plotly.graph_objs as go
from numpy.typing import ArrayLike, NDArray
from typing_extensions import deprecated

from qubex.typing import IQArray
from qubex.visualization import (
    COLORS,
    DEFAULT_HEIGHT,
    DEFAULT_TEMPLATE,
    DEFAULT_WIDTH,
    FIGURE_SIZE_IQ,
    FIGURE_SIZE_STANDARD,
    FIGURE_SIZE_TALL,
    IQ_AXIS_MARGIN_LEFT,
    IQ_AXIS_MARGIN_RIGHT,
    FigureSize,
    get_colors,
    get_config,
    make_bloch_vectors_figure,
    make_cdf_figure,
    make_fft_figure,
    make_figure,
    make_iq_scatter_figure,
    make_plot_figure,
    make_waveform_figure,
    plot as _plot,
    plot_bloch_vectors as _plot_bloch_vectors,
    plot_cdf as _plot_cdf,
    plot_fft as _plot_fft,
    plot_waveform as _plot_waveform,
    save_figure,
    scatter_iq_data as _scatter_iq_data,
    show_figure,
)

# TODO: Remove this compatibility shim after downstream imports migrate to `qubex.visualization`.
DEFAULT_IMAGES_DIR = "./images"


@deprecated(
    "Use `qubex.visualization.display_bloch_sphere`-equivalent call sites instead."
)
def display_bloch_sphere(bloch_vectors: NDArray) -> None:
    """Display Bloch-sphere visualization for the provided Bloch vectors."""
    import qctrlvisualizer as qcv

    qcv.display_bloch_sphere_from_bloch_vectors(bloch_vectors)


@deprecated("Use `qubex.visualization.save_figure` instead.")
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
    """Save a figure using the legacy helper name."""
    save_figure(
        fig,
        name=name,
        images_dir=images_dir,
        format=format,
        width=width,
        height=height,
        scale=scale,
    )


@deprecated("Use `qubex.visualization.plot` or `make_plot_figure` instead.")
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
    return_figure: bool = False,
    save_image: bool = False,
    **kwargs: Any,
) -> go.Figure | None:
    """Preserve the legacy plot API with optional figure return."""
    if return_figure:
        figure = make_plot_figure(
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
        if save_image:
            save_figure(
                figure,
                name="plot",
                width=width,
                height=height,
            )
        return figure

    _plot(
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
        save_image=save_image,
        **kwargs,
    )
    return None


@deprecated("Use `qubex.visualization.plot_cdf` or `make_cdf_figure` instead.")
def plot_cdf(
    data: ArrayLike | Mapping[Any, ArrayLike],
    *,
    xlim: tuple[float, float] | None = None,
    title: str = "Cumulative distribution",
    xlabel: str = "Value",
    ylabel: str = "Cumulative probability",
    width: int = FIGURE_SIZE_TALL.width,
    height: int = FIGURE_SIZE_TALL.height,
    template: str = DEFAULT_TEMPLATE,
    return_figure: bool = False,
    save_image: bool = False,
) -> go.Figure | None:
    """Preserve the legacy CDF API with optional figure return."""
    if return_figure:
        figure = make_cdf_figure(
            data,
            xlim=xlim,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
            template=template,
        )
        if save_image:
            save_figure(
                figure,
                name="plot_cdf",
                width=width,
                height=height,
            )
        return figure

    _plot_cdf(
        data,
        xlim=xlim,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        template=template,
        save_image=save_image,
    )
    return None


@deprecated("Use `qubex.visualization.plot_fft` or `make_fft_figure` instead.")
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
    return_figure: bool = False,
    save_image: bool = False,
    **kwargs: Any,
) -> go.Figure | None:
    """Preserve the legacy FFT API with optional figure return."""
    if return_figure:
        figure = make_fft_figure(
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
        if save_image:
            save_figure(
                figure,
                name="plot_fft",
                width=width,
                height=height,
            )
        return figure

    _plot_fft(
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
        save_image=save_image,
        **kwargs,
    )
    return None


@deprecated(
    "Use `qubex.visualization.plot_bloch_vectors` or `make_bloch_vectors_figure` instead."
)
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
    return_figure: bool = False,
    save_image: bool = False,
) -> go.Figure | None:
    """Preserve the legacy Bloch-vector API with optional figure return."""
    if return_figure:
        figure = make_bloch_vectors_figure(
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
        if save_image:
            save_figure(
                figure,
                name="plot_bloch_vectors",
                width=width,
                height=height,
            )
        return figure

    _plot_bloch_vectors(
        times,
        bloch_vectors,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        template=template,
        save_image=save_image,
    )
    return None


@deprecated(
    "Use `qubex.visualization.plot_waveform` or `make_waveform_figure` instead."
)
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
    return_figure: bool = False,
    save_image: bool = False,
) -> go.Figure | None:
    """Preserve the legacy waveform API with optional figure return."""
    if return_figure:
        figure = make_waveform_figure(
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
        if save_image:
            save_figure(
                figure,
                name="plot_waveform",
                width=width,
                height=height,
            )
        return figure

    _plot_waveform(
        data,
        sampling_period=sampling_period,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        template=template,
        save_image=save_image,
    )
    return None


@deprecated(
    "Use `qubex.visualization.scatter_iq_data` or `make_iq_scatter_figure` instead."
)
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
    return_figure: bool = False,
    save_image: bool = False,
) -> go.Figure | None:
    """Preserve the legacy IQ-scatter API with optional figure return."""
    if return_figure:
        figure = make_iq_scatter_figure(
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
        if save_image:
            save_figure(
                figure,
                name="plot_state_distribution",
                width=width,
                height=height,
            )
        return figure

    _scatter_iq_data(
        data,
        mode=mode,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        text=text,
        template=template,
        save_image=save_image,
    )
    return None


__all__ = [
    "COLORS",
    "DEFAULT_HEIGHT",
    "DEFAULT_IMAGES_DIR",
    "DEFAULT_TEMPLATE",
    "DEFAULT_WIDTH",
    "FIGURE_SIZE_IQ",
    "FIGURE_SIZE_STANDARD",
    "FIGURE_SIZE_TALL",
    "IQ_AXIS_MARGIN_LEFT",
    "IQ_AXIS_MARGIN_RIGHT",
    "FigureSize",
    "display_bloch_sphere",
    "get_colors",
    "get_config",
    "make_bloch_vectors_figure",
    "make_cdf_figure",
    "make_fft_figure",
    "make_figure",
    "make_iq_scatter_figure",
    "make_plot_figure",
    "make_waveform_figure",
    "plot",
    "plot_bloch_vectors",
    "plot_cdf",
    "plot_fft",
    "plot_waveform",
    "save_figure",
    "save_figure_image",
    "scatter_iq_data",
    "show_figure",
]
