from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Collection, Literal, Mapping

import numpy as np
import plotly.graph_objs as go
import qctrlvisualizer as qcv
from numpy.typing import ArrayLike, NDArray

from ..style import COLORS, get_colors, get_config
from ..typing import IQArray


def display_bloch_sphere(bloch_vectors: NDArray[np.float64]):
    qcv.display_bloch_sphere_from_bloch_vectors(bloch_vectors)


def save_figure_image(
    fig: go.Figure,
    name: str = "image",
    *,
    images_dir: Path | str = "./images",
    format: Literal["png", "svg", "jpeg", "webp"] = "png",
    width: int | None = None,
    height: int | None = None,
    scale: int | None = None,
):
    width = width or 600
    height = height or 300
    scale = scale or 3

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

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
    print(f"Image saved to {file_path}")


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
    template: str = "qubex",
    return_figure: bool = False,
    save_image: bool = False,
    **kwargs,
):
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
    if save_image:
        save_figure_image(
            fig,
            name="plot",
            width=width,
            height=height,
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
    template: str = "qubex",
    return_figure: bool = False,
    save_image: bool = False,
    **kwargs,
):
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
    if save_image:
        save_figure_image(
            fig,
            name="plot_fft",
            width=width,
            height=height,
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
    template: str = "qubex",
    return_figure: bool = False,
    save_image: bool = False,
):
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
    else:
        fig.show(
            config=get_config(
                filename="plot_bloch_vectors",
                width=width,
                height=height,
            )
        )
    if save_image:
        save_figure_image(
            fig,
            name="plot_bloch_vectors",
            width=width,
            height=height,
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
    template: str = "qubex",
    return_figure: bool = False,
    save_image: bool = False,
):
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
    if save_image:
        save_figure_image(
            fig,
            name="plot_waveform",
            width=width,
            height=height,
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
    template: str = "qubex",
    return_figure: bool = False,
    save_image: bool = False,
):
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
    if save_image:
        save_figure_image(
            fig,
            name="plot_state_distribution",
            width=width,
            height=height,
        )
