"""Visualization helpers backed by qxvisualizer plotting primitives."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from qxvisualizer.figure import (
    DEFAULT_TEMPLATE,
    FIGURE_SIZE_IQ,
    IQ_AXIS_MARGIN_LEFT,
    IQ_AXIS_MARGIN_RIGHT,
    make_figure,
    save_figure,
)
from qxvisualizer.plotting import (
    DEFAULT_IMAGES_DIR,
    make_bloch_vectors_figure,
    make_cdf_figure,
    make_fft_figure,
    make_iq_scatter_figure,
    make_plot_figure,
    make_waveform_figure,
    plot,
    plot_bloch_vectors,
    plot_cdf,
    plot_fft,
    plot_waveform,
    scatter_iq_data,
)
from qxvisualizer.style import get_colors


def make_classification_figure(
    *,
    target: str,
    data: NDArray,
    labels: NDArray,
    centers: Mapping[int, complex],
    stddevs: Mapping[int, float] | None = None,
    n_samples: int = 1000,
) -> go.Figure:
    """Build a state-classification I/Q figure for one prepared-state dataset."""
    plot_data = np.asarray(data)
    plot_labels = np.asarray(labels)

    if len(plot_data) > n_samples:
        plot_data = plot_data[:n_samples]
        plot_labels = plot_labels[:n_samples]

    x = np.real(plot_data)
    y = np.imag(plot_data)
    unique_labels = np.unique(plot_labels)
    colors = get_colors(alpha=0.8)

    max_candidates = [float(np.max(np.abs(plot_data))) if plot_data.size else 0.0]
    max_candidates.extend(float(np.abs(center)) for center in centers.values())
    if stddevs is not None:
        max_candidates.extend(
            float(np.abs(center) + 2 * stddevs[label])
            for label, center in centers.items()
            if label in stddevs
        )
    max_val = max(max_candidates, default=1.0)
    if max_val == 0:
        max_val = 1.0
    axis_range = [-max_val * 1.1, max_val * 1.1]
    dtick = max_val / 2

    fig = make_figure(
        template=DEFAULT_TEMPLATE,
        width=FIGURE_SIZE_IQ.width,
        height=FIGURE_SIZE_IQ.height,
    )
    for idx, label in enumerate(unique_labels):
        color = colors[idx % len(colors)]
        mask = plot_labels == label
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=f"|{label}⟩",
                marker=dict(
                    size=4,
                    color=f"rgba{color}",
                ),
            )
        )
    for label, center in centers.items():
        fig.add_trace(
            go.Scatter(
                x=[center.real],
                y=[center.imag],
                mode="markers",
                name=f"|{label}⟩",
                showlegend=True,
                marker=dict(
                    size=10,
                    color="black",
                    symbol="x",
                ),
            )
        )
    if stddevs is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        for label, center in centers.items():
            sigma = stddevs.get(label)
            if sigma is None:
                continue
            x_circle = center.real + 2 * sigma * np.cos(theta)
            y_circle = center.imag + 2 * sigma * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode="lines",
                    name=f"|{label}⟩ ± 2σ",
                    showlegend=True,
                    line=dict(
                        color="black",
                        width=2,
                        dash="dot",
                    ),
                )
            )

    fig.update_layout(
        title=f"State classification : {target}",
        xaxis_title="In-Phase (arb. units)",
        yaxis_title="Quadrature (arb. units)",
        showlegend=True,
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


__all__ = [
    "DEFAULT_IMAGES_DIR",
    "make_bloch_vectors_figure",
    "make_cdf_figure",
    "make_classification_figure",
    "make_fft_figure",
    "make_iq_scatter_figure",
    "make_plot_figure",
    "make_waveform_figure",
    "plot",
    "plot_bloch_vectors",
    "plot_cdf",
    "plot_fft",
    "plot_waveform",
    "save_figure",
    "scatter_iq_data",
]
