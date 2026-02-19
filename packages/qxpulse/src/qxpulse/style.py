"""Shared Plotly styling helpers for pulse visualizations."""

from __future__ import annotations

from typing import Any, Literal

import plotly.graph_objects as go
import plotly.io as pio

COLORS = [
    "#0C5DA5",
    "#00B945",
    "#FF9500",
    "#FF2C00",
    "#845B97",
    "#474747",
    "#9e9e9e",
]
FONT_FAMILY = "Times New Roman, Times, serif"
WIDTH = 600
HEIGHT = 300
MARGIN_L = 70
MARGIN_R = 70
MARGIN_B = 70
MARGIN_T = 70
TITLE_FONT_SIZE = 18
SUBTITLE_FONT_SIZE = 13
AXIS_TITLEFONT_SIZE = 16
AXIS_TICKFONT_SIZE = 14
LEGEND_FONT_SIZE = 14

pio.templates["qubex"] = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family=FONT_FAMILY,
        ),
        title=dict(
            font=dict(size=TITLE_FONT_SIZE),
            subtitle_font=dict(size=SUBTITLE_FONT_SIZE),
        ),
        xaxis=dict(
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False,
            linecolor="black",
            linewidth=1,
            ticks="inside",
            tickfont=dict(size=AXIS_TICKFONT_SIZE),
            titlefont=dict(size=AXIS_TITLEFONT_SIZE),
            domain=[0.0, 1.0],
        ),
        yaxis=dict(
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False,
            linecolor="black",
            linewidth=1,
            ticks="inside",
            tickfont=dict(size=AXIS_TICKFONT_SIZE),
            titlefont=dict(size=AXIS_TITLEFONT_SIZE),
            domain=[0.0, 1.0],
        ),
        legend=dict(
            font=dict(size=LEGEND_FONT_SIZE),
        ),
        modebar=dict(
            add=[],
            remove=[],
        ),
        autosize=False,
        width=WIDTH,
        height=HEIGHT,
        margin=dict(
            l=MARGIN_L,
            r=MARGIN_R,
            b=MARGIN_B,
            t=MARGIN_T,
        ),
        colorway=COLORS,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
    ),
    data=dict(
        scatter=[
            go.Scatter(
                mode="markers",
                marker=dict(size=6),
            )
        ]
    ),
)

pio.templates["square"] = go.layout.Template(
    layout=go.Layout(
        autosize=False,
        width=500,
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    ),
)


def apply_template(template: str) -> None:
    """Set the default Plotly template."""
    pio.templates.default = template


def hex_to_rgba(
    hex_color: str,
    alpha: float = 1.0,
) -> tuple:
    """Convert a HEX color string to an RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    return (*tuple(int(hex_color[i : i + 2], 16) for i in range(0, 6, 2)), alpha)


def get_colors(alpha: float = 1.0) -> list[tuple]:
    """Return the configured color palette as RGBA tuples."""
    return [hex_to_rgba(color, alpha) for color in COLORS]


def get_config(
    format: Literal["png", "svg", "jpeg", "webp"] = "png",
    filename: str = "image",
    scale: int = 3,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Return Plotly export configuration options."""
    options: dict[str, Any] = {
        "format": format,
        "filename": filename,
        "scale": scale,
    }
    if width is not None:
        options["width"] = width
    if height is not None:
        options["height"] = height
    return {"toImageButtonOptions": options}
