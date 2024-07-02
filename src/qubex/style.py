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
FONT_FAMILY = "Times New Roman"
WIDTH = 600
HEIGHT = 300
MARGIN_L = 70
MARGIN_R = 70
MARGIN_B = 70
MARGIN_T = 70
TITLE_FONT_SIZE = 18
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
            x=0.5,
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
        plot_bgcolor="white",
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


def apply_template(template: str):
    pio.templates.default = template


def hex_to_rgba(
    hex_color: str,
    alpha: float = 1.0,
) -> tuple:
    """
    Convert a HEX color string to an RGBA tuple.

    Parameters
    ----------
    hex_color : str
        The HEX color string (e.g., '#FFAABB').
    alpha : float, optional
        The alpha value (transparency) of the color, by default 1.0.

    Returns
    -------
    tuple
        The RGBA tuple (e.g., (255, 170, 187, 1.0)).
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in range(0, 6, 2)) + (alpha,)


def get_colors(
    alpha: float = 1.0,
) -> list[tuple]:
    """
    Get a list of RGBA colors.

    Parameters
    ----------
    alpha : float, optional
        The alpha value (transparency) of the colors, by default 1.0.

    Returns
    -------
    list[tuple]
        A list of RGBA colors.
    """
    return [hex_to_rgba(color, alpha) for color in COLORS]


def get_config(
    format: Literal["png", "svg", "jpeg", "webp"] = "svg",
    filename: str = "image",
    height: int | None = None,
    width: int | None = None,
) -> dict:
    options: dict[str, Any] = {
        "format": format,
        "filename": filename,
        "scale": 3,
    }
    if height is not None:
        options["height"] = height
    if width is not None:
        options["width"] = width
    return {"toImageButtonOptions": options}
