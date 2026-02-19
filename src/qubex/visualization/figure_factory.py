"""Factory helpers for creating and showing qubex-styled Plotly figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qubex.style import (
    HEIGHT as STYLE_DEFAULT_HEIGHT,
    WIDTH as STYLE_DEFAULT_WIDTH,
    get_config,
)

DEFAULT_TEMPLATE = "qubex"


@dataclass(frozen=True, slots=True)
class FigureSize:
    """Pair of width and height for a figure preset."""

    width: int
    height: int


FIGURE_SIZE_STANDARD = FigureSize(
    width=STYLE_DEFAULT_WIDTH,
    height=STYLE_DEFAULT_HEIGHT,
)
FIGURE_SIZE_TALL = FigureSize(
    width=STYLE_DEFAULT_WIDTH,
    height=400,
)
FIGURE_SIZE_IQ = FigureSize(
    width=500,
    height=400,
)

IQ_AXIS_MARGIN_LEFT = 120
IQ_AXIS_MARGIN_RIGHT = 120

MEASUREMENT_SCHEDULE_DEFAULT_WIDTH = 900
MEASUREMENT_SCHEDULE_ROW_HEIGHT = 90
MEASUREMENT_SCHEDULE_BASE_HEIGHT = 170

SEQUENCER_TIMELINE_DEFAULT_WIDTH = 900
SEQUENCER_TIMELINE_MIN_HEIGHT = 320
SEQUENCER_TIMELINE_BASE_HEIGHT = 120
SEQUENCER_TIMELINE_LANE_HEIGHT = 42

DEFAULT_WIDTH = FIGURE_SIZE_STANDARD.width
DEFAULT_HEIGHT = FIGURE_SIZE_STANDARD.height


def make_figure(
    *,
    template: str = DEFAULT_TEMPLATE,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Create a styled Plotly figure."""
    figure = go.Figure()
    _apply_figure_style(
        figure,
        template=template,
        width=width,
        height=height,
    )
    return figure


def make_subplots_figure(
    *,
    template: str = DEFAULT_TEMPLATE,
    width: int | None = None,
    height: int | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a styled Plotly subplots figure."""
    figure = make_subplots(**kwargs)
    _apply_figure_style(
        figure,
        template=template,
        width=width,
        height=height,
    )
    return figure


def show_figure(
    figure: go.Figure,
    *,
    filename: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Show a figure with qubex export configuration."""
    figure.show(
        config=get_config(
            filename=filename,
            width=_resolve_layout_dimension(figure, "width", default=width),
            height=_resolve_layout_dimension(figure, "height", default=height),
        )
    )


def _apply_figure_style(
    figure: go.Figure,
    *,
    template: str,
    width: int | None,
    height: int | None,
) -> None:
    """Apply common qubex style attributes to a figure."""
    layout_kwargs: dict[str, Any] = {"template": template}
    if width is not None:
        layout_kwargs["width"] = width
    if height is not None:
        layout_kwargs["height"] = height
    figure.update_layout(**layout_kwargs)


def _resolve_layout_dimension(
    figure: go.Figure,
    key: str,
    *,
    default: int | None,
) -> int:
    """Resolve a numeric layout dimension with a fallback."""
    layout = figure.to_dict().get("layout", {})
    if isinstance(layout, dict):
        value = layout.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    if default is not None:
        return default
    return DEFAULT_WIDTH if key == "width" else DEFAULT_HEIGHT
