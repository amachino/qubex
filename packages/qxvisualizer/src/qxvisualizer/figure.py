"""Helpers for creating, saving, and showing Plotly figures."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import plotly.graph_objects as go

from .style import (
    HEIGHT as STYLE_DEFAULT_HEIGHT,
    WIDTH as STYLE_DEFAULT_WIDTH,
    get_config,
)

logger = logging.getLogger(__name__)

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


def save_figure(
    figure: go.Figure,
    name: str = "image",
    *,
    images_dir: Path | str = "./images",
    format: Literal["png", "svg", "jpeg", "webp"] = "png",
    width: int | None = None,
    height: int | None = None,
    scale: int = 3,
) -> Path:
    """Save a figure image file and return the output path."""
    output_dir = Path(images_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_width = _resolve_layout_dimension(figure, "width", default=width)
    resolved_height = _resolve_layout_dimension(figure, "height", default=height)

    counter = 1
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"{current_date}_{name}_{counter}.{format}"

    while output_path.exists():
        counter += 1
        output_path = output_dir / f"{current_date}_{name}_{counter}.{format}"

    figure.write_image(
        output_path,
        format=format,
        width=resolved_width,
        height=resolved_height,
        scale=scale,
    )
    logger.info("Image saved to %s", output_path)
    return output_path


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
