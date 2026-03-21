"""Tests for Plotly figure export behavior."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

import qubex.visualization as viz


def test_save_figure_returns_output_path_when_export_succeeds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given export support, when saving a figure, then the output path is returned."""
    figure = go.Figure()

    def fake_write_image(
        self: go.Figure,
        file: Path,
        *,
        format: str,
        width: int,
        height: int,
        scale: int,
    ) -> None:
        assert self is figure
        assert format == "png"
        assert width == viz.DEFAULT_WIDTH
        assert height == viz.DEFAULT_HEIGHT
        assert scale == 3
        file.write_bytes(b"png")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)

    output_path = viz.save_figure(figure, name="export", images_dir=tmp_path)

    assert output_path is not None
    assert output_path.exists()
    assert output_path.parent == tmp_path
