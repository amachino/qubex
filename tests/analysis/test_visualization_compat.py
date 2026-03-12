"""Compatibility tests for `qubex.analysis.visualization`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objs as go
import pytest

from qubex.analysis import visualization as legacy_viz


def test_analysis_visualization_module_exposes_legacy_exports() -> None:
    """Given legacy analysis visualization import, when reading exports, then shim symbols are available."""
    assert legacy_viz.DEFAULT_TEMPLATE == "qubex"
    assert legacy_viz.make_plot_figure is not None
    assert legacy_viz.plot is not None
    assert legacy_viz.save_figure_image is not None
    assert legacy_viz.display_bloch_sphere is not None


def test_legacy_plot_returns_figure_when_requested() -> None:
    """Given legacy plot API, when return_figure is true, then a Plotly figure is returned."""
    with pytest.warns(
        DeprecationWarning,
        match="Use `qubex\\.visualization\\.plot` or `make_plot_figure` instead\\.",
    ):
        figure = legacy_viz.plot(
            y=np.array([0.0, 1.0, 2.0]),
            return_figure=True,
        )

    assert isinstance(figure, go.Figure)


def test_save_figure_image_delegates_to_current_helper(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given legacy save helper, when saving, then it delegates to the current save_figure API."""
    figure = go.Figure()
    calls: list[tuple[go.Figure, str, Path, str, int | None, int | None, int]] = []

    def fake_save_figure(
        fig: go.Figure,
        name: str = "image",
        *,
        images_dir: Path | str = "./images",
        format: str = "png",
        width: int | None = None,
        height: int | None = None,
        scale: int = 3,
    ) -> Path:
        calls.append(
            (
                fig,
                name,
                Path(images_dir),
                format,
                width,
                height,
                scale,
            )
        )
        return Path(images_dir) / f"{name}.{format}"

    monkeypatch.setattr(legacy_viz, "save_figure", fake_save_figure)

    with pytest.warns(
        DeprecationWarning,
        match="Use `qubex\\.visualization\\.save_figure` instead\\.",
    ):
        result = legacy_viz.save_figure_image(
            figure,
            name="compat",
            images_dir=tmp_path,
            format="svg",
            width=320,
            height=180,
            scale=2,
        )

    assert result is None
    assert calls == [(figure, "compat", tmp_path, "svg", 320, 180, 2)]
