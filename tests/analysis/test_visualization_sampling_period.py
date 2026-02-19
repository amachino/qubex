"""Tests for sampling-period behavior in analysis visualization."""

from __future__ import annotations

from typing import cast

import numpy as np
import plotly.graph_objs as go

import qubex.visualization as viz


def test_make_waveform_figure_uses_default_sampling_period() -> None:
    """Given no sampling period, when making waveform figure, then default 2.0 ns is used."""
    figure = viz.make_waveform_figure(np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]))

    trace = cast(go.Scatter, figure.data[0])
    assert np.allclose(np.asarray(trace.x), np.array([0.0, 2.0, 4.0]))


def test_plot_waveform_returns_none(monkeypatch) -> None:
    """Given plot waveform API, when plotting, then it returns None."""
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)
    result = viz.plot_waveform(np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]))
    assert result is None


def test_make_figure_applies_template_and_size() -> None:
    """Given figure factory input, when creating figure, then template and size are applied."""
    figure = viz.make_figure(template="qubex", width=321, height=123)
    assert figure.layout.width == 321
    assert figure.layout.height == 123
    assert figure.layout.template is not None


def test_make_plot_figure_uses_standard_size() -> None:
    """Given no size input, when making plot figure, then standard preset is used."""
    figure = viz.make_plot_figure(y=np.array([0.0, 1.0, 2.0]))
    assert figure.layout.width == viz.FIGURE_SIZE_STANDARD.width
    assert figure.layout.height == viz.FIGURE_SIZE_STANDARD.height


def test_qxvisualizer_exports_waveform_helper_with_default_sampling_period() -> None:
    """Given qxvisualizer API, when making waveform figure, then default 2.0 ns is used."""
    import qxvisualizer as qviz

    figure = qviz.make_waveform_figure(np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]))
    trace = cast(go.Scatter, figure.data[0])
    assert np.allclose(np.asarray(trace.x), np.array([0.0, 2.0, 4.0]))


def test_qubex_visualization_reuses_qxvisualizer_plotting_helpers() -> None:
    """Given plotting helper exports, when comparing modules, then qubex reuses qxvisualizer implementation."""
    import qxvisualizer as qviz

    assert viz.make_plot_figure is qviz.make_plot_figure
