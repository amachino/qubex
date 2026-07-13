"""Tests for IQ plotter color assignment."""

from __future__ import annotations

import numpy as np

import qubex.analysis.iq_plotter as iq_plotter
from qubex.analysis.iq_plotter import IQPlotter


def test_iq_plotter_uses_distinct_colors_for_distinct_labels(monkeypatch) -> None:
    """Given mixed labels, when initializing traces, then distinct labels use distinct colors."""
    monkeypatch.setattr(iq_plotter, "display", lambda *_args, **_kwargs: None)

    plotter = IQPlotter(
        state_centers={
            "Q05": {0: 0.0 + 0.0j, 1: 1.0 + 0.0j},
            "Q06": {0: 0.0 + 0.0j, 1: 0.0 + 1.0j},
        }
    )

    plotter.update(
        {
            "Q00": np.array([1.0 + 0.0j]),
            "Q05": np.array([0.0 + 1.0j]),
        }
    )

    widget = plotter.__dict__["_widget"]
    data_trace_colors = {
        trace.meta: trace.marker.color
        for trace in widget.data
        if getattr(trace, "meta", None) in {"Q00", "Q05"}
    }
    assert data_trace_colors["Q00"] != data_trace_colors["Q05"]


def test_iq_plotter_legendrank_follows_input_order(monkeypatch) -> None:
    """Given mixed labels, when creating traces, then legend rank follows input order."""
    monkeypatch.setattr(iq_plotter, "display", lambda *_args, **_kwargs: None)

    plotter = IQPlotter(
        state_centers={
            "Q05": {0: 0.0 + 0.0j, 1: 1.0 + 0.0j},
            "Q06": {0: 0.0 + 0.0j, 1: 0.0 + 1.0j},
        }
    )

    plotter.update(
        {
            "Q00": np.array([1.0 + 0.0j]),
            "Q05": np.array([0.0 + 1.0j]),
        }
    )

    widget = plotter.__dict__["_widget"]
    data_trace_ranks = {
        trace.meta: trace.legendrank
        for trace in widget.data
        if getattr(trace, "meta", None) in {"Q00", "Q05"}
    }
    assert data_trace_ranks == {"Q00": 0, "Q05": 1}


def test_iq_plotter_does_not_reassign_widget_data_on_initial_update(
    monkeypatch,
) -> None:
    """Given state traces, when first updating, then the widget trace collection is not reassigned."""
    monkeypatch.setattr(iq_plotter, "display", lambda *_args, **_kwargs: None)

    class _FakeFigureWidget:
        def __init__(self) -> None:
            self._data = []

        @property
        def data(self):
            return tuple(self._data)

        @data.setter
        def data(self, _value) -> None:
            raise AssertionError("FigureWidget.data should not be reassigned")

        def update_layout(self, **_kwargs) -> None:
            return None

        def add_trace(self, trace) -> None:
            self._data.append(trace)

        def add_scatter(self, **kwargs) -> None:
            self._data.append(iq_plotter.go.Scatter(**kwargs))

    monkeypatch.setattr(iq_plotter.go, "FigureWidget", _FakeFigureWidget)

    plotter = IQPlotter(
        state_centers={
            "Q05": {0: 0.0 + 0.0j, 1: 1.0 + 0.0j},
        }
    )

    plotter.update({"Q05": np.array([0.0 + 1.0j])})

    widget = plotter.__dict__["_widget"]
    data_traces = [
        trace for trace in widget.data if getattr(trace, "meta", None) == "Q05"
    ]

    assert len(data_traces) == 1
    assert np.array_equal(np.asarray(data_traces[0].x), np.array([0.0]))
    assert np.array_equal(np.asarray(data_traces[0].y), np.array([1.0]))


def test_iq_plotter_flattens_non_1d_inputs(monkeypatch) -> None:
    """Given non-1D IQ input, when updating, then scatter coordinates are flattened."""
    monkeypatch.setattr(iq_plotter, "display", lambda *_args, **_kwargs: None)

    plotter = IQPlotter()

    plotter.update(
        {
            "Q00": np.array(
                [[1.0 + 2.0j], [3.0 + 4.0j]],
                dtype=np.complex128,
            )
        }
    )

    widget = plotter.__dict__["_widget"]
    trace = next(
        trace for trace in widget.data if getattr(trace, "meta", None) == "Q00"
    )

    assert np.array_equal(np.asarray(trace.x), np.array([1.0, 3.0]))
    assert np.array_equal(np.asarray(trace.y), np.array([2.0, 4.0]))


def test_iq_plotter_show_displays_detached_figure_and_closes_widget(
    monkeypatch,
) -> None:
    """Given a live widget, when showing final output, then a detached Figure is shown."""
    monkeypatch.setattr(iq_plotter, "display", lambda *_args, **_kwargs: None)

    shown_figures = []

    def _record_figure_show(self, **kwargs) -> None:
        shown_figures.append((self, kwargs))

    def _fail_widget_show(self, **_kwargs) -> None:
        raise AssertionError("FigureWidget.show should not be used for final output")

    monkeypatch.setattr(iq_plotter.go.Figure, "show", _record_figure_show)
    monkeypatch.setattr(iq_plotter.go.FigureWidget, "show", _fail_widget_show)

    plotter = IQPlotter()
    plotter.update({"Q00": np.array([1.0 + 2.0j])})

    plotter.show()

    assert len(shown_figures) == 1
    shown_figure, kwargs = shown_figures[0]
    assert isinstance(shown_figure, iq_plotter.go.Figure)
    assert not isinstance(shown_figure, iq_plotter.go.FigureWidget)
    assert kwargs == {"config": iq_plotter.get_config()}
    assert plotter.__dict__["_closed"] is True


def test_iq_plotter_clear_closes_widget_once(monkeypatch) -> None:
    """Given a live widget, when clearing repeatedly, then widget resources close once."""
    plotter = IQPlotter()
    closed = []

    monkeypatch.setattr(
        plotter.__dict__["_widget"], "close", lambda: closed.append(True)
    )

    plotter.clear()
    plotter.clear()

    assert closed == [True]
