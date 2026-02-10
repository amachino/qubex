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
