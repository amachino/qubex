"""Tests for sampling-period behavior in analysis visualization."""

from __future__ import annotations

from typing import cast

import numpy as np
import plotly.graph_objs as go

from qubex.analysis import visualization as viz


def test_plot_waveform_uses_default_sampling_period() -> None:
    """Given no sampling period, when plotting waveform, then default 2.0 ns is used."""
    figure = viz.plot_waveform(
        np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
        return_figure=True,
    )

    assert figure is not None
    trace = cast(go.Scatter, figure.data[0])
    assert np.allclose(np.asarray(trace.x), np.array([0.0, 2.0, 4.0]))
