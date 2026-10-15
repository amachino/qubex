"""Tests for qxsimulator visualization compatibility exports."""

import qxsimulator.visualization as simulator_visualization
import qxvisualizer


def test_visualization_helpers_are_owned_by_qxvisualizer() -> None:
    """Qxsimulator visualization helpers should re-export qxvisualizer functions."""
    assert (
        simulator_visualization.make_bloch_vectors_figure
        is qxvisualizer.make_bloch_vectors_figure
    )
    assert simulator_visualization.plot_bloch_vectors is qxvisualizer.plot_bloch_vectors
