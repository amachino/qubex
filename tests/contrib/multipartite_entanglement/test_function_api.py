"""Tests for functional APIs in `qubex.contrib.experiment.multipartite_entanglement`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go

from qubex.contrib import (
    create_1d_cluster_sequence,
    create_connected_graphs,
    create_cz_rounds,
    create_entangle_sequence,
    create_ghz_sequence,
    create_graph_sequence,
    create_maximum_1d_chain,
    create_maximum_directed_tree,
    create_maximum_graph,
    create_maximum_spanning_tree,
    create_measurement_rounds,
    create_mqc_sequence,
    fourier_analysis,
    ghz_state_tomography,
    measure_1d_cluster_state,
    measure_bell_state_fidelities,
    measure_bell_states,
    measure_ghz_state,
    measure_graph_state,
    mqc_experiment,
    parity_oscillation,
    partial_transpose,
    visualize_graph,
)
from qubex.contrib.experiment import multipartite_entanglement as me


def test_all_multipartite_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then multipartite helpers are available."""
    assert callable(create_entangle_sequence)
    assert callable(create_ghz_sequence)
    assert callable(measure_ghz_state)
    assert callable(ghz_state_tomography)
    assert callable(create_mqc_sequence)
    assert callable(mqc_experiment)
    assert callable(fourier_analysis)
    assert callable(parity_oscillation)
    assert callable(create_1d_cluster_sequence)
    assert callable(measure_1d_cluster_state)
    assert callable(partial_transpose)
    assert callable(create_connected_graphs)
    assert callable(create_maximum_graph)
    assert callable(create_maximum_1d_chain)
    assert callable(create_maximum_spanning_tree)
    assert callable(create_maximum_directed_tree)
    assert callable(create_cz_rounds)
    assert callable(create_graph_sequence)
    assert callable(create_measurement_rounds)
    assert callable(visualize_graph)
    assert callable(measure_graph_state)
    assert callable(measure_bell_state_fidelities)
    assert callable(measure_bell_states)


def test_ghz_state_tomography_should_set_primary_and_named_figures(
    monkeypatch,
) -> None:
    """GHZ state tomography should expose primary and named figures on Result."""
    fig_raw = go.Figure()
    fig_mit = go.Figure()
    fig_mle = go.Figure()
    plot_calls = {"count": 0}

    monkeypatch.setattr(
        me,
        "resolve_shot_options",
        lambda **_kwargs: (32, 1_024.0),
    )
    monkeypatch.setattr(
        me,
        "measure_ghz_state",
        lambda *_args, **_kwargs: {
            "raw": [0.5, 0.0, 0.0, 0.5],
            "mitigated": [0.5, 0.0, 0.0, 0.5],
        },
    )
    monkeypatch.setattr(
        me,
        "create_density_matrix",
        lambda *_args, **_kwargs: np.diag([1.0, 0.0, 0.0, 0.0]).astype(
            np.complex128
        ),
    )

    def _plot_ghz_state_tomography(**_kwargs):
        plot_calls["count"] += 1
        if plot_calls["count"] == 1:
            return {"figure": fig_raw}
        if plot_calls["count"] == 2:
            return {"figure": fig_mit}
        return {"figure": fig_mle}

    monkeypatch.setattr(me, "plot_ghz_state_tomography", _plot_ghz_state_tomography)

    result = ghz_state_tomography(
        exp=cast(Any, SimpleNamespace()),
        entangle_steps=[("Q00", "Q01")],
        plot=False,
        show_sequence=False,
        save_image=False,
        readout_mitigation=True,
        mle_fit=True,
    )

    assert result.figure is fig_mle
    assert result.figures == {
        "raw": fig_raw,
        "mitigated": fig_mit,
        "mle": fig_mle,
    }
