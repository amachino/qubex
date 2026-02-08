"""Tests for functional APIs in `qubex.contrib.multipartite_entanglement`."""

from __future__ import annotations

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
