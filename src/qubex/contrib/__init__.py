"""Community-contributed experimental features."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS: dict[str, list[str]] = {
    "qubex.contrib.experiment.crosstalk_cross_resonance": [
        "cr_crosstalk_hamiltonian_tomography",
        "measure_cr_crosstalk",
    ],
    "qubex.contrib.experiment.multipartite_entanglement": [
        "create_1d_cluster_sequence",
        "create_connected_graphs",
        "create_cz_rounds",
        "create_entangle_sequence",
        "create_ghz_sequence",
        "create_graph_sequence",
        "create_maximum_1d_chain",
        "create_maximum_directed_tree",
        "create_maximum_graph",
        "create_maximum_spanning_tree",
        "create_measurement_rounds",
        "create_mqc_sequence",
        "fourier_analysis",
        "ghz_state_tomography",
        "measure_1d_cluster_state",
        "measure_bell_state_fidelities",
        "measure_bell_states",
        "measure_ghz_state",
        "measure_graph_state",
        "mqc_experiment",
        "parity_oscillation",
        "partial_transpose",
        "visualize_graph",
    ],
    "qubex.contrib.experiment.purity_benchmarking": [
        "interleaved_purity_benchmarking",
        "ipb_experiment",
        "pb_experiment_1q",
        "pb_experiment_2q",
        "purity_benchmarking",
        "purity_sequence_1q",
        "purity_sequence_2q",
    ],
    "qubex.contrib.experiment.rzx_gate": [
        "rzx",
        "rzx_gate_property",
    ],
    "qubex.contrib.experiment.simultaneous_coherence_measurement": [
        "simultaneous_coherence_measurement",
    ],
    "qubex.contrib.experiment.stark_characterization": [
        "stark_ramsey_experiment",
        "stark_t1_experiment",
    ],
    "qubex.contrib.gmm_linear_classification": [
        "build_gmm_linear_line_param",
        "build_gmm_linear_line_params",
    ],
}

_LAZY_ATTR_TO_MODULE = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}

__all__ = list(_LAZY_ATTR_TO_MODULE)


def __getattr__(name: str) -> Any:
    """Resolve contrib exports lazily to avoid package import cycles."""
    module_name = _LAZY_ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public names exposed by this package."""
    return sorted({*globals(), *__all__})
