"""Tests for top-level contrib module export."""

import qubex
from qubex import contrib


def test_contrib_module_is_exported_from_qubex() -> None:
    """`qubex` should export the `contrib` module."""
    assert "contrib" in qubex.__all__
    assert callable(contrib.measurement_induced_dephasing)
    assert callable(contrib.measurement_induced_dephasing_experiment)
    assert callable(contrib.measure_cr_crosstalk)
    assert callable(contrib.quantum_efficiency_measurement)
    assert callable(contrib.readout_snr)
    assert callable(contrib.sweep_readout_snr)
    assert callable(contrib.simultaneous_coherence_measurement)
    assert callable(contrib.purity_benchmarking)
