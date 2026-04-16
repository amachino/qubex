"""Tests for functional APIs in `qubex.contrib.experiment.quantum_efficiency_measurement`."""

from __future__ import annotations

from qubex.contrib import (
    measurement_induced_dephasing,
    measurement_induced_dephasing_experiment,
    quantum_efficiency_measurement,
    readout_snr,
    sweep_readout_snr,
)


def test_quantum_efficiency_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then quantum-efficiency helpers are available."""
    assert callable(measurement_induced_dephasing)
    assert callable(measurement_induced_dephasing_experiment)
    assert callable(readout_snr)
    assert callable(sweep_readout_snr)
    assert callable(quantum_efficiency_measurement)
