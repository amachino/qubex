"""Tests for top-level contrib module export."""

import qubex
from qubex import contrib
from qubex.contrib import experiment as experiment_contrib

_CR_PUBLIC_API = {
    "analyze_cr_pulse_health",
    "characterize_cr_pulse_health",
    "plot_cr_pulse_health",
}
_CR_MODULE_ONLY_API = {
    "DEFAULT_N_VALUES",
    "ChangeAssessment",
    "ControlRateHealthFit",
    "CrPulseHealthAnalysis",
    "CrPulseHealthFidelityEstimate",
    "CrPulseHealthMeasurements",
    "DecayHealthFit",
    "ExchangeHealthFit",
    "IdleHealthNoise",
    "ZX90Timing",
}


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
    assert callable(contrib.get_superconducting_gap)
    assert callable(contrib.get_resistance_charge)
    assert callable(contrib.analyze_cr_pulse_health)
    assert callable(contrib.characterize_cr_pulse_health)
    assert callable(contrib.plot_cr_pulse_health)


def test_cr_exports_are_limited_to_workflow_functions() -> None:
    """CR package exports should omit constants and data models."""
    for module in (contrib, experiment_contrib):
        exported = set(module.__all__)
        assert exported >= _CR_PUBLIC_API
        assert _CR_MODULE_ONLY_API.isdisjoint(exported)
