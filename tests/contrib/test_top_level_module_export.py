"""Tests for top-level contrib module export."""

import qubex
from qubex import contrib
from qubex.contrib import experiment as experiment_contrib

_CR_PUBLIC_API = {
    "CrOnNoise",
    "CrPulseCoherenceMeasurements",
    "IdleQubitNoise",
    "ZX90GateTiming",
    "analyze_cr_pulse_coherence",
    "analyze_cr_pulse_fidelity",
    "characterize_cr_pulse_coherence",
    "extract_zx90_gate_timing",
    "fit_cr_on_control_dephasing",
    "fit_cr_on_target_rotating_frame_dephasing",
    "fit_exponential_decay",
    "fit_target_leakage",
    "fit_target_t1rho",
    "fit_three_level_rate_model",
    "plot_cr_pulse_coherence",
    "prepare_cr_echo_decay_model",
    "simulate_cr_pulse_fidelity",
}
_CR_MODULE_ONLY_API = {
    "DEFAULT_N_VALUES",
    "CrEchoDecayModel",
    "CrOnDephasingFit",
    "CrPulseCoherenceAnalysis",
    "CrPulseFidelityAnalysis",
    "CrPulseFidelitySimulationResult",
    "ExponentialDecayFit",
    "TargetLeakageFit",
    "TargetLeakageModel",
    "TargetT1RhoFit",
    "TargetT1RhoModel",
    "ThreeLevelRateFit",
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
    assert callable(contrib.simulate_cr_pulse_fidelity)
    assert callable(contrib.prepare_cr_echo_decay_model)
    assert callable(contrib.fit_cr_on_control_dephasing)
    assert callable(contrib.fit_cr_on_target_rotating_frame_dephasing)


def test_cr_exports_are_limited_to_callable_workflows_and_input_models() -> None:
    """CR package exports should omit constants and return-only result types."""
    for module in (contrib, experiment_contrib):
        exported = set(module.__all__)
        assert exported >= _CR_PUBLIC_API
        assert _CR_MODULE_ONLY_API.isdisjoint(exported)
