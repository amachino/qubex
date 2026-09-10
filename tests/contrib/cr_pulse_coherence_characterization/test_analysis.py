"""Tests for CR-pulse coherence analysis helpers."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from qubex.contrib.experiment import cr_pulse_coherence_analysis as module
from qubex.contrib.experiment.cr_pulse_coherence_analysis import (
    CrOnDephasingFit,
    CrPulseCoherenceMeasurements,
    CrPulseFidelityAnalysis,
    analyze_cr_pulse_coherence,
    fit_exponential_decay,
    fit_target_t1rho,
    fit_three_level_rate_model,
)
from qubex.contrib.experiment.cr_pulse_fidelity_simulation import (
    CrOnNoise,
    CrPulseFidelitySimulationResult,
    IdleQubitNoise,
    ZX90GateTiming,
)


def _rate_matrix(rates: tuple[float, float, float, float]) -> np.ndarray:
    """Return the adjacent-transition three-level rate matrix."""
    ge_down, ge_up, ef_down, ef_up = rates
    return np.array(
        [
            [-ge_up, ge_down, 0.0],
            [ge_up, -(ge_down + ef_up), ef_down],
            [0.0, ef_up, -ef_down],
        ]
    )


def _trajectory(
    times: np.ndarray,
    initial: np.ndarray,
    rates: tuple[float, float, float, float],
) -> np.ndarray:
    """Generate an exact population trajectory for synthetic fit data."""
    matrix = _rate_matrix(rates)
    return np.stack([expm(matrix * time) @ initial for time in times])


def _control_echo_measurements() -> CrPulseCoherenceMeasurements:
    """Return a minimal structurally valid offline-analysis data set."""
    times = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    values = np.array([0.9, 0.8, 0.7])
    errors = np.full(3, 0.01)
    return CrPulseCoherenceMeasurements(
        control_qubit="Q0",
        target_qubit="Q1",
        protocols=("control_t2_echo",),
        pauli_components={"control_t2_echo": ("X",)},
        n_values=(0, 1, 2),
        times={"control_t2_echo": times},
        populations={},
        population_standard_errors={},
        target_polarization_standard_errors={},
        pauli_expectations={
            "control_t2_echo": {
                "X": {"actual": values, "reference": values},
            }
        },
        pauli_standard_errors={
            "control_t2_echo": {
                "X": {"actual": errors, "reference": errors},
            }
        },
    )


def _gef_measurements_without_target_leakage() -> CrPulseCoherenceMeasurements:
    """Return synthetic GEF data with decaying target polarization and constant Pf."""
    times = np.array([0, 400, 900, 1800, 3200, 5200, 8000.0])
    n_points = times.size
    control_rates = (2.5e-5, 4.0e-6, 0.0, 0.0)
    control_populations = {
        "control_ground": _trajectory(
            times,
            np.array([0.98, 0.02, 0.0]),
            control_rates,
        ),
        "control_excited": _trajectory(
            times,
            np.array([0.03, 0.97, 0.0]),
            control_rates,
        ),
    }
    target_polarizations = {
        "control_ground": 0.9 * np.exp(-times / 4200.0),
        "control_excited": -0.8 * np.exp(-times / 6300.0),
    }

    def target_population(polarization: np.ndarray) -> np.ndarray:
        computational_population = 0.998
        return np.column_stack(
            (
                computational_population * (1 - polarization) / 2,
                computational_population * (1 + polarization) / 2,
                np.full(n_points, 1 - computational_population),
            )
        )

    populations = {
        protocol: {
            kind: {
                "Q0": control_populations[protocol],
                "Q1": target_population(target_polarizations[protocol]),
            }
            for kind in ("actual", "reference")
        }
        for protocol in ("control_ground", "control_excited")
    }
    population_errors = {
        protocol: {
            kind: {qubit: np.full((n_points, 3), 0.001) for qubit in ("Q0", "Q1")}
            for kind in ("actual", "reference")
        }
        for protocol in populations
    }
    polarization_errors = {
        protocol: {kind: np.full(n_points, 0.002) for kind in ("actual", "reference")}
        for protocol in populations
    }
    return CrPulseCoherenceMeasurements(
        control_qubit="Q0",
        target_qubit="Q1",
        protocols=("control_ground", "control_excited"),
        pauli_components={},
        n_values=(0, 4, 9, 18, 32, 52, 80),
        times=dict.fromkeys(populations, times),
        populations=populations,
        population_standard_errors=population_errors,
        target_polarization_standard_errors=polarization_errors,
        pauli_expectations={},
        pauli_standard_errors={},
    )


def _all_protocol_measurements() -> CrPulseCoherenceMeasurements:
    """Add valid C/D observables to the synthetic A/B measurement set."""
    measurements = _gef_measurements_without_target_leakage()
    times = measurements.times["control_ground"]
    values = np.exp(-times / 10_000.0)
    errors = np.full(times.shape, 0.01)
    return replace(
        measurements,
        protocols=(
            "control_ground",
            "control_excited",
            "control_t2_echo",
            "target_t2rho_echo",
        ),
        pauli_components={
            "control_t2_echo": ("X",),
            "target_t2rho_echo": ("Z",),
        },
        times={
            **measurements.times,
            "control_t2_echo": times,
            "target_t2rho_echo": times,
        },
        pauli_expectations={
            "control_t2_echo": {
                "X": {"actual": values, "reference": values},
            },
            "target_t2rho_echo": {
                "Z": {"actual": values, "reference": values},
            },
        },
        pauli_standard_errors={
            "control_t2_echo": {
                "X": {"actual": errors, "reference": errors},
            },
            "target_t2rho_echo": {
                "Z": {"actual": errors, "reference": errors},
            },
        },
    )


def test_fidelity_uncertainty_defaults_to_auto_and_skips_partial_data() -> None:
    """The default should not reject or decorate a partial-protocol analysis."""
    parameter = inspect.signature(analyze_cr_pulse_coherence).parameters[
        "propagate_fidelity_uncertainty"
    ]

    analysis = analyze_cr_pulse_coherence(_control_echo_measurements())

    assert parameter.default is None
    assert analysis.fidelity_analysis is None
    assert analysis.fit_status["fidelity_uncertainty_requested"] is None
    assert analysis.fit_status["fidelity_uncertainty_mode"] == "auto"
    assert analysis.fit_status["fidelity_uncertainty_enabled"] is False
    assert analysis.fit_status["fidelity_uncertainty_success"] is None
    assert analysis.fit_status["fidelity_uncertainty_option_semantics"] == {
        "None": "auto",
        "True": "explicitly_enabled",
        "False": "disabled",
    }
    assert (
        analysis.fit_status["fidelity_uncertainty_covariance_approximation"]
        == "block_diagonal_between_fit_stages"
    )


@pytest.mark.parametrize(
    ("data_max", "expected_upper_limit"),
    [
        (0.0, 0.01),
        (0.003, 0.01),
        (0.012, 0.02),
        (0.04, 0.05),
        (0.12, 0.2),
        (0.9, 1.0),
    ],
)
def test_target_leakage_axis_uses_readable_automatic_upper_limits(
    data_max: float,
    expected_upper_limit: float,
) -> None:
    """Target-Pf plots should use a padded, bounded 1/2/5-series range."""
    upper_limit = module._resolve_target_leakage_upper_limit(  # noqa: SLF001
        np.array([np.nan, data_max, np.inf])
    )

    assert upper_limit == pytest.approx(expected_upper_limit)


def test_target_leakage_axis_is_resolved_per_protocol_from_fit_curves() -> None:
    """Each A/B target panel should include its own actual fitted Pf curve."""
    measurements = _gef_measurements_without_target_leakage()
    analysis = analyze_cr_pulse_coherence(measurements)
    target_fit = cast(module.TargetABForwardFit, analysis.fits["target_ab_forward"])
    curve_shape = target_fit.curve_n_values.shape
    target_fit = replace(
        target_fit,
        curve_target_f={
            "control_ground": np.full(curve_shape, 0.12),
            "control_excited": np.full(curve_shape, 0.04),
        },
    )
    analysis = replace(
        analysis,
        fits={**analysis.fits, "target_ab_forward": target_fit},
    )

    figures = module.plot_cr_pulse_coherence(measurements, analysis)

    ground_layout: Any = figures["control_ground_target_polarization"].layout
    excited_layout: Any = figures["control_excited_target_polarization"].layout
    assert tuple(ground_layout.yaxis2.range) == (0.0, 0.2)
    assert tuple(excited_layout.yaxis2.range) == (0.0, 0.05)


def test_explicit_fidelity_uncertainty_rejects_partial_data() -> None:
    """True should retain the strict all-protocol requirement."""
    with pytest.raises(ValueError, match="requires all four protocols"):
        analyze_cr_pulse_coherence(
            _control_echo_measurements(),
            propagate_fidelity_uncertainty=True,
        )


def test_uncertainty_auto_disables_with_disabled_simulation() -> None:
    """None should resolve to disabled when fidelity simulation is disabled."""
    analysis = analyze_cr_pulse_coherence(
        _all_protocol_measurements(),
        run_fidelity_simulation=False,
    )

    assert analysis.fit_status["fidelity_uncertainty_mode"] == "auto"
    assert analysis.fit_status["fidelity_uncertainty_enabled"] is False
    assert analysis.fidelity_uncertainty is None


def test_explicit_uncertainty_rejects_disabled_simulation() -> None:
    """True and run_fidelity_simulation=False are contradictory."""
    with pytest.raises(ValueError, match="requires fidelity simulation"):
        analyze_cr_pulse_coherence(
            _all_protocol_measurements(),
            run_fidelity_simulation=False,
            propagate_fidelity_uncertainty=True,
        )


def test_uncertainty_auto_runs_with_nominal_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None should enable propagation after a successful full-protocol simulation."""
    noise = CrOnNoise(0.0, 0.0, 0.0, 0.0, np.inf, 0.0, 0.0)
    simulation = CrPulseFidelitySimulationResult(1.0, 1.0, 1.0, 0.0, {})
    propagated: list[bool] = []

    monkeypatch.setattr(module, "_fit_measured_observables", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_base_cr_on_noise_from_fits",
        lambda _fits: (noise, None),
    )
    monkeypatch.setattr(
        module,
        "analyze_cr_pulse_fidelity",
        lambda *_args, **_kwargs: CrPulseFidelityAnalysis(
            success=True,
            message="ok",
            cr_on_noise=noise,
            control_dephasing_fit=None,
            target_dephasing_fit=None,
            simulation=simulation,
        ),
    )
    monkeypatch.setattr(
        module,
        "_fidelity_parameter_covariance_from_fits",
        lambda *_args: (np.zeros((9, 9)), ()),
    )

    def propagate(
        *_args: object,
        **_kwargs: object,
    ) -> module.CrPulseFidelityLinearUncertainty:
        propagated.append(True)
        return module._failed_fidelity_linear_uncertainty(  # noqa: SLF001
            "sentinel"
        )

    monkeypatch.setattr(module, "propagate_cr_pulse_fidelity_uncertainty", propagate)

    analysis = analyze_cr_pulse_coherence(
        _all_protocol_measurements(),
        zx90_gate=ZX90GateTiming(20.0, True, 4.0),
        control_idle_noise=IdleQubitNoise(np.inf, np.inf),
        target_idle_noise=IdleQubitNoise(np.inf, np.inf),
        control_x180_duration=4.0,
        target_x180_duration=4.0,
        calculate_conservative_scenario=False,
    )

    assert propagated == [True]
    assert analysis.fit_status["fidelity_uncertainty_mode"] == "auto"
    assert analysis.fit_status["fidelity_uncertainty_enabled"] is True
    assert analysis.fidelity_uncertainty is not None


def test_offline_analysis_validates_cross_field_measurement_structure() -> None:
    """Malformed saved data should fail with a clear structural error."""
    measurements = _control_echo_measurements()

    assert measurements.cr_pulse_counts == (0, 4, 8)

    with pytest.raises(TypeError, match="must be a NumPy array"):
        analyze_cr_pulse_coherence(
            replace(measurements, times={"control_t2_echo": [0.0, 10.0, 20.0]})  # type: ignore[dict-item]
        )


def test_offline_analysis_fits_target_when_control_pulse_durations_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing control-protocol pulse durations should not suppress the target fit."""
    control_measurements = _control_echo_measurements()
    times = control_measurements.times["control_t2_echo"]
    values = np.array([0.9, 0.8, 0.7])
    errors = np.full(3, 0.01)
    measurements = replace(
        control_measurements,
        protocols=("control_t2_echo", "target_t2rho_echo"),
        pauli_components={
            "control_t2_echo": ("X",),
            "target_t2rho_echo": ("Z",),
        },
        times={
            "control_t2_echo": times,
            "target_t2rho_echo": times,
        },
        pauli_expectations={
            "control_t2_echo": {
                "X": {"actual": values, "reference": values},
            },
            "target_t2rho_echo": {
                "Z": {"actual": values, "reference": values},
            },
        },
        pauli_standard_errors={
            "control_t2_echo": {
                "X": {"actual": errors, "reference": errors},
            },
            "target_t2rho_echo": {
                "Z": {"actual": errors, "reference": errors},
            },
        },
    )
    zero_noise = CrOnNoise(
        gamma_control_g_to_e=0.0,
        gamma_control_e_to_g=0.0,
        gamma_control_e_to_f=0.0,
        gamma_control_f_to_e=0.0,
        target_t1rho=np.inf,
        target_leakage_rate=0.0,
        target_seepage_rate=0.0,
    )
    target_fit = CrOnDephasingFit(
        success=True,
        message="ok",
        gamma_phi=2e-5,
        gamma_phi_error=1e-6,
        covariance=np.array([[1e-12]]),
        amplitude=0.9,
        offset=0.0,
        fitted_values=values,
        curve_n_values=np.array([0, 1, 2]),
        curve_values=values,
        r_squared=1.0,
    )

    monkeypatch.setattr(
        module,
        "_base_cr_on_noise_from_fits",
        lambda _fits: (zero_noise, None),
    )
    monkeypatch.setattr(module, "prepare_cr_echo_decay_model", lambda *args: object())
    monkeypatch.setattr(
        module,
        "fit_cr_on_control_dephasing",
        lambda *args: pytest.fail("control fit should be skipped"),
    )
    monkeypatch.setattr(
        module,
        "fit_cr_on_target_rotating_frame_dephasing",
        lambda *args: target_fit,
    )

    analysis = analyze_cr_pulse_coherence(
        measurements,
        zx90_gate=ZX90GateTiming(
            cr_lobe_duration=20.0,
            echo=True,
            control_pi_duration=4.0,
        ),
        control_idle_noise=IdleQubitNoise(t1=np.inf, t2_echo=np.inf),
        target_idle_noise=IdleQubitNoise(t1=np.inf, t2_echo=np.inf),
        run_fidelity_simulation=False,
    )

    assert isinstance(analysis.fidelity_analysis, CrPulseFidelityAnalysis)
    assert analysis.fidelity_analysis.target_dephasing_fit is target_fit
    assert analysis.fidelity_analysis.control_dephasing_fit is None
    assert analysis.fit_status["forward_fit_success"] == {
        "control_t2_echo": False,
        "target_t2rho_echo": True,
    }
    skip_reasons = cast(
        Mapping[str, str | None],
        analysis.fit_status["forward_fit_skip_reason"],
    )
    control_skip_reason = skip_reasons["control_t2_echo"]
    assert control_skip_reason is not None
    assert "control_x180_duration" in control_skip_reason
    assert skip_reasons["target_t2rho_echo"] is None


def test_conservative_scenario_refits_dephasing_from_base_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upper-bound leakage should be applied before C/D dephasing is refitted."""
    control_measurements = _control_echo_measurements()
    times = control_measurements.times["control_t2_echo"]
    values = np.array([0.9, 0.8, 0.7])
    errors = np.full(3, 0.01)
    measurements = replace(
        control_measurements,
        protocols=("control_t2_echo", "target_t2rho_echo"),
        pauli_components={
            "control_t2_echo": ("X",),
            "target_t2rho_echo": ("Z",),
        },
        times={
            "control_t2_echo": times,
            "target_t2rho_echo": times,
        },
        pauli_expectations={
            "control_t2_echo": {"X": {"actual": values, "reference": values}},
            "target_t2rho_echo": {"Z": {"actual": values, "reference": values}},
        },
        pauli_standard_errors={
            "control_t2_echo": {"X": {"actual": errors, "reference": errors}},
            "target_t2rho_echo": {"Z": {"actual": errors, "reference": errors}},
        },
    )
    base_noise = CrOnNoise(0.0, 0.0, 0.0, 0.0, np.inf, 0.0, 0.0)
    upper_bound_noise = replace(base_noise, target_leakage_rate=2e-5)
    simulation = CrPulseFidelitySimulationResult(0.999, 0.995, 0.99, 0.004, {})
    analyzed_noise: list[CrOnNoise] = []

    monkeypatch.setattr(
        module,
        "_base_cr_on_noise_from_fits",
        lambda _fits: (base_noise, None),
    )

    def conservative_noise(nominal: CrOnNoise, _fits: object) -> CrOnNoise:
        assert nominal is base_noise
        return upper_bound_noise

    monkeypatch.setattr(
        module, "_conservative_outward_leakage_noise", conservative_noise
    )

    def fake_fidelity_analysis(
        _gate: object,
        _n_values: object,
        _control_idle: object,
        _target_idle: object,
        cr_on_noise: CrOnNoise,
        **_kwargs: object,
    ) -> CrPulseFidelityAnalysis:
        analyzed_noise.append(cr_on_noise)
        fitted_noise = replace(
            cr_on_noise,
            gamma_phi_control=1e-5 if cr_on_noise.target_leakage_rate == 0 else 2e-6,
            gamma_phi_rho_target=(
                2e-5 if cr_on_noise.target_leakage_rate == 0 else 3e-6
            ),
        )
        return CrPulseFidelityAnalysis(
            success=True,
            message="ok",
            cr_on_noise=fitted_noise,
            control_dephasing_fit=None,
            target_dephasing_fit=None,
            simulation=simulation,
        )

    monkeypatch.setattr(module, "analyze_cr_pulse_fidelity", fake_fidelity_analysis)

    analysis = analyze_cr_pulse_coherence(
        measurements,
        zx90_gate=ZX90GateTiming(20.0, True, 4.0),
        control_idle_noise=IdleQubitNoise(np.inf, np.inf),
        target_idle_noise=IdleQubitNoise(np.inf, np.inf),
        control_x180_duration=4.0,
        target_x180_duration=4.0,
    )

    assert analyzed_noise == [base_noise, upper_bound_noise]
    conservative = analysis.fidelity_analysis
    assert conservative is not None
    assert conservative.conservative_simulation_95 is not None
    metadata = conservative.conservative_simulation_95.model_metadata
    assert metadata["dephasing_rates_refitted"] is True
    assert metadata["refitted_gamma_phi_control"] == 2e-6
    assert metadata["refitted_gamma_phi_rho_target"] == 3e-6


def test_failed_conservative_scenario_does_not_claim_dephasing_was_refitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conservative metadata should report only a completed C/D refit."""
    base_noise = CrOnNoise(0.0, 0.0, 0.0, 0.0, np.inf, 0.0, 0.0)
    simulation = CrPulseFidelitySimulationResult(0.999, 0.995, 0.99, 0.004, {})
    call_count = 0

    monkeypatch.setattr(module, "_fit_measured_observables", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_base_cr_on_noise_from_fits",
        lambda _fits: (base_noise, None),
    )
    monkeypatch.setattr(
        module,
        "_conservative_outward_leakage_noise",
        lambda *_args: replace(base_noise, target_leakage_rate=2e-5),
    )

    def analyze_fidelity(*_args: object, **_kwargs: object) -> CrPulseFidelityAnalysis:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("conservative refit failed")
        return CrPulseFidelityAnalysis(
            success=True,
            message="ok",
            cr_on_noise=base_noise,
            control_dephasing_fit=None,
            target_dephasing_fit=None,
            simulation=simulation,
        )

    monkeypatch.setattr(module, "analyze_cr_pulse_fidelity", analyze_fidelity)

    analysis = analyze_cr_pulse_coherence(
        _all_protocol_measurements(),
        zx90_gate=ZX90GateTiming(20.0, True, 4.0),
        control_idle_noise=IdleQubitNoise(np.inf, np.inf),
        target_idle_noise=IdleQubitNoise(np.inf, np.inf),
        control_x180_duration=4.0,
        target_x180_duration=4.0,
        propagate_fidelity_uncertainty=False,
    )

    status = cast(
        Mapping[str, object],
        analysis.fit_status["conservative_outward_scenario"],
    )
    assert status["dephasing_rates_refitted"] is False
    assert status["simulation_available"] is False
    assert status["message"] == "conservative refit failed"


def test_three_level_fit_recovers_four_independent_adjacent_rates() -> None:
    """A ground/excited fit should recover independent GE and EF rates."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 4.2e-5, 7.0e-6)
    initial_ground = np.array([0.97, 0.02, 0.01])
    initial_excited = np.array([0.03, 0.94, 0.03])

    fit = fit_three_level_rate_model(
        times,
        _trajectory(times, initial_ground, rates),
        _trajectory(times, initial_excited, rates),
    )

    assert fit.success
    assert_allclose(
        [
            fit.gamma_ge_down,
            fit.gamma_ge_up,
            fit.gamma_ef_down,
            fit.gamma_ef_up,
        ],
        rates,
        rtol=2e-4,
        atol=1e-10,
    )
    assert fit.t1_eff == pytest.approx(1 / (rates[0] + rates[1]), rel=2e-4)
    assert_allclose(
        fit.fitted_ground,
        _trajectory(times, initial_ground, rates),
        atol=1e-7,
    )
    assert_allclose(
        fit.fitted_excited,
        _trajectory(times, initial_excited, rates),
        atol=1e-7,
    )


def test_weighted_rate_fit_preserves_absolute_error_scale() -> None:
    """Doubling population errors should double fitted rate errors."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 4.2e-5, 7.0e-6)
    ground = _trajectory(times, np.array([0.97, 0.02, 0.01]), rates)
    excited = _trajectory(times, np.array([0.03, 0.94, 0.03]), rates)

    fit = fit_three_level_rate_model(
        times,
        ground,
        excited,
        np.full(ground.shape, 0.01),
        np.full(excited.shape, 0.01),
        relative_uncertainty_threshold=100.0,
    )
    doubled_error_fit = fit_three_level_rate_model(
        times,
        ground,
        excited,
        np.full(ground.shape, 0.02),
        np.full(excited.shape, 0.02),
        relative_uncertainty_threshold=100.0,
    )

    assert fit.gamma_ge_down_error > 0.0
    assert doubled_error_fit.gamma_ge_down_error / fit.gamma_ge_down_error == (
        pytest.approx(2.0, rel=1e-6)
    )


def test_partially_weighted_rate_fit_scales_covariance_from_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One weighted trajectory should not make every residual error absolute."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 4.2e-5, 7.0e-6)
    ground = _trajectory(times, np.array([0.97, 0.02, 0.01]), rates)
    excited = _trajectory(times, np.array([0.03, 0.94, 0.03]), rates)
    original_estimate_covariance = module._estimate_covariance  # noqa: SLF001
    absolute_weight_flags: list[bool] = []

    def record_absolute_weights(
        jacobian: np.ndarray,
        cost: float,
        residual_count: int,
        parameter_count: int,
        *,
        absolute_weights: bool,
    ) -> np.ndarray:
        absolute_weight_flags.append(absolute_weights)
        return original_estimate_covariance(
            jacobian,
            cost,
            residual_count,
            parameter_count,
            absolute_weights=absolute_weights,
        )

    monkeypatch.setattr(module, "_estimate_covariance", record_absolute_weights)

    fit_three_level_rate_model(
        times,
        ground,
        excited,
        standard_errors_ground=np.full(ground.shape, 0.01),
        relative_uncertainty_threshold=100.0,
    )

    assert absolute_weight_flags
    assert not any(absolute_weight_flags)


def test_control_rate_fit_falls_back_when_leakage_is_unresolved() -> None:
    """A GE-only trajectory should refit with both EF rates fixed to zero."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 0.0, 0.0)
    ground = _trajectory(times, np.array([0.98, 0.02, 0.0]), rates)
    excited = _trajectory(times, np.array([0.03, 0.97, 0.0]), rates)

    fit = fit_three_level_rate_model(times, ground, excited)

    assert fit.success
    assert fit.leakage_model == "none"
    assert fit.gamma_ef_down == 0.0
    assert fit.gamma_ef_up == 0.0
    assert np.all(np.isnan(fit.covariance[2:, :]))
    assert np.all(np.isnan(fit.covariance[:, 2:]))
    assert fit.gamma_ge_down == pytest.approx(rates[0], rel=1e-4)
    assert fit.gamma_ge_up == pytest.approx(rates[1], rel=1e-4)


def test_control_rate_fit_rejects_nearly_null_ef_direction() -> None:
    """An unpopulated F level should not make a large seepage rate identifiable."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 5.0e-3, 1.0e-12)
    ground = _trajectory(times, np.array([0.98, 0.02, 0.0]), rates)
    excited = _trajectory(times, np.array([0.03, 0.97, 0.0]), rates)
    errors = np.full(ground.shape, 0.001)

    fit = fit_three_level_rate_model(
        times,
        ground,
        excited,
        errors,
        errors,
    )

    assert fit.success
    assert fit.leakage_model == "none"
    assert fit.gamma_ef_up == 0.0
    assert fit.gamma_ef_down == 0.0


def test_control_leakage_upper_bound_improves_with_smaller_errors() -> None:
    """Absolute-error sensitivity should tighten the unresolved outward limit."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 0.0, 0.0)
    ground = _trajectory(times, np.array([0.98, 0.02, 0.0]), rates)
    excited = _trajectory(times, np.array([0.03, 0.97, 0.0]), rates)

    loose = fit_three_level_rate_model(
        times,
        ground,
        excited,
        np.full_like(ground, 0.01),
        np.full_like(excited, 0.01),
    )
    precise = fit_three_level_rate_model(
        times,
        ground,
        excited,
        np.full_like(ground, 0.003),
        np.full_like(excited, 0.003),
    )

    assert loose.gamma_ef_up == precise.gamma_ef_up == 0.0
    assert 0 < precise.gamma_ef_up_upper_95 < loose.gamma_ef_up_upper_95


def test_control_rate_fit_rejects_zero_ef_model_for_unresolved_fast_dynamics() -> None:
    """Visible saturated EF dynamics should fail when their rate scale is unresolved."""
    times = np.array([0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000.0])
    rates = (0.0, 0.0, 2.0e-3, 1.0e-3)
    ground = _trajectory(times, np.array([0.98, 0.02, 0.0]), rates)
    excited = _trajectory(times, np.array([0.03, 0.97, 0.0]), rates)
    errors = np.full(ground.shape, 0.001)

    fit = fit_three_level_rate_model(
        times,
        ground,
        excited,
        errors,
        errors,
    )

    assert not fit.success
    assert fit.leakage_model is None
    assert "E-F dynamics are visible" in fit.message
    assert np.isnan(fit.gamma_ge_down)
    assert np.isnan(fit.t1_eff)
    assert np.all(np.isnan(fit.covariance))
    assert np.all(np.isnan(fit.fitted_ground))


def test_no_target_leakage_uses_physical_actual_and_exponential_reference() -> None:
    """Actual A/B data use the physical model even when leakage is fixed to zero."""
    analysis = analyze_cr_pulse_coherence(_gef_measurements_without_target_leakage())

    leakage_fits = analysis.fits["target_leakage"]
    t1rho_fits = analysis.fits["target_t1rho"]
    assert isinstance(leakage_fits, dict)
    assert isinstance(t1rho_fits, dict)
    for protocol in ("control_ground", "control_excited"):
        for kind in ("actual", "reference"):
            assert leakage_fits[protocol][kind].model == "none"
        assert t1rho_fits[protocol]["actual"].model == "control_transition_forward"
        assert t1rho_fits[protocol]["reference"].model == "exponential"
        assert np.isfinite(leakage_fits[protocol]["actual"].leakage_rate_upper_95)
    control_fits = cast(
        Mapping[str, module.ThreeLevelRateFit], analysis.fits["control_rate_model"]
    )
    assert np.isfinite(control_fits["actual"].gamma_ef_up_upper_95)
    upper_bounds = cast(
        Mapping[str, float],
        analysis.fit_status["unresolved_outward_rate_upper_bounds_95"],
    )
    assert set(upper_bounds) == {
        "gamma_control_e_to_f",
        "target_leakage_rate_ground",
        "target_leakage_rate_excited",
    }


def test_unresolved_outward_upper_scenario_lowers_dissipative_fidelity() -> None:
    """The sensitivity scenario should increase only unresolved outward rates."""
    analysis = analyze_cr_pulse_coherence(_gef_measurements_without_target_leakage())
    nominal_noise = cast(CrOnNoise, analysis.base_cr_on_noise)
    conservative_noise = module._conservative_outward_leakage_noise(  # noqa: SLF001
        nominal_noise,
        analysis.fits,
    )
    assert conservative_noise is not None
    assert conservative_noise.gamma_control_e_to_f > 0
    assert conservative_noise.target_leakage_rate > 0
    assert conservative_noise.gamma_control_f_to_e == nominal_noise.gamma_control_f_to_e
    assert conservative_noise.target_seepage_rate == nominal_noise.target_seepage_rate

    timing = ZX90GateTiming(
        cr_lobe_duration=20.0,
        echo=True,
        control_pi_duration=4.0,
    )
    idle = IdleQubitNoise(t1=np.inf, t2_echo=np.inf)
    nominal = module.simulate_cr_pulse_fidelity(
        timing,
        idle,
        idle,
        nominal_noise,
    )
    conservative = module.simulate_cr_pulse_fidelity(
        timing,
        idle,
        idle,
        conservative_noise,
    )
    assert (
        conservative.cr_on_dissipative_limited_fidelity
        <= nominal.cr_on_dissipative_limited_fidelity
    )


def test_control_rate_fit_falls_back_when_only_seepage_is_unresolved() -> None:
    """An outward-only trajectory should refit with F-to-E fixed to zero."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 0.0, 7.0e-6)
    ground = _trajectory(times, np.array([0.98, 0.02, 0.0]), rates)
    excited = _trajectory(times, np.array([0.03, 0.97, 0.0]), rates)

    fit = fit_three_level_rate_model(times, ground, excited)

    assert fit.success
    assert fit.leakage_model == "outward_only"
    assert fit.gamma_ef_down == 0.0
    assert np.isnan(fit.gamma_ef_down_error)
    assert fit.gamma_ef_up == pytest.approx(rates[3], rel=1e-4)


def test_control_rate_fit_keeps_resolved_seepage_when_leakage_is_zero() -> None:
    """A seepage-only trajectory should not discard its resolved F-to-E rate."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 4.2e-5, 0.0)
    ground = _trajectory(times, np.array([0.88, 0.02, 0.10]), rates)
    excited = _trajectory(times, np.array([0.03, 0.82, 0.15]), rates)

    fit = fit_three_level_rate_model(times, ground, excited)

    assert fit.success
    assert fit.leakage_model == "inward_only"
    assert fit.gamma_ef_up == 0.0
    assert np.isnan(fit.gamma_ef_up_error)
    assert fit.gamma_ef_down == pytest.approx(rates[2], rel=1e-4)


def test_safe_control_rate_fit_preserves_results_on_numerical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A numerical rate-fit failure should produce a structured failed fit."""
    times = np.array([0.0, 1.0, 2.0])
    ground = np.tile(np.array([0.98, 0.02, 0.0]), (3, 1))
    excited = np.tile(np.array([0.03, 0.97, 0.0]), (3, 1))

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise np.linalg.LinAlgError("singular fit")

    monkeypatch.setattr(module, "fit_three_level_rate_model", fail_fit)

    fit = module._safe_fit_three_level_rate_model(  # noqa: SLF001
        times,
        ground,
        excited,
        None,
        None,
        1.0,
    )

    assert not fit.success
    assert "singular fit" in fit.message
    assert_allclose(fit.initial_ground, ground[0])
    assert_allclose(fit.initial_excited, excited[0])
    assert np.all(np.isnan(fit.fitted_ground))


def test_exponential_fit_recovers_offset_decay() -> None:
    """An exponential fit should recover amplitude, offset, and decay time."""
    times = np.array([0, 500, 1000, 2000, 3500, 5500, 8000.0])
    values = -0.12 + 0.91 * np.exp(-times / 3200.0)
    standard_errors = np.full(times.shape, 0.01)

    fit = fit_exponential_decay(times, values, standard_errors)

    assert fit.success
    assert fit.amplitude == pytest.approx(0.91, rel=1e-5)
    assert fit.offset == pytest.approx(-0.12, rel=1e-5)
    assert fit.tau == pytest.approx(3200.0, rel=1e-5)
    assert_allclose(fit.fitted_values, values, atol=1e-7)


def test_weighted_exponential_fit_preserves_absolute_error_scale() -> None:
    """Doubling known data errors should double fitted parameter errors."""
    times = np.array([0, 500, 1000, 2000, 3500, 5500, 8000.0])
    values = -0.12 + 0.91 * np.exp(-times / 3200.0)

    fit = fit_exponential_decay(times, values, np.full(times.shape, 0.01))
    doubled_error_fit = fit_exponential_decay(
        times,
        values,
        np.full(times.shape, 0.02),
    )

    assert fit.tau_error > 0.0
    assert doubled_error_fit.tau_error / fit.tau_error == pytest.approx(2.0, rel=1e-6)


def test_target_t1rho_rejects_decay_below_measurement_resolution() -> None:
    """A nominally finite but unobservably slow decay must remain unresolved."""
    times = np.linspace(0.0, 8000.0, 21)
    values = 0.8 * np.exp(-times / 1e9)

    fit = fit_target_t1rho(times, values, np.full(times.shape, 0.01))

    assert not fit.success
    assert "insufficient dynamic range" in fit.message
    assert np.isnan(fit.t1rho)
    assert np.isnan(fit.t1rho_error)
    assert np.all(np.isnan(fit.covariance))
    assert np.all(np.isnan(fit.fitted_values))


def test_exponential_fit_rejects_flat_curve() -> None:
    """A flat curve must not report an arbitrary finite decay time."""
    times = np.linspace(0.0, 8000.0, 21)

    fit = fit_exponential_decay(
        times,
        np.full(times.shape, 0.8),
        np.full(times.shape, 0.01),
    )

    assert not fit.success
    assert "insufficient dynamic range" in fit.message
    assert np.isnan(fit.tau)
    assert np.isnan(fit.tau_error)
    assert np.all(np.isnan(fit.covariance))
    assert np.all(np.isnan(fit.fitted_values))


@pytest.mark.parametrize(
    ("times", "values", "message"),
    [
        ([1, 2, 3], [0.0, 0.1, 0.2], "start at zero"),
        ([0, 2, 1], [0.0, 0.1, 0.2], "strictly increasing"),
    ],
)
def test_exponential_fit_validates_time_axis(
    times: list[int],
    values: list[float],
    message: str,
) -> None:
    """Decay fitting should reject a missing or unordered zero-time point."""
    with pytest.raises(ValueError, match=message):
        fit_exponential_decay(times, values)


def test_exponential_fit_rejects_scalar_values_cleanly() -> None:
    """A scalar value input should raise ValueError instead of leaking IndexError."""
    with pytest.raises(ValueError, match="values must be at least one-dimensional"):
        fit_exponential_decay([0.0, 1.0, 2.0], 1.0)


def test_exponential_fit_rejects_negative_standard_errors() -> None:
    """Finite negative standard errors should be rejected instead of ignored."""
    with pytest.raises(ValueError, match="standard_errors must be nonnegative"):
        fit_exponential_decay(
            [0.0, 1.0, 2.0],
            [1.0, 0.5, 0.25],
            [0.1, -0.1, 0.1],
        )


def test_effective_rate_error_uses_joint_covariance() -> None:
    """Effective target-rate summaries should retain joint-fit correlations."""
    value, error = module._mean_with_error(  # noqa: SLF001
        2.0,
        99.0,
        4.0,
        99.0,
        np.array([[4.0, 2.0], [2.0, 9.0]]),
    )
    fixed_zero_value, fixed_zero_error = module._mean_with_error(  # noqa: SLF001
        2.0,
        2.0,
        0.0,
        np.nan,
        np.array([[4.0, np.nan], [np.nan, np.nan]]),
    )

    assert value == 3.0
    assert error == pytest.approx(np.sqrt(17.0) / 2)
    assert fixed_zero_value == 1.0
    assert fixed_zero_error == 1.0


def _dephasing_fit(gamma_phi: float, variance: float) -> CrOnDephasingFit:
    """Return a minimal successful dephasing fit for covariance tests."""
    values = np.ones(3)
    return CrOnDephasingFit(
        success=True,
        message="ok",
        gamma_phi=gamma_phi,
        gamma_phi_error=np.sqrt(variance),
        covariance=np.array([[variance]]),
        amplitude=1.0,
        offset=0.0,
        fitted_values=values,
        curve_n_values=np.arange(3),
        curve_values=values,
        r_squared=1.0,
    )


def _fidelity_covariance_inputs() -> tuple[
    dict[str, object],
    CrPulseFidelityAnalysis,
    np.ndarray,
    np.ndarray,
]:
    """Build successful full-model fits with correlated rate covariance."""
    control_factor = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 1.1, 0.0, 0.0],
            [0.3, -0.1, 0.8, 0.0],
            [0.1, 0.2, -0.2, 0.9],
        ]
    )
    control_covariance = control_factor @ control_factor.T * 1e-12
    control_fit = replace(
        module._failed_rate_fit(  # noqa: SLF001
            "placeholder",
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            3,
        ),
        success=True,
        message="ok",
        leakage_model="full",
        gamma_ge_down=2e-5,
        gamma_ge_up=1e-5,
        gamma_ef_down=4e-6,
        gamma_ef_up=3e-6,
        covariance=control_covariance,
    )

    target_factor = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 1.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.8, 0.0, 0.0, 0.0],
            [0.0, -0.1, 0.2, 0.9, 0.0, 0.0],
            [0.3, 0.0, -0.1, 0.2, 0.7, 0.0],
            [0.1, 0.1, 0.0, -0.2, 0.3, 0.8],
        ]
    )
    target_rate_covariance = target_factor @ target_factor.T * 1e-12
    target_covariance = np.full((10, 10), np.nan)
    target_covariance[:6, :6] = target_rate_covariance
    target_fit = replace(
        module._failed_target_ab_forward_fit(  # noqa: SLF001
            "placeholder",
            np.arange(3),
        ),
        success=True,
        message="ok",
        leakage_model="full",
        target_t1rho_ground=4_000.0,
        target_t1rho_excited=6_000.0,
        leakage_rate_ground=3e-6,
        seepage_rate_ground=4e-6,
        leakage_rate_excited=5e-6,
        seepage_rate_excited=6e-6,
        covariance=target_covariance,
    )
    noise = CrOnNoise(
        gamma_control_g_to_e=control_fit.gamma_ge_up,
        gamma_control_e_to_g=control_fit.gamma_ge_down,
        gamma_control_e_to_f=control_fit.gamma_ef_up,
        gamma_control_f_to_e=control_fit.gamma_ef_down,
        target_t1rho=4_800.0,
        target_leakage_rate=4e-6,
        target_seepage_rate=5e-6,
        gamma_phi_control=7e-6,
        gamma_phi_rho_target=8e-6,
    )
    fidelity_analysis = CrPulseFidelityAnalysis(
        success=True,
        message="ok",
        cr_on_noise=noise,
        control_dephasing_fit=_dephasing_fit(7e-6, 2e-13),
        target_dephasing_fit=_dephasing_fit(8e-6, 3e-13),
        simulation=None,
    )
    return (
        {
            "control_rate_model": {"actual": control_fit},
            "target_ab_forward": target_fit,
        },
        fidelity_analysis,
        control_covariance,
        target_rate_covariance,
    )


def test_fidelity_control_covariance_is_reordered_to_noise_parameter_basis() -> None:
    """Control covariance should retain correlations under the explicit reorder."""
    fits, fidelity_analysis, source_covariance, _ = _fidelity_covariance_inputs()

    covariance, _ = module._fidelity_parameter_covariance_from_fits(  # noqa: SLF001
        fits,
        fidelity_analysis,
    )

    indices = (1, 0, 3, 2)
    assert_allclose(covariance[:4, :4], source_covariance[np.ix_(indices, indices)])


def test_fidelity_target_effective_covariance_uses_full_linear_transform() -> None:
    """Effective target rates should retain all A/B rate cross-correlations."""
    fits, fidelity_analysis, _, source_covariance = _fidelity_covariance_inputs()
    transform = np.array(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
        ]
    )

    covariance, _ = module._fidelity_parameter_covariance_from_fits(  # noqa: SLF001
        fits,
        fidelity_analysis,
    )

    assert_allclose(
        covariance[4:7, 4:7],
        transform @ source_covariance @ transform.T,
    )


def test_fixed_zero_rates_have_zero_conditional_covariance() -> None:
    """Reduced-model inactive rates should not contribute Gaussian variance."""
    fits, fidelity_analysis, _, _ = _fidelity_covariance_inputs()
    control_fit = cast(
        Mapping[str, module.ThreeLevelRateFit], fits["control_rate_model"]
    )["actual"]
    control_covariance = np.full((4, 4), np.nan)
    control_covariance[:2, :2] = np.array([[2e-12, 1e-13], [1e-13, 3e-12]])
    target_fit = cast(module.TargetABForwardFit, fits["target_ab_forward"])
    target_covariance = np.full((10, 10), np.nan)
    target_covariance[:2, :2] = np.array([[4e-12, 2e-13], [2e-13, 5e-12]])
    fits = {
        "control_rate_model": {
            "actual": replace(
                control_fit,
                leakage_model="none",
                gamma_ef_down=0.0,
                gamma_ef_up=0.0,
                covariance=control_covariance,
            )
        },
        "target_ab_forward": replace(
            target_fit,
            leakage_model="none",
            leakage_rate_ground=0.0,
            seepage_rate_ground=0.0,
            leakage_rate_excited=0.0,
            seepage_rate_excited=0.0,
            covariance=target_covariance,
        ),
    }
    noise = replace(
        cast(CrOnNoise, fidelity_analysis.cr_on_noise),
        gamma_control_e_to_f=0.0,
        gamma_control_f_to_e=0.0,
        target_leakage_rate=0.0,
        target_seepage_rate=0.0,
    )

    covariance, fixed = module._fidelity_parameter_covariance_from_fits(  # noqa: SLF001
        fits,
        replace(fidelity_analysis, cr_on_noise=noise),
    )

    assert_allclose(covariance[[2, 3, 5, 6], :], 0.0)
    assert {
        "gamma_control_e_to_f",
        "gamma_control_f_to_e",
        "gamma_target_leakage_effective",
        "gamma_target_seepage_effective",
    }.issubset(fixed)


def test_exact_zero_active_rate_is_not_reported_as_fixed() -> None:
    """Selected-model membership, rather than a zero estimate, defines fixed rates."""
    fits, fidelity_analysis, _, _ = _fidelity_covariance_inputs()
    control_fit = cast(
        Mapping[str, module.ThreeLevelRateFit], fits["control_rate_model"]
    )["actual"]
    covariance = control_fit.covariance.copy()
    covariance[3, :] = 0.0
    covariance[:, 3] = 0.0
    fits = {
        **fits,
        "control_rate_model": {
            "actual": replace(
                control_fit,
                gamma_ef_up=0.0,
                covariance=covariance,
            )
        },
    }
    noise = replace(
        cast(CrOnNoise, fidelity_analysis.cr_on_noise),
        gamma_control_e_to_f=0.0,
    )

    _, fixed = module._fidelity_parameter_covariance_from_fits(  # noqa: SLF001
        fits,
        replace(fidelity_analysis, cr_on_noise=noise),
    )

    assert "gamma_control_e_to_f" not in fixed


def test_active_nonfinite_rate_covariance_makes_uncertainty_unavailable() -> None:
    """A missing covariance for an active rate must not be treated as exact zero."""
    fits, fidelity_analysis, _, _ = _fidelity_covariance_inputs()
    control_fit = cast(
        Mapping[str, module.ThreeLevelRateFit], fits["control_rate_model"]
    )["actual"]
    invalid_covariance = control_fit.covariance.copy()
    invalid_covariance[0, :] = np.nan
    invalid_covariance[:, 0] = np.nan
    fits = {
        **fits,
        "control_rate_model": {
            "actual": replace(control_fit, covariance=invalid_covariance)
        },
    }

    with pytest.raises(ValueError, match="Active control-rate covariance"):
        module._fidelity_parameter_covariance_from_fits(  # noqa: SLF001
            fits,
            fidelity_analysis,
        )


def test_failed_fidelity_uncertainty_keeps_idle_variance_exactly_zero() -> None:
    """Unavailable CR-rate covariance does not imply uncertain idle T1/T2 inputs."""
    uncertainty = module._failed_fidelity_linear_uncertainty(  # noqa: SLF001
        "active covariance unavailable"
    )

    assert not uncertainty.success
    assert uncertainty.idle_coherence_limited_fidelity_standard_error == 0.0
    assert_allclose(uncertainty.output_covariance[0, :], 0.0)
    assert_allclose(uncertainty.output_covariance[:, 0], 0.0)
    assert_allclose(uncertainty.jacobian[0, :], 0.0)


def _linear_fidelity_simulation(
    _gate: object,
    _control_idle: object,
    _target_idle: object,
    noise: CrOnNoise,
) -> CrPulseFidelitySimulationResult:
    """Return a deterministic linear stand-in for Jacobian tests."""
    parameters = module._cr_on_noise_parameter_vector(noise)  # noqa: SLF001
    return CrPulseFidelitySimulationResult(
        idle_coherence_limited_fidelity=0.99,
        cr_on_coherence_limited_fidelity=(0.8 + 2 * parameters[0] + 3 * parameters[1]),
        cr_on_dissipative_limited_fidelity=(0.7 - 4 * parameters[0] + parameters[1]),
        average_leakage=0.1 + 6 * parameters[0] - 2 * parameters[1],
        model_metadata={},
    )


def _delta_method_noise(*, gamma_control_g_to_e: float = 0.1) -> CrOnNoise:
    """Return a valid noise model for local-propagation tests."""
    return CrOnNoise(
        gamma_control_g_to_e=gamma_control_g_to_e,
        gamma_control_e_to_g=0.1,
        gamma_control_e_to_f=0.0,
        gamma_control_f_to_e=0.0,
        target_t1rho=np.inf,
        target_leakage_rate=0.0,
        target_seepage_rate=0.0,
    )


def test_zero_parameter_covariance_gives_zero_fidelity_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact parameter vector should produce exact simulated outputs."""
    monkeypatch.setattr(
        module, "simulate_cr_pulse_fidelity", _linear_fidelity_simulation
    )

    uncertainty = module.propagate_cr_pulse_fidelity_uncertainty(
        ZX90GateTiming(20.0, False),
        IdleQubitNoise(np.inf, np.inf),
        IdleQubitNoise(np.inf, np.inf),
        _delta_method_noise(),
        np.zeros((9, 9)),
    )

    assert uncertainty.success
    assert_allclose(uncertainty.output_covariance, 0.0)
    assert uncertainty.cr_on_coherence_limited_fidelity_standard_error == 0.0
    assert set(uncertainty.finite_difference_schemes) == {"skipped_zero_variance"}
    assert uncertainty.parameter_names == (
        "gamma_control_g_to_e",
        "gamma_control_e_to_g",
        "gamma_control_e_to_f",
        "gamma_control_f_to_e",
        "gamma_target_1rho_effective",
        "gamma_target_leakage_effective",
        "gamma_target_seepage_effective",
        "gamma_phi_control",
        "gamma_phi_rho_target",
    )
    assert uncertainty.metadata["parameter_order"] == uncertainty.parameter_names
    assert (
        uncertainty.metadata["covariance_approximation"]
        == "supplied_parameter_covariance"
    )
    assert not uncertainty.metadata["ignored_fit_stage_cross_covariance"]
    assert not uncertainty.idle_noise_uncertainty_propagated


def test_single_parameter_fidelity_uncertainty_matches_derivative_times_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One varying rate should obey sigma_y = abs(dy/dtheta) sigma_theta."""
    monkeypatch.setattr(
        module, "simulate_cr_pulse_fidelity", _linear_fidelity_simulation
    )
    covariance = np.zeros((9, 9))
    covariance[0, 0] = 0.02**2

    uncertainty = module.propagate_cr_pulse_fidelity_uncertainty(
        ZX90GateTiming(20.0, False),
        IdleQubitNoise(np.inf, np.inf),
        IdleQubitNoise(np.inf, np.inf),
        _delta_method_noise(),
        covariance,
    )

    assert uncertainty.cr_on_coherence_limited_fidelity_standard_error == (
        pytest.approx(abs(2.0) * 0.02)
    )
    assert uncertainty.cr_on_dissipative_limited_fidelity_standard_error == (
        pytest.approx(abs(-4.0) * 0.02)
    )


def test_correlated_parameter_covariance_contributes_cross_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """J Sigma J.T should retain covariance between simultaneously fitted rates."""
    monkeypatch.setattr(
        module, "simulate_cr_pulse_fidelity", _linear_fidelity_simulation
    )
    covariance = np.zeros((9, 9))
    covariance[:2, :2] = np.array([[0.02**2, 3e-4], [3e-4, 0.03**2]])

    uncertainty = module.propagate_cr_pulse_fidelity_uncertainty(
        ZX90GateTiming(20.0, False),
        IdleQubitNoise(np.inf, np.inf),
        IdleQubitNoise(np.inf, np.inf),
        _delta_method_noise(),
        covariance,
    )

    gradient = np.array([2.0, 3.0])
    expected_variance = float(gradient @ covariance[:2, :2] @ gradient)
    assert uncertainty.output_covariance[1, 1] == pytest.approx(expected_variance)


def test_rate_boundary_uses_forward_difference_without_negative_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero rate with uncertainty should use only nonnegative perturbations."""
    evaluated_rates: list[float] = []

    def record_simulation(
        gate: object,
        control_idle: object,
        target_idle: object,
        noise: CrOnNoise,
    ) -> CrPulseFidelitySimulationResult:
        evaluated_rates.append(noise.gamma_control_g_to_e)
        return _linear_fidelity_simulation(gate, control_idle, target_idle, noise)

    monkeypatch.setattr(module, "simulate_cr_pulse_fidelity", record_simulation)
    covariance = np.zeros((9, 9))
    covariance[0, 0] = 0.02**2

    uncertainty = module.propagate_cr_pulse_fidelity_uncertainty(
        ZX90GateTiming(20.0, False),
        IdleQubitNoise(np.inf, np.inf),
        IdleQubitNoise(np.inf, np.inf),
        _delta_method_noise(gamma_control_g_to_e=0.0),
        covariance,
    )

    assert min(evaluated_rates) >= 0.0
    assert uncertainty.finite_difference_schemes[0] == "forward"
    assert uncertainty.jacobian[1, 0] == pytest.approx(2.0)


def test_fidelity_finite_difference_is_stable_across_step_factors() -> None:
    """A representative physical derivative should be stable for h/2, h, and 2h."""
    timing = ZX90GateTiming(
        cr_lobe_duration=20.0,
        echo=True,
        control_pi_duration=4.0,
    )
    idle = IdleQubitNoise(t1=50_000.0, t2_echo=40_000.0)
    noise = replace(
        _delta_method_noise(gamma_control_g_to_e=1e-5),
        gamma_phi_control=1e-3,
    )
    covariance = np.zeros((9, 9))
    covariance[7, 7] = 1e-12
    derivatives = []
    for relative_step in (0.5e-4, 1e-4, 2e-4):
        uncertainty = module.propagate_cr_pulse_fidelity_uncertainty(
            timing,
            idle,
            idle,
            noise,
            covariance,
            relative_difference_step=relative_step,
        )
        derivatives.append(uncertainty.jacobian[:, 7])

    assert_allclose(derivatives[0], derivatives[1], rtol=2e-5, atol=1e-10)
    assert_allclose(derivatives[2], derivatives[1], rtol=2e-5, atol=1e-10)
