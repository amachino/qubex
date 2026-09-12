"""Tests for offline CR-pulse health analysis."""

# ruff: noqa: SLF001

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import qubex.contrib.experiment.cr_pulse_health_check as health


def test_target_x_is_normalized_within_computational_subspace() -> None:
    """Target X should exclude measured F-state population from normalization."""
    populations = np.array(
        [
            [0.10, 0.80, 0.10],
            [0.18, 0.72, 0.10],
            [0.27, 0.63, 0.10],
        ]
    )
    zeros = np.zeros_like(populations)
    times = np.array([0.0, 1.0, 2.0])

    measurements = health.CrPulseHealthMeasurements(
        control_qubit="Q0",
        target_qubit="Q1",
        n_values=(0, 1, 2),
        total_times={
            protocol: times.copy()
            for protocol in (
                "control_ground",
                "control_excited",
                "control_t2_echo",
                "target_t2rho_echo",
            )
        },
        cr_active_times={
            protocol: times.copy()
            for protocol in (
                "control_ground",
                "control_excited",
                "control_t2_echo",
                "target_t2rho_echo",
            )
        },
        populations={
            protocol: {
                kind: {"Q0": populations.copy(), "Q1": populations.copy()}
                for kind in ("actual", "reference")
            }
            for protocol in ("control_ground", "control_excited")
        },
        population_standard_errors={
            protocol: {
                kind: {"Q0": zeros.copy(), "Q1": zeros.copy()}
                for kind in ("actual", "reference")
            }
            for protocol in ("control_ground", "control_excited")
        },
        target_x_standard_errors={
            protocol: {kind: np.zeros(3) for kind in ("actual", "reference")}
            for protocol in ("control_ground", "control_excited")
        },
        pauli_expectations={},
        pauli_standard_errors={},
        diagnostic_components_measured=False,
    )

    expected = (populations[:, 1] - populations[:, 0]) / (
        populations[:, 0] + populations[:, 1]
    )
    np.testing.assert_allclose(
        measurements.target_x["control_ground"]["actual"], expected
    )


def test_decay_health_keeps_flat_trace_at_nominal_zero_rate() -> None:
    """A flat decay trace should remain a stable nominal-zero-rate result."""
    times = np.linspace(0.0, 1000.0, 6)
    values = np.full(times.size, 0.83)
    errors = np.full(times.size, 1.0e-3)

    fit = health._fit_decay_health(
        times,
        values,
        errors,
        minimum_change=0.01,
        change_sigma_threshold=3.0,
        robust_loss="soft_l1",
    )

    assert fit.success
    assert fit.quality == "stable"
    assert fit.rate == 0.0
    assert np.isnan(fit.rate_standard_error)
    assert np.isinf(fit.time_constant)


def test_decay_health_recovers_exponential_rate() -> None:
    """A resolved exponential trace should recover its decay rate."""
    times = np.linspace(0.0, 6000.0, 10)
    expected_rate = 8.0e-5
    values = 0.15 + 0.8 * np.exp(-expected_rate * times)
    errors = np.full(times.size, 2.0e-4)

    fit = health._fit_decay_health(
        times,
        values,
        errors,
        minimum_change=0.01,
        change_sigma_threshold=3.0,
        robust_loss="soft_l1",
    )

    assert fit.success
    assert fit.quality in {"good", "fair"}
    assert fit.rate == pytest.approx(expected_rate, rel=1.0e-3)
    assert fit.time_constant == pytest.approx(1.0 / expected_rate, rel=1.0e-3)


@pytest.mark.parametrize(
    ("mode", "values", "expected_outward", "expected_inward"),
    [
        (
            "outward",
            lambda t, gamma: 1.0 - 0.98 * np.exp(-gamma * t),
            6.0e-5,
            0.0,
        ),
        (
            "inward",
            lambda t, gamma: 0.12 * np.exp(-gamma * t),
            0.0,
            6.0e-5,
        ),
    ],
)
def test_exchange_health_selects_directional_model(
    mode: str,
    values,
    expected_outward: float,
    expected_inward: float,
) -> None:
    """A one-way F-population trace should use its directional reduced model."""
    times = np.linspace(0.0, 6000.0, 10)
    gamma = 6.0e-5
    errors = np.full(times.size, 2.0e-4)

    fit = health._fit_exchange_health(
        times,
        values(times, gamma),
        errors,
        minimum_change=0.003,
        change_sigma_threshold=3.0,
        robust_loss="soft_l1",
    )

    assert fit.success
    assert fit.model == f"{mode}_only"
    assert fit.outward_rate == pytest.approx(expected_outward, rel=1.0e-3, abs=1e-12)
    assert fit.inward_rate == pytest.approx(expected_inward, rel=1.0e-3, abs=1e-12)


def test_exchange_health_recovers_resolved_two_way_exchange() -> None:
    """A saturated F-population trace should resolve outward and inward rates."""
    times = np.linspace(0.0, 20_000.0, 12)
    outward_rate = 6.0e-5
    inward_rate = 1.4e-4
    total_rate = outward_rate + inward_rate
    equilibrium = outward_rate / total_rate
    initial_population = 0.01
    values = equilibrium + (initial_population - equilibrium) * np.exp(
        -total_rate * times
    )

    fit = health._fit_exchange_health(
        times,
        values,
        np.full(times.size, 2.0e-4),
        minimum_change=0.003,
        change_sigma_threshold=3.0,
        robust_loss="soft_l1",
    )

    assert fit.success
    assert fit.model == "two_way"
    assert fit.outward_rate == pytest.approx(outward_rate, rel=1.0e-3)
    assert fit.inward_rate == pytest.approx(inward_rate, rel=1.0e-3)


def test_control_health_recovers_resolved_two_way_ef_exchange() -> None:
    """Joint control populations should resolve visible E-F exchange in both directions."""
    times = np.linspace(0.0, 20_000.0, 12)
    expected_rates = {
        "gamma_e_to_g": 5.0e-5,
        "gamma_g_to_e": 3.0e-5,
        "gamma_f_to_e": 1.4e-4,
        "gamma_e_to_f": 6.0e-5,
    }
    ground = health._population_trajectory(
        times, np.array([1.0, 0.0, 0.0]), expected_rates
    )
    excited = health._population_trajectory(
        times, np.array([0.0, 1.0, 0.0]), expected_rates
    )
    errors = np.full_like(ground, 2.0e-4)

    fit = health._fit_control_rates_health(
        times,
        times,
        ground,
        excited,
        errors,
        errors,
        population_minimum_change=0.001,
        leakage_minimum_change=0.001,
        change_sigma_threshold=2.0,
        robust_loss="soft_l1",
    )

    assert fit.success
    assert set(fit.active_rates) == set(expected_rates)
    for name, expected in expected_rates.items():
        assert fit.rates[name] == pytest.approx(expected, rel=0.01)


def test_control_health_uses_singular_full_population_covariance() -> None:
    """Control-rate fitting should whiten the resolved simplex covariance modes."""
    times = np.linspace(0.0, 20_000.0, 12)
    expected_rates = {
        "gamma_e_to_g": 5.0e-5,
        "gamma_g_to_e": 3.0e-5,
        "gamma_f_to_e": 1.4e-4,
        "gamma_e_to_f": 6.0e-5,
    }
    ground = health._population_trajectory(
        times, np.array([1.0, 0.0, 0.0]), expected_rates
    )
    excited = health._population_trajectory(
        times, np.array([0.0, 1.0, 0.0]), expected_rates
    )
    simplex_covariance = (2.0e-4) ** 2 * (np.eye(3) - np.ones((3, 3)) / 3.0)
    covariances = np.tile(simplex_covariance, (times.size, 1, 1))
    errors = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))

    fit = health._fit_control_rates_health(
        times,
        times,
        ground,
        excited,
        errors,
        errors,
        covariances,
        covariances,
        population_minimum_change=0.001,
        leakage_minimum_change=0.001,
        change_sigma_threshold=2.0,
        robust_loss="linear",
    )

    assert fit.success
    assert fit.population_weighting == "full_covariance"
    for name, expected in expected_rates.items():
        assert fit.rates[name] == pytest.approx(expected, rel=0.01)
        assert np.isfinite(fit.rate_standard_errors[name])


def test_control_health_falls_back_from_malformed_population_covariance() -> None:
    """Malformed full covariance should fall back to component standard errors."""
    times = np.linspace(0.0, 6000.0, 10)
    expected_rates = {
        "gamma_e_to_g": 5.0e-5,
        "gamma_g_to_e": 3.0e-5,
        "gamma_f_to_e": 0.0,
        "gamma_e_to_f": 2.0e-5,
    }
    ground = health._population_trajectory(
        times, np.array([1.0, 0.0, 0.0]), expected_rates
    )
    excited = health._population_trajectory(
        times, np.array([0.0, 1.0, 0.0]), expected_rates
    )
    errors = np.full_like(ground, 2.0e-4)
    malformed = np.tile(np.eye(3), (times.size, 1, 1))
    malformed[1, 0, 0] = -1.0

    fit = health._fit_control_rates_health(
        times,
        times,
        ground,
        excited,
        errors,
        errors,
        malformed,
        malformed,
        population_minimum_change=0.001,
        leakage_minimum_change=0.001,
        change_sigma_threshold=2.0,
        robust_loss="linear",
    )

    assert fit.success
    assert fit.population_weighting == "component_se_diagonal"


def test_target_x_error_uses_full_population_covariance() -> None:
    """Target-X uncertainty should retain the analytic G/E covariance term."""
    population = np.array([0.4, 0.5, 0.1])
    covariance = 1.0e-6 * np.array(
        [
            [4.0, -3.0, -1.0],
            [-3.0, 5.0, -2.0],
            [-1.0, -2.0, 3.0],
        ]
    )
    fit = SimpleNamespace(
        population=population,
        population_covariance=covariance,
        population_standard_error=np.sqrt(np.diag(covariance)),
    )
    denominator = population[0] + population[1]
    gradient = np.array(
        [
            -2.0 * population[1] / denominator**2,
            2.0 * population[0] / denominator**2,
        ]
    )
    expected = np.sqrt(gradient @ covariance[:2, :2] @ gradient)

    error = health._analytic_target_x_error(cast(Any, fit))

    assert error == pytest.approx(expected, rel=1.0e-12)


def test_decay_covariance_uses_supplied_errors_as_absolute_scale() -> None:
    """Exact data should retain finite uncertainty from supplied point errors."""
    times = np.linspace(0.0, 6000.0, 10)
    values = 0.15 + 0.8 * np.exp(-8.0e-5 * times)

    fit = health._fit_decay_health(
        times,
        values,
        np.full(times.size, 2.0e-4),
        minimum_change=0.01,
        change_sigma_threshold=3.0,
        robust_loss="linear",
    )

    assert fit.success
    assert 1.0e-9 < fit.rate_standard_error < 1.0e-4


def test_decay_fit_with_no_residual_degrees_of_freedom_is_poor() -> None:
    """A three-parameter fit to three points should not claim assessed quality."""
    times = np.array([0.0, 1000.0, 2000.0])
    values = 0.15 + 0.8 * np.exp(-8.0e-5 * times)

    fit = health._fit_decay_health(
        times,
        values,
        np.full(times.size, 2.0e-4),
        minimum_change=0.01,
        change_sigma_threshold=3.0,
        robust_loss="linear",
    )

    assert fit.success
    assert fit.quality == "poor"
    assert np.isnan(fit.reduced_chi_squared)
    assert "no residual degrees of freedom" in fit.message


def test_analyze_cr_pulse_health_recovers_synthetic_health_rates(
    synthetic_measurements,
) -> None:
    """The public analysis should recover all resolved synthetic health rates."""
    measurements, expected = synthetic_measurements

    analysis = health.analyze_cr_pulse_health(
        measurements,
        population_minimum_change=1.0e-3,
        leakage_minimum_change=1.0e-3,
        pauli_minimum_change=1.0e-3,
        change_sigma_threshold=2.0,
        zx90_echo_timing=health.ZX90Timing(
            cr_lobe_duration=50.0,
            echo=True,
            total_duration=200.0,
        ),
        control_idle_noise=health.IdleHealthNoise(t1=30_000.0, t2_echo=24_000.0),
        target_idle_noise=health.IdleHealthNoise(t1=35_000.0, t2_echo=28_000.0),
    )

    for name in (
        "gamma_control_e_to_g",
        "gamma_control_g_to_e",
        "gamma_control_e_to_f",
        "gamma_target_x_decay",
        "gamma_target_leakage",
        "gamma_control_xy_decay",
        "gamma_target_z_decay",
    ):
        assert analysis.cr_active_rates[name] == pytest.approx(expected[name], rel=0.03)

    assert analysis.cr_active_rates["gamma_control_f_to_e"] == 0.0
    assert analysis.cr_active_rates["gamma_target_seepage"] == 0.0

    assert analysis.rate_differences_from_reference[
        "gamma_control_xy_decay"
    ] == pytest.approx(expected["gamma_control_xy_decay"] - 2.0e-5, rel=0.03)
    assert analysis.rate_differences_from_reference[
        "gamma_target_z_decay"
    ] == pytest.approx(expected["gamma_target_z_decay"] - 1.0e-5, rel=0.03)

    assert analysis.fidelity.available
    assert 0.0 < analysis.fidelity.estimated_fidelity < 1.0
    assert 0.0 < analysis.fidelity.idle_coherence_limit < 1.0
    assert analysis.fidelity.average_leakage > 0.0
    assert (
        analysis.metadata["analysis_goal"] == "health_check_not_strict_identification"
    )


def test_rough_fidelity_decreases_when_cr_noise_is_increased() -> None:
    """The rough fidelity should decrease as effective CR noise increases."""
    control_idle = health.IdleHealthNoise(t1=30_000.0, t2_echo=24_000.0)
    target_idle = health.IdleHealthNoise(t1=35_000.0, t2_echo=28_000.0)

    low_noise = health._rough_fidelity_from_parameters(
        gate_total_duration=200.0,
        gate_cr_duration=100.0,
        control_idle=control_idle,
        target_idle=target_idle,
        control_xy_rate=4.0e-5,
        control_z_rate=4.0e-5,
        target_x_rate=4.0e-5,
        target_z_rate=4.0e-5,
        control_leakage_rate=0.0,
        target_leakage_rate=0.0,
    )[0]
    high_noise = health._rough_fidelity_from_parameters(
        gate_total_duration=200.0,
        gate_cr_duration=100.0,
        control_idle=control_idle,
        target_idle=target_idle,
        control_xy_rate=2.0e-4,
        control_z_rate=2.0e-4,
        target_x_rate=2.0e-4,
        target_z_rate=2.0e-4,
        control_leakage_rate=5.0e-5,
        target_leakage_rate=5.0e-5,
    )[0]

    assert high_noise < low_noise


def test_idle_noise_loading_rejects_boolean_lifetimes() -> None:
    """Boolean idle lifetimes should not be coerced to one nanosecond."""
    with pytest.raises(TypeError, match="t1 must be a real number"):
        health._load_idle_noise(
            cast(Any, SimpleNamespace()),
            "Q0",
            "Q1",
            {"Q0": True, "Q1": 30_000.0},
            {"Q0": 20_000.0, "Q1": 20_000.0},
        )


def test_fidelity_uncertainty_stays_unresolved_when_a_rate_error_is_missing() -> None:
    """Fidelity uncertainty should not treat a subthreshold rate as exact zero."""
    parameter_values = {"stable_rate": 0.0, "resolved_rate": 1.0}
    parameter_errors = {"stable_rate": np.nan, "resolved_rate": 0.1}

    error = health._fidelity_standard_error(
        parameter_values,
        parameter_errors,
        lambda parameters: parameters["stable_rate"] + parameters["resolved_rate"],
    )

    assert np.isnan(error)


def test_analysis_rejects_non_fully_active_control_state_times(
    synthetic_measurements,
) -> None:
    """Control-state data should require total time to equal CR-active time."""
    measurements, _ = synthetic_measurements
    active_times = dict(measurements.cr_active_times)
    active_times["control_ground"] = np.asarray(active_times["control_ground"]) * 0.5

    with pytest.raises(ValueError, match=r"control_ground.*fully CR-active"):
        health.analyze_cr_pulse_health(
            replace(measurements, cr_active_times=active_times),
            estimate_fidelity=False,
        )


def test_analysis_rejects_nonconstant_echo_duty_cycle(
    synthetic_measurements,
) -> None:
    """Echo correction should reject a duty cycle that varies across points."""
    measurements, _ = synthetic_measurements
    active_times = dict(measurements.cr_active_times)
    active_times["control_t2_echo"] = np.array(
        [0.0, 400.0, 790.0, 1200.0, 2000.0, 3200.0, 5200.0]
    )

    with pytest.raises(ValueError, match=r"control_t2_echo.*duty cycle"):
        health.analyze_cr_pulse_health(
            replace(measurements, cr_active_times=active_times),
            estimate_fidelity=False,
        )


def test_analysis_rejects_population_outside_probability_simplex(
    synthetic_measurements,
) -> None:
    """Offline GEF populations should remain physical probability vectors."""
    measurements, _ = synthetic_measurements
    populations = {
        protocol: {
            kind: {
                qubit: np.array(values, copy=True)
                for qubit, values in measurements.populations[protocol][kind].items()
            }
            for kind in ("actual", "reference")
        }
        for protocol in ("control_ground", "control_excited")
    }
    populations["control_ground"]["actual"]["Q0"][1] = [-0.1, 1.1, 0.0]

    with pytest.raises(ValueError, match="probability simplex"):
        health.analyze_cr_pulse_health(
            replace(measurements, populations=populations),
            estimate_fidelity=False,
        )


def test_analysis_preserves_other_results_when_target_x_is_undefined(
    synthetic_measurements,
) -> None:
    """Undefined target X should fail only that fit and preserve other health results."""
    measurements, _ = synthetic_measurements
    populations = {
        protocol: {
            kind: {
                qubit: np.array(values, copy=True)
                for qubit, values in measurements.populations[protocol][kind].items()
            }
            for kind in ("actual", "reference")
        }
        for protocol in ("control_ground", "control_excited")
    }
    populations["control_ground"]["actual"]["Q1"][-1] = [0.0, 0.0, 1.0]

    analysis = health.analyze_cr_pulse_health(
        replace(measurements, populations=populations),
        population_minimum_change=1.0e-3,
        leakage_minimum_change=1.0e-3,
        pauli_minimum_change=1.0e-3,
        change_sigma_threshold=2.0,
        estimate_fidelity=False,
    )

    target_x_fit = analysis.target_x_fits["control_ground"]["actual"]
    assert not target_x_fit.success
    assert target_x_fit.assessment.status == "invalid"
    assert np.isnan(analysis.cr_active_rates["gamma_target_x_decay"])
    assert analysis.control_rate_fits["actual"].success
    assert np.isfinite(analysis.cr_active_rates["gamma_control_e_to_g"])


def test_error_bars_preserve_missing_uncertainty() -> None:
    """Plotting should not turn unavailable uncertainty into exact zero error."""
    errors = np.array([np.nan, np.nan])

    error_bar = health._error_bar(errors)

    assert error_bar["visible"] is False
    assert np.all(np.isnan(np.asarray(error_bar["array"], dtype=float)))
