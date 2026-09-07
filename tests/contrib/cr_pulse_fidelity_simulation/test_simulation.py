"""Tests for the CR-on dissipative fidelity simulation."""

# ruff: noqa: SLF001

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib.experiment import (
    cr_pulse_coherence_fitting as fitting_module,
    cr_pulse_fidelity_simulation as module,
)
from qubex.contrib.experiment.cr_pulse_coherence_fitting import (
    fit_cr_on_dephasing,
    fit_target_leakage,
    fit_target_t1rho,
)
from qubex.contrib.experiment.cr_pulse_fidelity_simulation import (
    CrOnNoise,
    IdleQubitNoise,
    ZX90GateTiming,
    extract_zx90_gate_timing,
    prepare_cr_echo_decay_model,
    simulate_cr_pulse_fidelity,
)
from qubex.pulse import CrossResonance, Rect


def _gate_timing() -> ZX90GateTiming:
    """Return a small echoed ZX90 timing model."""
    return ZX90GateTiming(
        cr_lobe_duration=20.0,
        echo=True,
        control_pi_duration=4.0,
        echo_margin_duration=1.0,
    )


def _zero_idle_noise() -> IdleQubitNoise:
    """Return an infinite-lifetime idle model."""
    return IdleQubitNoise(t1=np.inf, t2_echo=np.inf)


def _zero_cr_noise() -> CrOnNoise:
    """Return a CR-on model with every rate disabled."""
    return CrOnNoise(
        gamma_control_g_to_e=0.0,
        gamma_control_e_to_g=0.0,
        gamma_control_e_to_f=0.0,
        gamma_control_f_to_e=0.0,
        target_t1rho=np.inf,
        target_leakage_rate=0.0,
        target_seepage_rate=0.0,
        gamma_phi_control=0.0,
        gamma_phi_rho_target=0.0,
    )


def test_noise_free_simulation_has_unit_fidelities_and_zero_leakage() -> None:
    """Every reported limit should be ideal when all rates vanish."""
    result = simulate_cr_pulse_fidelity(
        _gate_timing(),
        _zero_idle_noise(),
        _zero_idle_noise(),
        _zero_cr_noise(),
    )

    assert result.idle_coherence_limited_fidelity == pytest.approx(1.0, abs=1e-12)
    assert result.cr_on_coherence_limited_fidelity == pytest.approx(1.0, abs=1e-12)
    assert result.cr_on_dissipative_limited_fidelity == pytest.approx(1.0, abs=1e-12)
    assert result.average_leakage == pytest.approx(0.0, abs=1e-12)


def test_cr_on_coherence_matches_idle_for_equivalent_noise_channels() -> None:
    """Equivalent CR-active and idle channels should give the same limit."""
    control_idle = IdleQubitNoise(t1=50_000.0, t2_echo=70_000.0)
    target_idle = _zero_idle_noise()
    gamma_control_relaxation = 1 / control_idle.t1
    gamma_control_dephasing = 1 / control_idle.t2_echo - gamma_control_relaxation / 2
    cr_noise = replace(
        _zero_cr_noise(),
        gamma_control_e_to_g=gamma_control_relaxation,
        gamma_phi_control=gamma_control_dephasing,
    )

    result = simulate_cr_pulse_fidelity(
        _gate_timing(),
        control_idle,
        target_idle,
        cr_noise,
    )

    assert result.cr_on_coherence_limited_fidelity == pytest.approx(
        result.idle_coherence_limited_fidelity,
        abs=1e-12,
    )


def test_disabling_leakage_makes_coherence_and_dissipative_modes_equal() -> None:
    """Dissipative and coherence modes should agree without leakage channels."""
    cr_noise = CrOnNoise(
        gamma_control_g_to_e=2e-5,
        gamma_control_e_to_g=4e-5,
        gamma_control_e_to_f=0.0,
        gamma_control_f_to_e=0.0,
        target_t1rho=30_000.0,
        target_leakage_rate=0.0,
        target_seepage_rate=0.0,
        gamma_phi_control=1e-5,
        gamma_phi_rho_target=1.5e-5,
    )

    result = simulate_cr_pulse_fidelity(
        _gate_timing(),
        IdleQubitNoise(t1=50_000.0, t2_echo=70_000.0),
        IdleQubitNoise(t1=45_000.0, t2_echo=60_000.0),
        cr_noise,
    )

    assert result.cr_on_dissipative_limited_fidelity == pytest.approx(
        result.cr_on_coherence_limited_fidelity,
        abs=1e-12,
    )
    assert result.average_leakage == pytest.approx(0.0, abs=1e-12)


def test_positive_leakage_reduces_survival_and_is_reported() -> None:
    """Enabled outward channels should produce a positive average leakage."""
    cr_noise = CrOnNoise(
        gamma_control_g_to_e=0.0,
        gamma_control_e_to_g=0.0,
        gamma_control_e_to_f=8e-4,
        gamma_control_f_to_e=0.0,
        target_t1rho=np.inf,
        target_leakage_rate=5e-4,
        target_seepage_rate=0.0,
    )

    result = simulate_cr_pulse_fidelity(
        _gate_timing(),
        _zero_idle_noise(),
        _zero_idle_noise(),
        cr_noise,
    )

    assert result.average_leakage > 0.0
    assert result.cr_on_dissipative_limited_fidelity < (
        result.cr_on_coherence_limited_fidelity
    )


def test_cached_forward_gate_matches_full_gate_construction() -> None:
    """Cached fit generators should reproduce the canonical dissipative gate."""
    timing = _gate_timing()
    control_idle = IdleQubitNoise(t1=50_000.0, t2_echo=70_000.0)
    target_idle = IdleQubitNoise(t1=45_000.0, t2_echo=60_000.0)
    noise = CrOnNoise(
        gamma_control_g_to_e=2e-5,
        gamma_control_e_to_g=4e-5,
        gamma_control_e_to_f=1e-6,
        gamma_control_f_to_e=2e-6,
        target_t1rho=30_000.0,
        target_leakage_rate=1e-6,
        target_seepage_rate=2e-6,
        gamma_phi_control=1.5e-5,
        gamma_phi_rho_target=2.5e-5,
    )
    model = prepare_cr_echo_decay_model(
        timing,
        control_idle,
        target_idle,
        noise,
        4.0,
        6.0,
    )

    cached_gate = model.gate_map(
        noise.gamma_phi_control,
        noise.gamma_phi_rho_target,
    )
    canonical_gate = module._zx90_map(
        timing,
        control_idle,
        target_idle,
        noise,
        "cr_on_dissipative",
    )

    assert_allclose(cached_gate, canonical_gate, rtol=1e-12, atol=1e-12)


def test_extract_gate_timing_preserves_echo_pulse_and_margin_durations() -> None:
    """Timing extraction should separate CR-active, pi-pulse, and margin time."""
    gate = CrossResonance(
        control_qubit="Q0",
        target_qubit="Q1",
        cr_amplitude=0.2,
        cr_duration=20.0,
        echo=True,
        pi_pulse=Rect(duration=4.0, amplitude=0.1),
        pi_margin=2.0,
    )

    timing = extract_zx90_gate_timing(gate)

    assert timing.cr_lobe_duration == pytest.approx(20.0)
    assert timing.control_pi_duration == pytest.approx(4.0)
    assert timing.echo_margin_duration == pytest.approx(2.0)
    assert timing.duration == pytest.approx(gate.duration)


def test_joint_target_t1rho_fit_uses_shared_tau_and_separate_amplitudes() -> None:
    """The two target polarization curves should share one recovered lifetime."""
    times = np.array([0, 400, 900, 1800, 3200, 5200, 8000.0])
    tau = 4200.0
    ground = 0.93 * np.exp(-times / tau)
    excited = -0.78 * np.exp(-times / tau)

    fit = fit_target_t1rho(times, ground, excited)

    assert fit.success
    assert fit.t1rho == pytest.approx(tau, rel=1e-5)
    assert fit.amplitude_ground == pytest.approx(0.93, rel=1e-5)
    assert fit.amplitude_excited == pytest.approx(-0.78, rel=1e-5)
    assert_allclose(fit.fitted_ground, ground, atol=1e-8)
    assert_allclose(fit.fitted_excited, excited, atol=1e-8)


def test_joint_target_leakage_fit_recovers_common_leakage_and_seepage() -> None:
    """Two target Pf curves should recover their common effective rates."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    leakage = 2.0e-5
    seepage = 4.0e-5
    equilibrium = leakage / (leakage + seepage)
    ground = equilibrium + (0.01 - equilibrium) * np.exp(-(leakage + seepage) * times)
    excited = equilibrium + (0.04 - equilibrium) * np.exp(-(leakage + seepage) * times)

    fit = fit_target_leakage(times, ground, excited)

    assert fit.success
    assert fit.model == "leakage_and_seepage"
    assert fit.leakage_rate == pytest.approx(leakage, rel=1e-4)
    assert fit.seepage_rate == pytest.approx(seepage, rel=1e-4)
    assert_allclose(fit.fitted_ground, ground, atol=1e-8)
    assert_allclose(fit.fitted_excited, excited, atol=1e-8)


def test_target_leakage_fit_falls_back_when_seepage_is_unresolved() -> None:
    """A zero-seepage trajectory should use the leakage-only reduced model."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    leakage = 2.0e-5
    ground = 1 + (0.01 - 1) * np.exp(-leakage * times)
    excited = 1 + (0.04 - 1) * np.exp(-leakage * times)

    fit = fit_target_leakage(times, ground, excited)

    assert fit.success
    assert fit.model == "leakage_only"
    assert fit.leakage_rate == pytest.approx(leakage, rel=1e-4)
    assert fit.seepage_rate == 0.0


def test_full_echo_decay_forward_fit_recovers_both_cr_on_dephasing_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joint raw-curve fitting should recover control and target dephasing."""
    timing = _gate_timing()
    idle = IdleQubitNoise(t1=50_000.0, t2_echo=70_000.0)
    base_noise = CrOnNoise(
        gamma_control_g_to_e=2e-5,
        gamma_control_e_to_g=4e-5,
        gamma_control_e_to_f=1e-6,
        gamma_control_f_to_e=2e-6,
        target_t1rho=30_000.0,
        target_leakage_rate=1e-6,
        target_seepage_rate=2e-6,
    )
    expected_control = 1.5e-5
    expected_target = 2.5e-5
    fitted_noise = replace(
        base_noise,
        gamma_phi_control=expected_control,
        gamma_phi_rho_target=expected_target,
    )
    n_values = np.array([0, 1, 2, 4, 7])
    model = prepare_cr_echo_decay_model(
        timing,
        idle,
        idle,
        base_noise,
        4.0,
        6.0,
    )
    control_x, target_z = model.predict(
        n_values,
        fitted_noise.gamma_phi_control,
        fitted_noise.gamma_phi_rho_target,
    )
    covariance_parameter_counts: list[int] = []
    original_fit_covariance = fitting_module._estimate_covariance

    def capture_fit_covariance(
        jacobian: np.ndarray,
        cost: float,
        residual_count: int,
        parameter_count: int,
        *,
        absolute_weights: bool,
    ) -> np.ndarray:
        covariance_parameter_counts.append(parameter_count)
        return original_fit_covariance(
            jacobian,
            cost,
            residual_count,
            parameter_count,
            absolute_weights=absolute_weights,
        )

    monkeypatch.setattr(
        fitting_module,
        "_estimate_covariance",
        capture_fit_covariance,
    )

    fit = fit_cr_on_dephasing(
        model,
        n_values,
        0.9 * control_x + 0.03,
        0.8 * target_z + 0.05,
        np.full(n_values.shape, 0.003),
        np.full(n_values.shape, 0.003),
    )

    assert fit.success
    assert fit.gamma_phi_control == pytest.approx(expected_control, rel=2e-4)
    assert fit.gamma_phi_rho_target == pytest.approx(expected_target, rel=2e-4)
    assert fit.curve_n_values[-1] == n_values[-1]
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert covariance_parameter_counts == [6]


def test_forward_model_rejects_an_unechoed_gate() -> None:
    """The echo-decay predictor should require an echoed ZX90."""
    with pytest.raises(ValueError, match="echoed ZX90"):
        prepare_cr_echo_decay_model(
            ZX90GateTiming(cr_lobe_duration=20.0, echo=False),
            _zero_idle_noise(),
            _zero_idle_noise(),
            _zero_cr_noise(),
            4.0,
            4.0,
        )
