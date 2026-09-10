"""Tests for the CR-on dissipative fidelity simulation."""

# ruff: noqa: SLF001

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib.experiment import (
    cr_pulse_coherence_analysis as analysis_module,
    cr_pulse_fidelity_simulation as module,
)
from qubex.contrib.experiment.cr_pulse_coherence_analysis import (
    CrOnDephasingFit,
    fit_cr_on_control_dephasing,
    fit_cr_on_target_rotating_frame_dephasing,
    fit_cr_target_decay,
    fit_target_leakage,
    fit_target_t1rho,
)
from qubex.contrib.experiment.cr_pulse_fidelity_simulation import (
    CrOnNoise,
    IdleQubitNoise,
    ZX90GateTiming,
    extract_zx90_gate_timing,
    prepare_cr_echo_decay_model,
    prepare_cr_target_decay_model,
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


def test_fidelity_analysis_can_fit_one_curve_without_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reusable analysis API should support an independent control fit."""
    expected_fit = CrOnDephasingFit(
        success=True,
        message="ok",
        gamma_phi=2e-5,
        gamma_phi_error=1e-6,
        covariance=np.array([[1e-12]]),
        amplitude=0.9,
        offset=0.05,
        fitted_values=np.array([0.95, 0.85, 0.75]),
        curve_n_values=np.array([0, 1, 2]),
        curve_values=np.array([0.95, 0.85, 0.75]),
        r_squared=0.99,
    )
    monkeypatch.setattr(
        analysis_module,
        "prepare_cr_echo_decay_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        analysis_module,
        "fit_cr_on_control_dephasing",
        lambda *args, **kwargs: expected_fit,
    )

    result = analysis_module.analyze_cr_pulse_fidelity(
        _gate_timing(),
        [0, 1, 2],
        _zero_idle_noise(),
        _zero_idle_noise(),
        _zero_cr_noise(),
        control_x=[0.95, 0.85, 0.75],
        control_x180_duration=4.0,
        target_x180_duration=4.0,
        run_simulation=False,
    )

    assert result.success
    assert result.control_dephasing_fit is expected_fit
    assert result.target_dephasing_fit is None
    assert result.simulation is None
    assert result.cr_on_noise is not None
    assert result.cr_on_noise.gamma_phi_control == pytest.approx(2e-5)


def test_fidelity_analysis_requires_both_curves_for_explicit_simulation() -> None:
    """Explicit simulation should reject an incomplete pair of echo curves."""
    with pytest.raises(ValueError, match="both control_x and target_z"):
        analysis_module.analyze_cr_pulse_fidelity(
            _gate_timing(),
            [0, 1, 2],
            _zero_idle_noise(),
            _zero_idle_noise(),
            _zero_cr_noise(),
            control_x=[1.0, 0.9, 0.8],
            run_simulation=True,
        )


def test_control_forward_fit_requires_physical_x180_durations() -> None:
    """Offline control fitting should reject omitted X180 durations."""
    with pytest.raises(ValueError, match="control_x180_duration"):
        analysis_module.analyze_cr_pulse_fidelity(
            _gate_timing(),
            [0, 1, 2],
            _zero_idle_noise(),
            _zero_idle_noise(),
            _zero_cr_noise(),
            control_x=[1.0, 0.9, 0.8],
            run_simulation=False,
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
    assert (
        result.model_metadata["cr_lobe_rate_model"]
        == "same_for_positive_and_negative_drive_signs"
    )
    assert (
        result.model_metadata["cr_on_rate_target_state_dependence"]
        == "effective_average_not_explicitly_modeled"
    )


def test_explicitly_equal_negative_lobe_noise_matches_default() -> None:
    """Omitted negative-lobe noise should mean the positive-lobe model."""
    noise = replace(
        _zero_cr_noise(),
        gamma_control_e_to_g=3e-5,
        target_t1rho=40_000.0,
    )

    default = simulate_cr_pulse_fidelity(
        _gate_timing(),
        _zero_idle_noise(),
        _zero_idle_noise(),
        noise,
    )
    explicit = simulate_cr_pulse_fidelity(
        _gate_timing(),
        _zero_idle_noise(),
        _zero_idle_noise(),
        noise,
        cr_noise_negative=noise,
    )

    assert default == explicit


def test_un_echoed_simulation_reports_that_only_positive_noise_is_used() -> None:
    """An un-echoed gate should not claim to use a nonexistent negative lobe."""
    result = simulate_cr_pulse_fidelity(
        ZX90GateTiming(cr_lobe_duration=20.0, echo=False),
        _zero_idle_noise(),
        _zero_idle_noise(),
        _zero_cr_noise(),
    )

    assert result.model_metadata["cr_lobe_rate_model"] == "positive_lobe_only"


def test_independent_negative_lobe_noise_is_used() -> None:
    """A distinct negative-lobe model should affect the echoed gate result."""
    negative_noise = replace(
        _zero_cr_noise(),
        gamma_control_e_to_f=8e-4,
    )

    result = simulate_cr_pulse_fidelity(
        _gate_timing(),
        _zero_idle_noise(),
        _zero_idle_noise(),
        _zero_cr_noise(),
        cr_noise_negative=negative_noise,
    )

    assert result.average_leakage > 0.0
    assert (
        result.model_metadata["cr_lobe_rate_model"]
        == "independent_positive_and_negative_drive_signs"
    )

    default_model = prepare_cr_echo_decay_model(
        _gate_timing(),
        _zero_idle_noise(),
        _zero_idle_noise(),
        _zero_cr_noise(),
        4.0,
        6.0,
    )
    independent_model = prepare_cr_echo_decay_model(
        _gate_timing(),
        _zero_idle_noise(),
        _zero_idle_noise(),
        _zero_cr_noise(),
        4.0,
        6.0,
        cr_noise_negative=negative_noise,
    )
    n_values = np.array([0, 1, 2], dtype=np.int64)
    assert not np.allclose(
        independent_model.predict_control_x(n_values, 0.0),
        default_model.predict_control_x(n_values, 0.0),
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("control_pi_duration", [None, 0.0])
def test_manual_echoed_timing_requires_control_pi_duration(
    control_pi_duration: float | None,
) -> None:
    """Manual echoed timing must not silently assume an instantaneous pi pulse."""
    with pytest.raises(ValueError, match="control_pi_duration"):
        ZX90GateTiming(
            cr_lobe_duration=20.0,
            echo=True,
            control_pi_duration=control_pi_duration,
        )


def test_idle_noise_warns_when_pure_dephasing_is_clamped() -> None:
    """An unphysical T2 > 2*T1 input should emit the documented warning."""
    with pytest.warns(RuntimeWarning, match=r"t2_echo exceeds 2 \* t1"):
        noise = IdleQubitNoise(t1=10_000.0, t2_echo=30_000.0)

    assert module._idle_rates(noise) == pytest.approx((1e-4, 0.0, True))


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


def test_target_leakage_collapses_match_the_effective_pf_rate_equation() -> None:
    """Target qutrit collapses should realize dPf/dt=L(1-Pf)-S*Pf."""
    leakage = 2.0e-5
    seepage = 4.0e-5
    initial_pf = 0.1
    duration = 10_000.0
    noise = replace(
        _zero_cr_noise(),
        target_leakage_rate=leakage,
        target_seepage_rate=seepage,
    )
    plus = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
    f_state = np.array([0.0, 0.0, 1.0], dtype=complex)
    target_density = (1 - initial_pf) * np.outer(plus, plus.conj()) + initial_pf * (
        np.outer(f_state, f_state.conj())
    )
    control_ground = np.diag([1.0, 0.0, 0.0]).astype(complex)
    initial_density = np.kron(control_ground, target_density)
    initial_vector = initial_density.reshape(-1, order="F")
    channel = module._segment_map(
        np.zeros((module._FULL_DIMENSION,) * 2, dtype=complex),
        duration,
        module._cr_collapse_operators(noise, include_leakage=True),
    )

    final_density = (channel @ initial_vector).reshape((9, 9), order="F")
    simulated_pf = float(
        np.real(sum(final_density[index, index] for index in (2, 5, 8)))
    )
    equilibrium = leakage / (leakage + seepage)
    expected_pf = equilibrium + (initial_pf - equilibrium) * np.exp(
        -(leakage + seepage) * duration
    )

    assert simulated_pf == pytest.approx(expected_pf, abs=1e-12)


def test_projected_fidelity_formula_handles_uniform_erasure() -> None:
    """A q-times-identity projected channel should have Favg=survival=q."""
    survival_probability = 0.73
    computational_indices = (0, 1, 3, 4)
    surviving = np.eye(9, dtype=complex)
    surviving[list(computational_indices), list(computational_indices)] = np.sqrt(
        survival_probability
    )
    kraus = [surviving]
    for index in computational_indices:
        leaked = np.zeros((9, 9), dtype=complex)
        leaked[8, index] = np.sqrt(1 - survival_probability)
        kraus.append(leaked)
    channel = np.zeros(
        (module._SUPEROPERATOR_DIMENSION,) * 2,
        dtype=np.complex128,
    )
    for operator in kraus:
        channel += np.kron(operator.conj(), operator)

    average_fidelity, average_survival = module._fidelity_and_survival(
        channel,
        np.eye(module._SUPEROPERATOR_DIMENSION, dtype=complex),
    )

    assert average_fidelity == pytest.approx(survival_probability, abs=1e-12)
    assert average_survival == pytest.approx(survival_probability, abs=1e-12)


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


def test_noise_free_forward_model_keeps_primary_components_at_one() -> None:
    """Noise-free C and D forward predictions should remain flat at one."""
    model = prepare_cr_echo_decay_model(
        _gate_timing(),
        _zero_idle_noise(),
        _zero_idle_noise(),
        _zero_cr_noise(),
        4.0,
        6.0,
    )
    n_values = np.array([0, 1, 2, 5], dtype=np.int64)

    assert_allclose(model.predict_control_x(n_values, 0.0), 1.0, atol=1e-12)
    assert_allclose(model.predict_target_z(n_values, 0.0), 1.0, atol=1e-12)


def test_parallel_x_layer_does_not_stretch_the_shorter_pulse() -> None:
    """Different-duration simultaneous X pulses should stop independently."""
    control_duration = 4.0
    target_duration = 6.0
    idle_operators = module._idle_collapse_operators(
        IdleQubitNoise(t1=50_000.0, t2_echo=70_000.0),
        IdleQubitNoise(t1=45_000.0, t2_echo=60_000.0),
    )
    first_hamiltonian = np.pi * module._CONTROL_X / (
        2 * control_duration
    ) + np.pi * module._TARGET_X / (2 * target_duration)
    second_hamiltonian = np.pi * module._TARGET_X / (2 * target_duration)
    expected = module._compose(
        module._segment_map(first_hamiltonian, control_duration, idle_operators),
        module._segment_map(
            second_hamiltonian,
            target_duration - control_duration,
            idle_operators,
        ),
    )

    actual = module._parallel_x_layer_map(
        control_duration,
        np.pi,
        target_duration,
        np.pi,
        idle_operators,
    )

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_target_decay_model_reduces_to_single_exponential_without_switching() -> None:
    """Without control switching, A/B target X should follow their local T1rho."""
    duration = 20.0
    n_values = np.array([0, 1, 2, 4, 7], dtype=np.int64)
    model = prepare_cr_target_decay_model(duration, 0.0, 0.0, 0.0, 0.0)

    ground, excited = model.predict_pair(
        n_values,
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (0.9, -0.8),
        (0.01, 0.03),
        (1 / 4000.0, 1 / 7000.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )

    times = 4 * n_values * duration
    assert_allclose(ground.target_x, 0.9 * np.exp(-times / 4000.0), atol=1e-12)
    assert_allclose(excited.target_x, -0.8 * np.exp(-times / 7000.0), atol=1e-12)
    assert_allclose(ground.target_f, 0.01, atol=1e-12)
    assert_allclose(excited.target_f, 0.03, atol=1e-12)


def test_target_decay_forward_fit_recovers_switching_and_state_dependent_rates() -> (
    None
):
    """The joint fit should recover local target rates despite control switching."""
    duration = 100.0
    n_values = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 55])
    controls = (np.array([0.99, 0.01, 0.0]), np.array([0.01, 0.98, 0.01]))
    model = prepare_cr_target_decay_model(
        duration,
        2e-4,
        1e-4,
        1e-5,
        2e-5,
    )
    truth_t1rho = (3000.0, 7000.0)
    truth_leakage = (3e-5, 8e-5)
    truth_seepage = (2e-5, 5e-5)
    ground, excited = model.predict_pair(
        n_values,
        controls,
        (0.92, 0.88),
        (0.01, 0.02),
        (1 / truth_t1rho[0], 1 / truth_t1rho[1]),
        truth_leakage,
        truth_seepage,
    )

    fit = fit_cr_target_decay(
        model,
        n_values,
        *controls,
        ground.target_x,
        excited.target_x,
        ground.target_f,
        excited.target_f,
        *[np.full(n_values.shape, 1e-4) for _ in range(4)],
        relative_uncertainty_threshold=100.0,
    )

    assert fit.success
    assert fit.leakage_model == "full"
    assert fit.target_t1rho_ground == pytest.approx(truth_t1rho[0], rel=1e-5)
    assert fit.target_t1rho_excited == pytest.approx(truth_t1rho[1], rel=1e-5)
    assert fit.leakage_rate_ground == pytest.approx(truth_leakage[0], rel=1e-5)
    assert fit.leakage_rate_excited == pytest.approx(truth_leakage[1], rel=1e-5)
    assert fit.seepage_rate_ground == pytest.approx(truth_seepage[0], rel=1e-5)
    assert fit.seepage_rate_excited == pytest.approx(truth_seepage[1], rel=1e-5)
    assert fit.maximum_control_f_population > 0


def test_target_decay_forward_fit_selects_lowest_cost_successful_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary A/B fit should evaluate every start and retain minimum cost."""
    duration = 100.0
    n_values = np.array([0, 1, 2, 4, 7])
    controls = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    model = prepare_cr_target_decay_model(duration, 0.0, 0.0, 0.0, 0.0)
    desired_scaled_rate = 1e-2
    ground, excited = model.predict_pair(
        n_values,
        controls,
        (0.9, -0.8),
        (0.01, 0.02),
        (desired_scaled_rate / duration, desired_scaled_rate / duration),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    initial_rates: list[float] = []

    def fake_least_squares(
        _residual: object,
        x0: np.ndarray,
        **_kwargs: object,
    ) -> SimpleNamespace:
        initial_rates.append(float(x0[0]))
        parameters = np.asarray(x0, dtype=np.float64).copy()
        if np.isclose(x0[0], desired_scaled_rate):
            parameters[:6] = (
                desired_scaled_rate,
                desired_scaled_rate,
                0.9,
                -0.8,
                0.01,
                0.02,
            )
            cost = 0.0
        else:
            cost = 10.0
        jacobian = np.zeros((4 * n_values.size, parameters.size))
        jacobian[: parameters.size, :] = np.eye(parameters.size)
        return SimpleNamespace(
            success=True,
            cost=cost,
            x=parameters,
            jac=jacobian,
            message="mock converged",
        )

    monkeypatch.setattr(analysis_module, "least_squares", fake_least_squares)

    fit = fit_cr_target_decay(
        model,
        n_values,
        *controls,
        ground.target_x,
        excited.target_x,
        ground.target_f,
        excited.target_f,
        relative_uncertainty_threshold=100.0,
    )

    assert fit.success
    assert initial_rates == [1e-3, 1e-2]
    assert fit.target_t1rho_ground == pytest.approx(duration / desired_scaled_rate)
    assert fit.target_t1rho_excited == pytest.approx(duration / desired_scaled_rate)


def test_target_decay_forward_fit_reports_all_failed_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optimizer failures across candidate models should return a failed fit."""
    duration = 100.0
    n_values = np.array([0, 1, 2, 4, 7])
    controls = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    model = prepare_cr_target_decay_model(duration, 0.0, 0.0, 0.0, 0.0)
    ground, excited = model.predict_pair(
        n_values,
        controls,
        (0.9, -0.8),
        (0.01, 0.02),
        (1e-4, 2e-4),
        (0.0, 0.0),
        (0.0, 0.0),
    )

    def fail_least_squares(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(analysis_module, "least_squares", fail_least_squares)

    fit = fit_cr_target_decay(
        model,
        n_values,
        *controls,
        ground.target_x,
        excited.target_x,
        ground.target_f,
        excited.target_f,
    )

    assert not fit.success
    assert "not identifiable" in fit.message


def test_target_observable_reduction_matches_full_two_qutrit_lindblad() -> None:
    """The optimized observable propagation must equal the full density model."""
    model = prepare_cr_target_decay_model(100.0, 2e-4, 1e-4, 1e-5, 2e-5)
    n_values = np.array([0, 1, 3, 8])
    control = np.array([0.97, 0.02, 0.01])
    rates = ((1 / 3000.0, 1 / 7000.0), (3e-5, 8e-5), (2e-5, 5e-5))

    reduced = model.predict_pair(
        n_values,
        (control, control),
        (0.92, 0.92),
        (0.01, 0.01),
        *rates,
    )[0]
    full = model._predict_one(
        model.gate_map(*rates),
        4 * n_values,
        control,
        0.92,
        0.01,
    )

    assert_allclose(reduced.control_populations, full.control_populations, atol=1e-12)
    assert_allclose(reduced.target_x, full.target_x, atol=1e-12)
    assert_allclose(reduced.target_f, full.target_f, atol=1e-12)


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


def test_target_t1rho_fits_are_independent() -> None:
    """Each control-state target curve should recover its own lifetime."""
    times = np.array([0, 400, 900, 1800, 3200, 5200, 8000.0])
    ground_tau = 4200.0
    excited_tau = 6300.0
    ground = 0.93 * np.exp(-times / ground_tau)
    excited = -0.78 * np.exp(-times / excited_tau)

    ground_fit = fit_target_t1rho(times, ground)
    excited_fit = fit_target_t1rho(times, excited)

    assert ground_fit.success
    assert excited_fit.success
    assert ground_fit.model == excited_fit.model == "exponential"
    assert ground_fit.t1rho == pytest.approx(ground_tau, rel=1e-5)
    assert excited_fit.t1rho == pytest.approx(excited_tau, rel=1e-5)
    assert ground_fit.amplitude == pytest.approx(0.93, rel=1e-5)
    assert excited_fit.amplitude == pytest.approx(-0.78, rel=1e-5)
    assert_allclose(ground_fit.fitted_values, ground, atol=1e-8)
    assert_allclose(excited_fit.fitted_values, excited, atol=1e-8)


def test_target_t1rho_fit_removes_seepage_induced_polarization_dilution() -> None:
    """The leakage-corrected fit should recover the bare dressed decay time."""
    times = np.array([0, 400, 900, 1800, 3200, 5200, 8000.0])
    t1rho = 4200.0
    leakage = 2.0e-5
    seepage = 4.0e-5
    equilibrium = leakage / (leakage + seepage)
    pf_ground = equilibrium + (0.01 - equilibrium) * np.exp(
        -(leakage + seepage) * times
    )
    pf_excited = equilibrium + (0.04 - equilibrium) * np.exp(
        -(leakage + seepage) * times
    )

    def polarization(amplitude: float, pf: np.ndarray) -> np.ndarray:
        return (
            amplitude * (1 - pf[0]) * np.exp(-(1 / t1rho + leakage) * times) / (1 - pf)
        )

    ground = polarization(0.93, pf_ground)
    excited = polarization(-0.78, pf_excited)
    fit = fit_target_t1rho(
        times,
        ground,
        f_population=pf_ground,
        leakage_rate=leakage,
    )

    assert fit.success
    assert fit.model == "leakage_corrected"
    assert fit.leakage_rate_used == pytest.approx(leakage)
    assert fit.t1rho == pytest.approx(t1rho, rel=1e-5)
    assert_allclose(fit.fitted_values, ground, atol=1e-8)

    excited_fit = fit_target_t1rho(
        times,
        excited,
        f_population=pf_excited,
        leakage_rate=leakage,
    )
    assert excited_fit.success
    assert excited_fit.t1rho == pytest.approx(t1rho, rel=1e-5)

    leakage_fit = fit_target_leakage(times, pf_ground)
    dense_times = np.linspace(0.0, times[-1], 101)
    dense_pf_ground = equilibrium + (pf_ground[0] - equilibrium) * np.exp(
        -(leakage + seepage) * dense_times
    )
    expected_dense = (
        fit.amplitude
        * (1 - dense_pf_ground[0])
        * np.exp(-(1 / t1rho + leakage) * dense_times)
        / (1 - dense_pf_ground)
    )
    plotted_dense = analysis_module._target_t1rho_curve(
        fit,
        leakage_fit,
        dense_times,
    )
    assert_allclose(plotted_dense, expected_dense, atol=1e-8)


def test_target_leakage_fits_are_independent() -> None:
    """Each control-state target Pf curve should recover its own rates."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    leakage = 2.0e-5
    seepage = 4.0e-5
    equilibrium = leakage / (leakage + seepage)
    ground = equilibrium + (0.01 - equilibrium) * np.exp(-(leakage + seepage) * times)
    excited = equilibrium + (0.04 - equilibrium) * np.exp(-(leakage + seepage) * times)

    ground_fit = fit_target_leakage(times, ground)
    excited_fit = fit_target_leakage(times, excited)

    for fit, values in ((ground_fit, ground), (excited_fit, excited)):
        assert fit.success
        assert fit.model == "leakage_and_seepage"
        assert fit.leakage_rate == pytest.approx(leakage, rel=1e-4)
        assert fit.seepage_rate == pytest.approx(seepage, rel=1e-4)
        assert_allclose(fit.fitted_values, values, atol=1e-8)


def test_target_leakage_fit_continues_after_one_failed_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed initial guess should not discard later successful starts."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    leakage = 2.0e-5
    seepage = 4.0e-5
    equilibrium = leakage / (leakage + seepage)
    values = equilibrium + (0.01 - equilibrium) * np.exp(-(leakage + seepage) * times)
    original_least_squares: Callable[..., object] = analysis_module.least_squares
    attempt_count = 0

    def fail_first_start(*args: object, **kwargs: object):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise RuntimeError("failed initial guess")
        return original_least_squares(*args, **kwargs)

    monkeypatch.setattr(analysis_module, "least_squares", fail_first_start)

    fit = fit_target_leakage(times, values)

    assert attempt_count > 1
    assert fit.success
    assert fit.model == "leakage_and_seepage"


def test_target_leakage_fit_reports_all_failed_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete optimizer failure should remain a structured fit result."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    values = 0.3 + (0.01 - 0.3) * np.exp(-6.0e-5 * times)

    def fail_least_squares(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(analysis_module, "least_squares", fail_least_squares)

    fit = fit_target_leakage(times, values)

    assert not fit.success
    assert "least-squares starts failed" in fit.message
    assert fit.model is None
    assert np.all(np.isnan(fit.fitted_values))


def test_target_leakage_constant_curve_uses_zero_rate_model() -> None:
    """A constant target Pf curve should not resolve compensating nonzero rates."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    values = np.full(times.shape, 0.002)

    fit = fit_target_leakage(times, values, np.full(times.shape, 0.001))

    assert fit.success
    assert fit.model == "none"
    assert fit.leakage_rate == 0.0
    assert fit.seepage_rate == 0.0
    assert fit.initial_population == pytest.approx(0.002)
    assert np.all(np.isnan(fit.covariance))


@pytest.mark.parametrize("first_population", [0.001, 0.003])
def test_target_leakage_noisy_constant_curve_prefers_zero_rate_model(
    first_population: float,
) -> None:
    """An uncertain first-point fluctuation should not create a nonzero target rate."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    values = np.array([first_population, 0.0021, 0.0019, 0.0022, 0.0018, 0.002, 0.002])
    standard_errors = np.full(times.shape, 0.001)

    fit = fit_target_leakage(times, values, standard_errors)

    assert fit.success
    assert fit.model == "none"
    assert fit.leakage_rate == 0.0
    assert fit.seepage_rate == 0.0
    assert fit.initial_population == pytest.approx(np.mean(values))
    assert_allclose(fit.fitted_values, np.mean(values))


def test_target_leakage_unweighted_zero_model_fits_the_sample_mean() -> None:
    """An unweighted zero-rate fit should estimate its level from every point."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    values = np.array([0.0024, 0.0020, 0.0021, 0.0019, 0.0020, 0.0021, 0.0019])

    fit = fit_target_leakage(times, values)

    assert fit.success
    assert fit.model == "none"
    assert fit.initial_population == pytest.approx(np.mean(values))
    assert_allclose(fit.fitted_values, np.mean(values))


def test_target_leakage_rejects_zero_fallback_for_unresolved_fast_dynamics() -> None:
    """Visible saturated Pf dynamics should fail when their rate scale is unresolved."""
    times = np.array([0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000.0])
    leakage = 1.0e-3
    seepage = 2.0e-3
    equilibrium = leakage / (leakage + seepage)
    values = equilibrium + (0.002 - equilibrium) * np.exp(-(leakage + seepage) * times)

    fit = fit_target_leakage(times, values, np.full(times.shape, 0.001))

    assert not fit.success
    assert fit.model is None
    assert "dynamics are present" in fit.message
    assert np.isnan(fit.leakage_rate)
    assert np.isnan(fit.seepage_rate)
    assert np.isnan(fit.initial_population)
    assert np.all(np.isnan(fit.fitted_values))
    assert np.isnan(fit.r_squared)


def test_target_leakage_fit_falls_back_when_seepage_is_unresolved() -> None:
    """A first-point fluctuation should not bias a genuine leakage-only rate."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    leakage = 2.0e-5
    ground = 1 + (0.01 - 1) * np.exp(-leakage * times)
    ground[0] += 0.001
    fit = fit_target_leakage(times, ground, np.full(times.shape, 0.001))

    assert fit.success
    assert fit.model == "leakage_only"
    assert fit.leakage_rate == pytest.approx(leakage, rel=2e-3)
    assert fit.seepage_rate == 0.0
    assert fit.initial_population == pytest.approx(0.01, abs=5e-4)


def test_target_leakage_fit_keeps_resolved_seepage_when_leakage_is_zero() -> None:
    """A seepage-only trajectory should retain its measured inward rate."""
    times = np.array([0, 1000, 2500, 5000, 9000, 15_000, 25_000.0])
    seepage = 4.0e-5
    ground = 0.10 * np.exp(-seepage * times)
    fit = fit_target_leakage(times, ground)

    assert fit.success
    assert fit.model == "seepage_only"
    assert fit.leakage_rate == 0.0
    assert fit.seepage_rate == pytest.approx(seepage, rel=1e-4)


@pytest.mark.parametrize(
    ("prediction_method", "fit_function", "expected_rate"),
    [
        ("predict_control_x", fit_cr_on_control_dephasing, 1.5e-5),
        (
            "predict_target_z",
            fit_cr_on_target_rotating_frame_dephasing,
            2.5e-5,
        ),
    ],
)
def test_single_echo_decay_forward_fit_recovers_its_dephasing_rate(
    monkeypatch: pytest.MonkeyPatch,
    prediction_method: str,
    fit_function: Callable[..., CrOnDephasingFit],
    expected_rate: float,
) -> None:
    """Each raw echo curve should independently recover its dephasing rate."""
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
    n_values = np.array([0, 1, 2, 4, 7])
    model = prepare_cr_echo_decay_model(
        timing,
        idle,
        idle,
        base_noise,
        4.0,
        6.0,
    )
    prediction = getattr(model, prediction_method)(n_values, expected_rate)
    covariance_parameter_counts: list[int] = []
    original_fit_covariance = analysis_module._estimate_covariance

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
        analysis_module,
        "_estimate_covariance",
        capture_fit_covariance,
    )

    fit = fit_function(
        model,
        n_values,
        0.9 * prediction + 0.03,
        np.full(n_values.shape, 0.003),
    )

    assert fit.success
    assert fit.gamma_phi == pytest.approx(expected_rate, rel=2e-4)
    assert fit.curve_n_values[-1] == n_values[-1]
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert covariance_parameter_counts == [3]


class _SyntheticControlDecayModel:
    """Provide a small analytic control-X predictor for fit diagnostics."""

    cr_lobe_duration = 20.0

    def predict_control_x(
        self,
        n_values: np.ndarray,
        gamma_phi_control: float,
    ) -> np.ndarray:
        """Return an exponential whose rate is sensitive to gamma_phi_control."""
        return np.exp(-4 * n_values * self.cr_lobe_duration * gamma_phi_control)


class _RateInsensitiveControlDecayModel:
    """Provide a control-X predictor with no dephasing-rate sensitivity."""

    cr_lobe_duration = 20.0

    def predict_control_x(
        self,
        n_values: np.ndarray,
        gamma_phi_control: float,
    ) -> np.ndarray:
        """Return the same curve for every gamma_phi_control value."""
        del gamma_phi_control
        return np.exp(-0.2 * n_values)


def test_forward_fit_rejects_flat_observed_curve_after_spam_profiling() -> None:
    """A flat measured curve should not identify a CR-on dephasing rate."""
    n_values = np.array([0, 1, 2, 4, 7])

    fit = fit_cr_on_control_dephasing(
        _SyntheticControlDecayModel(),
        n_values,
        np.full(n_values.shape, 0.4),
        np.full(n_values.shape, 0.01),
    )

    assert not fit.success
    assert "insufficient dynamic range" in fit.message
    assert np.isnan(fit.gamma_phi)


def test_forward_fit_rejects_rate_insensitive_predictor() -> None:
    """A zero-sensitivity predictor should return no finite rate uncertainty."""
    n_values = np.array([0, 1, 2, 4, 7])
    observed = 0.9 * np.exp(-0.2 * n_values) + 0.03

    fit = fit_cr_on_control_dephasing(
        _RateInsensitiveControlDecayModel(),
        n_values,
        observed,
        np.full(n_values.shape, 0.01),
    )

    assert not fit.success
    assert "insufficient rate sensitivity" in fit.message
    assert np.all(np.isnan(fit.covariance))


def test_forward_fit_discards_parameters_when_optimization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsuccessful optimizer must not expose provisional fit parameters."""
    n_values = np.array([0, 1, 2, 4, 7])
    observed = 0.9 * np.exp(-0.2 * n_values) + 0.03
    monkeypatch.setattr(
        analysis_module,
        "least_squares",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            message="maximum evaluations exceeded",
            x=np.array([0.01]),
            jac=np.ones((n_values.size, 1)),
            cost=1.0,
        ),
    )

    fit = fit_cr_on_control_dephasing(
        _SyntheticControlDecayModel(),
        n_values,
        observed,
        np.full(n_values.shape, 0.01),
    )

    assert not fit.success
    assert "maximum evaluations exceeded" in fit.message
    assert np.isnan(fit.gamma_phi)
    assert np.all(np.isnan(fit.covariance))
    assert np.all(np.isnan(fit.fitted_values))


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
