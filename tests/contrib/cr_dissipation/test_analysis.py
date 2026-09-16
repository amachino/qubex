"""Fast synthetic tests for the CR dissipation v11 physical estimators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import qubex.contrib.experiment._cr_dissipation_analysis as analysis_module
from qubex.contrib.experiment._cr_dissipation_analysis import (
    _aggregate_value_error,
    _fidelity_limits,
    _pure_dephasing_component,
    _select_significant_nested_candidate,
    _simplex_boundary_override,
    _target_control_state_dependence_warnings,
    control_population_trajectory,
    fit_control_populations,
    fit_physical_dephasing,
    fit_target_exchange,
    fit_target_t1rho,
    nonnormalized_ge_expectation,
    target_exchange_trajectory,
    target_t1rho_trajectory,
)
from qubex.contrib.experiment._cr_dissipation_pulses import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    PROTOCOL_D,
    ProtocolSchedules,
    ZX90Descriptor,
)
from qubex.contrib.experiment._cr_dissipation_simulation import (
    X_CONTROL,
    CrNoiseRates,
    SemanticSegment,
    state_density,
)
from qubex.contrib.experiment._cr_dissipation_types import (
    CandidateFit,
    CrDissipationMeasurements,
    CrDissipationProtocolData,
    CrDissipationRateStatus,
    DecayRateFit,
    ExchangeRateFit,
    GefPopulationSeries,
    IdleNoiseParameters,
)
from qubex.pulse import FlatTop, PulseSchedule

from .conftest import measurements_from_protocol_data, population_series


def _candidate(
    name: str,
    names: tuple[str, ...],
    values: tuple[float, ...],
    errors: tuple[float, ...],
    aicc: float,
) -> CandidateFit:
    size = len(names)
    return CandidateFit(
        name,
        True,
        names,
        np.asarray(values),
        np.diag(np.square(errors)),
        np.asarray(errors),
        np.eye(size),
        np.zeros(10),
        aicc,
        aicc,
        aicc,
        10,
        size,
        "synthetic",
    )


def test_aicc_winner_is_iteratively_reduced_when_one_rate_is_sub_2sigma() -> None:
    """A complex initial winner drops only its insignificant nested rate."""
    candidates = {
        "L2": _candidate("L2", ("leak", "seep"), (4e-5, 2e-6), (5e-6, 2e-6), 0.0),
        "L1": _candidate("L1", ("leak",), (4e-5,), (5e-6,), 8.0),
        "L1b": _candidate("L1b", ("seep",), (2e-6,), (2e-6,), 15.0),
        "L0": _candidate("L0", (), (), (), 30.0),
    }

    selected = _select_significant_nested_candidate(
        candidates, {"leak": 0.0, "seep": 0.0}
    )

    assert selected is not None
    assert selected.name == "L1"


def test_boundary_fraction_is_only_an_auxiliary_flat_boundary_override() -> None:
    """A material occupancy change recovers dynamics hidden by clipping."""
    values = np.tile([0.55, 0.45, 0.0], (3, 1))
    base = population_series(values)
    confidence_interval = np.array([[0.50, 0.40, 0.0], [0.60, 0.50, 0.01]])
    initial = SimpleNamespace(
        confidence_interval=confidence_interval,
        boundary_fraction=np.array([0.0, 0.0, 0.9]),
    )
    changed = SimpleNamespace(
        confidence_interval=confidence_interval,
        boundary_fraction=np.array([0.0, 0.0, 0.5]),
    )
    stable = SimpleNamespace(
        confidence_interval=confidence_interval,
        boundary_fraction=np.array([0.0, 0.0, 0.8]),
    )

    assert _simplex_boundary_override(
        replace(base, bootstrap=(initial, changed, changed)),
        2,
        minimum_change=0.003,
    )
    assert not _simplex_boundary_override(
        replace(base, bootstrap=(initial, stable, stable)),
        2,
        minimum_change=0.003,
    )


def test_fidelity_fixed_metadata_matches_sigma_point_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only truly fixed primitive fallbacks are excluded from propagation."""
    primitive_order = (
        "control_e_to_g",
        "control_g_to_e",
        "control_e_to_f",
        "control_f_to_e",
        "target_t1rho",
        "target_leakage",
        "target_seepage",
        "control_pure_dephasing",
        "target_rotating_frame_pure_dephasing",
    )
    nominal: dict[str, float] = dict.fromkeys(primitive_order, 1e-5)
    statuses: dict[str, CrDissipationRateStatus] = dict.fromkeys(
        primitive_order, CrDissipationRateStatus.RESOLVED
    )
    statuses.update(
        {
            "control_e_to_g": CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED,
            "control_g_to_e": CrDissipationRateStatus.UNRESOLVED_ASSUMED_IDLE,
            "control_e_to_f": CrDissipationRateStatus.CONSISTENT_WITH_ZERO,
            "control_f_to_e": CrDissipationRateStatus.INCONSISTENT_RATE_DECOMPOSITION,
            "target_t1rho": CrDissipationRateStatus.PARTIALLY_UNRESOLVED,
            "control_transverse": CrDissipationRateStatus.PARTIALLY_UNRESOLVED,
            "target_t2rho": CrDissipationRateStatus.UNRESOLVED_ASSUMED_IDLE,
        }
    )
    covariance = np.eye(len(primitive_order)) * 1e-12
    propagated: list[set[str]] = []

    def capture(
        central: Mapping[str, float], *_args: Any, **_kwargs: Any
    ) -> tuple[float, bool]:
        propagated.append(set(central))
        return 1e-6, True

    # This test checks metadata bookkeeping, not the expensive 81x81 channel algebra.
    monkeypatch.setattr(analysis_module, "semantic_zx90", lambda *_a, **_k: ())
    monkeypatch.setattr(
        analysis_module, "noiseless_channel", lambda *_a, **_k: np.eye(1)
    )
    monkeypatch.setattr(analysis_module, "compose_channel", lambda *_a, **_k: np.eye(1))
    monkeypatch.setattr(
        analysis_module,
        "leakage_aware_average_fidelity",
        lambda *_a, **_k: (0.999, 0.999, 1.0),
    )
    monkeypatch.setattr(analysis_module, "propagate_fidelity_uncertainty", capture)
    fidelity, _ = _fidelity_limits(
        descriptor(),
        IdleNoiseParameters(float("inf"), float("inf")),
        IdleNoiseParameters(float("inf"), float("inf")),
        nominal,
        statuses,
        covariance,
        primitive_order,
    )

    expected_fixed = {
        "control_e_to_g",
        "control_g_to_e",
        "control_e_to_f",
        "control_f_to_e",
    }
    assert set(fidelity.fixed_unresolved_rates) == expected_fixed
    assert set(fidelity.fallback_rates_per_ns) == expected_fixed
    assert "control_transverse" not in fidelity.fixed_unresolved_rates
    assert "target_t2rho" not in fidelity.fixed_unresolved_rates
    assert propagated
    assert all("target_t1rho" in parameters for parameters in propagated)
    assert all(expected_fixed.isdisjoint(parameters) for parameters in propagated)


def test_pure_dephasing_component_uses_both_longitudinal_directions() -> None:
    """Control transverse decomposition includes upward and downward rates."""
    assert _pure_dephasing_component(8e-5, 6e-5) == pytest.approx(5e-5)
    assert _pure_dephasing_component(2e-5, 6e-5) == 0.0


def test_target_control_state_dependence_warning_is_diagnostic_only() -> None:
    """Only a significant resolved A/B difference emits the warning."""
    resolved = CrDissipationRateStatus.RESOLVED
    exchange_a = SimpleNamespace(
        leakage_rate_per_ns=4e-5,
        leakage_standard_error_per_ns=2e-6,
        leakage_status=resolved,
        seepage_rate_per_ns=2e-5,
        seepage_standard_error_per_ns=2e-6,
        seepage_status=resolved,
    )
    exchange_b = SimpleNamespace(
        leakage_rate_per_ns=4.1e-5,
        leakage_standard_error_per_ns=2e-6,
        leakage_status=resolved,
        seepage_rate_per_ns=2.1e-5,
        seepage_standard_error_per_ns=2e-6,
        seepage_status=resolved,
    )
    t1_a = SimpleNamespace(
        rate_per_ns=5e-5,
        rate_standard_error_per_ns=2e-6,
        status=resolved,
    )
    t1_same = SimpleNamespace(
        rate_per_ns=5.1e-5,
        rate_standard_error_per_ns=2e-6,
        status=resolved,
    )
    t1_different = SimpleNamespace(
        rate_per_ns=8e-5,
        rate_standard_error_per_ns=2e-6,
        status=resolved,
    )

    assert not _target_control_state_dependence_warnings(
        cast(DecayRateFit, t1_a),
        cast(DecayRateFit, t1_same),
        cast(ExchangeRateFit, exchange_a),
        cast(ExchangeRateFit, exchange_b),
    )
    before = _aggregate_value_error(
        t1_a.rate_per_ns,
        t1_a.rate_standard_error_per_ns,
        t1_different.rate_per_ns,
        t1_different.rate_standard_error_per_ns,
    )
    warnings = _target_control_state_dependence_warnings(
        cast(DecayRateFit, t1_a),
        cast(DecayRateFit, t1_different),
        cast(ExchangeRateFit, exchange_a),
        cast(ExchangeRateFit, exchange_b),
    )
    after = _aggregate_value_error(
        t1_a.rate_per_ns,
        t1_a.rate_standard_error_per_ns,
        t1_different.rate_per_ns,
        t1_different.rate_standard_error_per_ns,
    )

    assert [warning.affected_outputs for warning in warnings] == [("target_t1rho",)]
    assert all(
        warning.code == "target_rate_control_state_dependence" for warning in warnings
    )
    assert after == before


def test_idle_baseline_precedes_actual_fit_and_supplies_t1rho_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual selection starts only after the one-pass baseline is available."""
    events: list[object] = []
    baseline = SimpleNamespace(
        target_t1rho_by_protocol={PROTOCOL_A: 1.23e-4, PROTOCOL_B: 2.34e-4}
    )

    def extract(*_args: Any, **_kwargs: Any) -> Any:
        events.append("baseline")
        return baseline

    def control_fit(*_args: Any, **_kwargs: Any) -> Any:
        events.append("control")
        return SimpleNamespace()

    def target_fit(*_args: Any, **kwargs: Any) -> Any:
        events.append(kwargs["idle_equivalent_rate_per_ns"])
        raise RuntimeError("stop after observing the first actual T1rho fit")

    monkeypatch.setattr(analysis_module, "_extract_idle_equivalent_baseline", extract)
    monkeypatch.setattr(analysis_module, "fit_control_populations", control_fit)
    monkeypatch.setattr(analysis_module, "fit_target_t1rho", target_fit)

    with pytest.raises(RuntimeError, match="stop after observing"):
        analysis_module.analyze_cr_dissipation(
            cast(CrDissipationMeasurements, SimpleNamespace()),
            descriptor(),
            cast(ProtocolSchedules, SimpleNamespace()),
            IdleNoiseParameters(50_000.0, 40_000.0),
            IdleNoiseParameters(45_000.0, 35_000.0),
            covariance_rcond=1e-12,
        )

    assert events == ["baseline", "control", 1.23e-4]


def test_control_simplification_updates_status_and_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final joint-fit metadata comes from the post-significance model."""
    counts = np.array([0, 1, 2, 3, 5, 8], dtype=np.int64)
    control = population_series(np.tile([0.90, 0.08, 0.02], (counts.size, 1)))
    target = population_series(np.tile([0.50, 0.48, 0.02], (counts.size, 1)))
    ab = CrDissipationProtocolData(control_gef=control, target_gef=target)
    cd = CrDissipationProtocolData(
        primary_expectation=np.ones(counts.size),
        primary_standard_error=np.full(counts.size, 2e-4),
    )
    measurements = measurements_from_protocol_data(
        counts,
        {PROTOCOL_A: ab, PROTOCOL_B: ab, PROTOCOL_C: cd, PROTOCOL_D: cd},
    )
    idle = IdleNoiseParameters(50_000.0, 40_000.0)

    def fake_fit(**kwargs: Any) -> CandidateFit:
        name = str(kwargs["name"])
        names = tuple(cast(tuple[str, ...], kwargs["parameter_names"]))
        values = tuple(
            8e-5
            if item == "control_e_to_g"
            else 1e-6
            if item == "control_g_to_e"
            else 0.0
            for item in names
        )
        errors = tuple(5e-6 if item == "control_e_to_g" else 1e-6 for item in names)
        aicc = 0.0 if name == "G2 x F0" else 8.0 if name == "G1 x F0" else 30.0
        return _candidate(name, names, values, errors, aicc)

    monkeypatch.setattr(
        "qubex.contrib.experiment._cr_dissipation_analysis.fit_gls_candidate",
        fake_fit,
    )
    fitted = fit_control_populations(
        measurements,
        descriptor(),
        idle,
        covariance_rcond=1e-12,
        force_all_candidates=True,
    )

    assert fitted.selected_model == "G1 x F0"
    assert fitted.statuses["control_e_to_g"] == CrDissipationRateStatus.RESOLVED
    assert (
        fitted.statuses["control_g_to_e"]
        == CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED
    )
    assert fitted.covariance[0, 0] == pytest.approx(25e-12)
    np.testing.assert_array_equal(fitted.covariance[1:, :], 0.0)


def test_exchange_simplification_marks_insignificant_seepage_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sub-2-sigma seepage parameter is removed from an initial L2 winner."""
    counts = np.array([0, 1, 2, 3, 5, 8], dtype=np.int64)
    control = population_series(np.tile([0.90, 0.08, 0.02], (counts.size, 1)))
    target = population_series(np.tile([0.50, 0.48, 0.02], (counts.size, 1)))
    ab = CrDissipationProtocolData(control_gef=control, target_gef=target)
    cd = CrDissipationProtocolData(
        primary_expectation=np.ones(counts.size),
        primary_standard_error=np.full(counts.size, 2e-4),
    )
    measurements = measurements_from_protocol_data(
        counts,
        {PROTOCOL_A: ab, PROTOCOL_B: ab, PROTOCOL_C: cd, PROTOCOL_D: cd},
    )

    def fake_fit(**kwargs: Any) -> CandidateFit:
        name = str(kwargs["name"])
        names = tuple(cast(tuple[str, ...], kwargs["parameter_names"]))
        values = tuple(4e-5 if item.startswith("leakage") else 1e-6 for item in names)
        errors = tuple(5e-6 if item.startswith("leakage") else 1e-6 for item in names)
        aicc = 0.0 if name == "L2" else 8.0 if name == "L1" else 30.0
        return _candidate(name, names, values, errors, aicc)

    monkeypatch.setattr(
        "qubex.contrib.experiment._cr_dissipation_analysis.fit_gls_candidate",
        fake_fit,
    )
    fitted = fit_target_exchange(
        measurements, PROTOCOL_A, descriptor(), force_all_candidates=True
    )

    assert fitted.selected_model == "L1"
    assert fitted.leakage_status == CrDissipationRateStatus.RESOLVED
    assert fitted.seepage_status == CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED
    assert fitted.seepage_standard_error_per_ns is None
    np.testing.assert_array_equal(fitted.covariance[1], 0.0)


def descriptor() -> ZX90Descriptor:
    """Return a timing-only descriptor suitable for classical fits."""
    schedule = PulseSchedule()
    pi = FlatTop(duration=40.0, amplitude=0.2, tau=4.0)
    return ZX90Descriptor(
        cast(PulseSchedule, schedule),
        cast(PulseSchedule, schedule),
        cast(PulseSchedule, schedule),
        50.0,
        40.0,
        40.0,
        0.0,
        180.0,
        pi,
        None,
        None,
        (),
    )


def test_c_d_observable_is_not_leakage_normalized() -> None:
    """Fixed g/e polarization ratio loses contrast as f population grows."""
    population = np.array([[0.75, 0.25, 0.0], [0.60, 0.20, 0.20]])
    covariance = (
        np.array(
            [
                [[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]],
                np.eye(3),
            ]
        )
        * 1e-4
    )
    series = GefPopulationSeries(
        population,
        covariance,
        np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)),
        population.copy(),
    )

    values, errors, sign = nonnormalized_ge_expectation(series)

    assert sign == 1.0
    np.testing.assert_allclose(values, [0.5, 0.4])
    assert errors[0] == pytest.approx(np.sqrt((4.0 + 9.0 - 2.0) * 1e-4))


def test_joint_control_fit_recovers_four_rates_and_excludes_n_zero() -> None:
    """The A/B joint GLS fit recovers rates using only n>0 residuals."""
    counts = np.array([0, 1, 2, 3, 5, 8, 13], dtype=np.int64)
    desc = descriptor()
    idle = IdleNoiseParameters(50_000.0, 40_000.0)
    true_rates = {
        "control_e_to_g": 8e-5,
        "control_g_to_e": 4e-5,
        "control_e_to_f": 5e-5,
        "control_f_to_e": 2e-5,
    }
    a = control_population_trajectory(
        counts,
        [0.98, 0.02, 0.0],
        true_rates,
        cr_lobe_duration_ns=50.0,
        blank_duration_ns=40.0,
        idle_e_to_g_per_ns=idle.relaxation_rate_per_ns,
    )
    b = control_population_trajectory(
        counts,
        [0.03, 0.95, 0.02],
        true_rates,
        cr_lobe_duration_ns=50.0,
        blank_duration_ns=40.0,
        idle_e_to_g_per_ns=idle.relaxation_rate_per_ns,
    )
    flat_target = population_series(np.tile([0.5, 0.5, 0.0], (counts.size, 1)))
    dummy_primary = CrDissipationProtocolData(
        primary_expectation=np.ones(counts.size),
        primary_standard_error=np.full(counts.size, 2e-4),
    )
    measurements = measurements_from_protocol_data(
        counts,
        {
            PROTOCOL_A: CrDissipationProtocolData(
                control_gef=population_series(a),
                target_gef=flat_target,
                target_x_comp=np.ones(counts.size),
                target_x_comp_standard_error=np.full(counts.size, 2e-4),
            ),
            PROTOCOL_B: CrDissipationProtocolData(
                control_gef=population_series(b),
                target_gef=flat_target,
                target_x_comp=np.ones(counts.size),
                target_x_comp_standard_error=np.full(counts.size, 2e-4),
            ),
            PROTOCOL_C: dummy_primary,
            PROTOCOL_D: dummy_primary,
        },
    )

    fitted = fit_control_populations(
        measurements,
        desc,
        idle,
        covariance_rcond=1e-12,
        force_all_candidates=True,
    )

    assert fitted.success
    assert fitted.selected_model == "G2 x F2"
    for name, expected in true_rates.items():
        assert fitted.rates_per_ns[name] == pytest.approx(expected, rel=2e-3)
        assert fitted.statuses[name] == CrDissipationRateStatus.RESOLVED
    selected = fitted.candidates[fitted.selected_model]
    assert selected.n_observations == 2 * 2 * (counts.size - 1)


def test_target_t1rho_and_leakage_reduced_fits_recover_synthetic_rates() -> None:
    """A/B target estimators recover a free T1rho and two-way exchange."""
    counts = np.array([0, 1, 2, 3, 5, 8, 13], dtype=np.int64)
    desc = descriptor()
    idle = IdleNoiseParameters(45_000.0, 50_000.0)
    t1rho = 1.1e-4
    leakage = 4e-5
    seepage = 2e-5
    x = target_t1rho_trajectory(
        counts,
        0.96,
        t1rho,
        0.08,
        cr_lobe_duration_ns=50.0,
        blank_duration_ns=40.0,
        idle_transverse_rate_per_ns=idle.transverse_rate_per_ns,
    )
    pf = target_exchange_trajectory(
        counts,
        0.01,
        leakage,
        seepage,
        cr_lobe_duration_ns=50.0,
    )
    target_population = np.column_stack(
        [(1.0 - pf) * (1.0 + x) / 2.0, (1.0 - pf) * (1.0 - x) / 2.0, pf]
    )
    control = population_series(np.tile([1.0, 0.0, 0.0], (counts.size, 1)))
    protocol_data = CrDissipationProtocolData(
        control_gef=control,
        target_gef=population_series(target_population),
        target_x_comp=x,
        target_x_comp_standard_error=np.full(counts.size, 2e-4),
    )
    dummy = CrDissipationProtocolData(
        primary_expectation=np.ones(counts.size),
        primary_standard_error=np.full(counts.size, 2e-4),
    )
    measurements = measurements_from_protocol_data(
        counts,
        {
            PROTOCOL_A: protocol_data,
            PROTOCOL_B: protocol_data,
            PROTOCOL_C: dummy,
            PROTOCOL_D: dummy,
        },
    )

    decay = fit_target_t1rho(
        measurements,
        PROTOCOL_A,
        desc,
        idle,
        idle_equivalent_rate_per_ns=idle.transverse_rate_per_ns,
    )
    exchange = fit_target_exchange(
        measurements,
        PROTOCOL_A,
        desc,
        force_all_candidates=True,
    )

    assert decay.status == CrDissipationRateStatus.RESOLVED
    assert decay.rate_per_ns == pytest.approx(t1rho, rel=2e-3)
    assert exchange.selected_model == "L2"
    assert exchange.leakage_rate_per_ns == pytest.approx(leakage, rel=3e-3)
    assert exchange.seepage_rate_per_ns == pytest.approx(seepage, rel=3e-3)


def test_physical_forward_fit_selects_resolved_control_pure_dephasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C physical fit selects a significant free pure-dephasing rate."""
    counts = np.array([0, 1, 2, 3, 5, 8], dtype=np.int64)
    idle = IdleNoiseParameters(float("inf"), float("inf"))
    block = (
        SemanticSegment(
            80.0,
            np.zeros((9, 9), dtype=np.complex128),
            True,
            "CR",
        ),
    )

    true_rate = 7e-4
    fixed_rates = CrNoiseRates(control_e_to_g=2e-5)

    def fake_fit_gls_candidate(**kwargs: Any) -> CandidateFit:
        name = str(kwargs["name"])
        parameter_names = tuple(cast(tuple[str, ...], kwargs["parameter_names"]))

        if name == "zero_additional_dephasing":
            parameters = np.array([1.0, 0.0])
            errors = np.array([0.01, 0.01])
            aicc = 20.0
        elif name == "free_nonnegative" or name == "signed_diagnostic":
            parameters = np.array([true_rate, 1.0, 0.0])
            errors = np.array([5e-5, 0.01, 0.01])
            aicc = 0.0
        else:
            raise AssertionError(f"Unexpected candidate: {name}")

        size = len(parameter_names)

        return CandidateFit(
            name=name,
            success=True,
            parameter_names=parameter_names,
            parameters=parameters,
            covariance=np.diag(np.square(errors)),
            standard_errors=errors,
            jacobian=np.eye(len(counts), size),
            prediction=np.ones(len(counts)),
            chi_squared=0.0,
            reduced_chi_squared=0.0,
            aicc=aicc,
            n_observations=len(counts),
            n_parameters=size,
            message="synthetic",
        )

    monkeypatch.setattr(
        analysis_module,
        "fit_gls_candidate",
        fake_fit_gls_candidate,
    )

    fit = fit_physical_dephasing(
        np.ones(counts.size),
        np.full(counts.size, 2e-5),
        counts,
        block,
        initial_density=state_density("+x", "+x"),
        observable=X_CONTROL,
        control_idle=idle,
        target_idle=idle,
        fixed_rates=fixed_rates,
        fitted_role="control",
    )

    assert fit.success
    assert fit.selected_model == "free_nonnegative"
    assert fit.status == CrDissipationRateStatus.RESOLVED
    assert fit.rate_per_ns == pytest.approx(true_rate)
    assert fit.rate_standard_error_per_ns == pytest.approx(5e-5)
