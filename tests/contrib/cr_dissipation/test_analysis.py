"""Fast synthetic tests for the CR dissipation v11 physical estimators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import qubex.contrib.experiment._cr_dissipation.analysis as analysis_module
from qubex.contrib.experiment._cr_dissipation.analysis import (
    _aggregate_value_error,
    _change_detected,
    _fidelity_limits,
    _fit_idle_control_rates,
    _fit_idle_target_exchange,
    _fit_idle_target_t1rho,
    _pure_dephasing_component,
    _select_significant_nested_candidate,
    _signed_rate_difference,
    _simplex_boundary_override,
    _target_control_state_dependence_warnings,
    control_population_trajectory,
    fit_control_populations,
    fit_physical_dephasing,
    fit_target_exchange,
    fit_target_t1rho,
    nonnormalized_ge_expectation,
    simplex_boundary_bootstrap_required,
    target_exchange_trajectory,
    target_t1rho_trajectory,
)
from qubex.contrib.experiment._cr_dissipation.fit import (
    fit_profiled_affine_candidate,
)
from qubex.contrib.experiment._cr_dissipation.pulses import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    PROTOCOL_D,
    ProtocolSchedules,
    ZX90Descriptor,
)
from qubex.contrib.experiment._cr_dissipation.simulation import (
    X_CONTROL,
    CrNoiseRates,
    SemanticSegment,
    state_density,
)
from qubex.contrib.experiment._cr_dissipation.types import (
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


@pytest.mark.parametrize(
    ("constrained", "unconstrained", "expected"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.001, 0.002], True),
        ([0.0, 0.0, 0.0], [0.0, 0.004, 0.0], False),
        ([0.01, 0.0, 0.0], [0.01, 0.001, 0.002], False),
    ],
)
def test_simplex_boundary_bootstrap_is_requested_only_when_unconstrained_data_are_flat(
    constrained: list[float],
    unconstrained: list[float],
    expected: bool,
) -> None:
    """Bootstrap is reserved for an unresolved simplex-boundary diagnostic."""
    values = np.column_stack(
        (
            np.full(3, 0.5),
            np.full(3, 0.5),
            constrained,
        )
    )
    unconstrained_values = values.copy()
    unconstrained_values[:, 2] = unconstrained
    series = population_series(values)

    assert (
        simplex_boundary_bootstrap_required(
            replace(series, population_unconstrained=unconstrained_values),
            2,
            minimum_change=0.003,
        )
        is expected
    )


def test_change_detection_uses_independent_material_and_significance_maxima() -> None:
    """Material and significance maxima may occur at different points."""
    values = np.array([0.0, 0.02, 0.001])
    reference = np.zeros(3)
    errors = np.array([0.001, 0.1, 0.0001])

    assert _change_detected(
        values,
        errors,
        reference,
        minimum_change=0.01,
        include_initial_error=False,
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
            "control_g_to_e": CrDissipationRateStatus.PARTIALLY_UNRESOLVED,
            "control_e_to_f": CrDissipationRateStatus.CONSISTENT_WITH_ZERO,
            "control_f_to_e": CrDissipationRateStatus.INCONSISTENT_RATE_DECOMPOSITION,
            "target_t1rho": CrDissipationRateStatus.PARTIALLY_UNRESOLVED,
            "control_transverse": CrDissipationRateStatus.PARTIALLY_UNRESOLVED,
            "target_t2rho": CrDissipationRateStatus.PARTIALLY_UNRESOLVED,
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

    propagated.clear()
    disabled, _ = _fidelity_limits(
        descriptor(),
        IdleNoiseParameters(float("inf"), float("inf")),
        IdleNoiseParameters(float("inf"), float("inf")),
        nominal,
        statuses,
        covariance,
        primitive_order,
        compute_uncertainty=False,
    )
    assert not propagated
    assert disabled.cr_on_coherence_limited.standard_error is None
    assert disabled.cr_on_coherence_limited.message == (
        "Fidelity uncertainty calculation was disabled."
    )


def test_pure_dephasing_component_uses_both_longitudinal_directions() -> None:
    """Control transverse decomposition includes upward and downward rates."""
    assert _pure_dephasing_component(8e-5, 6e-5) == pytest.approx(5e-5)
    assert _pure_dephasing_component(2e-5, 6e-5) == 0.0
    assert np.isnan(_pure_dephasing_component(float("nan"), 6e-5))
    assert np.isnan(_pure_dephasing_component(8e-5, float("nan")))


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


def test_idle_baseline_precedes_actual_total_t1rho_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual selection starts only after the one-pass baseline is available."""
    events: list[object] = []
    baseline = SimpleNamespace()

    def extract(*_args: Any, **_kwargs: Any) -> Any:
        events.append("baseline")
        return baseline

    def control_fit(*_args: Any, **_kwargs: Any) -> Any:
        events.append("control")
        return SimpleNamespace()

    def target_fit(*_args: Any, **_kwargs: Any) -> Any:
        events.append("target_t1rho")
        raise RuntimeError("stop after observing the first actual T1rho fit")

    def exchange_fit(*_args: Any, **_kwargs: Any) -> Any:
        events.append("target_exchange")
        return SimpleNamespace(success=True)

    monkeypatch.setattr(analysis_module, "_extract_idle_equivalent_baseline", extract)
    monkeypatch.setattr(analysis_module, "fit_control_populations", control_fit)
    monkeypatch.setattr(analysis_module, "fit_target_exchange", exchange_fit)
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

    assert events == [
        "baseline",
        "control",
        "target_exchange",
        "target_exchange",
        "target_t1rho",
    ]


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
        "qubex.contrib.experiment._cr_dissipation.analysis.fit_gls_candidate",
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
        "qubex.contrib.experiment._cr_dissipation.analysis.fit_gls_candidate",
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


def test_exchange_fit_keeps_initial_rates_inside_long_duration_bounds() -> None:
    """Long CR lobes should not make otherwise identifiable fits fail at startup."""
    counts = np.array([0, 1, 2, 3, 5, 8], dtype=np.int64)
    cr_lobe_duration_ns = 100_000_000.0
    leakage_rate = 1e-9
    pf = target_exchange_trajectory(
        counts,
        0.0,
        leakage_rate,
        0.0,
        cr_lobe_duration_ns=cr_lobe_duration_ns,
    )
    target = population_series(
        np.column_stack((0.5 * (1.0 - pf), 0.5 * (1.0 - pf), pf)),
        error=1e-4,
    )
    control = population_series(np.tile([0.9, 0.1, 0.0], (counts.size, 1)))
    ab = CrDissipationProtocolData(control_gef=control, target_gef=target)
    cd = CrDissipationProtocolData(
        primary_expectation=np.ones(counts.size),
        primary_standard_error=np.full(counts.size, 1e-4),
    )
    measurements = measurements_from_protocol_data(
        counts,
        {PROTOCOL_A: ab, PROTOCOL_B: ab, PROTOCOL_C: cd, PROTOCOL_D: cd},
        cr_active_duration_ns=8.0 * cr_lobe_duration_ns,
    )
    long_descriptor = replace(descriptor(), cr_lobe_duration_ns=cr_lobe_duration_ns)

    fitted = fit_target_exchange(
        measurements,
        PROTOCOL_A,
        long_descriptor,
        force_all_candidates=True,
    )

    assert fitted.success
    assert fitted.leakage_rate_per_ns == pytest.approx(leakage_rate, rel=0.05)


def test_clear_target_dynamics_fail_when_only_null_candidate_is_identifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear leakage dynamics cannot fall back to an identifiable null fit."""
    counts = np.array([0, 1, 2, 3, 5], dtype=np.int64)
    target_values = np.column_stack(
        (
            0.5 - np.array([0.0, 0.01, 0.02, 0.03, 0.04]),
            np.full(counts.size, 0.5),
            np.array([0.0, 0.01, 0.02, 0.03, 0.04]),
        )
    )
    control = population_series(np.tile([0.9, 0.1, 0.0], (counts.size, 1)))
    ab = CrDissipationProtocolData(
        control_gef=control,
        target_gef=population_series(target_values, error=1e-4),
    )
    cd = CrDissipationProtocolData(
        primary_expectation=np.ones(counts.size),
        primary_standard_error=np.full(counts.size, 1e-4),
    )
    measurements = measurements_from_protocol_data(
        counts,
        {PROTOCOL_A: ab, PROTOCOL_B: ab, PROTOCOL_C: cd, PROTOCOL_D: cd},
    )

    def fake_fit(**kwargs: Any) -> CandidateFit:
        name = str(kwargs["name"])
        names = tuple(cast(tuple[str, ...], kwargs["parameter_names"]))
        candidate = _candidate(
            name,
            names,
            tuple(1e-5 for _ in names),
            tuple(1e-6 for _ in names),
            0.0,
        )
        return candidate if not names else replace(candidate, success=False, aicc=None)

    monkeypatch.setattr(analysis_module, "fit_gls_candidate", fake_fit)

    fitted = fit_target_exchange(measurements, PROTOCOL_A, descriptor())

    assert not fitted.success
    assert fitted.selected_model == "fit_failed"
    assert fitted.leakage_status == CrDissipationRateStatus.FIT_FAILED
    assert fitted.seepage_status == CrDissipationRateStatus.FIT_FAILED


def test_clear_control_dynamics_fail_when_only_null_candidate_is_identifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear control dynamics cannot fall back to an identifiable null fit."""
    counts = np.array([0, 1, 2, 3, 5], dtype=np.int64)
    pe = np.array([0.1, 0.13, 0.16, 0.19, 0.22])
    control = population_series(
        np.column_stack((1.0 - pe, pe, np.zeros(counts.size))), error=1e-4
    )
    target = population_series(np.tile([0.5, 0.5, 0.0], (counts.size, 1)))
    ab = CrDissipationProtocolData(control_gef=control, target_gef=target)
    cd = CrDissipationProtocolData(
        primary_expectation=np.ones(counts.size),
        primary_standard_error=np.full(counts.size, 1e-4),
    )
    measurements = measurements_from_protocol_data(
        counts,
        {PROTOCOL_A: ab, PROTOCOL_B: ab, PROTOCOL_C: cd, PROTOCOL_D: cd},
    )

    def fake_fit(**kwargs: Any) -> CandidateFit:
        name = str(kwargs["name"])
        names = tuple(cast(tuple[str, ...], kwargs["parameter_names"]))
        candidate = _candidate(
            name,
            names,
            tuple(1e-5 for _ in names),
            tuple(1e-6 for _ in names),
            0.0,
        )
        return candidate if not names else replace(candidate, success=False, aicc=None)

    monkeypatch.setattr(analysis_module, "fit_gls_candidate", fake_fit)

    fitted = fit_control_populations(
        measurements,
        descriptor(),
        IdleNoiseParameters(50_000.0, 40_000.0),
        covariance_rcond=1e-12,
    )

    assert not fitted.success
    assert fitted.selected_model == "fit_failed"
    assert set(fitted.statuses.values()) == {CrDissipationRateStatus.FIT_FAILED}


def test_idle_baseline_fits_return_nan_without_usable_observations() -> None:
    """Empty idle-fit datasets are unavailable rather than exact boundaries."""
    counts = np.array([0, 1, 2, 3, 5], dtype=np.int64)
    values = np.tile([0.9, 0.1, 0.0], (counts.size, 1))
    unavailable = GefPopulationSeries(
        values,
        np.full((counts.size, 3, 3), np.nan),
        np.full(values.shape, np.nan),
        values.copy(),
    )
    target = population_series(np.tile([0.5, 0.5, 0.0], (counts.size, 1)))
    ab = CrDissipationProtocolData(control_gef=unavailable, target_gef=target)
    cd = CrDissipationProtocolData(
        primary_expectation=np.ones(counts.size),
        primary_standard_error=np.full(counts.size, 1e-4),
    )
    measurements = measurements_from_protocol_data(
        counts,
        {PROTOCOL_A: ab, PROTOCOL_B: ab, PROTOCOL_C: cd, PROTOCOL_D: cd},
    )
    idle = IdleNoiseParameters(50_000.0, 40_000.0)
    synthetic = {PROTOCOL_A: values, PROTOCOL_B: values}
    active_time = counts.astype(float) * 400.0

    control_rates = _fit_idle_control_rates(
        measurements,
        descriptor(),
        idle,
        synthetic,
        covariance_rcond=1e-12,
    )
    t1rho = _fit_idle_target_t1rho(
        np.ones(counts.size),
        np.full(counts.size, np.nan),
        counts,
        descriptor(),
        idle,
        active_time,
    )
    leakage, seepage = _fit_idle_target_exchange(
        np.zeros(counts.size),
        np.full(counts.size, np.nan),
        counts,
        descriptor(),
        active_time,
    )

    assert all(np.isnan(value) for value in control_rates.values())
    assert np.isnan(t1rho)
    assert np.isnan(leakage)
    assert np.isnan(seepage)


def descriptor() -> ZX90Descriptor:
    """Return a timing-only descriptor suitable for classical fits."""
    schedule = PulseSchedule()
    pi = FlatTop(duration=40.0, amplitude=0.2, tau=4.0)
    return ZX90Descriptor(
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


def test_protocol_d_z_observable_keeps_the_pg_minus_pe_sign() -> None:
    """Direct D Z readout uses Pg-Pe without initial-sign normalization."""
    population = np.array([[0.2, 0.7, 0.1], [0.1, 0.5, 0.4]])
    covariance = np.tile(np.eye(3) * 1e-4, (2, 1, 1))
    series = GefPopulationSeries(
        population,
        covariance,
        np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)),
        population.copy(),
    )

    values, _, sign = nonnormalized_ge_expectation(
        series,
        align_initial_sign=False,
    )

    assert sign == 1.0
    np.testing.assert_allclose(values, [-0.5, -0.4])


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
    t1rho = idle.transverse_rate_per_ns
    leakage = 4e-5
    seepage = 2e-5
    x = target_t1rho_trajectory(
        counts,
        0.96,
        t1rho,
        0.08,
        initial_pf=0.01,
        leakage_rate_per_ns=leakage,
        seepage_rate_per_ns=seepage,
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

    exchange = fit_target_exchange(
        measurements,
        PROTOCOL_A,
        desc,
        force_all_candidates=True,
    )
    decay = fit_target_t1rho(
        measurements,
        PROTOCOL_A,
        desc,
        idle,
        exchange,
    )
    uncoupled_decay = fit_target_t1rho(
        measurements,
        PROTOCOL_A,
        desc,
        idle,
        replace(exchange, leakage_rate_per_ns=0.0, seepage_rate_per_ns=0.0),
    )

    assert decay.status == CrDissipationRateStatus.RESOLVED
    assert decay.selected_model == "free"
    assert decay.rate_per_ns == pytest.approx(t1rho, rel=2e-3)
    assert abs(decay.rate_per_ns - t1rho) < abs(uncoupled_decay.rate_per_ns - t1rho)
    assert exchange.selected_model == "L2"
    assert exchange.leakage_rate_per_ns == pytest.approx(leakage, rel=3e-3)
    assert exchange.seepage_rate_per_ns == pytest.approx(seepage, rel=3e-3)


def test_symmetric_target_leakage_does_not_mimic_t1rho_without_seepage() -> None:
    """Symmetric leakage alone leaves normalized computational X unchanged."""
    counts = np.array([0, 1, 2, 5, 8], dtype=np.int64)

    values = target_t1rho_trajectory(
        counts,
        0.93,
        0.0,
        0.0,
        initial_pf=0.02,
        leakage_rate_per_ns=8e-5,
        seepage_rate_per_ns=0.0,
        cr_lobe_duration_ns=50.0,
        blank_duration_ns=0.0,
        idle_transverse_rate_per_ns=0.0,
    )

    np.testing.assert_allclose(values, 0.93, rtol=0.0, atol=1e-12)


def test_t1rho_fit_recovers_zero_from_symmetric_leakage_only() -> None:
    """A fixed leakage-only Pf model does not bias fitted T1rho above zero."""
    counts = np.array([0, 1, 2, 3, 5, 8, 13], dtype=np.int64)
    desc = descriptor()
    idle = IdleNoiseParameters(float("inf"), float("inf"))
    leakage = 8e-5
    initial_pf = 0.02
    x = target_t1rho_trajectory(
        counts,
        0.93,
        0.0,
        0.0,
        initial_pf=initial_pf,
        leakage_rate_per_ns=leakage,
        seepage_rate_per_ns=0.0,
        cr_lobe_duration_ns=50.0,
        blank_duration_ns=40.0,
        idle_transverse_rate_per_ns=0.0,
    )
    pf = target_exchange_trajectory(
        counts,
        initial_pf,
        leakage,
        0.0,
        cr_lobe_duration_ns=50.0,
    )
    target_population = np.column_stack(
        [(1.0 - pf) * (1.0 + x) / 2.0, (1.0 - pf) * (1.0 - x) / 2.0, pf]
    )
    protocol_data = CrDissipationProtocolData(
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
    exchange = fit_target_exchange(
        measurements,
        PROTOCOL_A,
        desc,
        force_all_candidates=True,
    )

    decay = fit_target_t1rho(
        measurements,
        PROTOCOL_A,
        desc,
        idle,
        exchange,
    )

    assert exchange.leakage_rate_per_ns == pytest.approx(leakage, rel=3e-3)
    assert decay.rate_per_ns == pytest.approx(0.0, abs=1e-9)


def test_pure_dephasing_delta_is_signed_relative_to_idle_equivalent() -> None:
    """The derived pure-dephasing comparison preserves either sign."""
    assert _signed_rate_difference(2e-5, 2e-5) == pytest.approx(0.0, abs=1e-15)
    assert _signed_rate_difference(1e-5, 2e-5) == pytest.approx(-1e-5)


def test_profiled_affine_fit_recovers_one_nonlinear_rate() -> None:
    """Variable projection recovers rate, contrast, and offset."""
    times = np.arange(10, dtype=np.float64)
    true_rate = 0.18
    values = 0.72 * np.exp(-true_rate * times) + 0.11
    fit = fit_profiled_affine_candidate(
        name="synthetic",
        simulator=lambda rate: np.exp(-rate * times),
        values=values,
        standard_errors=np.full(times.size, 0.002),
        initial_rate=0.1,
        rate_bounds=(0.0, 1.0),
    )

    assert fit.success
    assert fit.parameters == pytest.approx((true_rate, 0.72, 0.11), rel=1e-5)
    assert fit.jacobian.shape == (times.size, 3)


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

    def fake_profiled_fit(**kwargs: Any) -> CandidateFit:
        name = str(kwargs["name"])
        parameter_names = (
            ("spam_amplitude", "spam_offset")
            if kwargs.get("fixed_rate") is not None
            else (
                "pure_dephasing_rate_per_ns",
                "spam_amplitude",
                "spam_offset",
            )
        )

        if name == "zero_total_dephasing":
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
        "fit_profiled_affine_candidate",
        fake_profiled_fit,
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


def test_physical_forward_fit_selects_zero_total_pure_dephasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C zero model means the full CR-active pure-dephasing rate is zero."""
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

    def fake_profiled_fit(**kwargs: Any) -> CandidateFit:
        name = str(kwargs["name"])
        names = (
            ("spam_amplitude", "spam_offset")
            if kwargs.get("fixed_rate") is not None
            else (
                "pure_dephasing_rate_per_ns",
                "spam_amplitude",
                "spam_offset",
            )
        )
        if name == "zero_total_dephasing":
            return _candidate(name, names, (1.0, 0.0), (0.01, 0.01), 0.0)
        return _candidate(
            name,
            names,
            (0.0, 1.0, 0.0),
            (1e-5, 0.01, 0.01),
            20.0,
        )

    monkeypatch.setattr(
        analysis_module, "fit_profiled_affine_candidate", fake_profiled_fit
    )
    monkeypatch.setattr(
        analysis_module,
        "simulate_repeated_observable",
        lambda *_args, **_kwargs: np.ones(counts.size),
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
        fixed_rates=CrNoiseRates(),
        fitted_role="control",
    )

    assert fit.selected_model == "zero_total_dephasing"
    assert fit.status == CrDissipationRateStatus.CONSISTENT_WITH_ZERO
    assert fit.rate_per_ns == 0.0


def test_negative_signed_diagnostic_projects_to_the_zero_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inconsistent signed diagnostic should use the zero model and its SPAM fit."""
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

    def fake_profiled_fit(**kwargs: Any) -> CandidateFit:
        name = str(kwargs["name"])
        if name == "zero_total_dephasing":
            return _candidate(
                name,
                ("spam_amplitude", "spam_offset"),
                (0.8, 0.1),
                (0.01, 0.01),
                20.0,
            )
        if name == "free_nonnegative":
            return _candidate(
                name,
                (
                    "pure_dephasing_rate_per_ns",
                    "spam_amplitude",
                    "spam_offset",
                ),
                (1e-3, 1.2, 0.2),
                (1e-4, 0.01, 0.01),
                0.0,
            )
        return _candidate(
            name,
            (
                "pure_dephasing_rate_per_ns",
                "spam_amplitude",
                "spam_offset",
            ),
            (-1e-3, 0.7, 0.05),
            (1e-4, 0.01, 0.01),
            0.0,
        )

    monkeypatch.setattr(
        analysis_module, "fit_profiled_affine_candidate", fake_profiled_fit
    )
    monkeypatch.setattr(
        analysis_module,
        "simulate_repeated_observable",
        lambda *_args, **_kwargs: np.ones(counts.size),
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
        fixed_rates=CrNoiseRates(),
        fitted_role="control",
    )

    assert fit.status == CrDissipationRateStatus.INCONSISTENT_RATE_DECOMPOSITION
    assert fit.selected_model == "zero_total_dephasing"
    assert fit.rate_per_ns == 0.0
    assert fit.spam_amplitude == pytest.approx(0.8)
    assert fit.spam_offset == pytest.approx(0.1)
    assert fit.fitted_values == pytest.approx(np.full(counts.size, 0.9))
