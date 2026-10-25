"""Synthetic tests for readout-MIST transition-matrix inference."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest

import qubex.contrib.experiment.readout_mist as mist_module
from qubex.contrib.experiment.readout_mist import (
    TRANSITION_ORDER,
    GefPopulationSeries,
    ReadoutMistConditionMeasurements,
    ReadoutMistMeasurements,
    _failed_fit,
    _matrix_figure,
    _population_figure,
    _profile_upper_bound,
    _transition_matrix_standard_error,
    analyze_readout_mist,
)


def _synthetic_measurements() -> tuple[ReadoutMistMeasurements, np.ndarray]:
    """Build noiseless three-state trajectories with finite rank-two covariance."""
    transition = np.array(
        [
            [0.984, 0.012, 0.007],
            [0.010, 0.978, 0.023],
            [0.006, 0.010, 0.970],
        ],
        dtype=np.float64,
    )
    return _measurements_from_transition(transition), transition


def _measurements_from_transition(
    transition: np.ndarray,
) -> ReadoutMistMeasurements:
    """Build deterministic trajectories for one supplied transition matrix."""
    initial: dict[str, np.ndarray] = {
        "initial_g": np.array([0.985, 0.012, 0.003], dtype=np.float64),
        "initial_e": np.array([0.018, 0.971, 0.011], dtype=np.float64),
        "initial_f": np.array([0.012, 0.026, 0.962], dtype=np.float64),
    }
    counts = np.array([0, 1, 2, 4, 8, 16, 32], dtype=np.int64)
    tangent_covariance = 2e-8 * np.array(
        [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]
    )
    protocols = {}
    for name, population_0 in initial.items():
        values = np.asarray(
            [np.linalg.matrix_power(transition, int(n)) @ population_0 for n in counts],
            dtype=np.float64,
        )
        covariance = np.repeat(tangent_covariance[None, :, :], counts.size, axis=0)
        protocols[name] = ReadoutMistConditionMeasurements(
            actual=GefPopulationSeries(
                population=values,
                covariance=covariance,
                standard_error=np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)),
                population_unconstrained=values.copy(),
            ),
            reference=None,
        )
    return ReadoutMistMeasurements(counts, protocols)


def test_joint_fit_recovers_transition_matrix_and_initial_populations() -> None:
    """The full fit should recover the matrix and its named transition convention."""
    measurements, expected = _synthetic_measurements()

    analysis = analyze_readout_mist(measurements, covariance_rcond=1e-12)

    np.testing.assert_allclose(
        analysis.actual.transition_matrix,
        expected,
        rtol=2e-4,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        analysis.actual.transition_matrix.sum(axis=0),
        np.ones(3),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(analysis.actual.transition_matrix >= 0.0)
    assert analysis.actual.success
    assert analysis.actual.jacobian_rank == analysis.actual.free_parameter_count
    assert all(analysis.actual.parameter_estimable.values())
    goodness = analysis.actual.goodness_of_fit
    assert goodness.weighted_residual_rms == pytest.approx(
        np.sqrt(goodness.reduced_chi_squared)
    )
    assert analysis.metadata["reference_interpretation"] is None
    assert analysis.metadata["profile_confidence_level"] is None
    np.testing.assert_allclose(
        [
            analysis.actual.transition_probabilities["g_to_e"].value,
            analysis.actual.transition_probabilities["e_to_g"].value,
            analysis.actual.transition_probabilities["g_to_f"].value,
            analysis.actual.transition_probabilities["e_to_f"].value,
            analysis.actual.transition_probabilities["f_to_g"].value,
            analysis.actual.transition_probabilities["f_to_e"].value,
        ],
        [
            expected[1, 0],
            expected[0, 1],
            expected[2, 0],
            expected[2, 1],
            expected[0, 2],
            expected[1, 2],
        ],
        rtol=1e-12,
        atol=1e-14,
    )
    assert analysis.actual.state_order == ("g", "e", "f")
    assert analysis.actual.probability_unit == "1/readout_cycle"
    assert tuple(analysis.actual.transition_probabilities) == TRANSITION_ORDER
    assert TRANSITION_ORDER == (
        "g_to_e",
        "e_to_g",
        "g_to_f",
        "e_to_f",
        "f_to_g",
        "f_to_e",
    )


def test_reduced_model_comparison_reports_candidate_aicc_values() -> None:
    """Reduced-model analysis should report candidates without selecting one."""
    transition = np.array(
        [
            [0.990, 0.020, 0.000],
            [0.010, 0.970, 0.030],
            [0.000, 0.010, 0.970],
        ],
        dtype=np.float64,
    )
    measurements = _measurements_from_transition(transition)

    analysis = analyze_readout_mist(
        measurements,
        fit_reduced_models=True,
    )

    comparison = analysis.actual_model_comparison
    assert comparison is not None
    assert set(comparison.candidates) == {"full", "adjacent_only", "no_leakage"}
    assert comparison.candidates["full"].fit is analysis.actual
    assert (
        min(
            comparison.candidates.values(),
            key=lambda candidate: candidate.aicc,
        ).fit.model_name
        == "adjacent_only"
    )
    adjacent = comparison.candidates["adjacent_only"].fit
    assert not adjacent.transition_probabilities["g_to_f"].estimable
    assert adjacent.transition_probabilities["g_to_f"].standard_error is None


def test_profile_upper_bound_solves_a_known_quadratic_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile search should find the one-sided threshold for a known objective."""
    measurements, _ = _synthetic_measurements()
    full_fit = analyze_readout_mist(measurements).actual
    estimate = full_fit.transition_probabilities["g_to_f"].value
    scale = 0.02

    def fit_quadratic_profile(*args: object, **kwargs: Any):
        spec = kwargs["spec"]
        probability = dict(spec.fixed_probabilities)["g_to_f"]
        return replace(
            full_fit,
            objective=full_fit.objective + ((probability - estimate) / scale) ** 2,
        )

    monkeypatch.setattr(mist_module, "_fit_model_condition", fit_quadratic_profile)

    upper_bound, reason = _profile_upper_bound(
        measurements,
        reference=False,
        covariance_rcond=1e-12,
        full_fit=full_fit,
        transition_name="g_to_f",
        confidence_level=0.95,
    )

    assert reason is None
    assert upper_bound == pytest.approx(estimate + 0.032897, abs=2e-6)


def test_profile_fit_failure_does_not_create_a_false_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed constrained fit should make the profile bound unavailable."""
    measurements, _ = _synthetic_measurements()
    full_fit = analyze_readout_mist(measurements).actual
    monkeypatch.setattr(
        mist_module,
        "_fit_model_condition",
        lambda *args, **kwargs: _failed_fit("synthetic failure"),
    )

    upper_bound, reason = _profile_upper_bound(
        measurements,
        reference=False,
        covariance_rcond=1e-12,
        full_fit=full_fit,
        transition_name="g_to_f",
        confidence_level=0.95,
    )

    assert upper_bound is None
    assert reason == "profile_fit_failed"


def test_population_figure_evaluates_fit_at_every_integer_repetition() -> None:
    """Fit lines should use every integer repetition through the largest count."""
    measurements, _ = _synthetic_measurements()
    analysis = analyze_readout_mist(measurements)

    figure = _population_figure(measurements, analysis)
    traces = cast(tuple[go.Scatter, ...], figure.data)

    fit_trace = next(trace for trace in traces if trace.name == "fit Pg")
    np.testing.assert_array_equal(
        np.asarray(fit_trace.x),
        np.arange(int(measurements.repetition_counts[-1]) + 1),
    )
    assert "WRMS=" in figure.layout.title.text


def test_matrix_figure_compares_actual_and_reference_with_uncertainty() -> None:
    """Matrix panels should share a 0.1-saturated scale and show available errors."""
    measurements, _ = _synthetic_measurements()
    with_reference = ReadoutMistMeasurements(
        measurements.repetition_counts,
        {
            name: ReadoutMistConditionMeasurements(
                actual=condition.actual,
                reference=condition.actual,
            )
            for name, condition in measurements.protocols.items()
        },
    )
    analysis = analyze_readout_mist(with_reference)

    figure = _matrix_figure(analysis)
    traces = cast(tuple[go.Heatmap, ...], figure.data)
    errors = _transition_matrix_standard_error(analysis.actual)
    covariance = analysis.actual.parameter_covariance
    expected_g_stay_error = np.sqrt(np.sum(covariance[np.ix_([0, 2], [0, 2])]))

    assert len(traces) == 2
    assert analysis.metadata["reference_interpretation"] == (
        "duration_matched_blank_cycle_diagnostic"
    )
    assert figure.layout.coloraxis.cmin == pytest.approx(0.0)
    assert figure.layout.coloraxis.cmax == pytest.approx(0.1)
    annotation_text = [annotation.text for annotation in figure.layout.annotations]
    assert annotation_text[0].startswith("Actual<br><sup>optimizer=success, χ²/DoF=")
    assert annotation_text[1].startswith("Reference<br><sup>optimizer=success, χ²/DoF=")
    for trace in traces:
        text = np.asarray(trace.text)
        assert "±" in text[0, 0]
        assert "±" in text[1, 0]
    assert errors[0, 0] == pytest.approx(expected_g_stay_error)
    assert errors[1, 0] == pytest.approx(
        analysis.actual.transition_probabilities["g_to_e"].standard_error
    )


def test_matrix_figure_marks_a_failed_reference_fit_as_unavailable() -> None:
    """A failed reference fit should not produce a meaningless NaN heatmap."""
    measurements, _ = _synthetic_measurements()
    analysis = analyze_readout_mist(measurements)
    failed_reference = _failed_fit("synthetic failure")

    figure = _matrix_figure(replace(analysis, reference=failed_reference))
    traces = cast(tuple[go.Heatmap, ...], figure.data)

    assert len(traces) == 1
    assert any(
        annotation.text == "Fit unavailable" for annotation in figure.layout.annotations
    )


def test_population_series_rejects_inconsistent_shapes() -> None:
    """Population records should reject covariance arrays on another grid."""
    with pytest.raises(ValueError, match="covariance"):
        GefPopulationSeries(
            population=np.full((2, 3), 1.0 / 3.0),
            covariance=np.zeros((1, 3, 3)),
            standard_error=np.zeros((2, 3)),
            population_unconstrained=np.full((2, 3), 1.0 / 3.0),
        )


def test_measurements_reject_partial_reference_data() -> None:
    """Reference trajectories should be present for every protocol or none."""
    measurements, _ = _synthetic_measurements()
    protocols = dict(measurements.protocols)
    initial_g = protocols["initial_g"]
    protocols["initial_g"] = ReadoutMistConditionMeasurements(
        actual=initial_g.actual,
        reference=initial_g.actual,
    )

    with pytest.raises(ValueError, match="all protocols or none"):
        ReadoutMistMeasurements(measurements.repetition_counts, protocols)


def test_failed_fit_keeps_the_parameter_covariance_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fit should retain labels for every covariance row and column."""
    measurements, _ = _synthetic_measurements()
    monkeypatch.setattr(mist_module, "_run_fit", lambda *args, **kwargs: None)

    fit = analyze_readout_mist(measurements).actual

    assert not fit.success
    assert fit.parameter_covariance.shape == (
        len(fit.parameter_order),
        len(fit.parameter_order),
    )
    assert fit.parameter_order[: len(TRANSITION_ORDER)] == TRANSITION_ORDER


@pytest.mark.parametrize("value", [True, -1.0, 1.0, float("nan")])
def test_analysis_rejects_invalid_covariance_cutoff(value: object) -> None:
    """The analysis cutoff should be a finite real value in the half-open unit interval."""
    measurements, _ = _synthetic_measurements()

    with pytest.raises((TypeError, ValueError), match="covariance_rcond"):
        analyze_readout_mist(
            measurements,
            covariance_rcond=value,  # type: ignore[arg-type]
        )
