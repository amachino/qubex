"""Tests for CR-pulse coherence analysis helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from qubex.contrib.experiment import cr_pulse_coherence_analysis as module
from qubex.contrib.experiment.cr_pulse_coherence_analysis import (
    CrPulseCoherenceMeasurements,
    analyze_cr_pulse_coherence,
    fit_exponential_decay,
    fit_three_level_rate_model,
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
        cr_pulse_counts=(0, 4, 8),
        times={"control_t2_echo": times},
        sequence_durations={
            "control_t2_echo": times + 2.0,
            "control_t2_echo_reference": times + 2.0,
        },
        populations={},
        population_standard_errors={},
        target_polarizations={},
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


def test_offline_analysis_validates_cross_field_measurement_structure() -> None:
    """Malformed saved data should fail with a clear structural error."""
    measurements = _control_echo_measurements()

    with pytest.raises(ValueError, match=r"cr_pulse_counts must equal 4 \* n_values"):
        analyze_cr_pulse_coherence(replace(measurements, cr_pulse_counts=(0, 5, 8)))

    with pytest.raises(ValueError, match="actual and reference sequence durations"):
        analyze_cr_pulse_coherence(
            replace(
                measurements,
                sequence_durations={
                    **measurements.sequence_durations,
                    "control_t2_echo_reference": np.array([2.0, 12.0, 23.0]),
                },
            )
        )

    with pytest.raises(TypeError, match="must be a NumPy array"):
        analyze_cr_pulse_coherence(
            replace(measurements, times={"control_t2_echo": [0.0, 10.0, 20.0]})  # type: ignore[dict-item]
        )


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
    assert fit.gamma_ge_down == pytest.approx(rates[0], rel=1e-4)
    assert fit.gamma_ge_up == pytest.approx(rates[1], rel=1e-4)


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
