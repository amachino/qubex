"""Tests for CR-pulse coherence characterization fitting helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from qubex.contrib.experiment.cr_pulse_coherence_characterization import (
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


def test_three_level_fit_recovers_four_independent_adjacent_rates() -> None:
    """A joint A/B fit should recover independent GE and EF transition rates."""
    times = np.array([0, 800, 1600, 3000, 5000, 8000, 13_000, 21_000.0])
    rates = (2.5e-5, 4.0e-6, 4.2e-5, 7.0e-6)
    initial_a = np.array([0.97, 0.02, 0.01])
    initial_b = np.array([0.03, 0.94, 0.03])

    fit = fit_three_level_rate_model(
        times,
        _trajectory(times, initial_a, rates),
        _trajectory(times, initial_b, rates),
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
    assert_allclose(fit.fitted_a, _trajectory(times, initial_a, rates), atol=1e-7)
    assert_allclose(fit.fitted_b, _trajectory(times, initial_b, rates), atol=1e-7)


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
