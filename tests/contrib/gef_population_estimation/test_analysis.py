"""Tests for GEF population estimation analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib.experiment.gef_population_estimation import (
    IQMomentSummary,
    fit_gef_population,
    reconstruct_gef_state_features,
    summarize_iq_shots,
)


def _summary(mean: np.ndarray, covariance: np.ndarray | None = None) -> IQMomentSummary:
    """Build a moment summary for deterministic fit tests."""
    if covariance is None:
        covariance = np.eye(5, dtype=np.float64)
    return IQMomentSummary(
        mean=np.asarray(mean, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        n_shots=100,
    )


def _unknown_means(
    state_features: np.ndarray,
    population: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return S1/S4/S5 feature means for one population vector."""
    h_g, h_e, h_f = state_features
    alpha, beta, gamma = population
    return {
        "s1": alpha * h_g + beta * h_e + gamma * h_f,
        "s4": beta * h_g + gamma * h_e + alpha * h_f,
        "s5": gamma * h_g + alpha * h_e + beta * h_f,
    }


def test_summarize_iq_shots_returns_full_second_moments_and_mean_covariance() -> None:
    """Complex IQ shots should produce the documented five moments and covariance."""
    iq = np.array([1 + 2j, 3 + 4j, -1 + 1j], dtype=np.complex128)
    features = np.column_stack(
        (
            iq.real,
            iq.imag,
            iq.real**2,
            iq.real * iq.imag,
            iq.imag**2,
        )
    )

    result = summarize_iq_shots(iq)

    assert_allclose(result.mean, np.mean(features, axis=0), rtol=1e-12, atol=1e-12)
    assert_allclose(
        result.covariance,
        np.cov(features, rowvar=False, ddof=1) / len(iq),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.n_shots == 3


@pytest.mark.parametrize(
    ("iq", "message"),
    [
        (np.array([1 + 1j]), "at least two"),
        (np.ones((2, 2), dtype=np.complex128), "one-dimensional"),
        (np.array([1 + 1j, complex(np.nan, 0)]), "finite"),
    ],
)
def test_summarize_iq_shots_rejects_invalid_shot_data(
    iq: np.ndarray,
    message: str,
) -> None:
    """Invalid IQ shot arrays should fail before moment estimation."""
    with pytest.raises(ValueError, match=message):
        summarize_iq_shots(iq)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (
            np.array(
                [
                    [1.0, 0.5, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "symmetric",
        ),
        (np.diag([1.0, 1.0, 1.0, 1.0, -0.1]), "positive semidefinite"),
    ],
)
def test_iq_moment_summary_rejects_invalid_covariance(
    covariance: np.ndarray,
    message: str,
) -> None:
    """A moment summary should reject matrices that are not covariances."""
    with pytest.raises(ValueError, match=message):
        IQMomentSummary(
            mean=np.zeros(5),
            covariance=covariance,
            n_shots=100,
        )


@pytest.mark.parametrize("n_shots", [True, 2.5])
def test_iq_moment_summary_rejects_noninteger_shot_counts(
    n_shots: object,
) -> None:
    """A moment summary should require an integer shot count."""
    with pytest.raises(TypeError, match="n_shots"):
        IQMomentSummary(
            mean=np.zeros(5),
            covariance=np.eye(5),
            n_shots=n_shots,  # type: ignore[arg-type]
        )


def test_reconstruct_state_features_is_independent_of_thermal_population() -> None:
    """Six calibration configurations should recover state features for unknown q."""
    state_features = np.array(
        [
            [-2.0, 0.5, 4.5, -0.8, 1.2],
            [0.2, 1.5, 1.7, 0.4, 3.5],
            [2.1, -0.3, 5.0, -0.6, 1.8],
        ]
    )
    h_g, h_e, h_f = state_features
    p = 0.83
    q = 1.0 - p
    summaries = {
        "c1": _summary(p * h_g + q * h_e),
        "c2": _summary(p * h_g + q * h_f),
        "c3": _summary(q * h_g + p * h_e),
        "c4": _summary(q * h_g + p * h_f),
        "c5": _summary(p * h_e + q * h_f),
        "c6": _summary(q * h_e + p * h_f),
    }

    result = reconstruct_gef_state_features(summaries)

    assert_allclose(result, state_features, rtol=1e-12, atol=1e-12)


def test_fit_recovers_population_when_centroids_do_not_separate_g_and_e() -> None:
    """Second moments and permutations should recover populations with equal g/e means."""
    state_features = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 4.0, 0.5, 1.0],
            [2.0, -1.0, 6.0, -1.0, 3.0],
        ]
    )
    expected = np.array([0.17, 0.36, 0.47])
    summaries = {
        name: _summary(mean, np.eye(5) * 1e-3)
        for name, mean in _unknown_means(state_features, expected).items()
    }

    result = fit_gef_population(summaries, state_features)

    assert result.success
    assert_allclose(result.population, expected, rtol=1e-7, atol=1e-8)
    assert_allclose(result.population_unconstrained, expected, rtol=1e-10, atol=1e-10)
    assert result.objective == pytest.approx(0.0, abs=1e-16)
    assert result.design_rank == 3


def test_fit_uses_pseudo_inverse_for_singular_feature_covariance() -> None:
    """Singular covariance directions should not prevent an identifiable fit."""
    state_features = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    expected = np.array([0.2, 0.3, 0.5])
    covariance = np.diag([0.01, 0.01, 0.0, 0.0, 0.0])
    summaries = {
        name: _summary(mean, covariance)
        for name, mean in _unknown_means(state_features, expected).items()
    }

    result = fit_gef_population(summaries, state_features)

    assert result.success
    assert_allclose(result.population, expected, rtol=1e-7, atol=1e-8)


def test_fit_constrains_an_unphysical_estimate_to_the_probability_simplex() -> None:
    """Constrained GLS should return nonnegative populations summing to one."""
    state_features = np.array(
        [
            [-1.0, 0.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 2.0],
            [2.0, -1.0, 5.0, -1.0, 3.0],
        ]
    )
    unphysical = np.array([-0.2, 0.45, 0.75])
    summaries = {
        name: _summary(mean, np.eye(5) * 1e-3)
        for name, mean in _unknown_means(state_features, unphysical).items()
    }

    result = fit_gef_population(summaries, state_features)

    assert result.success
    assert_allclose(result.population_unconstrained, unphysical, rtol=1e-10, atol=1e-10)
    assert np.all(result.population >= -1e-12)
    assert np.all(result.population <= 1.0 + 1e-12)
    assert np.sum(result.population) == pytest.approx(1.0, abs=1e-10)
    assert result.objective > 0.0


def test_fit_rejects_nonidentifiable_state_features() -> None:
    """A readout with no GEF contrast should fail instead of returning an arbitrary fit."""
    state_features = np.ones((3, 5), dtype=np.float64)
    summaries = {
        name: _summary(np.ones(5), np.eye(5) * 1e-3) for name in ("s1", "s4", "s5")
    }

    with pytest.raises(ValueError, match="not identifiable"):
        fit_gef_population(summaries, state_features)


@pytest.mark.parametrize(
    ("covariance_rcond", "error"),
    [
        (False, TypeError),
        (-1e-3, ValueError),
        (1.0, ValueError),
        (float("nan"), ValueError),
    ],
)
def test_fit_rejects_invalid_covariance_cutoffs(
    covariance_rcond: object,
    error: type[Exception],
) -> None:
    """The covariance cutoff should be a finite real in the half-open unit interval."""
    state_features = np.eye(3, 5)
    summaries = {
        name: _summary(mean)
        for name, mean in _unknown_means(
            state_features,
            np.array([0.2, 0.3, 0.5]),
        ).items()
    }

    with pytest.raises(error, match="covariance_rcond"):
        fit_gef_population(
            summaries,
            state_features,
            covariance_rcond=covariance_rcond,  # type: ignore[arg-type]
        )
