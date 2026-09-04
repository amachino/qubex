"""Tests for GEF population bootstrap uncertainty estimation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib.experiment.gef_population_estimation import (
    GefPopulationBootstrap,
    GefPopulationCalibration,
    bootstrap_gef_populations,
    reconstruct_gef_state_features,
    summarize_iq_shots,
)


def _state_samples() -> dict[str, np.ndarray]:
    """Return deterministic state-specific IQ samples."""
    rng = np.random.default_rng(2609)
    return {
        "g": rng.normal(-2.0, 0.30, 40) + 1j * rng.normal(0.2, 0.35, 40),
        "e": rng.normal(0.1, 0.50, 40) + 1j * rng.normal(1.8, 0.25, 40),
        "f": rng.normal(2.3, 0.35, 40) + 1j * rng.normal(-0.7, 0.55, 40),
    }


def _mixture(
    samples: Mapping[str, np.ndarray],
    weights: tuple[int, int, int],
) -> np.ndarray:
    """Build an empirical GEF mixture from state-specific samples."""
    return np.concatenate(
        [
            np.tile(samples[state], count)
            for state, count in zip(("g", "e", "f"), weights, strict=True)
        ]
    )


def _calibration(
    target: str,
    samples: Mapping[str, np.ndarray],
) -> GefPopulationCalibration:
    """Build a pure-state six-configuration calibration."""
    states = {
        "c1": "g",
        "c2": "g",
        "c3": "e",
        "c4": "f",
        "c5": "e",
        "c6": "f",
    }
    raw_iq = {name: samples[state] for name, state in states.items()}
    summaries = {name: summarize_iq_shots(iq) for name, iq in raw_iq.items()}
    return GefPopulationCalibration(
        target=target,
        state_features=reconstruct_gef_state_features(summaries),
        summaries=summaries,
        raw_iq=raw_iq,
        analyzer_duration=10.0,
    )


def _measurement_iq(
    samples: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Build one sequence of S1/S4/S5 IQ shots for one target."""
    return {
        "prepared": {
            "s1": {"Q0": _mixture(samples, (8, 12, 20))},
            "s4": {"Q0": _mixture(samples, (12, 20, 8))},
            "s5": {"Q0": _mixture(samples, (20, 8, 12))},
        }
    }


def test_bootstrap_returns_reproducible_physical_population_samples() -> None:
    """A fixed seed should produce reproducible physical bootstrap samples."""
    samples = _state_samples()
    calibration = {"Q0": _calibration("Q0", samples)}
    raw_iq = _measurement_iq(samples)

    first = bootstrap_gef_populations(
        calibration,
        raw_iq,
        n_resamples=40,
        seed=17,
        confidence_level=0.90,
    )["prepared"]["Q0"]
    second = bootstrap_gef_populations(
        calibration,
        raw_iq,
        n_resamples=40,
        seed=17,
        confidence_level=0.90,
    )["prepared"]["Q0"]

    assert isinstance(first, GefPopulationBootstrap)
    assert_allclose(first.point_estimate, [0.2, 0.3, 0.5], rtol=1e-6, atol=1e-7)
    assert_allclose(first.samples, second.samples, rtol=0.0, atol=0.0)
    assert first.samples.shape == (40, 3)
    assert first.unconstrained_samples.shape == (40, 3)
    assert first.standard_error is not None
    assert first.standard_error.shape == (3,)
    assert_allclose(
        first.standard_error,
        np.std(first.samples, axis=0, ddof=1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert first.confidence_interval is not None
    assert first.confidence_interval.shape == (2, 3)
    assert_allclose(
        first.confidence_interval,
        np.quantile(first.samples, [0.05, 0.95], axis=0),
        rtol=1e-12,
        atol=1e-12,
    )
    assert first.bias is not None
    assert_allclose(
        first.bias,
        np.mean(first.samples, axis=0) - first.point_estimate,
        rtol=1e-12,
        atol=1e-12,
    )
    assert first.boundary_fraction is not None
    assert first.successful_resamples == 40
    assert first.success_rate == pytest.approx(1.0)
    assert first.confidence_level == pytest.approx(0.90)
    assert first.minimum_success_rate == pytest.approx(0.8)
    assert first.seed == 17
    assert first.unavailable_reason is None
    assert np.all(first.samples >= -1e-12)
    assert np.all(first.samples <= 1.0 + 1e-12)
    assert_allclose(np.sum(first.samples, axis=1), 1.0, rtol=0.0, atol=1e-10)


def test_bootstrap_preserves_pairing_between_simultaneously_measured_targets() -> None:
    """Shared shot indices should preserve simultaneous-target correlations."""
    q0_samples = _state_samples()
    q1_samples = {state: 1.3 * iq for state, iq in q0_samples.items()}
    calibration = {
        "Q0": _calibration("Q0", q0_samples),
        "Q1": _calibration("Q1", q1_samples),
    }
    q0_raw = _measurement_iq(q0_samples)["prepared"]
    q1_raw = _measurement_iq(q1_samples)["prepared"]
    raw_iq = {
        "prepared": {
            configuration: {
                "Q0": q0_raw[configuration]["Q0"],
                "Q1": q1_raw[configuration]["Q0"],
            }
            for configuration in ("s1", "s4", "s5")
        }
    }

    result = bootstrap_gef_populations(
        calibration,
        raw_iq,
        n_resamples=30,
        seed=23,
    )["prepared"]

    assert_allclose(
        result["Q0"].samples,
        result["Q1"].samples,
        rtol=1e-8,
        atol=1e-9,
    )
    assert_allclose(
        result["Q0"].unconstrained_samples,
        result["Q1"].unconstrained_samples,
        rtol=1e-8,
        atol=1e-9,
    )


def test_bootstrap_can_be_disabled_without_resampling_calibration() -> None:
    """Zero resamples should return an explicit disabled uncertainty summary."""
    samples = _state_samples()
    calibration = _calibration("Q0", samples)
    calibration.raw_iq.clear()

    result = bootstrap_gef_populations(
        {"Q0": calibration},
        _measurement_iq(samples),
        n_resamples=0,
    )["prepared"]["Q0"]

    assert result.samples.shape == (0, 3)
    assert result.successful_resamples == 0
    assert result.success_rate is None
    assert result.standard_error is None
    assert result.confidence_interval is None
    assert result.unavailable_reason == "disabled"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_resamples": -1}, "n_resamples"),
        ({"n_resamples": True}, "n_resamples"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"confidence_level": 0.0}, "confidence_level"),
        ({"confidence_level": 1.0}, "confidence_level"),
    ],
)
def test_bootstrap_rejects_invalid_options(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Invalid bootstrap options should fail before resampling."""
    samples = _state_samples()

    with pytest.raises((TypeError, ValueError), match=message):
        bootstrap_gef_populations(
            {"Q0": _calibration("Q0", samples)},
            _measurement_iq(samples),
            **kwargs,  # type: ignore[arg-type]
        )


def test_bootstrap_rejects_missing_measurement_configuration() -> None:
    """Every sequence should contain S1, S4, and S5 raw IQ data."""
    samples = _state_samples()
    raw_iq = _measurement_iq(samples)
    del raw_iq["prepared"]["s5"]

    with pytest.raises(ValueError, match="s5"):
        bootstrap_gef_populations(
            {"Q0": _calibration("Q0", samples)},
            raw_iq,
            n_resamples=2,
        )


def test_bootstrap_requires_calibration_raw_iq_when_enabled() -> None:
    """Enabled bootstrap should require all six calibration IQ arrays."""
    samples = _state_samples()
    calibration = _calibration("Q0", samples)
    del calibration.raw_iq["c6"]

    with pytest.raises(ValueError, match="c6"):
        bootstrap_gef_populations(
            {"Q0": calibration},
            _measurement_iq(samples),
            n_resamples=2,
        )


def test_bootstrap_rejects_unpaired_simultaneous_target_shot_counts() -> None:
    """Simultaneous targets should have equal shot counts for paired resampling."""
    q0_samples = _state_samples()
    q1_samples = {state: 1.3 * iq for state, iq in q0_samples.items()}
    raw_iq = _measurement_iq(q0_samples)
    for configuration in ("s1", "s4", "s5"):
        raw_iq["prepared"][configuration]["Q1"] = raw_iq["prepared"][configuration][
            "Q0"
        ]
    raw_iq["prepared"]["s1"]["Q1"] = raw_iq["prepared"]["s1"]["Q1"][:-1]

    with pytest.raises(ValueError, match="equal shot counts"):
        bootstrap_gef_populations(
            {
                "Q0": _calibration("Q0", q0_samples),
                "Q1": _calibration("Q1", q1_samples),
            },
            raw_iq,
            n_resamples=2,
        )
