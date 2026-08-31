"""Offline tests of fixed-duration response-ridge and noisy GP planning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qubex.contrib.experiment.bswap_calibration.local_optimization import (
    estimate_response_ridge,
    plan_frequency_extensions,
    propose_gp_point,
    ridge_scout_points,
    select_conservative_plateau,
)


def _scout(slope: float = -0.45) -> tuple[np.ndarray, ...]:
    plan = ridge_scout_points(
        0.99, 4.61, prior_slope_ghz_per_amplitude=slope, frequency_points=13
    )
    points = np.asarray(plan["points"])
    detuning = points[:, 1] - (4.61 + slope * (points[:, 0] - 0.99))
    scores = 0.1 + 0.85 * np.exp(-((detuning / 0.00015) ** 2))
    return points[:, 0], points[:, 1], scores, np.full(len(scores), 1e-5)


def test_scout_uses_distinct_capped_amplitudes_and_prior_ridge() -> None:
    """A capped scout has three distinct amplitudes and follows the supplied prior slope."""
    plan = ridge_scout_points(0.99, 4.61, prior_slope_ghz_per_amplitude=-0.6)
    points = np.asarray(plan["points"])
    np.testing.assert_allclose(np.unique(points[:, 0]), [0.987, 0.9885, 0.99])
    for amplitude in np.unique(points[:, 0]):
        frequencies = points[points[:, 0] == amplitude, 1]
        assert np.mean(frequencies) == pytest.approx(4.61 - 0.6 * (amplitude - 0.99))
        assert np.ptp(frequencies) == pytest.approx(0.0012)
    assert plan["requires_acquisition"] is True
    assert plan["requested_shots"] == 0


@pytest.mark.parametrize("slope", [-0.3, -0.6])
def test_measured_ridge_recovers_tilt_with_peak_uncertainty(slope: float) -> None:
    """Interior scout peaks recover a tilted fixed-duration response ridge with uncertainty."""
    ridge = estimate_response_ridge(*_scout(slope), reference_amplitude=0.99)
    assert ridge["qualified"] is True
    assert ridge["label"] == "fixed-duration response ridge"
    assert ridge["frequency_ghz"] == pytest.approx(4.61, abs=1e-8)
    assert ridge["slope_ghz_per_amplitude"] == pytest.approx(slope, abs=1e-5)
    assert ridge["slope_standard_error"] > 0
    assert all(row["interior_peak"] for row in ridge["rows"])


def test_uncovered_tilt_requests_extension_instead_of_claiming_a_ridge() -> None:
    """A narrow unrotated scout reports missing coverage and does not authorize GP planning."""
    plan = ridge_scout_points(0.99, 4.61, prior_slope_ghz_per_amplitude=0.0)
    points = np.asarray(plan["points"])
    delta = points[:, 1] - (4.61 - 0.6 * (points[:, 0] - 0.99))
    scores = np.exp(-((delta / 0.0002) ** 2))
    data = points[:, 0], points[:, 1], scores, np.full(len(scores), 1e-5)
    ridge = estimate_response_ridge(*data, frequency_bounds_ghz=(4.607, 4.615))
    assert ridge["qualified"] is False
    assert ridge["suggested_frequency_extensions"]
    assert all(
        row["frequency_ghz"] <= 4.615 for row in ridge["suggested_frequency_extensions"]
    )
    with pytest.raises(ValueError, match="qualified response ridge"):
        propose_gp_point(
            *data,
            ridge=ridge,
            amplitude_bounds=(0.987, 0.99),
            frequency_bounds_ghz=(4.607, 4.615),
        )


def test_uncertain_flat_row_is_not_an_interior_peak() -> None:
    """A flat noisy response does not acquire an artificial peak through interpolation."""
    a, f, _, variance = _scout()
    ridge = estimate_response_ridge(a, f, np.full_like(a, 0.5), variance)
    assert ridge["qualified"] is False
    assert any(
        "curvature" in row["reason"] or "boundary" in row["reason"]
        for row in ridge["rows"]
    )
    assert all(
        not row.get("extension_direction_resolved", False) for row in ridge["rows"]
    )


def test_gp_handles_nonquadratic_narrow_ridge_and_avoids_observed_points() -> None:
    """A GP proposes a new physical point where a global quadratic has high shot-normalized residuals."""
    a, f, y, variance = _scout()
    x, z = (a - 0.9885) / 0.0015, (f - f.mean()) / 0.001
    design = np.column_stack([np.ones(len(a)), x, z, x * x, x * z, z * z])
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    assert float(np.mean(residual**2 / variance)) > 100
    ridge = estimate_response_ridge(a, f, y, variance)
    result = propose_gp_point(
        a,
        f,
        y,
        variance,
        ridge=ridge,
        amplitude_bounds=(0.987, 0.99),
        frequency_bounds_ghz=(4.607, 4.615),
        optimize_kernel=False,
        include_anchor=True,
    )
    candidate = result["candidate"]
    assert 0.987 <= candidate["amplitude"] <= 0.99
    assert 4.607 <= candidate["frequency_ghz"] <= 4.615
    assert not np.any(
        (abs(a - candidate["amplitude"]) < 1e-10)
        & (abs(f - candidate["frequency_ghz"]) < 1e-12)
    )
    assert result["requested_shots"] == 0
    assert result["validation_required"] is True
    assert (
        result["optional_anchor"]["purpose"]
        == "repeatability anchor, not independent validation"
    )


def test_known_noise_downweights_the_largest_noisy_observation() -> None:
    """The incumbent is selected from conservative posterior scores rather than the largest noisy datum."""
    a, f, y, variance = _scout()
    ridge = estimate_response_ridge(a, f, y, variance)
    a, f, y, variance = (
        np.r_[array, value]
        for array, value in zip(
            (a, f, y, variance), (0.989, 4.6109, 4.0, 4.0), strict=True
        )
    )
    result = propose_gp_point(
        a,
        f,
        y,
        variance,
        ridge=ridge,
        amplitude_bounds=(0.987, 0.99),
        frequency_bounds_ghz=(4.607, 4.615),
        optimize_kernel=False,
    )
    assert result["best_observed"]["index"] != len(a) - 1
    assert result["best_observed"]["observed_score"] < 1.1


def test_larger_known_noise_increases_posterior_uncertainty() -> None:
    """Known shot variance remains in the GP posterior instead of being discarded by normalization."""
    a, f, y, variance = _scout()
    ridge = estimate_response_ridge(a, f, y, variance)
    kwargs: dict[str, Any] = dict(
        ridge=ridge,
        amplitude_bounds=(0.987, 0.99),
        frequency_bounds_ghz=(4.607, 4.615),
        optimize_kernel=False,
        candidates=np.array([[0.9881, 4.61086]]),
    )
    small = propose_gp_point(a, f, y, variance, **kwargs)
    large = propose_gp_point(a, f, y, variance * 1000, **kwargs)
    assert (
        large["candidate"]["posterior_standard_deviation"]
        > small["candidate"]["posterior_standard_deviation"]
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, "not-a-number"])
def test_invalid_observation_values_fail_before_planning(bad: object) -> None:
    """Nonnumeric or nonfinite observations cannot be interpreted as optimizer evidence."""
    a, f, y, variance = _scout()
    broken = y.astype(object)
    broken[0] = bad
    with pytest.raises(ValueError, match="finite numeric"):
        estimate_response_ridge(a, f, broken, variance)


def test_repeats_need_explicit_permission_and_bounds_are_not_expanded() -> None:
    """An exhausted candidate set fails without scheduling an implicit repeat or expanding bounds."""
    a, f, y, variance = _scout()
    ridge = estimate_response_ridge(a, f, y, variance)
    kwargs: dict[str, Any] = dict(
        ridge=ridge,
        amplitude_bounds=(0.987, 0.99),
        frequency_bounds_ghz=(4.607, 4.615),
        candidates=np.column_stack([a, f]),
        optimize_kernel=False,
    )
    with pytest.raises(ValueError, match="unmeasured candidates"):
        propose_gp_point(a, f, y, variance, **kwargs)
    result = propose_gp_point(a, f, y, variance, allow_repeats=True, **kwargs)
    assert result["candidate"]["previously_measured"] is True


def test_ucb_with_fitted_kernel_returns_a_bounded_unqualified_candidate() -> None:
    """Fitted-kernel UCB remains a proposal and never reports gate qualification."""
    a, f, y, variance = _scout()
    result = propose_gp_point(
        a,
        f,
        y,
        variance,
        ridge=estimate_response_ridge(a, f, y, variance),
        amplitude_bounds=(0.987, 0.99),
        frequency_bounds_ghz=(4.607, 4.615),
        acquisition="ucb",
    )
    assert result["acquisition"] == "ucb"
    assert np.isfinite(result["candidate"]["acquisition_value"])
    assert result["validation_required"] is True
    assert 0.987 <= result["candidate"]["amplitude"] <= 0.99


def test_invalid_variance_or_scout_width_does_not_create_a_plan() -> None:
    """Zero shot variance and an impossible three-row scout fail before any proposal."""
    a, f, y, variance = _scout()
    variance[0] = 0.0
    with pytest.raises(ValueError, match="positive shot variances"):
        estimate_response_ridge(a, f, y, variance)
    with pytest.raises(ValueError, match="three distinct amplitudes"):
        ridge_scout_points(
            0.99, 4.61, amplitude_step=0.0015, amplitude_bounds=(0.989, 0.99)
        )


def test_extensions_touch_only_failed_rows_and_keep_physical_bounds() -> None:
    """Frequency extensions stay bounded, avoid measured pairs, and leave good rows alone."""
    scout = ridge_scout_points(0.99, 4.61)
    points = np.asarray(scout["points"])
    a, f = points.T
    scores = np.exp(-(((f - (4.61 - 0.6 * (a - 0.99))) / 0.0002) ** 2))
    variance = np.full(len(a), 1e-5)
    ridge = estimate_response_ridge(a, f, scores, variance)
    extension = plan_frequency_extensions(
        a,
        f,
        scores,
        variance,
        ridge=ridge,
        frequency_bounds_ghz=(4.608, 4.612),
        round_index=0,
    )
    failed = {r["amplitude"] for r in ridge["rows"] if not r["interior_peak"]}
    assert extension["points"]
    assert {p[0] for p in extension["points"]} <= failed
    assert all(4.608 <= p[1] <= 4.612 for p in extension["points"])
    assert len(extension["points"]) == len(set(map(tuple, extension["points"])))
    for amplitude, frequency in extension["points"]:
        assert not np.any((abs(a - amplitude) < 1e-10) & (abs(f - frequency) < 1e-12))
    assert extension["qualified_ridge"] is False
    assert extension["requested_shots"] == 0


def test_extension_round_budget_is_not_automatically_expanded() -> None:
    """The third request returns no extension after a two-round caller budget."""
    a, f, _, variance = _scout()
    scores = np.full(len(a), 0.5)
    ridge = estimate_response_ridge(a, f, scores, variance)
    result = plan_frequency_extensions(
        a,
        f,
        scores,
        variance,
        ridge=ridge,
        frequency_bounds_ghz=(4.608, 4.614),
        round_index=2,
        max_rounds=2,
    )
    assert result["exhausted"] is True
    assert result["points"] == []


def test_unresolved_curvature_gets_symmetric_coverage_not_a_false_peak() -> None:
    """An unresolved shallow interior bump proposes symmetric coverage without qualifying a ridge."""
    a, f, _, variance = _scout(0.0)
    scores = 0.9 + 1e-5 * np.exp(-(((f - 4.61) / 0.0002) ** 2))
    ridge = estimate_response_ridge(a, f, scores, variance)
    assert not ridge["suggested_frequency_extensions"]
    result = plan_frequency_extensions(
        a,
        f,
        scores,
        variance,
        ridge=ridge,
        frequency_bounds_ghz=(4.608, 4.612),
        round_index=0,
    )
    assert any(p[1] < f.min() for p in result["points"])
    assert any(p[1] > f.max() for p in result["points"])
    assert all(
        row["purpose"] == "symmetric coverage; curvature unresolved"
        for row in result["rows"]
    )
    assert result["qualified_ridge"] is False
    assert ridge["qualified"] is False


def test_root_plateau_retains_the_independent_seed_not_the_noisiest_point() -> None:
    """A high unresolved root plateau retains the supplied carrier and nearest seed amplitude."""
    plan = ridge_scout_points(0.99, 4.61, amplitude_step=0.0005)
    a, f = np.asarray(plan["points"]).T
    scores = np.full(len(a), 0.94)
    scores[(a == 0.99) & np.isclose(f, 4.61, atol=1e-12, rtol=0)] += 0.002
    result = select_conservative_plateau(
        a,
        f,
        scores,
        np.full(len(a), 1e-4),
        gate_kind="sqrt_bswap",
        seed_amplitude=0.98997,
        seed_frequency_ghz=4.610123,
    )
    assert result["accepted"] is True
    assert result["candidate"] == {"amplitude": 0.99, "frequency_ghz": 4.610123}
    assert result["qualified_ridge"] is False
    assert result["gp_allowed"] is False
    assert result["validation_required"] is True
    assert result["minimum_simultaneous_lower_bound"] >= 0.85


@pytest.mark.parametrize(
    "case", ["full", "low_score", "high_noise", "resolved_peak", "unbracketed_seed"]
)
def test_plateau_rejects_unsupported_fallbacks(case: str) -> None:
    """Only high-confidence, unresolved, seed-bracketing root rows admit the conservative fallback."""
    plan = ridge_scout_points(0.99, 4.61, amplitude_step=0.0005)
    a, f = np.asarray(plan["points"]).T
    scores = np.full(len(a), 0.94)
    variance = np.full(len(a), 1e-4)
    kind, seed_f = "sqrt_bswap", 4.61
    if case == "full":
        kind = "bswap"
    elif case == "low_score":
        scores[:] = 0.6
    elif case == "high_noise":
        variance[:] = 0.01
    elif case == "resolved_peak":
        scores += 0.05 * np.exp(-(((f - 4.61) / 0.0002) ** 2))
        variance[:] = 1e-7
    elif case == "unbracketed_seed":
        seed_f = 4.6109
    result = select_conservative_plateau(
        a,
        f,
        scores,
        variance,
        gate_kind=kind,
        seed_amplitude=0.99,
        seed_frequency_ghz=seed_f,
    )
    assert result["accepted"] is False
    assert result["candidate"] is None
    assert result["reasons"]


def test_resolved_shoulders_support_a_five_point_peak_when_top_three_are_noisy() -> (
    None
):
    """Available shoulders resolve peak curvature without changing the confidence threshold."""
    a = np.repeat([0.3, 0.4, 0.5], 7)
    f = np.tile(5.0 + np.linspace(-0.0006, 0.0006, 7), 3)
    y = np.tile([0.585, 0.700, 0.742, 0.787, 0.782, 0.749, 0.710], 3)
    variance = np.full(len(a), 0.0002)
    ridge = estimate_response_ridge(a, f, y, variance)
    assert ridge["qualified"] is True
    assert all(row["fit_points"] == 5 for row in ridge["rows"])
    assert all(row["local_fit_reduced_chi2"] < 5 for row in ridge["rows"])
    assert all(row["ci95_ghz"][1] < 5.0002 for row in ridge["rows"])


def test_five_point_peak_is_rejected_when_its_model_disagrees_with_shot_noise() -> None:
    """A wider fit cannot qualify a noisy top if its shoulders violate the quadratic model."""
    a = np.repeat([0.3, 0.4, 0.5], 7)
    f = np.tile(5.0 + np.linspace(-0.0006, 0.0006, 7), 3)
    y = np.tile([0.4, 0.1, 0.80, 0.81, 0.80, 0.75, 0.4], 3)
    ridge = estimate_response_ridge(a, f, y, np.full(len(a), 0.0002))
    assert ridge["qualified"] is False
    assert all("model" in row["reason"] for row in ridge["rows"])


def test_observed_incumbent_and_anchor_obey_the_same_residual_band_as_candidates() -> (
    None
):
    """A high observed score outside the ridge band cannot become the constrained incumbent."""
    a, f, y, variance = _scout()
    ridge = estimate_response_ridge(a, f, y, variance)
    a = np.r_[a, 0.9885]
    f = np.r_[f, 4.610675 + 0.0011]
    y = np.r_[y, 1.0]
    variance = np.r_[variance, 1e-8]
    result = propose_gp_point(
        a,
        f,
        y,
        variance,
        ridge=ridge,
        amplitude_bounds=(0.987, 0.99),
        frequency_bounds_ghz=(4.607, 4.615),
        optimize_kernel=False,
        include_anchor=True,
    )
    assert result["best_observed"]["index"] != len(a) - 1
    for point in (result["best_observed"], result["optional_anchor"]):
        center = ridge["frequency_ghz"] + ridge["slope_ghz_per_amplitude"] * (
            point["amplitude"] - ridge["reference_amplitude"]
        )
        assert abs(point["frequency_ghz"] - center) <= 0.0006 + 1e-12
