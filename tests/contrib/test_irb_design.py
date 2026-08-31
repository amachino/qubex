"""Uniform final-IRB designs preserve tail coverage, resolution and paired costs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from qubex.contrib.experiment.irb_design import (
    assess_irb_tail,
    estimate_irb_cost,
    plan_uniform_irb_depths,
)


def _plan(reference: float = 0.985, interleaved: float = 0.97, **kwargs):
    return plan_uniform_irb_depths(
        reference,
        interleaved,
        source_kind="qualified_pilot",
        source_label="independent pilot artifact",
        reference_qualified=True,
        interleaved_qualified=True,
        **kwargs,
    )


@pytest.mark.parametrize("rates", [(0.985, 0.97), (0.995, 0.99), (0.97, 0.965)])
def test_uniform_grid_resolves_fast_decay_and_covers_slow_tail(rates):
    """One common integer grid resolves both series and reaches five slow e-folds."""
    plan = _plan(*rates)
    depths = np.asarray(plan["depths"])
    assert depths[0] == 0
    assert np.all(np.diff(depths) == plan["uniform_step"])
    assert plan["fit_depths"] == depths[1:].tolist()
    assert plan["uniform_step"] <= plan["fast_decay_length"] / 4
    assert plan["maximum_depth"] >= 5 * plan["slow_decay_length"]
    assert plan["predicted_tail_reference"] < 0.01
    assert plan["predicted_tail_interleaved"] < 0.01
    assert 20 <= plan["positive_depth_count"] <= 60
    assert "not fitted" in plan["depth_zero_role"]


def test_wide_decay_ratio_is_not_truncated_to_sixty_depths() -> None:
    """A large REF/IRB decay disparity keeps every required depth beyond the preference."""
    plan = _plan(0.998, 0.95)
    assert plan["positive_depth_count"] > 60
    assert plan["maximum_depth"] >= 5 * plan["slow_decay_length"]
    assert plan["warnings"]


def test_pilot_confidence_bounds_control_resolution_and_tail() -> None:
    """Optional pilot bounds use the fastest lower rate and slowest upper rate."""
    plan = _plan(reference_p_bounds=(0.98, 0.99), interleaved_p_bounds=(0.96, 0.98))
    assert plan["fast_decay_length"] == pytest.approx(-1 / np.log(0.96))
    assert plan["slow_decay_length"] == pytest.approx(-1 / np.log(0.99))
    assert 0.99 ** plan["maximum_depth"] < 0.01
    assert plan["uses_pilot_bounds"]


def test_failed_pilot_requires_a_new_pilot_or_explicit_manual_prior() -> None:
    """A failed fit is not silently promoted to a usable planning decay."""
    with pytest.raises(ValueError, match="qualified"):
        plan_uniform_irb_depths(
            0.985,
            0.97,
            source_kind="qualified_pilot",
            source_label="failed pilot",
            reference_qualified=True,
            interleaved_qualified=False,
        )
    plan = plan_uniform_irb_depths(
        0.985,
        0.97,
        source_kind="manual_prior",
        source_label="explicit user-specified planning prior",
    )
    assert plan["source_kind"] == "manual_prior"
    assert plan["requires_independent_tail_validation"]


@pytest.mark.parametrize("p", [0, 1, -0.1, 1.001, np.nan])
def test_invalid_or_nondecaying_rates_are_rejected(p: float) -> None:
    """Finite decay lengths require rates strictly between zero and one."""
    with pytest.raises(ValueError, match="strictly between"):
        _plan(p, 0.97)


def test_unbounded_pilot_interval_and_unresolvable_fast_decay_are_rejected() -> None:
    """The planner neither caps an infinite tail nor undersamples a sub-grid decay."""
    with pytest.raises(ValueError, match="strictly between"):
        _plan(reference_p_bounds=(0.98, 1.0))
    with pytest.raises(ValueError, match="integer Clifford"):
        _plan(0.98, 0.7)


def test_planner_is_deterministic_and_has_no_timestamp_state() -> None:
    """Identical pilot inputs produce the same fully inspectable design."""
    assert _plan() == _plan()
    assert all(
        "time" not in key or key == "requires_independent_tail_validation"
        for key in _plan()
    )


def test_matched_cost_retains_seed_order_and_counts_depth_zero_twice() -> None:
    """REF and IRB share all seeds and depths, including both empty controls."""
    depths = list(range(0, 401, 10))
    seeds = list(range(20024, 20000, -1))
    cost = estimate_irb_cost(depths, seeds, 1024)
    assert cost["seeds"] == seeds
    assert cost["circuit_count"] == 2 * 24 * 41
    assert cost["requested_shots"] == 2_015_232
    assert cost["idle_wait_seconds"] == pytest.approx(2015.232)
    assert cost["estimated_wall_seconds"] is None
    assert cost["budget_status"] == "not_specified"


def test_actual_sequence_durations_and_setup_overhead_are_added_once() -> None:
    """Wall estimates include trailing idle, every shot's sequence and per-call setup."""
    depths, seeds = [0, 5, 10, 15, 20], [11, 12, 13, 14]
    durations = np.full((2, 4, 5), 20_000.0)
    durations[1] += 5_000.0
    cost = estimate_irb_cost(
        depths,
        seeds,
        512,
        circuit_durations_ns=durations,
        per_acquisition_setup_seconds=0.1,
        maximum_wall_seconds=30,
    )
    expected = 40 * 512 * 1e-3 + 512 * durations.sum() * 1e-9 + 40 * 0.1
    assert cost["estimated_wall_seconds"] == pytest.approx(expected)
    assert cost["budget_status"] == "within_declared_budgets"


def test_budget_failure_preserves_complete_uniform_range() -> None:
    """Insufficient shots or time is reported without shortening the decay range."""
    depths = list(range(0, 401, 10))
    cost = estimate_irb_cost(
        depths, list(range(24)), 1024, maximum_shots=1_000_000, maximum_wall_seconds=100
    )
    assert cost["budget_status"] == "exceeds_declared_budget"
    assert not cost["within_shot_budget"]
    assert not cost["within_time_budget"]
    assert cost["depths"] == depths


def test_unknown_circuit_overhead_does_not_become_zero_wall_cost() -> None:
    """An idle-only lower bound cannot certify a wall-time budget."""
    cost = estimate_irb_cost(
        [0, 5, 10, 15, 20], list(range(4)), 512, maximum_wall_seconds=100
    )
    assert cost["budget_status"] == "incomplete_time_estimate"
    assert cost["within_time_budget"] is None


def test_more_seeds_at_lower_shots_has_explicit_cost() -> None:
    """The 32-seed 512-shot alternative is not confused with 24 times 1024."""
    cost = estimate_irb_cost(list(range(0, 401, 10)), list(range(32)), 512)
    assert cost["requested_shots"] == 1_343_488
    assert cost["idle_wait_seconds"] == pytest.approx(1343.488)


def test_nonuniform_depths_duplicate_seeds_and_misaligned_durations_fail() -> None:
    """Cost inputs must preserve the same uniform paired acquisition design."""
    with pytest.raises(ValueError, match="uniform"):
        estimate_irb_cost([0, 1, 2, 4, 8], list(range(4)), 512)
    with pytest.raises(ValueError, match="unique"):
        estimate_irb_cost([0, 1, 2, 3, 4], [1, 1, 2, 3], 512)
    with pytest.raises(ValueError, match="shape"):
        estimate_irb_cost(
            [0, 1, 2, 3, 4], list(range(4)), 512, circuit_durations_ns=np.ones((4, 5))
        )


def _tail_example() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    depths = np.arange(0, 501, 10)
    arrays = {}
    fits = {}
    for mode, p in (("reference", 0.99), ("interleaved", 0.98)):
        arrays[mode] = np.tile(0.7 * p**depths + 0.25, (24, 1))
        fits[mode] = dict(
            p=p,
            amplitude=0.7,
            floor=0.25,
            errors=dict(p=1e-5, floor=0.001),
            fit_quality_pass=True,
        )
    return depths, arrays, dict(quote_as_irb_estimate=True, fits=fits)


def test_final_tail_uses_actual_data_and_final_fit_without_refitting() -> None:
    """Final tail evidence must agree with the final qualified fits, not only the pilot."""
    depths, arrays, analysis = _tail_example()
    before = deepcopy(analysis)
    result = assess_irb_tail(
        depths, arrays["reference"], arrays["interleaved"], analysis=analysis
    )
    assert result["passed"]
    assert analysis == before
    assert result["by_mode"]["reference"]["upper_remaining_contrast_fraction"] < 0.01


def test_final_slow_tail_or_bad_fit_is_not_accepted() -> None:
    """A slow final decay and an unqualified analysis each block the final result."""
    depths, arrays, analysis = _tail_example()
    analysis["fits"]["reference"]["p"] = 0.998
    result = assess_irb_tail(
        depths, arrays["reference"], arrays["interleaved"], analysis=analysis
    )
    assert not result["passed"]
    assert "tail" in " ".join(result["reasons"])
    analysis["quote_as_irb_estimate"] = False
    assert not assess_irb_tail(
        depths, arrays["reference"], arrays["interleaved"], analysis=analysis
    )["passed"]


def test_measured_tail_disagreement_is_not_hidden_by_a_small_fitted_p() -> None:
    """A displaced measured tail fails despite a fitted rate predicting convergence."""
    depths, arrays, analysis = _tail_example()
    arrays["reference"][:, -3:] += 0.08
    result = assess_irb_tail(
        depths, arrays["reference"], arrays["interleaved"], analysis=analysis
    )
    assert not result["passed"]
    assert not result["by_mode"]["reference"]["tail_floor_compatible"]


@pytest.mark.parametrize("failed_p", [1.0, np.nan])
def test_unqualified_fit_does_not_abort_raw_tail_reporting(failed_p: float) -> None:
    """A failed boundary/nonfinite fit retains raw evidence without evaluating its model."""
    depths, arrays, analysis = _tail_example()
    analysis["quote_as_irb_estimate"] = False
    analysis["fits"]["reference"].update(p=failed_p, fit_quality_pass=False)
    result = assess_irb_tail(
        depths, arrays["reference"], arrays["interleaved"], analysis=analysis
    )
    assert not result["passed"]
    assert not result["by_mode"]["reference"]["model_tail_evaluated"]
    assert np.isfinite(result["by_mode"]["reference"]["raw_tail_mean"])
