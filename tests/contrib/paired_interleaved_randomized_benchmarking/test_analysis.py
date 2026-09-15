"""Tests for paired interleaved randomized-benchmarking analysis."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from qubex.contrib import analyze_paired_irb
from qubex.experiment.models import Result


def _raw_result(
    reference_trials: np.ndarray,
    interleaved_trials: np.ndarray,
    *,
    n_cliffords: np.ndarray,
    dimension: int = 2,
    schema_version: int = 1,
    first_protocols: np.ndarray | None = None,
) -> Result:
    """Build a minimal measurement-only result for analysis tests."""
    n_trials = reference_trials.shape[1]
    if first_protocols is None:
        first_protocols = np.empty((len(n_cliffords), n_trials), dtype="<U11")
        for length_index in range(len(n_cliffords)):
            n_reference_first = n_trials // 2 + int(
                n_trials % 2 == 1 and length_index % 2 == 0
            )
            first_protocols[length_index, :n_reference_first] = "reference"
            first_protocols[length_index, n_reference_first:] = "interleaved"
    planned_order = tuple(
        {
            "length_index": length_index,
            "trial_index": trial_index,
            "first_protocol": str(first_protocols[length_index, trial_index]),
        }
        for length_index in range(len(n_cliffords))
        for trial_index in range(n_trials)
    )
    return Result(
        data={
            "Q0": {
                "reference": {"trials": reference_trials},
                "interleaved": {"trials": interleaved_trials},
                "acquisition": {
                    "n_cliffords": n_cliffords,
                    "n_trials": n_trials,
                    "n_shots": 1_000,
                    "seeds": np.arange(
                        len(n_cliffords) * n_trials,
                        dtype=np.int64,
                    ).reshape(len(n_cliffords), n_trials),
                    "planned_order": planned_order,
                },
                "metadata": {
                    "schema_version": schema_version,
                    "dimension": dimension,
                    "interleaved_clifford": "X90",
                },
            }
        }
    )


def test_identical_paired_trials_produce_unit_fidelity_and_zero_uncertainty() -> None:
    """Identical paired data should yield unit fidelity in every bootstrap fit."""
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    mean = 0.46 * 0.97**n_cliffords + 0.5
    offsets = np.linspace(-0.006, 0.006, 20)
    trials = mean[:, None] + offsets[None, :]
    raw = _raw_result(trials, trials.copy(), n_cliffords=n_cliffords)

    result = analyze_paired_irb(raw, n_bootstrap=40, bootstrap_seed=123)

    target = result["Q0"]
    assert target["gate_fidelity"] == pytest.approx(1.0, abs=1e-12)
    assert target["gate_fidelity_err"] == pytest.approx(0.0, abs=1e-12)
    assert target["gate_fidelity_ci68"] == pytest.approx((1.0, 1.0), abs=1e-12)
    assert target["gate_fidelity_ci95"] == pytest.approx((1.0, 1.0), abs=1e-12)
    assert target["bootstrap"]["n_success"] == 40
    assert target["bootstrap"]["valid"] is True
    assert target["bootstrap"]["parameter_bound_quality_valid"] is True
    assert target["bootstrap"]["resampling_method"] == "ab_ba_stratified_paired"
    assert target["bootstrap"]["stratum_counts"] == (
        {"reference_first": 10, "interleaved_first": 10},
    ) * len(n_cliffords)


def test_weighted_fit_recovers_known_synthetic_gate_fidelity() -> None:
    """Weighted full-data fits should recover a known synthetic IRB fidelity."""
    rng = np.random.default_rng(7)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    n_trials = 30
    p_reference = 0.985
    p_interleaved = 0.955
    paired_noise = rng.normal(0.0, 0.002, size=(len(n_cliffords), n_trials))
    reference = 0.47 * p_reference ** n_cliffords[:, None] + 0.49 + paired_noise
    interleaved = 0.46 * p_interleaved ** n_cliffords[:, None] + 0.50 + paired_noise
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    expected = 1.0 - 0.5 * (1.0 - p_interleaved / p_reference)

    result = analyze_paired_irb(raw, n_bootstrap=30, bootstrap_seed=9)

    target = result["Q0"]
    assert target["gate_fidelity"] == pytest.approx(expected, abs=2e-3)
    assert target["reference"]["std"].shape == n_cliffords.shape
    assert target["reference"]["sem"].shape == n_cliffords.shape
    np.testing.assert_allclose(
        target["reference"]["std"],
        np.std(reference, axis=1, ddof=1),
    )
    np.testing.assert_allclose(
        target["reference"]["sem"],
        np.std(reference, axis=1, ddof=1) / np.sqrt(n_trials),
    )
    assert target["reference"]["fit"]["absolute_sigma"] is True


def test_rb_fit_initializes_decay_from_the_largest_clifford_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RB fitting should scale its initial decay estimate to the measured range."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 10, 100], dtype=np.int64)
    mean = 0.4 * np.exp(-n_cliffords / 100.0) + 0.5
    sem_used = np.full_like(mean, 0.01)
    captured: dict[str, tuple[float, float, float]] = {}

    def fake_curve_fit(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Capture fit initialization while returning a valid optimizer result."""
        captured["p0"] = kwargs["p0"]
        return np.asarray([0.4, np.exp(-0.01), 0.5]), np.eye(3)

    monkeypatch.setattr(module, "curve_fit", fake_curve_fit)

    module._fit_rb_decay(n_cliffords, mean, sem_used)  # noqa: SLF001

    np.testing.assert_allclose(
        captured["p0"],
        (
            float(mean[0] - np.min(mean)),
            np.exp(-1.0 / 100.0),
            float(np.min(mean)),
        ),
        rtol=1e-7,
        atol=1e-9,
    )


def _evaluate_noisy_grid_design(
    grid: np.ndarray,
    *,
    n_trials: int,
    p_reference: float,
    p_interleaved: float,
    n_repeats: int,
    data_seed: int,
    bootstrap_seed: int,
) -> dict[str, float]:
    """Return deterministic Monte Carlo precision and cost metrics for one grid."""
    expected_fidelity = 1.0 - 0.5 * (1.0 - p_interleaved / p_reference)
    p_reference_estimates = []
    p_interleaved_estimates = []
    fidelity_estimates = []
    bootstrap_sigmas = []
    coverage = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(data_seed + repeat)
        shape = (len(grid), n_trials)
        paired_noise = rng.normal(0.0, 0.004, size=shape)
        reference_noise = rng.normal(0.0, 0.002, size=shape)
        interleaved_noise = rng.normal(0.0, 0.002, size=shape)
        reference = (
            0.48 * p_reference ** grid[:, None] + 0.50 + paired_noise + reference_noise
        )
        interleaved = (
            0.48 * p_interleaved ** grid[:, None]
            + 0.50
            + paired_noise
            + interleaved_noise
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = analyze_paired_irb(
                _raw_result(reference, interleaved, n_cliffords=grid),
                n_bootstrap=30,
                bootstrap_seed=bootstrap_seed + repeat,
            )["Q0"]
        p_reference_estimates.append(result["reference"]["fit"]["p"])
        p_interleaved_estimates.append(result["interleaved"]["fit"]["p"])
        fidelity_estimates.append(result["gate_fidelity"])
        if result["gate_fidelity_err"] is not None:
            bootstrap_sigmas.append(result["gate_fidelity_err"])
        ci95 = result["gate_fidelity_ci95"]
        if ci95 is not None:
            coverage.append(float(ci95[0] <= expected_fidelity <= ci95[1]))
    return {
        "p_reference_bias": float(np.mean(p_reference_estimates) - p_reference),
        "p_interleaved_bias": float(np.mean(p_interleaved_estimates) - p_interleaved),
        "fidelity_bias": float(np.mean(fidelity_estimates) - expected_fidelity),
        "fidelity_sd": float(np.std(fidelity_estimates, ddof=1)),
        "bootstrap_sigma": float(np.mean(bootstrap_sigmas)),
        "ci95_coverage": float(np.mean(coverage)),
        "n_lengths": float(len(grid)),
        "measurement_pairs": float(len(grid) * n_trials),
        "sequence_cost": float(2 * n_trials * np.sum(grid)),
    }


def test_decay_adapted_grid_improves_high_fidelity_precision_at_equal_budget() -> None:
    """Adaptive long-length placement should improve noisy high-fidelity fits."""
    fixed_grid = np.asarray(
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        dtype=np.int64,
    )
    adaptive_grid = np.asarray(
        [0, 1, 105, 211, 356, 703, 1203, 1386, 1896, 2048],
        dtype=np.int64,
    )
    design_options = {
        "p_reference": 0.9995,
        "p_interleaved": 0.9990,
        "n_repeats": 16,
        "data_seed": 10_000,
        "bootstrap_seed": 20_000,
    }
    fixed = _evaluate_noisy_grid_design(
        fixed_grid,
        n_trials=24,
        **design_options,
    )
    adaptive = _evaluate_noisy_grid_design(
        adaptive_grid,
        n_trials=31,
        **design_options,
    )
    assert abs(adaptive["p_reference_bias"]) < 5e-6
    assert abs(adaptive["p_interleaved_bias"]) < 5e-6
    assert abs(adaptive["fidelity_bias"]) < 3e-6
    assert adaptive["fidelity_sd"] < 0.8 * fixed["fidelity_sd"]
    assert adaptive["bootstrap_sigma"] < fixed["bootstrap_sigma"]
    assert adaptive["ci95_coverage"] >= fixed["ci95_coverage"]
    assert adaptive["ci95_coverage"] >= 0.8
    assert abs(adaptive["measurement_pairs"] - fixed["measurement_pairs"]) <= 2
    assert adaptive["n_lengths"] < fixed["n_lengths"]
    assert adaptive["sequence_cost"] > fixed["sequence_cost"]


def test_parallel_shared_grid_preserves_noisy_multi_scale_precision() -> None:
    """An unthinned shared union should preserve fits across decay scales."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    decay_pairs = {
        "slow": (0.9995, 0.9990),
        "medium": (0.9950, 0.9900),
        "fast": (0.9800, 0.9500),
    }
    fallback = module._default_n_cliffords(2_048)  # noqa: SLF001
    selections = {
        target: module._select_decay_adapted_main_grid(  # noqa: SLF001
            (
                SimpleNamespace(decay_parameter=p_reference),
                SimpleNamespace(decay_parameter=p_interleaved),
            ),
            remaining_fractions=(0.90, 0.70, 0.50, 0.30, 0.15),
            maximum=2_048,
            fallback_grid=fallback,
        )
        for target, (p_reference, p_interleaved) in decay_pairs.items()
    }
    shared_grid = module._parallel_shared_main_grid(selections)  # noqa: SLF001

    for index, (target, (p_reference, p_interleaved)) in enumerate(decay_pairs.items()):
        common = {
            "n_trials": 24,
            "p_reference": p_reference,
            "p_interleaved": p_interleaved,
            "n_repeats": 8,
            "data_seed": 30_000 + 1_000 * index,
            "bootstrap_seed": 40_000 + 1_000 * index,
        }
        independent = _evaluate_noisy_grid_design(
            selections[target].selected_grid,
            **common,
        )
        shared = _evaluate_noisy_grid_design(shared_grid, **common)

        assert set(selections[target].selected_grid).issubset(shared_grid)
        assert abs(shared["p_reference_bias"]) < 2e-4
        assert abs(shared["p_interleaved_bias"]) < 2e-4
        assert abs(shared["fidelity_bias"]) < 2e-4
        assert shared["fidelity_sd"] <= 1.5 * independent["fidelity_sd"]
        assert shared["bootstrap_sigma"] <= 1.5 * independent["bootstrap_sigma"]


def test_extended_default_contrast_set_improves_tail_precision() -> None:
    """The 2% default tail should trade sequence depth for fit precision."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    fraction_designs = {
        "A": (0.90, 0.70, 0.50, 0.30, 0.15, 0.05, 0.02),
        "B": (0.90, 0.75, 0.60, 0.45, 0.30, 0.20),
        "C": (0.80, 0.60, 0.40, 0.20, 0.10),
    }
    fits = (
        SimpleNamespace(decay_parameter=0.995),
        SimpleNamespace(decay_parameter=0.990),
    )
    fallback = module._default_n_cliffords(2_048)  # noqa: SLF001
    selections = {
        name: module._select_decay_adapted_main_grid(  # noqa: SLF001
            fits,
            remaining_fractions=fractions,
            maximum=2_048,
            fallback_grid=fallback,
        )
        for name, fractions in fraction_designs.items()
    }
    grids = {name: selection.selected_grid for name, selection in selections.items()}
    trials = {"A": 28, "B": 31, "C": 34}
    np.testing.assert_array_equal(
        grids["A"],
        [0, 1, 10, 35, 69, 120, 189, 298, 389, 598, 780],
    )
    np.testing.assert_array_equal(selections["A"].reference_tail_grid, [598, 780])
    metrics = {
        name: _evaluate_noisy_grid_design(
            grid,
            n_trials=trials[name],
            p_reference=0.995,
            p_interleaved=0.990,
            n_repeats=20,
            data_seed=30_000,
            bootstrap_seed=40_000,
        )
        for name, grid in grids.items()
    }

    assert max(abs(item["fidelity_bias"]) for item in metrics.values()) < 2e-6
    assert min(item["ci95_coverage"] for item in metrics.values()) >= 0.9
    assert metrics["A"]["fidelity_sd"] < metrics["B"]["fidelity_sd"]
    assert metrics["A"]["bootstrap_sigma"] < metrics["B"]["bootstrap_sigma"]
    assert metrics["A"]["fidelity_sd"] < metrics["C"]["fidelity_sd"]
    assert metrics["B"]["sequence_cost"] < metrics["C"]["sequence_cost"]
    assert metrics["C"]["sequence_cost"] < metrics["A"]["sequence_cost"]
    assert (
        max(item["measurement_pairs"] for item in metrics.values())
        - min(item["measurement_pairs"] for item in metrics.values())
        <= 4
    )


def test_two_qubit_analysis_uses_dimension_four_for_fidelity() -> None:
    """The 2Q result should use d=4 in its point fit and paired bootstrap."""
    rng = np.random.default_rng(71)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    p_reference = 0.985
    p_interleaved = 0.955
    paired_noise = rng.normal(0.0, 0.002, size=(len(n_cliffords), 30))
    reference = 0.47 * p_reference ** n_cliffords[:, None] + 0.49 + paired_noise
    interleaved = 0.46 * p_interleaved ** n_cliffords[:, None] + 0.50 + paired_noise
    raw = _raw_result(
        reference,
        interleaved,
        n_cliffords=n_cliffords,
        dimension=4,
    )
    expected = 1.0 - 0.75 * (1.0 - p_interleaved / p_reference)

    result = analyze_paired_irb(raw, n_bootstrap=20, bootstrap_seed=19)

    target = result["Q0"]
    assert target["gate_fidelity"] == pytest.approx(expected, abs=2e-3)
    assert target["metadata"]["dimension"] == 4
    assert target["bootstrap"]["n_success"] == 20
    assert target["bootstrap"]["valid"] is True
    assert target["gate_fidelity_ci95"] is not None


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    """Reanalysis should reproduce the complete bootstrap distribution."""
    rng = np.random.default_rng(2)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49
    reference = reference + rng.normal(0.0, 0.003, size=(6, 20))
    interleaved = reference - 0.02 * (1.0 - 0.96 ** n_cliffords[:, None])
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    first = analyze_paired_irb(raw, n_bootstrap=25, bootstrap_seed=44)
    second = analyze_paired_irb(raw, n_bootstrap=25, bootstrap_seed=44)

    np.testing.assert_array_equal(
        first["Q0"]["bootstrap"]["fidelities"],
        second["Q0"]["bootstrap"]["fidelities"],
    )
    np.testing.assert_array_equal(
        first["Q0"]["bootstrap"]["p_reference"],
        second["Q0"]["bootstrap"]["p_reference"],
    )


def test_stratified_resampling_preserves_odd_ab_ba_counts_and_pairing() -> None:
    """Each resample should retain actual odd stratum counts and paired indices."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    first_protocols = np.asarray(
        [
            ["reference", "reference", "interleaved", "reference", "interleaved"],
            ["interleaved", "reference", "interleaved", "reference", "interleaved"],
        ]
    )
    reference = np.arange(10, dtype=np.float64).reshape(2, 5)
    interleaved = reference + 100.0
    rng = np.random.default_rng(17)

    for _ in range(30):
        indices = module._stratified_paired_resample_indices(  # noqa: SLF001
            first_protocols,
            rng=rng,
        )
        row_indices = np.arange(2)[:, None]

        np.testing.assert_array_equal(
            first_protocols[row_indices, indices],
            first_protocols,
        )
        np.testing.assert_array_equal(
            interleaved[row_indices, indices] - reference[row_indices, indices],
            np.full((2, 5), 100.0),
        )


def test_bootstrap_reports_actual_odd_ab_ba_counts_from_acquisition() -> None:
    """Bootstrap metadata should report each length's observed odd stratum counts."""
    rng = np.random.default_rng(64)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference_mean = 0.48 * 0.98**n_cliffords + 0.49
    interleaved_mean = 0.47 * 0.95**n_cliffords + 0.50
    noise = rng.normal(0.0, 0.002, size=(6, 5))
    reference = reference_mean[:, None] + noise
    interleaved = interleaved_mean[:, None] + noise
    first_protocols = np.asarray(
        [
            ["reference", "reference", "reference", "interleaved", "interleaved"],
            ["reference", "reference", "interleaved", "interleaved", "interleaved"],
        ]
        * 3
    )
    raw = _raw_result(
        reference,
        interleaved,
        n_cliffords=n_cliffords,
        first_protocols=first_protocols,
    )

    with pytest.warns(RuntimeWarning, match="n_trials"):
        result = analyze_paired_irb(raw, n_bootstrap=5, bootstrap_seed=4)

    assert (
        result["Q0"]["bootstrap"]["stratum_counts"]
        == (
            {"reference_first": 3, "interleaved_first": 2},
            {"reference_first": 2, "interleaved_first": 3},
        )
        * 3
    )


@pytest.mark.parametrize("n_trials", [2, 3])
def test_small_ab_ba_strata_do_not_report_primary_uncertainty(
    n_trials: int,
) -> None:
    """Two or three trials should retain the point fit but invalidate resampling."""
    rng = np.random.default_rng(200 + n_trials)
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    shape = (len(n_cliffords), n_trials)
    paired_noise = rng.normal(0.0, 0.003, size=shape)
    reference = (
        0.48 * 0.98 ** n_cliffords[:, None]
        + 0.50
        + paired_noise
        + rng.normal(0.0, 0.001, size=shape)
    )
    interleaved = (
        0.48 * 0.95 ** n_cliffords[:, None]
        + 0.50
        + paired_noise
        + rng.normal(0.0, 0.001, size=shape)
    )
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.warns(RuntimeWarning) as warning_records:
        result = analyze_paired_irb(raw, n_bootstrap=20, bootstrap_seed=17)

    target = result["Q0"]
    bootstrap = target["bootstrap"]
    assert np.isfinite(target["gate_fidelity"])
    assert bootstrap["optimizer_valid"] is False
    assert bootstrap["resampling_valid"] is False
    assert bootstrap["execution_skipped"] is True
    assert bootstrap["n_attempted"] == 0
    assert bootstrap["n_success"] == 0
    assert bootstrap["valid"] is False
    assert target["diagnostics"]["bootstrap_optimizer_valid"] is False
    assert target["diagnostics"]["bootstrap_resampling_valid"] is False
    assert bootstrap["invalid_reasons"] == ("insufficient_samples_per_stratum",)
    assert target["gate_fidelity_err"] is None
    assert target["gate_fidelity_ci68"] is None
    assert target["gate_fidelity_ci95"] is None
    assert target["uncertainty_method"] == "paired_bootstrap_resampling_invalid"
    assert any(
        "two samples per AB/BA stratum" in str(record.message)
        for record in warning_records
    )


def test_invalid_bootstrap_resampling_skips_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degenerate AB/BA design must not run meaningless bootstrap fits."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16], dtype=np.int64)
    offsets = np.asarray([-0.002, 0.002])
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49 + offsets
    interleaved = 0.47 * 0.95 ** n_cliffords[:, None] + 0.50 + offsets
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    original_fit = module._fit_rb_decay  # noqa: SLF001
    call_count = 0

    def count_fits(*args: Any, **kwargs: Any) -> Any:
        """Count the two required full-data fits and any unexpected fit."""
        nonlocal call_count
        call_count += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_rb_decay", count_fits)

    with pytest.warns(RuntimeWarning) as warning_records:
        result = analyze_paired_irb(raw, n_bootstrap=3, bootstrap_seed=17)

    bootstrap = result["Q0"]["bootstrap"]
    assert call_count == 2
    assert bootstrap["execution_skipped"] is True
    assert bootstrap["n_attempted"] == 0
    assert bootstrap["n_success"] == 0
    assert bootstrap["invalid_reasons"] == ("insufficient_samples_per_stratum",)
    warning_messages = tuple(str(record.message) for record in warning_records)
    assert any(
        "two samples per AB/BA stratum" in message for message in warning_messages
    )
    assert not any("Fewer than two" in message for message in warning_messages)


def test_four_trials_support_non_degenerate_stratified_bootstrap() -> None:
    """Two samples in each AB/BA stratum should permit varying uncertainty."""
    rng = np.random.default_rng(204)
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    n_trials = 4
    shape = (len(n_cliffords), n_trials)
    paired_noise = rng.normal(0.0, 0.004, size=shape)
    reference = (
        0.48 * 0.98 ** n_cliffords[:, None]
        + 0.50
        + paired_noise
        + rng.normal(0.0, 0.002, size=shape)
    )
    interleaved = (
        0.48 * 0.95 ** n_cliffords[:, None]
        + 0.50
        + paired_noise
        + rng.normal(0.0, 0.002, size=shape)
    )
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.warns(RuntimeWarning, match="n_trials"):
        result = analyze_paired_irb(raw, n_bootstrap=40, bootstrap_seed=19)

    target = result["Q0"]
    bootstrap = target["bootstrap"]
    assert bootstrap["stratum_counts"] == (
        {"reference_first": 2, "interleaved_first": 2},
    ) * len(n_cliffords)
    assert bootstrap["optimizer_valid"] is True
    assert bootstrap["resampling_valid"] is True
    assert bootstrap["valid"] is True
    assert target["diagnostics"]["bootstrap_optimizer_valid"] is True
    assert target["diagnostics"]["bootstrap_resampling_valid"] is True
    assert bootstrap["invalid_reasons"] == ()
    assert np.ptp(bootstrap["fidelities"]) > 0.0
    assert bootstrap["fidelity_std"] > 0.0
    assert target["gate_fidelity_err"] > 0.0
    assert target["gate_fidelity_ci95"] is not None


def test_stratification_avoids_variance_from_artificial_order_imbalance() -> None:
    """Stratification should remove variance caused only by changing AB/BA counts."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    first_protocols = np.asarray(
        [["reference", "reference", "reference", "interleaved", "interleaved"]]
    )
    order_effect = np.asarray([[1.0, 1.0, 1.0, 0.0, 0.0]])
    no_order_effect = np.ones((1, 5), dtype=np.float64)
    ordinary_rng = np.random.default_rng(91)
    stratified_rng = np.random.default_rng(91)
    ordinary_means = []
    stratified_means = []

    for _ in range(200):
        ordinary_indices = ordinary_rng.integers(0, 5, size=(1, 5))
        stratified_indices = module._stratified_paired_resample_indices(  # noqa: SLF001
            first_protocols,
            rng=stratified_rng,
        )
        ordinary_means.append(float(np.mean(order_effect[:, ordinary_indices[0]])))
        stratified_means.append(float(np.mean(order_effect[:, stratified_indices[0]])))
        assert np.mean(no_order_effect[:, ordinary_indices[0]]) == pytest.approx(1.0)
        assert np.mean(no_order_effect[:, stratified_indices[0]]) == pytest.approx(1.0)

    assert np.var(ordinary_means) > 0.01
    assert np.var(stratified_means) == pytest.approx(0.0, abs=1e-15)


def test_ordinary_and_stratified_resampling_agree_without_order_effect() -> None:
    """Equivalent AB/BA strata should give comparable bootstrap mean variance."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    first_protocols = np.asarray([[*["reference"] * 15, *["interleaved"] * 15]])
    shared_values = np.linspace(-1.0, 1.0, 15)
    no_order_effect = np.asarray([[*shared_values, *shared_values]])
    ordinary_rng = np.random.default_rng(29)
    stratified_rng = np.random.default_rng(29)
    ordinary_means = []
    stratified_means = []

    for _ in range(2_000):
        ordinary_indices = ordinary_rng.integers(0, 30, size=(1, 30))
        stratified_indices = module._stratified_paired_resample_indices(  # noqa: SLF001
            first_protocols,
            rng=stratified_rng,
        )
        ordinary_means.append(float(np.mean(no_order_effect[:, ordinary_indices[0]])))
        stratified_means.append(
            float(np.mean(no_order_effect[:, stratified_indices[0]]))
        )

    assert np.mean(stratified_means) == pytest.approx(
        np.mean(ordinary_means),
        abs=0.01,
    )
    assert np.var(stratified_means) == pytest.approx(
        np.var(ordinary_means),
        rel=0.1,
    )


def test_analysis_rejects_unpaired_trial_shapes() -> None:
    """Analysis should reject trial matrices that cannot be paired cell-wise."""
    n_cliffords = np.array([0, 1, 2, 4], dtype=np.int64)
    reference = np.full((4, 3), 0.9)
    interleaved = np.full((4, 2), 0.8)
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.raises(ValueError, match="same shape"):
        analyze_paired_irb(raw, n_bootstrap=10)


def test_analysis_rejects_n_trials_metadata_mismatch() -> None:
    """Recorded trial count must match the paired matrix columns."""
    n_cliffords = np.array([0, 1, 2, 4], dtype=np.int64)
    trials = np.full((4, 3), 0.9)
    raw = _raw_result(trials, trials.copy(), n_cliffords=n_cliffords)
    raw.data["Q0"]["acquisition"]["n_trials"] = 4

    with pytest.raises(ValueError, match=r"n_trials.*columns"):
        analyze_paired_irb(raw, n_bootstrap=0)


def test_analysis_rejects_an_unknown_raw_schema_version() -> None:
    """Analysis should fail explicitly when the raw schema is unsupported."""
    n_cliffords = np.array([0, 1, 2, 4], dtype=np.int64)
    trials = np.full((4, 3), 0.9)
    raw = _raw_result(
        trials,
        trials.copy(),
        n_cliffords=n_cliffords,
        schema_version=2,
    )

    with pytest.raises(ValueError, match=r"schema_version.*2"):
        analyze_paired_irb(raw, n_bootstrap=0)


def test_analysis_rejects_inconsistent_target_dimension_metadata() -> None:
    """Target kind and Hilbert-space dimension must describe the same system."""
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    offsets = np.linspace(-0.003, 0.003, 20)
    mean = 0.48 * 0.97**n_cliffords + 0.5
    trials = mean[:, None] + offsets[None, :]
    raw = _raw_result(trials, trials.copy(), n_cliffords=n_cliffords, dimension=2)
    raw.data["Q0"]["metadata"]["is_two_qubit"] = True

    with pytest.raises(ValueError, match=r"dimension.*is_two_qubit"):
        analyze_paired_irb(raw, n_bootstrap=0)


def test_analysis_requires_complete_acquisition_order_metadata() -> None:
    """Schema-v1 analysis should require one AB/BA label for every trial cell."""
    n_cliffords = np.array([0, 1, 2, 4], dtype=np.int64)
    offsets = np.linspace(-0.003, 0.003, 4)
    mean = 0.48 * 0.97**n_cliffords + 0.5
    trials = mean[:, None] + offsets[None, :]
    raw = _raw_result(trials, trials.copy(), n_cliffords=n_cliffords)
    del raw.data["Q0"]["acquisition"]["planned_order"]

    with pytest.raises(TypeError, match="planned_order"):
        analyze_paired_irb(raw, n_bootstrap=0)


@pytest.mark.parametrize(
    "invalid_case",
    ["wrong_count", "invalid_cell", "duplicate_cell", "invalid_protocol", "imbalanced"],
)
def test_analysis_rejects_invalid_acquisition_order_metadata(
    invalid_case: str,
) -> None:
    """Malformed or unbalanced AB/BA metadata should fail before resampling."""
    n_cliffords = np.array([0, 1, 2, 4], dtype=np.int64)
    offsets = np.linspace(-0.003, 0.003, 4)
    mean = 0.48 * 0.97**n_cliffords + 0.5
    trials = mean[:, None] + offsets[None, :]
    raw = _raw_result(trials, trials.copy(), n_cliffords=n_cliffords)
    plan = [dict(pair) for pair in raw.data["Q0"]["acquisition"]["planned_order"]]
    if invalid_case == "wrong_count":
        plan.pop()
    elif invalid_case == "invalid_cell":
        plan[0]["length_index"] = len(n_cliffords)
    elif invalid_case == "duplicate_cell":
        plan[1]["length_index"] = plan[0]["length_index"]
        plan[1]["trial_index"] = plan[0]["trial_index"]
    elif invalid_case == "invalid_protocol":
        plan[0]["first_protocol"] = "unknown"
    else:
        for pair in plan[:4]:
            pair["first_protocol"] = "reference"
    raw.data["Q0"]["acquisition"]["planned_order"] = plan

    with pytest.raises(ValueError, match=r"planned_order|acquisition-plan|AB/BA"):
        analyze_paired_irb(raw, n_bootstrap=0)


@pytest.mark.parametrize(
    ("error_bar", "error_type"),
    [(1, TypeError), ("variance", ValueError)],
)
def test_analysis_rejects_invalid_error_bar_values(
    error_bar: Any,
    error_type: type[Exception],
) -> None:
    """Error-bar type errors and unsupported names should remain distinguishable."""
    n_cliffords = np.array([0, 1, 2, 4], dtype=np.int64)
    offsets = np.array([-0.01, 0.0, 0.01])
    means = 0.48 * 0.97**n_cliffords + 0.5
    trials = means[:, None] + offsets[None, :]
    raw = _raw_result(trials, trials.copy(), n_cliffords=n_cliffords)

    with pytest.raises(error_type, match="error_bar"):
        analyze_paired_irb(
            raw,
            n_bootstrap=0,
            error_bar=error_bar,
        )


def test_automatic_sem_floor_uses_the_paired_empirical_sem_scale() -> None:
    """Zero-SEM points should receive a finite weight comparable to other points."""
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    offsets = np.linspace(-1.0, 1.0, 20)
    reference_mean = 0.48 * 0.98**n_cliffords + 0.49
    interleaved_mean = 0.47 * 0.95**n_cliffords + 0.50
    reference_scales = np.array([0.0, 0.002, 0.003, 0.004, 0.005, 0.006])
    interleaved_scales = np.array([0.0, 0.003, 0.004, 0.005, 0.006, 0.007])
    reference = reference_mean[:, None] + reference_scales[:, None] * offsets
    interleaved = interleaved_mean[:, None] + interleaved_scales[:, None] * offsets
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    reference_sem = np.std(reference, axis=1, ddof=1) / np.sqrt(20)
    interleaved_sem = np.std(interleaved, axis=1, ddof=1) / np.sqrt(20)
    reference_sem[reference_scales == 0.0] = 0.0
    interleaved_sem[interleaved_scales == 0.0] = 0.0
    positive_sem = np.concatenate(
        (reference_sem[reference_sem > 0], interleaved_sem[interleaved_sem > 0])
    )
    expected_floor = 0.25 * float(np.median(positive_sem))

    with pytest.warns(RuntimeWarning, match="SEM floor"):
        result = analyze_paired_irb(raw, n_bootstrap=0)

    target = result["Q0"]
    assert target["diagnostics"]["sem_floor"] == pytest.approx(expected_floor)
    assert target["diagnostics"]["sem_floor_mode"] == "automatic"
    assert target["reference"]["sem_used"][0] == pytest.approx(expected_floor)
    assert target["interleaved"]["sem_used"][0] == pytest.approx(expected_floor)
    assert target["diagnostics"]["sem_floor_applied"] == {
        "reference": 1,
        "interleaved": 1,
    }


def test_all_zero_sem_uses_probability_scale_fallback_during_bootstrap() -> None:
    """Exact rows should fit and bootstrap without machine-epsilon weights."""
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference_mean = 0.48 * 0.98**n_cliffords + 0.49
    interleaved_mean = 0.47 * 0.95**n_cliffords + 0.50
    reference = np.broadcast_to(reference_mean[:, None], (6, 20)).copy()
    interleaved = np.broadcast_to(interleaved_mean[:, None], (6, 20)).copy()
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.warns(RuntimeWarning) as warning_records:
        result = analyze_paired_irb(raw, n_bootstrap=20, bootstrap_seed=7)

    assert any("SEM floor" in str(record.message) for record in warning_records)
    assert any("non-empirical" in str(record.message) for record in warning_records)

    target = result["Q0"]
    assert target["diagnostics"]["sem_floor"] == pytest.approx(1e-3)
    assert target["diagnostics"]["sem_floor_applied"] == {
        "reference": 6,
        "interleaved": 6,
    }
    assert np.all(np.isfinite(target["reference"]["fit"]["predicted"]))
    assert np.all(np.isfinite(target["interleaved"]["fit"]["predicted"]))
    assert target["bootstrap"]["n_success"] == 20
    assert target["bootstrap"]["valid"] is True


def test_explicit_sem_floor_is_recorded_as_a_fixed_analysis_option() -> None:
    """A numeric SEM floor should remain fixed and identifiable in the result."""
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    offsets = np.linspace(-0.002, 0.002, 20)
    reference_mean = 0.48 * 0.98**n_cliffords + 0.49
    interleaved_mean = 0.47 * 0.95**n_cliffords + 0.50
    reference = reference_mean[:, None] + offsets[None, :]
    interleaved = interleaved_mean[:, None] + offsets[None, :]
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.warns(RuntimeWarning) as warning_records:
        result = analyze_paired_irb(raw, n_bootstrap=5, sem_floor=0.01)

    assert any("SEM floor" in str(record.message) for record in warning_records)
    assert any("non-empirical" in str(record.message) for record in warning_records)

    diagnostics = result["Q0"]["diagnostics"]
    assert diagnostics["sem_floor"] == pytest.approx(0.01)
    assert diagnostics["sem_floor_mode"] == "fixed"
    assert diagnostics["fit_covariance_scale_empirical"] is False
    assert diagnostics["gate_fidelity_err_fit_covariance"] is None


def test_sem_floor_candidates_are_stable_for_typical_synthetic_data() -> None:
    """Reasonable SEM floors should agree on typical fit and bootstrap outputs."""
    rng = np.random.default_rng(2026)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    n_trials = 30
    p_reference = 0.985
    p_interleaved = 0.955
    reference_mean = 0.47 * p_reference**n_cliffords + 0.49
    interleaved_mean = 0.46 * p_interleaved**n_cliffords + 0.50
    noise = rng.normal(size=(len(n_cliffords), n_trials))
    scales = np.linspace(0.0015, 0.004, len(n_cliffords))[:, None]
    reference = reference_mean[:, None] + scales * noise
    interleaved = (
        interleaved_mean[:, None]
        + scales * noise
        + 0.0006 * rng.normal(size=noise.shape)
    )
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    combined_sem = np.concatenate(
        (
            np.std(reference, axis=1, ddof=1) / np.sqrt(n_trials),
            np.std(interleaved, axis=1, ddof=1) / np.sqrt(n_trials),
        )
    )
    positive_sem = combined_sem[combined_sem > 0.0]
    median_sem = float(np.median(positive_sem))
    candidate_floors = (
        0.1 * median_sem,
        0.25 * median_sem,
        float(np.percentile(positive_sem, 10)),
        0.5 * median_sem,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        targets = [
            analyze_paired_irb(
                raw,
                n_bootstrap=30,
                bootstrap_seed=7,
                sem_floor=floor,
            )["Q0"]
            for floor in candidate_floors
        ]

    p_reference_values = [target["reference"]["fit"]["p"] for target in targets]
    p_interleaved_values = [target["interleaved"]["fit"]["p"] for target in targets]
    fidelities = [target["gate_fidelity"] for target in targets]
    bootstrap_sigmas = [target["gate_fidelity_err"] for target in targets]
    ci95 = [target["gate_fidelity_ci95"] for target in targets]
    residual_rms = [
        float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        (
                            target["reference"]["fit"]["residuals"],
                            target["interleaved"]["fit"]["residuals"],
                        )
                    )
                    ** 2
                )
            )
        )
        for target in targets
    ]

    assert np.ptp(p_reference_values) < 5e-5
    assert np.ptp(p_interleaved_values) < 5e-5
    assert np.ptp(fidelities) < 1e-5
    assert all(value is not None for value in bootstrap_sigmas)
    assert np.ptp(bootstrap_sigmas) < 1e-5
    assert all(value is not None for value in ci95)
    assert np.ptp([value[0] for value in ci95]) < 5e-5
    assert np.ptp([value[1] for value in ci95]) < 5e-5
    assert np.ptp(residual_rms) < 5e-6


def test_sem_floor_candidates_are_stable_when_only_m_zero_has_zero_sem() -> None:
    """A zero-SEM origin should not make fit or bootstrap outputs floor-sensitive."""
    rng = np.random.default_rng(2028)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    n_trials = 30
    reference_mean = 0.47 * 0.985**n_cliffords + 0.49
    interleaved_mean = 0.46 * 0.955**n_cliffords + 0.50
    noise = rng.normal(size=(len(n_cliffords), n_trials))
    scales = np.full((len(n_cliffords), 1), 0.003)
    scales[0] = 0.0
    reference = reference_mean[:, None] + scales * noise
    interleaved = interleaved_mean[:, None] + scales * noise
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    combined_sem = np.concatenate(
        (
            np.std(reference, axis=1, ddof=1) / np.sqrt(n_trials),
            np.std(interleaved, axis=1, ddof=1) / np.sqrt(n_trials),
        )
    )
    positive_sem = combined_sem[combined_sem > 0.0]
    median_sem = float(np.median(positive_sem))
    candidate_floors = (
        0.1 * median_sem,
        0.25 * median_sem,
        float(np.percentile(positive_sem, 10)),
        0.5 * median_sem,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        targets = [
            analyze_paired_irb(
                raw,
                n_bootstrap=20,
                bootstrap_seed=7,
                sem_floor=floor,
            )["Q0"]
            for floor in candidate_floors
        ]

    assert np.ptp([target["reference"]["fit"]["p"] for target in targets]) < 2e-4
    assert np.ptp([target["interleaved"]["fit"]["p"] for target in targets]) < 2e-4
    assert np.ptp([target["gate_fidelity"] for target in targets]) < 1e-5
    assert np.ptp([target["gate_fidelity_err"] for target in targets]) < 3e-5
    assert np.ptp([target["gate_fidelity_ci95"][0] for target in targets]) < 1e-4
    assert np.ptp([target["gate_fidelity_ci95"][1] for target in targets]) < 1e-4


def test_conservative_sem_floor_reduces_tiny_sem_point_influence() -> None:
    """The default floor should reduce bias from one implausibly precise point."""
    rng = np.random.default_rng(2027)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    n_trials = 30
    p_reference = 0.985
    p_interleaved = 0.955
    reference_mean = 0.47 * p_reference**n_cliffords + 0.49
    interleaved_mean = 0.46 * p_interleaved**n_cliffords + 0.50
    noise = rng.normal(size=(len(n_cliffords), n_trials))
    scales = np.full((len(n_cliffords), 1), 0.003)
    scales[2] = 1e-6
    reference = reference_mean[:, None] + scales * noise
    interleaved = interleaved_mean[:, None] + scales * noise
    reference[2] += 0.004
    interleaved[2] -= 0.004
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    combined_sem = np.concatenate(
        (
            np.std(reference, axis=1, ddof=1) / np.sqrt(n_trials),
            np.std(interleaved, axis=1, ddof=1) / np.sqrt(n_trials),
        )
    )
    median_sem = float(np.median(combined_sem[combined_sem > 0.0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low_floor = analyze_paired_irb(
            raw,
            n_bootstrap=30,
            bootstrap_seed=7,
            sem_floor=0.1 * median_sem,
        )["Q0"]
        default_floor = analyze_paired_irb(
            raw,
            n_bootstrap=30,
            bootstrap_seed=7,
        )["Q0"]

    expected_fidelity = 1.0 - 0.5 * (1.0 - p_interleaved / p_reference)
    low_residual = np.linalg.norm(
        np.concatenate(
            (
                low_floor["reference"]["fit"]["residuals"],
                low_floor["interleaved"]["fit"]["residuals"],
            )
        )
    )
    default_residual = np.linalg.norm(
        np.concatenate(
            (
                default_floor["reference"]["fit"]["residuals"],
                default_floor["interleaved"]["fit"]["residuals"],
            )
        )
    )

    assert default_floor["diagnostics"]["sem_floor"] == pytest.approx(0.25 * median_sem)
    assert abs(default_floor["gate_fidelity"] - expected_fidelity) < abs(
        low_floor["gate_fidelity"] - expected_fidelity
    )
    assert default_residual < low_residual
    assert default_floor["gate_fidelity_err"] is not None
    assert default_floor["gate_fidelity_ci95"] is not None


def test_fit_bound_hits_are_reported_without_rejecting_bootstrap_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound-hit fits should remain sampled while making fit quality invalid."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    rng = np.random.default_rng(31)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49
    reference = reference + rng.normal(0.0, 0.003, size=(6, 20))
    interleaved = reference - 0.01 * (1.0 - 0.95 ** n_cliffords[:, None])
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    original_fit = module._fit_rb_decay  # noqa: SLF001
    call_count = 0

    def mark_selected_fits_at_bounds(*args: Any, **kwargs: Any) -> Any:
        """Mark full fits and selected successful bootstrap fits at bounds."""
        nonlocal call_count
        call_count += 1
        fit = original_fit(*args, **kwargs)
        if call_count == 1:
            return replace(fit, parameters_at_bound=("A",))
        if call_count == 2:
            return replace(fit, parameters_at_bound=("p",))
        replicate_index = (call_count - 3) // 2
        is_reference = (call_count - 3) % 2 == 0
        if is_reference and replicate_index < 6:
            return replace(fit, parameters_at_bound=("p",))
        if not is_reference and replicate_index < 4:
            return replace(fit, parameters_at_bound=("A",))
        return fit

    monkeypatch.setattr(module, "_fit_rb_decay", mark_selected_fits_at_bounds)

    with pytest.warns(RuntimeWarning) as warning_records:
        result = analyze_paired_irb(raw, n_bootstrap=20, bootstrap_seed=12)

    target = result["Q0"]
    bootstrap = target["bootstrap"]
    diagnostics = target["diagnostics"]
    assert target["gate_fidelity"] is not None
    assert target["gate_fidelity_err"] is not None
    assert bootstrap["n_success"] == 20
    assert bootstrap["valid"] is True
    assert bootstrap["parameter_bound_quality_valid"] is False
    assert bootstrap["parameter_bound_quality_thresholds"] == {
        "any_bound_fraction_max": 0.10,
        "p_bound_fraction_max": 0.05,
    }
    assert bootstrap["reference_any_bound_count"] == 6
    assert bootstrap["reference_any_bound_fraction"] == pytest.approx(0.3)
    assert bootstrap["reference_p_bound_count"] == 6
    assert bootstrap["reference_p_bound_fraction"] == pytest.approx(0.3)
    assert bootstrap["interleaved_any_bound_count"] == 4
    assert bootstrap["interleaved_any_bound_fraction"] == pytest.approx(0.2)
    assert bootstrap["interleaved_p_bound_count"] == 0
    assert bootstrap["interleaved_p_bound_fraction"] == pytest.approx(0.0)
    assert diagnostics["parameter_bound_quality_valid"] is False
    assert diagnostics["reference_p_at_bound"] is False
    assert diagnostics["interleaved_p_at_bound"] is True
    assert diagnostics["bootstrap_parameter_bound_quality_valid"] is False
    assert any(
        "bootstrap" in str(record.message) and "bound" in str(record.message)
        for record in warning_records
    )


def test_bootstrap_skips_failed_fits_and_retains_successful_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed bootstrap fit should not discard valid replicates or raw data."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    rng = np.random.default_rng(8)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    trials = 0.47 * 0.97 ** n_cliffords[:, None] + 0.5
    trials = trials + rng.normal(0.0, 0.003, size=(6, 20))
    raw = _raw_result(trials, trials - 0.002, n_cliffords=n_cliffords)
    original_fit = module._fit_rb_decay  # noqa: SLF001
    call_count = 0

    def fail_first_bootstrap_fit(*args: Any, **kwargs: Any) -> Any:
        """Fail only the first fit after the two full-data fits."""
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("synthetic optimizer failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_rb_decay", fail_first_bootstrap_fit)

    result = analyze_paired_irb(raw, n_bootstrap=10, bootstrap_seed=1)

    bootstrap = result["Q0"]["bootstrap"]
    assert bootstrap["n_success"] == 9
    assert bootstrap["n_failed"] == 1
    assert bootstrap["fidelities"].shape == (9,)


def test_analysis_reports_when_every_bootstrap_fit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero successful replicates should explicitly report unavailable intervals."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    rng = np.random.default_rng(21)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    trials = 0.47 * 0.97 ** n_cliffords[:, None] + 0.5
    trials = trials + rng.normal(0.0, 0.003, size=(6, 20))
    raw = _raw_result(trials, trials - 0.002, n_cliffords=n_cliffords)
    original_fit = module._fit_rb_decay  # noqa: SLF001
    call_count = 0

    def fail_every_bootstrap_fit(*args: Any, **kwargs: Any) -> Any:
        """Allow full-data fits and reject all bootstrap fits."""
        nonlocal call_count
        call_count += 1
        if call_count > 2:
            raise RuntimeError("synthetic optimizer failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_rb_decay", fail_every_bootstrap_fit)

    with pytest.warns(RuntimeWarning) as warning_records:
        result = analyze_paired_irb(raw, n_bootstrap=3, bootstrap_seed=2)

    assert any("Fewer than two" in str(record.message) for record in warning_records)
    bootstrap_warnings = [
        record
        for record in warning_records
        if "bootstrap replicates" in str(record.message)
    ]
    assert len(bootstrap_warnings) == 1
    target = result["Q0"]
    assert target["bootstrap"]["n_success"] == 0
    assert target["gate_fidelity_err"] is None
    assert target["gate_fidelity_ci95"] is None
    assert target["uncertainty_method"] == "unavailable"
    assert target["bootstrap"]["valid"] is False
    assert target["bootstrap"]["optimizer_valid"] is False
    assert target["bootstrap"]["parameter_bound_quality_valid"] is None


def test_low_bootstrap_success_rate_invalidates_primary_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected subset of bootstrap fits must not define primary uncertainty."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    rng = np.random.default_rng(22)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49
    reference = reference + rng.normal(0.0, 0.003, size=(6, 20))
    interleaved = reference - 0.01 * (1.0 - 0.95 ** n_cliffords[:, None])
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    original_fit = module._fit_rb_decay  # noqa: SLF001
    call_count = 0

    def reject_three_bootstrap_replicates(*args: Any, **kwargs: Any) -> Any:
        """Allow full-data fits, reject three replicates, then allow two."""
        nonlocal call_count
        call_count += 1
        if 3 <= call_count <= 5:
            raise RuntimeError("synthetic optimizer failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_rb_decay", reject_three_bootstrap_replicates)
    monkeypatch.setattr(module.go.Figure, "show", lambda self: None)

    with pytest.warns(RuntimeWarning, match="80%"):
        result = analyze_paired_irb(
            raw,
            n_bootstrap=5,
            bootstrap_seed=3,
            plot=True,
        )

    target = result["Q0"]
    assert target["bootstrap"]["n_success"] == 2
    assert target["bootstrap"]["success_rate"] == pytest.approx(0.4)
    assert target["bootstrap"]["valid"] is False
    assert target["bootstrap"]["optimizer_valid"] is False
    assert target["bootstrap"]["parameter_bound_quality_valid"] is True
    assert target["bootstrap"]["fidelity_std"] is not None
    assert target["gate_fidelity_err"] is None
    assert target["gate_fidelity_ci68"] is None
    assert target["gate_fidelity_ci95"] is None
    assert target["uncertainty_method"] == "paired_bootstrap_unreliable"
    assert result.figure is not None
    figure: Any = result.figure
    annotation = figure.layout.annotations[0].text
    assert "bootstrap σ = unavailable" in annotation
    assert "95% CI = unavailable" in annotation


def test_fidelity_estimates_outside_the_physical_range_are_not_clipped() -> None:
    """Finite-sample fidelity values above one should remain visible diagnostics."""
    rng = np.random.default_rng(12)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    paired_noise = rng.normal(0.0, 0.001, size=(8, 20))
    reference = 0.47 * 0.96 ** n_cliffords[:, None] + 0.50 + paired_noise
    interleaved = 0.47 * 0.99 ** n_cliffords[:, None] + 0.50 + paired_noise
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.warns(RuntimeWarning, match="not clipped"):
        result = analyze_paired_irb(raw, n_bootstrap=20, bootstrap_seed=6)

    assert result["Q0"]["gate_fidelity"] > 1.0
    assert np.any(result["Q0"]["bootstrap"]["fidelities"] > 1.0)


def test_one_successful_bootstrap_sample_does_not_define_an_interval() -> None:
    """A single bootstrap value should not be reported as a confidence interval."""
    rng = np.random.default_rng(18)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49
    reference = reference + rng.normal(0.0, 0.003, size=(6, 20))
    interleaved = reference - 0.01 * (1.0 - 0.95 ** n_cliffords[:, None])
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.warns(RuntimeWarning, match="Fewer than two"):
        result = analyze_paired_irb(raw, n_bootstrap=1, bootstrap_seed=4)

    assert result["Q0"]["bootstrap"]["n_success"] == 1
    assert result["Q0"]["bootstrap"]["valid"] is False
    assert result["Q0"]["gate_fidelity_err"] is None
    assert result["Q0"]["uncertainty_method"] == "unavailable"


def test_unavailable_fit_covariance_does_not_discard_decay_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit covariance failure should remain diagnostic when decay fits converge."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    rng = np.random.default_rng(20)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49
    reference = reference + rng.normal(0.0, 0.003, size=(6, 20))
    interleaved = reference - 0.01 * (1.0 - 0.95 ** n_cliffords[:, None])
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)
    original_curve_fit = module.curve_fit

    def return_invalid_covariance(*args: Any, **kwargs: Any) -> Any:
        """Retain converged parameters while replacing covariance with infinity."""
        parameters, _ = original_curve_fit(*args, **kwargs)
        return parameters, np.full((3, 3), np.inf)

    monkeypatch.setattr(module, "curve_fit", return_invalid_covariance)

    with pytest.warns(RuntimeWarning, match="covariance is unavailable"):
        result = analyze_paired_irb(raw, n_bootstrap=2, bootstrap_seed=5)

    target = result["Q0"]
    assert target["reference"]["fit"]["covariance"] is None
    assert target["reference"]["fit"]["standard_errors"] is None
    assert target["bootstrap"]["n_success"] == 2
    assert target["diagnostics"]["gate_fidelity_err_fit_covariance"] is None
    assert target["gate_fidelity_ci68"] is not None
    assert target["gate_fidelity_ci95"] is not None


def test_all_zero_sem_marks_fit_covariance_scale_as_nonempirical() -> None:
    """All-zero SEM should suppress covariance-derived fidelity uncertainty."""
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference_mean = 0.48 * 0.98**n_cliffords + 0.49
    interleaved_mean = 0.47 * 0.95**n_cliffords + 0.50
    reference = np.broadcast_to(reference_mean[:, None], (6, 20)).copy()
    interleaved = np.broadcast_to(interleaved_mean[:, None], (6, 20)).copy()
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.warns(RuntimeWarning) as warning_records:
        result = analyze_paired_irb(raw, n_bootstrap=0)

    assert any("SEM floor" in str(record.message) for record in warning_records)
    assert any("non-empirical" in str(record.message) for record in warning_records)

    diagnostics = result["Q0"]["diagnostics"]
    assert diagnostics["fit_covariance_scale_empirical"] is False
    assert diagnostics["gate_fidelity_err_fit_covariance"] is None


def test_fewer_than_six_clifford_lengths_produces_a_diagnostic_warning() -> None:
    """A low-degree-of-freedom decay fit should remain usable but be flagged."""
    n_cliffords = np.array([0, 1, 2, 4, 8], dtype=np.int64)
    offsets = np.linspace(-0.003, 0.003, 20)
    reference_mean = 0.48 * 0.98**n_cliffords + 0.49
    interleaved_mean = 0.47 * 0.95**n_cliffords + 0.50
    reference = reference_mean[:, None] + offsets[None, :]
    interleaved = interleaved_mean[:, None] + offsets[None, :]
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    with pytest.warns(RuntimeWarning, match="Clifford-length points"):
        result = analyze_paired_irb(raw, n_bootstrap=0)

    assert any(
        "Clifford-length points" in message
        for message in result["Q0"]["diagnostics"]["warnings"]
    )


@pytest.mark.parametrize(
    ("error_bar", "statistics_key"),
    [(None, None), ("sem", "sem"), ("std", "std")],
)
def test_figure_uses_the_selected_error_bars(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error_bar: str | None,
    statistics_key: str | None,
) -> None:
    """The public analysis API should plot the selected mean uncertainty."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    monkeypatch.setattr(module.go.Figure, "show", lambda self: None)
    rng = np.random.default_rng(19)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49
    reference = reference + rng.normal(0.0, 0.003, size=(6, 20))
    interleaved = reference - 0.01 * (1.0 - 0.95 ** n_cliffords[:, None])
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    result = analyze_paired_irb(
        raw,
        n_bootstrap=0,
        error_bar=error_bar,  # type: ignore[arg-type]
        plot=True,
    )

    assert result["Q0"]["gate_fidelity_err"] is None
    assert result.figure is not None
    figure: Any = result.figure
    assert len(figure.data) == 4
    assert figure.data[0].name == "Reference"
    assert figure.data[0].mode == "lines"
    assert figure.data[0].line.color == "#0C5DA5"
    assert figure.data[1].mode == "markers"
    assert figure.data[1].marker.color == "#0C5DA5"
    assert figure.data[1].showlegend is False
    assert figure.data[2].name == "Interleaved"
    assert figure.data[2].mode == "lines"
    assert figure.data[2].line.color == "#00B945"
    assert figure.data[3].mode == "markers"
    assert figure.data[3].marker.color == "#00B945"
    assert figure.data[3].showlegend is False
    assert figure.layout.yaxis.title.text == "Normalized signal"
    if statistics_key is None:
        assert figure.data[1].error_y.array is None
        assert figure.data[3].error_y.array is None
    else:
        np.testing.assert_allclose(
            figure.data[1].error_y.array,
            result["Q0"]["reference"][statistics_key],
        )
        np.testing.assert_allclose(
            figure.data[3].error_y.array,
            result["Q0"]["interleaved"][statistics_key],
        )


def test_figure_title_reports_fidelity_and_uncertainty_in_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Figure annotations should not mix fractions and percentage points."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    monkeypatch.setattr(module.go.Figure, "show", lambda self: None)
    rng = np.random.default_rng(23)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49
    reference = reference + rng.normal(0.0, 0.003, size=(6, 20))
    interleaved = reference - 0.01 * (1.0 - 0.95 ** n_cliffords[:, None])
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    result = analyze_paired_irb(raw, n_bootstrap=10, bootstrap_seed=4, plot=True)

    target = result["Q0"]
    figure: Any = result.figure
    title = figure.layout.title.text
    annotation = figure.layout.annotations[0].text
    ci95 = target["gate_fidelity_ci95"]
    assert ci95 is not None
    assert title == "Paired interleaved randomized benchmarking of X90 : Q0"
    assert f"F = {100.0 * target['gate_fidelity']:.4f} %" in annotation
    assert f"bootstrap σ = {100.0 * target['gate_fidelity_err']:.4f} pp" in annotation
    assert f"95% CI = [{100.0 * ci95[0]:.4f}, {100.0 * ci95[1]:.4f}] %" in annotation


def test_save_image_builds_and_saves_the_figure_without_displaying_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving should create one named figure even when interactive plot is disabled."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    save_figure = Mock()
    monkeypatch.setattr(module.viz, "save_figure", save_figure)
    rng = np.random.default_rng(29)
    n_cliffords = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = 0.48 * 0.98 ** n_cliffords[:, None] + 0.49
    reference = reference + rng.normal(0.0, 0.003, size=(6, 20))
    interleaved = reference - 0.01 * (1.0 - 0.95 ** n_cliffords[:, None])
    raw = _raw_result(reference, interleaved, n_cliffords=n_cliffords)

    result = analyze_paired_irb(raw, n_bootstrap=0, save_image=True)

    assert result.figure is not None
    save_figure.assert_called_once_with(
        result.figure,
        name="paired_interleaved_randomized_benchmarking_Q0",
    )
