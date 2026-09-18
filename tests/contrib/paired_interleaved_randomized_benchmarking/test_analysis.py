"""Focused statistical tests for paired interleaved randomized benchmarking."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

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
    """Build a minimal valid measurement-only result for analysis tests."""
    n_trials = reference_trials.shape[1]
    if first_protocols is None:
        first_protocols = np.empty((len(n_cliffords), n_trials), dtype="<U11")
        for length_index in range(len(n_cliffords)):
            half = n_trials // 2
            first_protocols[length_index, :half] = "reference"
            first_protocols[length_index, half:] = "interleaved"
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


def _synthetic_trials(
    *,
    p_reference: float = 0.985,
    p_interleaved: float = 0.955,
    n_trials: int = 20,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    rng = np.random.default_rng(seed)
    paired_noise = rng.normal(0.0, 0.002, size=(len(grid), n_trials))
    reference = 0.47 * p_reference ** grid[:, None] + 0.49 + paired_noise
    interleaved = 0.46 * p_interleaved ** grid[:, None] + 0.50 + paired_noise
    return grid, reference, interleaved


def test_weighted_fit_recovers_known_synthetic_gate_fidelity() -> None:
    """The core weighted fit should recover a known paired-IRB fidelity."""
    grid, reference, interleaved = _synthetic_trials()
    expected = 1.0 - 0.5 * (1.0 - 0.955 / 0.985)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = analyze_paired_irb(
            _raw_result(reference, interleaved, n_cliffords=grid),
            n_bootstrap=0,
        )

    target = result["Q0"]
    assert target["gate_fidelity"] == pytest.approx(expected, abs=2e-3)
    assert target["reference"]["fit"]["absolute_sigma"] is True
    assert target["reference"]["std"].shape == grid.shape
    assert target["reference"]["sem"].shape == grid.shape


def test_two_qubit_analysis_uses_dimension_four_for_fidelity() -> None:
    """The same decay ratio should use d=4 for a two-qubit target."""
    grid, reference, interleaved = _synthetic_trials(seed=71)
    expected = 1.0 - 0.75 * (1.0 - 0.955 / 0.985)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = analyze_paired_irb(
            _raw_result(
                reference,
                interleaved,
                n_cliffords=grid,
                dimension=4,
            ),
            n_bootstrap=0,
        )

    assert result["Q0"]["gate_fidelity"] == pytest.approx(expected, abs=2e-3)
    assert result["Q0"]["metadata"]["dimension"] == 4


def test_paired_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    """A fixed bootstrap seed should reproduce the complete paired distribution."""
    grid, reference, interleaved = _synthetic_trials(seed=2)
    raw = _raw_result(reference, interleaved, n_cliffords=grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        first = analyze_paired_irb(raw, n_bootstrap=12, bootstrap_seed=44)
        second = analyze_paired_irb(raw, n_bootstrap=12, bootstrap_seed=44)

    np.testing.assert_array_equal(
        first["Q0"]["bootstrap"]["fidelities"],
        second["Q0"]["bootstrap"]["fidelities"],
    )
    np.testing.assert_array_equal(
        first["Q0"]["bootstrap"]["p_reference"],
        second["Q0"]["bootstrap"]["p_reference"],
    )


def test_stratified_resampling_preserves_ab_ba_strata_and_pairing() -> None:
    """Resampling should preserve order strata and use one index for both arms."""
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
    indices = module._stratified_paired_resample_indices(  # noqa: SLF001
        first_protocols,
        rng=np.random.default_rng(17),
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


def test_too_small_ab_ba_strata_skip_primary_bootstrap_uncertainty() -> None:
    """Two total trials cannot provide two samples in both AB/BA strata."""
    grid, reference, interleaved = _synthetic_trials(n_trials=2, seed=5)
    raw = _raw_result(reference, interleaved, n_cliffords=grid)
    with pytest.warns(RuntimeWarning, match="two samples per AB/BA stratum"):
        result = analyze_paired_irb(raw, n_bootstrap=8, bootstrap_seed=3)

    bootstrap = result["Q0"]["bootstrap"]
    assert bootstrap["execution_skipped"] is True
    assert bootstrap["n_attempted"] == 0
    assert result["Q0"]["gate_fidelity_err"] is None


def test_invalid_trial_shape_is_rejected_before_fitting() -> None:
    """Reference and interleaved trial matrices must stay exactly paired."""
    grid = np.array([0, 1, 2, 4, 8, 16], dtype=np.int64)
    reference = np.zeros((6, 4))
    interleaved = np.zeros((6, 3))
    raw = _raw_result(
        reference,
        np.zeros_like(reference),
        n_cliffords=grid,
    )
    raw["Q0"]["interleaved"]["trials"] = interleaved

    with pytest.raises(ValueError, match="same shape"):
        analyze_paired_irb(raw, n_bootstrap=0)


def test_fidelity_estimates_outside_the_physical_range_are_not_clipped() -> None:
    """A finite-sample fidelity above one should remain an explicit diagnostic."""
    grid, reference, interleaved = _synthetic_trials(
        p_reference=0.96,
        p_interleaved=0.99,
        seed=12,
    )
    raw = _raw_result(reference, interleaved, n_cliffords=grid)
    with pytest.warns(RuntimeWarning, match="not clipped"):
        result = analyze_paired_irb(raw, n_bootstrap=0)

    assert result["Q0"]["gate_fidelity"] > 1.0


def test_figure_reports_bootstrap_ci_and_reduced_chi_square_without_chi_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reduced chi-square belongs in the figure, not in a RuntimeWarning."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    monkeypatch.setattr(module.go.Figure, "show", lambda self: None)
    grid, reference, interleaved = _synthetic_trials(seed=23)
    # Add a small systematic short-length deviation to make chi-square visibly > 1.
    reference[0] += 0.008
    interleaved[0] += 0.008
    raw = _raw_result(reference, interleaved, n_cliffords=grid)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", RuntimeWarning)
        result = analyze_paired_irb(
            raw,
            n_bootstrap=8,
            bootstrap_seed=4,
            sem_floor=1e-6,
            plot=True,
        )

    assert all("chi-square" not in str(record.message) for record in records)
    target = result["Q0"]
    figure: Any = result.figure
    annotation = figure.layout.annotations[0].text
    ci95 = target["gate_fidelity_ci95"]
    assert ci95 is not None
    assert f"95% CI = [{100.0 * ci95[0]:.4f}, {100.0 * ci95[1]:.4f}] %" in annotation
    assert "reduced χ² (ref / IRB) =" in annotation
    assert annotation.index("95% CI =") < annotation.index("reduced χ²")
    chi = target["diagnostics"]["reduced_chi_square"]
    assert f"{chi['reference']:.3g}" in annotation
    assert f"{chi['interleaved']:.3g}" in annotation
