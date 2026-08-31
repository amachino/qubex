"""Pure tests for model-free selection from five local damped-row fits."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from qubex.contrib.experiment.bswap_calibration.chevron_fit import (
    select_local_operational_candidate,
)


def row(
    score: float,
    *,
    rate: float = 0.9,
    chi2: float = 1.5,
    vis00: float | None = None,
    vis11: float | None = None,
    duration: float = 640.0,
    warnings: list[str] | None = None,
    available: bool = True,
) -> dict[str, Any]:
    """Return one internally consistent synthetic damped-transfer row fit."""
    offset = 0.05
    visibility = (score - offset) / (1 - offset)
    return dict(
        parameters=dict(
            rate_mhz=rate,
            ramp_phase_rad=np.pi - 2 * np.pi * rate * (duration - 60) / 1000,
            offset_00=offset,
            offset_11=offset,
            visibility_00=visibility if vis00 is None else vis00,
            visibility_11=visibility if vis11 is None else vis11,
            decay_rate_00_per_us=0.0,
            decay_rate_11_per_us=0.0,
        ),
        local_standard_errors=dict(rate_mhz=0.005),
        bswap=dict(
            available=available,
            grid_duration_ns=duration,
            neighbors_ns=[duration - 2, duration, duration + 2],
        ),
        sqrt_bswap=dict(available=True),
        reduced_chi2=chi2,
        warnings=[] if warnings is None else warnings,
        ramp_ns=30.0,
    )


def test_selects_by_worst_direction_then_mean_without_frequency_curve_fit() -> None:
    """Operational ranking uses predicted local-row transfer, never a global curve."""
    frequencies = 4.7719 + np.arange(-2, 3) * 0.00006
    fits = [row(score) for score in (0.68, 0.75, 0.82, 0.79, 0.70)]
    result = select_local_operational_candidate(
        frequencies, fits, seed_frequency_ghz=4.7719
    )
    assert result["candidate_generation_passed"]
    assert not result["adoption_qualified"]
    assert result["independent_holdout_required"]
    assert result["operational_response_candidate"]["row_index"] == 2
    assert result["operational_response_candidate"]["frequency_ghz"] == pytest.approx(
        frequencies[2]
    )
    assert result["operational_response_candidate"]["duration_ns"] == 640.0
    assert result["operational_response_candidate"]["primary_score"] == pytest.approx(
        0.82
    )
    assert "frequency_profile" not in result
    assert "resonance_frequency_ghz" not in result
    assert "rate_squared" not in result


def test_secondary_mean_breaks_equal_minimum_score() -> None:
    """Equal worst-direction predictions use the larger two-direction mean."""
    frequencies = 4.7719 + np.arange(-2, 3) * 0.00006
    fits = [row(0.7) for _ in frequencies]
    fits[1] = row(0.75, vis00=(0.75 - 0.05) / 0.95, vis11=(0.82 - 0.05) / 0.95)
    fits[3] = row(0.75, vis00=(0.75 - 0.05) / 0.95, vis11=(0.78 - 0.05) / 0.95)
    result = select_local_operational_candidate(
        frequencies, fits, seed_frequency_ghz=frequencies[2]
    )
    assert result["operational_response_candidate"]["row_index"] == 1


def test_ties_use_seed_distance_then_lower_frequency() -> None:
    """Exact score ties choose nearest seed and then lower measured frequency."""
    frequencies = 4.7719 + np.arange(-2, 3) * 0.00006
    fits = [row(0.7), row(0.8), row(0.7), row(0.8), row(0.7)]
    result = select_local_operational_candidate(
        frequencies, fits, seed_frequency_ghz=frequencies[2]
    )
    assert result["operational_response_candidate"]["row_index"] == 1


def test_residual_warning_is_retained_but_chi3_gate_is_explicit() -> None:
    """Shot-residual warnings remain visible while the predeclared chi3 gate applies."""
    frequencies = 4.7719 + np.arange(-2, 3) * 0.00006
    warning = "Residuals exceed shot noise; inspect the effective-model fit"
    fits = [row(0.75) for _ in frequencies]
    fits[2] = row(0.81, chi2=2.8, warnings=[warning])
    result = select_local_operational_candidate(
        frequencies, fits, seed_frequency_ghz=frequencies[2]
    )
    assert result["candidate_generation_passed"]
    assert not result["row_models_all_warning_free"]
    assert result["rows"][2]["warnings"] == [warning]
    assert warning in result["warnings"]


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"available": False}, "full_candidate_unavailable"),
        ({"chi2": 3.01}, "reduced_chi2_exceeds_limit"),
        ({"vis00": 0.59}, "visibility_00_below_minimum"),
        ({"vis11": 0.59}, "visibility_11_below_minimum"),
        ({"rate": 0.3}, "rate_not_interior"),
        (
            {
                "warnings": [
                    "Low fitted contrast; phase-angle candidates are unreliable"
                ]
            },
            "unsupported_fit_warning",
        ),
    ],
)
def test_every_row_must_pass_before_any_candidate_is_returned(
    update: dict[str, Any], reason: str
) -> None:
    """A failing row prevents selecting around inconvenient training evidence."""
    frequencies = 4.7719 + np.arange(-2, 3) * 0.00006
    kwargs = deepcopy(update)
    if "available" in kwargs:
        kwargs["available"] = kwargs.pop("available")
    fits = [row(0.75) for _ in frequencies]
    fits[4] = row(0.8, **kwargs)
    result = select_local_operational_candidate(
        frequencies, fits, seed_frequency_ghz=frequencies[2]
    )
    assert not result["candidate_generation_passed"]
    assert result["operational_response_candidate"] is None
    assert f"row_4:{reason}" in result["reasons"]


def test_real_unavailable_payload_without_duration_is_a_scientific_failure() -> None:
    """The duration helper's unavailable payload is retained without a schema exception."""
    frequencies = 4.7719 + np.arange(-2, 3) * 0.00006
    fits = [row(0.75) for _ in frequencies]
    fits[3]["bswap"] = {"available": False, "reason": "outside measured range"}
    result = select_local_operational_candidate(
        frequencies, fits, seed_frequency_ghz=frequencies[2]
    )
    assert not result["candidate_generation_passed"]
    assert result["operational_response_candidate"] is None
    assert result["rows"][3]["full_duration_ns"] is None
    assert "row_3:full_candidate_unavailable" in result["reasons"]


@pytest.mark.parametrize(
    "frequencies",
    [
        [4.7] * 5,
        [4.7, 4.7001, 4.7002, 4.7004, 4.7003],
        [4.7, 4.7001, 4.7002, 4.7003],
        [4.7, 4.7001, np.nan, 4.7003, 4.7004],
    ],
)
def test_frequency_grid_must_be_five_increasing_finite_ghz(frequencies: Any) -> None:
    """Malformed or nonlocal frequency grids fail instead of changing ranking semantics."""
    with pytest.raises(ValueError, match=r"five.*frequency"):
        select_local_operational_candidate(
            frequencies, [row(0.8)] * len(frequencies), seed_frequency_ghz=4.7
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grid_duration_ns", 641.0),
        ("grid_duration_ns", np.nan),
        ("rate_mhz", np.nan),
        ("rate_mhz_error", 0.0),
    ],
)
def test_malformed_row_numbers_raise(field: str, value: float) -> None:
    """Nonfinite or off-grid local-fit payloads are schema errors, not failed science."""
    frequencies = 4.7719 + np.arange(-2, 3) * 0.00006
    fits = [row(0.8) for _ in frequencies]
    if field == "grid_duration_ns":
        fits[0]["bswap"][field] = value
    elif field == "rate_mhz":
        fits[0]["parameters"][field] = value
    else:
        fits[0]["local_standard_errors"]["rate_mhz"] = value
    with pytest.raises(ValueError, match=r"row 0|grid_duration"):
        select_local_operational_candidate(
            frequencies, fits, seed_frequency_ghz=frequencies[2]
        )


@pytest.mark.parametrize("mismatch", ["ramp", "neighbors"])
def test_rows_share_one_ramp_and_retain_the_native_candidate(mismatch: str) -> None:
    """Cross-row timing mismatch and a missing native candidate are schema errors."""
    frequencies = 4.7719 + np.arange(-2, 3) * 0.00006
    fits = [row(0.8) for _ in frequencies]
    if mismatch == "ramp":
        fits[3]["ramp_ns"] = 32.0
    else:
        fits[3]["bswap"]["neighbors_ns"] = [638.0, 642.0]
    with pytest.raises(ValueError, match=r"same ramp|off-grid"):
        select_local_operational_candidate(
            frequencies, fits, seed_frequency_ghz=frequencies[2]
        )


def test_inputs_and_row_fits_are_not_mutated() -> None:
    """Candidate generation is pure and retains all raw row-fit diagnostics."""
    frequencies = (4.7719 + np.arange(-2, 3) * 0.00006).tolist()
    fits = [row(0.7 + 0.02 * i) for i in range(5)]
    before = deepcopy(fits)
    select_local_operational_candidate(
        frequencies, fits, seed_frequency_ghz=frequencies[2]
    )
    assert fits == before
