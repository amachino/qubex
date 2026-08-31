"""Synthetic count-port tests, not hardware or analog-response evidence."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from qubex.contrib.experiment.bswap_calibration import optimization as module
from qubex.contrib.experiment.bswap_calibration.optimization import (
    QualificationError,
    ShotBudget,
    calibrate_sizzle,
    estimate_phase_cycle,
    optimize_squad,
    phase_cycle_zz,
    recenter_amplitude_frequency,
    short_gate_score,
)
from qubex.contrib.experiment.bswap_calibration.pulses import (
    ideal_circuit_unitary,
    local_xy,
    local_z,
    make_squad_pulse,
)


def _recipe(kind: str) -> Any:
    pre = 0.31 if kind == "sqrt_bswap" else 0.0
    return {
        "gate_kind": kind,
        "amplitude": 0.9,
        "frequency_ghz": 4.61,
        "duration_ns": 260.0 if kind == "bswap" else 140.0,
        "ramp_ns": 16.0,
        "design_delta_scale": 1.0,
        "cd_strength": 1.0,
        "window": {"type": "hann"},
        "gate_start_ns": 24.0,
        "cancel_amplitude_ratio": 0.0,
        "cancel_phase_rad": 0.0,
        "phase_calibration": {
            "pre_active_rad": pre,
            "post_active_rad": 0.42,
            "post_passive_rad": -0.27,
        },
    }


class SyntheticCounts:
    """
    Ideal reduced gate with known controllable ZZ and shape-dependent noise.

    This port only generates synthetic counts. Compiler waveform tests live in
    test_bswap_campaign_pulses; this model is not a physical-Z hardware audit.
    """

    qubits = ("Q035", "Q034")
    rabi_scale = 0.636
    references: ClassVar[dict[str, float]] = {"Q035": 4.4, "Q034": 4.837}
    session_id = "synthetic-no-hardware"

    def __init__(
        self,
        *,
        phi: float = 0.12,
        ratio_slope: float = 4.0,
        shape_loss: bool = False,
        maximum_visibility: float = 1.0,
    ) -> None:
        """Initialize a synthetic count port without hardware access."""
        self.recipes = {kind: _recipe(kind) for kind in ("bswap", "sqrt_bswap")}
        self.phi, self.ratio_slope, self.shape_loss = phi, ratio_slope, shape_loss
        self.maximum_visibility = maximum_visibility
        self.calls = []
        self.fail_validation = False

    def acquire(
        self,
        gates: Any,
        directory: Any,
        label: str,
        *,
        prepared: Any = ("0", "0"),
        basis: str = "ZZ",
        delay_ns: float = 0.0,
        shots: int = 512,
        recipes: Any = None,
    ) -> Any:
        """Return deterministic four-state counts from the synthetic model."""
        recipes = self.recipes if recipes is None else recipes
        self.calls.append(
            {
                "directory": str(directory),
                "label": label,
                "shots": shots,
                "recipes": deepcopy(recipes),
                "prepared": prepared,
                "basis": basis,
                "gates": gates,
                "delay_ns": delay_ns,
            }
        )
        vectors = {
            "0": np.array([1, 0]),
            "1": np.array([0, 1]),
            "+": np.array([1, 1]) / np.sqrt(2),
            "-": np.array([1, -1]) / np.sqrt(2),
            "+i": np.array([1, 1j]) / np.sqrt(2),
            "-i": np.array([1, -1j]) / np.sqrt(2),
        }
        vector = np.kron(*(vectors[s] for s in prepared))
        visibility = 1.0
        for gate in gates:
            kind = "bswap" if gate == "BSWAP" else "sqrt_bswap"
            rec = recipes[kind]
            phi = self.phi - self.ratio_slope * rec.get(
                "cancel_amplitude_ratio", 0
            ) * np.cos(rec.get("cancel_phase_rad", 0))
            true_pre = np.array([0.31 if kind == "sqrt_bswap" else 0.0, 0.0])
            true_post = np.array([0.42, -0.27])
            actual = (
                local_z(true_post)
                @ ideal_circuit_unitary([gate], zz_phases={kind: 2 * phi})
                @ local_z(true_pre)
            )
            # Logical effect of the separately waveform-tested phase transport.
            cal = rec["phase_calibration"]
            pre = np.array([cal["pre_active_rad"], cal.get("pre_passive_rad", 0.0)])
            post = np.array([cal["post_active_rad"], cal["post_passive_rad"]])
            phase = pre.sum() / 2
            logical = (
                local_z(-pre - post + [phase, phase])
                @ actual
                @ local_z([-phase, -phase])
            )
            vector = logical @ vector
            loss = 0.0
            if self.shape_loss:
                loss = (
                    0.20 * (rec.get("design_delta_scale", 1) - 1.25) ** 2
                    + 0.20 * (rec.get("cd_strength", 1) - 0.85) ** 2
                )
            visibility *= self.maximum_visibility - loss
        rho = (
            visibility * np.outer(vector, vector.conj())
            + (1 - visibility) * np.eye(4) / 4
        )
        analysis = np.eye(4, dtype=complex)
        for qi, axis in enumerate(basis):
            if axis != "Z":
                analysis = local_xy(qi, -np.pi / 2 if axis == "X" else 0.0) @ analysis
        probability = np.diag(analysis @ rho @ analysis.conj().T).real
        if self.fail_validation and "independent_zz" in str(directory):
            probability = np.ones(4) / 4
        probability = np.maximum(probability, 0.0)
        probability /= probability.sum()
        counts = np.floor(probability * shots).astype(int)
        residual = shots - int(counts.sum())
        counts[np.argsort(probability * shots - counts)[-residual:]] += (
            1 if residual else 0
        )
        return {
            "counts": counts.tolist(),
            "shots": shots,
            "label": label,
            "prepared": prepared,
            "basis": basis,
            "synthetic": True,
        }


@pytest.mark.parametrize("kind", ["bswap", "sqrt_bswap"])
@pytest.mark.parametrize("phi", [-0.16, 0.12])
def test_phase_cycle_recovers_signed_integrated_zz_and_root_visibility(
    tmp_path: Path, kind: str, phi: float
) -> None:
    """Phase cycle recovers signed integrated zz and root visibility."""
    port = SyntheticCounts(phi=phi)
    result = phase_cycle_zz(
        port, kind, port.recipes[kind], tmp_path, shots=8192, bootstrap=40
    )
    estimate = result["estimate"]
    assert estimate["Phi_ZZ_mean_rad"] == pytest.approx(phi, abs=2e-4)
    assert estimate["zz_phase_rad"] == pytest.approx(2 * phi, abs=4e-4)
    assert estimate["minimum_coherence_fraction"] == pytest.approx(1.0, abs=1e-3)
    assert estimate["direction_disagreement_rad"] < 3e-4
    assert len(port.calls) == 32
    assert estimate["ci95_mean_rad"][0] < phi < estimate["ci95_mean_rad"][1]


def test_phase_cycle_rejects_missing_or_noninteger_counts(tmp_path: Path) -> None:
    """Phase cycle rejects missing or noninteger counts."""
    port = SyntheticCounts()
    result = phase_cycle_zz(port, "bswap", port.recipes["bswap"], tmp_path, bootstrap=8)
    with pytest.raises(ValueError, match="32 settings"):
        estimate_phase_cycle(result["rows"][:-1], kind="bswap", bootstrap=8)
    rows = deepcopy(result["rows"])
    rows[0]["counts"][0] += 0.1
    with pytest.raises(ValueError, match="integer counts"):
        estimate_phase_cycle(rows, kind="bswap", bootstrap=8)


@pytest.mark.parametrize("kind", ["bswap", "sqrt_bswap"])
def test_sizzle_qualifies_independent_null_preserving_duration_and_inputs(
    tmp_path: Path, kind: str
) -> None:
    """Sizzle qualifies independent null preserving duration and inputs."""
    port = SyntheticCounts(phi=0.12)
    original = deepcopy(port.recipes)
    qualified, summary = calibrate_sizzle(
        port,
        kind,
        tmp_path,
        shots=1024,
        validation_shots=8192,
        bootstrap=30,
        recenter=False,
        max_total_shots=1_000_000,
    )
    assert summary["qualified"]
    assert qualified["null_shot_interval_passed"]
    assert qualified["cancel_amplitude_ratio"] == pytest.approx(0.03, abs=0.001)
    assert abs(qualified["cancel_phase_rad"]) < 0.01
    assert qualified["duration_ns"] == original[kind]["duration_ns"]
    assert qualified["ramp_ns"] == 16.0
    assert port.recipes == original
    assert qualified["phase_reference_session_id"] == port.session_id
    calls = [r for r in port.calls if "independent_zz" in r["directory"]]
    assert len(calls) == 32
    assert all(r["shots"] == 8192 for r in calls)
    assert summary["budget"]["requested_shots"] == sum(r["shots"] for r in port.calls)
    with pytest.raises(FileExistsError):
        calibrate_sizzle(port, kind, tmp_path)


def test_sizzle_does_not_manufacture_null_without_sign_bracket(tmp_path: Path) -> None:
    """Sizzle does not manufacture null without sign bracket."""
    port = SyntheticCounts(phi=0.6, ratio_slope=1.0)
    with pytest.raises(QualificationError, match="sign bracket"):
        calibrate_sizzle(
            port,
            "bswap",
            tmp_path,
            shots=2048,
            validation_shots=8192,
            recenter=False,
            bootstrap=20,
        )
    saved = json.loads((tmp_path / "sizzle_summary.json").read_text())
    assert not saved["qualified"]
    assert saved["status"] == "failed"
    assert not (tmp_path / "qualified_recipe.json").exists()


def test_failed_fresh_sizzle_validation_is_not_reused_for_selection(
    tmp_path: Path,
) -> None:
    """Failed fresh sizzle validation is not reused for selection."""
    port = SyntheticCounts()
    port.fail_validation = True
    with pytest.raises(QualificationError, match="fresh ZZ/population"):
        calibrate_sizzle(
            port,
            "bswap",
            tmp_path,
            shots=1024,
            validation_shots=8192,
            recenter=False,
            bootstrap=20,
        )
    assert (tmp_path / "frozen_recipe.json").exists()
    assert not (tmp_path / "qualified_recipe.json").exists()
    assert len([r for r in port.calls if "independent_zz" in r["directory"]]) == 32


def test_shot_budget_stops_before_extra_hardware_request_and_saves_failure(
    tmp_path: Path,
) -> None:
    """Shot budget stops before extra hardware request and saves failure."""
    port = SyntheticCounts()
    with pytest.raises(QualificationError, match="budget exhausted"):
        calibrate_sizzle(
            port,
            "bswap",
            tmp_path,
            shots=512,
            max_total_shots=513,
            recenter=False,
            bootstrap=8,
        )
    assert len(port.calls) == 1
    summary = json.loads((tmp_path / "sizzle_summary.json").read_text())
    assert summary["budget"]["requested_shots"] == 512
    assert not summary["qualified"]


def test_short_score_is_coherence_sensitive_with_unchanged_basis_populations(
    tmp_path: Path,
) -> None:
    """Short score is coherence sensitive with unchanged basis populations."""
    port = SyntheticCounts(phi=0.0)
    good = short_gate_score(
        port, "sqrt_bswap", port.recipes["sqrt_bswap"], tmp_path / "good", shots=4096
    )
    bad_port = SyntheticCounts(phi=0.0, maximum_visibility=0.5)
    bad = short_gate_score(
        bad_port,
        "sqrt_bswap",
        bad_port.recipes["sqrt_bswap"],
        tmp_path / "bad",
        shots=4096,
    )
    assert good["score"] > 0.999
    assert bad["score"] < 0.6
    assert good["score"] - bad["score"] > 0.4
    assert len(good["rows"]) == 4


def test_squad_optimizer_fixed_family_fresh_validation_and_physical_k(
    tmp_path: Path,
) -> None:
    """Squad optimizer fixed family fresh validation and physical k."""
    port = SyntheticCounts(phi=0.0, shape_loss=True)
    original = deepcopy(port.recipes)
    result, summary = optimize_squad(
        port,
        "sqrt_bswap",
        tmp_path,
        max_evaluations=5,
        shots=1024,
        validation_shots=4096,
        max_total_shots=1_000_000,
        recenter=False,
        bootstrap=20,
    )
    assert summary["qualified"]
    assert summary["evaluations"] <= 5
    assert result["duration_ns"] == original["sqrt_bswap"]["duration_ns"]
    assert result["ramp_ns"] == original["sqrt_bswap"]["ramp_ns"]
    assert port.recipes == original
    assert 0.6 <= result["design_delta_scale"] <= 1.6
    assert 0.3 <= result["cd_strength"] <= 1.5
    assert summary["rabi_conversion_fixed"] == port.rabi_scale
    expected = make_squad_pulse(
        result,
        rabi_ghz_per_amplitude=port.rabi_scale,
        transition_frequency_ghz=port.references["Q035"],
    )
    assert expected.scale == pytest.approx(1 / (2 * np.pi * port.rabi_scale))
    trials = json.loads((tmp_path / "optimization_trials.json").read_text())["trials"]
    for trial in trials:
        if trial["status"] == "scored":
            assert trial["recipe"]["window"] == {"type": "hann"}
            assert (
                trial["recipe"]["duration_ns"] == original["sqrt_bswap"]["duration_ns"]
            )
    assert all("independent" not in row["directory"] for row in port.calls[:36])
    assert summary["budget"]["requested_shots"] == sum(r["shots"] for r in port.calls)
    assert (tmp_path / "frozen_recipe.json").exists()


def test_squad_bad_budget_and_lost_coherence_fail_closed(tmp_path: Path) -> None:
    """Squad bad budget and lost coherence fail closed."""
    port = SyntheticCounts(phi=0.0, maximum_visibility=0.6)
    with pytest.raises(QualificationError):
        optimize_squad(
            port,
            "bswap",
            tmp_path,
            max_evaluations=3,
            shots=1024,
            validation_shots=2048,
            recenter=False,
            max_total_shots=1_000_000,
        )
    summary = json.loads((tmp_path / "optimization_summary.json").read_text())
    assert not summary["qualified"]
    assert not (tmp_path / "qualified_recipe.json").exists()


def test_budget_validation() -> None:
    """Budget validation."""
    with pytest.raises(ValueError, match="positive integer"):
        ShotBudget(0)
    budget = ShotBudget(10)
    budget.reserve(7)
    with pytest.raises(QualificationError):
        budget.reserve(4)
    assert budget.requested == 7


class RidgeCounts(SyntheticCounts):
    """Synthetic narrow response ridge, without a hardware interpretation."""

    center_frequency = 4.61
    confirmation_visibility: float | None = None

    def acquire(self, gates: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply bounded depolarization about a known sloping carrier ridge."""
        kind = "bswap" if gates[0] == "BSWAP" else "sqrt_bswap"
        recipe = kwargs["recipes"][kind]
        center = self.center_frequency - 0.2 * (recipe["amplitude"] - 0.9)
        offset = (recipe["frequency_ghz"] - center) / 0.00045
        self.maximum_visibility = 0.98 * np.exp(-0.5 * offset**2)
        if self.confirmation_visibility is not None and "candidate_" in args[1]:
            self.maximum_visibility = self.confirmation_visibility
        return super().acquire(gates, *args, **kwargs)


@pytest.mark.parametrize("kind", ["bswap", "sqrt_bswap"])
def test_local_recenter_executes_fixed_duration_coherence_aware_map(
    tmp_path: Path, kind: str
) -> None:
    """Local recenter executes fixed duration coherence aware map."""
    port = RidgeCounts(phi=0.0)
    base = deepcopy(port.recipes[kind])
    result, summary = recenter_amplitude_frequency(
        port, kind, base, tmp_path, shots=1024
    )
    assert result["duration_ns"] == base["duration_ns"]
    assert result["ramp_ns"] == base["ramp_ns"]
    assert result["amplitude"] == pytest.approx(0.9, abs=0.001)
    assert abs(result["frequency_ghz"] - 4.61) < 0.0001
    assert summary["fit"]["reduced_chi2"] < 5
    assert summary["fit"]["qualified_ridge"]
    assert summary["fit"]["gp_allowed"]
    assert summary["confirmation"]["passed"]
    assert summary["population"]["minimum_population_agreement"] > 0.97
    assert {r["recipes"][kind]["duration_ns"] for r in port.calls} == {
        base["duration_ns"]
    }
    assert len(port.calls) == (50 if kind == "bswap" else 234)
    assert summary["plan"]["initial_points"] == 21
    assert summary["plan"]["amplitude_step"] == 0.0005
    assert summary["budget"]["requested_shots"] <= summary["plan"]["maximum_shots"]
    assert base == port.recipes[kind]
    for call in port.calls:
        observed = call["recipes"][kind]
        for key in (
            "duration_ns",
            "ramp_ns",
            "design_delta_scale",
            "cd_strength",
            "window",
            "cancel_amplitude_ratio",
            "cancel_phase_rad",
            "gate_start_ns",
        ):
            assert observed[key] == base[key]
    assert summary["plan"]["fixed_rabi_scale"] == port.rabi_scale


def test_recenter_root_plateau_retains_seed_without_gp(tmp_path: Path) -> None:
    """A root plateau retains its independent seed without qualifying a ridge."""
    port = SyntheticCounts(phi=0.0)
    base = deepcopy(port.recipes["sqrt_bswap"])
    result, summary = recenter_amplitude_frequency(
        port, "sqrt_bswap", base, tmp_path, shots=1024
    )
    assert result["frequency_ghz"] == base["frequency_ghz"]
    assert result["amplitude"] == base["amplitude"]
    assert summary["fit"]["plateau"]["accepted"]
    assert not summary["fit"]["qualified_ridge"]
    assert not summary["fit"]["gp_allowed"]
    assert summary["confirmation"]["passed"]
    assert len(port.calls) == 234


def test_recenter_rejects_full_flat_response_with_saved_evidence(
    tmp_path: Path,
) -> None:
    """A full-gate flat response cannot bypass the ridge qualification."""
    port = SyntheticCounts(phi=0.0)
    with pytest.raises(QualificationError, match="response ridge"):
        recenter_amplitude_frequency(
            port,
            "bswap",
            port.recipes["bswap"],
            tmp_path,
            shots=1024,
            max_extension_rounds=0,
        )
    evidence = json.loads((tmp_path / "local_fit.json").read_text())
    assert not evidence["qualified_ridge"]
    assert not evidence["plateau"]["accepted"]
    assert len(port.calls) == 42
    assert not (tmp_path / "recentered_recipe.json").exists()


def test_recenter_shot_budget_remains_hard(tmp_path: Path) -> None:
    """Recenter cannot reinterpret a shot-budget failure as a scientific fallback."""
    port = RidgeCounts(phi=0.0)
    with pytest.raises(QualificationError, match="budget exhausted"):
        recenter_amplitude_frequency(
            port,
            "bswap",
            port.recipes["bswap"],
            tmp_path,
            shots=256,
            budget=ShotBudget(256),
        )
    assert len(port.calls) == 1
    assert not (tmp_path / "recentered_recipe.json").exists()


def test_recenter_extends_boundary_rows_within_fixed_bounds(tmp_path: Path) -> None:
    """Missing peak coverage gets bounded added frequencies, not a fake peak."""
    port = RidgeCounts(phi=0.0)
    port.center_frequency = 4.6108
    result, summary = recenter_amplitude_frequency(
        port, "bswap", port.recipes["bswap"], tmp_path, shots=2048
    )
    assert summary["fit"]["qualified_ridge"]
    assert summary["fit"]["extensions"]
    assert 21 < summary["fit"]["observed_points"] <= 84
    observed = json.loads((tmp_path / "local_observations.json").read_text())
    points = {(row["amplitude"], row["frequency_ghz"]) for row in observed}
    assert len(points) == len(observed)
    assert all(4.608 <= frequency <= 4.612 for _, frequency in points)
    assert (result["amplitude"], result["frequency_ghz"]) in points
    assert {call["recipes"]["bswap"]["ramp_ns"] for call in port.calls} == {16}


def test_recenter_point_cap_rejects_without_promotion(tmp_path: Path) -> None:
    """Exhausted frequency coverage is saved and cannot exceed the point cap."""
    port = RidgeCounts(phi=0.0)
    port.center_frequency = 4.6115
    with pytest.raises(QualificationError, match="response ridge"):
        recenter_amplitude_frequency(
            port,
            "bswap",
            port.recipes["bswap"],
            tmp_path,
            shots=1024,
            max_scout_points=24,
        )
    evidence = json.loads((tmp_path / "local_fit.json").read_text())
    assert evidence["observed_points"] == 24
    assert not evidence["qualified_ridge"]
    assert any(row["limited_by_point_cap"] for row in evidence["extensions"])
    assert len(port.calls) == 48
    assert not (tmp_path / "recentered_recipe.json").exists()


def test_recenter_rejects_changed_conversion_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed measured K terminates before another fixed-family request."""
    port = RidgeCounts(phi=0.0)
    acquire = port.acquire

    def change_scale(*args: Any, **kwargs: Any) -> Any:
        result = acquire(*args, **kwargs)
        port.rabi_scale = 0.7
        return result

    monkeypatch.setattr(port, "acquire", change_scale)
    with pytest.raises(ValueError, match="Rabi scale changed"):
        recenter_amplitude_frequency(
            port, "bswap", port.recipes["bswap"], tmp_path, shots=1024
        )
    assert len(port.calls) == 1
    assert not (tmp_path / "recentered_recipe.json").exists()


@pytest.mark.parametrize("visibility", [0.2, 0.85])
def test_recenter_rejects_fresh_score_failure(
    tmp_path: Path, visibility: float
) -> None:
    """Low absolute score or a resolved fresh degradation rejects the candidate."""
    port = RidgeCounts(phi=0.0)
    port.confirmation_visibility = visibility
    with pytest.raises(QualificationError, match="score confirmation failed"):
        recenter_amplitude_frequency(
            port, "bswap", port.recipes["bswap"], tmp_path, shots=2048
        )
    confirmation = json.loads((tmp_path / "confirmation.json").read_text())
    assert not confirmation["passed"]
    assert len(port.calls) == 46
    assert not (tmp_path / "recentered_recipe.json").exists()


@pytest.mark.parametrize(
    "controls",
    [
        {"frequency_points": 6},
        {"frequency_points": 8},
        {"max_scout_points": 20},
        {"amplitude_step": float("nan")},
        {"max_extension_rounds": -1},
        {"confirmation_shots": 0},
        {"minimum_score_lower_bound": float("nan")},
        {"frequency_search_half_width_mhz": 0.1},
    ],
)
def test_recenter_invalid_controls_never_acquire(tmp_path: Path, controls: Any) -> None:
    """Malformed production controls fail before any count request."""
    port = RidgeCounts(phi=0.0)
    with pytest.raises(ValueError, match=r"integer|bounds|finite"):
        recenter_amplitude_frequency(
            port, "bswap", port.recipes["bswap"], tmp_path, **controls
        )
    assert not port.calls


def test_selected_on_shape_requalifies_exact_waveform_sizzle_null(
    tmp_path: Path,
) -> None:
    """Selected on shape requalifies exact waveform sizzle null."""
    port = SyntheticCounts(phi=0.12, shape_loss=True)
    port.recipes["bswap"]["cancel_amplitude_ratio"] = 0.03
    port.recipes["bswap"]["null_shot_interval_passed"] = True
    result, summary = optimize_squad(
        port,
        "bswap",
        tmp_path,
        max_evaluations=3,
        shots=1024,
        validation_shots=2048,
        null_validation_shots=8192,
        max_total_shots=2_000_000,
        recenter=False,
        bootstrap=20,
    )
    assert summary["qualified"]
    assert summary["sizzle_requalification"]["qualified"]
    assert result["null_shot_interval_passed"]
    assert result["sizzle_calibration_directory"].endswith("selected_shape_sizzle")
    assert result["phase_reference_session_id"] == port.session_id
    null_inputs = [
        r["recipes"]["bswap"]
        for r in port.calls
        if "selected_shape_sizzle/independent_zz" in r["directory"]
    ]
    assert len(null_inputs) == 32
    assert all(
        r["design_delta_scale"] == result["design_delta_scale"]
        and r["cd_strength"] == result["cd_strength"]
        for r in null_inputs
    )


@pytest.mark.parametrize("workflow", ["sizzle", "squad"])
@pytest.mark.parametrize("kind", ["bswap", "sqrt_bswap"])
def test_smoke_exercises_two_real_candidates_without_qualification(
    tmp_path: Path, workflow: str, kind: str
) -> None:
    """Low-quality smoke data exercises changed pulses without a null or fidelity claim."""
    port = SyntheticCounts(phi=0.12, ratio_slope=0.0, maximum_visibility=0.4)
    port.recipes[kind]["zz_phase_rad"] = 0.24
    port.recipes[kind].update(
        qualified=True,
        scientific_qualified=True,
        null_shot_interval_passed=True,
        shape_validation_passed=True,
    )
    original = deepcopy(port.recipes)
    function = calibrate_sizzle if workflow == "sizzle" else optimize_squad
    result, summary = function(
        port,
        kind,
        tmp_path,
        shots=64,
        validation_shots=128,
        max_total_shots=20000,
        bootstrap=8,
        smoke_mode=True,
    )
    assert summary["status"] == "smoke_only"
    assert summary["smoke_only"]
    assert not summary["qualified"]
    assert not summary["scientific_qualified"]
    assert summary["evaluations"] == 2
    assert summary["coverage"]["changed_waveform_acquired"]
    assert summary["coverage"]["changed_waveform_returned"]
    assert summary["coverage"]["skipped_full_optimization"]
    assert result["smoke_only"]
    assert not result["null_shot_interval_passed"]
    assert not result["shape_validation_passed"]
    assert "phase_calibration" in result
    assert result["duration_ns"] == original[kind]["duration_ns"]
    assert port.recipes == original
    assert len(port.calls) == 144
    assert len([r for r in port.calls if "/zz/" in r["directory"] + "/"]) == 64
    assert (tmp_path / "provisional_recipe.json").exists()
    assert not (tmp_path / "qualified_recipe.json").exists()
    for path in tmp_path.glob("smoke_candidate_*/phase/phase_calibration.json"):
        phase_record = json.loads(path.read_text())
        assert all(
            not phase_record[key]
            for key in (
                "qualified",
                "scientific_qualified",
                "null_shot_interval_passed",
                "shape_validation_passed",
            )
        )
    assert summary["budget"]["requested_shots"] == sum(r["shots"] for r in port.calls)


def test_smoke_unidentified_changed_phase_retains_only_unchanged_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unidentified changed pulse is recorded but never receives invented phases."""
    port = SyntheticCounts(phi=0.12)
    port.recipes["bswap"]["zz_phase_rad"] = 0.24
    original = deepcopy(port.recipes["bswap"])
    actual_fit = module.fit_local_phases
    calls = 0

    def fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic unidentifiable changed phase")
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(module, "fit_local_phases", fit)
    result, summary = calibrate_sizzle(
        port,
        "bswap",
        tmp_path,
        shots=64,
        validation_shots=64,
        bootstrap=8,
        smoke_mode=True,
    )
    assert result["phase_calibration"] == original["phase_calibration"]
    assert result["zz_phase_rad"] == original["zz_phase_rad"]
    assert result["cancel_amplitude_ratio"] == original["cancel_amplitude_ratio"]
    assert summary["selected_trial"] == 0
    assert summary["coverage"]["changed_waveform_acquired"]
    assert not summary["coverage"]["changed_waveform_returned"]
    assert summary["coverage"]["unexercised_coverage"]
    assert not summary["candidates"][1]["phase_identified"]
    assert "phase_calibration" not in summary["candidates"][1]["recipe"]
    assert "SMOKE_UNCHANGED_SEED" in capsys.readouterr().out


@pytest.mark.parametrize("workflow", ["sizzle", "squad"])
def test_smoke_budget_exhaustion_remains_hard(tmp_path: Path, workflow: str) -> None:
    """Smoke never converts an exhausted shot budget into a successful continuation."""
    port = SyntheticCounts()
    function = calibrate_sizzle if workflow == "sizzle" else optimize_squad
    with pytest.raises(QualificationError, match="budget exhausted"):
        function(
            port,
            "bswap",
            tmp_path,
            shots=64,
            max_total_shots=65,
            bootstrap=8,
            smoke_mode=True,
        )
    assert len(port.calls) == 1
    filename = (
        "sizzle_summary.json" if workflow == "sizzle" else "optimization_summary.json"
    )
    summary = json.loads((tmp_path / filename).read_text())
    assert summary["status"] == "failed"
    assert not summary["qualified"]


def test_smoke_invalid_waveform_fails_before_measurement(tmp_path: Path) -> None:
    """Smoke does not clip or skip an invalid physical waveform."""
    port = SyntheticCounts()
    port.recipes["bswap"]["amplitude"] = 1.2
    with pytest.raises(ValueError, match="amplitude"):
        optimize_squad(port, "bswap", tmp_path, smoke_mode=True)
    assert port.calls == []


def test_smoke_budget_during_phase_acquisition_is_not_softened(tmp_path: Path) -> None:
    """The scientific phase fallback cannot catch a budget failure inside acquisition."""
    port = SyntheticCounts()
    with pytest.raises(QualificationError, match="budget exhausted"):
        calibrate_sizzle(
            port,
            "bswap",
            tmp_path,
            shots=64,
            max_total_shots=33 * 64 + 1,
            bootstrap=8,
            smoke_mode=True,
        )
    assert len(port.calls) == 33
    assert not any("population" in row["directory"] for row in port.calls)


@pytest.mark.parametrize("failure", ["counts", "deadline"])
def test_smoke_data_and_deadline_errors_remain_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Malformed counts and execution deadlines never become smoke-quality warnings."""
    port = SyntheticCounts()
    original = port.acquire

    def acquire(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if failure == "deadline":
            raise TimeoutError("synthetic deadline")
        row = original(*args, **kwargs)
        row["counts"][0] += 0.5
        return row

    monkeypatch.setattr(port, "acquire", acquire)
    with pytest.raises((ValueError, TimeoutError)):
        calibrate_sizzle(
            port, "bswap", tmp_path, shots=64, bootstrap=8, smoke_mode=True
        )
    saved = json.loads((tmp_path / "sizzle_summary.json").read_text())
    assert saved["status"] == "failed"
    assert not (tmp_path / "provisional_recipe.json").exists()


def test_smoke_records_bad_model_quality_without_qualifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finite but poor phase model remains an explicitly unqualified smoke result."""
    port = SyntheticCounts()
    original = module.fit_local_phases

    def fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        result["coherence_residual_rms"] = 0.25
        return result

    monkeypatch.setattr(module, "fit_local_phases", fit)
    result, summary = optimize_squad(
        port,
        "bswap",
        tmp_path,
        shots=64,
        validation_shots=64,
        bootstrap=8,
        smoke_mode=True,
    )
    assert summary["coverage"]["changed_waveform_returned"]
    assert all(not row["phase_model_qualified"] for row in summary["candidates"])
    assert result["phase_calibration"]["coherence_residual_rms"] == 0.25
    assert not result["qualified"]
