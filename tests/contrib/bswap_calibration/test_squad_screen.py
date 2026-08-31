"""Controller tests for a bounded SQUAD screen; synthetic counts only."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qxpulse import Arbitrary

from qubex.contrib.experiment.bswap_calibration import squad_screen as module
from qubex.contrib.experiment.bswap_calibration.optimization import QualificationError
from qubex.contrib.experiment.bswap_calibration.pulses import (
    ideal_circuit_unitary,
    local_xy,
    local_z,
)


def _estimate(good: bool = True) -> dict[str, Any]:
    interval = [-0.005, 0.005] if good else [0.04, 0.06]
    return dict(
        Phi_ZZ_mean_rad=0.0 if good else 0.05,
        zz_phase_rad=0.0 if good else 0.1,
        ci95_active_rad=interval,
        ci95_passive_rad=interval,
        ci95_mean_rad=interval,
        direction_disagreement_rad=0.001,
        minimum_coherence_fraction=0.9,
    )


class CountsPort:
    """Record attempted synthetic acquisitions without connecting to any device."""

    def __init__(self) -> None:
        self.qubits = ("A", "P")
        self.rabi_scale = 0.636
        self.references = {"A": 4.4, "P": 4.837}
        self.targets = {"D": 4.612, "C": 4.612}
        self.drive_label, self.cancel_label = "D", "C"
        self.session_id = "synthetic-epoch"
        self.deadline = None
        self.x90 = {
            q: Arbitrary(np.full(12, 0.08), sampling_period=2) for q in self.qubits
        }
        self.xpi = {
            q: Arbitrary(np.full(12, 0.16), sampling_period=2) for q in self.qubits
        }
        self.classifiers = {
            q: SimpleNamespace(
                phase=0.0,
                scale=1.0,
                label_map={0: 0, 1: 1},
                model=SimpleNamespace(
                    weights_=np.array([0.5, 0.5]), precisions_cholesky_=np.ones(2)
                ),
            )
            for q in self.qubits
        }
        self.recipes = {}
        for kind in ("bswap", "sqrt_bswap"):
            self.recipes[kind] = dict(
                gate_kind=kind,
                amplitude=0.9,
                frequency_ghz=4.6118,
                duration_ns=260.0 if kind == "bswap" else 140.0,
                ramp_ns=16.0,
                window={"type": "hann"},
                design_delta_scale=1.0,
                cd_strength=1.0,
                gate_start_ns=24.0,
                cancel_amplitude_ratio=0.03,
                cancel_phase_rad=0.0,
                phase_reference_session_id=self.session_id,
                phase_reference_id="synthetic-ref",
                null_shot_interval_passed=True,
                null_validation=_estimate(),
                tolerance_phi_zz_rad=0.02,
                phase_calibration=dict(
                    pre_active_rad=0.3 if kind == "sqrt_bswap" else 0.0,
                    post_active_rad=0.4,
                    post_passive_rad=-0.2,
                    zz_phase_rad=0.0,
                ),
            )
        self.calls: list[dict[str, Any]] = []
        self.change_classifier_precision = False
        self.timeout = self.malformed = self.change_scale = self.change_classifier = (
            False
        )

    def acquire(
        self,
        gates: Any,
        directory: Any,
        label: str,
        *,
        shots: int,
        recipes: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a synthetic raw count row or the requested injected failure."""
        self.calls.append(
            dict(
                directory=str(directory),
                shots=shots,
                recipes=deepcopy(recipes),
                label=label,
            )
        )
        if self.timeout:
            raise TimeoutError("synthetic deadline")
        if self.change_scale:
            self.rabi_scale *= 1.1
        if self.change_classifier:
            self.classifiers["A"].phase += 0.1
        if self.change_classifier_precision:
            self.classifiers["A"].model.precisions_cholesky_[0] *= 1.1
        return dict(
            counts=[shots, 0] if self.malformed else [shots, 0, 0, 0], shots=shots
        )


class Stages:
    """Replace already separately tested numerical protocols with counted stages."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.scores = {1.0: 0.78, 1.3: 0.80, 1.6: 0.84, 2.0: 0.82}
        self.final_scores: dict[float, float] = {}
        self.selection_se = 0.002
        self.final_se = 0.002
        self.null_calls: list[dict[str, Any]] = []
        self.recenter_after_null_changes = False
        self.fail_new_null = False
        self.baseline_null_good = True
        monkeypatch.setattr(module, "recenter_amplitude_frequency", self.recenter)
        monkeypatch.setattr(module, "_refresh_phases", self.phases)
        monkeypatch.setattr(module, "short_gate_score", self.score)
        monkeypatch.setattr(module, "calibrate_sizzle", self.null)
        monkeypatch.setattr(module, "phase_cycle_zz", self.cycle)

    @staticmethod
    def request(
        port: Any,
        kind: str,
        recipe: Any,
        directory: Any,
        budget: Any,
        shots: int,
        settings: int,
    ) -> None:
        """Count each requested setting through the guarded public acquisition port."""
        for index in range(settings):
            budget.reserve(shots)
            selected = deepcopy(port.recipes)
            selected[kind] = deepcopy(recipe)
            port.acquire(
                ["BSWAP" if kind == "bswap" else "RAW_SQRT_BSWAP"],
                directory,
                str(index),
                shots=shots,
                recipes=selected,
            )

    def recenter(
        self,
        port: Any,
        kind: str,
        recipe: Any,
        directory: Any,
        *,
        shots: int,
        budget: Any,
    ) -> tuple[dict, dict]:
        """Charge the actual 21-point root recenter and fourfold confirmation rates."""
        bases = 5 if kind == "sqrt_bswap" else 1
        self.request(port, kind, recipe, directory, budget, shots, 42 * bases)
        self.request(port, kind, recipe, directory, budget, 4 * shots, 4 * bases + 4)
        result = deepcopy(recipe)
        if self.recenter_after_null_changes and "post_null_recenter" in str(directory):
            result["frequency_ghz"] += 0.0001
        return result, dict(population=dict(minimum_population_agreement=0.9))

    def phases(
        self,
        port: Any,
        kind: str,
        recipe: Any,
        directory: Any,
        *,
        shots: int,
        budget: Any,
    ) -> dict:
        """Keep supplied measured phases while charging all 36 tomography settings."""
        self.request(port, kind, recipe, directory, budget, shots, 36)
        return deepcopy(recipe)

    def score(
        self,
        port: Any,
        kind: str,
        recipe: Any,
        directory: Any,
        *,
        shots: int,
        budget: Any,
        validation: bool = False,
    ) -> dict:
        """Return predeclared synthetic scores and account for all 40 settings."""
        self.request(port, kind, recipe, directory, budget, shots, 40)
        scale = float(recipe.get("design_delta_scale", 1.0))
        score = (
            self.final_scores.get(scale, self.scores[scale])
            if validation
            else self.scores[scale]
        )
        se = self.final_se if validation else self.selection_se
        return dict(
            score=score,
            shot_standard_error=se,
            ranking_score=score - 1.96 * se,
            minimum_state_overlap=score - 0.01,
            population=dict(minimum_population_agreement=0.9),
            validation=validation,
        )

    def null(
        self,
        port: Any,
        kind: str,
        directory: Any,
        *,
        recipe: Any,
        shots: int,
        validation_shots: int,
        recenter: bool,
        budget: Any,
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        """Charge a two-point fixed-control null search with real validation shot counts."""
        self.null_calls.append(
            dict(
                recipe=deepcopy(recipe),
                shots=shots,
                validation_shots=validation_shots,
                recenter=recenter,
            )
        )
        self.request(port, kind, recipe, directory, budget, shots, 160 + 36 * 2 + 68)
        if self.fail_new_null:
            raise QualificationError("synthetic unresolved new-shape null")
        self.request(port, kind, recipe, directory, budget, validation_shots, 36)
        qualified = deepcopy(recipe)
        qualified["null_shot_interval_passed"] = True
        return qualified, dict(qualified=True)

    def cycle(
        self,
        port: Any,
        kind: str,
        recipe: Any,
        directory: Any,
        *,
        shots: int,
        budget: Any,
        **kwargs: Any,
    ) -> dict:
        """Charge a complete final ZZ phase cycle, including the baseline fallback."""
        self.request(port, kind, recipe, directory, budget, shots, 32)
        good = (
            self.baseline_null_good
            if recipe.get("design_delta_scale", 1.0) == 1
            else True
        )
        return dict(estimate=_estimate(good))


def test_screen_preserves_baseline_and_accepts_only_freshly_validated_improvement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline competes in both stages while every requested shot is counted."""
    port = CountsPort()
    original = deepcopy(port.recipes)
    stages = Stages(monkeypatch)
    recipe, summary = module.screen_squad_gain_hypotheses(
        port, "sqrt_bswap", tmp_path, bootstrap=8
    )
    assert summary["qualified"]
    assert summary["selected_shape_changed"]
    assert summary["selected_pair"] == [1.6, 1.6]
    assert recipe["design_delta_scale"] == 1.6
    assert recipe["null_shot_interval_passed"]
    changed_training = [
        call for call in port.calls if "candidates" in call["directory"]
    ]
    assert all(
        not call["recipes"]["sqrt_bswap"]["null_shot_interval_passed"]
        for call in changed_training
    )
    assert port.recipes == original
    assert summary["budget"]["requested_shots"] == sum(
        call["shots"] for call in port.calls
    )
    assert len(stages.null_calls) == 1
    assert stages.null_calls[0]["recenter"] is False
    assert stages.null_calls[0]["shots"] == 2048
    assert (tmp_path / "qualified_recipe.json").is_file()


def test_unresolved_second_stage_keeps_verified_baseline_without_null_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Higher point estimates below shot uncertainty do not trigger another shape's null."""
    port = CountsPort()
    stages = Stages(monkeypatch)
    stages.scores = {1.0: 0.8, 1.3: 0.8001, 1.6: 0.8002, 2.0: 0.8003}
    recipe, summary = module.screen_squad_gain_hypotheses(
        port, "sqrt_bswap", tmp_path, allow_endpoint_extension=False, bootstrap=8
    )
    assert summary["retained_baseline"]
    assert recipe["design_delta_scale"] == 1
    assert not stages.null_calls
    assert summary["selection_training"]["role"] == "second_stage_training"
    assert not summary["resolved_improvement"]


def test_final_nonimprovement_cannot_select_an_alternate_changed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed final selected candidate returns only the independently checked baseline."""
    port = CountsPort()
    stages = Stages(monkeypatch)
    stages.final_scores = {1.0: 0.8, 1.6: 0.77, 2.0: 0.95}
    recipe, summary = module.screen_squad_gain_hypotheses(
        port, "sqrt_bswap", tmp_path, bootstrap=8
    )
    assert summary["retained_baseline"]
    assert recipe["design_delta_scale"] == 1
    assert len(stages.null_calls) == 1
    assert not summary["final_validation"]["candidate_accepted"]


def test_postnull_recenter_change_gets_exactly_one_additional_null_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new carrier after the one ON-map is requalified once without an endless loop."""
    port = CountsPort()
    stages = Stages(monkeypatch)
    stages.recenter_after_null_changes = True
    recipe, summary = module.screen_squad_gain_hypotheses(
        port,
        "sqrt_bswap",
        tmp_path,
        candidate_pairs=[(1.6, 1.6)],
        allow_endpoint_extension=False,
        bootstrap=8,
    )
    assert summary["selected_shape_changed"]
    assert len(stages.null_calls) == 2
    assert stages.null_calls[1]["recipe"]["frequency_ghz"] == pytest.approx(
        recipe["frequency_ghz"]
    )


def test_training_partition_exhaustion_uses_only_reserved_baseline_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A predeclared training cap retains a verified baseline and records incomplete search."""
    port = CountsPort()
    Stages(monkeypatch)
    _, summary = module.screen_squad_gain_hypotheses(
        port, "sqrt_bswap", tmp_path, max_total_shots=700_000, bootstrap=8
    )
    assert summary["status"] == "baseline_retained_budget_limit"
    assert summary["qualified"]
    assert summary["retained_baseline"]
    assert not summary["screen_complete"]
    assert summary["budget"]["requested_shots"] <= 700_000
    assert any(
        "final_baseline" in call["directory"] and call["shots"] == 8192
        for call in port.calls
    )


@pytest.mark.parametrize(
    ("failure", "exception"),
    [
        ("timeout", TimeoutError),
        ("malformed", ValueError),
        ("change_scale", module.ScreenIdentityError),
        ("change_classifier", module.ScreenIdentityError),
        ("change_classifier_precision", module.ScreenIdentityError),
    ],
)
def test_safety_failures_propagate_instead_of_becoming_baseline_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    exception: type[Exception],
) -> None:
    """Deadline, malformed counts and changed physical scale never use scientific fallback."""
    port = CountsPort()
    Stages(monkeypatch)
    setattr(port, failure, True)
    with pytest.raises(exception):
        module.screen_squad_gain_hypotheses(port, "sqrt_bswap", tmp_path, bootstrap=8)
    assert not (tmp_path / "qualified_recipe.json").exists()


def test_empty_candidate_list_verifies_full_baseline_without_shape_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full-gate baseline-only policy performs exactly its reserved validation."""
    port = CountsPort()
    stages = Stages(monkeypatch)
    _, summary = module.screen_squad_gain_hypotheses(
        port,
        "bswap",
        tmp_path,
        candidate_pairs=[],
        allow_endpoint_extension=False,
        bootstrap=8,
    )
    assert summary["retained_baseline"]
    assert not summary["screen_complete"]
    assert summary["budget"]["requested_shots"] == 344064
    assert not stages.null_calls


def test_failed_baseline_null_cannot_be_called_an_acceptable_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A baseline that no longer passes its own current ZZ test stops the screen."""
    port = CountsPort()
    stages = Stages(monkeypatch)
    stages.baseline_null_good = False
    with pytest.raises(QualificationError, match="baseline"):
        module.screen_squad_gain_hypotheses(
            port, "bswap", tmp_path, candidate_pairs=[], bootstrap=8
        )
    assert not (tmp_path / "qualified_recipe.json").exists()


def test_new_shape_null_failure_preserves_evidence_and_verifies_only_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresolved new-shape null is not promoted and cannot select another candidate."""
    port = CountsPort()
    stages = Stages(monkeypatch)
    stages.fail_new_null = True
    recipe, summary = module.screen_squad_gain_hypotheses(
        port, "sqrt_bswap", tmp_path, bootstrap=8
    )
    assert summary["retained_baseline"]
    assert recipe["design_delta_scale"] == 1
    assert len(stages.null_calls) == 1


def test_nonunit_baseline_design_and_uncoupled_pairs_are_rejected_before_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gain-hypothesis screen requires its explicitly defined unit-design baseline."""
    port = CountsPort()
    Stages(monkeypatch)
    with pytest.raises(ValueError, match="coupled"):
        module.screen_squad_gain_hypotheses(
            port, "sqrt_bswap", tmp_path / "bad-pair", candidate_pairs=[(1.3, 1.0)]
        )
    port.recipes["sqrt_bswap"]["design_delta_scale"] = 1.1
    with pytest.raises(ValueError, match="baseline"):
        module.screen_squad_gain_hypotheses(port, "sqrt_bswap", tmp_path / "bad-base")
    assert not port.calls


def test_waveform_manifest_uses_actual_coupled_design_not_historical_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sample provenance follows current carrier/scale while retaining the old delta as history."""
    import json

    port = CountsPort()
    port.recipes["sqrt_bswap"]["squad_delta_control"] = 123.0
    Stages(monkeypatch)
    module.screen_squad_gain_hypotheses(
        port, "sqrt_bswap", tmp_path, candidate_pairs=[(1.6, 1.6)], bootstrap=8
    )
    manifest = json.loads(
        (tmp_path / "frozen_candidate_waveform/waveform_manifest.json").read_text()
    )
    assert manifest["historical_squad_delta_control"] == 123.0
    assert not manifest["historical_delta_used_for_materialization"]
    assert manifest["design_delta_rad_per_ns"] == pytest.approx(
        1.6 * 2 * np.pi * (4.4 - manifest["carrier_ghz"])
    )
    assert manifest["c_over_s"] == pytest.approx(1.0)
    with np.load(manifest["envelope_npz"], allow_pickle=False) as stored:
        assert np.max(np.abs(stored["iq_command"])) == pytest.approx(
            manifest["peak_complex_command"]
        )


class AnalyticPort(CountsPort):
    """Synthetic logical gate responses exercise the real phase and score protocols."""

    def acquire(
        self,
        gates: Any,
        directory: Any,
        label: str,
        *,
        shots: int,
        recipes: Any,
        prepared: Any = ("0", "0"),
        basis: str = "ZZ",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate a synthetic calibrated zero-ZZ gate and ordinary analysis rotations."""
        self.calls.append(
            dict(
                directory=str(directory),
                shots=shots,
                recipes=deepcopy(recipes),
                label=label,
            )
        )
        vectors = {
            "0": np.array([1, 0]),
            "1": np.array([0, 1]),
            "+": np.array([1, 1]) / np.sqrt(2),
            "-": np.array([1, -1]) / np.sqrt(2),
            "+i": np.array([1, 1j]) / np.sqrt(2),
            "-i": np.array([1, -1j]) / np.sqrt(2),
        }
        vector = np.kron(vectors[prepared[0]], vectors[prepared[1]])
        for gate in gates:
            kind = "bswap" if gate == "BSWAP" else "sqrt_bswap"
            true = self.recipes[kind]["phase_calibration"]
            requested = recipes[kind]["phase_calibration"]
            pre = np.array(
                [requested["pre_active_rad"], requested.get("pre_passive_rad", 0)]
            )
            post = np.array(
                [requested["post_active_rad"], requested["post_passive_rad"]]
            )
            raw = (
                local_z([true["post_active_rad"], true["post_passive_rad"]])
                @ ideal_circuit_unitary([gate])
                @ local_z([true["pre_active_rad"], 0])
            )
            phase = pre.sum() / 2
            logical = (
                local_z(-pre - post + [phase, phase]) @ raw @ local_z([-phase, -phase])
            )
            vector = logical @ vector
        for qubit, axis in enumerate(basis):
            if axis != "Z":
                vector = local_xy(qubit, -np.pi / 2 if axis == "X" else 0) @ vector
        probabilities = np.abs(vector) ** 2
        probabilities /= probabilities.sum()
        counts = np.floor(probabilities * shots).astype(int)
        remaining = shots - int(counts.sum())
        if remaining:
            counts[np.argsort(probabilities * shots - counts)[-remaining:]] += 1
        return dict(counts=counts.tolist(), shots=shots, synthetic=True)


def test_baseline_only_integrates_real_phase_and_score_helpers(tmp_path: Path) -> None:
    """Real count-based phase cycling and repeated-state scores can validate the frozen baseline."""
    port = AnalyticPort()
    original = deepcopy(port.recipes)
    recipe, summary = module.screen_squad_gain_hypotheses(
        port, "sqrt_bswap", tmp_path, candidate_pairs=[], bootstrap=20
    )
    assert summary["qualified"]
    assert summary["retained_baseline"]
    assert recipe["design_delta_scale"] == 1.0
    assert summary["budget"]["requested_shots"] == 344064
    assert len(port.calls) == 72
    assert port.recipes == original


def test_full_transfer_candidate_gets_its_own_null_and_keeps_root_record_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root-selected pair is only a design input, never transferred full-gate qualification."""
    port = CountsPort()
    port.recipes["sqrt_bswap"].update(design_delta_scale=1.6, cd_strength=1.6)
    original = deepcopy(port.recipes)
    stages = Stages(monkeypatch)
    recipe, summary = module.screen_squad_gain_hypotheses(
        port,
        "bswap",
        tmp_path,
        candidate_pairs=[(1.6, 1.6)],
        allow_endpoint_extension=False,
        bootstrap=8,
    )
    assert summary["selected_shape_changed"]
    assert recipe["duration_ns"] == 260
    assert len(stages.null_calls) == 1
    assert stages.null_calls[0]["recipe"]["gate_kind"] == "bswap"
    assert port.recipes == original


def test_nested_null_budget_exhaustion_still_leaves_baseline_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precision-null partial acquisitions cannot consume the protected baseline validation."""
    port = CountsPort()
    stages = Stages(monkeypatch)
    _, summary = module.screen_squad_gain_hypotheses(
        port,
        "sqrt_bswap",
        tmp_path,
        candidate_pairs=[(1.6, 1.6)],
        max_total_shots=1_200_000,
        bootstrap=8,
    )
    assert len(stages.null_calls) == 1
    assert summary["status"] == "baseline_retained_budget_limit"
    assert summary["final_validation"]["baseline"]["passed"]
    assert summary["budget"]["requested_shots"] == sum(
        call["shots"] for call in port.calls
    )


def test_recenter_cannot_change_the_fixed_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A returned duration change is rejected before a different family's phase acquisition."""
    port = CountsPort()
    stages = Stages(monkeypatch)
    original_recenter = stages.recenter

    def bad_recenter(*args: Any, **kwargs: Any) -> Any:
        recipe, report = original_recenter(*args, **kwargs)
        recipe["duration_ns"] += 2
        return recipe, report

    monkeypatch.setattr(module, "recenter_amplitude_frequency", bad_recenter)
    with pytest.raises(module.ScreenIdentityError, match="duration"):
        module.screen_squad_gain_hypotheses(port, "sqrt_bswap", tmp_path, bootstrap=8)
    assert not (tmp_path / "qualified_recipe.json").exists()


def test_returned_zz_model_uses_final_prebenchmark_null_without_changing_local_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected fresh null replaces only the benchmark ZZ model and preserves old provenance."""
    port = CountsPort()
    port.recipes["bswap"]["zz_phase_rad"] = 0.006
    before = deepcopy(port.recipes["bswap"])
    Stages(monkeypatch)
    selected, summary = module.screen_squad_gain_hypotheses(
        port, "bswap", tmp_path, candidate_pairs=[], bootstrap=8
    )
    assert (
        selected["zz_phase_rad"]
        == summary["final_validation"]["baseline"]["null"]["zz_phase_rad"]
    )
    assert selected["shape_screen_prior_zz_model"]["zz_phase_rad"] == 0.006
    assert selected["phase_calibration"] == before["phase_calibration"]
    assert selected["zz_model_source"]["case"] == "baseline"
    assert port.recipes["bswap"] == before
