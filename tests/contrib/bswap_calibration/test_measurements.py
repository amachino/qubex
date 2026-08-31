"""Fake-experiment integration tests for count-preserving bSWAP measurements."""

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from qxpulse import PulseSchedule, Rect

from qubex.contrib.experiment.bswap_calibration import measurements as module
from qubex.contrib.experiment.bswap_calibration.irb import NativeBSWAPCache
from qubex.contrib.experiment.bswap_calibration.measurements import (
    CampaignMeasurements,
    acquire_irb,
    acquire_xeb,
    fit_unclipped_decay,
)
from qubex.contrib.experiment.bswap_calibration.pulses import (
    ideal_circuit_unitary,
    xeb_circuit,
)
from qubex.contrib.experiment.bswap_calibration.tomography import state_vector
from qubex.experiment.experiment import Experiment


class _Classifier:
    def __init__(self, *, failure: bool = False, fractional: bool = False) -> None:
        self.failure = failure
        self.fractional = fractional

    def predict(self, values: NDArray[np.complex128]) -> NDArray[np.float64]:
        if self.failure:
            raise ValueError("synthetic classification failure")
        if self.fractional:
            return np.full(len(values), 0.5)
        return values.real.astype(float)


class _Experiment:
    def __init__(self) -> None:
        self.classifiers = {q: _Classifier() for q in ("A", "P")}
        self.pulse = SimpleNamespace(
            get_drag_hpi_pulse=lambda _: Rect(duration=16, amplitude=0.1),
            get_drag_pi_pulse=lambda _: Rect(duration=24, amplitude=0.2),
            rabi_params={"A": SimpleNamespace(frequency=0.04)},
        )
        self.params = SimpleNamespace(get_control_amplitude=lambda _: 0.5)
        self.targets = {
            q: SimpleNamespace(frequency=f)
            for q, f in {"A": 5.0, "P": 4.4, "D": 4.7, "C": 4.71}.items()
        }
        self.counts: list[list[int]] = []
        self.calls: list[dict[str, Any]] = []

    def measure(self, sequence: PulseSchedule, **options: Any) -> SimpleNamespace:
        self.calls.append({"sequence": sequence, **deepcopy(options)})
        shots = options["n_shots"]
        counts = self.counts.pop(0) if self.counts else [shots, 0, 0, 0]
        states = np.repeat(np.arange(4), counts)
        return SimpleNamespace(
            data={
                "A": SimpleNamespace(kerneled=(states // 2).astype(complex)),
                "P": SimpleNamespace(kerneled=(states % 2).astype(complex)),
            }
        )


def _recipes() -> dict[str, dict[str, Any]]:
    common = dict(
        amplitude=0.5,
        frequency_ghz=4.7,
        ramp_ns=16.0,
        cd_strength=0.5,
        design_delta_scale=1.0,
        window={"type": "hann"},
        gate_start_ns=24.0,
        phase_calibration=dict(
            pre_active_rad=0.0, post_active_rad=0.1, post_passive_rad=-0.2
        ),
        zz_phase_rad=0.4,
    )
    return {
        "bswap": dict(common, gate_kind="bswap", duration_ns=200.0),
        "sqrt_bswap": dict(
            common, gate_kind="sqrt_bswap", duration_ns=116.0, frequency_ghz=4.7006
        ),
    }


def _measurements(tmp_path: Path) -> tuple[CampaignMeasurements, _Experiment]:
    exp = _Experiment()
    measurements = CampaignMeasurements(
        cast(Experiment, exp),
        tmp_path / "no_metadata_needed",
        _recipes(),
        qubits=("A", "P"),
        drive_label="D",
        cancel_label="C",
    )
    return measurements, exp


def test_acquire_uses_frozen_target_references(tmp_path: Path) -> None:
    """Compilation and measurement use the same captured custom-target frequencies."""
    measurements, exp = _measurements(tmp_path)
    exp.targets["D"].frequency = 4.72
    row = measurements.acquire(["BSWAP"], tmp_path, "fixed_target", shots=16)

    assert exp.calls[0]["frequencies"] == {"D": 4.7, "C": 4.71}
    assert exp.calls[0]["enable_dsp_classification"] is False
    assert row["counts"] == [16, 0, 0, 0]
    assert not measurements.run.exists()


@pytest.mark.parametrize("failure", ["exception", "fractional"])
def test_raw_iq_survives_classifier_failure(tmp_path: Path, failure: str) -> None:
    """Raw per-shot IQ is retained before classifier errors or invalid labels fail."""
    measurements, exp = _measurements(tmp_path)
    exp.classifiers["A"] = _Classifier(
        failure=failure == "exception", fractional=failure == "fractional"
    )
    with pytest.raises(ValueError, match=r"classification|classifier"):
        measurements.acquire(["BSWAP"], tmp_path, "invalid_classifier", shots=16)

    paths = list(tmp_path.glob("*.npz"))
    assert len(paths) == 1
    with np.load(paths[0]) as saved:
        assert saved["iq"].shape == (2, 16)


def test_assignment_columns_and_unclipped_normalization(tmp_path: Path) -> None:
    """Assignment columns are prepared states and inverse values are never clipped."""
    measurements, exp = _measurements(tmp_path)
    columns = [[8, 1, 1, 0], [2, 7, 0, 1], [1, 1, 8, 0], [0, 2, 1, 7]]
    exp.counts = deepcopy(columns)
    report = measurements.calibrate_assignment(tmp_path / "assignment", shots=10)
    expected = np.asarray(columns).T / 10
    np.testing.assert_allclose(report["matrix"], expected, atol=1e-15, rtol=0)
    assert report["convention"] == "C[reported, prepared]"

    exp.counts = [[10, 0, 0, 0]]
    row = measurements.acquire([], tmp_path, "normalized", shots=10)
    corrected = np.linalg.solve(expected, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        row["mitigated_probabilities_unclipped"], corrected, atol=1e-14, rtol=0
    )
    assert corrected.min() < 0
    assert corrected.max() > 1
    assert row["raw_probabilities"] == [1.0, 0.0, 0.0, 0.0]
    assert "not detector-only" in row["mitigation_scope"]


@pytest.mark.parametrize("gate_name", ["BSWAP", "RAW_SQRT_BSWAP"])
def test_acquire_xeb_freezes_raw_target_and_compares_same_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_name: str
) -> None:
    """Both targets use one frozen 64-pattern acquisition despite later recipe changes."""
    measurements, _ = _measurements(tmp_path)
    frozen = deepcopy(measurements.recipes)
    zz = {kind: recipe["zz_phase_rad"] for kind, recipe in frozen.items()}
    measurements.assignment = np.eye(4)
    measurements.assignment_source = "synthetic identity response"
    acquired_recipes: list[dict[str, Any]] = []
    original_acquire = measurements.acquire

    def acquire(
        gates: list[Any], directory: Path, label: str, **options: Any
    ) -> dict[str, Any]:
        acquired_recipes.append(deepcopy(options["recipes"]))
        kind = "bswap" if gate_name == "BSWAP" else "sqrt_bswap"
        measurements.recipes[kind]["zz_phase_rad"] = 1.2
        measurements.recipes[kind]["frequency_ghz"] += 0.001
        measurements.recipes[kind]["phase_calibration"]["post_active_rad"] = 0.8
        return original_acquire(gates, directory, label, **options)

    def summarize(
        rows: list[dict[str, Any]],
        depths: list[int],
        *,
        mitigated: bool = False,
        probability_key: str = "ideal_probabilities",
    ) -> dict[str, Any]:
        return {
            "depths": depths,
            "scores": [0.0] * len(depths),
            "errors": [0.1] * len(depths),
            "probability_key": probability_key,
            "mitigated": mitigated,
        }

    monkeypatch.setattr(measurements, "acquire", acquire)
    monkeypatch.setattr(module, "summarize_xeb", summarize)
    summary, rows = acquire_xeb(
        measurements, gate_name, tmp_path, depths=[2, 3], seeds=[1, 2], shots=16
    )
    assert len(rows) == 4
    assert len(acquired_recipes) == 4
    assert all(recipe == frozen for recipe in acquired_recipes)
    assert (
        summary["target"] == f"frozen independently calibrated raw-ZZ model {gate_name}"
    )
    assert (
        summary["zero_zz_diagnostic"]["target"] == f"frozen zero-ZZ ideal {gate_name}"
    )
    assert (
        summary["zero_zz_diagnostic"]["probability_key"]
        == "zero_zz_ideal_probabilities"
    )
    assert summary["mitigated_unclipped"]["mitigated"]
    assert summary["zero_zz_diagnostic"]["mitigated_unclipped"]["mitigated"]
    assert not summary["target_refitted_on_benchmark"]
    saved_target = json.loads(Path(summary["frozen_target_file"]).read_text())
    assert saved_target["recipes"] == frozen
    assert saved_target["zz_phases_rad"] == zz
    different_targets = False
    for row in rows:
        plan = xeb_circuit(row["depth"], row["seed"], gate_name)
        assert row["local_indices"] == plan["local_indices"]
        assert row["gates"] == plan["gates"]
        assert np.asarray(row["local_indices"]).shape == (row["depth"] + 1, 2)
        assert np.max(row["local_indices"]) > 7
        np.testing.assert_allclose(
            row["zero_zz_ideal_probabilities"],
            plan["ideal_probabilities"],
            atol=1e-14,
            rtol=0,
        )
        expected_raw = (
            np.abs(ideal_circuit_unitary(plan["gates"], zz_phases=zz)[:, 0]) ** 2
        )
        np.testing.assert_allclose(
            row["raw_model_probabilities"], expected_raw, atol=1e-14, rtol=0
        )
        assert row["ideal_probabilities"] == row["raw_model_probabilities"]
        saved_row = json.loads(Path(row["iq_file"]).with_suffix(".json").read_text())
        assert saved_row["raw_model_probabilities"] == row["raw_model_probabilities"]
        assert saved_row["frozen_target_sha256"] == row["frozen_target_sha256"]
        different_targets |= not np.allclose(
            expected_raw, plan["ideal_probabilities"], atol=1e-10, rtol=0
        )
    assert different_targets


def test_irb_primary_analysis_never_receives_normalized_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw IRB and out-of-range combined-SPAM diagnostics stay separate."""
    measurements, _ = _measurements(tmp_path)
    captured: dict[str, Any] = {}

    def acquire(
        gates: list[str], directory: Path, label: str, *, shots: int
    ) -> dict[str, Any]:
        raw = 80 if label.startswith("reference") else 70
        normalized = 1.2 if label.startswith("reference") else -0.2
        return {
            "counts": [raw, shots - raw, 0, 0],
            "mitigated_probabilities_unclipped": [normalized, 0, 0, 0],
        }

    def analyze(
        depths: list[int], reference: np.ndarray, interleaved: np.ndarray
    ) -> dict[str, Any]:
        captured["reference"], captured["interleaved"] = (
            reference.copy(),
            interleaved.copy(),
        )
        return {"fidelity_estimate": 0.91, "quote_as_irb_estimate": True}

    def diagnostic(depths: list[int], values: np.ndarray) -> dict[str, Any]:
        return {
            "input_min": float(values.min()),
            "input_max": float(values.max()),
            "clipping": False,
        }

    monkeypatch.setattr(measurements, "acquire", acquire)
    monkeypatch.setattr(module, "make_irb_circuit", lambda *args: [])
    monkeypatch.setattr(module, "analyze_irb", analyze)
    monkeypatch.setattr(module, "fit_unclipped_decay", diagnostic)
    summary, _ = acquire_irb(
        measurements,
        cast(NativeBSWAPCache, SimpleNamespace()),
        "BSWAP",
        tmp_path,
        depths=[0, 1],
        seeds=[1, 2],
        shots=100,
    )
    np.testing.assert_allclose(captured["reference"], 0.8, atol=0, rtol=0)
    np.testing.assert_allclose(captured["interleaved"], 0.7, atol=0, rtol=0)
    normalized = summary["mitigated_unclipped"]
    assert normalized["fits"]["reference"]["input_max"] == 1.2
    assert normalized["fits"]["interleaved"]["input_min"] == -0.2
    assert "fidelity_estimate" not in normalized
    assert summary["fidelity_estimate"] == 0.91


def test_unclipped_decay_retains_out_of_range_data() -> None:
    """Diagnostic fitting retains negative and above-one normalization outcomes."""
    depths = np.array([0, 1, 2, 4, 8, 16, 32])
    values = np.tile(1.5 * 0.9**depths - 0.2, (8, 1))
    result = fit_unclipped_decay(depths, values)
    assert result["input_min"] < 0
    assert result["input_max"] > 1
    assert result["clipping"] is False
    np.testing.assert_allclose(result["means"], values[0], atol=1e-14, rtol=0)


def test_xeb_observable_uses_selected_target_on_same_counts() -> None:
    """Raw-model and zero-ZZ observables differ without another acquisition."""
    rows = [
        dict(
            depth=depth,
            counts=[50, 20, 20, 10],
            ideal_probabilities=[0.5, 0.2, 0.2, 0.1],
            zero_zz_ideal_probabilities=[0.7, 0.1, 0.1, 0.1],
        )
        for depth in [1, 2, 4, 8]
        for _ in range(4)
    ]
    raw = module.summarize_xeb(rows, [1, 2, 4, 8])
    zero = module.summarize_xeb(
        rows, [1, 2, 4, 8], probability_key="zero_zz_ideal_probabilities"
    )
    np.testing.assert_allclose(raw["scores"], 1.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(zero["scores"], 5 / 9, atol=1e-14, rtol=0)
    assert "single-gate fidelity" in raw["claim"]
    assert not raw["decay_identified"]


def test_gate_checks_report_state_overlap_not_gate_fidelity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw-ZZ model agreement is labeled as input-state overlap, never gate fidelity."""
    measurements, _ = _measurements(tmp_path)
    zz = {kind: recipe["zz_phase_rad"] for kind, recipe in measurements.recipes.items()}

    def tomography(
        gates: list[str], state: tuple[str, str], *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        vector = ideal_circuit_unitary(gates, zz_phases=zz) @ state_vector(state)
        return np.outer(vector, vector.conj()), np.zeros((9, 4), dtype=int)

    monkeypatch.setattr(measurements, "tomography", tomography)
    results = measurements.gate_checks(["BSWAP"], tmp_path, shots=16)
    coherent = [row for row in results["BSWAP"] if "raw_model_state_overlap" in row]
    assert coherent
    for row in coherent:
        assert row["raw_model_state_overlap"] == pytest.approx(1, abs=1e-12)
        assert "ideal_state_overlap" in row
        assert "gate_fidelity" not in row


def test_expired_deadline_stops_before_measurement(tmp_path: Path) -> None:
    """An expired aware deadline prevents all measurement requests."""
    measurements, exp = _measurements(tmp_path)
    measurements.deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(TimeoutError, match="Reservation"):
        measurements.acquire([], tmp_path, "expired")
    assert exp.calls == []


@pytest.mark.parametrize("benchmark", ["irb", "xeb"])
def test_nonvector_benchmark_depths_fail_before_measurement(
    tmp_path: Path, benchmark: str
) -> None:
    """Malformed scan shapes fail before any measurement request is made."""
    measurements, exp = _measurements(tmp_path)
    if benchmark == "irb":
        with pytest.raises(ValueError, match="one-dimensional"):
            acquire_irb(
                measurements,
                cast(NativeBSWAPCache, SimpleNamespace()),
                "BSWAP",
                tmp_path,
                depths=np.array([[1, 2]]),
                seeds=[1, 2],
            )
    else:
        with pytest.raises(ValueError, match="one-dimensional"):
            acquire_xeb(
                measurements,
                "BSWAP",
                tmp_path,
                depths=np.array([[1, 2]]),
                seeds=[1, 2],
            )
    assert exp.calls == []


def test_xeb_existing_target_manifest_is_not_overwritten(tmp_path: Path) -> None:
    """A repeated destination cannot replace an earlier frozen-target manifest."""
    measurements, exp = _measurements(tmp_path)
    manifest = tmp_path / "xeb_frozen_targets.json"
    manifest.write_text("previous frozen target")
    with pytest.raises(FileExistsError, match="already exist"):
        acquire_xeb(measurements, "BSWAP", tmp_path, depths=[1, 2], seeds=[1, 2])
    assert manifest.read_text() == "previous frozen target"
    assert exp.calls == []


def test_xeb_depth_zero_is_rejected_before_acquisition(tmp_path: Path) -> None:
    """Unidentifiable terminal-only depth zero fails before spending shots."""
    measurements, exp = _measurements(tmp_path)
    with pytest.raises(ValueError, match="depth 0"):
        acquire_xeb(measurements, "BSWAP", tmp_path, depths=[0, 1, 2], seeds=[1, 2])
    assert exp.calls == []
