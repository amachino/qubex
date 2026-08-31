"""Synthetic/fake-callback tests for fixed-waveform resonance anchors."""

import json
from copy import deepcopy

import numpy as np
import pytest
from qxpulse import Rect

from qubex.contrib.experiment.bswap_calibration.drift import (
    fit_resonance_anchor,
    plot_resonance_history,
    record_resonance_anchor,
)
from qubex.contrib.experiment.bswap_calibration.pulses import make_squad_pulse


class FakeMeasurements:
    """Provide synthetic count-based measurements for anchor tests."""

    def __init__(self, *, center_offset_mhz=0.25, fail_after=None):
        self.qubits = ("A", "P")
        self.references = {"A": 5.0, "P": 4.4}
        self.targets = {"drive": 4.7, "cancel": 4.7}
        self.rabi_scale = 0.08
        self.session_id = "same-session"
        self.x90 = {
            q: Rect(duration=16, amplitude=0.1, sampling_period=2) for q in self.qubits
        }
        self.xpi = {
            q: Rect(duration=24, amplitude=0.1, sampling_period=2) for q in self.qubits
        }
        self.classifiers = {q: {"frozen_model": q} for q in self.qubits}
        recipe = dict(
            gate_kind="bswap",
            amplitude=0.5,
            frequency_ghz=4.7,
            duration_ns=200.0,
            ramp_ns=16.0,
            cd_strength=0.5,
            design_delta_scale=1.15,
            window={"type": "hann"},
            gate_start_ns=24.0,
            phase_calibration=dict(
                pre_active_rad=0.0, post_active_rad=0.2, post_passive_rad=-0.3
            ),
            cancel_amplitude_ratio=0.1,
            cancel_phase_rad=0.4,
        )
        self.recipes = {
            "bswap": recipe,
            "sqrt_bswap": dict(recipe, gate_kind="sqrt_bswap", duration_ns=116.0),
        }
        self.calls = []
        self.center_offset_mhz = center_offset_mhz
        self.fail_after = fail_after

    def acquire(self, gates, directory, label, *, prepared, basis, shots, recipes):
        """Return synthetic all-shot counts without accessing hardware."""
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("synthetic interruption")
        self.calls.append(
            dict(gates=gates, prepared=prepared, recipes=deepcopy(recipes), shots=shots)
        )
        offset = 1e3 * (
            recipes["bswap"]["frequency_ghz"] - self.recipes["bswap"]["frequency_ghz"]
        )
        probability = 0.9 - 0.10 * (offset - self.center_offset_mhz) ** 2
        successes = round(shots * probability)
        counts = (
            [shots - successes, 0, 0, successes]
            if prepared == ("0", "0")
            else [successes, 0, 0, shots - successes]
        )
        return dict(counts=counts, timestamp="2026-08-31T12:00:00.000001+09:00")


def synthetic_counts(probabilities, shots=10000):
    """Return deterministic bidirectional counts for a transfer curve."""
    probabilities = np.broadcast_to(probabilities, (2, len(probabilities)))
    success = np.rint(probabilities * shots).astype(int)
    result = np.zeros((2, success.shape[1], 4), dtype=int)
    result[0, :, 3], result[0, :, 0] = success[0], shots - success[0]
    result[1, :, 0], result[1, :, 3] = success[1], shots - success[1]
    return result


def test_record_anchor_freezes_waveform_and_preserves_recipes(tmp_path):
    """An anchor should preserve the physical waveform and calibrated recipes."""
    measurements = FakeMeasurements()
    original = deepcopy(measurements.recipes)
    result = record_resonance_anchor(measurements, tmp_path, shots=4096)
    assert measurements.recipes == original
    assert result["qualified"], result["fit"]["reasons"]
    assert abs(result["fit"]["center_ghz"] - 4.70025) < 2e-5
    assert result["fit"]["center_sem_mhz"] > 0
    assert len(measurements.calls) == 18
    assert {call["prepared"] for call in measurements.calls} == {("0", "0"), ("1", "1")}
    nominal = make_squad_pulse(
        original["bswap"], rabi_ghz_per_amplitude=0.08, transition_frequency_ghz=5.0
    )
    for call in measurements.calls:
        assert call["gates"] == ["BSWAP"]
        recipe = call["recipes"]["bswap"]
        assert {
            k: v
            for k, v in recipe.items()
            if k not in ("frequency_ghz", "design_delta_scale")
        } == {
            k: v
            for k, v in original["bswap"].items()
            if k not in ("frequency_ghz", "design_delta_scale")
        }
        pulse = make_squad_pulse(
            recipe, rabi_ghz_per_amplitude=0.08, transition_frequency_ghz=5.0
        )
        np.testing.assert_allclose(pulse.values, nominal.values, atol=1e-13, rtol=1e-13)
    saved = np.load(result["data_file"])
    assert saved["counts"].shape == (2, 9, 4)
    assert np.all(saved["counts"].sum(axis=-1) == 4096)
    assert (saved["timestamps"] != "").all()
    assert json.loads(
        (
            tmp_path
            / result["anchor_directory"].split("/")[-1]
            / "resonance_anchor.json"
        ).read_text()
    )["qualified"]


def test_baseline_identity_is_stable_and_changes_with_settings(tmp_path):
    """Baseline identifiers should track comparable acquisition settings."""
    measurements = FakeMeasurements()
    first = record_resonance_anchor(measurements, tmp_path, shots=4096)
    second = record_resonance_anchor(measurements, tmp_path, shots=4096)
    assert first["baseline_id"] == second["baseline_id"]
    assert first["recipe_fingerprint"] == second["recipe_fingerprint"]
    assert first["anchor_directory"] != second["anchor_directory"]
    measurements.recipes["bswap"]["amplitude"] = 0.49
    changed = record_resonance_anchor(measurements, tmp_path, shots=4096)
    assert changed["baseline_id"] != first["baseline_id"]
    figure = plot_resonance_history(
        [first, second, changed], output_html=tmp_path / "history.html"
    )
    traces = figure.to_plotly_json()["data"]
    assert len(traces) == 2
    assert sorted(len(trace["x"]) for trace in traces) == [1, 2]
    assert (tmp_path / "history.html").exists()


def test_baseline_tracks_other_recipe_preparation_and_classifier_changes(tmp_path):
    """Preparation timing and classifiers should contribute to baseline identity."""
    measurements = FakeMeasurements()
    first = record_resonance_anchor(measurements, tmp_path, shots=4096)
    measurements.recipes["sqrt_bswap"]["gate_start_ns"] = 32.0
    second = record_resonance_anchor(measurements, tmp_path, shots=4096)
    assert first["recipe_fingerprint"] == second["recipe_fingerprint"]
    assert first["baseline_id"] != second["baseline_id"]
    measurements.classifiers["A"]["frozen_model"] = "new-model"
    third = record_resonance_anchor(measurements, tmp_path, shots=4096)
    assert third["baseline_id"] != second["baseline_id"]


@pytest.mark.parametrize("kind", ["flat", "convex", "outside"])
def test_unresolved_or_noninterior_peak_is_not_qualified(kind):
    """An unresolved or exterior peak should not be qualified."""
    offsets = np.linspace(-1.5, 1.5, 9)
    probability = {
        "flat": np.full(9, 0.5),
        "convex": 0.3 + 0.1 * offsets**2,
        "outside": 0.8 - 0.03 * (offsets - 2.0) ** 2,
    }[kind]
    result = fit_resonance_anchor(4.7 + offsets / 1000, synthetic_counts(probability))
    assert not result["qualified"]
    assert result["center_ghz"] is None
    assert result["center_sem_mhz"] is None


def test_bidirectional_disagreement_is_unqualified():
    """Disagreeing forward and reverse centers should not be qualified."""
    offsets = np.linspace(-1.5, 1.5, 9)
    counts = synthetic_counts(0.9 - 0.1 * (offsets - 0.3) ** 2)
    counts[1] = synthetic_counts(0.9 - 0.1 * (offsets + 0.3) ** 2)[1]
    result = fit_resonance_anchor(4.7 + offsets / 1000, counts)
    assert not result["qualified"]
    assert "bidirectional_centers_disagree" in result["reasons"]


def test_failed_acquisition_retains_partial_evidence(tmp_path):
    """Interrupted acquisition should retain its partial observations."""
    measurements = FakeMeasurements(fail_after=3)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        record_resonance_anchor(measurements, tmp_path)
    summary_path = next(tmp_path.rglob("resonance_anchor.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "failed"
    assert not summary["qualified"]
    assert summary["completed_at"] is not None
    assert np.isfinite(np.load(summary["data_file"])["counts"]).all(axis=-1).sum() == 3
    assert len(plot_resonance_history(tmp_path).to_plotly_json()["data"]) == 0


@pytest.mark.parametrize(
    "arguments", [dict(shots=0), dict(npoints=5), dict(npoints=8), dict(span_mhz=-1.0)]
)
def test_invalid_parameters_do_not_acquire(tmp_path, arguments):
    """Invalid anchor settings should fail before measurement."""
    measurements = FakeMeasurements()
    with pytest.raises(ValueError, match=r"shots|npoints|span_mhz"):
        record_resonance_anchor(measurements, tmp_path, **arguments)
    assert not measurements.calls
    assert not list(tmp_path.iterdir())


def test_zero_detuning_crossing_is_rejected_before_acquisition(tmp_path):
    """A sweep crossing zero detuning should fail before measurement."""
    measurements = FakeMeasurements()
    with pytest.raises(ValueError, match="zero detuning"):
        record_resonance_anchor(measurements, tmp_path, span_mhz=400.0)
    assert not measurements.calls
