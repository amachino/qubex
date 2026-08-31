"""Offline sampled-waveform/frame tests; no hardware or analog-fidelity claims."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from qxpulse import FlatTop, Rect

from qubex.contrib.experiment.bswap_calibration.pulses import (
    compile_campaign,
    ideal_circuit_unitary,
    local_xy,
    local_z,
    make_squad_pulse,
    measured_phase_vectors,
    single_qubit_ensemble64,
    xeb_circuit,
)

QUBITS = ("A", "P")
REFERENCES = {"A": 4.401, "P": 4.837}
TARGETS = {"D": 4.612, "C": 4.618}
GAIN = 0.636


def recipe(kind: str = "bswap", **updates) -> Any:
    """Return a synthetic measured-phase gate recipe."""
    result = {
        "gate_kind": kind,
        "amplitude": 0.91,
        "frequency_ghz": 4.6118,
        "duration_ns": 264.0 if kind == "bswap" else 144.0,
        "ramp_ns": 16.0,
        "cd_strength": 1.0,
        "design_delta_scale": 1.0,
        "window": {"type": "hann"},
        "gate_start_ns": 24.0,
        "phase_calibration": {
            "pre_active_rad": 0.0 if kind == "bswap" else 0.73,
            "post_active_rad": 0.61,
            "post_passive_rad": -0.82,
        },
        # Deliberately disagree: compiler must NOT read inverse corrections.
        "pre_vz_rad": [2.3, 0.8],
        "post_vz_rad": [-1.1, 1.7],
        "cancel_amplitude_ratio": 0.04,
        "cancel_phase_rad": 0.37,
    }
    result.update(updates)
    return result


def ingredients(recipes: Any = None) -> Any:
    """Return compiler inputs with fixed references and synthetic local pulses."""
    return dict(
        recipes=recipes
        or {
            "bswap": recipe(),
            "sqrt_bswap": recipe("sqrt_bswap", frequency_ghz=4.6126),
        },
        qubits=QUBITS,
        drive_label="D",
        cancel_label="C",
        target_frequencies_ghz=TARGETS,
        reference_frequencies_ghz=REFERENCES,
        rabi_ghz_per_amplitude=GAIN,
        x90={q: Rect(duration=16, amplitude=0.11) for q in QUBITS},
        xpi={q: Rect(duration=24, amplitude=0.16) for q in QUBITS},
    )


def waveform(rec: dict[str, Any]) -> Any:
    """Construct the synthetic recipe with the declared physical Rabi scale."""
    return make_squad_pulse(
        rec, rabi_ghz_per_amplitude=GAIN, transition_frequency_ghz=REFERENCES["A"]
    )


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.7])
@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_squad_angular_conversion_and_cd_sign(scale: float, strength: float) -> None:
    """Squad angular conversion and cd sign."""
    rec = recipe(design_delta_scale=scale, cd_strength=strength)
    pulse = waveform(rec)
    k = 2 * np.pi * GAIN
    delta = scale * 2 * np.pi * (REFERENCES["A"] - rec["frequency_ghz"])
    angular = FlatTop(
        duration=rec["duration_ns"],
        amplitude=k * rec["amplitude"],
        tau=16,
        type="Squad",
        delta=delta,
        window={"type": "hann"},
        correction_type="CD",
        correction_factor=strength,
        sampling_period=2.0,
    )
    np.testing.assert_allclose(k * pulse.values, angular.values, atol=1e-13)
    i = pulse.values.real
    expected_q = (
        -strength * delta * np.gradient(i, pulse.times) / (delta**2 + (k * i) ** 2)
    )
    np.testing.assert_allclose(pulse.values.imag, expected_q, atol=1e-13)
    assert pulse.duration == rec["duration_ns"]
    assert pulse.tau == 16
    assert np.max(i) == pytest.approx(rec["amplitude"])


def test_carrier_adaptive_design_and_window_copy() -> None:
    """Carrier adaptive design and window copy."""
    first = recipe(window={"type": "tukey", "rise_end": 0.2, "fall_start": 0.7})
    a = waveform(first)
    second = {**first, "frequency_ghz": first["frequency_ghz"] + 0.005}
    b = waveform(second)
    assert a.delta != b.delta
    assert not np.allclose(a.values, b.values)
    before = a.values.copy()
    first["window"]["rise_end"] = 0.4
    np.testing.assert_array_equal(a.values, before)


@pytest.mark.parametrize(
    "updates",
    [
        {"duration_ns": 31},
        {"duration_ns": 33},
        {"ramp_ns": 15},
        {"design_delta_scale": 0},
        {"design_delta_scale": -1},
        {"frequency_ghz": REFERENCES["A"]},
        {"amplitude": 1.01},
        {"amplitude": np.nan},
        {"window": "hann"},
        {"cd_strength": 1e5},
    ],
)
def test_invalid_pulse_parameters_fail(updates: dict[str, Any]) -> None:
    """Invalid pulse parameters fail."""
    with pytest.raises((ValueError, TypeError)):
        waveform(recipe(**updates))


def _samples(schedule: Any, label: str) -> Any:
    return np.asarray(schedule.get_sequence(label).get_values(apply_frame_shifts=True))


def _event_samples(
    schedule: Any, label: str, event: dict[str, Any], origin: float
) -> Any:
    start = round((event["start_ns"] - origin) / 2)
    length = round(event["duration_ns"] / 2)
    return _samples(schedule, label)[start : start + length]


def _phase_ratio(values: Any, expected: Any) -> Any:
    mask = np.abs(expected) > 1e-9
    ratio = values[mask] / expected[mask]
    np.testing.assert_allclose(np.abs(ratio), 1.0, atol=1e-12)
    phase = np.angle(np.mean(ratio))
    np.testing.assert_allclose(ratio, np.exp(1j * phase), atol=2e-11)
    return float(phase)


def test_exact_emitted_iq_mixed_carriers_and_same_carrier_cancel() -> None:
    """Exact emitted iq mixed carriers and same carrier cancel."""
    kw = ingredients()
    origin = 126.0
    gates = [("XY", 0.4, 1.2), ("VZ", 0.7, -0.9), "BSWAP", "ROOT_PAIR", "XX90"]
    sequence, report = compile_campaign(
        gates,
        **kw,
        global_start_ns=origin,
        prepared=("+", "+i"),
        basis="XY",
        delay_ns=18.0,
    )
    assert len([e for e in report["events"] if e["kind"] != "local"]) == 5
    for event in report["events"]:
        if event["kind"] == "local":
            pulse = (kw["xpi"] if event["angle_rad"] > 2 else kw["x90"])[
                QUBITS[event["qubit"]]
            ]
            emitted = _event_samples(sequence, QUBITS[event["qubit"]], event, origin)
            expected = pulse.values * np.exp(1j * event["source_phase_rad"])
            np.testing.assert_allclose(emitted, expected, atol=1e-12)
            continue
        rec = kw["recipes"][event["kind"]]
        pulse = waveform(rec)
        corrected = {}
        for label, scale, relative in (
            ("D", 1.0, 0.0),
            ("C", rec["cancel_amplitude_ratio"], rec["cancel_phase_rad"]),
        ):
            emitted = _event_samples(sequence, label, event, origin)
            offset = rec["frequency_ghz"] - TARGETS[label]
            absolute_time = event["start_ns"] + pulse.times
            expected = (
                scale
                * pulse.values
                * np.exp(1j * (event["logical_drive_phase_rad"] + relative))
            )
            expected *= np.exp(-2j * np.pi * offset * absolute_time)
            np.testing.assert_allclose(emitted, expected, atol=2e-11)
            corrected[label] = emitted * np.exp(2j * np.pi * offset * absolute_time)
        np.testing.assert_allclose(
            corrected["C"],
            corrected["D"]
            * rec["cancel_amplitude_ratio"]
            * np.exp(1j * rec["cancel_phase_rad"]),
            atol=2e-11,
        )


def _replay_emitted(
    schedule: Any, report: Any, kw: dict[str, Any], zz: dict[str, float]
) -> Any:
    """
    Replay actual emitted axes and calibrated primitive blackboxes.

    VirtualZ is NEVER applied as an extra physical operation. Its effect is
    read from the actual analysis pulse's IQ, just like all other local axes.
    This does not simulate the nonlinear analog response of a transmon.
    """
    state = np.array([1, 0, 0, 0], complex)
    origin = report["global_start_ns"]
    for event in sorted(
        report["events"], key=lambda e: (e["start_ns"], e.get("qubit", 2))
    ):
        kind = event["kind"]
        if kind == "local":
            qi, angle = event["qubit"], event["angle_rad"]
            pulse = (kw["xpi"] if angle > 2 else kw["x90"])[QUBITS[qi]]
            values = _event_samples(schedule, QUBITS[qi], event, origin)
            phase = _phase_ratio(values, pulse.values)
            operation = local_xy(qi, phase, angle)
        else:
            rec = kw["recipes"][kind]
            pulse = waveform(rec)
            values = _event_samples(schedule, "D", event, origin)
            offset = rec["frequency_ghz"] - TARGETS["D"]
            values = values * np.exp(
                2j * np.pi * offset * (event["start_ns"] + pulse.times)
            )
            phase = _phase_ratio(values, pulse.values)
            rate = np.pi * (2 * rec["frequency_ghz"] - sum(REFERENCES.values()))
            effective_phase = phase - rate * (event["start_ns"] - rec["gate_start_ns"])
            pre, post = measured_phase_vectors(rec)
            target = ideal_circuit_unitary(
                ["BSWAP" if kind == "bswap" else "RAW_SQRT_BSWAP"], zz_phases=zz
            )
            blackbox = local_z(post) @ target @ local_z(pre)
            operation = (
                local_z([effective_phase] * 2)
                @ blackbox
                @ local_z([-effective_phase] * 2)
            )
        state = operation @ state
    return np.abs(state) ** 2


@pytest.mark.parametrize("gate_name", ["BSWAP", "RAW_SQRT_BSWAP", "ROOT_PAIR", "XX90"])
def test_actual_waveform_frame_transport_and_analysis(gate_name: str) -> None:
    """Actual waveform frame transport and analysis."""
    rng = np.random.default_rng(8531)
    for trial in range(12):
        kw = ingredients()
        for kind, rec in kw["recipes"].items():
            pre, pa, pp = rng.uniform(-np.pi, np.pi, 3)
            rec["phase_calibration"] = {
                "pre_active_rad": pre if kind == "sqrt_bswap" else 0.0,
                "post_active_rad": pa,
                "post_passive_rad": pp,
            }
        zz = {"bswap": 0.37, "sqrt_bswap": -0.19}
        gates = xeb_circuit(3, trial + 110, gate_name)["gates"]
        gates = ["XI90", "ZI90", ("VZ", -0.31, 0.73), *gates, ("IDLE", 12.0), "IX90"]
        prepared = ("+", "+i")
        basis = ("XY", "YX", "XX", "YY")[trial % 4]
        schedule, report = compile_campaign(
            gates,
            **kw,
            prepared=prepared,
            basis=basis,
            global_start_ns=64.0,
            delay_ns=float(trial * 18),
        )
        actual = _replay_emitted(schedule, report, kw, zz)
        initial = np.kron([1, 1], [1, 1j]) / 2
        expected_state = ideal_circuit_unitary(gates, zz_phases=zz) @ initial
        for qi, axis in enumerate(basis):
            expected_state = (
                local_xy(qi, -np.pi / 2 if axis == "X" else 0.0) @ expected_state
            )
        np.testing.assert_allclose(actual, np.abs(expected_state) ** 2, atol=4e-11)


def test_no_phase_zero_fill_and_inverse_corrections_are_ignored() -> None:
    """No phase zero fill and inverse corrections are ignored."""
    kw = ingredients()
    del kw["recipes"]["bswap"]["phase_calibration"]
    with pytest.raises(KeyError, match="phase_calibration"):
        compile_campaign(["BSWAP"], **kw)
    rec = recipe()
    pre, post = measured_phase_vectors(rec)
    np.testing.assert_allclose(pre, [0, 0])
    np.testing.assert_allclose(post, [0.61, -0.82])


def test_64_ensemble_and_frozen_single_root_contract() -> None:
    """64 ensemble and frozen single root contract."""
    ensemble = single_qubit_ensemble64()
    assert len(ensemble) == len(set(ensemble)) == 64
    for axis, z in ensemble:
        kw = ingredients()
        gates = [("XY", axis, -0.3), ("VZ", z, 0.7), "RAW_SQRT_BSWAP"]
        schedule, report = compile_campaign(gates, **kw, basis="YX")
        actual = _replay_emitted(schedule, report, kw, {})
        expected = (
            local_xy(0, 0.0)
            @ local_xy(1, -np.pi / 2)
            @ ideal_circuit_unitary(gates)[:, 0]
        )
        np.testing.assert_allclose(actual, np.abs(expected) ** 2, atol=3e-11)
    plan = xeb_circuit(11, 7834, "RAW_SQRT_BSWAP")
    local = xeb_circuit(11, 7834, None)
    assert plan["local_indices"] == local["local_indices"]
    assert plan["gates"].count("RAW_SQRT_BSWAP") == 11
    assert not plan["target_refitted_on_benchmark"]
    np.testing.assert_allclose(
        plan["ideal_probabilities"],
        np.abs(ideal_circuit_unitary(plan["gates"])[:, 0]) ** 2,
    )


def test_root_pair_target_and_echo_cancel_model_zz() -> None:
    """Root pair target and echo cancel model zz."""
    np.testing.assert_allclose(
        ideal_circuit_unitary(["ROOT_PAIR"]),
        ideal_circuit_unitary(["BSWAP"]),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        ideal_circuit_unitary(["XX90"], zz_phases={"sqrt_bswap": 0.8}),
        ideal_circuit_unitary(["XX90"]),
        atol=1e-14,
    )


def test_terminal_layer_resolves_depth_one_full_bswap_xeb_contrast() -> None:
    """A terminal random local layer exposes nonuniform full-bSWAP output probabilities."""
    probabilities = np.array(
        [xeb_circuit(1, seed, "BSWAP")["ideal_probabilities"] for seed in range(128)]
    )
    contrast = np.sum((probabilities - 0.25) ** 2, axis=1)
    assert float(np.mean(contrast)) > 0.05
    assert np.count_nonzero(contrast > 1e-10) > 64


@pytest.mark.parametrize("depth", [0, 1, 3, 11])
@pytest.mark.parametrize("gate_name", ["BSWAP", "RAW_SQRT_BSWAP", "ROOT_PAIR", "XX90"])
def test_terminal_layer_preserves_entangling_depth_and_paired_local_indices(
    depth: int, gate_name: str
) -> None:
    """Terminal local randomization adds no entangler and matches the local-only reference."""
    plan = xeb_circuit(depth, 4121, gate_name)
    reference = xeb_circuit(depth, 4121, None)
    assert plan["terminal_layer"] is True
    assert len(plan["local_indices"]) == depth + 1
    assert plan["local_indices"] == reference["local_indices"]
    assert plan["gates"].count(gate_name) == depth
    assert plan["gates"][-2][0] == "XY"
    assert plan["gates"][-1][0] == "VZ"
    assert len(reference["gates"]) == 2 * (depth + 1)
    assert plan == xeb_circuit(depth, 4121, gate_name)
    np.testing.assert_allclose(
        plan["ideal_probabilities"],
        np.abs(ideal_circuit_unitary(plan["gates"])[:, 0]) ** 2,
        atol=1e-14,
    )


def test_terminal_layer_opt_out_documents_the_uniform_depth_one_counterexample() -> (
    None
):
    """Disabling terminal randomization retains the known uniform full-bSWAP example."""
    plan = xeb_circuit(1, 123, "BSWAP", terminal_layer=False)
    assert plan["terminal_layer"] is False
    assert len(plan["local_indices"]) == 1
    assert plan["gates"][-1] == "BSWAP"
    np.testing.assert_allclose(plan["ideal_probabilities"], 0.25, atol=1e-14)


def test_target_frequency_bound_and_cancel_off() -> None:
    """Target frequency bound and cancel off."""
    kw = ingredients()
    kw["target_frequencies_ghz"] = {"D": 4.0, "C": 4.0}
    with pytest.raises(ValueError, match="frequency offset"):
        compile_campaign(["BSWAP"], **kw)
    kw = ingredients()
    kw["recipes"]["bswap"]["cancel_amplitude_ratio"] = 0.0
    sequence, report = compile_campaign(["BSWAP"], **kw)
    np.testing.assert_array_equal(_samples(sequence, "C"), 0.0)
    assert report["terminal_frame_exported"]


def test_preparation_is_padded_before_shorter_drag_and_no_input_mutation() -> None:
    """Preparation is padded before shorter drag and no input mutation."""
    kw = ingredients()
    kw["x90"]["P"] = Rect(duration=20, amplitude=0.12)
    before = deepcopy(kw["recipes"])
    _, report = compile_campaign(["BSWAP"], **kw, prepared=("+", "+"))
    prep = [e for e in report["events"] if e.get("context") == "preparation"]
    assert [e["start_ns"] for e in prep] == [8.0, 4.0]
    assert all(e["start_ns"] + e["duration_ns"] == 24.0 for e in prep)
    assert kw["recipes"] == before


def test_incoming_frame_changes_physical_preparation_and_analysis_axes() -> None:
    """Incoming frame changes physical preparation and analysis axes."""
    kw = ingredients()
    gates = [("XY", 0.3, -0.4), "RAW_SQRT_BSWAP", "BSWAP"]
    schedule, report = compile_campaign(
        gates, **kw, prepared=("+", "+i"), initial_frame=(0.91, -0.33), basis="XY"
    )
    actual = _replay_emitted(schedule, report, kw, {})
    expected = ideal_circuit_unitary(gates) @ (np.kron([1, 1], [1, 1j]) / 2)
    expected = local_xy(1, 0.0) @ local_xy(0, -np.pi / 2) @ expected
    np.testing.assert_allclose(actual, np.abs(expected) ** 2, atol=2e-11)


def test_zero_plateau_still_contains_both_ramps() -> None:
    """Zero plateau still contains both ramps."""
    pulse = waveform(recipe(duration_ns=32.0))
    assert pulse.duration == 32.0
    assert len(pulse.values) == 16
    assert np.isfinite(pulse.values).all()
