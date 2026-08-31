"""Offline matrix, emitted-phase and estimator tests; no hardware measurements."""

import numpy as np
import pytest
from qxpulse import Blank, Rect

from qubex.contrib.experiment.bswap_calibration.irb import (
    NativeBSWAPCache,
    circuit_unitary,
    compile_bswap,
    x90_matrix,
)
from qubex.contrib.experiment.bswap_calibration.irb_analysis import (
    analyze_irb,
    decay,
    fit_decay,
)
from qubex.contrib.experiment.bswap_calibration.tomography import (
    local_z,
    raw_unitary,
    state_vector,
)


def test_native_table_and_paired_circuits(native_bswap_cache_file):
    """A portable native cache should close matched reference and interleaved circuits."""
    cache = NativeBSWAPCache(native_bswap_cache_file)
    for depth in (0, 1, 2, 4, 16, 64):
        for seed in range(10):
            reference = cache.circuit(depth, seed, False)
            interleaved = cache.circuit(depth, seed, True)
            assert reference["random_indices"] == interleaved["random_indices"]
            for circuit in (reference, interleaved):
                assert abs(circuit["ideal_closure_error"]) < 1e-10
                assert set(circuit["gates"]) <= {
                    "XI90",
                    "IX90",
                    "ZI90",
                    "IZ90",
                    "BSWAP",
                }


def test_emitted_phase_transport_including_preparation_and_analysis():
    """Compiled phases should transport preparation and analysis frames consistently."""
    rng = np.random.default_rng(123)
    qubits = ["A", "P"]
    x90 = {q: Rect(duration=16, amplitude=0.1) for q in qubits}
    xpi = {q: Rect(duration=24, amplitude=0.1) for q in qubits}
    gate = Rect(duration=100, amplitude=0.8)
    names = ["XI90", "IX90", "ZI90", "IZ90", "BSWAP"]
    for _ in range(24):
        post = rng.uniform(-np.pi, np.pi, 2)
        zeta = 0.37
        rate = float(rng.uniform(-0.5, 0.5))
        delay = float(rng.integers(0, 1000) * 2)
        gates = list(rng.choice(names, size=12))
        sequence, _ = compile_bswap(
            gates,
            qubits=qubits,
            drive_label="D",
            sizzle_label="S",
            gate=gate,
            x90=x90,
            xpi=xpi,
            post_vz=-post,
            placement_rate=rate,
            reference_start=24,
            delay_ns=delay,
            prepared=("+", "+"),
            basis="XY",
        )
        events = []
        for label in sequence.labels:
            time = 0.0
            for pulse in sequence.get_sequence(label).get_flattened_waveforms(True):
                if not isinstance(pulse, Blank):
                    events.append((time, label, pulse))
                time += pulse.duration
        psi = np.array([1, 0, 0, 0], dtype=complex)
        for time, label, pulse in sorted(events, key=lambda e: (e[0], e[1])):
            if label == "D":
                phase = pulse.phase - rate * (time - 24)
                op = (
                    local_z([phase, phase])
                    @ raw_unitary(
                        "bswap",
                        post_active=float(post[0]),
                        post_passive=float(post[1]),
                        zeta=zeta,
                    )
                    @ local_z([-phase, -phase])
                )
            else:
                op = x90_matrix(0 if label == "A" else 1, pulse.phase)
            psi = op @ psi
        expected = (
            x90_matrix(1, 0)
            @ x90_matrix(0, -np.pi / 2)
            @ circuit_unitary(gates, zeta)
            @ state_vector(("+", "+"))
        )
        np.testing.assert_allclose(abs(psi) ** 2, abs(expected) ** 2, atol=1e-10)


def test_irb_estimator_and_invalid_reference():
    """IRB should estimate a resolved decay and reject a collapsed reference."""
    depths = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    rng = np.random.default_rng(54)
    reference = decay(depths, 0.70, 0.98, 0.25)[None, :] + rng.normal(
        0, 0.002, (24, len(depths))
    )
    interleaved = decay(depths, 0.70, 0.97, 0.25)[None, :] + rng.normal(
        0, 0.002, (24, len(depths))
    )
    result = analyze_irb(depths, reference, interleaved, bootstrap=32)
    assert result["quote_as_irb_estimate"], result["reasons"]
    assert abs(result["fidelity_estimate"] - (0.25 + 0.75 * 0.97 / 0.98)) < 0.005
    collapsed = np.full_like(reference, 0.25)
    rejected = analyze_irb(depths, collapsed, collapsed, bootstrap=16)
    assert not rejected["quote_as_irb_estimate"]
    assert rejected["fidelity_estimate"] is None


def _synthetic_irb_data():
    """Return complete raw-probability arrays with well-identified decays."""
    depths = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    reference = np.repeat(decay(depths, 0.70, 0.98, 0.25)[None, :], 24, axis=0)
    interleaved = np.repeat(decay(depths, 0.70, 0.97, 0.25)[None, :], 24, axis=0)
    return depths, reference, interleaved


@pytest.mark.parametrize("bootstrap", [0, -1, 1.5, True])
def test_irb_requires_positive_integer_bootstrap(bootstrap):
    """Reject bootstrap settings that cannot produce the requested resampling."""
    with pytest.raises(ValueError, match="positive integer"):
        analyze_irb(*_synthetic_irb_data(), bootstrap=bootstrap)


def test_irb_rejects_unstable_bootstrap_fits():
    """Do not count non-identifiable resample fits as bootstrap successes."""
    depths, _, _ = _synthetic_irb_data()
    rates = [0.98, 0.98, 0.996, 0.996]
    reference = np.array([decay(depths, 0.70, p, 0.25) for p in rates])
    interleaved = np.array([decay(depths, 0.70, p - 0.01, 0.25) for p in rates])
    result = analyze_irb(depths, reference, interleaved, bootstrap=32, seed=42)
    assert all(fit["fit_quality_pass"] for fit in result["fits"].values())
    assert result["bootstrap_successes"] < 0.9 * result["bootstrap_requested"]
    assert "bootstrap_instability" in result["reasons"]
    assert not result["quote_as_irb_estimate"]
    assert result["fidelity_estimate"] is None


def test_irb_requires_usable_bootstrap_interval():
    """A single accepted draw cannot supply a usable percentile interval."""
    result = analyze_irb(*_synthetic_irb_data(), bootstrap=1)
    assert result["bootstrap_successes"] == 1
    assert result["statistical_interval_95"] is None
    assert "bootstrap_interval_unavailable" in result["reasons"]
    assert not result["quote_as_irb_estimate"]


@pytest.mark.parametrize("bad_probability", [-0.01, 1.01, np.nan, np.inf])
@pytest.mark.parametrize("mode", [0, 1])
def test_irb_rejects_malformed_raw_depth_zero_controls(bad_probability, mode):
    """Validate raw controls even though depth zero is excluded from fitting."""
    depths, reference, interleaved = _synthetic_irb_data()
    (reference, interleaved)[mode][:, 0] = bad_probability
    with pytest.raises(ValueError, match="probabilities"):
        analyze_irb(depths, reference, interleaved, bootstrap=8)
    with pytest.raises(ValueError, match="probabilities"):
        fit_decay(depths, (reference, interleaved)[mode])


@pytest.mark.parametrize("bad_depth", [-1, 0.5, np.nan, np.inf, 2])
def test_irb_rejects_malformed_depths(bad_depth):
    """Reject invalid or duplicated depths rather than silently omitting them."""
    depths, reference, interleaved = _synthetic_irb_data()
    depths = depths.astype(float)
    depths[0] = bad_depth
    with pytest.raises(ValueError, match="Depths"):
        analyze_irb(depths, reference, interleaved, bootstrap=8)
    with pytest.raises(ValueError, match="Depths"):
        fit_decay(depths, reference)
