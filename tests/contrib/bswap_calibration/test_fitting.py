"""Synthetic regression tests for offline, fixed-target calibration fits."""

import numpy as np
import pytest

from qubex.contrib.experiment.bswap_calibration.chevron_fit import (
    chevron_model,
    fit_bidirectional_chevron,
)
from qubex.contrib.experiment.bswap_calibration.duration_fit import (
    damped_transfer,
    fit_damped_duration,
)
from qubex.contrib.experiment.bswap_calibration.irb_analysis import decay
from qubex.contrib.experiment.bswap_calibration.local_fit import fit_local_map


def test_chevron_fit_recovers_shared_resonance_and_preserves_shapes():
    """A synthetic joint chevron should recover its carrier and rate."""
    frequencies = 4.612 + np.linspace(-0.005, 0.005, 21)
    durations = np.r_[0.0, np.arange(32.0, 1233.0, 40.0)]
    parameters = [0.2, 2.1, 16.0, 1.5, 0.02, 0.04, 0.8, 0.75, 5000.0, 4000.0]
    probabilities = chevron_model(parameters, (frequencies - 4.612) * 1000, durations)
    counts = np.rint(probabilities * 20000)
    result = fit_bidirectional_chevron(frequencies, durations, counts, 20000)
    assert result["resonance_frequency_ghz"] == pytest.approx(4.6122, abs=1e-5)
    assert result["parameters"]["rate_mhz"] == pytest.approx(2.1, abs=0.01)
    assert result["residual"].shape == (2, 21, 32)
    np.testing.assert_allclose(result["observed"], counts / 20000, atol=1e-12)


def test_duration_fit_uses_ramp_phase_for_both_native_grid_candidates():
    """Full and root durations should restore both ramps after phase fitting."""
    flat = np.arange(0.0, 1001.0, 10.0)
    parameters = [2.0, 0.2, 0.03, 0.04, 0.85, 0.80, 0.1, 0.2]
    counts = np.rint(damped_transfer(parameters, flat) * 20000)
    result = fit_damped_duration(flat + 32, counts, 20000, ramp_ns=16.0)
    for name, angle in (("bswap", np.pi), ("sqrt_bswap", np.pi / 2)):
        candidate = result[name]
        expected = 32 + (angle - 0.2) * 1000 / (4 * np.pi)
        assert candidate["available"]
        assert candidate["duration_ns"] == pytest.approx(expected, abs=0.2)
        assert candidate["grid_duration_ns"] % 2 == 0
        assert candidate["flat_duration_ns"] + 32 == pytest.approx(
            candidate["duration_ns"], abs=1e-10
        )
    assert (
        abs(result["sqrt_bswap"]["duration_ns"] - result["bswap"]["duration_ns"] / 2)
        > 1
    )


def test_local_fit_recovers_an_interior_amplitude_frequency_candidate():
    """A local quadratic should retain both direction axes and find its interior peak."""
    amplitudes = np.linspace(0.94, 0.99, 5)
    frequencies = np.linspace(4.6116, 4.6124, 5)
    aa, ff = np.meshgrid(amplitudes, frequencies, indexing="ij")
    probability = 0.8 - 60 * (aa - 0.975) ** 2 - 5e5 * (ff - 4.6121) ** 2
    result = fit_local_map(amplitudes, frequencies, np.stack([probability] * 2), 20000)
    assert result["amplitude"] == pytest.approx(0.975, abs=0.002)
    assert result["frequency_ghz"] == pytest.approx(4.6121, abs=2e-5)
    assert not result["boundary"]
    np.testing.assert_allclose(result["residual"], 0.0, atol=1e-10)


def test_decay_accepts_array_like_depths():
    """The decay model should accept list coordinates without changing values."""
    np.testing.assert_allclose(
        decay([0, 1, 2], 0.7, 0.9, 0.25), [0.95, 0.88, 0.817], atol=1e-12
    )
