"""Tests for `qubex.contrib.experiment.quantum_efficiency_measurement` analysis helpers."""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest

from qubex.contrib.experiment.quantum_efficiency_measurement import (
    _fit_ramsey_fringe,
    analyze_quantum_efficiency,
    compute_readout_snr,
    measurement_induced_dephasing,
    measurement_induced_dephasing_experiment,
    quantum_efficiency_measurement,
    readout_snr,
    sweep_readout_snr,
)
from qubex.experiment import Experiment


def _build_analysis_inputs() -> dict[str, Any]:
    """Build one synthetic dataset with known SNR and dephasing parameters."""
    amplitudes = np.linspace(0.01, 0.20, 10)
    phases = np.linspace(0.0, 4.0 * np.pi, 41)
    expected_sigma_m = 0.05
    expected_slope = 10.0
    b = 0.45
    probability_rows = []
    ground_raw = []
    excited_raw = []

    for amplitude in amplitudes:
        rho01 = b * np.exp(-(amplitude**2) / (2.0 * expected_sigma_m**2))
        sigma_z = 2.0 * rho01 * np.cos(phases + 0.1)
        probability_rows.append(0.5 * (1.0 - sigma_z))

        projected_ground = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        projected_excited = projected_ground + expected_slope * amplitude
        ground_raw.append(projected_ground.astype(np.complex128))
        excited_raw.append(projected_excited.astype(np.complex128))

    return {
        "amplitudes": amplitudes,
        "phases": phases,
        "ramsey_probabilities": np.asarray(probability_rows, dtype=np.float64),
        "ground_raw": ground_raw,
        "excited_raw": excited_raw,
        "expected_sigma_m": expected_sigma_m,
        "expected_slope": expected_slope,
    }


def test_analyze_quantum_efficiency_should_recover_fit_based_eta() -> None:
    """Given synthetic amplitude data, when analyzed, then the fit-based QE matches the known model."""
    dataset = _build_analysis_inputs()
    result = analyze_quantum_efficiency(
        exp=object(),
        target="Q00",
        readout_amplitudes=dataset["amplitudes"],
        ramsey_phases=dataset["phases"],
        ramsey_excited_probabilities=dataset["ramsey_probabilities"],
        ground_state_raw=dataset["ground_raw"],
        excited_state_raw=dataset["excited_raw"],
        plot=False,
    )

    expected_eta = (
        0.5 * dataset["expected_slope"] ** 2 * dataset["expected_sigma_m"] ** 2
    )
    np.testing.assert_allclose(result["quantum_efficiency"], expected_eta, rtol=5e-2)
    assert result["snr_fit"]["a"] == pytest.approx(dataset["expected_slope"], rel=5e-2)
    assert result["measurement_induced_dephasing_fit"]["sigma_m"] == pytest.approx(
        dataset["expected_sigma_m"], rel=5e-2
    )
    assert isinstance(result.get_figure(), go.Figure)
    assert isinstance(result.get_figure("snr"), go.Figure)
    assert isinstance(result.get_figure("dephasing"), go.Figure)


def test_compute_readout_snr_should_support_waveform_raw_iq() -> None:
    """Given waveform IQ shots, when computing SNR, then the projected SNR is positive and stable."""
    ground = np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j],
            [0.1 + 0.0j, 1.1 + 0.0j],
            [-0.1 + 0.0j, 0.9 + 0.0j],
        ],
        dtype=np.complex128,
    )
    excited = np.array(
        [
            [1.0 + 0.0j, 2.0 + 0.0j],
            [1.1 + 0.0j, 2.1 + 0.0j],
            [0.9 + 0.0j, 1.9 + 0.0j],
        ],
        dtype=np.complex128,
    )

    summary = compute_readout_snr(ground, excited)

    assert summary["snr"] > 0
    assert np.asarray(summary["projected_ground"]).shape == (3,)
    assert np.asarray(summary["projected_excited"]).shape == (3,)


def test_analyze_quantum_efficiency_should_reject_shape_mismatch() -> None:
    """Given mismatched Ramsey shapes, when analyzing, then a clear ValueError is raised."""
    dataset = _build_analysis_inputs()

    with pytest.raises(ValueError, match="ramsey_excited_probabilities"):
        analyze_quantum_efficiency(
            exp=object(),
            target="Q00",
            readout_amplitudes=dataset["amplitudes"],
            ramsey_phases=dataset["phases"],
            ramsey_excited_probabilities=dataset["ramsey_probabilities"][:-1],
            ground_state_raw=dataset["ground_raw"],
            excited_state_raw=dataset["excited_raw"],
            plot=False,
        )


def test_analyze_quantum_efficiency_should_reject_invalid_raw_shape() -> None:
    """Given invalid raw IQ arrays, when analyzing, then a clear ValueError is raised."""
    dataset = _build_analysis_inputs()
    invalid_ground = [np.zeros((2, 2, 2), dtype=np.float64)] * len(
        dataset["amplitudes"]
    )

    with pytest.raises(ValueError, match="ground_state_raw"):
        analyze_quantum_efficiency(
            exp=object(),
            target="Q00",
            readout_amplitudes=dataset["amplitudes"],
            ramsey_phases=dataset["phases"],
            ramsey_excited_probabilities=dataset["ramsey_probabilities"],
            ground_state_raw=invalid_ground,
            excited_state_raw=dataset["excited_raw"],
            plot=False,
        )


def test_public_experiment_helpers_should_run_with_stubbed_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a stub experiment, when calling the public helpers, then each returns a Result with figures."""
    dataset = _build_analysis_inputs()
    amplitudes = dataset["amplitudes"]
    phases = dataset["phases"]
    probability_map = {
        float(amplitude): np.asarray(
            dataset["ramsey_probabilities"][index], dtype=np.float64
        )
        for index, amplitude in enumerate(amplitudes)
    }
    probability_map[0.0] = 0.5 * (1.0 - 2.0 * 0.45 * np.cos(phases + 0.1))
    raw_map = {
        float(amplitude): (
            np.asarray(dataset["ground_raw"][index], dtype=np.complex128),
            np.asarray(dataset["excited_raw"][index], dtype=np.complex128),
        )
        for index, amplitude in enumerate(amplitudes)
    }

    def _fake_measure_ramsey_fringe(
        _exp: Experiment,
        _target: str,
        *,
        readout_amplitude: float,
        phase_range: np.ndarray,
        n_shots: int | None,
        shot_interval: float | None,
    ) -> dict[str, object]:
        del n_shots, shot_interval
        phase_index = np.array(
            [int(np.argmin(np.abs(phases - phase))) for phase in phase_range],
            dtype=np.int64,
        )
        probabilities = probability_map[float(readout_amplitude)][phase_index]
        return dict(_fit_ramsey_fringe(phase_range, probabilities))

    def _fake_measure_readout_snr(
        _exp: Experiment,
        _target: str,
        *,
        readout_amplitude: float,
        n_shots: int,
        shot_interval: float | None,
        readout_duration: float | None,
    ) -> Any:
        del n_shots, shot_interval, readout_duration
        ground_raw, excited_raw = raw_map[float(readout_amplitude)]
        return compute_readout_snr(ground_raw, excited_raw)

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(
        qe_module, "_measure_ramsey_fringe", _fake_measure_ramsey_fringe
    )
    monkeypatch.setattr(qe_module, "_measure_readout_snr", _fake_measure_readout_snr)

    experiment = cast(Experiment, object())
    single_dephasing_result = measurement_induced_dephasing(
        experiment,
        "Q00",
        readout_amplitude=float(amplitudes[3]),
        use_reference=True,
        phase_range=phases,
        plot=False,
    )
    dephasing_result = measurement_induced_dephasing_experiment(
        experiment,
        "Q00",
        amplitude_range=amplitudes,
        phase_range=phases,
        plot=False,
    )
    single_snr_result = readout_snr(
        experiment,
        "Q00",
        readout_amplitude=float(amplitudes[3]),
        n_shots=1024,
        plot=False,
    )
    snr_result = sweep_readout_snr(
        experiment,
        "Q00",
        amplitude_range=amplitudes,
        n_shots=1024,
        plot=False,
    )
    qe_result = quantum_efficiency_measurement(
        experiment,
        "Q00",
        amplitude_range=amplitudes,
        phase_range=phases,
        n_shots=1024,
        plot=False,
    )

    assert single_dephasing_result["measurement_induced_dephasing"] >= 0
    assert isinstance(single_dephasing_result.get_figure(), go.Figure)
    assert isinstance(dephasing_result.get_figure(), go.Figure)
    assert single_snr_result["snr"] > 0
    assert isinstance(single_snr_result.get_figure(), go.Figure)
    assert isinstance(snr_result.get_figure(), go.Figure)
    assert isinstance(qe_result.get_figure(), go.Figure)
    assert qe_result["quantum_efficiency"] > 0
