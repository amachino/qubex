"""Tests for the adaptive final chevron measurement."""

from unittest.mock import Mock

import numpy as np
import pytest
from plotly.graph_objects import Figure

from qubex.contrib.experiment import chevron_matched_transform as chevron
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result


def test_adaptive_uses_standard_chevron_for_final_measurement(monkeypatch) -> None:
    """Adaptive calibration should reuse standard chevron fits and measured data."""
    times = np.array([0.0, 10.0, 20.0])
    detunings = np.array([-0.01, 0.01])
    values = np.arange(6).reshape(3, 2)
    figure = Figure()
    final = Result(
        data={
            "time_range": times,
            "detuning_range": detunings,
            "frequencies": {"Q1": 5.01},
            "chevron_data": {"Q1": values},
            "rabi_rates": {"Q1": np.array([0.02, 0.02])},
            "rabi_fit_r2": {"Q1": np.array([0.9, 0.95])},
            "resonant_frequencies": {"Q1": 5.012},
        },
        figures={"Q1": figure},
    )
    exp = Mock(spec=Experiment)
    exp.chevron_pattern.return_value = final
    search = Mock(
        return_value=(
            Result(
                data={
                    "detuning_range": detunings,
                    "time_range": times,
                    "chevron_data": values,
                }
            ),
            Result(
                data={
                    "omega_q": 5.01,
                    "omega_rabi": 0.025,
                    "peak_background_rms_ratio": 20.0,
                    "transform": values,
                }
            ),
        )
    )
    analyze = Mock(
        return_value=Result(
            data={
                "omega_q": 5.011,
                "omega_rabi": 0.0125,
                "peak_background_rms_ratio": 15.0,
                "transform": values,
            }
        )
    )
    monkeypatch.setattr(chevron, "_measure_and_analyze_chevron", search)
    monkeypatch.setattr(chevron, "analyze_chevron_matched_transform", analyze)

    result = chevron.estimate_qubit_frequency_from_chevron_adaptive(
        exp,
        "Q1",
        frequencies={"Q1": 5.0},
        amplitudes={"Q1": 0.4},
        final_time_range=times,
        final_detuning_range=detunings,
        n_shots=128,
        shot_interval=1000.0,
        plot=False,
        save_image=False,
    )

    assert search.call_count == 1
    kwargs = exp.chevron_pattern.call_args.kwargs
    assert kwargs["targets"] == ["Q1"]
    assert kwargs["frequencies"] == {"Q1": 5.01}
    assert kwargs["amplitudes"]["Q1"] == pytest.approx(0.2)
    assert kwargs["n_shots"] == 128
    assert kwargs["shot_interval"] == 1000.0
    assert kwargs["plot"] is False
    assert kwargs["save_image"] is False
    np.testing.assert_array_equal(kwargs["time_range"], times)
    np.testing.assert_array_equal(kwargs["detuning_range"], detunings)
    assert analyze.call_args.args[0] is final
    assert result.data["resonant_frequencies"]["Q1"] == pytest.approx(5.012)
    assert result.data["results"]["Q1"]["omega_q"] == pytest.approx(5.012)
    assert set(result.data) == {
        "results",
        "time_range",
        "detuning_range",
        "resonant_frequencies",
        "target_amplitudes",
        "peak_background_rms_ratios",
        "search_results",
    }
    assert set(result.data["results"]["Q1"]) == {
        "omega_q",
        "omega_rabi",
        "peak_background_rms_ratio",
        "frequency_used",
        "amplitude_used",
        "chevron_data",
        "transform",
    }
    np.testing.assert_array_equal(result.data["results"]["Q1"]["chevron_data"], values)
    assert result.data["target_amplitudes"]["Q1"] == pytest.approx(0.2)
    assert result.figures is not None
    assert result.figures["Q1_measurement"] is figure
