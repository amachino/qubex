"""Tests for functional APIs in `qubex.contrib.experiment.ckp_characterization`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go

import qubex.contrib.experiment.ckp_characterization as ckp_characterization
from qubex.contrib import (
    ckp_measurement_v2,
    filtered_ckp_experiment,
)
from qubex.experiment import Experiment
from qubex.experiment.models import Result


def test_all_ckp_characterization_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then CKP characterization helpers are available."""
    assert callable(ckp_measurement_v2)
    assert callable(filtered_ckp_experiment)


def test_filtered_ckp_experiment_returns_all_figures_without_display_or_save(
    monkeypatch,
) -> None:
    """Filtered CKP returns its four QDash figures when display and saving are disabled."""
    heatmap_g = go.Figure()
    heatmap_e = go.Figure()
    optimization_figure = go.Figure()

    def fake_ckp_measurement_v2(*_args: Any, **kwargs: Any) -> Result:
        initial_state = kwargs["qubit_initial_state"]
        is_no_drive = kwargs["resonator_drive_amplitude"] == 0.0
        if is_no_drive:
            resonance_frequencies = [5.0 if initial_state == "0" else 5.1]
            return Result(data={"qubit_resonance_frequencies": resonance_frequencies})

        resonance_frequencies = [4.9, 5.0] if initial_state == "0" else [5.0, 5.1]
        return Result(
            data={
                "valid_resonator_frequencies": [6.0, 6.1],
                "qubit_resonance_frequencies": resonance_frequencies,
                "qubit_resonance_frequency_errors": [0.001, 0.001],
            },
            figure=heatmap_g if initial_state == "0" else heatmap_e,
        )

    initial_guess = SimpleNamespace(
        C=0.01,
        omega_r=6.05,
        omega_p=6.1,
        J=0.02,
        kappa=0.03,
    )
    fit = SimpleNamespace(
        success=True,
        message="success",
        omega_r_g=6.0,
        omega_r_e=6.1,
        omega_p=6.2,
        J=0.02,
        kappa=0.03,
        C=0.01,
        chi=0.1,
        A2=0.04,
        A=0.2,
        omega_r_g_error=0.001,
        omega_r_e_error=0.001,
        omega_p_error=0.001,
        J_error=0.001,
        kappa_error=0.001,
        C_error=0.001,
        chi_error=0.001,
        A2_error=0.001,
        reduced_chi2=1.0,
        r2=0.99,
        covariance=np.eye(6),
    )

    def fake_readout_optimization(**kwargs: Any) -> Result:
        assert kwargs["plot"] is False
        assert kwargs["save_image"] is False
        return Result(
            data={
                "optimal_readout_frequency": 6.05,
                "n_limited_A2_at_optimal_frequency": 0.02,
            },
            figure=optimization_figure,
        )

    monkeypatch.setattr(
        ckp_characterization,
        "ckp_measurement_v2",
        fake_ckp_measurement_v2,
    )
    monkeypatch.setattr(
        ckp_characterization,
        "estimate_filtered_ckp_initial_params",
        lambda **_kwargs: initial_guess,
    )
    monkeypatch.setattr(
        ckp_characterization,
        "fit_filtered_ckp_two_traces",
        lambda **_kwargs: fit,
    )
    monkeypatch.setattr(
        ckp_characterization,
        "filtered_ckp_model",
        lambda x, **_kwargs: np.zeros_like(x),
    )
    monkeypatch.setattr(
        ckp_characterization,
        "filtered_ckp_photon_number",
        lambda x, **_kwargs: np.ones_like(x),
    )
    monkeypatch.setattr(
        ckp_characterization,
        "estimate_optimal_readout_frequency_from_ckp",
        fake_readout_optimization,
    )

    result = filtered_ckp_experiment(
        exp=cast(Experiment, object()),
        target="Q00",
        qubit_detuning_range=[-0.01, 0.01],
        resonator_detuning_range=[-0.01, 0.01],
        resonator_drive_amplitude=0.2,
        n_shots=4,
        plot=False,
        save_image=False,
        enable_rough_search=False,
    )

    figures = result.figures
    primary_figure = result.figure
    assert figures is not None
    assert primary_figure is not None
    assert primary_figure is figures["ckp_fit"]
    assert len(primary_figure.to_plotly_json()["data"]) == 4
    assert figures == {
        "ckp_fit": primary_figure,
        "ckp_heatmap_g": heatmap_g,
        "ckp_heatmap_e": heatmap_e,
        "readout_optimization": optimization_figure,
    }
