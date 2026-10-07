"""Tests for functional APIs in `qubex.contrib.experiment.ckp_characterization`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest

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
        assert kwargs["readout_amplitude"] == 0.4
        initial_state = kwargs["qubit_initial_state"]
        is_no_drive = kwargs["resonator_drive_amplitude"] == 0.0
        if is_no_drive:
            resonance_frequencies = [5.0 if initial_state == "0" else 5.1]
            return Result(data={"qubit_resonance_frequencies": resonance_frequencies})

        assert kwargs["resonator_drive_amplitude"] == 0.2
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

    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            resolve_qubit_label=lambda target: target,
            resolve_read_label=lambda target: f"R{target[1:]}",
            targets={
                "Q00": SimpleNamespace(frequency=5.0),
                "R00": SimpleNamespace(frequency=6.0),
            },
            params=SimpleNamespace(get_readout_amplitude=lambda _target: 0.4),
        )
    )

    result = filtered_ckp_experiment(
        exp=cast(Experiment, exp),
        target="Q00",
        qubit_detuning_range=[-0.01, 0.01],
        resonator_detuning_range=[-0.01, 0.01],
        readout_amplitude=0.4,
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
    assert result.data["qubit_frequency"] == 5.0
    assert result.data["readout_frequency"] == 6.0
    assert result.data["readout_amplitude"] == 0.4


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("qubit_frequency", 0.0),
        ("readout_frequency", float("inf")),
        ("readout_amplitude", float("nan")),
    ],
)
def test_filtered_ckp_rejects_nonpositive_or_nonfinite_overrides(
    name: str,
    value: float,
) -> None:
    """Filtered CKP rejects invalid calibration overrides before measurement."""
    with pytest.raises(ValueError, match=f"{name} must be positive and finite"):
        filtered_ckp_experiment(
            exp=cast(Experiment, object()),
            target="Q00",
            **cast(Any, {name: value}),
        )


def test_filtered_ckp_scopes_frequency_overrides_and_restores_them(
    monkeypatch,
) -> None:
    """Filtered CKP applies explicit sweep centers only for the experiment call."""
    active_frequencies: dict[str, float] = {}

    @contextmanager
    def modified_frequencies(frequencies: dict[str, float]) -> Iterator[None]:
        active_frequencies.update(frequencies)
        try:
            yield
        finally:
            active_frequencies.clear()

    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            resolve_qubit_label=lambda target: target,
            resolve_read_label=lambda target: f"R{target[1:]}",
        ),
        modified_frequencies=modified_frequencies,
    )
    result = Result(data={"completed": True})

    def fake_run(exp: Experiment, **kwargs: Any) -> Result:
        assert exp is not None
        assert active_frequencies == {"Q00": 5.1, "R00": 6.2}
        assert kwargs["readout_amplitude"] == 0.3
        return result

    monkeypatch.setattr(ckp_characterization, "_run_filtered_ckp_experiment", fake_run)

    actual = filtered_ckp_experiment(
        exp=cast(Experiment, exp),
        target="Q00",
        qubit_frequency=5.1,
        readout_frequency=6.2,
        readout_amplitude=0.3,
    )

    assert actual is result
    assert active_frequencies == {}


def test_filtered_ckp_restores_frequency_overrides_after_failure(
    monkeypatch,
) -> None:
    """A failing CKP run must not leak a temporary frequency override."""
    active_frequencies: dict[str, float] = {}

    @contextmanager
    def modified_frequencies(frequencies: dict[str, float]) -> Iterator[None]:
        active_frequencies.update(frequencies)
        try:
            yield
        finally:
            active_frequencies.clear()

    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            resolve_qubit_label=lambda target: target,
            resolve_read_label=lambda target: f"R{target[1:]}",
        ),
        modified_frequencies=modified_frequencies,
    )

    def fail_run(_exp: Experiment, **_kwargs: Any) -> Result:
        assert active_frequencies == {"Q00": 5.1}
        raise RuntimeError("CKP failed")

    monkeypatch.setattr(ckp_characterization, "_run_filtered_ckp_experiment", fail_run)

    with pytest.raises(RuntimeError, match="CKP failed"):
        filtered_ckp_experiment(
            exp=cast(Experiment, exp),
            target="Q00",
            qubit_frequency=5.1,
        )

    assert active_frequencies == {}
