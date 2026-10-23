"""Tests for functional APIs in `qubex.contrib.experiment.cpmg_noise_spectroscopy`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest

from qubex.contrib import (
    cpmg_noise_spectroscopy,
    plot_cpmg_results,
)
from qubex.contrib.experiment.cpmg_noise_spectroscopy import _make_frequency_plan


def test_all_cpmg_noise_spectroscopy_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then CPMG noise spectroscopy helpers are available."""
    assert callable(cpmg_noise_spectroscopy)
    assert callable(plot_cpmg_results)


def test_cpmg_frequency_plan_rejects_zero_frequency() -> None:
    """Given f=0, when planning frequencies, then validation fails."""
    with pytest.raises(ValueError, match="positive"):
        _make_frequency_plan(
            np.array([0.01, 0.0, 0.02]),
            sampling_period=2.0,
            pi_duration=20.0,
            max_pulse_duty_ratio=0.25,
        )


def test_cpmg_frequency_plan_rejects_negative_frequency() -> None:
    """Given a negative frequency, when planning, then validation fails."""
    with pytest.raises(ValueError, match="positive"):
        _make_frequency_plan(
            np.array([-0.01]),
            sampling_period=2.0,
            pi_duration=20.0,
            max_pulse_duty_ratio=0.25,
        )


def test_plot_cpmg_results_defaults_to_both_summary_modes() -> None:
    """Given an in-memory payload, the default summary should show T2 and Gamma2."""
    exp = SimpleNamespace()
    payload = {
        "timestamp": "20260609_000000",
        "targets": ["Q00"],
        "n_repeats": 2,
        "frequency_range": [0.0, 0.01],
        "half_tau_range": [0.0, 25.0],
        "time_range": [100.0, 200.0],
        "t2_matrix": np.array([[[1000.0, 2000.0], [1200.0, 2200.0]]]),
        "failed_fits": [],
        "save_dir": None,
    }

    fig = plot_cpmg_results(
        exp,  # type: ignore[arg-type]
        payload,
        show_t1=False,
        plot=False,
        save_image=False,
    )

    assert isinstance(fig, go.Figure)
    figure_data = cast(list[dict[str, Any]], fig.to_dict()["data"])
    assert len(figure_data) == 2
    np.testing.assert_allclose(figure_data[0]["x"], [0.0, 10.0])
    np.testing.assert_array_equal(figure_data[0]["customdata"], [2, 2])
    assert figure_data[0].get("yaxis", "y") == "y"
    assert figure_data[1]["yaxis"] == "y2"
    assert fig.layout.yaxis2.overlaying == "y"
    assert fig.layout.xaxis.rangemode == "tozero"
    assert fig.layout.yaxis.rangemode == "tozero"
    assert fig.layout.yaxis2.rangemode == "tozero"
    assert cast(float, fig.layout.legend.x) > 1.0
    assert cast(int, fig.layout.margin.r) >= 200


@pytest.mark.parametrize("mode", ["t2", "gamma2"])
def test_plot_cpmg_results_keeps_single_summary_modes(mode: str) -> None:
    """Given an existing summary mode, plotting should still return one trace."""
    payload = {
        "timestamp": "20260609_000000",
        "targets": ["Q00"],
        "n_repeats": 1,
        "frequency_range": [0.001],
        "half_tau_range": [250.0],
        "time_range": [1000.0],
        "t2_matrix": np.array([[[2000.0]]]),
        "failed_fits": [],
        "save_dir": None,
    }

    fig = plot_cpmg_results(
        SimpleNamespace(),  # type: ignore[arg-type]
        payload,
        mode=mode,  # type: ignore[arg-type]
        show_t1=False,
        plot=False,
    )

    traces = cast(tuple[object, ...], fig.data)
    assert len(traces) == 1


def test_plot_cpmg_results_rejects_malformed_error_matrix() -> None:
    """An in-memory error matrix with the wrong shape should fail clearly."""
    payload = {
        "timestamp": "20260609_000000",
        "targets": ["Q00"],
        "n_repeats": 1,
        "frequency_range": [0.001],
        "half_tau_range": [250.0],
        "t2_matrix": np.array([[[2000.0]]]),
        "t2_error_matrix": np.ones((2, 2)),
    }

    with pytest.raises(ValueError, match="t2_error_matrix has shape"):
        plot_cpmg_results(
            SimpleNamespace(),  # type: ignore[arg-type]
            payload,
            show_t1=False,
            plot=False,
        )
