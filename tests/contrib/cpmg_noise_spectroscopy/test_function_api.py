"""Tests for functional APIs in `qubex.contrib.experiment.cpmg_noise_spectroscopy`."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.typing import ArrayLike

from qubex.contrib import (
    cpmg_noise_spectroscopy,
    plot_cpmg_results,
)
from qubex.contrib.experiment.cpmg_noise_spectroscopy import _make_frequency_plan


def test_all_cpmg_noise_spectroscopy_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then CPMG noise spectroscopy helpers are available."""
    assert callable(cpmg_noise_spectroscopy)
    assert callable(plot_cpmg_results)


def test_public_experiment_excludes_removed_pulse_count_limits() -> None:
    """The public workflow should expose only time-based decay-point planning."""
    parameters = inspect.signature(cpmg_noise_spectroscopy).parameters

    assert "n_pi_range" not in parameters
    assert "max_pi_pulses" not in parameters
    assert "max_sequence_time_ns" not in parameters
    assert parameters["show_t1"].default is False


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
    assert fig.layout.xaxis.rangemode is None
    assert fig.layout.yaxis.rangemode == "tozero"
    assert fig.layout.yaxis2.rangemode == "tozero"
    assert 1.0 < cast(float, fig.layout.legend.x) <= 1.15
    assert cast(int, fig.layout.margin.r) < 240


def test_plot_cpmg_results_does_not_show_t1_by_default() -> None:
    """The summary should not load or display a T1 reference unless requested."""

    class FailingLoader:
        def load_param_data(self, _name: str) -> object:
            raise AssertionError("T1 loading should be disabled by default")

    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            system_manager=SimpleNamespace(config_loader=FailingLoader())
        )
    )
    payload = {
        "timestamp": "20260609_000000",
        "targets": ["Q00"],
        "n_repeats": 1,
        "frequency_range": [0.001],
        "half_tau_range": [250.0],
        "t2_matrix": np.array([[[2000.0]]]),
    }

    figure = plot_cpmg_results(exp, payload, mode="t2", plot=False)  # type: ignore[arg-type]

    shapes = cast(tuple[object, ...], figure.layout.shapes)
    assert len(shapes) == 0


def test_plot_cpmg_results_separates_quality_warning_points() -> None:
    """Quality-flagged fits should be shown separately from reliable points."""
    payload = {
        "timestamp": "20260609_000000",
        "targets": ["Q00"],
        "n_repeats": 1,
        "frequency_range": [0.001, 0.002],
        "half_tau_range": [250.0, 125.0],
        "t2_matrix": np.array([[[20_000.0, 25_000.0]]]),
        "fit_records": [
            {
                "target": "Q00",
                "repeat_index": 0,
                "requested_frequency_index": 0,
                "fit_status": "ok",
                "quality_flags": [],
            },
            {
                "target": "Q00",
                "repeat_index": 0,
                "requested_frequency_index": 1,
                "fit_status": "ok",
                "quality_flags": ["large_relative_error"],
            },
        ],
    }

    figure = plot_cpmg_results(
        SimpleNamespace(),  # type: ignore[arg-type]
        payload,
        mode="t2",
        plot=False,
    )

    traces = cast(tuple[object, ...], figure.data)
    assert len(traces) == 2
    reliable, warning = (cast(go.Scatter, trace) for trace in traces)
    reliable_x = np.asarray(cast(ArrayLike, reliable.x), dtype=np.float64)
    reliable_y = np.asarray(cast(ArrayLike, reliable.y), dtype=np.float64)
    warning_x = np.asarray(cast(ArrayLike, warning.x), dtype=np.float64)
    np.testing.assert_allclose(reliable_x, [1.0, 2.0])
    np.testing.assert_array_equal(np.isfinite(reliable_y), [True, False])
    assert reliable.mode == "markers+lines"
    np.testing.assert_allclose(warning_x, [2.0])
    assert warning.mode == "markers"
    warning_data = cast(dict[str, Any], warning.to_plotly_json())
    assert warning_data["marker"]["symbol"] == "diamond-open"
    assert "quality warning" in cast(str, warning.name)


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
    assert cast(float, fig.layout.legend.x) == pytest.approx(1.02)


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
