"""Tests for functional APIs in `qubex.contrib.experiment.cpmg_noise_spectroscopy`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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


def test_cpmg_frequency_plan_keeps_zero_reference_first() -> None:
    """Given f=0, when planning frequencies, then zero reference stays first."""
    plan = _make_frequency_plan(
        np.array([0.01, 0.0, 0.02]),
        sampling_period=2.0,
    )

    assert plan[0].frequency == 0.0
    assert plan[0].half_tau == 0.0


def test_cpmg_frequency_plan_rejects_negative_frequency() -> None:
    """Given a negative frequency, when planning, then validation fails."""
    with pytest.raises(ValueError, match="non-negative"):
        _make_frequency_plan(np.array([-0.01]), sampling_period=2.0)


def test_plot_cpmg_results_accepts_in_memory_payload() -> None:
    """Given an in-memory CPMG payload, when plotting, then a summary figure is returned."""
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

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [0.0, 10.0]
