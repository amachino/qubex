"""Tests for sampling-period resolution in experiment utilities."""

from __future__ import annotations

import numpy as np

from qubex.experiment.experiment_util import (
    DEFAULT_BACKEND_SAMPLING_PERIOD_NS,
    ExperimentUtil,
)


def test_discretize_time_range_uses_backend_sampling_period_by_default(
    monkeypatch,
) -> None:
    """Given backend dt, when discretizing without dt, then utility uses backend sampling period."""
    backend_controller = type("_BackendController", (), {"sampling_period_ns": 0.4})()
    system_manager = type(
        "_SystemManager",
        (),
        {"backend_controller": backend_controller},
    )()
    monkeypatch.setattr(
        "qubex.experiment.experiment_util.SystemManager.shared",
        lambda: system_manager,
    )

    discretized = ExperimentUtil.discretize_time_range(
        np.array([0.21, 0.61], dtype=float),
    )

    assert np.allclose(discretized, np.array([0.4, 0.8]))


def test_resolve_sampling_period_uses_default_before_backend_selection(
    monkeypatch,
) -> None:
    """Given no selected backend, when resolving sampling period, then the default is returned."""

    class _SystemManager:
        @property
        def backend_controller(self) -> object:
            raise RuntimeError("Backend controller is not initialized.")

    monkeypatch.setattr(
        "qubex.experiment.experiment_util.SystemManager.shared",
        _SystemManager,
    )

    sampling_period = ExperimentUtil.resolve_sampling_period()

    assert sampling_period == float(DEFAULT_BACKEND_SAMPLING_PERIOD_NS)
