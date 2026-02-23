"""Tests for session-lifecycle ownership boundaries of ExperimentContext."""

from __future__ import annotations

from qubex.experiment.experiment_context import ExperimentContext


def test_context_does_not_expose_session_lifecycle_methods() -> None:
    """Given context class, when checking session APIs, then session lifecycle methods are not exposed."""
    assert not hasattr(ExperimentContext, "session_service")
    assert not hasattr(ExperimentContext, "is_connected")
    assert not hasattr(ExperimentContext, "disconnect")
    assert not hasattr(ExperimentContext, "check_status")
    assert not hasattr(ExperimentContext, "connect")
    assert not hasattr(ExperimentContext, "linkup")
    assert not hasattr(ExperimentContext, "resync_clocks")
    assert not hasattr(ExperimentContext, "configure")
    assert not hasattr(ExperimentContext, "reload")
