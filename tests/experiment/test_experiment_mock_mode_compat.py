"""Tests for Experiment mock-mode compatibility contract."""

from __future__ import annotations

from typing import Any, cast

from qubex.experiment import experiment as experiment_module
from qubex.experiment.experiment import Experiment


def test_experiment_init_forwards_mock_mode_to_context(monkeypatch) -> None:
    """Given mock_mode, when creating Experiment, then ExperimentContext receives that value."""
    called: dict[str, object] = {}
    service_kwargs: list[dict[str, object]] = []

    class _ExperimentContext:
        def __init__(self, **kwargs: object) -> None:
            called["context_kwargs"] = kwargs

    class _Service:
        def __init__(self, **kwargs: object) -> None:
            service_kwargs.append(dict(kwargs))

    monkeypatch.setattr(experiment_module, "ExperimentContext", _ExperimentContext)
    monkeypatch.setattr(experiment_module, "PulseService", _Service)
    monkeypatch.setattr(experiment_module, "MeasurementService", _Service)
    monkeypatch.setattr(experiment_module, "CalibrationService", _Service)
    monkeypatch.setattr(experiment_module, "CharacterizationService", _Service)
    monkeypatch.setattr(experiment_module, "BenchmarkingService", _Service)
    monkeypatch.setattr(experiment_module, "OptimizationService", _Service)
    monkeypatch.setattr(experiment_module, "SessionService", _Service)

    Experiment(
        chip_id="TEST",
        qubits=["Q00"],
        mock_mode=True,
        configuration_mode="ge-cr-cr",
    )

    context_kwargs = called["context_kwargs"]
    assert isinstance(context_kwargs, dict)
    assert context_kwargs["mock_mode"] is True
    assert any("experiment_context" in entry for entry in service_kwargs)


def test_experiment_init_creates_session_service_with_context(monkeypatch) -> None:
    """Given experiment init, when services are created, then session service receives experiment context."""
    called: dict[str, object] = {}

    class _ExperimentContext:
        def __init__(self, **kwargs: object) -> None:
            called["context"] = self
            called["context_kwargs"] = kwargs

    class _SessionService:
        def __init__(self, **_: object) -> None:
            called["session_kwargs"] = _

    class _Service:
        def __init__(self, **_: object) -> None:
            return None

    monkeypatch.setattr(experiment_module, "ExperimentContext", _ExperimentContext)
    monkeypatch.setattr(experiment_module, "SessionService", _SessionService)
    monkeypatch.setattr(experiment_module, "PulseService", _Service)
    monkeypatch.setattr(experiment_module, "MeasurementService", _Service)
    monkeypatch.setattr(experiment_module, "CalibrationService", _Service)
    monkeypatch.setattr(experiment_module, "CharacterizationService", _Service)
    monkeypatch.setattr(experiment_module, "BenchmarkingService", _Service)
    monkeypatch.setattr(experiment_module, "OptimizationService", _Service)

    Experiment(
        chip_id="TEST",
        qubits=["Q00"],
        mock_mode=True,
        configuration_mode="ge-cr-cr",
    )

    assert called["session_kwargs"] == {"experiment_context": called["context"]}


def test_experiment_init_forwards_backend_controller_to_context(
    monkeypatch,
) -> None:
    """Given a custom backend controller, ExperimentContext should receive it."""
    called: dict[str, object] = {}

    class _ExperimentContext:
        def __init__(self, **kwargs: object) -> None:
            called["context_kwargs"] = kwargs

    class _Service:
        def __init__(self, **_: object) -> None:
            return None

    backend_controller = object()

    monkeypatch.setattr(experiment_module, "ExperimentContext", _ExperimentContext)
    monkeypatch.setattr(experiment_module, "SessionService", _Service)
    monkeypatch.setattr(experiment_module, "PulseService", _Service)
    monkeypatch.setattr(experiment_module, "MeasurementService", _Service)
    monkeypatch.setattr(experiment_module, "CalibrationService", _Service)
    monkeypatch.setattr(experiment_module, "CharacterizationService", _Service)
    monkeypatch.setattr(experiment_module, "BenchmarkingService", _Service)
    monkeypatch.setattr(experiment_module, "OptimizationService", _Service)

    Experiment(
        chip_id="TEST",
        qubits=["Q00"],
        backend_controller=cast(Any, backend_controller),
    )

    context_kwargs = called["context_kwargs"]
    assert isinstance(context_kwargs, dict)
    assert context_kwargs["backend_controller"] is backend_controller
