"""Tests for Experiment mock-mode compatibility contract."""

from __future__ import annotations

from qubex.experiment import experiment as experiment_module
from qubex.experiment.experiment import Experiment


def test_experiment_init_forwards_mock_mode_to_context(monkeypatch) -> None:
    """Given mock_mode, when creating Experiment, then ExperimentContext receives that value."""
    called: dict[str, object] = {}

    class _ExperimentContext:
        def __init__(self, **kwargs: object) -> None:
            called["context_kwargs"] = kwargs

    class _Service:
        def __init__(self, **_: object) -> None:
            pass

    monkeypatch.setattr(experiment_module, "ExperimentContext", _ExperimentContext)
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

    context_kwargs = called["context_kwargs"]
    assert isinstance(context_kwargs, dict)
    assert context_kwargs["mock_mode"] is True
