"""Tests for measurement-default resolution in experiment context."""

from __future__ import annotations

from types import SimpleNamespace

from qubex.experiment import experiment_context as experiment_context_module
from qubex.experiment.experiment_context import ExperimentContext


class _SystemManagerStub:
    def __init__(self, *, measurement_defaults: dict[str, object]) -> None:
        self.config_loader = SimpleNamespace(chip_id="TESTCHIP")
        self.experiment_system = SimpleNamespace(
            measurement_defaults=measurement_defaults
        )

    def load(self, **_: object) -> None:
        return None


class _MeasurementStub:
    def __init__(self, **_: object) -> None:
        return None


class _NoteStub:
    def __init__(self, **_: object) -> None:
        return None


def test_init_uses_measurement_defaults_for_readout_timing(monkeypatch) -> None:
    """Given measurement defaults yaml, when ExperimentContext is created, then readout timing uses configured values."""
    system_manager = _SystemManagerStub(
        measurement_defaults={
            "readout": {
                "duration_ns": 512.0,
                "pre_margin_ns": 16.0,
                "post_margin_ns": 96.0,
            }
        }
    )
    monkeypatch.setattr(
        experiment_context_module.SystemManager,
        "shared",
        staticmethod(lambda: system_manager),
    )
    monkeypatch.setattr(experiment_context_module, "Measurement", _MeasurementStub)
    monkeypatch.setattr(experiment_context_module, "ExperimentNote", _NoteStub)
    monkeypatch.setattr(experiment_context_module, "CalibrationNote", _NoteStub)
    monkeypatch.setattr(ExperimentContext, "_load_skew_file", lambda self: None)
    monkeypatch.setattr(ExperimentContext, "_load_classifiers", lambda self: None)
    monkeypatch.setattr(
        ExperimentContext,
        "print_environment",
        lambda self, verbose=False: None,
    )

    context = ExperimentContext(system_id="SYS-A")

    assert context.readout_duration == 512.0
    assert context.readout_pre_margin == 16.0
    assert context.readout_post_margin == 96.0
