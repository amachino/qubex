"""Tests for classifier loading behavior in experiment context."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qubex.experiment.experiment_context import ExperimentContext
from qubex.measurement import StateClassifier


class _MeasurementStub:
    def __init__(self) -> None:
        self.classifiers: dict[str, object] = {}

    def update_classifiers(self, classifiers: dict[str, object]) -> None:
        self.classifiers.update(classifiers)


class _TestExperimentContext(ExperimentContext):
    def load_classifiers(self) -> None:
        self._load_classifiers()


def _make_context(tmp_path: Path) -> _TestExperimentContext:
    """Create a minimal context instance for classifier-loading tests."""
    context = object.__new__(_TestExperimentContext)
    context.__dict__["_qubits"] = ["Q00", "Q01"]
    context.__dict__["_classifier_dir"] = tmp_path
    context.__dict__["_chip_id"] = "test-chip"
    context.__dict__["_measurement"] = _MeasurementStub()

    classifier_path = tmp_path / "test-chip"
    classifier_path.mkdir()
    for qubit in context.qubit_labels:
        (classifier_path / f"{qubit}.pkl").write_bytes(b"classifier")

    return context


def test_load_classifiers_warns_and_skips_compatibility_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given incompatible classifier pickle, when loading, then a warning is emitted and the classifier is skipped."""
    context = _make_context(tmp_path)
    loaded_classifier = object()

    def _load(path: Path | str) -> object:
        if Path(path).name == "Q00.pkl":
            raise ModuleNotFoundError("No module named 'legacy_classifier_module'")
        return loaded_classifier

    monkeypatch.setattr(StateClassifier, "load", staticmethod(_load))

    caplog.set_level(logging.WARNING, logger="qubex.experiment.experiment_context")
    context.load_classifiers()

    assert "Failed to load state classifier for Q00" in caplog.text
    assert "compatibility issue" in caplog.text
    assert "The classifier was skipped." in caplog.text
    assert "Q00.pkl" in caplog.text
    assert "legacy_classifier_module" in caplog.text

    assert context.classifiers == {"Q01": loaded_classifier}


def test_load_classifiers_propagates_non_compatibility_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given unrelated classifier load failure, when loading, then the original error is raised."""
    context = _make_context(tmp_path)

    def _load(_path: Path | str) -> object:
        raise ValueError("broken classifier payload")

    monkeypatch.setattr(StateClassifier, "load", staticmethod(_load))

    with pytest.raises(ValueError, match="broken classifier payload"):
        context.load_classifiers()
