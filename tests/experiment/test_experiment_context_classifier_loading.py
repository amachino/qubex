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


def _make_context(tmp_path: Path) -> ExperimentContext:
    """Create a minimal context instance for classifier-loading tests."""
    context = object.__new__(ExperimentContext)
    context.__dict__["_qubits"] = ["Q00", "Q01"]
    context.__dict__["_classifier_dir"] = tmp_path
    context.__dict__["_chip_id"] = "test-chip"
    context.__dict__["_measurement"] = _MeasurementStub()

    classifier_path = tmp_path / "test-chip"
    classifier_path.mkdir(parents=True)
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
    context.load_classifier()

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
    loaded_classifier = object()

    def _load(path: Path | str) -> object:
        if Path(path).name == "Q01.pkl":
            raise ValueError("broken classifier payload")
        return loaded_classifier

    monkeypatch.setattr(StateClassifier, "load", staticmethod(_load))

    with pytest.raises(ValueError, match="broken classifier payload"):
        context.load_classifier()

    assert context.classifiers == {}


def test_load_classifier_uses_custom_classifier_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a custom classifier directory, when loading, then classifiers are read from that directory."""
    default_dir = tmp_path / "default"
    custom_dir = tmp_path / "custom" / "test-chip"
    context = _make_context(default_dir)
    custom_dir.mkdir(parents=True)
    for qubit in context.qubit_labels:
        (custom_dir / f"{qubit}.pkl").write_bytes(b"classifier")

    loaded_paths: list[Path] = []

    def _load(path: Path | str) -> object:
        loaded_paths.append(Path(path))
        return object()

    monkeypatch.setattr(StateClassifier, "load", staticmethod(_load))

    context.load_classifier(path=custom_dir)

    assert loaded_paths == [
        custom_dir / "Q00.pkl",
        custom_dir / "Q01.pkl",
    ]


def test_load_classifier_rejects_missing_directory(tmp_path: Path) -> None:
    """Given a missing classifier directory, when loading, then a file-not-found error is raised."""
    context = _make_context(tmp_path / "default")
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=str(missing_dir)):
        context.load_classifier(path=missing_dir)


def test_load_classifier_rejects_directory_without_classifier_files(
    tmp_path: Path,
) -> None:
    """Given an empty classifier directory, when loading, then a file-not-found error is raised."""
    context = _make_context(tmp_path / "default")
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No classifier files"):
        context.load_classifier(path=empty_dir)
