"""Tests for skew-file warning behavior in experiment context."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qubex.experiment.experiment_context import ExperimentContext


class _BackendControllerStub:
    pass


class _ExperimentSystemStub:
    def __init__(self, *, box_count: int) -> None:
        self.control_system = type(
            "_ControlSystem",
            (),
            {"boxes": [object() for _ in range(box_count)]},
        )()


class _ConfigLoaderStub:
    def __init__(self, *, config_path: Path, backend_kind: str) -> None:
        self.config_path = config_path
        self.backend_kind = backend_kind


class _TestExperimentContext(ExperimentContext):
    def __init__(
        self,
        *,
        config_path: Path,
        backend_kind: str,
        box_count: int,
    ) -> None:
        self._config_loader_stub = _ConfigLoaderStub(
            config_path=config_path,
            backend_kind=backend_kind,
        )
        self._experiment_system_stub = _ExperimentSystemStub(box_count=box_count)
        self._backend_controller_stub = _BackendControllerStub()

    @property
    def config_loader(self) -> _ConfigLoaderStub:
        return self._config_loader_stub

    @property
    def experiment_system(self) -> _ExperimentSystemStub:
        return self._experiment_system_stub

    @property
    def control_system(self) -> object:
        return self._experiment_system_stub.control_system

    @property
    def backend_controller(self) -> _BackendControllerStub:
        return self._backend_controller_stub

    def load_skew_file(self) -> None:
        self._load_skew_file()


def test_load_skew_file_warns_for_missing_multi_box_quel1(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given missing skew file on multi-box QuEL-1, when context loads it, then a warning is logged."""
    context = _TestExperimentContext(
        config_path=tmp_path,
        backend_kind="quel1",
        box_count=2,
    )

    caplog.set_level(logging.WARNING, logger="qubex.experiment.experiment_context")

    context.load_skew_file()

    assert "Skew file not found" in caplog.text


def test_load_skew_file_skips_warning_for_missing_single_box_quel1(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given missing skew file on single-box QuEL-1, when context loads it, then no warning is logged."""
    context = _TestExperimentContext(
        config_path=tmp_path,
        backend_kind="quel1",
        box_count=1,
    )

    caplog.set_level(logging.WARNING, logger="qubex.experiment.experiment_context")

    context.load_skew_file()

    assert "Skew file not found" not in caplog.text


def test_load_skew_file_skips_warning_for_missing_quel3(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given missing skew file on QuEL-3, when context loads it, then no warning is logged."""
    context = _TestExperimentContext(
        config_path=tmp_path,
        backend_kind="quel3",
        box_count=2,
    )

    caplog.set_level(logging.WARNING, logger="qubex.experiment.experiment_context")

    context.load_skew_file()

    assert "Skew file not found" not in caplog.text
