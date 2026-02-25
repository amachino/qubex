# ruff: noqa: SLF001

"""Tests for QuEL-1 runtime context initialization behavior."""

from __future__ import annotations

from typing import Any

import pytest

from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext


class _FakeQubeCalib:
    pass


class _FakeDriver:
    DEFAULT_SAMPLING_PERIOD = 0.8

    def __init__(self) -> None:
        self._qubecalib = _FakeQubeCalib()

    def QubeCalib(self) -> _FakeQubeCalib:
        """Return fake qubecalib instance."""
        return self._qubecalib


class _FailingQubeCalibDriver:
    DEFAULT_SAMPLING_PERIOD = 1.2

    def QubeCalib(self) -> Any:
        """Raise a dependency failure for qubecalib construction."""
        raise RuntimeError("failed to import qubecalib")


def test_runtime_context_constructor_loads_driver_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given default constructor, runtime context loads driver and qubecalib defaults."""
    driver = _FakeDriver()
    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_runtime_context.load_quel1_driver",
        lambda: driver,
    )

    context = Quel1RuntimeContext()

    assert context.driver is driver
    assert context.qubecalib is driver._qubecalib
    assert context.sampling_period == 0.8


def test_runtime_context_constructor_handles_qubecalib_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given failing qubecalib construction, qubecalib property raises ModuleNotFoundError."""
    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_runtime_context.load_quel1_driver",
        lambda: _FailingQubeCalibDriver(),
    )

    context = Quel1RuntimeContext()

    with pytest.raises(ModuleNotFoundError) as exc:
        _ = context.qubecalib
    assert exc.value.name == "qubecalib"
