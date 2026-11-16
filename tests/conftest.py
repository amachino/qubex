"""Shared fixtures for isolated backend boundary tests."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType

import pytest


@pytest.fixture
def stub_classification_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the driver line-set data type without the optional driver package."""

    @dataclass(frozen=True)
    class ClassificationLineSet:
        line0: tuple[float, float, float]
        line1: tuple[float, float, float]

    module = ModuleType("qxdriver_quel1.classification")
    module.__dict__["ClassificationLineSet"] = ClassificationLineSet
    monkeypatch.setitem(sys.modules, "qxdriver_quel1.classification", module)
