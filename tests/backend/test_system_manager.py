"""Tests for SystemManager backend settings collection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qubex.backend.control_system import PortType
from qubex.backend.system_manager import SystemManager


@dataclass(frozen=True)
class FakePort:
    """Port stub for backend settings tests."""

    number: int
    type: PortType


@dataclass(frozen=True)
class FakeBox:
    """Box stub for backend settings tests."""

    id: str
    ports: tuple[FakePort, ...]


class FakeBoxPool:
    """Minimal box pool cache holder."""

    def __init__(self) -> None:
        self._box_config_cache: dict[str, dict] = {}


class FakeBackendController:
    """Backend controller stub for backend settings tests."""

    def __init__(self, configs: dict[str, dict]) -> None:
        self._configs = configs
        self.boxpool = FakeBoxPool()

    def dump_box(self, box_id: str) -> dict:
        """Return a predefined box configuration."""
        return self._configs.get(box_id, {})


class FakeExperimentSystem:
    """Experiment system stub with box lookup."""

    def __init__(self, boxes: list[FakeBox]) -> None:
        self._boxes = {box.id: box for box in boxes}

    def get_box(self, box_id: str) -> FakeBox:
        """Return a fake box."""
        return self._boxes[box_id]


@pytest.mark.parametrize("parallel", [True, False])
def test_fetch_backend_settings_collects_ports(
    monkeypatch: pytest.MonkeyPatch,
    parallel: bool,
) -> None:
    """Given boxes, when fetching settings, then cache and ports are filled."""
    # Arrange
    box_a = FakeBox(
        id="A",
        ports=(
            FakePort(number=1, type=PortType.CTRL),
            FakePort(number=2, type=PortType.MNTR_OUT),
            FakePort(number=3, type=PortType.READ_IN),
        ),
    )
    box_b = FakeBox(
        id="B",
        ports=(FakePort(number=4, type=PortType.READ_OUT),),
    )
    configs = {
        "A": {"ports": {1: {"mode": "ctrl"}, 3: {"mode": "read"}}},
        "B": {"ports": {4: {"mode": "read"}}},
    }
    manager = SystemManager.shared()
    monkeypatch.setattr(manager, "_backend_controller", FakeBackendController(configs))
    monkeypatch.setattr(
        manager, "_experiment_system", FakeExperimentSystem([box_a, box_b])
    )

    # Act
    result = manager._fetch_backend_settings(  # noqa: SLF001
        ["A", "B"],
        parallel=parallel,
    )

    # Assert
    assert result == {
        "A": {"ports": {1: {"mode": "ctrl"}, 3: {"mode": "read"}}},
        "B": {"ports": {4: {"mode": "read"}}},
    }
    assert manager.backend_controller.boxpool._box_config_cache == configs  # noqa: SLF001
