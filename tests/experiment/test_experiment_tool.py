"""Tests for experiment_tool helpers."""

from __future__ import annotations

from dataclasses import dataclass

from qubex.experiment import experiment_tool
from qubex.system.control_system import PortType


@dataclass(frozen=True)
class FakePort:
    """Port stub for box info tests."""

    number: int
    type: PortType


@dataclass(frozen=True)
class FakeBox:
    """Box stub for box info tests."""

    id: str
    ports: tuple[FakePort, ...]


class FakeExperimentSystem:
    """Experiment-system stub with box lookup."""

    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes
        self._boxes = {box.id: box for box in boxes}

    def get_box(self, box_id: str) -> FakeBox:
        """Return a fake box."""
        return self._boxes[box_id]


class FakeBackendController:
    """Backend-controller stub with dump recording."""

    def __init__(self) -> None:
        self.dumped_box_ids: list[str] = []

    def dump_box(self, box_id: str) -> dict:
        """Return a fake dump for the requested box."""
        self.dumped_box_ids.append(box_id)
        return {
            "ports": {
                1: {
                    "direction": "out",
                    "sideband": "L",
                    "lo_freq": 10_000_000_000,
                    "cnco_freq": 1_500,
                    "vatt": 2_048,
                    "fullscale_current": 40_527,
                    "channels": {0: {"fnco_freq": 100}},
                }
            }
        }


class FakeQuel1Box:
    """Quel1Box stub recording reconnect calls."""

    def __init__(self) -> None:
        self.background_noise_thresholds: list[int] = []

    def reconnect(self, *, background_noise_threshold: int) -> None:
        """Record reconnect threshold."""
        self.background_noise_thresholds.append(background_noise_threshold)


class FakeBackendControllerWithGetBox(FakeBackendController):
    """Backend-controller stub with box lookup."""

    def __init__(self, boxes: dict[str, FakeQuel1Box]) -> None:
        super().__init__()
        self._boxes = boxes

    def get_box(self, box_id: str) -> FakeQuel1Box:
        """Return a fake Quel1 box."""
        return self._boxes[box_id]


@dataclass
class FakeSystemManager:
    """System-manager stub for experiment_tool tests."""

    experiment_system: FakeExperimentSystem
    backend_controller: FakeBackendController


def test_print_box_info_fetch_uses_dump_box(monkeypatch) -> None:
    """Given fetch mode, when printing box info, then dump_box is used."""
    fake_manager = FakeSystemManager(
        experiment_system=FakeExperimentSystem(
            [FakeBox(id="A", ports=(FakePort(number=1, type=PortType.CTRL),))]
        ),
        backend_controller=FakeBackendController(),
    )
    monkeypatch.setattr(experiment_tool, "system_manager", fake_manager)
    monkeypatch.setattr(experiment_tool.console, "print", lambda *_: None)

    experiment_tool.print_box_info("A", fetch=True)

    assert fake_manager.backend_controller.dumped_box_ids == ["A"]


def test_get_quel1_box_reconnects_box_with_default_threshold(monkeypatch) -> None:
    """Given a backend box, when requesting Quel1 box, then reconnect is called once."""
    box = FakeQuel1Box()
    fake_manager = FakeSystemManager(
        experiment_system=FakeExperimentSystem([]),
        backend_controller=FakeBackendControllerWithGetBox({"U15A": box}),
    )
    monkeypatch.setattr(experiment_tool, "system_manager", fake_manager)

    returned_box = experiment_tool.get_quel1_box("U15A")

    assert returned_box is box
    assert box.background_noise_thresholds == [10000]
