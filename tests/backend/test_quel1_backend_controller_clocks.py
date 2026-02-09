# ruff: noqa: SLF001

"""Tests for clock interactions in Quel1BackendController."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import pytest

from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


@dataclass
class _ClockmasterSetting:
    ipaddr: str


@dataclass
class _BoxSetting:
    ipaddr_sss: str


class _FakeSysDb:
    def __init__(self, *, include_clockmaster: bool = True) -> None:
        self._box_settings = {
            "A": _BoxSetting("10.0.1.1"),
            "B": _BoxSetting("10.0.1.2"),
        }
        self._clockmaster_setting = (
            _ClockmasterSetting("192.0.2.1") if include_clockmaster else None
        )

    def asdict(self) -> dict[str, Any]:
        """Return a minimal system config payload."""
        return {
            "box_settings": {
                name: {"ipaddr_sss": setting.ipaddr_sss}
                for name, setting in self._box_settings.items()
            }
        }


class _FakeQubeCalib:
    def __init__(self, *, include_clockmaster: bool = True) -> None:
        self.sysdb = _FakeSysDb(include_clockmaster=include_clockmaster)
        self.system_config_database = self.sysdb


class _FakeSequencerClient:
    calls: ClassVar[list[str]] = []
    readings: ClassVar[dict[str, tuple[bool, int, int]]] = {}

    def __init__(self, *, target_ipaddr: str) -> None:
        self._target_ipaddr = target_ipaddr

    def read_clock(self) -> tuple[bool, int, int]:
        """Return the preconfigured reading for the target."""
        self.calls.append(self._target_ipaddr)
        return self.readings[self._target_ipaddr]


class _FakeQuBEMasterClient:
    def __init__(self, master_ipaddr: str) -> None:
        self.master_ipaddr = master_ipaddr
        self.kick_calls: list[list[str]] = []

    def kick_clock_synch(self, target_addrs: list[str]) -> bool:
        """Record target addresses and report success."""
        self.kick_calls.append(list(target_addrs))
        return True


def _make_controller(*, include_clockmaster: bool = True) -> Quel1BackendController:
    controller = Quel1BackendController()
    cast(Any, controller)._qubecalib = _FakeQubeCalib(
        include_clockmaster=include_clockmaster
    )
    return controller


def test_read_clocks_uses_sequencer_clients(monkeypatch) -> None:
    """Given boxes, read_clocks returns sequencer client readings."""
    controller = _make_controller()
    _FakeSequencerClient.calls = []
    _FakeSequencerClient.readings = {
        "10.0.1.1": (True, 123, 456),
        "10.0.1.2": (True, 789, 1011),
    }
    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_backend_controller.SequencerClient",
        _FakeSequencerClient,
    )

    result = controller.read_clocks(["A", "B"])

    assert result == [(True, 123, 456), (True, 789, 1011)]
    assert _FakeSequencerClient.calls == ["10.0.1.1", "10.0.1.2"]


def test_resync_clocks_kicks_clockmaster(monkeypatch) -> None:
    """Given multiple boxes, resync_clocks kicks clockmaster and checks."""
    controller = _make_controller()
    _FakeSequencerClient.readings = {
        "10.0.1.1": (True, 123_456_789_000, 123_456_789_999),
        "10.0.1.2": (True, 123_456_789_000, 123_456_789_999),
    }
    master = _FakeQuBEMasterClient("192.0.2.1")

    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_backend_controller.SequencerClient",
        _FakeSequencerClient,
    )
    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_backend_controller.QuBEMasterClient",
        lambda master_ipaddr: master,
    )

    assert controller.resync_clocks(["A", "B"]) is True
    assert master.master_ipaddr == "192.0.2.1"
    assert master.kick_calls == [["10.0.1.1", "10.0.1.2"]]


def test_resync_clocks_raises_without_clockmaster(monkeypatch) -> None:
    """Given no clockmaster, resync_clocks raises ValueError."""
    controller = _make_controller(include_clockmaster=False)
    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_backend_controller.QuBEMasterClient",
        _FakeQuBEMasterClient,
    )

    with pytest.raises(ValueError, match="clock master is not found"):
        controller.resync_clocks(["A", "B"])
