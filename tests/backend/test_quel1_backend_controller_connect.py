# ruff: noqa: SLF001

"""Tests for connect mode switching in Quel1BackendController."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import pytest

from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


@dataclass
class _ClockmasterSetting:
    ipaddr: str


@dataclass
class _BoxSetting:
    ipaddr_wss: str
    ipaddr_sss: str
    ipaddr_css: str
    boxtype: str


class _FakeBox:
    def __init__(self, name: str) -> None:
        self.name = name
        self.reconnect_count = 0

    def reconnect(self) -> None:
        """Increment reconnect count after a tiny delay."""
        time.sleep(0.005)
        self.reconnect_count += 1


class _FakeBoxPool:
    def __init__(self) -> None:
        self._boxes: dict[str, tuple[_FakeBox, object]] = {}
        self.clockmaster_ip: str | None = None

    def create_clock_master(self, *, ipaddr: str) -> None:
        """Store clockmaster IP."""
        self.clockmaster_ip = ipaddr

    def create(
        self,
        box_name: str,
        *,
        ipaddr_wss: str,
        ipaddr_sss: str,
        ipaddr_css: str,
        boxtype: str,
    ) -> _FakeBox:
        """Create and register a fake box."""
        _ = (ipaddr_wss, ipaddr_sss, ipaddr_css, boxtype)
        box = _FakeBox(box_name)
        self._boxes[box_name] = (box, object())
        return box


class _FakeSysDb:
    def __init__(self) -> None:
        self._clockmaster_setting = _ClockmasterSetting("192.0.2.1")
        self._box_settings = {
            "A": _BoxSetting("10.0.0.1", "10.0.1.1", "10.0.2.1", "type-a"),
            "B": _BoxSetting("10.0.0.2", "10.0.1.2", "10.0.2.2", "type-b"),
        }
        self.create_quel1system_calls: list[tuple[str, ...]] = []

    def create_quel1system(self, *box_names: str) -> str:
        """Record and return a fake system marker."""
        self.create_quel1system_calls.append(tuple(box_names))
        return "legacy-system"


class _FakeQubeCalib:
    def __init__(self) -> None:
        self.sysdb = _FakeSysDb()
        self.system_config_database = self.sysdb
        self.create_boxpool_calls: list[tuple[str, ...]] = []

    def create_boxpool(self, *box_names: str) -> str:
        """Record and return a fake legacy boxpool marker."""
        self.create_boxpool_calls.append(tuple(box_names))
        return "legacy-boxpool"


def _make_controller() -> Quel1BackendController:
    controller = Quel1BackendController()
    cast(Any, controller)._qubecalib = _FakeQubeCalib()
    return controller


def test_connect_uses_legacy_mode_by_default() -> None:
    """Given default mode, connect uses legacy qubecalib APIs."""
    controller = _make_controller()
    controller.create_resource_map = lambda _type: {}  # type: ignore[method-assign]

    controller.connect(["A", "B"])

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.create_boxpool_calls == [("A", "B")]
    assert qubecalib.sysdb.create_quel1system_calls == [("A", "B")]
    assert controller._boxpool == "legacy-boxpool"
    assert controller._quel1system == "legacy-system"


def test_connect_parallel_mode_bypasses_legacy_create_boxpool(monkeypatch) -> None:
    """Given parallel mode, connect bypasses legacy boxpool creation."""
    controller = _make_controller()
    controller.create_resource_map = lambda _type: {}  # type: ignore[method-assign]
    fake_quel1_system = object()

    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_backend_controller.BoxPool",
        _FakeBoxPool,
    )

    def _fake_create_from_boxpool(self: Quel1BackendController, box_names: list[str]):
        assert box_names == ["A", "B"]
        return fake_quel1_system

    monkeypatch.setattr(
        Quel1BackendController,
        "_create_quel1system_from_boxpool",
        _fake_create_from_boxpool,
    )

    controller.connect(["A", "B"], parallel=True)

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.create_boxpool_calls == []
    assert qubecalib.sysdb.create_quel1system_calls == []
    assert isinstance(controller._boxpool, _FakeBoxPool)
    assert controller._quel1system is fake_quel1_system


def test_create_boxpool_parallel_reconnects_all_boxes(monkeypatch) -> None:
    """Given valid boxes, parallel pool creation reconnects each box once."""
    controller = _make_controller()
    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_backend_controller.BoxPool",
        _FakeBoxPool,
    )

    boxpool = cast(_FakeBoxPool, controller._create_boxpool_parallel(["A", "B"]))

    assert isinstance(boxpool, _FakeBoxPool)
    assert boxpool.clockmaster_ip == "192.0.2.1"
    assert set(boxpool._boxes) == {"A", "B"}
    assert boxpool._boxes["A"][0].reconnect_count == 1
    assert boxpool._boxes["B"][0].reconnect_count == 1


def test_create_boxpool_parallel_raises_for_unknown_box(monkeypatch) -> None:
    """Given an unknown box, parallel pool creation raises ValueError."""
    controller = _make_controller()
    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_backend_controller.BoxPool",
        _FakeBoxPool,
    )

    with pytest.raises(ValueError, match=r"box\(Z\) is not defined") as exc:
        controller._create_boxpool_parallel(["A", "Z"])
    assert exc.value.args
    if exc.value.args:
        assert "box(Z) is not defined" in str(exc)
