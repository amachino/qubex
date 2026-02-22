# ruff: noqa: SLF001

"""Tests for connect mode switching in Quel1BackendController."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
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
        self._linkstatus: dict[str, bool] = {}
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
        self.create_box_calls: list[tuple[str, bool]] = []

    def create_box(self, box_name: str, reconnect: bool = True) -> _FakeBox:
        """Create a fake box and record call arguments."""
        self.create_box_calls.append((box_name, reconnect))
        return _FakeBox(box_name)

    def create_quel1system(self, *box_names: str) -> str:
        """Record and return a fake system marker."""
        self.create_quel1system_calls.append(tuple(box_names))
        return "legacy-system"


class _FakeQubeCalib:
    def __init__(self) -> None:
        self.sysdb = _FakeSysDb()
        self.system_config_database = self.sysdb


def _make_controller() -> Quel1BackendController:
    controller = Quel1BackendController()
    cast(Any, controller)._qubecalib = _FakeQubeCalib()
    return controller


def _override_driver_classes(
    controller: Quel1BackendController, **overrides: Any
) -> None:
    """Replace selected driver classes in one controller instance."""
    cast(Any, controller)._driver = replace(cast(Any, controller)._driver, **overrides)


def test_connect_uses_boxpool_by_default(monkeypatch) -> None:
    """Given default mode, connect builds a boxpool and system."""
    controller = _make_controller()
    controller.create_resource_map = lambda _type: {}  # type: ignore[method-assign]
    fake_quel1_system = object()

    _override_driver_classes(controller, BoxPool=_FakeBoxPool)

    def _fake_create_from_boxpool(self: Quel1BackendController, box_names: list[str]):
        assert box_names == ["A", "B"]
        return fake_quel1_system

    monkeypatch.setattr(
        Quel1BackendController,
        "_create_quel1system_from_boxpool",
        _fake_create_from_boxpool,
    )

    controller.connect(["A", "B"])

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.sysdb.create_quel1system_calls == []
    assert isinstance(controller._connection_manager.boxpool, _FakeBoxPool)
    assert controller._connection_manager.quel1system is fake_quel1_system


def test_connect_parallel_mode_bypasses_legacy_create_boxpool(monkeypatch) -> None:
    """Given parallel mode, connect builds a boxpool and system."""
    controller = _make_controller()
    controller.create_resource_map = lambda _type: {}  # type: ignore[method-assign]
    fake_quel1_system = object()

    _override_driver_classes(controller, BoxPool=_FakeBoxPool)

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
    assert qubecalib.sysdb.create_quel1system_calls == []
    assert isinstance(controller._connection_manager.boxpool, _FakeBoxPool)
    assert controller._connection_manager.quel1system is fake_quel1_system


def test_create_boxpool_reconnects_all_boxes(monkeypatch) -> None:
    """Given valid boxes, pool creation reconnects each box once."""
    controller = _make_controller()
    _override_driver_classes(
        controller,
        BoxPool=_FakeBoxPool,
        SequencerClient=lambda _ipaddr: object(),
    )

    boxpool = cast(_FakeBoxPool, controller._create_boxpool(["A", "B"]))

    assert isinstance(boxpool, _FakeBoxPool)
    assert boxpool.clockmaster_ip == "192.0.2.1"
    assert set(boxpool._boxes) == {"A", "B"}
    assert boxpool._boxes["A"][0].reconnect_count == 1
    assert boxpool._boxes["B"][0].reconnect_count == 1


def test_create_boxpool_raises_for_unknown_box(monkeypatch) -> None:
    """Given an unknown box, pool creation raises ValueError."""
    controller = _make_controller()
    _override_driver_classes(controller, BoxPool=_FakeBoxPool)

    with pytest.raises(ValueError, match=r"box\(Z\) is not defined") as exc:
        controller._create_boxpool(["A", "Z"])
    assert exc.value.args
    if exc.value.args:
        assert "box(Z) is not defined" in str(exc)


def test_get_box_returns_existing_box_without_reconnect(monkeypatch) -> None:
    """Given pooled box, get_box returns it without reconnecting."""
    controller = _make_controller()
    _override_driver_classes(
        controller,
        BoxPool=_FakeBoxPool,
        SequencerClient=lambda _ipaddr: object(),
    )
    boxpool = cast(_FakeBoxPool, controller._create_boxpool(["A"]))
    controller._connection_manager.set_boxpool(cast(Any, boxpool))
    monkeypatch.setattr(controller, "_check_box_availability", lambda _: None)

    reconnect_count_before = boxpool._boxes["A"][0].reconnect_count
    box = controller.get_box("A")

    assert box is boxpool._boxes["A"][0]
    assert boxpool._boxes["A"][0].reconnect_count == reconnect_count_before


def test_connect_skips_reconnect_when_already_connected(monkeypatch) -> None:
    """Given existing connection, connect skips creating a new boxpool."""
    controller = _make_controller()
    existing_system = object()
    controller._connection_manager.set_quel1system(cast(Any, existing_system))

    def _raise_if_called(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("connect path should be skipped when already connected")

    monkeypatch.setattr(controller, "_create_boxpool", _raise_if_called)
    monkeypatch.setattr(
        controller, "_create_quel1system_from_boxpool", _raise_if_called
    )
    monkeypatch.setattr(controller, "create_resource_map", _raise_if_called)

    controller.connect(["A"])

    assert controller._connection_manager.quel1system is existing_system
