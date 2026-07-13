# ruff: noqa: SLF001

"""Tests for stateful QuEL-1 connection manager behavior."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pytest

from qubex.backend.quel1.compat.qubecalib_protocols import (
    BoxPoolProtocol,
    Quel1SystemProtocol,
)
from qubex.backend.quel1.managers.connection_manager import Quel1ConnectionManager
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext


class _FakeDriver:
    pass


class _FakeQubeCalib:
    pass


class _FakeBoxPool:
    def __init__(self) -> None:
        self._boxes: dict[str, tuple[object, object]] = {}
        self._linkstatus: dict[str, bool] = {}
        self._box_config_cache: dict[str, dict[str, Any]] = {}


class _FakeQuel1System:
    boxes: dict[str, object]
    box: dict[str, object]
    _clockmaster: object
    timing_shift: dict[str, int]
    displacement: int
    config_cache: dict[str, dict[str, Any]]
    config_fetched_at: datetime | None

    def __init__(self) -> None:
        self.boxes = {}
        self.box = {}
        self._clockmaster = object()
        self.timing_shift = {}
        self.displacement = 0
        self.config_cache = {}
        self.config_fetched_at = None


def _make_manager() -> Quel1ConnectionManager:
    runtime_context = Quel1RuntimeContext(
        driver=_FakeDriver(),  # type: ignore[arg-type]
        qubecalib=_FakeQubeCalib(),  # type: ignore[arg-type]
        sampling_period_ns=2.0,
    )
    return Quel1ConnectionManager(runtime_context=runtime_context)


def test_connect_stores_connected_runtime_state() -> None:
    """Given connect collaborators, when connect runs, then runtime state is stored in manager."""
    manager = _make_manager()
    boxpool = _FakeBoxPool()
    quel1system = _FakeQuel1System()

    def _resolve_box_names(_box_names: str | list[str] | None) -> list[str]:
        return ["A", "B"]

    def _create_boxpool(box_names: list[str], *, parallel: bool) -> BoxPoolProtocol:
        assert box_names == ["A", "B"]
        assert parallel is True
        return boxpool  # type: ignore[return-value]

    def _create_quel1system_from_boxpool(
        box_names: list[str],
    ) -> Quel1SystemProtocol:
        assert box_names == ["A", "B"]
        assert manager.boxpool is boxpool
        return quel1system  # type: ignore[return-value]

    def _create_resource_map(kind: str) -> dict[str, dict]:
        assert manager.boxpool is boxpool
        assert manager.quel1system is quel1system
        if kind == "cap":
            return {"cap-target": {"kind": "cap"}}
        return {"gen-target": {"kind": "gen"}}

    manager._resolve_box_names = _resolve_box_names  # type: ignore[method-assign]
    manager._create_boxpool = _create_boxpool  # type: ignore[method-assign]
    manager._create_quel1system_from_boxpool = (  # type: ignore[method-assign]
        _create_quel1system_from_boxpool
    )
    manager._create_resource_map = _create_resource_map  # type: ignore[method-assign]

    manager.connect(
        box_names=["A", "B"],
        parallel=True,
    )

    assert manager.is_connected
    assert manager.boxpool is boxpool
    assert manager.quel1system is quel1system
    assert manager.cap_resource_map == {"cap-target": {"kind": "cap"}}
    assert manager.gen_resource_map == {"gen-target": {"kind": "gen"}}


def test_connect_skips_when_already_connected() -> None:
    """Given connected state, when connect is called again, then create pipeline is skipped."""
    manager = _make_manager()
    boxpool = _FakeBoxPool()
    boxpool._boxes["A"] = (object(), object())
    manager.set_connected_state(
        boxpool=boxpool,  # type: ignore[arg-type]
        quel1system=_FakeQuel1System(),  # type: ignore[arg-type]
        cap_resource_map={},
        gen_resource_map={},
    )
    called = False

    def _unexpected_create_boxpool(
        box_names: list[str],
        *,
        parallel: bool,
    ) -> BoxPoolProtocol:
        nonlocal called
        _ = (box_names, parallel)
        called = True
        return _FakeBoxPool()  # type: ignore[return-value]

    manager._create_boxpool = _unexpected_create_boxpool  # type: ignore[method-assign]

    manager.connect(
        box_names=["A"],
        parallel=False,
    )

    assert called is False


def test_connect_reconnects_when_requested_boxes_are_missing() -> None:
    """Given connected subset, when connect requests missing boxes, then it reconnects."""
    manager = _make_manager()
    existing_boxpool = _FakeBoxPool()
    existing_boxpool._boxes["A"] = (object(), object())
    manager.set_connected_state(
        boxpool=existing_boxpool,  # type: ignore[arg-type]
        quel1system=_FakeQuel1System(),  # type: ignore[arg-type]
        cap_resource_map={},
        gen_resource_map={},
    )
    events: list[str] = []
    rebuilt_boxpool = _FakeBoxPool()
    rebuilt_quel1system = _FakeQuel1System()

    def _disconnect() -> None:
        events.append("disconnect")
        manager.clear_connected_state()

    def _create_boxpool(box_names: list[str], *, parallel: bool) -> BoxPoolProtocol:
        events.append(f"create_boxpool:{box_names}:{parallel}")
        return rebuilt_boxpool  # type: ignore[return-value]

    def _create_quel1system_from_boxpool(box_names: list[str]) -> Quel1SystemProtocol:
        _ = box_names
        events.append("create_quel1system")
        return rebuilt_quel1system  # type: ignore[return-value]

    manager.disconnect = _disconnect  # type: ignore[method-assign]
    manager._create_boxpool = _create_boxpool  # type: ignore[method-assign]
    manager._create_quel1system_from_boxpool = (  # type: ignore[method-assign]
        _create_quel1system_from_boxpool
    )
    manager._create_resource_map = lambda _kind: {}  # type: ignore[method-assign]

    manager.connect(
        box_names=["A", "B"],
        parallel=False,
    )

    assert events[0] == "disconnect"
    assert events[1] == "create_boxpool:['A', 'B']:False"
    assert events[2] == "create_quel1system"
    assert manager.boxpool is rebuilt_boxpool


def test_disconnect_clears_connected_runtime_state() -> None:
    """Given connected state, when disconnect runs, then state and resources are cleared."""
    manager = _make_manager()
    disconnected: list[object] = []
    resources: list[object] = ["clockmaster", "box-a"]
    manager.set_connected_state(
        boxpool=_FakeBoxPool(),  # type: ignore[arg-type]
        quel1system=_FakeQuel1System(),  # type: ignore[arg-type]
        cap_resource_map={"cap": {}},
        gen_resource_map={"gen": {}},
    )

    def _collect_held_resources() -> list[object]:
        return resources

    def _disconnect_resource_safely(resource: object) -> None:
        disconnected.append(resource)

    manager._collect_held_resources = _collect_held_resources  # type: ignore[method-assign]
    manager._disconnect_resource_safely = _disconnect_resource_safely  # type: ignore[method-assign]

    manager.disconnect()

    assert disconnected == resources
    assert manager.is_connected is False
    with pytest.raises(ValueError, match="Boxes not connected"):
        _ = manager.boxpool
    with pytest.raises(ValueError, match="Boxes not connected"):
        _ = manager.quel1system
    with pytest.raises(ValueError, match="Boxes not connected"):
        _ = manager.cap_resource_map
    with pytest.raises(ValueError, match="Boxes not connected"):
        _ = manager.gen_resource_map


def test_linkup_reemits_capture_warning_with_box_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given quel_ic warning, when linkup runs, then manager re-emits warning with box id."""

    class _FakeBox:
        def link_status(self) -> dict[int, bool]:
            return {0: True}

        def reconnect(self, **_kwargs: Any) -> None:
            logging.getLogger("quel_ic_config.linkupper").warning(
                "failed establishment of capture link of fddc-#5 of mxfe-#0"
            )
            logging.getLogger("quel_ic_config.quel1_box_intrinsic").error(
                "JESD204C tx-link of AD9082-#0 is not working properly, it must be linked up"
            )

    manager = _make_manager()
    manager._runtime_context.validate_box_availability = (  # type: ignore[method-assign]
        lambda _box_name: None
    )
    boxpool = _FakeBoxPool()
    boxpool._boxes["B0"] = (_FakeBox(), object())
    manager.set_connected_state(
        boxpool=boxpool,  # type: ignore[arg-type]
        quel1system=_FakeQuel1System(),  # type: ignore[arg-type]
        cap_resource_map={},
        gen_resource_map={},
    )

    with caplog.at_level(logging.WARNING):
        manager.linkup(box_name="B0", noise_threshold=10_000)

    assert any(
        "B0 : failed establishment of capture link" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        "B0 : JESD204C tx-link of AD9082-#0 is not working properly"
        in record.getMessage()
        for record in caplog.records
    )
    assert all(
        not (
            record.name.startswith("quel_ic_config")
            and "failed establishment of capture link" in record.getMessage()
        )
        for record in caplog.records
    )
