"""Tests for stateful QuEL-1 connection manager behavior."""

from __future__ import annotations

from typing import Any

from qubex.backend.quel1.managers.connection_manager import Quel1ConnectionManager


def test_connect_stores_connected_runtime_state() -> None:
    """Given connect callbacks, when connect runs, then runtime state is stored in manager."""
    manager = Quel1ConnectionManager()
    boxpool = object()
    quel1system = object()

    def _create_quel1system_from_boxpool(box_names: list[str]) -> object:
        assert box_names == ["A", "B"]
        assert manager.boxpool is boxpool
        return quel1system

    def _create_resource_map(kind: str) -> dict[str, dict]:
        assert manager.boxpool is boxpool
        assert manager.quel1system is quel1system
        if kind == "cap":
            return {"cap-target": {"kind": "cap"}}
        return {"gen-target": {"kind": "gen"}}

    manager.connect(
        box_names=["A", "B"],
        available_boxes=lambda: ("A", "B"),
        parallel=None,
        default_parallel_mode=True,
        create_boxpool=lambda names, parallel: boxpool,
        create_quel1system_from_boxpool=_create_quel1system_from_boxpool,
        create_resource_map=_create_resource_map,
    )

    assert manager.is_connected
    assert manager.boxpool is boxpool
    assert manager.quel1system is quel1system
    assert manager.cap_resource_map == {"cap-target": {"kind": "cap"}}
    assert manager.gen_resource_map == {"gen-target": {"kind": "gen"}}


def test_connect_skips_when_already_connected() -> None:
    """Given connected state, when connect is called again, then no callback is executed."""
    manager = Quel1ConnectionManager()
    manager.set_connected_state(
        boxpool=object(),
        quel1system=object(),
        cap_resource_map={},
        gen_resource_map={},
    )

    called = False

    def _unexpected(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        return object()

    manager.connect(
        box_names=["A"],
        available_boxes=lambda: ("A",),
        parallel=False,
        default_parallel_mode=True,
        create_boxpool=_unexpected,
        create_quel1system_from_boxpool=_unexpected,
        create_resource_map=_unexpected,
    )

    assert called is False


def test_disconnect_clears_connected_runtime_state() -> None:
    """Given connected state, when disconnect runs, then state and resources are cleared."""
    manager = Quel1ConnectionManager()
    disconnected: list[str] = []
    resources = ["clockmaster", "box-a"]
    manager.set_connected_state(
        boxpool=object(),
        quel1system=object(),
        cap_resource_map={"cap": {}},
        gen_resource_map={"gen": {}},
    )

    manager.disconnect(
        collect_held_resources=lambda: resources,
        disconnect_resource_safely=lambda resource: disconnected.append(resource),
    )

    assert disconnected == resources
    assert manager.is_connected is False
    assert manager.boxpool is None
    assert manager.quel1system is None
    assert manager.cap_resource_map is None
    assert manager.gen_resource_map is None
