"""Tests for stateful QuEL-1 connection manager behavior."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from qubex.backend.quel1.compat.qubecalib_protocols import (
    BoxPoolProtocol,
    Quel1SystemProtocol,
)
from qubex.backend.quel1.managers.connection_manager import Quel1ConnectionManager
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext


class _FakeClockmaster:
    pass


class _FakeSequencerClient:
    def read_counter(self) -> tuple[bool, int, int]:
        return True, 0, 0


class _FakeBox:
    boxtype = "quel1-a"

    def reconnect(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def relinkup(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def link_status(self) -> dict[int, bool]:
        return {0: True}

    def get_input_ports(self) -> tuple[int, ...]:
        return (0,)

    def get_output_ports(self) -> tuple[int, ...]:
        return (1,)

    def dump_box(self) -> dict[str, Any]:
        return {}

    def dump_port(self, port: int | tuple[int, int]) -> dict[str, Any]:
        _ = port
        return {}

    def config_port(
        self,
        *,
        port: int | tuple[int, int],
        lo_freq: float | None = None,
        cnco_freq: float | None = None,
        vatt: int | None = None,
        sideband: str | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
    ) -> None:
        _ = (
            port,
            lo_freq,
            cnco_freq,
            vatt,
            sideband,
            fullscale_current,
            rfswitch,
        )

    def config_channel(
        self,
        *,
        port: int | tuple[int, int],
        channel: int,
        fnco_freq: float | None = None,
    ) -> None:
        _ = (port, channel, fnco_freq)

    def config_runit(
        self,
        *,
        port: int | tuple[int, int],
        runit: int,
        fnco_freq: float | None = None,
    ) -> None:
        _ = (port, runit, fnco_freq)


class _FakeBoxPool:
    def __init__(self) -> None:
        box = _FakeBox()
        sequencer = _FakeSequencerClient()
        self._boxes = {"A": (box, sequencer), "B": (box, sequencer)}
        self._linkstatus = {"A": True, "B": True}
        self._box_config_cache: dict[str, dict[str, Any]] = {}

    def create_clock_master(self, *, ipaddr: str) -> None:
        _ = ipaddr

    def create(
        self,
        box_name: str,
        *,
        ipaddr_wss: str,
        ipaddr_sss: str,
        ipaddr_css: str,
        boxtype: str,
    ) -> _FakeBox:
        _ = (box_name, ipaddr_wss, ipaddr_sss, ipaddr_css, boxtype)
        return _FakeBox()

    def get_box(
        self,
        box_name: str,
    ) -> tuple[_FakeBox, _FakeSequencerClient]:
        return self._boxes[box_name]

    def get_port_direction(self, box_name: str, port: int | tuple[int, int]) -> str:
        _ = (box_name, port)
        return "in"

    def ensure_box_config_cache(
        self,
        *,
        box_name: str,
        box: _FakeBox,
    ) -> dict[str, Any]:
        _ = box
        return self._box_config_cache.setdefault(box_name, {})


class _FakeQuel1System:
    boxes: dict[str, _FakeBox]
    box: dict[str, _FakeBox]
    _clockmaster: _FakeClockmaster
    timing_shift: dict[str, int]
    displacement: int
    config_cache: dict[str, dict[str, Any]]
    config_fetched_at: datetime | None

    def __init__(self) -> None:
        box = _FakeBox()
        self.boxes = {"A": box, "B": box}
        self.box = self.boxes
        self._clockmaster = _FakeClockmaster()
        self.timing_shift = {}
        self.displacement = 0
        self.config_cache = {}
        self.config_fetched_at = None

    @classmethod
    def create(
        cls,
        *,
        clockmaster: _FakeClockmaster,
        boxes: tuple[object, ...],
        update_copnfig_cache: bool = False,
    ) -> _FakeQuel1System:
        _ = (clockmaster, boxes, update_copnfig_cache)
        return cls()


def test_connect_stores_connected_runtime_state() -> None:
    """Given connect callbacks, when connect runs, then runtime state is stored in manager."""
    manager = Quel1ConnectionManager(runtime_context=Quel1RuntimeContext())
    boxpool = cast(BoxPoolProtocol, _FakeBoxPool())
    quel1system = cast(Quel1SystemProtocol, _FakeQuel1System())

    def _create_quel1system_from_boxpool(
        box_names: list[str],
    ) -> Quel1SystemProtocol:
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
    manager = Quel1ConnectionManager(runtime_context=Quel1RuntimeContext())
    manager.set_connected_state(
        boxpool=cast(BoxPoolProtocol, _FakeBoxPool()),
        quel1system=cast(Quel1SystemProtocol, _FakeQuel1System()),
        cap_resource_map={},
        gen_resource_map={},
    )

    called = False

    def _unexpected_boxpool(_names: list[str], _parallel: bool) -> BoxPoolProtocol:
        nonlocal called
        called = True
        return cast(BoxPoolProtocol, _FakeBoxPool())

    def _unexpected_quel1system(_names: list[str]) -> Quel1SystemProtocol:
        nonlocal called
        called = True
        return cast(Quel1SystemProtocol, _FakeQuel1System())

    def _unexpected_resource_map(_kind: str) -> dict[str, dict]:
        nonlocal called
        called = True
        return {}

    manager.connect(
        box_names=["A"],
        available_boxes=lambda: ("A",),
        parallel=False,
        default_parallel_mode=True,
        create_boxpool=_unexpected_boxpool,
        create_quel1system_from_boxpool=_unexpected_quel1system,
        create_resource_map=_unexpected_resource_map,
    )

    assert called is False


def test_disconnect_clears_connected_runtime_state() -> None:
    """Given connected state, when disconnect runs, then state and resources are cleared."""
    manager = Quel1ConnectionManager(runtime_context=Quel1RuntimeContext())
    disconnected: list[object] = []
    resources: list[object] = ["clockmaster", "box-a"]
    manager.set_connected_state(
        boxpool=cast(BoxPoolProtocol, _FakeBoxPool()),
        quel1system=cast(Quel1SystemProtocol, _FakeQuel1System()),
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
