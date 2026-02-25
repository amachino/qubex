"""Tests for skew measurement flow in QuEL-1 skew manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, cast

from qubex.backend.quel1.managers.skew_manager import Quel1SkewManager
from qubex.backend.quel1.quel1_backend_constants import RELAXED_NOISE_THRESHOLD


class _FakeBox:
    def __init__(self, *, name: str) -> None:
        self.name = name
        self.reconnect_calls: list[dict[str, Any]] = []

    def reconnect(self, **kwargs: Any) -> None:
        self.reconnect_calls.append(kwargs)


class _FakeSysDb:
    def __init__(self) -> None:
        self.create_box_calls: list[tuple[str, bool]] = []
        self.created_boxes: dict[str, _FakeBox] = {}

    def create_box(self, box_name: str, *, reconnect: bool = True) -> _FakeBox:
        self.create_box_calls.append((box_name, reconnect))
        box = _FakeBox(name=box_name)
        self.created_boxes[box_name] = box
        return box


class _FakeQubeCalib:
    def __init__(self, *, sysdb: _FakeSysDb) -> None:
        self.system_config_database = sysdb


class _FakeSkewSystem:
    def __init__(
        self,
        *,
        boxes: dict[str, _FakeBox],
        clockmaster: object,
    ) -> None:
        self.boxes = boxes
        self._clockmaster = clockmaster
        self.resync_calls = 0

    def resync(self) -> None:
        self.resync_calls += 1


class _FakeSkewRuntime:
    def __init__(self, *, system: _FakeSkewSystem) -> None:
        self.system = system
        self.measure_calls = 0
        self.estimate_calls = 0
        self.plot_calls = 0

    def measure(self) -> None:
        self.measure_calls += 1

    def estimate(self) -> None:
        self.estimate_calls += 1

    def plot(self) -> dict[str, str]:
        self.plot_calls += 1
        return {"figure": "ok"}


@dataclass
class _FakeNamedBox:
    name: str
    box: _FakeBox


class _FakeQuel1SystemClass:
    create_calls: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def create(
        cls,
        *,
        clockmaster: object,
        boxes: list[_FakeNamedBox],
        update_copnfig_cache: bool = False,
    ) -> _FakeSkewSystem:
        cls.create_calls.append(
            {
                "clockmaster": clockmaster,
                "boxes": boxes,
                "update_copnfig_cache": update_copnfig_cache,
            }
        )
        return _FakeSkewSystem(
            boxes={named.name: named.box for named in boxes},
            clockmaster=clockmaster,
        )


class _FakeSkewClass:
    from_yaml_calls: ClassVar[list[dict[str, Any]]] = []
    created_runtimes: ClassVar[list[_FakeSkewRuntime]] = []

    @classmethod
    def from_yaml(cls, path: str, **kwargs: Any) -> _FakeSkewRuntime:
        cls.from_yaml_calls.append({"path": path, **kwargs})
        system = cast(_FakeSkewSystem, kwargs["system"])
        runtime = _FakeSkewRuntime(system=system)
        cls.created_runtimes.append(runtime)
        return runtime


class _FakeQuBEMasterClient:
    create_calls: ClassVar[list[str]] = []

    def __init__(self, ipaddr: str) -> None:
        self.ipaddr = ipaddr
        _FakeQuBEMasterClient.create_calls.append(ipaddr)


class _FakeDriver:
    Skew = _FakeSkewClass
    NamedBox = _FakeNamedBox
    Quel1System = _FakeQuel1SystemClass
    QuBEMasterClient = _FakeQuBEMasterClient


class _FakeRuntimeContext:
    def __init__(
        self,
        *,
        available_boxes: list[str],
        is_connected: bool,
        connected_system: _FakeSkewSystem | None,
        sysdb: _FakeSysDb,
    ) -> None:
        self.driver = _FakeDriver()
        self.qubecalib = _FakeQubeCalib(sysdb=sysdb)
        self._available_boxes = available_boxes
        self._is_connected = is_connected
        self._connected_system = connected_system
        self.validated_box_names: list[str] = []

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def quel1system(self) -> _FakeSkewSystem:
        if self._connected_system is None:
            raise ValueError("Not connected")
        return self._connected_system

    def validate_box_availability(self, box_name: str) -> None:
        self.validated_box_names.append(box_name)
        if box_name not in self._available_boxes:
            raise ValueError(f"Box {box_name} not available")


def _reset_fakes() -> None:
    _FakeSkewClass.from_yaml_calls.clear()
    _FakeSkewClass.created_runtimes.clear()
    _FakeQuel1SystemClass.create_calls.clear()
    _FakeQuBEMasterClient.create_calls.clear()


def test_run_skew_measurement_reuses_connected_boxes_without_reconnect() -> None:
    """Given connected boxes exist, when measuring skew, then manager reuses them without creating new boxes."""
    _reset_fakes()
    sysdb = _FakeSysDb()
    clockmaster = object()
    connected_system = _FakeSkewSystem(
        boxes={
            "A": _FakeBox(name="A"),
            "B": _FakeBox(name="B"),
        },
        clockmaster=clockmaster,
    )
    runtime_context = _FakeRuntimeContext(
        available_boxes=["A", "B"],
        is_connected=True,
        connected_system=connected_system,
        sysdb=sysdb,
    )
    manager = Quel1SkewManager(runtime_context=cast(Any, runtime_context))

    skew, fig = manager.run_skew_measurement(
        skew_yaml_path="skew.yaml",
        box_yaml_path="box.yaml",
        clockmaster_ip="192.0.2.1",
        box_names=["A", "B"],
        estimate=True,
    )

    assert fig == {"figure": "ok"}
    assert skew is _FakeSkewClass.created_runtimes[-1]
    assert sysdb.create_box_calls == []
    assert _FakeQuBEMasterClient.create_calls == []
    assert runtime_context.validated_box_names == ["A", "B"]
    create_call = _FakeQuel1SystemClass.create_calls[-1]
    assert create_call["clockmaster"] is clockmaster
    assert [named.name for named in create_call["boxes"]] == ["A", "B"]
    assert create_call["boxes"][0].box is connected_system.boxes["A"]
    assert create_call["boxes"][1].box is connected_system.boxes["B"]


def test_run_skew_measurement_adds_missing_reference_box_with_relaxed_reconnect() -> (
    None
):
    """Given connected runtime missing reference box, when measuring skew, then manager creates missing box with relaxed reconnect."""
    _reset_fakes()
    sysdb = _FakeSysDb()
    connected_system = _FakeSkewSystem(
        boxes={"A": _FakeBox(name="A")},
        clockmaster=object(),
    )
    runtime_context = _FakeRuntimeContext(
        available_boxes=["A", "R"],
        is_connected=True,
        connected_system=connected_system,
        sysdb=sysdb,
    )
    manager = Quel1SkewManager(runtime_context=cast(Any, runtime_context))

    manager.run_skew_measurement(
        skew_yaml_path="skew.yaml",
        box_yaml_path="box.yaml",
        clockmaster_ip="192.0.2.1",
        box_names=["A", "R"],
        estimate=False,
    )

    assert sysdb.create_box_calls == [("R", False)]
    assert sysdb.created_boxes["R"].reconnect_calls == [
        {"background_noise_threshold": RELAXED_NOISE_THRESHOLD}
    ]
    assert _FakeQuBEMasterClient.create_calls == []
    create_call = _FakeQuel1SystemClass.create_calls[-1]
    assert [named.name for named in create_call["boxes"]] == ["A", "R"]
    assert create_call["boxes"][0].box is connected_system.boxes["A"]
    assert create_call["boxes"][1].box is sysdb.created_boxes["R"]


def test_run_skew_measurement_creates_all_boxes_with_relaxed_reconnect_when_disconnected() -> (
    None
):
    """Given disconnected runtime, when measuring skew, then manager creates all boxes and clockmaster explicitly."""
    _reset_fakes()
    sysdb = _FakeSysDb()
    runtime_context = _FakeRuntimeContext(
        available_boxes=["A", "R"],
        is_connected=False,
        connected_system=None,
        sysdb=sysdb,
    )
    manager = Quel1SkewManager(runtime_context=cast(Any, runtime_context))

    manager.run_skew_measurement(
        skew_yaml_path="skew.yaml",
        box_yaml_path="box.yaml",
        clockmaster_ip="192.0.2.1",
        box_names=["A", "R"],
        estimate=False,
    )

    assert sysdb.create_box_calls == [("A", False), ("R", False)]
    assert sysdb.created_boxes["A"].reconnect_calls == [
        {"background_noise_threshold": RELAXED_NOISE_THRESHOLD}
    ]
    assert sysdb.created_boxes["R"].reconnect_calls == [
        {"background_noise_threshold": RELAXED_NOISE_THRESHOLD}
    ]
    assert _FakeQuBEMasterClient.create_calls == ["192.0.2.1"]
    from_yaml_call = _FakeSkewClass.from_yaml_calls[-1]
    assert "boxes" not in from_yaml_call
    assert "system" in from_yaml_call
