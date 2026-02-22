# ruff: noqa: SLF001

"""Tests for qubecalib delegation methods in Quel1BackendController."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import qubex.backend.quel1.compat.sequencer as sequencer_module
from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


@dataclass
class _ChannelRelation:
    channel: str
    target: str


class _FakeSysDb:
    def __init__(self) -> None:
        self._relation_channel_target: list[tuple[str, str]] = []
        self.load_skew_yaml_calls: list[str] = []

    def load_skew_yaml(self, path: str) -> None:
        """Record load_skew_yaml calls."""
        self.load_skew_yaml_calls.append(path)


class _FakeQubeCalib:
    def __init__(self) -> None:
        self.sysdb = _FakeSysDb()
        self.define_clockmaster_calls: list[dict[str, Any]] = []
        self.define_box_calls: list[dict[str, Any]] = []
        self.define_port_calls: list[dict[str, Any]] = []
        self.define_channel_calls: list[dict[str, Any]] = []

    def define_clockmaster(self, **kwargs: Any) -> None:
        """Record define_clockmaster calls."""
        self.define_clockmaster_calls.append(kwargs)

    def define_box(self, **kwargs: Any) -> None:
        """Record define_box calls."""
        self.define_box_calls.append(kwargs)

    def define_port(self, **kwargs: Any) -> None:
        """Record define_port calls."""
        self.define_port_calls.append(kwargs)

    def define_channel(self, **kwargs: Any) -> None:
        """Record define_channel calls."""
        self.define_channel_calls.append(kwargs)


def _make_controller() -> Quel1BackendController:
    controller = Quel1BackendController()
    fake_qubecalib = _FakeQubeCalib()
    cast(Any, controller)._qubecalib = fake_qubecalib
    cast(Any, controller)._runtime_context._qubecalib = fake_qubecalib
    return controller


def test_define_helpers_delegate_to_qubecalib() -> None:
    """Given helper methods, when called, then qubecalib methods receive the same kwargs."""
    controller = _make_controller()

    controller.define_clockmaster(ipaddr="192.0.2.11", reset=True)
    controller.define_box(
        box_name="Q00",
        ipaddr_wss="192.0.2.21",
        boxtype="quel1-a",
    )
    controller.define_port(
        port_name="Q00-1",
        box_name="Q00",
        port_number=1,
    )
    controller.define_channel(
        channel_name="Q00-1-0",
        port_name="Q00-1",
        channel_number=0,
        ndelay_or_nwait=16,
    )

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.define_clockmaster_calls == [
        {"ipaddr": "192.0.2.11", "reset": True}
    ]
    assert qubecalib.define_box_calls == [
        {
            "box_name": "Q00",
            "ipaddr_wss": "192.0.2.21",
            "boxtype": "quel1-a",
        }
    ]
    assert qubecalib.define_port_calls == [
        {
            "port_name": "Q00-1",
            "box_name": "Q00",
            "port_number": 1,
        }
    ]
    assert qubecalib.define_channel_calls == [
        {
            "channel_name": "Q00-1-0",
            "port_name": "Q00-1",
            "channel_number": 0,
            "ndelay_or_nwait": 16,
        }
    ]


def test_load_skew_yaml_delegates_to_sysdb() -> None:
    """Given a path, when loading skew yaml, then sysdb.load_skew_yaml is called once."""
    controller = _make_controller()

    controller.load_skew_yaml("skew.yaml")

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.sysdb.load_skew_yaml_calls == ["skew.yaml"]


def test_add_channel_target_relation_is_idempotent() -> None:
    """Given duplicate relation requests, when appending, then relation is added only once."""
    controller = _make_controller()

    controller.add_channel_target_relation("Q00-1-0", "Q00-1")
    controller.add_channel_target_relation("Q00-1-0", "Q00-1")

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.sysdb._relation_channel_target == [("Q00-1-0", "Q00-1")]


def test_create_quel1_sequencer_passes_driver_for_constructor_compatibility(
    monkeypatch: Any,
) -> None:
    """Given controller sequencer creation, when constructing, then driver is passed for compatibility."""
    controller = _make_controller()
    fake_driver = object()
    controller._connection_manager.set_quel1system(cast(Any, fake_driver))
    created_kwargs: dict[str, Any] = {}

    class _FakeSequencer:
        def __init__(self, **kwargs: Any) -> None:
            created_kwargs.update(kwargs)

    monkeypatch.setattr(sequencer_module, "Quel1Sequencer", _FakeSequencer)

    controller.create_quel1_sequencer(
        gen_sampled_sequence={},
        cap_sampled_sequence={},
        resource_map={},
        interval=128,
    )

    assert created_kwargs["driver"] is fake_driver
    assert created_kwargs["sysdb"] is controller.qubecalib.sysdb
