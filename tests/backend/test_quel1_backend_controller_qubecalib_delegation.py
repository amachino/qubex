# ruff: noqa: SLF001

"""Tests for qubecalib delegation methods in Quel1BackendController."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

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


def _write_skew_yaml(tmp_path: Path, *, wait: int) -> Path:
    path = tmp_path / "skew.yaml"
    path.write_text(
        f"box_setting:\n  Q00:\n    slot: 0\n    wait: {wait}\ntime_to_start: 0\n",
        encoding="utf-8",
    )
    return path


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


def test_load_skew_yaml_delegates_to_sysdb(tmp_path: Path) -> None:
    """Given a path, when loading skew yaml, then sysdb.load_skew_yaml is called once."""
    controller = _make_controller()
    path = _write_skew_yaml(tmp_path, wait=250)

    controller.load_skew_yaml(path)

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.sysdb.load_skew_yaml_calls == [str(path)]


def test_load_skew_yaml_rejects_negative_wait(tmp_path: Path) -> None:
    """Given a negative wait, when loading skew yaml, then ValueError is raised before delegation."""
    controller = _make_controller()
    path = _write_skew_yaml(tmp_path, wait=-1)

    with pytest.raises(ValueError, match="wait must be non-negative"):
        controller.load_skew_yaml(path)

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.sysdb.load_skew_yaml_calls == []


def test_update_skew_updates_yaml_and_reloads_sysdb(tmp_path: Path) -> None:
    """Given a skew file, when update_skew is called, then waits are updated and sysdb reloads the file."""
    controller = _make_controller()
    path = tmp_path / "skew.yaml"
    path.write_text(
        """
box_setting:
  Q00:
    slot: 0
    wait: 0
  Q01:
    slot: 1
    wait: 1
time_to_start: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = controller.update_skew(
        file_path=path,
        wait=250,
        box_names=["Q01"],
        backup=True,
    )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert result["wait"] == 250
    assert result["file_path"] == path
    assert result["backup_path"] == path.with_suffix(".yaml.bak")
    assert result["box_names"] == ["Q01"]
    assert payload["box_setting"]["Q00"]["wait"] == 0
    assert payload["box_setting"]["Q01"]["wait"] == 250
    assert qubecalib.sysdb.load_skew_yaml_calls == [str(path)]


def test_add_channel_target_relation_is_idempotent() -> None:
    """Given duplicate relation requests, when appending, then relation is added only once."""
    controller = _make_controller()

    controller.add_channel_target_relation("Q00-1-0", "Q00-1")
    controller.add_channel_target_relation("Q00-1-0", "Q00-1")

    qubecalib = cast(_FakeQubeCalib, controller.qubecalib)
    assert qubecalib.sysdb._relation_channel_target == [("Q00-1-0", "Q00-1")]
