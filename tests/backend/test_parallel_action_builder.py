"""Tests for parallel action builder setting conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qubex.backend.quel1.execution.parallel_action_builder import (
    _convert_to_box_setting_dict,
)


@dataclass(frozen=True)
class _RunitRef:
    box: str
    port: str
    runit: int


@dataclass(frozen=True)
class _AwgRef:
    box: str
    port: str
    channel: int


@dataclass(frozen=True)
class _RunitConfig:
    runit: _RunitRef
    cprm: Any


@dataclass(frozen=True)
class _AwgConfig:
    awg: _AwgRef
    wseq: Any


@dataclass(frozen=True)
class _TriggerConfig:
    trigger_awg: _AwgRef
    triggerd_port: str


@dataclass(frozen=True)
class _FakeRunitId:
    box: str
    port: str
    runit: int


@dataclass(frozen=True)
class _FakeAwgId:
    box: str
    port: str
    channel: int


@dataclass(frozen=True)
class _FakeRunitSetting:
    runit: _FakeRunitId
    cprm: Any


@dataclass(frozen=True)
class _FakeAwgSetting:
    awg: _FakeAwgId
    wseq: Any


@dataclass(frozen=True)
class _FakeTriggerSetting:
    trigger_awg: _FakeAwgId
    triggerd_port: str


def test_convert_to_box_setting_dict_keeps_box_in_rebuilt_driver_ids() -> None:
    """Given mixed settings, when converting by box, then rebuilt IDs keep box/port/channel fields."""
    # Arrange
    settings = [
        _RunitConfig(runit=_RunitRef(box="B0", port="P0", runit=1), cprm="CP"),
        _AwgConfig(awg=_AwgRef(box="B0", port="P1", channel=2), wseq="WQ"),
        _TriggerConfig(
            trigger_awg=_AwgRef(box="B1", port="P2", channel=3),
            triggerd_port="TP",
        ),
    ]

    # Act
    result = _convert_to_box_setting_dict(
        settings=settings,
        awg_id_class=_FakeAwgId,
        awg_setting_class=_FakeAwgSetting,
        runit_id_class=_FakeRunitId,
        runit_setting_class=_FakeRunitSetting,
        trigger_setting_class=_FakeTriggerSetting,
    )

    # Assert
    assert set(result) == {"B0", "B1"}
    runit_setting = next(item for item in result["B0"] if hasattr(item, "cprm"))
    awg_setting = next(item for item in result["B0"] if hasattr(item, "wseq"))
    trigger_setting = result["B1"][0]

    assert runit_setting.runit == _FakeRunitId(box="B0", port="P0", runit=1)
    assert awg_setting.awg == _FakeAwgId(box="B0", port="P1", channel=2)
    assert trigger_setting.trigger_awg == _FakeAwgId(box="B1", port="P2", channel=3)
