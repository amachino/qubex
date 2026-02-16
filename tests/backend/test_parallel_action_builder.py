"""Tests for parallel action builder setting conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import qubex.backend.quel1.execution.parallel_action_builder as parallel_action_builder
from qubex.backend.quel1.execution.parallel_action_builder import (
    _convert_to_box_setting_dict,
    build_parallel_multi_action,
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


@dataclass(frozen=True)
class _SingleLikeRunitId:
    port: str
    runit: int


@dataclass(frozen=True)
class _SingleLikeAwgId:
    port: str
    channel: int


@dataclass(frozen=True)
class _SingleLikeRunitSetting:
    runit: _SingleLikeRunitId
    cprm: Any


@dataclass(frozen=True)
class _SingleLikeAwgSetting:
    awg: _SingleLikeAwgId
    wseq: Any


@dataclass(frozen=True)
class _SingleLikeTriggerSetting:
    trigger_awg: _SingleLikeAwgId
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


def test_convert_to_box_setting_dict_supports_single_style_ids_without_box() -> None:
    """Given single-style IDs, conversion rebuilds settings without requiring a box field."""
    settings = [
        _RunitConfig(runit=_RunitRef(box="B0", port="P0", runit=1), cprm="CP"),
        _AwgConfig(awg=_AwgRef(box="B0", port="P1", channel=2), wseq="WQ"),
        _TriggerConfig(
            trigger_awg=_AwgRef(box="B0", port="P1", channel=2),
            triggerd_port="TP",
        ),
    ]

    result = _convert_to_box_setting_dict(
        settings=settings,
        awg_id_class=_SingleLikeAwgId,
        awg_setting_class=_SingleLikeAwgSetting,
        runit_id_class=_SingleLikeRunitId,
        runit_setting_class=_SingleLikeRunitSetting,
        trigger_setting_class=_SingleLikeTriggerSetting,
    )

    runit_setting = next(item for item in result["B0"] if hasattr(item, "cprm"))
    awg_setting = next(item for item in result["B0"] if hasattr(item, "wseq"))
    trigger_setting = next(
        item for item in result["B0"] if hasattr(item, "triggerd_port")
    )
    assert runit_setting.runit == _SingleLikeRunitId(port="P0", runit=1)
    assert awg_setting.awg == _SingleLikeAwgId(port="P1", channel=2)
    assert trigger_setting.trigger_awg == _SingleLikeAwgId(port="P1", channel=2)


def test_build_parallel_multi_action_uses_driver_single_setting_classes(
    monkeypatch,
) -> None:
    """Given split common/single classes, parallel build passes driver Single* setting classes to SingleAction."""

    @dataclass(frozen=True)
    class _CommonRunitId:
        box: str
        port: str
        runit: int

    @dataclass(frozen=True)
    class _CommonRunitSetting:
        runit: _CommonRunitId
        cprm: Any

    @dataclass(frozen=True)
    class _SingleRunitId:
        port: str
        runit: int

    @dataclass(frozen=True)
    class _SingleRunitSetting:
        runit: _SingleRunitId
        cprm: Any

    class _SingleAction:
        @classmethod
        def build(cls, *, box: object, settings: list[object]) -> object:
            _ = box
            assert settings
            assert all(isinstance(item, _SingleRunitSetting) for item in settings)
            return SimpleNamespace(_cprms={})

    class _FakeMultiAction:
        @staticmethod
        def _get_reference_box_name(actions: dict[str, object]) -> str:
            return next(iter(actions))

    fake_driver = SimpleNamespace(
        MultiAction=_FakeMultiAction,
        SingleAction=_SingleAction,
        AwgId=_FakeAwgId,
        AwgSetting=_FakeAwgSetting,
        RunitId=_CommonRunitId,
        RunitSetting=_CommonRunitSetting,
        TriggerSetting=_FakeTriggerSetting,
        SingleAwgId=_SingleLikeAwgId,
        SingleAwgSetting=_SingleLikeAwgSetting,
        SingleRunitId=_SingleRunitId,
        SingleRunitSetting=_SingleRunitSetting,
        SingleTriggerSetting=_SingleLikeTriggerSetting,
    )
    monkeypatch.setattr(
        parallel_action_builder, "load_quel1_driver", lambda: fake_driver
    )
    monkeypatch.setattr(parallel_action_builder, "adapt_quel1_box", lambda box: box)

    settings = [
        _CommonRunitSetting(runit=_CommonRunitId(box="B0", port="P0", runit=1), cprm=1),
        _CommonRunitSetting(runit=_CommonRunitId(box="B1", port="P1", runit=2), cprm=2),
    ]
    system = SimpleNamespace(
        boxes={"B0": object(), "B1": object()},
        box={"B0": object(), "B1": object()},
        _clockmaster=SimpleNamespace(read_clock=lambda: 0),
        timing_shift={"B0": 0, "B1": 0},
        displacement=0,
    )

    _, cprms = build_parallel_multi_action(
        system=cast(Any, system),
        settings=settings,
        action_builder=cast(Any, (lambda **_: None)),
        logger=logging.getLogger(__name__),
    )

    assert cprms == {}
