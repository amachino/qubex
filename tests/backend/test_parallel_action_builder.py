"""Tests for parallel action builder setting conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import qubex.backend.quel1.compat.parallel_action_builder as parallel_action_builder
from qubex.backend.quel1.compat.parallel_action_builder import (
    ClockHealthCheckOptions,
    QubexMultiAction,
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


@dataclass(frozen=True)
class _UnknownSetting:
    payload: str


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
        awg_id_class=cast(Any, _FakeAwgId),
        awg_setting_class=cast(Any, _FakeAwgSetting),
        runit_id_class=cast(Any, _FakeRunitId),
        runit_setting_class=cast(Any, _FakeRunitSetting),
        trigger_setting_class=cast(Any, _FakeTriggerSetting),
    )

    # Assert
    assert set(result) == {"B0", "B1"}
    runit_setting = cast(
        _FakeRunitSetting, next(item for item in result["B0"] if hasattr(item, "cprm"))
    )
    awg_setting = cast(
        _FakeAwgSetting, next(item for item in result["B0"] if hasattr(item, "wseq"))
    )
    trigger_setting = cast(_FakeTriggerSetting, result["B1"][0])

    assert runit_setting.runit == _FakeRunitId(box="B0", port="P0", runit=1)
    assert awg_setting.awg == _FakeAwgId(box="B0", port="P1", channel=2)
    assert trigger_setting.trigger_awg == _FakeAwgId(box="B1", port="P2", channel=3)


def test_convert_to_box_setting_dict_supports_single_style_ids_without_box() -> None:
    """Given single-style IDs, when converting settings by box, then settings are rebuilt without requiring a box field."""
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
        awg_id_class=cast(Any, _SingleLikeAwgId),
        awg_setting_class=cast(Any, _SingleLikeAwgSetting),
        runit_id_class=cast(Any, _SingleLikeRunitId),
        runit_setting_class=cast(Any, _SingleLikeRunitSetting),
        trigger_setting_class=cast(Any, _SingleLikeTriggerSetting),
    )

    runit_setting = cast(
        _SingleLikeRunitSetting,
        next(item for item in result["B0"] if hasattr(item, "cprm")),
    )
    awg_setting = cast(
        _SingleLikeAwgSetting,
        next(item for item in result["B0"] if hasattr(item, "wseq")),
    )
    trigger_setting = cast(
        _SingleLikeTriggerSetting,
        next(item for item in result["B0"] if hasattr(item, "triggerd_port")),
    )
    assert runit_setting.runit == _SingleLikeRunitId(port="P0", runit=1)
    assert awg_setting.awg == _SingleLikeAwgId(port="P1", channel=2)
    assert trigger_setting.trigger_awg == _SingleLikeAwgId(port="P1", channel=2)


def test_convert_to_box_setting_dict_ignores_unknown_settings() -> None:
    """Given unknown settings, when converting settings by box, then unknown entries are skipped."""
    settings = [
        _UnknownSetting(payload="noop"),
        _RunitConfig(runit=_RunitRef(box="B0", port="P0", runit=1), cprm="CP"),
    ]

    result = _convert_to_box_setting_dict(
        settings=settings,
        awg_id_class=cast(Any, _SingleLikeAwgId),
        awg_setting_class=cast(Any, _SingleLikeAwgSetting),
        runit_id_class=cast(Any, _SingleLikeRunitId),
        runit_setting_class=cast(Any, _SingleLikeRunitSetting),
        trigger_setting_class=cast(Any, _SingleLikeTriggerSetting),
    )

    assert set(result) == {"B0"}
    assert len(result["B0"]) == 1


def test_build_parallel_multi_action_uses_driver_single_setting_classes(
    monkeypatch,
) -> None:
    """Given split common/single classes, when building parallel multi action, then driver Single* setting classes are passed to SingleAction."""

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


def test_build_parallel_multi_action_single_box_path_uses_action_builder(
    monkeypatch,
) -> None:
    """Given one box, when building parallel multi action, then action_builder path is used and capture map is returned."""

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
            _ = settings
            raise AssertionError("single build must not run on single-box fast path")

    fake_driver = SimpleNamespace(
        MultiAction=SimpleNamespace,
        SingleAction=_SingleAction,
        SingleAwgId=_SingleLikeAwgId,
        SingleAwgSetting=_SingleLikeAwgSetting,
        SingleRunitId=_SingleRunitId,
        SingleRunitSetting=_SingleRunitSetting,
        SingleTriggerSetting=_SingleLikeTriggerSetting,
    )
    monkeypatch.setattr(
        parallel_action_builder, "load_quel1_driver", lambda: fake_driver
    )

    action_single = SimpleNamespace(
        _cprms={
            _SingleRunitId(port="P0", runit=1): "CP0",
            _SingleRunitId(port="P0", runit=2): "CP1",
        }
    )
    action_obj = SimpleNamespace(_action=("B0", action_single))
    action_builder_calls: dict[str, Any] = {}

    def _action_builder(**kwargs: Any) -> object:
        action_builder_calls.update(kwargs)
        return action_obj

    settings = [
        _CommonRunitSetting(runit=_CommonRunitId(box="B0", port="P0", runit=1), cprm=1)
    ]
    system = SimpleNamespace(
        boxes={"B0": object()},
        box={"B0": object()},
        _clockmaster=SimpleNamespace(read_clock=lambda: 0),
        timing_shift={"B0": 0},
        displacement=0,
    )

    action, cprms = build_parallel_multi_action(
        system=cast(Any, system),
        settings=settings,
        action_builder=cast(Any, _action_builder),
        logger=logging.getLogger(__name__),
    )

    assert action is action_obj
    assert action_builder_calls["system"] is system
    assert action_builder_calls["settings"] == settings
    assert cprms == {("B0", "P0", 1): "CP0", ("B0", "P0", 2): "CP1"}


@dataclass
class _CompletedWavegenTask:
    def result(self) -> None:
        """Return immediately for scheduled wavegen."""


@dataclass
class _FakeWavegenBox:
    current_time: int = 100
    latest_sysref_time: int = 0
    reservations: list[tuple[set[tuple[str, int]], int | None]] = field(
        default_factory=list
    )

    def get_current_timecounter(self) -> int:
        """Return the mocked current time counter."""
        return self.current_time

    def get_latest_sysref_timecounter(self) -> int:
        """Return the mocked latest SYSREF time counter."""
        return self.latest_sysref_time

    def start_wavegen(
        self,
        channels: set[tuple[str, int]],
        timecounter: int | None = None,
    ) -> _CompletedWavegenTask:
        """Record one emission reservation."""
        self.reservations.append((channels, timecounter))
        return _CompletedWavegenTask()


@dataclass
class _FakeTriggeredAction:
    box: _FakeWavegenBox
    _wseqs: list[SimpleNamespace]
    _triggers: dict[str, SimpleNamespace]
    _cprms: dict[object, object] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        box: _FakeWavegenBox,
        settings: list[object],
    ) -> _FakeTriggeredAction:
        """Build one fake action for protocol compatibility in tests."""
        _ = settings
        return cls(box=box, _wseqs=[], _triggers={})


def test_qubecalib_emit_at_reserves_triggered_boxes() -> None:
    """Qubecalib compatibility should reserve emission for triggered boxes."""
    box = _FakeWavegenBox()
    action = _FakeTriggeredAction(
        box=box,
        _wseqs=[SimpleNamespace(port="P0", channel=1)],
        _triggers={"P1": SimpleNamespace(port="P0", channel=1)},
    )
    multi_action = QubexMultiAction(
        _system=cast(
            Any,
            SimpleNamespace(
                box={"B0": box},
                timing_shift={"B0": 0},
                displacement=0,
            ),
        ),
        _actions=cast(Any, MappingProxyType({"B0": action})),
        _estimated_timediff=MappingProxyType({"B0": 0}),
        _reference_box_name="B0",
        _ref_sysref_time_offset=0,
        _clock_options=ClockHealthCheckOptions(),
        _logger=logging.getLogger(__name__),
        _emit_triggered_boxes=True,
    )

    multi_action.emit_at(min_time_offset=16)

    assert box.reservations == [({("P0", 1)}, 128)]


def test_qxdriver_emit_at_skips_triggered_boxes() -> None:
    """Qxdriver compatibility should not reserve emission for triggered boxes."""
    box = _FakeWavegenBox()
    action = _FakeTriggeredAction(
        box=box,
        _wseqs=[SimpleNamespace(port="P0", channel=1)],
        _triggers={"P1": SimpleNamespace(port="P0", channel=1)},
    )
    multi_action = QubexMultiAction(
        _system=cast(
            Any,
            SimpleNamespace(
                box={"B0": box},
                timing_shift={"B0": 0},
                displacement=0,
            ),
        ),
        _actions=cast(Any, MappingProxyType({"B0": action})),
        _estimated_timediff=MappingProxyType({"B0": 0}),
        _reference_box_name="B0",
        _ref_sysref_time_offset=0,
        _clock_options=ClockHealthCheckOptions(),
        _logger=logging.getLogger(__name__),
        _emit_triggered_boxes=False,
    )

    multi_action.emit_at(min_time_offset=16)

    assert box.reservations == []
