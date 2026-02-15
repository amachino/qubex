# ruff: noqa: SLF001

"""Tests for temporary qubecalib compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qubex.backend.quel1.qubecalib_compat import (
    add_executor_command,
    clear_boxpool_config_cache,
    clear_system_box_cache,
    db_box_settings,
    db_clockmaster_setting,
    db_port_settings,
    db_target_settings,
    ensure_channel_target_relation,
    get_boxpool_boxes,
    get_boxpool_config_cache,
    register_existing_box,
    replace_boxpool_config_cache,
    replace_system_box_cache,
    set_system_config_fetched_at,
    update_boxpool_config_cache,
    update_system_box_cache,
)


@dataclass
class _Setting:
    value: str


class _ModernDb:
    def __init__(self) -> None:
        self.clockmaster_setting = _Setting("clock")
        self.box_settings = {"A": _Setting("box")}
        self.port_settings = {"P": _Setting("port")}
        self.target_settings = {"T": _Setting("target")}
        self.relation_channel_target: list[tuple[str, str]] = []

    def assign_target_to_channel(self, *, channel: str, target: str) -> None:
        """Append relation through modern API."""
        self.relation_channel_target.append((channel, target))


class _LegacyDb:
    def __init__(self) -> None:
        self._clockmaster_setting = _Setting("clock")
        self._box_settings = {"A": _Setting("box")}
        self._port_settings = {"P": _Setting("port")}
        self._target_settings = {"T": _Setting("target")}
        self._relation_channel_target: list[tuple[str, str]] = []


class _ModernBoxPool:
    def __init__(self) -> None:
        self.boxes: dict[str, tuple[Any, Any]] = {}
        self.box_config_cache: dict[str, Any] = {}

    def register_existing_box(self, *, box_name: str, box: Any, sequencer: Any) -> None:
        """Store existing box through modern API."""
        self.boxes[box_name] = (box, sequencer)

    def clear_box_config_cache(self) -> None:
        """Clear cache through modern API."""
        self.box_config_cache.clear()

    def replace_box_config_cache(self, box_configs: dict[str, Any]) -> None:
        """Replace cache through modern API."""
        self.box_config_cache = box_configs

    def update_box_config_cache(self, box_configs: dict[str, Any]) -> None:
        """Update cache through modern API."""
        self.box_config_cache.update(box_configs)


class _LegacyBoxPool:
    def __init__(self) -> None:
        self._boxes: dict[str, tuple[Any, Any]] = {}
        self._linkstatus: dict[str, bool] = {}
        self._box_config_cache: dict[str, Any] = {}


class _Queue:
    def __init__(self) -> None:
        self.commands: list[Any] = []

    def add_command(self, command: Any) -> None:
        """Record added command."""
        self.commands.append(command)


class _ModernQubeCalib:
    def __init__(self) -> None:
        self.executor = _Queue()


class _LegacyQubeCalib:
    def __init__(self) -> None:
        self._executor = _Queue()


class _SystemWithBoxCache:
    def __init__(self) -> None:
        self.box_cache: dict[str, Any] = {}
        self.config_fetched_at: Any = None


class _SystemWithConfigCache:
    def __init__(self) -> None:
        self.config_cache: dict[str, Any] = {}
        self.config_fetched_at: Any = None


def test_db_accessors_support_public_and_private_attributes() -> None:
    """Given modern and legacy DB objects, accessors resolve both attribute styles."""
    modern = _ModernDb()
    legacy = _LegacyDb()

    assert db_clockmaster_setting(modern).value == "clock"
    assert db_clockmaster_setting(legacy).value == "clock"
    assert db_box_settings(modern)["A"].value == "box"
    assert db_box_settings(legacy)["A"].value == "box"
    assert db_port_settings(modern)["P"].value == "port"
    assert db_port_settings(legacy)["P"].value == "port"
    assert db_target_settings(modern)["T"].value == "target"
    assert db_target_settings(legacy)["T"].value == "target"


def test_boxpool_helpers_support_modern_and_legacy_shapes() -> None:
    """Given modern and legacy boxpool objects, helpers update both layout variants."""
    modern = _ModernBoxPool()
    legacy = _LegacyBoxPool()

    register_existing_box(modern, box_name="A", box="box-A", sequencer="seq-A")
    register_existing_box(legacy, box_name="A", box="box-A", sequencer="seq-A")
    assert get_boxpool_boxes(modern)["A"] == ("box-A", "seq-A")
    assert get_boxpool_boxes(legacy)["A"] == ("box-A", "seq-A")
    assert legacy._linkstatus["A"] is False

    replace_boxpool_config_cache(modern, {"A": {"v": 1}})
    replace_boxpool_config_cache(legacy, {"A": {"v": 1}})
    update_boxpool_config_cache(modern, {"B": {"v": 2}})
    update_boxpool_config_cache(legacy, {"B": {"v": 2}})
    assert get_boxpool_config_cache(modern) == {"A": {"v": 1}, "B": {"v": 2}}
    assert get_boxpool_config_cache(legacy) == {"A": {"v": 1}, "B": {"v": 2}}

    clear_boxpool_config_cache(modern)
    clear_boxpool_config_cache(legacy)
    assert get_boxpool_config_cache(modern) == {}
    assert get_boxpool_config_cache(legacy) == {}


def test_add_executor_command_supports_modern_and_legacy_qubecalib() -> None:
    """Given modern and legacy QubeCalib shapes, sequencer command is enqueued."""
    modern = _ModernQubeCalib()
    legacy = _LegacyQubeCalib()

    add_executor_command(modern, "seq-1")
    add_executor_command(legacy, "seq-2")

    assert modern.executor.commands == ["seq-1"]
    assert legacy._executor.commands == ["seq-2"]


def test_ensure_channel_target_relation_supports_both_db_shapes() -> None:
    """Given modern and legacy DB relation APIs, relation insertion remains idempotent."""
    modern = _ModernDb()
    legacy = _LegacyDb()

    assert ensure_channel_target_relation(modern, channel_name="C0", target_name="Q0")
    assert not ensure_channel_target_relation(
        modern, channel_name="C0", target_name="Q0"
    )
    assert modern.relation_channel_target == [("C0", "Q0")]

    assert ensure_channel_target_relation(legacy, channel_name="C1", target_name="Q1")
    assert not ensure_channel_target_relation(
        legacy, channel_name="C1", target_name="Q1"
    )
    assert legacy._relation_channel_target == [("C1", "Q1")]


def test_system_cache_helpers_support_box_cache_and_config_cache() -> None:
    """Given two system cache shapes, helper operations update cache and timestamp."""
    with_box_cache = _SystemWithBoxCache()
    with_config_cache = _SystemWithConfigCache()

    replace_system_box_cache(with_box_cache, {"A": {"value": 1}})
    replace_system_box_cache(with_config_cache, {"B": {"value": 2}})
    assert with_box_cache.box_cache == {"A": {"value": 1}}
    assert with_config_cache.config_cache == {"B": {"value": 2}}

    update_system_box_cache(with_box_cache, {"C": {"value": 3}})
    update_system_box_cache(with_config_cache, {"D": {"value": 4}})
    assert with_box_cache.box_cache == {"A": {"value": 1}, "C": {"value": 3}}
    assert with_config_cache.config_cache == {
        "B": {"value": 2},
        "D": {"value": 4},
    }
    assert with_box_cache.config_fetched_at is not None
    assert with_config_cache.config_fetched_at is not None

    clear_system_box_cache(with_box_cache)
    clear_system_box_cache(with_config_cache)
    assert with_box_cache.box_cache == {}
    assert with_config_cache.config_cache == {}
    assert with_box_cache.config_fetched_at is None
    assert with_config_cache.config_fetched_at is None

    set_system_config_fetched_at(with_box_cache, "stamp")
    set_system_config_fetched_at(with_config_cache, "stamp")
    assert with_box_cache.config_fetched_at == "stamp"
    assert with_config_cache.config_fetched_at == "stamp"
