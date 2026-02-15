# ruff: noqa: SLF001

"""Temporary compatibility helpers for qubecalib 0.8/0.10 migration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any


def _get_public_or_private(obj: Any, public_name: str, private_name: str) -> Any:
    """Resolve an attribute from public name first, then legacy private name."""
    if hasattr(obj, public_name):
        return getattr(obj, public_name)
    return getattr(obj, private_name)


def db_clockmaster_setting(db: Any) -> Any:
    """Return the clockmaster setting for both modern and legacy sysdb layouts."""
    return _get_public_or_private(db, "clockmaster_setting", "_clockmaster_setting")


def db_box_settings(db: Any) -> dict[str, Any]:
    """Return box settings for both modern and legacy sysdb layouts."""
    return _get_public_or_private(db, "box_settings", "_box_settings")


def db_port_settings(db: Any) -> dict[str, Any]:
    """Return port settings for both modern and legacy sysdb layouts."""
    return _get_public_or_private(db, "port_settings", "_port_settings")


def db_target_settings(db: Any) -> dict[str, Any]:
    """Return target settings for both modern and legacy sysdb layouts."""
    return _get_public_or_private(db, "target_settings", "_target_settings")


def db_relation_channel_target(db: Any) -> list[tuple[str, str]]:
    """Return channel-target relations for both modern and legacy sysdb layouts."""
    return _get_public_or_private(
        db, "relation_channel_target", "_relation_channel_target"
    )


def ensure_channel_target_relation(
    db: Any,
    *,
    channel_name: str,
    target_name: str,
) -> bool:
    """Ensure one channel-target relation exists and return whether it was added."""
    relation = (channel_name, target_name)
    relations = db_relation_channel_target(db)
    if relation in relations:
        return False

    assign = getattr(db, "assign_target_to_channel", None)
    if callable(assign):
        assign(channel=channel_name, target=target_name)
    else:
        relations.append(relation)
    return True


def get_boxpool_boxes(boxpool: Any) -> dict[str, Any]:
    """Return pooled boxes for both modern and legacy boxpool layouts."""
    return _get_public_or_private(boxpool, "boxes", "_boxes")


def register_existing_box(
    boxpool: Any,
    *,
    box_name: str,
    box: Any,
    sequencer: Any,
) -> None:
    """Register an already-created box in modern and legacy boxpool variants."""
    register = getattr(boxpool, "register_existing_box", None)
    if callable(register):
        register(box_name=box_name, box=box, sequencer=sequencer)
        return

    boxpool._boxes[box_name] = (box, sequencer)
    boxpool._linkstatus[box_name] = False


def get_boxpool_config_cache(boxpool: Any) -> dict[str, Any]:
    """Return box-config cache for both modern and legacy boxpool layouts."""
    return _get_public_or_private(boxpool, "box_config_cache", "_box_config_cache")


def clear_boxpool_config_cache(boxpool: Any) -> None:
    """Clear box-config cache for both modern and legacy boxpool layouts."""
    clear = getattr(boxpool, "clear_box_config_cache", None)
    if callable(clear):
        clear()
        return

    boxpool._box_config_cache.clear()


def replace_boxpool_config_cache(boxpool: Any, box_configs: dict[str, Any]) -> None:
    """Replace box-config cache for both modern and legacy boxpool layouts."""
    replacement = deepcopy(box_configs)
    replace = getattr(boxpool, "replace_box_config_cache", None)
    if callable(replace):
        replace(replacement)
        return

    boxpool._box_config_cache = replacement


def update_boxpool_config_cache(boxpool: Any, box_configs: dict[str, Any]) -> None:
    """Update box-config cache for both modern and legacy boxpool layouts."""
    update = getattr(boxpool, "update_box_config_cache", None)
    if callable(update):
        update(deepcopy(box_configs))
        return

    for box_name, box_config in box_configs.items():
        boxpool._box_config_cache[box_name] = deepcopy(box_config)


def get_system_box_cache(system: Any) -> dict[str, Any] | None:
    """Return mutable system-side box cache for both 0.8 and 0.10 layouts."""
    box_cache = getattr(system, "box_cache", None)
    if isinstance(box_cache, dict):
        return box_cache

    config_cache = getattr(system, "config_cache", None)
    if isinstance(config_cache, dict):
        return config_cache

    return None


def set_system_config_fetched_at(system: Any, fetched_at: Any) -> None:
    """Set config timestamp when supported by the system object."""
    if hasattr(system, "config_fetched_at"):
        system.config_fetched_at = fetched_at


def clear_system_box_cache(system: Any) -> None:
    """Clear system-side box cache and reset fetched timestamp."""
    cache = get_system_box_cache(system)
    if cache is not None:
        cache.clear()
    set_system_config_fetched_at(system, None)


def replace_system_box_cache(system: Any, box_configs: dict[str, Any]) -> None:
    """Replace system-side box cache and refresh fetched timestamp."""
    cache = get_system_box_cache(system)
    if cache is None:
        return

    cache.clear()
    for box_name, box_config in box_configs.items():
        cache[box_name] = deepcopy(box_config)

    set_system_config_fetched_at(system, datetime.now() if cache else None)


def update_system_box_cache(system: Any, box_configs: dict[str, Any]) -> None:
    """Update system-side box cache and refresh fetched timestamp."""
    cache = get_system_box_cache(system)
    if cache is None:
        return

    for box_name, box_config in box_configs.items():
        cache[box_name] = deepcopy(box_config)

    if cache:
        set_system_config_fetched_at(system, datetime.now())


def add_executor_command(qubecalib: Any, sequencer: Any) -> None:
    """Add a sequencer command via modern executor or legacy _executor."""
    executor = getattr(qubecalib, "executor", None)
    if executor is None:
        executor = qubecalib._executor
    executor.add_command(sequencer)
