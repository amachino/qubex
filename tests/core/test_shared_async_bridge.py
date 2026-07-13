"""Tests for shared async bridge helper behavior."""

from __future__ import annotations

import time

from qubex.core.async_bridge import (
    close_shared_async_bridge,
    get_shared_async_bridge,
)


def test_shared_async_bridge_reuses_instance_for_same_key() -> None:
    """Given same key, when requesting shared bridge twice, then instance is reused."""
    key = f"shared-bridge-{time.monotonic_ns()}"
    try:
        bridge_a = get_shared_async_bridge(
            key=key,
            thread_name=f"{key}-thread",
        )
        bridge_b = get_shared_async_bridge(
            key=key,
            thread_name=f"{key}-thread",
        )
        assert bridge_a is bridge_b
    finally:
        close_shared_async_bridge(key=key)


def test_shared_async_bridge_recreates_closed_instance() -> None:
    """Given closed shared bridge, when requesting again, then new instance is created."""
    key = f"closed-bridge-{time.monotonic_ns()}"
    bridge_a = get_shared_async_bridge(
        key=key,
        thread_name=f"{key}-thread-a",
    )
    bridge_a.close()
    try:
        bridge_b = get_shared_async_bridge(
            key=key,
            thread_name=f"{key}-thread-b",
        )
        assert bridge_b is not bridge_a
        assert not bridge_b.closed
    finally:
        close_shared_async_bridge(key=key)


def test_shared_async_bridge_uses_different_instance_for_different_key() -> None:
    """Given different keys, when requesting shared bridges, then instances differ."""
    key_a = f"shared-a-{time.monotonic_ns()}"
    key_b = f"shared-b-{time.monotonic_ns()}"
    try:
        bridge_a = get_shared_async_bridge(
            key=key_a,
            thread_name=f"{key_a}-thread",
        )
        bridge_b = get_shared_async_bridge(
            key=key_b,
            thread_name=f"{key_b}-thread",
        )
        assert bridge_a is not bridge_b
    finally:
        close_shared_async_bridge(key=key_a)
        close_shared_async_bridge(key=key_b)
