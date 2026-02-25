"""Tests for public async bridge behavior."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Generator

import pytest

from qubex.core.async_bridge import AsyncBridge


async def _return_value(value: int) -> int:
    return value


@pytest.fixture
def bridge() -> Generator[AsyncBridge, None, None]:
    """Given bridge fixture, when test finishes, then bridge loop is closed."""
    async_bridge = AsyncBridge(
        default_timeout=1.0,
        startup_timeout=1.0,
        thread_name="qubex-test-async-bridge",
    )
    try:
        yield async_bridge
    finally:
        async_bridge.close()


def test_run_without_running_loop_returns_result(bridge: AsyncBridge) -> None:
    """Given no active loop, when running async, then it returns coroutine result."""
    result = bridge.run(lambda: _return_value(7))

    assert result == 7


def test_run_inside_running_loop_preserves_contextvars(bridge: AsyncBridge) -> None:
    """Given active loop with contextvar, when running async, then copied context is visible."""
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="")

    async def _read_marker() -> str:
        return marker.get()

    async def _invoke() -> str:
        marker.set("captured")
        return bridge.run(lambda: _read_marker(), timeout=1.0)

    result = asyncio.run(_invoke())

    assert result == "captured"


def test_run_inside_running_loop_cancels_on_timeout(bridge: AsyncBridge) -> None:
    """Given active loop and timeout, when async hangs, then bridge cancels coroutine."""
    cancelled = threading.Event()

    async def _hang_forever() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def _invoke() -> None:
        with pytest.raises(TimeoutError):
            bridge.run(lambda: _hang_forever(), timeout=0.01)

    asyncio.run(_invoke())

    assert cancelled.wait(timeout=1.0)


def test_run_inside_running_loop_propagates_cancelled_error(
    bridge: AsyncBridge,
) -> None:
    """Given active loop, when coroutine is cancelled, then cancellation propagates."""

    async def _cancelled() -> int:
        raise asyncio.CancelledError

    async def _invoke() -> None:
        with pytest.raises(asyncio.CancelledError):
            bridge.run(lambda: _cancelled(), timeout=1.0)

    asyncio.run(_invoke())


def test_run_after_close_raises_runtime_error() -> None:
    """Given closed bridge, when running async, then runtime error is raised."""
    bridge = AsyncBridge(default_timeout=1.0, startup_timeout=1.0)
    bridge.close()

    with pytest.raises(RuntimeError, match="closed"):
        bridge.run(lambda: _return_value(1))


def test_startup_timeout_requests_stop_and_closes_loop() -> None:
    """Given delayed startup, when startup times out, then bridge requests stop."""
    original_new_event_loop = asyncio.new_event_loop
    thread_name = f"timeout-bridge-{time.monotonic_ns()}"

    def delayed_new_event_loop() -> asyncio.AbstractEventLoop:
        time.sleep(0.05)
        return original_new_event_loop()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(asyncio, "new_event_loop", delayed_new_event_loop)
        with pytest.raises(RuntimeError, match="Failed to start AsyncBridge"):
            AsyncBridge(startup_timeout=0.01, thread_name=thread_name)

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not any(
            thread.name == thread_name and thread.is_alive()
            for thread in threading.enumerate()
        ):
            break
        time.sleep(0.01)

    assert not any(
        thread.name == thread_name and thread.is_alive()
        for thread in threading.enumerate()
    )
