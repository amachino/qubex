"""Tests for public async bridge behavior."""

from __future__ import annotations

import asyncio
import contextvars
import signal
import threading
import time
from collections.abc import Generator
from types import FrameType

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


@pytest.mark.skipif(
    not hasattr(signal, "pthread_kill") or not hasattr(signal, "SIGUSR1"),
    reason="requires thread-directed signal support",
)
def test_run_inside_running_loop_cancels_on_keyboard_interrupt(
    bridge: AsyncBridge,
) -> None:
    """Given an interrupted wait, bridge should cancel its background task."""
    started = threading.Event()
    cancelled = threading.Event()

    async def _hang_forever() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def _raise_keyboard_interrupt(
        signal_number: int,
        frame: FrameType | None,
    ) -> None:
        del signal_number, frame
        raise KeyboardInterrupt

    main_thread_id = threading.get_ident()

    def _interrupt_after_start() -> None:
        assert started.wait(timeout=10.0)
        signal.pthread_kill(main_thread_id, signal.SIGUSR1)

    async def _invoke() -> None:
        with pytest.raises(KeyboardInterrupt):
            bridge.run(lambda: _hang_forever(), timeout=10.0)

    previous_handler = signal.signal(signal.SIGUSR1, _raise_keyboard_interrupt)
    try:
        interrupter = threading.Thread(target=_interrupt_after_start)
        interrupter.start()
        try:
            asyncio.run(_invoke())
        finally:
            interrupter.join(timeout=1.0)
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)

    assert not interrupter.is_alive()
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
    startup_gate = threading.Event()

    def delayed_new_event_loop() -> asyncio.AbstractEventLoop:
        startup_gate.wait(timeout=0.05)
        return original_new_event_loop()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(asyncio, "new_event_loop", delayed_new_event_loop)
        with pytest.raises(RuntimeError, match="Failed to start AsyncBridge"):
            AsyncBridge(startup_timeout=0.01, thread_name=thread_name)

    lingering_thread = next(
        (
            thread
            for thread in threading.enumerate()
            if thread.name == thread_name and thread.is_alive()
        ),
        None,
    )
    if lingering_thread is not None:
        lingering_thread.join(timeout=1.0)

    assert not any(
        thread.name == thread_name and thread.is_alive()
        for thread in threading.enumerate()
    )
