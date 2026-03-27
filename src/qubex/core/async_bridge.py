"""Async bridge utilities for synchronous API boundaries."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import threading
from collections.abc import Awaitable, Callable
from typing import Final, TypeVar

T = TypeVar("T")

DEFAULT_TIMEOUT_SECONDS: Final[float] = 300.0
DEFAULT_STARTUP_TIMEOUT_SECONDS: Final[float] = 5.0
THREAD_JOIN_TIMEOUT_SECONDS: Final[float] = 1.0
DEFAULT_SHARED_THREAD_PREFIX: Final[str] = "qx-async-bridge"

_SHARED_BRIDGE_LOCK = threading.Lock()
_SHARED_BRIDGES: dict[str, AsyncBridge] = {}


async def _invoke_factory(factory: Callable[[], Awaitable[T]]) -> T:
    """Invoke one awaitable factory as a coroutine."""
    return await factory()


async def _await_result(awaitable: Awaitable[T]) -> T:
    """Await a generic awaitable in direct-run mode."""
    return await awaitable


class AsyncBridge:
    """
    Run awaitable factories from synchronous code safely.

    Examples
    --------
    >>> import asyncio
    >>> from qubex.core.async_bridge import AsyncBridge
    >>> async def delayed_print(message: str) -> None:
    ...     await asyncio.sleep(1.0)
    ...     print(message)
    >>> with AsyncBridge(default_timeout=2.0) as bridge:
    ...     bridge.run(lambda: delayed_print("hello"))
    hello
    """

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        thread_name: str = "qubex-async-bridge",
    ) -> None:
        """
        Initialize async bridge with one dedicated event-loop thread.

        Parameters
        ----------
        default_timeout : float, optional
            Default timeout in seconds for blocking bridge calls.
        startup_timeout : float, optional
            Timeout in seconds while waiting for bridge-loop startup.
        thread_name : str, optional
            Name of the dedicated bridge thread.
        """
        self._default_timeout = default_timeout
        self._startup_timeout = startup_timeout
        self._thread_name = thread_name
        self._state_lock = threading.Lock()
        self._loop_ready = threading.Event()
        self._stop_requested = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._startup_error: BaseException | None = None
        self._start_loop_thread()

    @property
    def default_timeout(self) -> float:
        """Return default bridge timeout in seconds."""
        return self._default_timeout

    @property
    def closed(self) -> bool:
        """Return whether the bridge is closed."""
        with self._state_lock:
            return self._closed

    def run(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        timeout: float | None = None,
    ) -> T:
        """
        Run one awaitable factory from synchronous code.

        Parameters
        ----------
        factory : Callable[[], Awaitable[T]]
            Awaitable factory to execute.
        timeout : float | None, optional
            Timeout in seconds for bridge execution when a loop is already active.

        Returns
        -------
        T
            Resolved value from the awaitable factory.

        Examples
        --------
        >>> import asyncio
        >>> from qubex.core.async_bridge import AsyncBridge
        >>> async def delayed_message(message: str) -> str:
        ...     await asyncio.sleep(1.0)
        ...     return message
        >>> bridge = AsyncBridge(default_timeout=2.0)
        >>> bridge.run(lambda: delayed_message("done"), timeout=1.5)
        'done'
        >>> bridge.close()
        """
        self._ensure_not_closed()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_await_result(factory()))

        if threading.current_thread() is self._thread:
            raise RuntimeError(
                "AsyncBridge.run cannot block from its own event-loop thread."
            )

        bridge_future = self._submit(
            factory=factory,
            context=contextvars.copy_context(),
        )
        wait_timeout = self._default_timeout if timeout is None else timeout
        try:
            return bridge_future.result(timeout=wait_timeout)
        except concurrent.futures.TimeoutError as error:
            bridge_future.cancel()
            raise TimeoutError(
                "Timed out while waiting for asynchronous bridge execution."
            ) from error

    def close(self) -> None:
        """Stop the dedicated bridge loop and close this bridge."""
        self._stop_requested.set()
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread

        if thread is None:
            return

        if loop is not None and not loop.is_closed():
            if threading.current_thread() is thread:
                loop.stop()
                return
            loop.call_soon_threadsafe(loop.stop)

        if threading.current_thread() is thread:
            return

        thread.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)

    def _submit(
        self,
        *,
        factory: Callable[[], Awaitable[T]],
        context: contextvars.Context,
    ) -> concurrent.futures.Future[T]:
        """Submit one awaitable factory to the dedicated bridge loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("AsyncBridge event loop is unavailable.")

        bridge_future: concurrent.futures.Future[T] = concurrent.futures.Future()
        task_holder: dict[str, asyncio.Task[T]] = {}

        def _on_task_done(task: asyncio.Task[T]) -> None:
            if bridge_future.done():
                return
            if task.cancelled():
                bridge_future.set_exception(asyncio.CancelledError())
                return
            error = task.exception()
            if error is not None:
                bridge_future.set_exception(error)
                return
            bridge_future.set_result(task.result())

        def _schedule_task() -> None:
            if bridge_future.cancelled():
                return
            if self.closed:
                bridge_future.set_exception(
                    RuntimeError("AsyncBridge is already closed.")
                )
                return
            try:
                task = context.run(loop.create_task, _invoke_factory(factory))
            except BaseException as error:  # pragma: no cover - passthrough guard
                if not bridge_future.done():
                    bridge_future.set_exception(error)
                return
            task_holder["task"] = task
            task.add_done_callback(_on_task_done)

        def _cancel_task() -> None:
            task = task_holder.get("task")
            if task is not None and not task.done():
                task.cancel()

        def _on_bridge_done(future: concurrent.futures.Future[T]) -> None:
            if not future.cancelled():
                return
            if loop.is_closed():  # pragma: no cover - defensive guard
                return
            loop.call_soon_threadsafe(_cancel_task)

        bridge_future.add_done_callback(_on_bridge_done)
        loop.call_soon_threadsafe(_schedule_task)
        return bridge_future

    def _ensure_not_closed(self) -> None:
        """Raise runtime error when bridge has already been closed."""
        if self.closed:
            raise RuntimeError("AsyncBridge is already closed.")

    def _start_loop_thread(self) -> None:
        """Start dedicated event-loop thread and wait for readiness."""
        thread = threading.Thread(
            target=self._run_loop,
            name=self._thread_name,
            daemon=True,
        )
        self._thread = thread
        thread.start()
        if not self._loop_ready.wait(timeout=self._startup_timeout):
            self.close()
            raise RuntimeError("Failed to start AsyncBridge event loop.")
        if self._startup_error is not None:
            startup_error = self._startup_error
            self.close()
            raise RuntimeError(
                "Failed to start AsyncBridge event loop."
            ) from startup_error

    def _run_loop(self) -> None:
        """Own dedicated event-loop lifecycle."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        if self._stop_requested.is_set():
            self._loop_ready.set()
            loop.close()
            return
        # Signal readiness only after the loop starts serving callbacks.
        loop.call_soon(self._loop_ready.set)
        try:
            loop.run_forever()
        except BaseException as error:  # pragma: no cover - fatal guard
            self._startup_error = error
            raise
        finally:
            pending_tasks = asyncio.all_tasks(loop=loop)
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                loop.run_until_complete(
                    asyncio.gather(*pending_tasks, return_exceptions=True)
                )
            loop.close()

    def __enter__(self) -> AsyncBridge:
        """Return bridge as context manager resource."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close bridge when leaving context manager scope."""
        self.close()


def get_shared_async_bridge(
    *,
    key: str,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    thread_name: str | None = None,
) -> AsyncBridge:
    """
    Return one process-wide shared async bridge for the given key.

    Parameters
    ----------
    key : str
        Stable key used to identify one shared bridge instance.
    default_timeout : float, optional
        Default timeout in seconds for bridge calls.
    startup_timeout : float, optional
        Startup timeout in seconds for newly created bridge loops.
    thread_name : str | None, optional
        Dedicated bridge thread name. Defaults to
        ``f"{DEFAULT_SHARED_THREAD_PREFIX}-{key}"``.

    Examples
    --------
    >>> from qubex.core.async_bridge import (
    ...     close_shared_async_bridge,
    ...     get_shared_async_bridge,
    ... )
    >>> async def _sample() -> int:
    ...     return 1
    >>> bridge = get_shared_async_bridge(key="measurement")
    >>> result = bridge.run(lambda: _sample())
    >>> result
    1
    >>> close_shared_async_bridge(key="measurement")
    """
    with _SHARED_BRIDGE_LOCK:
        bridge = _SHARED_BRIDGES.get(key)
        if bridge is None or bridge.closed:
            resolved_thread_name = (
                f"{DEFAULT_SHARED_THREAD_PREFIX}-{key}"
                if thread_name is None
                else thread_name
            )
            bridge = AsyncBridge(
                default_timeout=default_timeout,
                startup_timeout=startup_timeout,
                thread_name=resolved_thread_name,
            )
            _SHARED_BRIDGES[key] = bridge
        return bridge


def close_shared_async_bridge(*, key: str) -> None:
    """
    Close one shared async bridge identified by key when it exists.

    Examples
    --------
    >>> from qubex.core.async_bridge import (
    ...     close_shared_async_bridge,
    ...     get_shared_async_bridge,
    ... )
    >>> _ = get_shared_async_bridge(key="temporary")
    >>> close_shared_async_bridge(key="temporary")
    """
    with _SHARED_BRIDGE_LOCK:
        bridge = _SHARED_BRIDGES.pop(key, None)
    if bridge is not None:
        bridge.close()
