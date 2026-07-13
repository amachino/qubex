"""Script version of the AsyncBridge notebook example."""

from __future__ import annotations

import asyncio
import threading

from qubex.core import AsyncBridge


async def delayed_print(message: str, delay: float) -> None:
    """Sleep and print one message."""
    await asyncio.sleep(delay)
    print(message)


async def delayed_sum(x: int, y: int, delay: float) -> int:
    """Sleep and return sum of two integers."""
    await asyncio.sleep(delay)
    return x + y


async def report_thread() -> str:
    """Return current thread name after a short delay."""
    await asyncio.sleep(0.1)
    return threading.current_thread().name


def run_sync_bridge_example() -> None:
    """Run an awaitable from synchronous code."""
    print("## 1) Run an awaitable from synchronous code")
    with AsyncBridge() as bridge:
        bridge.run(lambda: delayed_print("hello", delay=2.0))


def run_return_value_example() -> None:
    """Run awaitable and consume its return value."""
    print("## 2) Return a value")
    with AsyncBridge() as bridge:
        result = bridge.run(lambda: delayed_sum(2, 3, delay=1.0))
    print(f"result = {result}")


async def run_timeout_example() -> None:
    """Show timeout handling during bridge execution."""
    print("## 3) Handle timeout")
    with AsyncBridge(default_timeout=0.3) as bridge:
        try:
            bridge.run(lambda: delayed_print("this will timeout", delay=1.0))
        except TimeoutError as exc:
            print(f"Caught expected timeout: {exc}")


async def run_inside_running_loop_example() -> None:
    """Call bridge from inside a running event loop."""
    print("## 4) Call from inside a running event loop")
    with AsyncBridge(thread_name="async-bridge") as bridge:
        caller_thread = threading.current_thread().name
        bridge_thread = bridge.run(lambda: report_thread())
        print(f"caller thread: {caller_thread}")
        print(f"bridge thread: {bridge_thread}")


def main() -> None:
    """Run all AsyncBridge script examples in order."""
    run_sync_bridge_example()
    run_return_value_example()
    asyncio.run(run_timeout_example())
    asyncio.run(run_inside_running_loop_example())


if __name__ == "__main__":
    main()
