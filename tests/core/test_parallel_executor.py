"""Tests for core parallel executor helpers."""

from __future__ import annotations

import time

from qubex.core.parallel_executor import run_parallel


def test_run_parallel_handles_errors_in_completion_order() -> None:
    """Given mixed worker durations, when run_parallel handles errors, then callback order follows completion."""
    handled_items: list[str] = []

    def _worker(item: str) -> None:
        if item == "slow_ok":
            time.sleep(0.05)
            return
        if item == "slow_fail":
            time.sleep(0.03)
            raise RuntimeError("slow failure")
        if item == "fast_fail":
            raise RuntimeError("fast failure")
        raise AssertionError(f"Unexpected item: {item}")

    def _on_error(item: str, exc: BaseException) -> None:
        del exc
        handled_items.append(item)

    run_parallel(
        ["slow_ok", "slow_fail", "fast_fail"],
        _worker,
        on_error=_on_error,
        max_workers=3,
    )

    assert handled_items == ["fast_fail", "slow_fail"]
