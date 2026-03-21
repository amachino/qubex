"""Tests for core parallel executor helpers."""

from __future__ import annotations

from threading import Event

from qubex.core.parallel_executor import run_parallel


def test_run_parallel_handles_errors_in_completion_order() -> None:
    """Given mixed worker durations, when run_parallel handles errors, then callback order follows completion."""
    handled_items: list[str] = []
    allow_slow_fail = Event()
    allow_slow_ok = Event()

    def _worker(item: str) -> None:
        if item == "slow_ok":
            if not allow_slow_ok.wait(timeout=1.0):
                raise TimeoutError("slow_ok was not released")
            return
        if item == "slow_fail":
            if not allow_slow_fail.wait(timeout=1.0):
                raise TimeoutError("slow_fail was not released")
            raise RuntimeError("slow failure")
        if item == "fast_fail":
            raise RuntimeError("fast failure")
        raise AssertionError(f"Unexpected item: {item}")

    def _on_error(item: str, exc: BaseException) -> None:
        del exc
        handled_items.append(item)
        if item == "fast_fail":
            allow_slow_fail.set()
        if item == "slow_fail":
            allow_slow_ok.set()

    run_parallel(
        ["slow_ok", "slow_fail", "fast_fail"],
        _worker,
        on_error=_on_error,
        max_workers=3,
    )

    assert handled_items == ["fast_fail", "slow_fail"]
