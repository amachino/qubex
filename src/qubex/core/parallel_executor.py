"""Helpers for generic parallel execution with `ThreadPoolExecutor`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

DEFAULT_MAX_WORKERS = 32


def run_parallel(
    items: Sequence[T],
    worker: Callable[[T], object],
    *,
    max_workers: int | None = None,
    on_error: Callable[[T, BaseException], None] | None = None,
) -> None:
    """
    Execute `worker` for each item in parallel.

    Parameters
    ----------
    items : Sequence[T]
        Items to process.
    worker : Callable[[T], object]
        Worker function called once per item.
    max_workers : int | None, optional
        Maximum number of worker threads. If `None`, uses
        `DEFAULT_MAX_WORKERS`.
    on_error : Callable[[T, BaseException], None] | None, optional
        Optional per-item error handler. If omitted, exceptions are re-raised.
    """
    if not items:
        return
    resolved_max_workers = DEFAULT_MAX_WORKERS if max_workers is None else max_workers
    if resolved_max_workers < 1:
        raise ValueError("`max_workers` must be at least 1.")
    workers = min(resolved_max_workers, len(items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(worker, item): item for item in items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                future.result()
            except BaseException as exc:
                if on_error is None:
                    raise
                on_error(item, exc)


def run_parallel_map(
    items: Sequence[T],
    worker: Callable[[T], V],
    *,
    key: Callable[[T], K],
    max_workers: int | None = None,
    as_completed_order: bool = False,
    on_error: Callable[[T, BaseException], V] | None = None,
) -> dict[K, V]:
    """
    Execute `worker` for each item in parallel and collect results by key.

    Parameters
    ----------
    items : Sequence[T]
        Items to process.
    worker : Callable[[T], V]
        Worker function called once per item.
    key : Callable[[T], K]
        Function to derive output key from an item.
    max_workers : int | None, optional
        Maximum number of worker threads. If `None`, uses
        `DEFAULT_MAX_WORKERS`.
    as_completed_order : bool, optional
        If `True`, consume futures in completion order.
    on_error : Callable[[T, BaseException], V] | None, optional
        Optional per-item fallback result provider on errors. If omitted,
        exceptions are re-raised.

    Returns
    -------
    dict[K, V]
        Mapping from item key to worker result.
    """
    if not items:
        return {}
    resolved_max_workers = DEFAULT_MAX_WORKERS if max_workers is None else max_workers
    if resolved_max_workers < 1:
        raise ValueError("`max_workers` must be at least 1.")
    workers = min(resolved_max_workers, len(items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(worker, item): item for item in items}
        futures = (
            as_completed(future_to_item)
            if as_completed_order
            else tuple(future_to_item.keys())
        )
        results: dict[K, V] = {}
        for future in futures:
            item = future_to_item[future]
            try:
                results[key(item)] = future.result()
            except BaseException as exc:
                if on_error is None:
                    raise
                results[key(item)] = on_error(item, exc)
        return results
