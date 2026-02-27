"""Helpers for handling deprecated keyword aliases in contrib experiment APIs."""

from __future__ import annotations

import warnings
from typing import Any


def resolve_shot_options(
    *,
    n_shots: int | None,
    shot_interval: float | None,
    deprecated_options: dict[str, Any],
    function_name: str,
) -> tuple[int | None, float | None]:
    """Resolve deprecated shot option aliases for contrib APIs."""
    if "shots" in deprecated_options:
        warnings.warn(
            f"`shots` is deprecated in `{function_name}`; use `n_shots`.",
            DeprecationWarning,
            stacklevel=3,
        )
        if n_shots is None:
            n_shots = deprecated_options.pop("shots")
        else:
            deprecated_options.pop("shots")

    if "interval" in deprecated_options:
        warnings.warn(
            f"`interval` is deprecated in `{function_name}`; use `shot_interval`.",
            DeprecationWarning,
            stacklevel=3,
        )
        if shot_interval is None:
            shot_interval = deprecated_options.pop("interval")
        else:
            deprecated_options.pop("interval")

    if deprecated_options:
        unknown = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unknown}")

    return n_shots, shot_interval
