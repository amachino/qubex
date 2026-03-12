"""Helpers for handling deprecated keyword aliases in contrib experiment APIs."""

from __future__ import annotations

from typing import Any

from qubex._deprecated_options import (
    DeprecatedOptionSpec,
    normalize_deprecated_options,
)


def resolve_shot_options(
    *,
    n_shots: int | None,
    shot_interval: float | None,
    deprecated_options: dict[str, Any],
    function_name: str,
) -> tuple[int | None, float | None]:
    """Resolve deprecated shot option aliases for contrib APIs."""
    normalized = normalize_deprecated_options(
        values={
            "n_shots": n_shots,
            "shot_interval": shot_interval,
        },
        deprecated_options=deprecated_options,
        specs=(
            DeprecatedOptionSpec(
                "shots",
                "n_shots",
                warning_message=(
                    f"`shots` is deprecated in `{function_name}`; use `n_shots`."
                ),
            ),
            DeprecatedOptionSpec(
                "interval",
                "shot_interval",
                warning_message=(
                    "`interval` is deprecated in "
                    f"`{function_name}`; use `shot_interval`."
                ),
            ),
        ),
        stacklevel=4,
    )

    return normalized["n_shots"], normalized["shot_interval"]
