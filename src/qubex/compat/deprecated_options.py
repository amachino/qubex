"""Helpers for normalizing deprecated keyword options."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from qubex.core.sentinel import MISSING


@dataclass(frozen=True)
class DeprecatedOptionSpec:
    """Describe one deprecated keyword normalization rule."""

    deprecated_name: str
    replacement_name: str
    warning_message: str | None = None
    default: Any = MISSING


def resolve_deprecated_option(
    *,
    value: Any,
    deprecated_options: dict[str, Any],
    deprecated_name: str,
    replacement_name: str,
    default: Any = MISSING,
    warning_message: str | None = None,
    stacklevel: int = 3,
) -> Any:
    """Resolve one deprecated keyword alias and return the normalized value."""
    legacy_value = deprecated_options.pop(deprecated_name, MISSING)
    if legacy_value is not MISSING and legacy_value is not None:
        warnings.warn(
            (
                warning_message
                if warning_message is not None
                else f"`{deprecated_name}` is deprecated; use `{replacement_name}`."
            ),
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        if value is not None and value != legacy_value:
            raise ValueError(
                f"`{deprecated_name}` conflicts with `{replacement_name}`. "
                f"Provide only `{replacement_name}`."
            )
        return legacy_value

    if value is None and default is not MISSING:
        return default

    return value


def normalize_deprecated_options(
    *,
    values: dict[str, Any],
    deprecated_options: dict[str, Any],
    specs: tuple[DeprecatedOptionSpec, ...] | list[DeprecatedOptionSpec],
    stacklevel: int = 3,
) -> dict[str, Any]:
    """Resolve deprecated keyword aliases and reject unexpected leftovers."""
    normalized, remaining = partition_deprecated_options(
        values=values,
        deprecated_options=deprecated_options,
        specs=specs,
        stacklevel=stacklevel,
    )

    if remaining:
        joined = ", ".join(f"`{key}`" for key in sorted(remaining))
        raise TypeError(f"Unexpected keyword argument(s): {joined}")

    return normalized


def partition_deprecated_options(
    *,
    values: dict[str, Any],
    deprecated_options: dict[str, Any],
    specs: tuple[DeprecatedOptionSpec, ...] | list[DeprecatedOptionSpec],
    stacklevel: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve deprecated keyword aliases and return unresolved leftovers."""
    remaining = dict(deprecated_options)
    normalized = dict(values)

    for spec in specs:
        normalized[spec.replacement_name] = resolve_deprecated_option(
            value=normalized.get(spec.replacement_name),
            deprecated_options=remaining,
            deprecated_name=spec.deprecated_name,
            replacement_name=spec.replacement_name,
            default=spec.default,
            warning_message=spec.warning_message,
            stacklevel=stacklevel,
        )

    return normalized, remaining
