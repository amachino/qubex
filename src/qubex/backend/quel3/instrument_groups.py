"""Internal alias conventions for grouped QuEL-3 transmitter instruments."""

from __future__ import annotations

from typing import Final

TRANSMITTER_INSTRUMENT_MULTIPLICITY: Final = 4
_TRANSMITTER_SUFFIXES: Final = frozenset(range(TRANSMITTER_INSTRUMENT_MULTIPLICITY))


def build_transmitter_aliases(alias: str) -> tuple[str, ...]:
    """Return the four physical transmitter aliases for one logical alias."""
    return tuple(
        f"{alias}-{index}" for index in range(TRANSMITTER_INSTRUMENT_MULTIPLICITY)
    )


def split_transmitter_alias(alias: str) -> tuple[str, int | None]:
    """Split one terminal transmitter-group suffix from an alias."""
    base_alias, separator, raw_index = alias.rpartition("-")
    if not separator or not base_alias or not raw_index.isdigit():
        return alias, None
    index = int(raw_index)
    if index not in _TRANSMITTER_SUFFIXES:
        return alias, None
    return base_alias, index


def is_transmitter_role(role: object) -> bool:
    """Return whether an enum-like role value identifies a transmitter."""
    role_name = getattr(role, "name", role)
    return str(role_name).rsplit(".", maxsplit=1)[-1].upper() == "TRANSMITTER"
