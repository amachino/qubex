"""Helpers for normalizing tunits quantities to scalar runtime units."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar, cast

from qxcore import Frequency, Time


class SupportsValueInBaseUnits(Protocol):
    """Protocol for quantity types exposing value in base units."""

    def value_in_base_units(self) -> float:
        """Return value converted to base units."""
        ...


Q = TypeVar("Q", bound=SupportsValueInBaseUnits)
Number = float | int


def normalize_quantity(
    value: Number | Q,
    *,
    quantity_type: type[Q],
    scale_from_base: float,
) -> float:
    """Normalize a scalar quantity to float using base-unit scaling."""
    if isinstance(value, quantity_type):
        return float(value.value_in_base_units() * scale_from_base)
    return float(cast(Number, value))


def normalize_quantity_mapping(
    values: Mapping[str, Number | Q] | None,
    *,
    quantity_type: type[Q],
    scale_from_base: float,
) -> dict[str, float] | None:
    """Normalize a mapping of scalar quantities to float."""
    if values is None:
        return None
    return {
        label: normalize_quantity(
            value,
            quantity_type=quantity_type,
            scale_from_base=scale_from_base,
        )
        for label, value in values.items()
    }


def normalize_time_to_ns(value: Number | Time | None) -> float | None:
    """Normalize time value to ns as float."""
    if value is None:
        return None
    return normalize_quantity(
        value,
        quantity_type=Time,
        scale_from_base=1e9,
    )


def normalize_frequency_to_ghz(value: Number | Frequency) -> float:
    """Normalize frequency value to GHz as float."""
    return normalize_quantity(
        value,
        quantity_type=Frequency,
        scale_from_base=1e-9,
    )


def normalize_frequencies_to_ghz(
    values: Mapping[str, Number | Frequency] | None,
) -> dict[str, float] | None:
    """Normalize frequency mapping to GHz floats."""
    return normalize_quantity_mapping(
        values,
        quantity_type=Frequency,
        scale_from_base=1e-9,
    )
