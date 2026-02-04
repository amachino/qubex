"""NetCDF-specific serialization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import tunits

from .constants import DATA_TYPE_KEY, TYPE_TUNITS_VALUE_ARRAY
from .json_serializer import deserialize_tunits


def deserialize_tunits_value_array(
    *,
    units: list[dict[str, Any]],
    values: np.ndarray,
) -> tunits.ValueArray:
    """Deserialize a tunits.ValueArray from units metadata and numeric values."""
    payload: dict[str, Any] = {
        DATA_TYPE_KEY: TYPE_TUNITS_VALUE_ARRAY,
        "units": units,
        "shape": list(values.shape),
    }

    flat = np.asarray(values).reshape(-1)
    if np.iscomplexobj(flat):
        payload["complexes"] = {
            "values": [
                {"real": float(item.real), "imaginary": float(item.imag)}
                for item in flat
            ]
        }
    else:
        payload["reals"] = {"values": [float(item) for item in flat]}

    restored = deserialize_tunits(payload)
    if isinstance(restored, tunits.ValueArray):
        return restored
    raise TypeError("Failed to deserialize tunits.ValueArray.")
