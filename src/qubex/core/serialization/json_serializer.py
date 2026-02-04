"""JSON serialization utilities for core models."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypeGuard

import numpy as np
import tunits
from google.protobuf.json_format import MessageToDict, ParseDict
from tunits.proto import tunits_pb2

from .constants import (
    DATA_COMPLEX_IMAG_KEY,
    DATA_COMPLEX_REAL_KEY,
    DATA_NUMPY_PREFIX,
    DATA_PYTHON_PREFIX,
    DATA_TUNITS_PREFIX,
    DATA_TYPE_KEY,
)

_logger = logging.getLogger(__name__)


def is_numpy(value: Any) -> TypeGuard[np.ndarray | np.generic]:
    """Return whether `value` is a NumPy scalar or array."""
    return isinstance(value, (np.ndarray, np.generic))


def is_tunits(value: Any) -> TypeGuard[tunits.Value | tunits.ValueArray]:
    """Return whether `value` is a tunits value or value-array."""
    return isinstance(value, (tunits.Value, tunits.ValueArray))


def is_complex(value: Any) -> TypeGuard[complex]:
    """Return whether `value` is a complex scalar."""
    return isinstance(value, complex)


def is_custom_serializable_class(cls: type[Any]) -> bool:
    """Return whether class is one of the custom-serialized value types."""
    return issubclass(
        cls,
        (
            complex,
            np.ndarray,
            np.generic,
            tunits.Value,
            tunits.ValueArray,
        ),
    )


def serialize_complex(value: complex) -> dict[str, Any]:
    """Serialize a Python complex value."""
    class_name = value.__class__.__name__
    return {
        DATA_COMPLEX_REAL_KEY: float(value.real),
        DATA_COMPLEX_IMAG_KEY: float(value.imag),
        DATA_TYPE_KEY: f"{DATA_PYTHON_PREFIX}{class_name}",
    }


def serialize_tunits(value: tunits.Value | tunits.ValueArray) -> dict[str, Any]:
    """Serialize a tunits value to a JSON-compatible dict via protobuf."""
    class_name = value.__class__.__name__
    message = value.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[DATA_TYPE_KEY] = f"{DATA_TUNITS_PREFIX}{class_name}"
    return data


def serialize_numpy(value: np.ndarray | np.generic) -> dict[str, Any]:
    """Serialize a NumPy value to a JSON-compatible dict via tunits protobuf."""
    if isinstance(value, np.ndarray):
        value_tunits = tunits.ValueArray(value)
    else:
        value_tunits = tunits.Value(value.item())
    class_name = value.__class__.__name__
    message = value_tunits.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[DATA_TYPE_KEY] = f"{DATA_NUMPY_PREFIX}{class_name}"
    return data


def deserialize_numpy(payload: dict[str, Any]) -> np.ndarray | np.generic:
    """Deserialize a NumPy payload."""
    data = dict(payload)
    type_name: str = data.pop(DATA_TYPE_KEY)
    class_name = type_name.removeprefix(DATA_NUMPY_PREFIX)
    try:
        cls = getattr(np, class_name)
    except AttributeError as exc:
        raise TypeError(f"Unknown numpy class: {class_name}") from exc
    if not isinstance(cls, type):
        raise TypeError(f"Unknown numpy class: {class_name}")
    if issubclass(cls, np.ndarray):
        message = ParseDict(data, tunits_pb2.ValueArray())
        value_tunits = tunits.ValueArray.from_proto(message)
        return value_tunits.value
    if issubclass(cls, np.generic):
        message = ParseDict(data, tunits_pb2.Value())
        value_tunits = tunits.Value.from_proto(message)
        return cls(value_tunits.value)
    raise TypeError(f"Unknown numpy class: {class_name}")


def deserialize_tunits(payload: dict[str, Any]) -> tunits.Value | tunits.ValueArray:
    """Deserialize a tunits payload."""
    data = dict(payload)
    type_name: str = data.pop(DATA_TYPE_KEY)
    class_name = type_name.removeprefix(DATA_TUNITS_PREFIX)
    try:
        cls = getattr(tunits, class_name)
    except AttributeError as exc:
        raise TypeError(f"Unknown tunits class: {class_name}") from exc
    if not isinstance(cls, type):
        raise TypeError(f"Unknown tunits class: {class_name}")
    if issubclass(cls, tunits.Value):
        message = ParseDict(data, tunits_pb2.Value())
        return cls.from_proto(message)
    if issubclass(cls, tunits.ValueArray):
        message = ParseDict(data, tunits_pb2.ValueArray())
        return cls.from_proto(message)
    raise TypeError(f"Unknown tunits class: {class_name}")


def deserialize_complex(payload: dict[str, Any]) -> complex:
    """Deserialize a Python complex payload."""
    data = dict(payload)
    type_name: str = data.pop(DATA_TYPE_KEY)
    class_name = type_name.removeprefix(DATA_PYTHON_PREFIX)
    if class_name == "complex":
        return complex(data[DATA_COMPLEX_REAL_KEY], data[DATA_COMPLEX_IMAG_KEY])
    raise TypeError(f"Unknown complex class: {type_name}")


def serialize_value(value: Any) -> Any:
    """Serialize nested values with NumPy/tunits/complex support."""
    if is_numpy(value):
        return serialize_numpy(value)
    if is_complex(value):
        return serialize_complex(value)
    if is_tunits(value):
        return serialize_tunits(value)
    if isinstance(value, Mapping):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    return value


def deserialize_value(value: Any) -> Any:
    """Deserialize nested values serialized by `serialize_value`."""
    if isinstance(value, Mapping):
        type_tag = value.get(DATA_TYPE_KEY)
        if isinstance(type_tag, str):
            payload = dict(value)
            if type_tag.startswith(DATA_NUMPY_PREFIX):
                return deserialize_numpy(payload)
            if type_tag.startswith(DATA_TUNITS_PREFIX):
                return deserialize_tunits(payload)
            if type_tag.startswith(DATA_PYTHON_PREFIX):
                return deserialize_complex(payload)
            _logger.warning(f"Unknown type during deserialization: {type_tag}")
        return {k: deserialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [deserialize_value(v) for v in value]
    return value
