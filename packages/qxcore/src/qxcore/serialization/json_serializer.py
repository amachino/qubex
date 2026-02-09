"""JSON serialization utilities for core models."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import tunits as _tunits_runtime
from google.protobuf.json_format import MessageToDict, ParseDict

if TYPE_CHECKING:
    import tunits as _tunits

    TunitsValue = _tunits.Value
    TunitsValueArray = _tunits.ValueArray
else:
    TunitsValue = Any
    TunitsValueArray = Any

from .constants import (
    DATA_COMPLEX_IMAG_KEY,
    DATA_COMPLEX_REAL_KEY,
    DATA_NUMPY_PREFIX,
    DATA_PYTHON_PREFIX,
    DATA_TUNITS_PREFIX,
    DATA_TYPE_KEY,
)

_logger = logging.getLogger(__name__)


def to_canonical_json(value: Any) -> str:
    """
    Serialize a value to canonical JSON.

    Parameters
    ----------
    value : Any
        JSON-serializable value.

    Returns
    -------
    str
        Canonical JSON string with stable key ordering and compact separators.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=True,
    )


def is_numpy(value: Any) -> TypeGuard[np.ndarray | np.generic]:
    """
    Return whether `value` is a NumPy scalar or array.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        `True` if the value is a NumPy array or scalar.
    """
    return isinstance(value, (np.ndarray, np.generic))


def is_tunits(value: Any) -> TypeGuard[TunitsValue | TunitsValueArray]:
    """
    Return whether `value` is a tunits value or value-array.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        `True` if the value is a tunits `Value` or `ValueArray`.
    """
    return isinstance(value, (_tunits_runtime.Value, _tunits_runtime.ValueArray))


def is_complex(value: Any) -> TypeGuard[complex]:
    """
    Return whether `value` is a complex scalar.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        `True` if the value is a `complex` instance.
    """
    return isinstance(value, complex)


def is_custom_serializable_class(cls: type[Any]) -> bool:
    """
    Return whether a class is serialized by custom logic.

    Parameters
    ----------
    cls : type[Any]
        Class to inspect.

    Returns
    -------
    bool
        `True` if the class matches a supported custom serialization type.
    """
    if issubclass(cls, (complex, np.ndarray, np.generic)):
        return True
    return issubclass(cls, (_tunits_runtime.Value, _tunits_runtime.ValueArray))


def serialize_complex(value: complex) -> dict[str, Any]:
    """
    Serialize a Python complex value to a JSON-friendly payload.

    Parameters
    ----------
    value : complex
        Complex number to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-compatible payload describing the complex value.
    """
    class_name = value.__class__.__name__
    return {
        DATA_COMPLEX_REAL_KEY: float(value.real),
        DATA_COMPLEX_IMAG_KEY: float(value.imag),
        DATA_TYPE_KEY: f"{DATA_PYTHON_PREFIX}{class_name}",
    }


def serialize_tunits(value: TunitsValue | TunitsValueArray) -> dict[str, Any]:
    """
    Serialize a tunits value to a JSON-compatible dict via protobuf.

    Parameters
    ----------
    value : TunitsValue | TunitsValueArray
        Tunits value or value array to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-compatible payload with tunits metadata.
    """
    class_name = value.__class__.__name__
    message = value.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[DATA_TYPE_KEY] = f"{DATA_TUNITS_PREFIX}{class_name}"
    return data


def serialize_numpy(value: np.ndarray | np.generic) -> dict[str, Any]:
    """
    Serialize a NumPy value to a JSON-compatible dict.

    Parameters
    ----------
    value : np.ndarray | np.generic
        NumPy array or scalar to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-compatible payload describing the NumPy value.
    """
    if isinstance(value, np.ndarray):
        value_tunits = _tunits_runtime.ValueArray(value)
    else:
        value_tunits = _tunits_runtime.Value(value.item())
    class_name = value.__class__.__name__
    message = value_tunits.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[DATA_TYPE_KEY] = f"{DATA_NUMPY_PREFIX}{class_name}"
    return data


def deserialize_numpy(payload: dict[str, Any]) -> np.ndarray | np.generic:
    """
    Deserialize a NumPy payload.

    Parameters
    ----------
    payload : dict[str, Any]
        JSON-compatible NumPy payload.

    Returns
    -------
    np.ndarray | np.generic
        Restored NumPy array or scalar.

    Raises
    ------
    TypeError
        If the payload references an unknown NumPy class.
    """
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
        from tunits.proto import tunits_pb2

        message = ParseDict(data, tunits_pb2.ValueArray())
        value_tunits = _tunits_runtime.ValueArray.from_proto(message)
        return value_tunits.value
    if issubclass(cls, np.generic):
        from tunits.proto import tunits_pb2

        message = ParseDict(data, tunits_pb2.Value())
        value_tunits = _tunits_runtime.Value.from_proto(message)
        return cls(value_tunits.value)
    raise TypeError(f"Unknown numpy class: {class_name}")


def deserialize_tunits(payload: dict[str, Any]) -> TunitsValue | TunitsValueArray:
    """
    Deserialize a tunits payload.

    Parameters
    ----------
    payload : dict[str, Any]
        JSON-compatible tunits payload.

    Returns
    -------
    TunitsValue | TunitsValueArray
        Restored tunits value or value array.

    Raises
    ------
    TypeError
        If the payload references an unknown tunits class.
    """
    data = dict(payload)
    type_name: str = data.pop(DATA_TYPE_KEY)
    class_name = type_name.removeprefix(DATA_TUNITS_PREFIX)
    try:
        cls = getattr(_tunits_runtime, class_name)
    except AttributeError as exc:
        raise TypeError(f"Unknown tunits class: {class_name}") from exc
    if not isinstance(cls, type):
        raise TypeError(f"Unknown tunits class: {class_name}")
    if issubclass(cls, _tunits_runtime.Value):
        from tunits.proto import tunits_pb2

        message = ParseDict(data, tunits_pb2.Value())
        return cls.from_proto(message)
    if issubclass(cls, _tunits_runtime.ValueArray):
        from tunits.proto import tunits_pb2

        message = ParseDict(data, tunits_pb2.ValueArray())
        return cls.from_proto(message)
    raise TypeError(f"Unknown tunits class: {class_name}")


def deserialize_complex(payload: dict[str, Any]) -> complex:
    """
    Deserialize a Python complex payload.

    Parameters
    ----------
    payload : dict[str, Any]
        JSON-compatible complex payload.

    Returns
    -------
    complex
        Restored complex value.

    Raises
    ------
    TypeError
        If the payload references an unknown complex class.
    """
    data = dict(payload)
    type_name: str = data.pop(DATA_TYPE_KEY)
    class_name = type_name.removeprefix(DATA_PYTHON_PREFIX)
    if class_name == "complex":
        return complex(data[DATA_COMPLEX_REAL_KEY], data[DATA_COMPLEX_IMAG_KEY])
    raise TypeError(f"Unknown complex class: {type_name}")


def serialize_value(value: Any) -> Any:
    """
    Serialize nested values with NumPy, tunits, and complex support.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    Any
        JSON-compatible payload with type metadata where needed.
    """
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
    """
    Deserialize nested values serialized by `serialize_value`.

    Parameters
    ----------
    value : Any
        Serialized payload.

    Returns
    -------
    Any
        Restored Python values with NumPy/tunits/complex support.
    """
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
