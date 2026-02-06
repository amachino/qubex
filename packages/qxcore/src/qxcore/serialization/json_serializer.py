"""JSON serialization utilities for core models."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, overload

import numpy as np
from google.protobuf.json_format import MessageToDict, ParseDict

if TYPE_CHECKING:
    import tunits as _tunits

    TunitsValue = _tunits.Value
    TunitsValueArray = _tunits.ValueArray
else:
    TunitsValue = Any
    TunitsValueArray = Any

try:
    import tunits as _tunits_runtime

    _HAS_TUNITS = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency path.
    _tunits_runtime = None
    _HAS_TUNITS = False

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


def is_tunits(value: Any) -> TypeGuard[TunitsValue | TunitsValueArray]:
    """Return whether `value` is a tunits value or value-array."""
    tunits_module = _get_tunits_module(optional=True)
    if tunits_module is None:
        return False
    return isinstance(value, (tunits_module.Value, tunits_module.ValueArray))


def is_complex(value: Any) -> TypeGuard[complex]:
    """Return whether `value` is a complex scalar."""
    return isinstance(value, complex)


def is_custom_serializable_class(cls: type[Any]) -> bool:
    """Return whether class is one of the custom-serialized value types."""
    if issubclass(cls, (complex, np.ndarray, np.generic)):
        return True
    tunits_module = _get_tunits_module(optional=True)
    if tunits_module is None:
        return False
    return issubclass(cls, (tunits_module.Value, tunits_module.ValueArray))


def serialize_complex(value: complex) -> dict[str, Any]:
    """Serialize a Python complex value."""
    class_name = value.__class__.__name__
    return {
        DATA_COMPLEX_REAL_KEY: float(value.real),
        DATA_COMPLEX_IMAG_KEY: float(value.imag),
        DATA_TYPE_KEY: f"{DATA_PYTHON_PREFIX}{class_name}",
    }


def serialize_tunits(value: TunitsValue | TunitsValueArray) -> dict[str, Any]:
    """Serialize a tunits value to a JSON-compatible dict via protobuf."""
    _require_tunits()
    class_name = value.__class__.__name__
    message = value.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[DATA_TYPE_KEY] = f"{DATA_TUNITS_PREFIX}{class_name}"
    return data


def serialize_numpy(value: np.ndarray | np.generic) -> dict[str, Any]:
    """Serialize a NumPy value to a JSON-compatible dict via tunits protobuf."""
    if not _HAS_TUNITS:
        array = np.asarray(value)
        return {
            DATA_TYPE_KEY: f"{DATA_NUMPY_PREFIX}{value.__class__.__name__}",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.reshape(-1).tolist(),
        }

    tunits_module = _get_tunits_module()
    if isinstance(value, np.ndarray):
        value_tunits = tunits_module.ValueArray(value)
    else:
        value_tunits = tunits_module.Value(value.item())
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

    if not _HAS_TUNITS:
        dtype = np.dtype(str(data["dtype"]))
        shape = tuple(int(v) for v in data["shape"])
        flat = np.asarray(data["data"], dtype=dtype)
        restored = flat.reshape(shape)
        if issubclass(cls, np.ndarray):
            return restored
        if issubclass(cls, np.generic):
            return cls(restored.item())
        raise TypeError(f"Unknown numpy class: {class_name}")

    if issubclass(cls, np.ndarray):
        from tunits.proto import tunits_pb2

        message = ParseDict(data, tunits_pb2.ValueArray())
        tunits_module = _get_tunits_module()
        value_tunits = tunits_module.ValueArray.from_proto(message)
        return value_tunits.value
    if issubclass(cls, np.generic):
        from tunits.proto import tunits_pb2

        message = ParseDict(data, tunits_pb2.Value())
        tunits_module = _get_tunits_module()
        value_tunits = tunits_module.Value.from_proto(message)
        return cls(value_tunits.value)
    raise TypeError(f"Unknown numpy class: {class_name}")


def deserialize_tunits(payload: dict[str, Any]) -> TunitsValue | TunitsValueArray:
    """Deserialize a tunits payload."""
    _require_tunits()
    data = dict(payload)
    type_name: str = data.pop(DATA_TYPE_KEY)
    class_name = type_name.removeprefix(DATA_TUNITS_PREFIX)
    tunits_module = _get_tunits_module()
    try:
        cls = getattr(tunits_module, class_name)
    except AttributeError as exc:
        raise TypeError(f"Unknown tunits class: {class_name}") from exc
    if not isinstance(cls, type):
        raise TypeError(f"Unknown tunits class: {class_name}")
    if issubclass(cls, tunits_module.Value):
        from tunits.proto import tunits_pb2

        message = ParseDict(data, tunits_pb2.Value())
        return cls.from_proto(message)
    if issubclass(cls, tunits_module.ValueArray):
        from tunits.proto import tunits_pb2

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


def _require_tunits() -> None:
    if _get_tunits_module(optional=True) is None:
        raise ModuleNotFoundError(
            "Optional dependency 'tunits' is required. Install extras: [tunits]."
        )


@overload
def _get_tunits_module(*, optional: Literal[True]) -> Any | None: ...


@overload
def _get_tunits_module(*, optional: Literal[False] = False) -> Any: ...


def _get_tunits_module(*, optional: bool = False) -> Any | None:
    if _tunits_runtime is not None:
        return _tunits_runtime
    if optional:
        return None
    raise ModuleNotFoundError(
        "Optional dependency 'tunits' is required. Install extras: [tunits]."
    )
