"""Seralizable base model with NumPy and tunits support."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, TypeGuard

import numpy as np
import tunits
from google.protobuf.json_format import MessageToDict, ParseDict
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    model_serializer,
)
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from tunits.proto import tunits_pb2
from typing_extensions import Self

SERIALIZATION_VERSION = 0

_META_KEY = "__meta__"
_META_VERSION_KEY = "version"

_DATA_TYPE_KEY = "__type__"
_DATA_NUMPY_PREFIX = "numpy."
_DATA_TUNITS_PREFIX = "tunits."
_DATA_PYTHON_PREFIX = "python."
_DATA_COMPLEX_REAL_KEY = "real"
_DATA_COMPLEX_IMAG_KEY = "imag"

logger = logging.getLogger(__name__)


def _is_numpy(value: Any) -> TypeGuard[np.ndarray | np.generic]:
    return isinstance(value, (np.ndarray, np.generic))


def _is_tunits(value: Any) -> TypeGuard[tunits.Value | tunits.ValueArray]:
    return isinstance(value, (tunits.Value, tunits.ValueArray))


def _is_complex(value: Any) -> TypeGuard[complex]:
    return isinstance(value, complex)


def _complex_to_dict(value: complex) -> dict[str, Any]:
    class_name = value.__class__.__name__
    return {
        _DATA_COMPLEX_REAL_KEY: float(value.real),
        _DATA_COMPLEX_IMAG_KEY: float(value.imag),
        _DATA_TYPE_KEY: f"{_DATA_PYTHON_PREFIX}{class_name}",
    }


def _tunits_to_dict(value: tunits.Value | tunits.ValueArray) -> dict[str, Any]:
    class_name = value.__class__.__name__
    message = value.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[_DATA_TYPE_KEY] = f"{_DATA_TUNITS_PREFIX}{class_name}"
    return data


def _numpy_to_dict(value: np.ndarray | np.generic) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value_tunits = tunits.ValueArray(value)
    else:
        value_tunits = tunits.Value(value.item())
    class_name = value.__class__.__name__
    message = value_tunits.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[_DATA_TYPE_KEY] = f"{_DATA_NUMPY_PREFIX}{class_name}"
    return data


def _serialize(obj: Any) -> Any:
    if _is_numpy(obj):
        return _numpy_to_dict(obj)
    if _is_complex(obj):
        return _complex_to_dict(obj)
    if _is_tunits(obj):
        return _tunits_to_dict(obj)
    if isinstance(obj, Mapping):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _numpy_from_dict(value: dict[str, Any]) -> np.ndarray | np.generic:
    payload = dict(value)  # make a copy to avoid modifying the original
    type_name: str = payload.pop(_DATA_TYPE_KEY)
    class_name = type_name.removeprefix(_DATA_NUMPY_PREFIX)
    try:
        cls = getattr(np, class_name)
    except AttributeError as exc:
        raise TypeError(f"Unknown numpy class: {class_name}") from exc
    if not isinstance(cls, type):
        raise TypeError(f"Unknown numpy class: {class_name}")
    # Use tunits as an intermediary for deserialization
    if issubclass(cls, np.ndarray):
        message = ParseDict(payload, tunits_pb2.ValueArray())
        value_tunits = tunits.ValueArray.from_proto(message)
        return value_tunits.value
    elif issubclass(cls, np.generic):
        message = ParseDict(payload, tunits_pb2.Value())
        value_tunits = tunits.Value.from_proto(message)
        return cls(value_tunits.value)
    else:
        raise TypeError(f"Unknown numpy class: {class_name}")


def _tunits_from_dict(value: dict[str, Any]) -> tunits.Value | tunits.ValueArray:
    payload = dict(value)  # make a copy to avoid modifying the original
    type_name: str = payload.pop(_DATA_TYPE_KEY)
    class_name = type_name.removeprefix(_DATA_TUNITS_PREFIX)
    try:
        cls = getattr(tunits, class_name)
    except AttributeError as exc:
        raise TypeError(f"Unknown tunits class: {class_name}") from exc
    if not isinstance(cls, type):
        raise TypeError(f"Unknown tunits class: {class_name}")
    if issubclass(cls, tunits.Value):
        message = ParseDict(payload, tunits_pb2.Value())
        return cls.from_proto(message)
    elif issubclass(cls, tunits.ValueArray):
        message = ParseDict(payload, tunits_pb2.ValueArray())
        return cls.from_proto(message)
    else:
        raise TypeError(f"Unknown tunits class: {class_name}")


def _complex_from_dict(value: dict[str, Any]) -> complex:
    payload = dict(value)  # make a copy to avoid modifying the original
    type_name: str = payload.pop(_DATA_TYPE_KEY)
    class_name = type_name.removeprefix(_DATA_PYTHON_PREFIX)
    if class_name == "complex":
        return complex(payload[_DATA_COMPLEX_REAL_KEY], payload[_DATA_COMPLEX_IMAG_KEY])
    raise TypeError(f"Unknown complex class: {type_name}")


def _deserialize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        type_tag = obj.get(_DATA_TYPE_KEY)
        if isinstance(type_tag, str):
            payload = dict(obj)
            if type_tag.startswith(_DATA_NUMPY_PREFIX):
                return _numpy_from_dict(payload)
            elif type_tag.startswith(_DATA_TUNITS_PREFIX):
                return _tunits_from_dict(payload)
            elif type_tag.startswith(_DATA_PYTHON_PREFIX):
                return _complex_from_dict(payload)
            else:
                logger.warning(f"Unknown type during deserialization: {type_tag}")
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deserialize(v) for v in obj]
    return obj


def _is_custom_class(cls: type[Any]) -> bool:
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


class NumpyTunitsJsonSchema(GenerateJsonSchema):
    """JSON schema generator that supports NumPy and tunits types."""

    def handle_invalid_for_json_schema(
        self,
        schema: Any,
        error_info: Any,
    ) -> JsonSchemaValue:
        """Handle unsupported schema nodes for custom types."""
        if schema.get("type") == "is-instance":
            cls = schema.get("cls")
            if isinstance(cls, type) and _is_custom_class(cls):
                return {"type": "object"}
        return super().handle_invalid_for_json_schema(schema, error_info)


class Model(BaseModel):
    """Base model with custom serialization helpers."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @model_serializer(mode="wrap")
    def _serialize_model(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> Any:
        """Serialize the model with custom value handling."""
        data = handler(self)
        return _serialize(data)

    @classmethod
    def json_schema(cls, **kwargs) -> dict[str, Any]:
        """Return the JSON schema for the model."""
        kwargs.setdefault("schema_generator", NumpyTunitsJsonSchema)
        return cls.model_json_schema(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a model instance from a dictionary."""
        payload = dict(data)
        payload.pop(_META_KEY, None)
        return cls.model_validate(_deserialize(payload))

    @classmethod
    def from_json(cls, data: str) -> Self:
        """Create a model instance from a JSON string."""
        payload = json.loads(data)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.pop(_META_KEY, None)
        return cls.model_validate(_deserialize(payload))

    def to_dict(self) -> dict:
        """Serialize the model to a dictionary."""
        data = self.model_dump()
        if isinstance(data, dict):
            data[_META_KEY] = {_META_VERSION_KEY: SERIALIZATION_VERSION}
        return data

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the model to a JSON string."""
        # NOTE: Pydantic's built-in model_dump_json does not support custom serialization well.
        # return self.model_dump_json(indent=indent)
        data = self.to_dict()
        return json.dumps(data, ensure_ascii=False, indent=indent)


class MutableModel(Model):
    """Mutable variant of the base model."""

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
    )
