from __future__ import annotations

import json
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
from tunits.proto import tunits_pb2

_TYPE_KEY = "__type__"

_NUMPY_PREFIX = "numpy."
_TUNITS_PREFIX = "tunits."
_COLLECTION_PREFIX = "collection."

_COLLECTION_ITEMS_KEY = "items"


def _is_numpy(value: Any) -> TypeGuard[np.ndarray | np.generic]:
    return isinstance(value, (np.ndarray, np.generic))


def _is_tunits(value: Any) -> TypeGuard[tunits.Value | tunits.ValueArray]:
    return isinstance(value, (tunits.Value, tunits.ValueArray))


def _tunits_to_dict(value: tunits.Value | tunits.ValueArray) -> dict[str, Any]:
    class_name = value.__class__.__name__
    message = value.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[_TYPE_KEY] = f"{_TUNITS_PREFIX}{class_name}"
    return data


def _numpy_to_dict(value: np.ndarray | np.generic) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value_tunits = tunits.ValueArray(value)
    else:
        value_tunits = tunits.Value(value.item())
    class_name = value.__class__.__name__
    message = value_tunits.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data[_TYPE_KEY] = f"{_NUMPY_PREFIX}{class_name}"
    return data


def _collection_to_dict(
    value: tuple[Any, ...] | set[Any] | frozenset[Any],
) -> dict[str, Any]:
    if isinstance(value, tuple):
        collection_type = "tuple"
    elif isinstance(value, frozenset):
        collection_type = "frozenset"
    else:
        collection_type = "set"
    return {
        _TYPE_KEY: f"{_COLLECTION_PREFIX}{collection_type}",
        _COLLECTION_ITEMS_KEY: [_serialize(item) for item in value],
    }


def _serialize(obj: Any) -> Any:
    if _is_numpy(obj):
        return _numpy_to_dict(obj)
    if _is_tunits(obj):
        return _tunits_to_dict(obj)
    if isinstance(obj, (tuple, set, frozenset)):
        return _collection_to_dict(obj)
    if isinstance(obj, Mapping):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _is_numpy_dict(value: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(value, dict) or _TYPE_KEY not in value:
        return False
    return value[_TYPE_KEY].startswith(_NUMPY_PREFIX)


def _is_tunits_dict(value: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(value, dict) or _TYPE_KEY not in value:
        return False
    return value[_TYPE_KEY].startswith(_TUNITS_PREFIX)


def _is_collection_dict(value: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(value, dict) or _TYPE_KEY not in value:
        return False
    return value[_TYPE_KEY].startswith(_COLLECTION_PREFIX)


def _numpy_from_dict(value: dict[str, Any]) -> np.ndarray | np.generic:
    payload = dict(value)  # make a copy to avoid modifying the original
    type_name: str = payload.pop(_TYPE_KEY)
    class_name = type_name.removeprefix(_NUMPY_PREFIX)
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
    type_name: str = payload.pop(_TYPE_KEY)
    class_name = type_name.removeprefix(_TUNITS_PREFIX)
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


def _collection_from_dict(
    value: dict[str, Any],
) -> tuple[Any, ...] | set[Any] | frozenset[Any]:
    payload = dict(value)
    type_name: str = payload.pop(_TYPE_KEY)
    items = payload.get(_COLLECTION_ITEMS_KEY, [])
    if not isinstance(items, list):
        raise TypeError("Collection payload must contain a list under 'items'.")
    deserialized_items = [_deserialize(item) for item in items]
    kind = type_name.removeprefix(_COLLECTION_PREFIX)
    if kind == "tuple":
        return tuple(deserialized_items)
    if kind == "frozenset":
        return frozenset(deserialized_items)
    if kind == "set":
        return set(deserialized_items)
    raise TypeError(f"Unknown collection type: {kind}")


def _deserialize(obj: Any) -> Any:
    if _is_numpy_dict(obj):
        return _numpy_from_dict(obj)
    if _is_tunits_dict(obj):
        return _tunits_from_dict(obj)
    if _is_collection_dict(obj):
        return _collection_from_dict(obj)
    if isinstance(obj, Mapping):
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize(v) for v in obj]
    return obj


class Model(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @model_serializer(mode="wrap")
    def _serialize_model(
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        data = handler(self)
        return _serialize(data)

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()

    @classmethod
    def from_dict(cls, data: dict):
        return cls.model_validate(_deserialize(data))

    @classmethod
    def from_json(cls, data: str):
        return cls.model_validate(_deserialize(json.loads(data)))

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, indent: int | None = None) -> str:
        # NOTE: Pydantic's built-in model_dump_json does not support custom serialization well.
        # return self.model_dump_json(indent=indent)
        data = self.model_dump()
        return json.dumps(data, ensure_ascii=False, indent=indent)


class MutableModel(Model):
    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
    )
