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

_NUMPY_PREFIX = "numpy."
_TUNITS_PREFIX = "tunits."


def _is_numpy(value: Any) -> TypeGuard[np.ndarray | np.generic]:
    return isinstance(value, (np.ndarray, np.generic))


def _is_tunits(value: Any) -> TypeGuard[tunits.Value | tunits.ValueArray]:
    return isinstance(value, (tunits.Value, tunits.ValueArray))


def _tunits_to_dict(value: tunits.Value | tunits.ValueArray) -> dict[str, Any]:
    class_name = value.__class__.__name__
    message = value.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data["__type__"] = f"{_TUNITS_PREFIX}{class_name}"
    return data


def _numpy_to_dict(value: np.ndarray | np.generic) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value_tunits = tunits.ValueArray(value)
    else:
        value_tunits = tunits.Value(value.item())
    class_name = value.__class__.__name__
    message = value_tunits.to_proto()
    data = MessageToDict(message, preserving_proto_field_name=True)
    data["__type__"] = f"{_NUMPY_PREFIX}{class_name}"
    return data


def _serialize(obj: Any) -> Any:
    if _is_numpy(obj):
        return _numpy_to_dict(obj)
    if _is_tunits(obj):
        return _tunits_to_dict(obj)
    if isinstance(obj, Mapping):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialize(v) for v in obj]
    return obj


def _is_numpy_dict(value: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(value, dict) or "__type__" not in value:
        return False
    return value["__type__"].startswith(_NUMPY_PREFIX)


def _is_tunits_dict(value: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(value, dict) or "__type__" not in value:
        return False
    return value["__type__"].startswith(_TUNITS_PREFIX)


def _numpy_from_dict(value: dict[str, Any]) -> np.ndarray | np.generic:
    payload = dict(value)  # make a copy to avoid modifying the original
    type_name: str = payload.pop("__type__")
    class_name = type_name.removeprefix(_NUMPY_PREFIX)
    cls = getattr(np, class_name)
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
    type_name: str = payload.pop("__type__")
    class_name = type_name.removeprefix(_TUNITS_PREFIX)
    cls = getattr(tunits, class_name)
    if issubclass(cls, tunits.Value):
        message = ParseDict(payload, tunits_pb2.Value())
        return cls.from_proto(message)
    elif issubclass(cls, tunits.ValueArray):
        message = ParseDict(payload, tunits_pb2.ValueArray())
        return cls.from_proto(message)
    else:
        raise TypeError(f"Unknown tunits class: {class_name}")


def _deserialize(obj: Any) -> Any:
    if _is_numpy_dict(obj):
        return _numpy_from_dict(obj)
    if _is_tunits_dict(obj):
        return _tunits_from_dict(obj)
    if isinstance(obj, Mapping):
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
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
