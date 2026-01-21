from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeGuard

import tunits
from google.protobuf.json_format import MessageToDict
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    model_serializer,
)


def _is_tunits_value(value: Any) -> TypeGuard[tunits.Value]:
    return isinstance(value, tunits.Value)


def _tunits_to_dict(value: tunits.Value) -> dict[str, Any]:
    proto = value.to_proto()
    return MessageToDict(proto, preserving_proto_field_name=True)


def _serialize(obj: Any) -> Any:
    if _is_tunits_value(obj):
        return _tunits_to_dict(obj)

    if isinstance(obj, Mapping):
        return {k: _serialize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_serialize(v) for v in obj]

    return obj


class Model(BaseModel):
    model_config = ConfigDict(
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
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, data: str):
        return cls.model_validate_json(data)

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, indent: int | None = None) -> str:
        return self.model_dump_json(indent=indent)


class MutableModel(Model):
    model_config = ConfigDict(validate_assignment=True)


class ImmutableModel(Model):
    model_config = ConfigDict(frozen=True)
