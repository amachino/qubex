"""Seralizable base model with NumPy and tunits support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    model_serializer,
)
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from typing_extensions import Self

from qubex.core.serialization import (
    FORMAT_NAME,
    FORMAT_VERSION,
    META_FORMAT_KEY,
    META_KEY,
    META_VERSION_KEY,
    deserialize_value,
    is_custom_serializable_class,
    serialize_value,
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
            if isinstance(cls, type) and is_custom_serializable_class(cls):
                return {"type": "object"}
        return super().handle_invalid_for_json_schema(schema, error_info)


class Model(BaseModel):
    """Base model with custom serialization helpers."""

    format_name: ClassVar[str] = FORMAT_NAME
    format_version: ClassVar[int] = FORMAT_VERSION

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
        return serialize_value(data)

    @classmethod
    def json_schema(cls, **kwargs) -> dict[str, Any]:
        """Return the JSON schema for the model."""
        kwargs.setdefault("schema_generator", NumpyTunitsJsonSchema)
        return cls.model_json_schema(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a model instance from a dictionary."""
        payload = dict(data)
        payload.pop(META_KEY, None)
        return cls.model_validate(deserialize_value(payload))

    @classmethod
    def from_json(cls, data: str) -> Self:
        """Create a model instance from a JSON string."""
        payload = json.loads(data)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.pop(META_KEY, None)
        return cls.model_validate(deserialize_value(payload))

    def to_dict(self) -> dict:
        """Serialize the model to a dictionary."""
        data = self.model_dump()
        if isinstance(data, dict):
            data[META_KEY] = {
                META_FORMAT_KEY: self.format_name,
                META_VERSION_KEY: self.format_version,
            }
        return data

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the model to a JSON string."""
        # NOTE: Pydantic's built-in model_dump_json does not support custom serialization well.
        # return self.model_dump_json(indent=indent)
        data = self.to_dict()
        return json.dumps(data, ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Save the model as a JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(self.to_json(indent=indent), encoding="utf-8")
        return path_obj

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        """Load the model from a JSON file."""
        path_obj = Path(path)
        return cls.from_json(path_obj.read_text(encoding="utf-8"))


class MutableModel(Model):
    """Mutable variant of the base model."""

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
    )
