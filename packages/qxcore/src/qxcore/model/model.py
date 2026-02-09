"""Serializable base model with NumPy and tunits support."""

from __future__ import annotations

import hashlib
import json
from functools import cached_property
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

from qxcore.serialization import (
    FORMAT_NAME,
    FORMAT_VERSION,
    META_FORMAT_KEY,
    META_KEY,
    META_VERSION_KEY,
    deserialize_value,
    is_custom_serializable_class,
    serialize_value,
    to_canonical_json,
)


class NumpyTunitsJsonSchema(GenerateJsonSchema):
    """
    JSON schema generator that supports NumPy and tunits types.

    Notes
    -----
    This overrides Pydantic's schema generation for instances that are serialized
    through custom handlers, allowing them to appear as generic objects in the
    schema output.
    """

    def handle_invalid_for_json_schema(
        self,
        schema: Any,
        error_info: Any,
    ) -> JsonSchemaValue:
        """
        Handle unsupported schema nodes for custom types.

        Parameters
        ----------
        schema : Any
            Schema payload that Pydantic could not render.
        error_info : Any
            Error information provided by Pydantic.

        Returns
        -------
        JsonSchemaValue
            Fallback schema value for custom serialized types.
        """
        if schema.get("type") == "is-instance":
            cls = schema.get("cls")
            if isinstance(cls, type) and is_custom_serializable_class(cls):
                return {"type": "object"}
        return super().handle_invalid_for_json_schema(schema, error_info)


class Model(BaseModel):
    """
    Base model with custom serialization helpers.

    Notes
    -----
    Values such as NumPy arrays, complex numbers, and tunits values are encoded
    using custom serializers so they remain JSON compatible.
    """

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
        """
        Serialize the model with custom value handling.

        Parameters
        ----------
        handler : SerializerFunctionWrapHandler
            Pydantic serializer wrapper.

        Returns
        -------
        Any
            Serialized payload with custom type metadata.
        """
        data = handler(self)
        return serialize_value(data)

    @classmethod
    def json_schema(cls, **kwargs) -> dict[str, Any]:
        """
        Return the JSON schema for the model.

        Parameters
        ----------
        **kwargs
            Additional arguments forwarded to `model_json_schema`.

        Returns
        -------
        dict[str, Any]
            JSON schema with custom type support.
        """
        kwargs.setdefault("schema_generator", NumpyTunitsJsonSchema)
        return cls.model_json_schema(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """
        Create a model instance from a dictionary.

        Parameters
        ----------
        data : dict
            Input payload, optionally including metadata.

        Returns
        -------
        Self
            Restored model instance.
        """
        payload = dict(data)
        payload.pop(META_KEY, None)
        return cls.model_validate(deserialize_value(payload))

    @classmethod
    def from_json(cls, data: str) -> Self:
        """
        Create a model instance from a JSON string.

        Parameters
        ----------
        data : str
            JSON document.

        Returns
        -------
        Self
            Restored model instance.
        """
        payload = json.loads(data)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.pop(META_KEY, None)
        return cls.model_validate(deserialize_value(payload))

    def to_dict(self) -> dict:
        """
        Serialize the model to a dictionary.

        Returns
        -------
        dict
            Serialized payload with format metadata.
        """
        data = self.model_dump()
        if isinstance(data, dict):
            data[META_KEY] = {
                META_FORMAT_KEY: self.format_name,
                META_VERSION_KEY: self.format_version,
            }
        return data

    def to_json(self, indent: int | None = None) -> str:
        """
        Serialize the model to a JSON string.

        Parameters
        ----------
        indent : int | None, default=None
            JSON indentation level.

        Returns
        -------
        str
            JSON document.
        """
        # NOTE: Pydantic's built-in model_dump_json does not support custom serialization well.
        # return self.model_dump_json(indent=indent)
        data = self.to_dict()
        return json.dumps(data, ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """
        Save the model as a JSON file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        indent : int | None, default=2
            JSON indentation level.

        Returns
        -------
        Path
            Path to the saved file.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(self.to_json(indent=indent), encoding="utf-8")
        return path_obj

    def _canonical_hash_bytes(self) -> bytes:
        """
        Return canonical bytes used for hash computation.

        Returns
        -------
        bytes
            UTF-8 encoded canonical payload bytes.
        """
        canonical = to_canonical_json(self.to_dict())
        namespace = (
            f"{self.__class__.__module__}.{self.__class__.__qualname__}:"
            f"{self.format_version}:"
        )
        return f"{namespace}{canonical}".encode()

    def _compute_hash(self) -> str:
        """
        Compute a SHA-256 hash of this model.

        Returns
        -------
        str
            SHA-256 digest as a hexadecimal string.
        """
        return hashlib.sha256(self._canonical_hash_bytes()).hexdigest()

    @cached_property
    def hash(self) -> str:
        """
        Return the cached SHA-256 hash of this model.

        Returns
        -------
        str
            SHA-256 digest as a hexadecimal string.
        """
        return self._compute_hash()

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        """
        Load the model from a JSON file.

        Parameters
        ----------
        path : str | Path
            Input JSON file path.

        Returns
        -------
        Self
            Restored model instance.
        """
        path_obj = Path(path)
        return cls.from_json(path_obj.read_text(encoding="utf-8"))


class MutableModel(Model):
    """
    Mutable variant of the base model.

    Notes
    -----
    Assignment validation is enabled to ensure field invariants.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
    )

    @property
    def hash(self) -> str:
        """
        Return a SHA-256 hash of this mutable model.

        Returns
        -------
        str
            SHA-256 digest as a hexadecimal string.
        """
        return self._compute_hash()
