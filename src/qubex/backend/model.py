"""Backend model base class with Pydantic helpers."""

from __future__ import annotations

from typing import cast

from pydantic import ConfigDict, RootModel, TypeAdapter
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(validate_assignment=True))
class Model:
    """Base dataclass model with JSON helpers."""

    @classmethod
    def _pd_class(cls) -> TypeAdapter:
        """Return a TypeAdapter for the model class."""
        return TypeAdapter(cls)

    @classmethod
    def json_schema(cls) -> dict:
        """Return the JSON schema for the model."""
        return cls._pd_class().json_schema()

    @classmethod
    def load(cls, data: dict | str) -> Model:
        """Load a model instance from dict or JSON string."""
        if isinstance(data, str):
            obj = cls._pd_class().validate_json(data)
        else:
            obj = cls._pd_class().validate_python(data)
        return obj

    def _pd_model(self):
        """Return a RootModel wrapper for serialization."""
        return RootModel(self)

    def to_dict(self) -> dict:
        """Serialize the model to a dictionary."""
        dumped = self._pd_model().model_dump()
        return cast(dict, dumped)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the model to a JSON string."""
        return self._pd_model().model_dump_json(indent=indent)

    @property
    def hash(self) -> int:
        """Return a hash of the JSON representation."""
        return hash(self.to_json(indent=0))
