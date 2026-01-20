from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()

    @classmethod
    def load(cls, data: dict | str):
        if isinstance(data, dict):
            obj = cls.model_validate(data)
        else:
            obj = cls.model_validate_json(data)
        return obj

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, indent: int | None = None) -> str:
        return self.model_dump_json(indent=indent)


class MutableModel(Model):
    model_config = ConfigDict(validate_assignment=True)


class ImmutableModel(Model):
    model_config = ConfigDict(frozen=True)
