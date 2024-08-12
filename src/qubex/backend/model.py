from __future__ import annotations

from pydantic import RootModel, TypeAdapter


class Model:
    @classmethod
    def _pd_class(cls):
        return TypeAdapter(cls)

    @classmethod
    def json_schema(cls):
        return cls._pd_class().json_schema()

    @classmethod
    def load(cls, data: dict | str):
        if isinstance(data, str):
            obj = cls._pd_class().validate_json(data)
        else:
            obj = cls._pd_class().validate_python(data)
        return obj

    def _pd_model(self):
        return RootModel(self)

    def to_dict(self) -> dict:
        return self._pd_model().model_dump()  # type: ignore

    def to_json(self, indent: int | None = None) -> str:
        return self._pd_model().model_dump_json(indent=indent)

    @property
    def hash(self) -> int:
        return hash(self.to_json(indent=0))
