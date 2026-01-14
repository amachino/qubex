from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any, Generic

from typing_extensions import Self, TypeVar

from .experiment_context import ExperimentContext


@dataclass
class ExperimentTaskResult:
    """Base class for experiment task results."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    data: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} created_at={self.created_at} data={{...}}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "created_at": self.created_at,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a result instance from a dictionary."""
        kwargs = {}
        if "data" in data:
            kwargs["data"] = data["data"]
        if "created_at" in data:
            kwargs["created_at"] = data["created_at"]
        return cls(**kwargs)

    def to_json(self, **kwargs) -> str:
        """Convert the result to a JSON string."""
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_json(cls, json_str: str, **kwargs) -> Self:
        """Create a result instance from a JSON string."""
        return cls.from_dict(json.loads(json_str, **kwargs))


T = TypeVar(
    "T",
    bound=ExperimentTaskResult,
    default=ExperimentTaskResult,
    covariant=True,
)


@dataclass
class ExperimentTask(ABC, Generic[T]):
    """
    Base class for all experiment tasks (Command Pattern).
    """

    @abstractmethod
    def execute(self, ctx: ExperimentContext) -> T:
        """
        Execute the experiment task logic.

        Args:
            ctx: The experiment context, providing access to services.

        Returns:
            The experiment result.
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the task parameters to a dictionary.
        Assumes the task is implemented as a dataclass.
        """
        if is_dataclass(self):
            return asdict(self)
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is not a dataclass. "
            "Please implement `to_dict` or decorate the class with `@dataclass`."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a task instance from a dictionary.
        Assumes the task is implemented as a dataclass.
        """
        if is_dataclass(cls):
            return cls(**data)
        raise NotImplementedError(
            f"'{cls.__name__}' is not a dataclass. "
            "Please implement `from_dict` or decorate the class with `@dataclass`."
        )

    def to_json(self, **kwargs) -> str:
        """
        Convert the task parameters to a JSON string.
        """
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_json(cls, json_str: str, **kwargs) -> Self:
        """
        Create a task instance from a JSON string.
        """
        return cls.from_dict(json.loads(json_str, **kwargs))
