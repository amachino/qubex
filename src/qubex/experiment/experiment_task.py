from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from .experiment_context import ExperimentContext


class ExperimentTaskResult(Protocol):
    """Protocol for experiment task results."""


T = TypeVar("T", bound=ExperimentTaskResult)


class ExperimentTask(Protocol, Generic[T]):
    """Protocol for experiment tasks executed by `Experiment`."""

    def execute(self, ctx: ExperimentContext) -> T:
        """Execute the task using the given experiment context."""
        ...
