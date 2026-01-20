from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from .experiment import Experiment


class ExperimentTaskResult(Protocol):
    """Protocol for experiment task results."""


T = TypeVar("T", bound=ExperimentTaskResult, covariant=True)


class ExperimentTask(Protocol, Generic[T]):
    """Protocol for experiment tasks executed by `Experiment`."""

    def execute(self, exp: Experiment) -> T:
        """Execute the task using the given Experiment object."""
        ...
