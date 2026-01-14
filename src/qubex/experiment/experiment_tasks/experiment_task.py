from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from qubex.experiment.experiment_context import ExperimentContext

T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class TaskResult(Generic[T]):
    """
    Immutable container for the result of an experiment task.

    Parameters
    ----------
    success : bool
        Whether the task completed successfully.
    data : T | None
        The main result data of the task.
    message : str
        A message describing the result or error.
    artifacts : dict[str, Any]
        Additional artifacts produced by the task.
    """

    success: bool
    data: T | None = None
    message: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentTask(ABC, Generic[T]):
    """
    Immutable Command definition for an experiment task.
    Concrete tasks should implement the `execute` method.
    """

    @abstractmethod
    def execute(self, ctx: ExperimentContext) -> TaskResult[T]:
        """
        Executes the task within the given experiment context.

        Parameters
        ----------
        ctx : ExperimentContext
            The experiment context to use for execution.

        Returns
        -------
        TaskResult[T]
            The result of the task execution.
        """
        ...
