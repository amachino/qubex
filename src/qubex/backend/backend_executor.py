"""Backend execution abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

ExecutionModeOption: TypeAlias = Literal["serial", "parallel"]


@dataclass(frozen=True)
class BackendExecutionRequest:
    """Backend-neutral execution request."""

    payload: Any
    execution_mode: ExecutionModeOption | None = None
    clock_health_checks: bool | None = None


BackendExecutionResult: TypeAlias = Any


class BackendExecutor(Protocol):
    """Protocol for backend execution of prepared requests."""

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute a prepared backend request."""
        ...
