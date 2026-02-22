"""Backend execution abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias


@dataclass(frozen=True)
class BackendExecutionRequest:
    """Backend-neutral execution request."""

    payload: Any


BackendExecutionResult: TypeAlias = Any


class BackendExecutor(Protocol):
    """Protocol for backend execution of prepared requests."""

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute a prepared backend request."""
        ...
