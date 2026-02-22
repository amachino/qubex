"""QuEL-3 backend executor implementation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .quel3_execution_payload import Quel3ExecutionPayload


@runtime_checkable
class _BackendExecuteHook(Protocol):
    """Protocol for backend controllers exposing execute(request=...)."""

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute one backend request."""
        ...


class Quel3BackendExecutor:
    """QuEL-3 backend executor delegating to backend-controller execute API."""

    def __init__(self, *, backend_controller: Any) -> None:
        self._backend_controller = backend_controller

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute QuEL-3 payload using backend-controller execute API."""
        payload = request.payload
        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3BackendExecutor expects `Quel3ExecutionPayload` payload."
            )
        if not isinstance(self._backend_controller, _BackendExecuteHook):
            raise TypeError(
                "Quel3 backend execution requires backend_controller.execute(request=...)."
            )
        return self._backend_controller.execute(request=request)
