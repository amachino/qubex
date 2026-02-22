"""QuEL-3 backend executor implementation."""

from __future__ import annotations

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .quel3_execution_payload import Quel3ExecutionPayload


class Quel3BackendExecutor:
    """QuEL-3 backend executor delegating to backend-controller execute API."""

    def __init__(
        self,
        *,
        backend_controller: object,
    ) -> None:
        self._backend_controller = backend_controller

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute QuEL-3 payload using backend-controller execute API."""
        payload = request.payload
        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3BackendExecutor expects `Quel3ExecutionPayload` payload."
            )
        execute = getattr(self._backend_controller, "execute", None)
        if not callable(execute):
            raise TypeError(
                "Quel3 backend execution requires backend_controller.execute(request=...)."
            )
        return execute(request=request)
