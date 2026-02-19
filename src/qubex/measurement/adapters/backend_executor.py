"""Backend executor helpers for measurement adapters."""

from __future__ import annotations

from typing import Any

from qubex.backend import (
    BackendExecutionRequest,
    BackendExecutionResult,
)
from qubex.backend.quel3 import Quel3ExecutionPayload


class Quel3BackendExecutor:
    """Quel3 backend executor delegating to backend-controller hook."""

    def __init__(self, *, backend_controller: Any) -> None:
        self._backend_controller = backend_controller

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute Quel3 payload using backend-controller integration hook."""
        payload = request.payload
        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3BackendExecutor expects `Quel3ExecutionPayload` payload."
            )
        execute_impl = getattr(self._backend_controller, "execute_measurement", None)
        if not callable(execute_impl):
            raise TypeError(
                "Quel3 backend execution requires backend_controller.execute_measurement(payload=...)."
            )
        return execute_impl(payload=payload)
