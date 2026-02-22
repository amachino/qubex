"""QuEL-3 backend executor implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .quel3_execution_payload import Quel3ExecutionPayload


@runtime_checkable
class _Quel3ExecutionHook(Protocol):
    """Protocol for backend controllers exposing QuEL-3 execution hook."""

    def execute_measurement(self, *, payload: Quel3ExecutionPayload) -> object:
        """Execute one QuEL-3 backend payload."""
        ...


class Quel3BackendExecutor:
    """QuEL-3 backend executor delegating to backend-controller hook."""

    def __init__(self, *, backend_controller: object) -> None:
        self._backend_controller = backend_controller

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute QuEL-3 payload using backend-controller integration hook."""
        payload = request.payload
        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3BackendExecutor expects `Quel3ExecutionPayload` payload."
            )
        if not isinstance(self._backend_controller, _Quel3ExecutionHook):
            raise TypeError(
                "Quel3 backend execution requires backend_controller.execute_measurement(payload=...)."
            )
        return self._backend_controller.execute_measurement(payload=payload)
