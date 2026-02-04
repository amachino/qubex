"""Backend execution abstraction and QuEL implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .device_controller import DeviceController, RawResult


@dataclass(frozen=True)
class BackendExecutionRequest:
    """Backend-neutral execution request."""

    payload: Any


@dataclass(frozen=True)
class QuelExecutionPayload:
    """QuEL-specific execution payload for backend request."""

    sequencer: Any
    repeats: int
    integral_mode: str
    dsp_demodulation: bool
    enable_sum: bool
    enable_classification: bool
    line_param0: tuple[float, float, float] | None = None
    line_param1: tuple[float, float, float] | None = None


BackendResult = RawResult


class BackendExecutor(Protocol):
    """Protocol for backend execution of prepared requests."""

    def execute(self, *, request: BackendExecutionRequest) -> BackendResult:
        """Execute a prepared backend request."""
        ...


class QuelBackendExecutor:
    """QuEL backend executor using `DeviceController.execute_sequencer`."""

    def __init__(self, *, device_controller: DeviceController) -> None:
        self._device_controller = device_controller

    def execute(self, *, request: BackendExecutionRequest) -> BackendResult:
        """Execute a prepared request on QuEL hardware."""
        payload = request.payload
        if not isinstance(payload, QuelExecutionPayload):
            raise TypeError(
                "QuelBackendExecutor expects `QuelExecutionPayload` payload."
            )
        return self._device_controller.execute_sequencer(
            sequencer=payload.sequencer,
            repeats=payload.repeats,
            integral_mode=payload.integral_mode,
            dsp_demodulation=payload.dsp_demodulation,
            enable_sum=payload.enable_sum,
            enable_classification=payload.enable_classification,
            line_param0=payload.line_param0,
            line_param1=payload.line_param1,
        )
