"""QuEL backend execution implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .quel1_backend_controller import Quel1BackendController


@dataclass(frozen=True)
class Quel1ExecutionPayload:
    """QuEL-specific execution payload for backend request."""

    sequencer: Any
    repeats: int
    integral_mode: str
    dsp_demodulation: bool
    enable_sum: bool
    enable_classification: bool
    line_param0: tuple[float, float, float]
    line_param1: tuple[float, float, float]


class Quel1BackendExecutor:
    """QuEL backend executor using `Quel1BackendController.execute_sequencer`."""

    def __init__(self, *, device_controller: Quel1BackendController) -> None:
        self._device_controller = device_controller

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute a prepared request on QuEL hardware."""
        payload = request.payload
        if not isinstance(payload, Quel1ExecutionPayload):
            raise TypeError(
                "Quel1BackendExecutor expects `Quel1ExecutionPayload` payload."
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
