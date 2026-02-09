"""QuEL backend execution implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .quel1_backend_controller import Quel1BackendController


@dataclass(frozen=True)
class Quel1ExecutionPayload:
    """QuEL-1 specific execution payload for backend request."""

    sequencer: Any
    repeats: int
    integral_mode: str
    dsp_demodulation: bool
    enable_sum: bool
    enable_classification: bool
    line_param0: tuple[float, float, float]
    line_param1: tuple[float, float, float]


class Quel1BackendExecutor:
    """QuEL-1 backend executor using `Quel1BackendController.execute_sequencer`."""

    def __init__(
        self,
        *,
        backend_controller: Quel1BackendController,
        execution_mode: Literal["legacy", "parallel"] = "parallel",
    ) -> None:
        """
        Initialize the backend executor.

        Parameters
        ----------
        backend_controller : Quel1BackendController
            Backend controller used to execute sequencers.
        execution_mode : {"legacy", "parallel"}, optional
            Execution path selector. ``"legacy"`` uses qubecalib's direct action
            flow, while ``"parallel"`` uses the qubex-side parallelized flow.
        """
        if execution_mode not in {"legacy", "parallel"}:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")
        self._backend_controller = backend_controller
        self._execution_mode = execution_mode

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute a prepared request on QuEL-1 hardware."""
        payload = request.payload
        if not isinstance(payload, Quel1ExecutionPayload):
            raise TypeError(
                "Quel1BackendExecutor expects `Quel1ExecutionPayload` payload."
            )
        execute_impl = (
            self._backend_controller.execute_sequencer
            if self._execution_mode == "legacy"
            else self._backend_controller.execute_sequencer_parallel
        )
        return execute_impl(
            sequencer=payload.sequencer,
            repeats=payload.repeats,
            integral_mode=payload.integral_mode,
            dsp_demodulation=payload.dsp_demodulation,
            enable_sum=payload.enable_sum,
            enable_classification=payload.enable_classification,
            line_param0=payload.line_param0,
            line_param1=payload.line_param1,
        )
