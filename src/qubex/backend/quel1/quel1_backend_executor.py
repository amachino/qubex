"""QuEL backend execution implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .quel1_backend_constants import (
    DEFAULT_CLOCK_HEALTH_CHECKS,
    DEFAULT_EXECUTION_MODE,
    ExecutionMode,
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
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> None:
        """
        Initialize the backend executor.

        Parameters
        ----------
        backend_controller : Quel1BackendController
            Backend controller used to execute sequencers.
        execution_mode : ExecutionMode | None, optional
            Execution path selector. ``"legacy"`` uses qubecalib's direct action
            flow, while ``"parallel"`` uses the qubex-side parallelized flow.
            If ``None``, ``qubex.backend.quel1.DEFAULT_EXECUTION_MODE`` is used.
        clock_health_checks : bool | None, optional
            Whether to enable clock-health checks in parallel execution.
            If ``None``, ``qubex.backend.quel1.DEFAULT_CLOCK_HEALTH_CHECKS`` is used.
        """
        if execution_mode is None:
            execution_mode = DEFAULT_EXECUTION_MODE
        if clock_health_checks is None:
            clock_health_checks = DEFAULT_CLOCK_HEALTH_CHECKS
        if execution_mode not in {"legacy", "parallel"}:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")
        self._backend_controller = backend_controller
        self._execution_mode = execution_mode
        self._clock_health_checks = clock_health_checks

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
        execute_kwargs = dict(
            sequencer=payload.sequencer,
            repeats=payload.repeats,
            integral_mode=payload.integral_mode,
            dsp_demodulation=payload.dsp_demodulation,
            enable_sum=payload.enable_sum,
            enable_classification=payload.enable_classification,
            line_param0=payload.line_param0,
            line_param1=payload.line_param1,
        )
        if self._execution_mode == "parallel":
            execute_kwargs["clock_health_checks"] = self._clock_health_checks
        return execute_impl(**execute_kwargs)
