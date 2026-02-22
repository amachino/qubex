"""QuEL backend execution implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)
from qubex.backend.quel1.compat.sequencer import Quel1Sequencer

from .quel1_backend_constants import (
    DEFAULT_CLOCK_HEALTH_CHECKS,
    DEFAULT_EXECUTION_MODE,
    ExecutionMode,
)
from .quel1_backend_controller import Quel1BackendController
from .quel1_execution_payload import Quel1ExecutionPayload

if TYPE_CHECKING:
    from .compat.qubecalib_protocols import (
        CapSampledSequenceProtocol,
        GenSampledSequenceProtocol,
        SequencerProtocol as Sequencer,
    )


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
            Execution path selector. `"serial"` uses qubecalib's direct action
            flow, while `"parallel"` uses the qubex-side parallelized flow.
            If `None`, `qubex.backend.quel1.DEFAULT_EXECUTION_MODE` is used.
        clock_health_checks : bool | None, optional
            Whether to enable clock-health checks in parallel execution.
            If `None`, `qubex.backend.quel1.DEFAULT_CLOCK_HEALTH_CHECKS` is used.
        """
        if execution_mode is None:
            execution_mode = DEFAULT_EXECUTION_MODE
        if clock_health_checks is None:
            clock_health_checks = DEFAULT_CLOCK_HEALTH_CHECKS
        if execution_mode not in {"serial", "parallel"}:
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
        sequencer = self._create_quel1_sequencer(
            gen_sampled_sequence=payload.gen_sampled_sequence,
            cap_sampled_sequence=payload.cap_sampled_sequence,
            resource_map=payload.resource_map,
            interval=payload.interval,
        )
        if self._execution_mode == "parallel":
            return self._backend_controller.execute_sequencer_parallel(
                sequencer=sequencer,
                repeats=payload.repeats,
                integral_mode=payload.integral_mode,
                dsp_demodulation=payload.dsp_demodulation,
                enable_sum=payload.enable_sum,
                enable_classification=payload.enable_classification,
                line_param0=payload.line_param0,
                line_param1=payload.line_param1,
                clock_health_checks=self._clock_health_checks,
            )
        return self._backend_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=payload.repeats,
            integral_mode=payload.integral_mode,
            dsp_demodulation=payload.dsp_demodulation,
            enable_sum=payload.enable_sum,
            enable_classification=payload.enable_classification,
            line_param0=payload.line_param0,
            line_param1=payload.line_param1,
        )

    def _create_quel1_sequencer(
        self,
        *,
        gen_sampled_sequence: dict[str, GenSampledSequenceProtocol],
        cap_sampled_sequence: dict[str, CapSampledSequenceProtocol],
        resource_map: dict[str, list[dict]],
        interval: int,
    ) -> Sequencer:
        """Create QuEL-1 sequencer instance from prepared execution payload."""
        return Quel1Sequencer(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map=resource_map,  # type: ignore[arg-type]
            interval=interval,
            sysdb=self._backend_controller.qubecalib.sysdb,
            # Keep passing connected system for constructor compatibility across
            # old/new driver packages.
            driver=self._backend_controller.quel1system,
        )
