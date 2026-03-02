"""Execution manager for QuEL-1 backend controller."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from qubex.backend.backend_controller import (
    BackendExecutionRequest,
    BackendExecutionResult,
)
from qubex.backend.quel1.compat.parallel_action_builder import (
    ClockHealthCheckOptions,
)
from qubex.backend.quel1.compat.qubecalib_protocols import SequencerProtocol
from qubex.backend.quel1.compat.sequencer import Quel1Sequencer
from qubex.backend.quel1.compat.sequencer_execution_engine import (
    SequencerExecutionEngine,
)
from qubex.backend.quel1.quel1_backend_constants import (
    DEFAULT_CLOCK_HEALTH_CHECKS,
    DEFAULT_EXECUTION_MODE,
    ExecutionMode,
)
from qubex.backend.quel1.quel1_backend_execution_result import (
    Quel1BackendExecutionResult,
)
from qubex.backend.quel1.quel1_execution_payload import Quel1ExecutionPayload
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        CapSampledSequenceProtocol,
        GenSampledSequenceProtocol,
        SequencerProtocol as Sequencer,
    )


class Quel1ExecutionManager:
    """Handle backend execution entrypoints for QuEL-1 controller."""

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context

    def execute_sync(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """
        Execute a prepared backend request.

        Parameters
        ----------
        request : BackendExecutionRequest
            Backend execution request.
        execution_mode : ExecutionMode | None, optional
            Backend execution mode selector.
        clock_health_checks : bool | None, optional
            Whether to enable clock diagnostics on parallel path.

        Returns
        -------
        BackendExecutionResult
            Backend-specific execution result.
        """
        payload = request.payload
        if not isinstance(payload, Quel1ExecutionPayload):
            raise TypeError(
                "Quel1ExecutionManager expects `Quel1ExecutionPayload` payload."
            )
        mode = execution_mode or DEFAULT_EXECUTION_MODE
        if mode not in {"serial", "parallel"}:
            raise ValueError(f"Unsupported execution mode: {mode}")
        if clock_health_checks is None:
            clock_health_checks = DEFAULT_CLOCK_HEALTH_CHECKS
        sequencer = self._create_quel1_sequencer(
            gen_sampled_sequence=payload.gen_sampled_sequence,
            cap_sampled_sequence=payload.cap_sampled_sequence,
            resource_map=payload.resource_map,
            interval=payload.interval,
        )
        if mode == "parallel":
            return self._execute_sequencer_parallel(
                sequencer=sequencer,
                repeats=payload.repeats,
                integral_mode=payload.integral_mode,
                dsp_demodulation=payload.dsp_demodulation,
                software_demodulation=False,
                enable_sum=payload.enable_sum,
                enable_classification=payload.enable_classification,
                line_param0=payload.line_param0,
                line_param1=payload.line_param1,
                clock_health_checks=clock_health_checks,
            )
        return self._execute_sequencer(
            sequencer=sequencer,
            repeats=payload.repeats,
            integral_mode=payload.integral_mode,
            dsp_demodulation=payload.dsp_demodulation,
            software_demodulation=False,
            enable_sum=payload.enable_sum,
            enable_classification=payload.enable_classification,
            line_param0=payload.line_param0,
            line_param1=payload.line_param1,
        )

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """
        Execute a prepared backend request asynchronously.

        Parameters
        ----------
        request : BackendExecutionRequest
            Backend execution request.
        execution_mode : ExecutionMode | None, optional
            Backend execution mode selector.
        clock_health_checks : bool | None, optional
            Whether to enable clock diagnostics on parallel path.

        Returns
        -------
        BackendExecutionResult
            Backend-specific execution result.
        """
        return await asyncio.to_thread(
            self.execute_sync,
            request=request,
            execution_mode=execution_mode,
            clock_health_checks=clock_health_checks,
        )

    def _create_quel1_sequencer(
        self,
        *,
        gen_sampled_sequence: dict[str, GenSampledSequenceProtocol],
        cap_sampled_sequence: dict[str, CapSampledSequenceProtocol],
        resource_map: dict[str, list[dict[str, Any]]],
        interval: int,
    ) -> Sequencer:
        """Create QuEL-1 sequencer instance from prepared execution payload."""
        return Quel1Sequencer(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map=resource_map,  # type: ignore[arg-type]
            interval=interval,
            sysdb=self._runtime_context.qubecalib.sysdb,
            # Keep passing connected system for constructor compatibility across
            # old/new driver packages.
            driver=self._runtime_context.quel1system,
        )

    def _execute_sequencer(
        self,
        *,
        sequencer: SequencerProtocol,
        repeats: int,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
        enable_sum: bool,
        enable_classification: bool,
        line_param0: tuple[float, float, float] | None,
        line_param1: tuple[float, float, float] | None,
    ) -> Quel1BackendExecutionResult:
        """Execute a sequencer through serial qubecalib path."""
        SequencerExecutionEngine.set_measurement_options(
            sequencer=sequencer,
            repeats=repeats,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
            enable_sum=enable_sum,
            enable_classification=enable_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
        status, data, config = sequencer.execute(self._runtime_context.boxpool)
        return Quel1BackendExecutionResult(
            status=status,
            data=data,
            config=config,
        )

    def _execute_sequencer_parallel(
        self,
        *,
        sequencer: SequencerProtocol,
        repeats: int,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
        enable_sum: bool,
        enable_classification: bool,
        line_param0: tuple[float, float, float] | None,
        line_param1: tuple[float, float, float] | None,
        clock_health_checks: bool,
    ) -> Quel1BackendExecutionResult:
        """Execute a sequencer through parallelized multi-box action path."""
        SequencerExecutionEngine.set_measurement_options(
            sequencer=sequencer,
            repeats=repeats,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
            enable_sum=enable_sum,
            enable_classification=enable_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
        parsed_status, parsed_data, parsed_config = (
            SequencerExecutionEngine.execute_parallel(
                sequencer=sequencer,
                boxpool=self._runtime_context.boxpool,
                system=self._runtime_context.quel1system,
                action_builder=self._runtime_context.driver.Action.build,
                runit_setting_factory=self._runtime_context.driver.RunitSetting,
                runit_id_factory=self._runtime_context.driver.RunitId,
                awg_setting_factory=self._runtime_context.driver.AwgSetting,
                awg_id_factory=self._runtime_context.driver.AwgId,
                logger=logger,
                clock_health_checks=(
                    None
                    if not clock_health_checks
                    else ClockHealthCheckOptions(
                        read_master_clock=True,
                        read_box_latched_clock_on_build=True,
                        measure_average_sysref_offset=True,
                        validate_sysref_fluctuation_on_emit=True,
                    )
                ),
            )
        )
        return Quel1BackendExecutionResult(
            status=parsed_status,
            data=parsed_data,
            config=parsed_config,
        )
