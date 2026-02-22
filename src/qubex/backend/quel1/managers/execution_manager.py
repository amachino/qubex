"""Execution manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
    BackendExecutor,
)
from qubex.backend.quel1.compat.parallel_action_builder import (
    ClockHealthCheckOptions,
)
from qubex.backend.quel1.compat.qubecalib_protocols import SequencerProtocol
from qubex.backend.quel1.compat.sequencer import Quel1Sequencer
from qubex.backend.quel1.compat.sequencer_execution_engine import (
    SequencerExecutionEngine,
)
from qubex.backend.quel1.quel1_backend_raw_result import (
    Quel1BackendRawResult,
    make_backend_raw_result,
)
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        CapSampledSequenceProtocol,
        GenSampledSequenceProtocol,
        Quel1SystemProtocol as Quel1System,
        SequencerProtocol as Sequencer,
    )


class Quel1ExecutionManager:
    """Handle backend execution entrypoints for QuEL-1 controller."""

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context

    def execute(
        self,
        *,
        request: BackendExecutionRequest,
        executor: BackendExecutor,
    ) -> BackendExecutionResult:
        """
        Execute a prepared backend request with a configured executor.

        Parameters
        ----------
        request : BackendExecutionRequest
            Backend execution request.
        executor : BackendExecutor
            Backend executor that handles request execution.

        Returns
        -------
        BackendExecutionResult
            Backend-specific execution result.
        """
        return executor.execute(request=request)

    def create_quel1_sequencer(
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

    def execute_sequencer(
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
    ) -> Quel1BackendRawResult:
        """
        Execute a sequencer through serial qubecalib path.

        Parameters
        ----------
        sequencer : SequencerProtocol
            Sequencer to execute.
        repeats : int
            Repeat count.
        integral_mode : str
            Integral mode.
        dsp_demodulation : bool
            DSP demodulation enable flag.
        software_demodulation : bool
            Software demodulation enable flag.
        enable_sum : bool
            DSP sum enable flag.
        enable_classification : bool
            DSP classification enable flag.
        line_param0 : tuple[float, float, float] | None
            Classifier line parameter 0.
        line_param1 : tuple[float, float, float] | None
            Classifier line parameter 1.

        Returns
        -------
        Quel1BackendRawResult
            Parsed backend result.
        """
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
        status, data, config = sequencer.execute(self._require_boxpool())
        return make_backend_raw_result(
            status=status,
            data=data,
            config=config,
        )

    def execute_sequencer_parallel(
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
    ) -> Quel1BackendRawResult:
        """
        Execute a sequencer through parallelized multi-box action path.

        Parameters
        ----------
        sequencer : SequencerProtocol
            Sequencer to execute.
        repeats : int
            Repeat count.
        integral_mode : str
            Integral mode.
        dsp_demodulation : bool
            DSP demodulation enable flag.
        software_demodulation : bool
            Software demodulation enable flag.
        enable_sum : bool
            DSP sum enable flag.
        enable_classification : bool
            DSP classification enable flag.
        line_param0 : tuple[float, float, float] | None
            Classifier line parameter 0.
        line_param1 : tuple[float, float, float] | None
            Classifier line parameter 1.
        clock_health_checks : bool
            Whether to enable clock health diagnostics.

        Returns
        -------
        Quel1BackendRawResult
            Parsed backend result.
        """
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
                boxpool=self._require_boxpool(),
                system=self._require_quel1system(),
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
        return make_backend_raw_result(
            status=parsed_status,
            data=parsed_data,
            config=parsed_config,
        )

    def _require_boxpool(self) -> BoxPool:
        """Return connected boxpool or raise when runtime is disconnected."""
        return self._runtime_context.boxpool

    def _require_quel1system(self) -> Quel1System:
        """Return connected Quel1System or raise when runtime is disconnected."""
        return self._runtime_context.quel1system
