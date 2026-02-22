"""Execution manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
    BackendExecutor,
)
from qubex.backend.quel1.compat.parallel_action_builder import (
    ClockHealthCheckOptions,
)
from qubex.backend.quel1.compat.qubecalib_protocols import SequencerProtocol
from qubex.backend.quel1.compat.sequencer_execution_engine import (
    ActionBuilder,
    AwgIdFactory,
    AwgSettingFactory,
    RunitIdFactory,
    RunitSettingFactory,
    SequencerExecutionEngine,
)
from qubex.backend.quel1.quel1_backend_raw_result import Quel1BackendRawResult
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        Quel1SystemProtocol as Quel1System,
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
        make_backend_raw_result: Callable[..., Quel1BackendRawResult],
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
        make_backend_raw_result : Callable[..., Quel1BackendRawResult]
            Result-container factory.

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
        action_builder: ActionBuilder,
        runit_setting_factory: RunitSettingFactory,
        runit_id_factory: RunitIdFactory,
        awg_setting_factory: AwgSettingFactory,
        awg_id_factory: AwgIdFactory,
        make_backend_raw_result: Callable[..., Quel1BackendRawResult],
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
        action_builder : ActionBuilder
            Action builder callable.
        runit_setting_factory : RunitSettingFactory
            Runit-setting factory.
        runit_id_factory : RunitIdFactory
            Runit-id factory.
        awg_setting_factory : AwgSettingFactory
            AWG-setting factory.
        awg_id_factory : AwgIdFactory
            AWG-id factory.
        make_backend_raw_result : Callable[..., Quel1BackendRawResult]
            Result-container factory.

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
                action_builder=action_builder,
                runit_setting_factory=runit_setting_factory,
                runit_id_factory=runit_id_factory,
                awg_setting_factory=awg_setting_factory,
                awg_id_factory=awg_id_factory,
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
