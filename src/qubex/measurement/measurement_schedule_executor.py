"""Measurement schedule execution orchestrator."""

from __future__ import annotations

from qubex.backend import (
    BackendExecutor,
    ExperimentSystem,
)
from qubex.backend.quel1 import (
    SAMPLING_PERIOD,
    ExecutionMode,
    Quel1BackendController,
    Quel1BackendExecutor,
)

from .measurement_backend_adapter import (
    MeasurementBackendAdapter,
    Quel1MeasurementBackendAdapter,
)
from .measurement_result_factory import MeasurementResultFactory
from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult
from .models.measurement_schedule import MeasurementSchedule


class MeasurementScheduleExecutor:
    """Execute measurement schedules with adapter/executor/result factory."""

    def __init__(
        self,
        *,
        backend_executor: BackendExecutor,
        measurement_backend_adapter: MeasurementBackendAdapter,
        measurement_result_factory: MeasurementResultFactory,
        backend_controller: Quel1BackendController,
    ) -> None:
        self._backend_executor = backend_executor
        self._measurement_backend_adapter = measurement_backend_adapter
        self._measurement_result_factory = measurement_result_factory
        self._backend_controller = backend_controller

    @classmethod
    def create_default(
        cls,
        *,
        backend_controller: Quel1BackendController,
        experiment_system: ExperimentSystem,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> MeasurementScheduleExecutor:
        """
        Create the default QuEL-backed schedule executor.

        Parameters
        ----------
        backend_controller : Quel1BackendController
            Backend controller bound to connected hardware.
        experiment_system : ExperimentSystem
            Experiment-system model used by adapter/result conversion.
        execution_mode : ExecutionMode | None, optional
            Backend execution mode.
        clock_health_checks : bool | None, optional
            Whether to enable additional clock-health I/O in parallel mode.
        """
        return cls(
            backend_executor=Quel1BackendExecutor(
                backend_controller=backend_controller,
                execution_mode=execution_mode,
                clock_health_checks=clock_health_checks,
            ),
            measurement_backend_adapter=Quel1MeasurementBackendAdapter(
                backend_controller=backend_controller,
                experiment_system=experiment_system,
                sampling_period=cls._resolve_sampling_period_ns(backend_controller),
            ),
            measurement_result_factory=MeasurementResultFactory(
                experiment_system=experiment_system,
            ),
            backend_controller=backend_controller,
        )

    @staticmethod
    def _resolve_sampling_period_ns(
        backend_controller: Quel1BackendController,
    ) -> float:
        """Resolve sampling period (ns) from backend-controller contract."""
        sampling_period = getattr(backend_controller, "DEFAULT_SAMPLING_PERIOD", None)
        if isinstance(sampling_period, (int, float)):
            return float(sampling_period)
        return SAMPLING_PERIOD

    def execute(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        """
        Execute a measurement schedule with the given configuration.

        Parameters
        ----------
        schedule : MeasurementSchedule
            The measurement schedule.
        config : MeasurementConfig
            The measurement configuration.

        Returns
        -------
        MeasurementResult
            The measurement result.
        """
        self._measurement_backend_adapter.validate_schedule(schedule)
        request = self._measurement_backend_adapter.build_execution_request(
            schedule=schedule,
            config=config,
        )
        backend_result = self._backend_executor.execute(
            request=request,
        )
        result = self._measurement_result_factory.create(
            backend_result=backend_result,
            measurement_config=config,
            device_config=self._backend_controller.box_config,
        )
        return result
