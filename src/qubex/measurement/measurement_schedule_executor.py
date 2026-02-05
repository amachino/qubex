"""Measurement schedule execution orchestrator."""

from __future__ import annotations

from qubex.backend import (
    BackendExecutor,
    DeviceController,
    ExperimentSystem,
    Quel1BackendExecutor,
)

from .measurement_backend_adapter import (
    MeasurementBackendAdapter,
    QuelMeasurementBackendAdapter,
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
        device_controller: DeviceController,
    ) -> None:
        self._backend_executor = backend_executor
        self._measurement_backend_adapter = measurement_backend_adapter
        self._measurement_result_factory = measurement_result_factory
        self._device_controller = device_controller

    @classmethod
    def create_default(
        cls,
        *,
        device_controller: DeviceController,
        experiment_system: ExperimentSystem,
    ) -> MeasurementScheduleExecutor:
        """Create the default QuEL-backed schedule executor."""
        return cls(
            backend_executor=Quel1BackendExecutor(
                device_controller=device_controller,
            ),
            measurement_backend_adapter=QuelMeasurementBackendAdapter(
                device_controller=device_controller,
                experiment_system=experiment_system,
            ),
            measurement_result_factory=MeasurementResultFactory(
                experiment_system=experiment_system,
            ),
            device_controller=device_controller,
        )

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
            device_config=self._device_controller.box_config,
        )
        return result
