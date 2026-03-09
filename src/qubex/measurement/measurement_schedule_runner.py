"""Measurement schedule execution orchestrator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from qubex.backend import (
    BackendController,
    BackendExecutionRequest,
)
from qubex.backend.quel1 import (
    ExecutionMode,
    Quel1BackendController,
)
from qubex.backend.quel3 import Quel3BackendController
from qubex.system import ExperimentSystem

from .adapters import (
    MeasurementBackendAdapter,
    Quel1MeasurementBackendAdapter,
    Quel3MeasurementBackendAdapter,
)
from .measurement_constraint_profile import MeasurementConstraintProfile
from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult
from .models.measurement_schedule import MeasurementSchedule
from .models.quel1_measurement_options import Quel1MeasurementOptions


class MeasurementScheduleRunner:
    """Execute measurement schedules with adapter and backend controller."""

    def __init__(
        self,
        *,
        measurement_backend_adapter: MeasurementBackendAdapter | None = None,
        backend_controller: BackendController,
        experiment_system: ExperimentSystem | None = None,
        constraint_profile: MeasurementConstraintProfile | None = None,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> None:
        if measurement_backend_adapter is None:
            if experiment_system is None:
                raise ValueError(
                    "experiment_system is required when "
                    "measurement_backend_adapter is not provided."
                )
            if isinstance(backend_controller, Quel3BackendController):
                constraint_profile = MeasurementConstraintProfile.quel3(
                    backend_controller.sampling_period_ns
                )
                measurement_backend_adapter = Quel3MeasurementBackendAdapter(
                    backend_controller=backend_controller,
                    experiment_system=experiment_system,
                    constraint_profile=constraint_profile,
                )
            elif isinstance(backend_controller, Quel1BackendController):
                constraint_profile = MeasurementConstraintProfile.quel1(
                    backend_controller.sampling_period_ns
                )
                measurement_backend_adapter = Quel1MeasurementBackendAdapter(
                    backend_controller=backend_controller,
                    experiment_system=experiment_system,
                    constraint_profile=constraint_profile,
                )
            else:
                raise TypeError(
                    "Unsupported backend controller for measurement adapter selection."
                )

        self._measurement_backend_adapter: MeasurementBackendAdapter = cast(
            MeasurementBackendAdapter,
            measurement_backend_adapter,
        )
        self._backend_controller = backend_controller
        self._constraint_profile = constraint_profile
        self._execution_mode: ExecutionMode | None = execution_mode
        self._clock_health_checks: bool | None = clock_health_checks

    def _build_result(
        self,
        *,
        backend_result: object,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        """Build canonical measurement result via measurement backend adapter."""
        if isinstance(backend_result, MeasurementResult):
            return backend_result

        box_config = getattr(self._backend_controller, "box_config", None)
        if isinstance(box_config, Mapping):
            device_config: dict[str, Any] = dict(box_config)
        else:
            device_config = {}

        capture_decimation_factor = getattr(
            self._backend_controller,
            "CAPTURE_DECIMATION_FACTOR",
            None,
        )
        if not (
            isinstance(capture_decimation_factor, int) and capture_decimation_factor > 0
        ):
            raise ValueError(
                "backend_controller.CAPTURE_DECIMATION_FACTOR must be a positive integer."
            )

        sampling_period = self._backend_controller.sampling_period_ns
        if config.shot_averaging:
            sampling_period = sampling_period * capture_decimation_factor

        return self._measurement_backend_adapter.build_measurement_result(
            backend_result=backend_result,
            measurement_config=config,
            device_config=device_config,
            sampling_period=sampling_period,
        )

    def execute_sync(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        """Execute a measurement schedule synchronously with the given configuration."""
        request = self._prepare_execution(
            schedule=schedule,
            config=config,
            quel1_options=quel1_options,
        )
        if self._execution_mode is None and self._clock_health_checks is None:
            backend_result = self._backend_controller.execute_sync(request=request)
        else:
            backend_result = self._backend_controller.execute_sync(
                request=request,
                execution_mode=self._execution_mode,
                clock_health_checks=self._clock_health_checks,
            )
        return self._build_result(backend_result=backend_result, config=config)

    async def execute_async(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        """
        Execute a measurement schedule with the given configuration.

        Parameters
        ----------
        schedule : MeasurementSchedule
            Measurement schedule.

        config : MeasurementConfig
            Measurement configuration.


        Returns
        -------
        MeasurementResult
            Measurement result.

        """
        request = self._prepare_execution(
            schedule=schedule,
            config=config,
            quel1_options=quel1_options,
        )
        if self._execution_mode is None and self._clock_health_checks is None:
            backend_result = await self._backend_controller.execute_async(
                request=request
            )
        else:
            backend_result = await self._backend_controller.execute_async(
                request=request,
                execution_mode=self._execution_mode,
                clock_health_checks=self._clock_health_checks,
            )
        return self._build_result(
            backend_result=backend_result,
            config=config,
        )

    def _prepare_execution(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> BackendExecutionRequest:
        """Validate one execution request and return backend request."""
        self._measurement_backend_adapter.validate_schedule(schedule)
        if quel1_options is None:
            request = self._measurement_backend_adapter.build_execution_request(
                schedule=schedule,
                config=config,
            )
        else:
            request = self._measurement_backend_adapter.build_execution_request(
                schedule=schedule,
                config=config,
                quel1_options=quel1_options,
            )
        return request
