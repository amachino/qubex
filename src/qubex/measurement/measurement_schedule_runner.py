"""Measurement schedule execution orchestrator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from qubex.backend import (
    BackendController,
    ExperimentSystem,
)
from qubex.backend.quel1 import (
    ExecutionMode,
    Quel1BackendController,
)
from qubex.backend.quel3 import Quel3BackendController

from .adapters import (
    MeasurementBackendAdapter,
    Quel1MeasurementBackendAdapter,
    Quel3MeasurementBackendAdapter,
)
from .measurement_constraint_profile import MeasurementConstraintProfile
from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult
from .models.measurement_schedule import MeasurementSchedule


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
                    backend_controller.sampling_period
                )
                measurement_backend_adapter = Quel3MeasurementBackendAdapter(
                    backend_controller=backend_controller,
                    experiment_system=experiment_system,
                    constraint_profile=constraint_profile,
                )
            elif isinstance(backend_controller, Quel1BackendController):
                constraint_profile = MeasurementConstraintProfile.quel1(
                    backend_controller.sampling_period
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

        self._measurement_backend_adapter = measurement_backend_adapter
        self._backend_controller = backend_controller
        self._constraint_profile = constraint_profile
        self._execution_mode = execution_mode
        self._clock_health_checks = clock_health_checks

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

        backend_stride = getattr(
            self._backend_controller,
            "MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE",
            None,
        )
        if isinstance(backend_stride, int) and backend_stride > 0:
            avg_sample_stride = backend_stride
        else:
            word_length_samples = (
                self._constraint_profile.word_length_samples
                if isinstance(self._constraint_profile, MeasurementConstraintProfile)
                else None
            )
            avg_sample_stride = (
                int(word_length_samples)
                if isinstance(word_length_samples, int) and word_length_samples > 0
                else 4
            )

        return self._measurement_backend_adapter.build_measurement_result(
            backend_result=backend_result,
            measurement_config=config,
            device_config=device_config,
            sampling_period_ns=self._backend_controller.sampling_period,
            avg_sample_stride=avg_sample_stride,
        )

    async def execute(
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
        options: dict[str, object] = {}
        if self._execution_mode is not None:
            options["execution_mode"] = self._execution_mode
        if self._clock_health_checks is not None:
            options["clock_health_checks"] = self._clock_health_checks
        if not options:
            backend_result = await self._backend_controller.execute(request=request)
        else:
            backend_result = await cast(Any, self._backend_controller).execute(
                request=request,
                **options,
            )
        return self._build_result(
            backend_result=backend_result,
            config=config,
        )
