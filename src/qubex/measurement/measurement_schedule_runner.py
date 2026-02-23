"""Measurement schedule execution orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from qubex.backend import (
    BackendController,
    ExperimentSystem,
)
from qubex.backend.quel1 import (
    ExecutionMode,
)

from .adapters import (
    MeasurementBackendAdapter,
    Quel1MeasurementBackendAdapter,
    Quel3MeasurementBackendAdapter,
)
from .measurement_constraint_profile import MeasurementConstraintProfile
from .measurement_result_factory import MeasurementResultFactory
from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult
from .models.measurement_schedule import MeasurementSchedule


class MeasurementScheduleRunner:
    """Execute measurement schedules with adapter/executor/result factory."""

    def __init__(
        self,
        *,
        measurement_backend_adapter: MeasurementBackendAdapter,
        measurement_result_factory: MeasurementResultFactory,
        backend_controller: BackendController,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> None:
        self._measurement_backend_adapter = measurement_backend_adapter
        self._measurement_result_factory = measurement_result_factory
        self._backend_controller = backend_controller
        self._execution_mode = execution_mode
        self._clock_health_checks = clock_health_checks

    @classmethod
    def create_default(
        cls,
        *,
        backend_controller: BackendController,
        experiment_system: ExperimentSystem,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> MeasurementScheduleRunner:
        """
        Create the default QuEL-backed schedule executor.

        Parameters
        ----------
        backend_controller : BackendController
            Backend controller bound to connected hardware.
        experiment_system : ExperimentSystem
            Experiment-system model used by adapter/result conversion.
        execution_mode : ExecutionMode | None, optional
            Backend execution mode.
        clock_health_checks : bool | None, optional
            Whether to enable additional clock-health I/O in parallel mode.
        """
        backend_kind = cls._resolve_backend_kind(backend_controller)
        constraint_profile = cls._resolve_constraint_profile(backend_controller)
        return cls(
            measurement_backend_adapter=cls._create_backend_adapter(
                backend_controller=backend_controller,
                experiment_system=experiment_system,
                constraint_profile=constraint_profile,
                backend_kind=backend_kind,
            ),
            measurement_result_factory=cls._create_result_factory(
                backend_controller=backend_controller,
                experiment_system=experiment_system,
            ),
            backend_controller=backend_controller,
            execution_mode=execution_mode,
            clock_health_checks=clock_health_checks,
        )

    @staticmethod
    def _create_backend_adapter(
        *,
        backend_controller: BackendController,
        experiment_system: ExperimentSystem,
        constraint_profile: MeasurementConstraintProfile,
        backend_kind: str,
    ) -> MeasurementBackendAdapter:
        """Create backend adapter with optional backend-specific factory hook."""
        factory = getattr(
            backend_controller,
            "create_measurement_backend_adapter",
            None,
        )
        if isinstance(factory, Callable):
            return factory(
                experiment_system=experiment_system,
                constraint_profile=constraint_profile,
            )
        if backend_kind == "quel3":
            return Quel3MeasurementBackendAdapter(
                backend_controller=backend_controller,
                experiment_system=experiment_system,
                constraint_profile=constraint_profile,
            )
        return Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, backend_controller),
            experiment_system=experiment_system,
            constraint_profile=constraint_profile,
        )

    @staticmethod
    def _create_result_factory(
        *,
        backend_controller: BackendController,
        experiment_system: ExperimentSystem,
    ) -> MeasurementResultFactory:
        """Create result factory with optional backend-specific factory hook."""
        factory = getattr(
            backend_controller,
            "create_measurement_result_factory",
            None,
        )
        if isinstance(factory, Callable):
            return factory(experiment_system=experiment_system)
        return MeasurementResultFactory(
            experiment_system=experiment_system,
        )

    @staticmethod
    def _resolve_sampling_period_ns(
        backend_controller: BackendController,
    ) -> float:
        """Resolve sampling period (ns) from backend-controller contract."""
        return backend_controller.sampling_period

    @classmethod
    def _resolve_constraint_profile(
        cls,
        backend_controller: BackendController,
    ) -> MeasurementConstraintProfile:
        """Resolve backend constraint profile from controller capability hints."""
        profile = getattr(backend_controller, "MEASUREMENT_CONSTRAINT_PROFILE", None)
        if isinstance(profile, MeasurementConstraintProfile):
            return profile

        sampling_period = cls._resolve_sampling_period_ns(backend_controller)
        mode = getattr(backend_controller, "MEASUREMENT_CONSTRAINT_MODE", "quel1")
        if mode == "quel3":
            return MeasurementConstraintProfile.quel3(sampling_period)
        return MeasurementConstraintProfile.quel1(sampling_period)

    @staticmethod
    def _resolve_backend_kind(backend_controller: BackendController) -> str:
        """Resolve backend kind hint used for default adapter selection."""
        backend_kind = getattr(backend_controller, "MEASUREMENT_BACKEND_KIND", None)
        if backend_kind in {"quel1", "quel3"}:
            return backend_kind
        return "quel1"

    @staticmethod
    def _resolve_avg_sample_stride(
        backend_controller: BackendController,
        constraint_profile: MeasurementConstraintProfile | None,
    ) -> int:
        """Resolve AVG-mode time-stride multiplier from backend capability hints."""
        backend_stride = getattr(
            backend_controller,
            "MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE",
            None,
        )
        if isinstance(backend_stride, int) and backend_stride > 0:
            return backend_stride
        if (
            isinstance(constraint_profile, MeasurementConstraintProfile)
            and constraint_profile.word_length_samples is not None
            and constraint_profile.word_length_samples > 0
        ):
            return int(constraint_profile.word_length_samples)
        return 4

    @staticmethod
    def _resolve_device_config(
        backend_controller: BackendController,
    ) -> dict[str, Any]:
        """Resolve backend device config if supported by the controller."""
        box_config = getattr(backend_controller, "box_config", None)
        if isinstance(box_config, dict):
            return box_config
        return {}

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
        if self._execution_mode is None and self._clock_health_checks is None:
            backend_result = self._backend_controller.execute(request=request)
        else:
            backend_result = cast(Any, self._backend_controller).execute(
                request=request,
                execution_mode=self._execution_mode,
                clock_health_checks=self._clock_health_checks,
            )
        if isinstance(backend_result, MeasurementResult):
            return backend_result
        result = self._measurement_result_factory.create(
            backend_result=backend_result,
            measurement_config=config,
            device_config=self._resolve_device_config(self._backend_controller),
            sampling_period_ns=getattr(
                self._measurement_backend_adapter,
                "sampling_period",
                self._resolve_sampling_period_ns(self._backend_controller),
            ),
            avg_sample_stride=self._resolve_avg_sample_stride(
                self._backend_controller,
                getattr(self._measurement_backend_adapter, "constraint_profile", None),
            ),
        )
        return result

    async def execute_async(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        """
        Execute a measurement schedule asynchronously with the given configuration.

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
        if self._execution_mode is None and self._clock_health_checks is None:
            backend_result = await self._backend_controller.execute_async(
                request=request
            )
        else:
            backend_result = await cast(Any, self._backend_controller).execute_async(
                request=request,
                execution_mode=self._execution_mode,
                clock_health_checks=self._clock_health_checks,
            )
        if isinstance(backend_result, MeasurementResult):
            return backend_result
        result = self._measurement_result_factory.create(
            backend_result=backend_result,
            measurement_config=config,
            device_config=self._resolve_device_config(self._backend_controller),
            sampling_period_ns=getattr(
                self._measurement_backend_adapter,
                "sampling_period",
                self._resolve_sampling_period_ns(self._backend_controller),
            ),
            avg_sample_stride=self._resolve_avg_sample_stride(
                self._backend_controller,
                getattr(self._measurement_backend_adapter, "constraint_profile", None),
            ),
        )
        return result
