"""Measurement schedule execution orchestrator."""

from __future__ import annotations

from collections.abc import Callable

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
from .measurement_constraint_profile import MeasurementConstraintProfile
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
        constraint_profile = cls._resolve_constraint_profile(backend_controller)
        return cls(
            backend_executor=cls._create_backend_executor(
                backend_controller=backend_controller,
                execution_mode=execution_mode,
                clock_health_checks=clock_health_checks,
            ),
            measurement_backend_adapter=cls._create_backend_adapter(
                backend_controller=backend_controller,
                experiment_system=experiment_system,
                constraint_profile=constraint_profile,
            ),
            measurement_result_factory=cls._create_result_factory(
                backend_controller=backend_controller,
                experiment_system=experiment_system,
            ),
            backend_controller=backend_controller,
        )

    @staticmethod
    def _create_backend_executor(
        *,
        backend_controller: Quel1BackendController,
        execution_mode: ExecutionMode | None,
        clock_health_checks: bool | None,
    ) -> BackendExecutor:
        """Create backend executor with optional backend-specific factory hook."""
        factory = getattr(
            backend_controller,
            "create_measurement_backend_executor",
            None,
        )
        if isinstance(factory, Callable):
            return factory(
                execution_mode=execution_mode,
                clock_health_checks=clock_health_checks,
            )
        return Quel1BackendExecutor(
            backend_controller=backend_controller,
            execution_mode=execution_mode,
            clock_health_checks=clock_health_checks,
        )

    @staticmethod
    def _create_backend_adapter(
        *,
        backend_controller: Quel1BackendController,
        experiment_system: ExperimentSystem,
        constraint_profile: MeasurementConstraintProfile,
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
        return Quel1MeasurementBackendAdapter(
            backend_controller=backend_controller,
            experiment_system=experiment_system,
            constraint_profile=constraint_profile,
        )

    @staticmethod
    def _create_result_factory(
        *,
        backend_controller: Quel1BackendController,
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
        backend_controller: Quel1BackendController,
    ) -> float:
        """Resolve sampling period (ns) from backend-controller contract."""
        sampling_period = getattr(backend_controller, "DEFAULT_SAMPLING_PERIOD", None)
        if isinstance(sampling_period, (int, float)):
            return float(sampling_period)
        return SAMPLING_PERIOD

    @classmethod
    def _resolve_constraint_profile(
        cls,
        backend_controller: Quel1BackendController,
    ) -> MeasurementConstraintProfile:
        """Resolve backend constraint profile from controller capability hints."""
        profile = getattr(backend_controller, "MEASUREMENT_CONSTRAINT_PROFILE", None)
        if isinstance(profile, MeasurementConstraintProfile):
            return profile

        sampling_period = cls._resolve_sampling_period_ns(backend_controller)
        mode = getattr(backend_controller, "MEASUREMENT_CONSTRAINT_MODE", "strict")
        if mode == "relaxed":
            return MeasurementConstraintProfile.relaxed(sampling_period)
        return MeasurementConstraintProfile.strict_quel1(sampling_period)

    @staticmethod
    def _resolve_avg_sample_stride(
        backend_controller: Quel1BackendController,
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
