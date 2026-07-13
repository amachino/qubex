"""Measurement backend adapter protocol and compatibility exports."""

from __future__ import annotations

from typing import Any, Protocol

from qubex.backend import BackendExecutionRequest
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions

from .quel1_backend_adapter import Quel1MeasurementBackendAdapter
from .quel3_backend_adapter import Quel3MeasurementBackendAdapter


class MeasurementBackendAdapter(Protocol):
    """Protocol for converting measurement requests into backend requests."""

    def validate_schedule(self, schedule: MeasurementSchedule) -> None:
        """Validate backend-specific constraints for a measurement schedule."""
        ...

    def build_execution_request(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> BackendExecutionRequest:
        """Build backend execution request from measurement schedule/config."""
        ...

    def build_measurement_result(
        self,
        *,
        backend_result: Any,
        measurement_config: MeasurementConfig,
        device_config: dict,
        sampling_period: float,
    ) -> MeasurementResult:
        """Build canonical result from a backend-specific result payload."""
        ...


__all__ = [
    "MeasurementBackendAdapter",
    "Quel1MeasurementBackendAdapter",
    "Quel3MeasurementBackendAdapter",
]
