"""Measurement backend adapter protocol and compatibility exports."""

from __future__ import annotations

from typing import Protocol

from qubex.backend import BackendExecutionRequest
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule

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
    ) -> BackendExecutionRequest:
        """Build backend execution request from measurement schedule/config."""
        ...


__all__ = [
    "MeasurementBackendAdapter",
    "Quel1MeasurementBackendAdapter",
    "Quel3MeasurementBackendAdapter",
]
