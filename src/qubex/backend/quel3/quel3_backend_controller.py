"""QuEL-3 backend controller implemented through quelware-client."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)
from qubex.backend.controller_types import BackendController

from .managers.connection_manager import Quel3ConnectionManager
from .managers.execution_manager import ExecutionMode, Quel3ExecutionManager
from .quel3_execution_payload import Quel3ExecutionPayload
from .quel3_runtime_context import Quel3RuntimeContext

QUEL3_DEFAULT_SAMPLING_PERIOD_NS = 0.4


class Quel3BackendController(BackendController):
    """Control and execute QuEL-3 measurements through quelware-client."""

    MEASUREMENT_BACKEND_KIND: Literal["quel3"] = "quel3"
    MEASUREMENT_CONSTRAINT_MODE: Literal["quel3"] = "quel3"
    MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE: int = 4
    DEFAULT_SAMPLING_PERIOD: float = QUEL3_DEFAULT_SAMPLING_PERIOD_NS

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        sampling_period_ns: float | None = None,
        alias_map: Mapping[str, str] | None = None,
        quelware_endpoint: str | None = None,
        quelware_port: int | None = None,
    ) -> None:
        """
        Initialize a QuEL-3 backend controller.

        Parameters
        ----------
        config_path : str | Path | None, optional
            Reserved for API compatibility.
        sampling_period_ns : float | None, optional
            Session sampling period used by measurement-layer adapters.
        alias_map : Mapping[str, str] | None, optional
            Optional target-label to instrument-alias mapping.
        quelware_endpoint : str | None, optional
            Quelware API endpoint. Defaults to "localhost".
        quelware_port : int | None, optional
            Quelware API port. Defaults to 50051.
        """
        del config_path
        self._default_sampling_period = (
            sampling_period_ns
            if sampling_period_ns is not None
            else self.DEFAULT_SAMPLING_PERIOD
        )

        self._runtime_context = Quel3RuntimeContext(
            alias_map=dict(alias_map or {}),
            quelware_endpoint=(
                quelware_endpoint if quelware_endpoint is not None else "localhost"
            ),
            quelware_port=quelware_port if quelware_port is not None else 50051,
            default_sampling_period=self._default_sampling_period,
            measurement_result_avg_sample_stride=self.MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE,
        )
        self._connection_manager = Quel3ConnectionManager(
            runtime_context=self._runtime_context
        )
        self._execution_manager = Quel3ExecutionManager(
            runtime_context=self._runtime_context
        )

    @property
    def hash(self) -> int:
        """Return stable hash from runtime state."""
        return self._connection_manager.hash

    @property
    def is_connected(self) -> bool:
        """Return whether backend resources are connected."""
        return self._connection_manager.is_connected

    def connect(
        self,
        box_names: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """Connect backend resources for selected boxes."""
        self._connection_manager.connect(
            box_names=box_names,
            parallel=parallel,
        )

    def disconnect(self) -> None:
        """Disconnect backend resources."""
        self._connection_manager.disconnect()

    def set_instrument_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Replace full target-to-alias mapping for quelware execution."""
        self._connection_manager.set_alias_map(alias_map)

    def update_instrument_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Update target-to-alias mapping for quelware execution."""
        self._connection_manager.update_alias_map(alias_map)

    def resolve_instrument_alias(self, target: str) -> str:
        """Resolve quelware instrument alias for a measurement target."""
        return self._execution_manager.resolve_instrument_alias(target)

    @property
    def default_sampling_period(self) -> float:
        """Return instance-scoped default sampling period in ns."""
        return self._default_sampling_period

    def execute(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request using QuEL-3 execution defaults."""
        del execution_mode, clock_health_checks

        payload = request.payload
        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3 backend execution expects `Quel3ExecutionPayload` payload."
            )

        return self._execution_manager.execute_payload(payload=payload)
