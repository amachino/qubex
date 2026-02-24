"""
QuEL-3 backend controller implementing the shared measurement-facing contract.

This module defines the QuEL-3 concrete `BackendController` implementation
built on quelware-client managers and runtime context.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from qubex.backend.backend_controller import (
    BackendController,
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .managers.connection_manager import Quel3ConnectionManager
from .managers.execution_manager import Quel3ExecutionManager
from .quel3_runtime_context import Quel3RuntimeContext

QUEL3_SAMPLING_PERIOD_NS = 0.4


class Quel3BackendController(BackendController):
    """
    QuEL-3 backend controller for session lifecycle and execution dispatch.

    The controller provides the required shared `BackendController` API for the
    measurement layer and routes concrete operations to QuEL-3 manager classes.
    Backend-specific capabilities are intentionally kept outside the shared
    contract.
    """

    MEASUREMENT_BACKEND_KIND: Literal["quel3"] = "quel3"
    MEASUREMENT_CONSTRAINT_MODE: Literal["quel3"] = "quel3"
    MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE: int = 4

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
        # Kept for API compatibility with QuEL-1 constructor signature.
        del config_path
        sampling_period = (
            sampling_period_ns
            if sampling_period_ns is not None
            else QUEL3_SAMPLING_PERIOD_NS
        )

        self._runtime_context = Quel3RuntimeContext(
            alias_map=dict(alias_map or {}),
            quelware_endpoint=(
                quelware_endpoint if quelware_endpoint is not None else "localhost"
            ),
            quelware_port=quelware_port if quelware_port is not None else 50051,
            sampling_period=sampling_period,
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

    @property
    def instrument_alias_map(self) -> Mapping[str, str]:
        """Return configured target-to-instrument alias mapping."""
        return self._runtime_context.alias_map

    @property
    def sampling_period(self) -> float:
        """Return backend sampling period in ns."""
        return self._runtime_context.sampling_period

    def execute(
        self,
        *,
        request: BackendExecutionRequest,
    ) -> BackendExecutionResult:
        """Execute a backend request using QuEL-3 execution defaults."""
        return self._execution_manager.execute(request=request)

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
    ) -> BackendExecutionResult:
        """Execute a backend request asynchronously using QuEL-3 defaults."""
        return await self._execution_manager.execute_async(request=request)
