"""
QuEL-3 backend controller implementing the shared measurement-facing contract.

This module defines the QuEL-3 concrete `BackendController` implementation
built on quelware-client managers.
"""

from __future__ import annotations

from qubex.backend.backend_controller import (
    BackendController,
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .managers.connection_manager import Quel3ConnectionManager
from .managers.execution_manager import Quel3ExecutionManager


class Quel3BackendController(BackendController):
    """
    QuEL-3 backend controller for session lifecycle and execution dispatch.

    The controller provides the required shared `BackendController` API for the
    measurement layer and routes concrete operations to QuEL-3 manager classes.
    Backend-specific capabilities are intentionally kept outside the shared
    contract.
    """

    SAMPLING_PERIOD_NS: float = 0.4
    CAPTURE_DECIMATION_FACTOR: int = 4

    def __init__(
        self,
        *,
        quelware_endpoint: str | None = None,
        quelware_port: int | None = None,
    ) -> None:
        """
        Initialize a QuEL-3 backend controller.

        Parameters
        ----------
        quelware_endpoint : str | None, optional
            quelware API endpoint. Defaults to "localhost".
        quelware_port : int | None, optional
            quelware API port. Defaults to 50051.
        """
        self._sampling_period = self.SAMPLING_PERIOD_NS
        endpoint = quelware_endpoint if quelware_endpoint is not None else "localhost"
        port = quelware_port if quelware_port is not None else 50051

        self._connection_manager = Quel3ConnectionManager(
            quelware_endpoint=endpoint,
            quelware_port=port,
        )
        self._execution_manager = Quel3ExecutionManager(
            quelware_endpoint=endpoint,
            quelware_port=port,
            sampling_period=self._sampling_period,
            capture_decimation_factor=self.CAPTURE_DECIMATION_FACTOR,
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

    @property
    def sampling_period(self) -> float:
        """Return backend sampling period in ns."""
        return self._sampling_period

    def execute_sync(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: str | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request synchronously using QuEL-3 defaults."""
        del execution_mode, clock_health_checks
        return self._execution_manager.execute_sync(request=request)

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: str | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request asynchronously using QuEL-3 defaults."""
        del execution_mode, clock_health_checks
        return await self._execution_manager.execute_async(request=request)
