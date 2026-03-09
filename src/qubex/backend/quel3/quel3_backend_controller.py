"""
QuEL-3 backend controller implementing the shared measurement-facing contract.

This module defines the QuEL-3 concrete `BackendController` implementation
built on quelware-client managers.
"""

from __future__ import annotations

from collections.abc import Sequence

from qubex.backend.backend_controller import (
    BackendController,
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .managers import (
    Quel3ConfigurationManager,
    Quel3ConnectionManager,
    Quel3ExecutionManager,
)
from .models import InstrumentDeployRequest


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
        configuration_manager: Quel3ConfigurationManager | None = None,
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
        self._quelware_endpoint = endpoint
        self._quelware_port = port

        self._connection_manager = Quel3ConnectionManager(
            quelware_endpoint=endpoint,
            quelware_port=port,
        )
        self._configuration_manager = (
            configuration_manager
            if configuration_manager is not None
            else Quel3ConfigurationManager(
                quelware_endpoint=endpoint,
                quelware_port=port,
            )
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
        return hash(
            (
                self._connection_manager.hash,
                tuple(sorted(self._configuration_manager.target_alias_map.items())),
                tuple(
                    sorted(
                        self._configuration_manager.last_deployed_instrument_infos.keys()
                    )
                ),
            )
        )

    @property
    def is_connected(self) -> bool:
        """Return whether backend resources are connected."""
        return self._connection_manager.is_connected

    @property
    def quelware_endpoint(self) -> str:
        """Return configured quelware endpoint."""
        return self._quelware_endpoint

    @property
    def quelware_port(self) -> int:
        """Return configured quelware port."""
        return self._quelware_port

    @property
    def configuration_manager(self) -> Quel3ConfigurationManager:
        """Return backend-side QuEL-3 configuration manager."""
        return self._configuration_manager

    @property
    def target_alias_map(self) -> dict[str, str]:
        """Return deployed target-to-alias mapping from backend runtime state."""
        return self._configuration_manager.target_alias_map

    @property
    def last_deployed_instrument_infos(self) -> dict[str, tuple[object, ...]]:
        """Return deployed instrument infos from backend runtime state."""
        return self._configuration_manager.last_deployed_instrument_infos

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

    def deploy_instruments(
        self,
        *,
        requests: Sequence[InstrumentDeployRequest],
    ) -> dict[str, tuple[object, ...]]:
        """Deploy QuEL-3 instruments for the provided requests."""
        return self._configuration_manager.deploy_instruments(requests=requests)

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
