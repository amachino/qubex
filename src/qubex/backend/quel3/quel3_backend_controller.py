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
from qubex.backend.quel3.infra import Quel3ClientMode
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol

from .managers import (
    Quel3ConfigurationManager,
    Quel3ConnectionManager,
    Quel3ExecutionManager,
    Quel3RuntimeConfig,
    Quel3SessionManager,
)
from .models import InstrumentDeployRequest
from .quel3_backend_constants import CAPTURE_DECIMATION_FACTOR, SAMPLING_PERIOD_NS


class Quel3BackendController(BackendController):
    """
    QuEL-3 backend controller for session lifecycle and execution dispatch.

    The controller provides the required shared `BackendController` API for the
    measurement layer and routes concrete operations to QuEL-3 manager classes.
    Backend-specific capabilities are intentionally kept outside the shared
    contract.
    """

    SAMPLING_PERIOD_NS: float = SAMPLING_PERIOD_NS
    CAPTURE_DECIMATION_FACTOR: int = CAPTURE_DECIMATION_FACTOR

    def __init__(
        self,
        *,
        quelware_endpoint: str | None = None,
        quelware_port: int | None = None,
        client_mode: str | None = None,
        quelware_pat_path: str | None = None,
        connection_manager: Quel3ConnectionManager | None = None,
        session_manager: Quel3SessionManager | None = None,
        configuration_manager: Quel3ConfigurationManager | None = None,
        execution_manager: Quel3ExecutionManager | None = None,
    ) -> None:
        """
        Initialize a QuEL-3 backend controller.

        Parameters
        ----------
        quelware_endpoint : str | None, optional
            quelware API endpoint. Defaults to "localhost".
        quelware_port : int | None, optional
            quelware API port. Defaults to 50051.
        connection_manager : Quel3ConnectionManager | None, optional
            Injected connection manager for testing or customization.
        session_manager : Quel3SessionManager | None, optional
            Injected session manager for testing or customization.
        configuration_manager : Quel3ConfigurationManager | None, optional
            Injected configuration manager for testing or customization.
        execution_manager : Quel3ExecutionManager | None, optional
            Injected execution manager for testing or customization.
        """
        runtime_config = Quel3RuntimeConfig(
            endpoint=quelware_endpoint or "localhost",
            port=50051 if quelware_port is None else quelware_port,
            client_mode=client_mode or "server",
            pat_path=quelware_pat_path,
        )
        self._sampling_period_ns = (
            execution_manager.sampling_period_ns
            if execution_manager is not None
            else self.SAMPLING_PERIOD_NS
        )
        self._runtime_config = runtime_config

        self._connection_manager = (
            connection_manager
            if connection_manager is not None
            else Quel3ConnectionManager(
                runtime_config=runtime_config,
            )
        )
        self._session_manager = (
            session_manager
            if session_manager is not None
            else Quel3SessionManager(
                runtime_config=runtime_config,
            )
        )
        self._configuration_manager = (
            configuration_manager
            if configuration_manager is not None
            else Quel3ConfigurationManager(
                runtime_config=runtime_config,
            )
        )
        self._execution_manager = (
            execution_manager
            if execution_manager is not None
            else Quel3ExecutionManager(
                runtime_config=runtime_config,
                sampling_period_ns=self._sampling_period_ns,
                capture_decimation_factor=self.CAPTURE_DECIMATION_FACTOR,
                session_manager=self._session_manager,
            )
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
        return self._runtime_config.endpoint

    @property
    def quelware_port(self) -> int:
        """Return configured quelware port."""
        return self._runtime_config.port

    @property
    def client_mode(self) -> Quel3ClientMode:
        """Return configured quelware client mode."""
        return self._runtime_config.client_mode_value

    @property
    def quelware_pat_path(self) -> str | None:
        """Return configured quelware personal access token path."""
        return self._runtime_config.pat_path

    @property
    def runtime_config(self) -> Quel3RuntimeConfig:
        """Return configured quelware runtime settings."""
        return self._runtime_config

    @property
    def configuration_manager(self) -> Quel3ConfigurationManager:
        """Return backend-side QuEL-3 configuration manager."""
        return self._configuration_manager

    @property
    def connection_manager(self) -> Quel3ConnectionManager:
        """Return backend-side QuEL-3 connection manager."""
        return self._connection_manager

    @property
    def session_manager(self) -> Quel3SessionManager:
        """Return backend-side QuEL-3 session manager."""
        return self._session_manager

    @property
    def execution_manager(self) -> Quel3ExecutionManager:
        """Return backend-side QuEL-3 execution manager."""
        return self._execution_manager

    @property
    def target_alias_map(self) -> dict[tuple[str, str], str]:
        """Return deployed box-and-target to runtime-alias mapping."""
        return self._configuration_manager.target_alias_map

    @property
    def last_deployed_instrument_infos(
        self,
    ) -> dict[str, tuple[InstrumentInfoProtocol, ...]]:
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
        self._configuration_manager.refresh_instrument_cache()

    def disconnect(self) -> None:
        """Disconnect backend resources."""
        self._connection_manager.disconnect()

    def deploy_instruments(
        self,
        *,
        requests: Sequence[InstrumentDeployRequest],
        parallel: bool = True,
    ) -> dict[str, tuple[InstrumentInfoProtocol, ...]]:
        """Deploy QuEL-3 instruments for the provided requests."""
        return self._configuration_manager.deploy_instruments(
            requests=requests,
            parallel=parallel,
        )

    @property
    def sampling_period_ns(self) -> float:
        """Return backend sampling period in ns."""
        return self._sampling_period_ns

    def execute_sync(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: str | None = None,
        clock_health_checks: bool | None = None,
        parallel: bool = True,
    ) -> BackendExecutionResult:
        """Execute a backend request synchronously using QuEL-3 defaults."""
        del execution_mode, clock_health_checks
        return self._execution_manager.execute_sync(
            request=request,
            parallel=parallel,
        )

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: str | None = None,
        clock_health_checks: bool | None = None,
        parallel: bool = True,
    ) -> BackendExecutionResult:
        """Execute a backend request asynchronously using QuEL-3 defaults."""
        del execution_mode, clock_health_checks
        return await self._execution_manager.execute_async(
            request=request,
            parallel=parallel,
        )

    async def execute_batch_async(
        self,
        *,
        requests: Sequence[BackendExecutionRequest],
        execution_mode: str | None = None,
        clock_health_checks: bool | None = None,
        parallel: bool = True,
    ) -> list[BackendExecutionResult]:
        """Execute multiple backend requests as one resolved QuEL-3 batch."""
        del execution_mode, clock_health_checks
        return await self._execution_manager.execute_batch_async(
            requests=tuple(requests),
            parallel=parallel,
        )
