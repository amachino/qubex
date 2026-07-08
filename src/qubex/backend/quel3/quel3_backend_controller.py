"""
QuEL-3 backend controller implementing the shared measurement-facing contract.

This module defines the QuEL-3 concrete `BackendController` implementation
built on quelware-client managers.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console

from qubex.backend.backend_controller import (
    BackendController,
    BackendExecutionRequest,
    BackendExecutionResult,
)
from qubex.backend.quel3.formatting import format_quel3_hardware_state
from qubex.backend.quel3.infra import Quel3ClientMode
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol

from .managers import (
    Quel3ConfigurationManager,
    Quel3ConnectionManager,
    Quel3ExecutionManager,
    Quel3HardwareStateReader,
    Quel3RuntimeConfig,
    Quel3SessionManager,
)
from .models import InstrumentDeployRequest, Quel3HardwareState, Quel3HardwareStateView
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
        hardware_state_reader: Quel3HardwareStateReader | None = None,
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
        hardware_state_reader : Quel3HardwareStateReader | None, optional
            Injected hardware-state reader for testing or customization.
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
        self._hardware_state_reader = (
            hardware_state_reader
            if hardware_state_reader is not None
            else Quel3HardwareStateReader(
                runtime_config=runtime_config,
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
    def hardware_state_reader(self) -> Quel3HardwareStateReader:
        """Return backend-side QuEL-3 hardware-state reader."""
        return self._hardware_state_reader

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

    def get_hardware_state(
        self,
        *,
        unit_labels: Sequence[str] = (),
        port_ids: Sequence[str] = (),
        instrument_aliases: Sequence[str] = (),
        include_diagnostics: bool = False,
        parallel: bool = True,
        timeout_seconds: float | None = None,
    ) -> Quel3HardwareState:
        """
        Collect one structured QuEL-3 hardware-state snapshot.

        Parameters
        ----------
        unit_labels : Sequence[str], optional
            Unit labels to inspect. Empty means all discovered units.
        port_ids : Sequence[str], optional
            Full port IDs or local port IDs used to filter ports and
            instruments.
        instrument_aliases : Sequence[str], optional
            Unit-qualified aliases or local aliases used to filter instruments
            and their related ports.
        include_diagnostics : bool, optional
            Whether to collect diagnostic dumps for the final visible ports.
        parallel : bool, optional
            Whether resource reads should run concurrently.
        timeout_seconds : float | None, optional
            Timeout for the synchronous hardware-state collection call.
        """
        return self._hardware_state_reader.collect_state(
            unit_labels=tuple(unit_labels),
            port_ids=tuple(port_ids),
            instrument_aliases=tuple(instrument_aliases),
            include_diagnostics=include_diagnostics,
            parallel=parallel,
            timeout_seconds=timeout_seconds,
        )

    def print_hardware_state(
        self,
        *,
        view: Quel3HardwareStateView = "summary",
        unit_labels: Sequence[str] = (),
        port_ids: Sequence[str] = (),
        instrument_aliases: Sequence[str] = (),
        include_diagnostics: bool | None = None,
        parallel: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        """
        Print one QuEL-3 hardware-state view with Rich.

        Parameters
        ----------
        view : Quel3HardwareStateView, optional
            Rendered view name.
        unit_labels : Sequence[str], optional
            Unit labels to inspect. Empty means all discovered units.
        port_ids : Sequence[str], optional
            Full port IDs or local port IDs used to filter ports and
            instruments.
        instrument_aliases : Sequence[str], optional
            Unit-qualified aliases or local aliases used to filter instruments
            and their related ports.
        include_diagnostics : bool | None, optional
            Whether to collect diagnostic dumps. `None` includes diagnostics
            for the `diagnostics` and `all` views.
        parallel : bool, optional
            Whether resource reads should run concurrently.
        timeout_seconds : float | None, optional
            Timeout for the synchronous hardware-state collection call.
        """
        if include_diagnostics is None:
            include_diagnostics = view in ("diagnostics", "all")
        state = self.get_hardware_state(
            unit_labels=unit_labels,
            port_ids=port_ids,
            instrument_aliases=instrument_aliases,
            include_diagnostics=include_diagnostics,
            parallel=parallel,
            timeout_seconds=timeout_seconds,
        )
        output_console = Console(highlight=False)
        output_console.print(format_quel3_hardware_state(state, view=view))

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
