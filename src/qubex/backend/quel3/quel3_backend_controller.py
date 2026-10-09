"""
QuEL-3 backend controller implementing the shared measurement-facing contract.

This module defines the QuEL-3 concrete `BackendController` implementation
built on quelware-client managers and tools.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt

from qubex.backend.backend_controller import (
    BackendController,
    BackendExecutionRequest,
    BackendExecutionResult,
)
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol

from .formatters import Quel3ResourceView, print_resource_snapshot
from .infra import Quel3ClientMode, Quel3ResourceReader, Quel3RuntimeConfig
from .managers import (
    Quel3ConfigurationManager,
    Quel3ConnectionManager,
    Quel3ExecutionManager,
    Quel3SessionManager,
)
from .models import (
    InstrumentConfiguration,
    InstrumentSpec,
    Quel3ExecutionOptions,
    Quel3ExecutionPlan,
    Quel3ResourceLevel,
    Quel3ResourceSnapshot,
)
from .quel3_backend_constants import CAPTURE_DECIMATION_FACTOR, SAMPLING_PERIOD_NS
from .tools import Quel3MonitorTool

if TYPE_CHECKING:
    from qxpulse import PulseSchedule


class Quel3BackendController(BackendController):
    """
    QuEL-3 backend controller for session lifecycle and execution dispatch.

    The controller provides the required shared `BackendController` API for the
    measurement layer and delegates operations to QuEL-3 managers and tools.
    Backend-specific capabilities are intentionally kept outside the shared
    contract.
    """

    SAMPLING_PERIOD_NS: float = SAMPLING_PERIOD_NS
    CAPTURE_DECIMATION_FACTOR: int = CAPTURE_DECIMATION_FACTOR

    @classmethod
    def from_config_mapping(
        cls,
        value: Mapping[str, object] | None,
    ) -> Quel3BackendController:
        """Create a controller from one QuEL-3 system configuration."""
        if value is None:
            return cls()
        return cls(
            runtime_config=Quel3RuntimeConfig.from_mapping(value),
            execution_options=(
                Quel3ExecutionOptions.model_validate(value["execution"])
                if "execution" in value
                else None
            ),
            cable_delay_ns=cast(
                Mapping[str, Mapping[str, float]] | None,
                value.get("cable_delay_ns"),
            ),
        )

    def __init__(
        self,
        *,
        quelware_endpoint: str | None = None,
        quelware_port: int | None = None,
        client_mode: str | None = None,
        quelware_pat_path: str | None = None,
        runtime_config: Quel3RuntimeConfig | None = None,
        cable_delay_ns: Mapping[str, Mapping[str, float]] | None = None,
        connection_manager: Quel3ConnectionManager | None = None,
        session_manager: Quel3SessionManager | None = None,
        configuration_manager: Quel3ConfigurationManager | None = None,
        execution_manager: Quel3ExecutionManager | None = None,
        resource_reader: Quel3ResourceReader | None = None,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> None:
        """
        Initialize a QuEL-3 backend controller.

        Parameters
        ----------
        quelware_endpoint : str | None, optional
            quelware API endpoint. Defaults to "localhost".
        quelware_port : int | None, optional
            quelware API port. Defaults to 50051.
        client_mode : str | None, optional
            quelware client runtime mode. Defaults to "server".
        quelware_pat_path : str | None, optional
            Path to a quelware personal access token file.
        runtime_config : Quel3RuntimeConfig | None, optional
            Prebuilt runtime settings. Cannot be combined with quelware options.
        cable_delay_ns : Mapping[str, Mapping[str, float]] | None, optional
            Cable delays in ns keyed by unit, then local quelware port ID. All
            configured ports determine the reference, including unused ports.
            Unspecified ports have zero cable delay. None preserves legacy
            execution without timing directives. Empty settings explicitly
            apply zero offsets. Capture delay and payload timing are unchanged.
        connection_manager : Quel3ConnectionManager | None, optional
            Injected connection manager for testing or customization.
        session_manager : Quel3SessionManager | None, optional
            Injected session manager for testing or customization.
        configuration_manager : Quel3ConfigurationManager | None, optional
            Injected configuration manager for testing or customization.
        execution_manager : Quel3ExecutionManager | None, optional
            Injected execution manager for testing or customization.
        resource_reader : Quel3ResourceReader | None, optional
            Injected resource reader for testing or customization.
        execution_options : Quel3ExecutionOptions | None, optional
            Default resource limits and adjacent-job packing settings.
        """
        if runtime_config is not None and any(
            value is not None
            for value in (
                quelware_endpoint,
                quelware_port,
                client_mode,
                quelware_pat_path,
            )
        ):
            raise ValueError(
                "runtime_config cannot be combined with quelware runtime options"
            )
        resolved_runtime_config = runtime_config or Quel3RuntimeConfig(
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
        self._runtime_config = resolved_runtime_config
        self._instrument_cache = InstrumentCache()

        self._connection_manager = (
            connection_manager
            if connection_manager is not None
            else Quel3ConnectionManager(
                runtime_config=resolved_runtime_config,
            )
        )
        self._session_manager = (
            session_manager
            if session_manager is not None
            else Quel3SessionManager(
                runtime_config=resolved_runtime_config,
            )
        )
        self._configuration_manager = (
            configuration_manager
            if configuration_manager is not None
            else Quel3ConfigurationManager(
                runtime_config=resolved_runtime_config,
            )
        )
        self._execution_manager = (
            execution_manager
            if execution_manager is not None
            else Quel3ExecutionManager(
                runtime_config=resolved_runtime_config,
                sampling_period_ns=self._sampling_period_ns,
                capture_decimation_factor=self.CAPTURE_DECIMATION_FACTOR,
                session_manager=self._session_manager,
                execution_options=execution_options,
            )
        )
        if execution_manager is not None and execution_options is not None:
            self.execution_options = execution_options
        if cable_delay_ns is not None:
            self.cable_delay_ns = cable_delay_ns
        self._resource_reader = (
            resource_reader
            if resource_reader is not None
            else Quel3ResourceReader(
                runtime_config=resolved_runtime_config,
            )
        )

        self._monitor_tool = Quel3MonitorTool(
            configuration_manager=self._configuration_manager,
            execution_manager=self._execution_manager,
            resource_reader=self._resource_reader,
            instrument_cache=self._instrument_cache,
        )

    @property
    def cable_delay_ns(self) -> dict[str, dict[str, float]]:
        """Return copied cable delays in ns keyed by unit, then local port ID."""
        return self._execution_manager.cable_delay_ns

    @cable_delay_ns.setter
    def cable_delay_ns(self, delays: Mapping[str, Mapping[str, float]]) -> None:
        """
        Replace default per-port cable delays for subsequent execution batches.

        A payload's `cable_delay_ns` overrides these defaults without merging.
        Effective delays are validated before execution.

        All configured ports determine the maximum delay, including unused
        ports. Each instrument receives the maximum minus its port's delay;
        missing ports have zero cable delay. A transceiver's offset applies
        to both transmission and capture, preserving capture delay. Offsets
        must match the instrument sampling grid. An empty dictionary resets
        offsets to zero. Input and returned dictionaries are copied; assign
        the complete dictionary to update settings. No hardware IO occurs.

        Examples
        --------
        >>> controller.cable_delay_ns = {
        ...     "unit-a": {"tx_p04": 20.0, "trx_p00p01": 80.0}
        ... }
        """
        self._execution_manager.cable_delay_ns = delays

    @property
    def hash(self) -> int:
        """Return stable hash from runtime state."""
        return hash(
            (
                self._connection_manager.hash,
                self._instrument_cache.hash,
                tuple(
                    sorted(
                        (unit, tuple(sorted(ports.items())))
                        for unit, ports in self.cable_delay_ns.items()
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
    def quelware_port(self) -> int | None:
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
    def resource_reader(self) -> Quel3ResourceReader:
        """Return backend-side QuEL-3 resource reader."""
        return self._resource_reader

    def connect(
        self,
        box_names: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Connect to quelware and load existing instruments into the execution cache.

        Notes
        -----
        Each call validates the selected unit labels and replaces the cache with
        instruments from those units. `box_names` contains QuEL-3 unit labels;
        a string selects one unit, `None` selects all units, and an empty list
        probes the endpoint without loading instruments. Duplicate normalized
        aliases emit a warning and keep the last instrument in readback order,
        without changing hardware. Connection, unit-label
        validation, or readback failure leaves the controller disconnected with
        an empty cache and propagates the error.
        """
        self._instrument_cache.clear()
        unit_labels = [box_names] if isinstance(box_names, str) else box_names
        try:
            self._connection_manager.connect(
                unit_labels=unit_labels,
                parallel=parallel,
            )
            if unit_labels is not None and not unit_labels:
                return
            instrument_infos = self._resource_reader.read_instrument_infos(
                unit_labels=() if unit_labels is None else unit_labels,
                parallel=True if parallel is None else parallel,
            )
            self._instrument_cache.replace_all(
                instrument_infos=instrument_infos,
                allow_duplicate_aliases=True,
            )
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect backend resources."""
        self._connection_manager.disconnect()
        self._instrument_cache.clear()

    def clear_instruments(self, *, unit_label: str, parallel: bool = True) -> None:
        """
        Delete every instrument on the specified unit and invalidate its cache.

        Parameters
        ----------
        unit_label : str
            Exact label of the unit whose instruments should be deleted.
        parallel : bool, default=True
            Whether to delete instruments on different ports concurrently.

        Raises
        ------
        ValueError
            The unit label is empty or was not discovered.

        Notes
        -----
        Other units and their cached instruments are retained. The selected
        unit is uncached before writing and stays uncached on failure. Partial
        deletions are not rolled back. This operation is independent of connect.
        """
        self._configuration_manager.clear_instruments(
            unit_label=unit_label,
            instrument_cache=self._instrument_cache,
            parallel=parallel,
        )

    def configure_monitor_mode(
        self,
        *,
        unit_label: str,
        mode: Literal["open", "loopback"] = "open",
        clear_instruments: bool = False,
    ) -> str:
        """
        Set a unit's monitor mode before deploying output and monitor instruments.

        Parameters
        ----------
        unit_label : str
            Exact QuEL-3 unit label.
        mode : {"open", "loopback"}, default="open"
            Normal external output or monitor loopback mode.
        clear_instruments : bool, default=False
            Delete all instruments on this unit before configuring the mode.

        Returns
        -------
        str
            Applied mode reported by quelware.

        Raises
        ------
        RuntimeError
            If this unit has cached instruments and `clear_instruments` is false.

        Notes
        -----
        Quelware also rejects a mode change if uncached instruments remain
        deployed on the unit. With `clear_instruments=True`, every instrument
        on this unit is deleted and its cache is invalidated. Instruments are
        not automatically restored. Re-deploy them after changing the mode.
        """
        return self._configuration_manager.configure_monitor_mode(
            unit_label=unit_label,
            instrument_cache=self._instrument_cache,
            mode=mode,
            clear_instruments=clear_instruments,
        )

    def deploy_instrument(
        self,
        *,
        instrument: InstrumentSpec,
        append: bool = True,
        parallel: bool = True,
    ) -> InstrumentInfoProtocol:
        """
        Deploy one instrument and return its complete hardware information.

        Parameters
        ----------
        instrument : InstrumentSpec
            Instrument definition with a unit-qualified port ID.
        append : bool, default=True
            Add or replace this alias while preserving the port's other
            instruments. If false, replace every instrument on the specified
            port with this one.
        parallel : bool, default=True
            Whether hardware reads may run concurrently.

        Raises
        ------
        ValueError
            Hardware readback is invalid.

        Notes
        -----
        Alias replacement is handled by quelware without a pre-deployment read.
        Append does not retry the deployment request. Both modes refresh the
        entire touched port. Write or readback failure leaves that port absent
        from the execution cache.
        """
        return self._configuration_manager.deploy_instrument(
            instrument=instrument,
            instrument_cache=self._instrument_cache,
            resource_reader=self._resource_reader,
            append=append,
            parallel=parallel,
        )

    def modify_target_frequencies(
        self, target_frequencies_ghz: dict[str, float]
    ) -> None:
        """Redeploy target instruments at the specified frequencies in GHz."""
        if not target_frequencies_ghz:
            return
        instruments = {
            instrument.alias: instrument
            for instrument in self.get_instrument_configuration().instruments
        }
        for alias, frequency in target_frequencies_ghz.items():
            instrument = instruments[alias]
            margin = (
                instrument.frequency_range_max_hz - instrument.frequency_range_min_hz
            ) / 2
            self.deploy_instrument(
                instrument=InstrumentSpec(
                    port_id=instrument.port_id,
                    alias=alias,
                    role=instrument.role,
                    frequency_range_min_hz=frequency * 1e9 - margin,
                    frequency_range_max_hz=frequency * 1e9 + margin,
                ),
                append=True,
            )

    def deploy_instruments(
        self,
        *,
        configuration: InstrumentConfiguration,
        parallel: bool = True,
    ) -> dict[str, InstrumentInfoProtocol]:
        """
        Deploy instruments and read their complete information back from hardware.

        Only the requested ports are replaced. Their old cached identities are
        discarded before writing and remain absent if deployment or readback fails.
        An empty configuration leaves hardware and cache unchanged.
        """
        return self._configuration_manager.deploy_instruments(
            configuration=configuration,
            instrument_cache=self._instrument_cache,
            resource_reader=self._resource_reader,
            parallel=parallel,
        )

    def get_instrument_configuration(self) -> InstrumentConfiguration:
        """
        Return deployable settings from the last successful connect, deploy, or refresh.

        This method reads the instrument cache without contacting hardware.
        """
        return self._configuration_manager.get_instrument_configuration(
            instrument_cache=self._instrument_cache,
        )

    def save_instrument_configuration(self, path: str | Path) -> Path:
        """
        Save the last confirmed instrument configuration as YAML.

        The output file is overwritten. Call `refresh_instrument_cache()` first
        to save the current hardware configuration.
        """
        return self._configuration_manager.save_instrument_configuration(
            path,
            instrument_cache=self._instrument_cache,
        )

    def load_instrument_configuration(
        self, path: str | Path
    ) -> InstrumentConfiguration:
        """
        Load and validate an instrument configuration from YAML.

        Loading does not deploy instruments or change the execution cache.
        Pass the returned configuration to `deploy_instruments()` to apply it.
        """
        return self._configuration_manager.load_instrument_configuration(path)

    def refresh_instrument_cache(
        self,
        *,
        unit_labels: Sequence[str] | None = None,
        parallel: bool = True,
    ) -> dict[str, InstrumentInfoProtocol]:
        """
        Refresh instrument information from hardware for all or selected units.

        `None` selects all units; an empty sequence performs no reads or updates.
        The selected scope is replaced only after successful acquisition and
        validation. Return only the instruments fetched by this call.
        """
        return self._configuration_manager.refresh_instrument_cache(
            instrument_cache=self._instrument_cache,
            resource_reader=self._resource_reader,
            unit_labels=unit_labels,
            parallel=parallel,
        )

    def get_resource_snapshot(
        self,
        *,
        unit_labels: Sequence[str] = (),
        port_ids: Sequence[str] = (),
        instrument_aliases: Sequence[str] = (),
        level: Quel3ResourceLevel = "instrument",
        parallel: bool = True,
        timeout_seconds: float | None = None,
    ) -> Quel3ResourceSnapshot:
        """
        Collect one structured QuEL-3 resource snapshot.

        Read the current quelware resources without changing the execution cache. The
        snapshot can contain partial results and acquisition issues. Instrument
        configuration is obtained separately from the last confirmed cache with
        `get_instrument_configuration()`.

        Parameters
        ----------
        unit_labels : Sequence[str], optional
            Unit labels to inspect. Empty means all discovered units.
        port_ids : Sequence[str], optional
            Full port IDs or local port IDs used to filter ports and
            instruments. Requires `port` level or higher.
        instrument_aliases : Sequence[str], optional
            Unit-qualified aliases or local aliases used to filter instruments
            and their related ports. Requires `instrument` level or higher.
        level : Quel3ResourceLevel, optional
            Cumulative acquisition depth: `unit` reads unit controls, `port`
            adds ports, `instrument` adds instruments, and `diagnosis` adds
            expensive port diagnostic dumps. Defaults to `instrument`.
        parallel : bool, optional
            Whether resource reads should run concurrently.
        timeout_seconds : float | None, optional
            Timeout for the synchronous resource snapshot collection call.

        Returns
        -------
        Quel3ResourceSnapshot
            Observed resources, including partial results and acquisition issues.

        Raises
        ------
        ValueError
            If the level is unsupported or filters require a higher level.
        """
        return self._resource_reader.collect_snapshot(
            unit_labels=tuple(unit_labels),
            port_ids=tuple(port_ids),
            instrument_aliases=tuple(instrument_aliases),
            level=level,
            parallel=parallel,
            timeout_seconds=timeout_seconds,
        )

    def print_resource_snapshot(
        self,
        *,
        view: Quel3ResourceView = "summary",
        unit_labels: Sequence[str] = (),
        port_ids: Sequence[str] = (),
        instrument_aliases: Sequence[str] = (),
        parallel: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        """
        Print one QuEL-3 resource snapshot view with Rich.

        Parameters
        ----------
        view : Quel3ResourceView, optional
            Rendered view name. `units`, `ports`, and `instruments` select the
            corresponding cumulative acquisition level. `summary` uses
            `instrument`; `diagnostics` and `all` use `diagnosis`.
        unit_labels : Sequence[str], optional
            Unit labels to inspect. Empty means all discovered units.
        port_ids : Sequence[str], optional
            Full port IDs or local port IDs used to filter ports and
            instruments. Requires `port` level or higher.
        instrument_aliases : Sequence[str], optional
            Unit-qualified aliases or local aliases used to filter instruments
            and their related ports. Requires `instrument` level or higher.
        parallel : bool, optional
            Whether resource reads should run concurrently.
        timeout_seconds : float | None, optional
            Timeout for the synchronous resource snapshot collection call.

        Raises
        ------
        ValueError
            If the view is unsupported or filters require a higher level.
            Validation runs before resource reads.
        """
        level_by_view: dict[Quel3ResourceView, Quel3ResourceLevel] = {
            "summary": "instrument",
            "units": "unit",
            "ports": "port",
            "instruments": "instrument",
            "diagnostics": "diagnosis",
            "all": "diagnosis",
        }
        if view not in level_by_view:
            raise ValueError(f"Unsupported QuEL-3 resource snapshot view: {view!r}")
        snapshot = self.get_resource_snapshot(
            unit_labels=unit_labels,
            port_ids=port_ids,
            instrument_aliases=instrument_aliases,
            level=level_by_view[view],
            parallel=parallel,
            timeout_seconds=timeout_seconds,
        )
        print_resource_snapshot(snapshot, view=view)

    @property
    def sampling_period_ns(self) -> float:
        """Return backend sampling period in ns."""
        return self._sampling_period_ns

    def run_monitor_schedule(
        self,
        *,
        pulse_schedule: PulseSchedule,
        capture_start_ns: float = 0.0,
        capture_length_ns: float | None = None,
        n_iterations: int = 1,
        shot_interval_ns: float = 0.0,
        parallel: bool = True,
    ) -> dict[str, npt.NDArray[np.complex128]]:
        """
        Run each schedule target on the monitor port and restore the unit state.

        Parameters
        ----------
        pulse_schedule : PulseSchedule
            Valid schedule whose target names uniquely identify live output
            instrument aliases on a single unit.
        capture_start_ns : float, default=0.0
            Capture start time relative to the output trigger, in ns.
        capture_length_ns : float | None, optional
            Capture duration in ns. Defaults to the schedule duration.
        n_iterations : int, default=1
            Number of unaveraged captures.
        shot_interval_ns : float, default=0.0
            Idle interval between iterations, in ns.
        parallel : bool, default=True
            Whether to parallelize instrument execution phases.

        Returns
        -------
        dict[str, NDArray[np.complex128]]
            Raw IQ arrays keyed by schedule target, each with shape
            `(n_iterations, samples)`.

        Raises
        ------
        ValueError
            If a target has no live instrument, its alias is ambiguous, targets
            span multiple units, or schedule or capture settings are invalid.

        Notes
        -----
        The unit is inferred from live instruments matching the schedule's
        targets. All targets are validated before clearing any instruments.
        Schedule frequencies are in GHz. A target without a frequency uses the
        center of its live instrument's frequency range. The same frequency is
        applied to the output and monitor instruments. The method reads the
        live instruments, clears the selected unit, enters loopback mode, and
        executes targets sequentially. It clears temporary instruments and
        restores the original monitor mode and instrument configuration even
        if deployment or execution fails. Blanks advance event offsets without
        allocating zero-filled waveform samples.
        """
        return self._monitor_tool.run_schedule(
            pulse_schedule=pulse_schedule,
            capture_start_ns=capture_start_ns,
            capture_length_ns=capture_length_ns,
            n_iterations=n_iterations,
            shot_interval_ns=shot_interval_ns,
            parallel=parallel,
        )

    @property
    def execution_options(self) -> Quel3ExecutionOptions:
        """Return execution defaults owned by the execution manager."""
        return self._execution_manager.execution_options

    @execution_options.setter
    def execution_options(self, options: Quel3ExecutionOptions) -> None:
        """Replace execution defaults in the execution manager."""
        self._execution_manager.execution_options = options

    def plan_execution(
        self,
        requests: Sequence[BackendExecutionRequest],
        *,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> Quel3ExecutionPlan:
        """
        Inspect execution resources and shot ranges without hardware IO.

        Use cached instrument information from connect, deployment, or refresh.
        Input indices and half-open shot ranges in the returned plan are
        zero-based. Passing options replaces the complete controller defaults.
        Jobs are packed and executed in input order.
        """
        return self._execution_manager.plan_execution(
            requests=requests,
            instrument_cache=self._instrument_cache,
            execution_options=execution_options,
        )

    def execute_sync(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: str | None = None,
        clock_health_checks: bool | None = None,
        parallel: bool = True,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request synchronously using QuEL-3 defaults."""
        del execution_mode, clock_health_checks
        return self._execution_manager.execute_sync(
            request=request,
            instrument_cache=self._instrument_cache,
            parallel=parallel,
            execution_options=execution_options,
        )

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: str | None = None,
        clock_health_checks: bool | None = None,
        parallel: bool = True,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request asynchronously using QuEL-3 defaults."""
        del execution_mode, clock_health_checks
        return await self._execution_manager.execute_async(
            request=request,
            instrument_cache=self._instrument_cache,
            parallel=parallel,
            execution_options=execution_options,
        )

    async def execute_batch_async(
        self,
        *,
        requests: Sequence[BackendExecutionRequest],
        execution_mode: str | None = None,
        clock_health_checks: bool | None = None,
        parallel: bool = True,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> list[BackendExecutionResult]:
        """Execute multiple backend requests as one resolved QuEL-3 batch."""
        del execution_mode, clock_health_checks
        return await self._execution_manager.execute_batch_async(
            requests=tuple(requests),
            instrument_cache=self._instrument_cache,
            parallel=parallel,
            execution_options=execution_options,
        )
