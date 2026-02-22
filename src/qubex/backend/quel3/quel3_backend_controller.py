"""QuEL-3 backend controller implemented through quelware-client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
    BackendExecutor,
)
from qubex.backend.controller_types import BackendController

from .managers.clock_manager import Quel3ClockManager
from .managers.configuration_manager import Quel3ConfigurationManager
from .managers.connection_manager import Quel3ConnectionManager
from .managers.execution_manager import ExecutionMode, Quel3ExecutionManager
from .quel3_runtime_context import Quel3RuntimeContext

QUEL3_DEFAULT_SAMPLING_PERIOD_NS = 0.4

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        QubeCalibProtocol as QubeCalib,
        Quel1BoxCommonProtocol as Quel1Box,
    )


class _FigureLike(Protocol):
    """Minimal plotting object protocol used by skew-measurement API."""

    def update_layout(self, *, title: str, width: int) -> None:
        """Update plot layout metadata."""
        ...

    def show(self) -> None:
        """Render plot output."""
        ...


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
        trigger_wait: int | None = None,
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
        trigger_wait : int | None, optional
            Trigger wait count passed to quelware session trigger.
            Defaults to 1_000_000.
        """
        del config_path
        if sampling_period_ns is not None:
            self.DEFAULT_SAMPLING_PERIOD = float(sampling_period_ns)

        self._runtime_context = Quel3RuntimeContext(
            alias_map=dict(alias_map or {}),
            quelware_endpoint=(
                quelware_endpoint if quelware_endpoint is not None else "localhost"
            ),
            quelware_port=int(quelware_port) if quelware_port is not None else 50051,
            trigger_wait=int(trigger_wait) if trigger_wait is not None else 1_000_000,
            default_sampling_period=float(self.DEFAULT_SAMPLING_PERIOD),
            measurement_result_avg_sample_stride=self.MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE,
        )
        self._connection_manager = Quel3ConnectionManager(
            runtime_context=self._runtime_context
        )
        self._clock_manager = Quel3ClockManager(runtime_context=self._runtime_context)
        self._execution_manager = Quel3ExecutionManager(
            runtime_context=self._runtime_context
        )
        self._configuration_manager = Quel3ConfigurationManager()

    @property
    def hash(self) -> int:
        """Return stable hash from runtime and backend configuration state."""
        return hash((self._connection_manager.hash, self._configuration_manager.hash))

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

    @staticmethod
    def load_skew_yaml(file_path: str | Path) -> None:
        """Accept skew-file load API as a no-op for QuEL-3."""
        del file_path

    @staticmethod
    def sync_clocks(box_list: list[str]) -> bool:
        """Keep clock-sync API as a no-op for QuEL-3."""
        del box_list
        return True

    @staticmethod
    def resync_clocks(box_list: list[str]) -> bool:
        """Keep clock-resync API as a no-op for QuEL-3."""
        del box_list
        return True

    @staticmethod
    def relinkup_boxes(
        box_list: list[str],
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """Keep relinkup API as a no-op for QuEL-3."""
        del box_list, noise_threshold, parallel

    @staticmethod
    def reset_clockmaster(ipaddr: str) -> bool:
        """Keep clockmaster-reset API as a no-op for QuEL-3."""
        del ipaddr
        return True

    def set_instrument_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Replace full target-to-alias mapping for quelware execution."""
        self._connection_manager.set_alias_map(alias_map)

    def update_instrument_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Update target-to-alias mapping for quelware execution."""
        self._connection_manager.update_alias_map(alias_map)

    def resolve_instrument_alias(self, target: str) -> str:
        """Resolve quelware instrument alias for a measurement target."""
        return self._execution_manager.resolve_instrument_alias(target)

    def execute(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request using QuEL-3 execution defaults."""
        from .quel3_backend_executor import Quel3BackendExecutor

        factory = getattr(self, "create_measurement_backend_executor", None)
        manager_factory = None
        if callable(factory):
            manager_factory = lambda mode, clock: cast(
                BackendExecutor,
                factory(
                    execution_mode=mode,
                    clock_health_checks=clock,
                ),
            )

        return self._execution_manager.execute(
            request=request,
            execution_mode=execution_mode,
            clock_health_checks=clock_health_checks,
            create_default_executor=lambda mode, clock: Quel3BackendExecutor(
                backend_controller=self,
            ),
            create_measurement_backend_executor=manager_factory,
        )

    def execute_measurement(self, *, payload: object) -> object:
        """Execute one QuEL-3 measurement payload through quelware-client."""
        return self._execution_manager.execute_measurement(payload=payload)

    @classmethod
    def _build_measurement_result(
        cls,
        *,
        payload: object,
        shot_samples: dict[str, dict[str, list[np.ndarray]]],
        sampling_period_ns: float | None,
    ) -> object:
        """Build canonical measurement result from per-shot capture samples."""
        return Quel3ExecutionManager.build_measurement_result(
            payload=payload,
            shot_samples=shot_samples,
            sampling_period_ns=sampling_period_ns,
            default_sampling_period=cls.DEFAULT_SAMPLING_PERIOD,
            avg_sample_stride=cls.MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE,
        )

    @staticmethod
    def _load_quelware_api() -> tuple[object, object, object, object, object]:
        """Import quelware helpers lazily and return required symbols."""
        return cast(
            tuple[object, object, object, object, object],
            Quel3ExecutionManager.load_quelware_api(),
        )

    @staticmethod
    def _append_local_quelware_paths() -> None:
        """Append local quelware source paths when present in the workspace."""
        Quel3ExecutionManager.append_local_quelware_paths()

    @staticmethod
    def get_qubecalib() -> QubeCalib:
        """Reject QuEL-1 specific qubecalib accessor for QuEL-3 backend."""
        raise NotImplementedError("QuEL-3 backend does not provide qubecalib access.")

    @staticmethod
    def get_box(box_name: str) -> Quel1Box:
        """Reject QuEL-1 specific direct box accessor for QuEL-3 backend."""
        del box_name
        raise NotImplementedError("QuEL-3 backend does not expose Quel1Box handles.")

    @staticmethod
    def run_skew_measurement(
        *,
        skew_yaml_path: Path,
        box_yaml_path: Path,
        clockmaster_ip: str,
        box_names: Sequence[str],
        estimate: bool,
    ) -> tuple[dict[str, object], _FigureLike]:
        """Reject QuEL-1 specific skew-measurement API for QuEL-3 backend."""
        del skew_yaml_path, box_yaml_path, clockmaster_ip, box_names, estimate
        raise NotImplementedError("QuEL-3 backend does not support skew measurement.")

    @staticmethod
    def dump_box(box_name: str) -> dict:
        """Reject QuEL-1 specific box-dump API for QuEL-3 backend."""
        del box_name
        raise NotImplementedError("QuEL-3 backend does not expose box dump APIs.")

    @staticmethod
    def dump_port(box_name: str, port_number: int | tuple[int, int]) -> dict:
        """Reject QuEL-1 specific port-dump API for QuEL-3 backend."""
        del box_name, port_number
        raise NotImplementedError("QuEL-3 backend does not expose port dump APIs.")

    def clear_command_queue(self) -> None:
        """Clear pending backend command queue."""
        self._configuration_manager.clear_command_queue()

    def clear_cache(self) -> None:
        """Clear transient backend-side configuration cache."""
        self._configuration_manager.clear_cache()

    def set_box_options(self, box_options: dict[str, tuple[str, ...]]) -> None:
        """Set optional per-box options."""
        self._configuration_manager.set_box_options(box_options)

    def define_clockmaster(self, *, ipaddr: str, reset: bool = True) -> None:
        """Define backend clockmaster metadata."""
        self._configuration_manager.define_clockmaster(ipaddr=ipaddr, reset=reset)

    def define_box(
        self,
        *,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
    ) -> None:
        """Define one backend box."""
        self._configuration_manager.define_box(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=boxtype,
        )

    def define_port(
        self,
        *,
        port_name: str,
        box_name: str,
        port_number: int | tuple[int, int],
    ) -> None:
        """Define one backend port."""
        self._configuration_manager.define_port(
            port_name=port_name,
            box_name=box_name,
            port_number=port_number,
        )

    def define_channel(
        self,
        *,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> None:
        """Define one backend channel."""
        self._configuration_manager.define_channel(
            channel_name=channel_name,
            port_name=port_name,
            channel_number=channel_number,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def add_channel_target_relation(self, channel_name: str, target_name: str) -> None:
        """Define one channel-to-target relation."""
        self._configuration_manager.add_channel_target_relation(
            channel_name,
            target_name,
        )

    def define_target(
        self,
        *,
        target_name: str,
        channel_name: str,
        target_frequency: float,
    ) -> None:
        """Define one backend target."""
        self._configuration_manager.define_target(
            target_name=target_name,
            channel_name=channel_name,
            target_frequency=target_frequency,
        )

    def modify_target_frequencies(self, frequencies: dict[str, float]) -> None:
        """Update target-frequency definitions."""
        self._configuration_manager.modify_target_frequencies(frequencies)

    def config_port(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        lo_freq: float | None = None,
        cnco_freq: float | None = None,
        vatt: int | None = None,
        sideband: str | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
    ) -> None:
        """Store one port-configuration update."""
        self._configuration_manager.config_port(
            box_name,
            port=port,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )

    def config_channel(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        channel: int,
        fnco_freq: float | None = None,
    ) -> None:
        """Store one channel-configuration update."""
        self._configuration_manager.config_channel(
            box_name,
            port=port,
            channel=channel,
            fnco_freq=fnco_freq,
        )

    def config_runit(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        runit: int,
        fnco_freq: float | None = None,
    ) -> None:
        """Store one runit-configuration update."""
        self._configuration_manager.config_runit(
            box_name,
            port=port,
            runit=runit,
            fnco_freq=fnco_freq,
        )

    def initialize_awg_and_capunits(
        self,
        box_names: str | Sequence[str],
        *,
        parallel: bool | None = None,
    ) -> None:
        """Keep QuEL-1 compatibility API as a no-op for QuEL-3."""
        self._configuration_manager.initialize_awg_and_capunits(
            box_names,
            parallel=parallel,
        )
