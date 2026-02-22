# ruff: noqa: SLF001

"""QuEL-1 backend controller using qube-calib."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)
from qubex.backend.controller_types import BackendController
from qubex.backend.parallel_box_executor import run_parallel_map

from .compat.box_adapter import adapt_quel1_box
from .compat.driver_loader import load_quel1_driver
from .managers import (
    Quel1ClockManager,
    Quel1ConfigurationManager,
    Quel1ConnectionManager,
    Quel1ExecutionManager,
)
from .quel1_backend_constants import (
    DEFAULT_CAPTURE_DELAY,
    DEFAULT_EXECUTION_MODE,
    WORD_LENGTH,
    ExecutionMode,
)
from .quel1_runtime_context import Quel1RuntimeContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.control_system import Box
    from qubex.backend.experiment_system import ExperimentSystem

    from .compat.qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        BoxSettingProtocol as BoxSetting,
        QubeCalibProtocol as QubeCalib,
        Quel1BoxCommonProtocol as Quel1Box,
        Quel1ConfigOptionProtocol as Quel1ConfigOption,
        Quel1SystemProtocol as Quel1System,
        SequencerProtocol as Sequencer,
    )

# TODO: use appropriate noise threshold
_RELAXED_NOISE_THRESHOLD = 10000
_MAX_BOX_PARALLEL_WORKERS = 32
_DEFAULT_PARALLEL_MODE = DEFAULT_EXECUTION_MODE == "parallel"
_QUEL1SE_R8_AWG_OPTIONS = {
    "se8_mxfe1_awg1331",
    "se8_mxfe1_awg2222",
    "se8_mxfe1_awg3113",
}
_QUEL1SE_R8_DEFAULT_AWG_OPTION = "se8_mxfe1_awg2222"


def _resolve_quel1se_r8_awg_option(options: list[str]) -> str:
    awg_options = [label for label in options if label in _QUEL1SE_R8_AWG_OPTIONS]
    if len(awg_options) > 1:
        raise ValueError("Multiple AWG options are not allowed for quel1se-riken8.")
    if len(awg_options) == 1:
        return awg_options[0]
    return _QUEL1SE_R8_DEFAULT_AWG_OPTION


@dataclass
class Quel1BackendRawResult:
    """Raw status, data, and config returned from qube-calib execution."""

    status: dict
    data: dict
    config: dict


class Quel1BackendController(BackendController):
    """Control and query device state through qube-calib."""

    # Measurement capability hints consumed by measurement-layer selector logic.
    MEASUREMENT_BACKEND_KIND: Literal["quel1"] = "quel1"
    MEASUREMENT_CONSTRAINT_MODE: Literal["quel1"] = "quel1"
    MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE: int = WORD_LENGTH

    def __init__(
        self,
        config_path: str | Path | None = None,
    ):
        self._driver = load_quel1_driver()

        try:
            if config_path is None:
                self._qubecalib = self._driver.QubeCalib()
            else:
                try:
                    self._qubecalib = self._driver.QubeCalib(str(config_path))
                except FileNotFoundError:
                    logger.warning(f"Configuration file {config_path} not found.")
                    raise
        except Exception:
            self._qubecalib = None
        self._box_options: dict[str, tuple[str, ...]] = {}
        self._runtime_context = Quel1RuntimeContext()
        self._connection_manager = Quel1ConnectionManager(
            runtime_context=self._runtime_context
        )
        self._clock_manager = Quel1ClockManager(runtime_context=self._runtime_context)
        self._execution_manager = Quel1ExecutionManager(
            runtime_context=self._runtime_context
        )
        self._configuration_manager = Quel1ConfigurationManager(
            runtime_context=self._runtime_context
        )

    @property
    def is_connected(self) -> bool:
        """Return whether the hardware is connected."""
        return self._connection_manager.is_connected

    @property
    def qubecalib(self) -> QubeCalib:
        """Return the QubeCalib instance or raise if unavailable."""
        if self._qubecalib is None:
            raise ModuleNotFoundError(name="qubecalib")
        return self._qubecalib

    def get_qubecalib(self) -> QubeCalib:
        """
        Return the underlying QubeCalib instance.

        Returns
        -------
        QubeCalib
            QubeCalib instance.
        """
        return self.qubecalib

    @property
    def box_config(self) -> dict[str, Any]:
        """Get the box configuration."""
        boxpool = self._connection_manager.boxpool
        if boxpool is None:
            box_config = {}
        else:
            box_config = boxpool._box_config_cache
        return box_config

    @property
    def system_config(self) -> dict[str, Any]:
        """Get the system configuration."""
        config = self.qubecalib.system_config_database.asdict()
        return config

    @property
    def system_config_json(self) -> str:
        """Get the system configuration as JSON."""
        config = self.qubecalib.system_config_database.asjson()
        return config

    @property
    def box_settings(self) -> dict[str, Any]:
        """Get the box settings."""
        return self.system_config["box_settings"]

    @property
    def port_settings(self) -> dict[str, Any]:
        """Get the port settings."""
        return self.system_config["port_settings"]

    @property
    def target_settings(self) -> dict[str, Any]:
        """Get the target settings."""
        return self.system_config["target_settings"]

    @property
    def available_boxes(self) -> list[str]:
        """
        Get the list of available boxes.

        Returns
        -------
        list[str]
            List of available boxes.
        """
        return list(self.box_settings.keys())

    @property
    def boxpool(self) -> BoxPool:
        """
        Get the boxpool.

        Returns
        -------
        BoxPool
            The boxpool.
        """
        boxpool = self._connection_manager.boxpool
        if boxpool is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        return boxpool

    @property
    def quel1system(self) -> Quel1System:
        """
        Get the Quel1 system.

        Returns
        -------
        Quel1System
            The Quel1 system.
        """
        quel1system = self._connection_manager.quel1system
        if quel1system is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        return quel1system

    @property
    def cap_resource_map(self) -> dict[str, dict]:
        """
        Get the cap resource map.

        Returns
        -------
        dict[str, dict]
            The cap resource map.
        """
        cap_resource_map = self._connection_manager.cap_resource_map
        if cap_resource_map is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        return cap_resource_map

    @property
    def gen_resource_map(self) -> dict[str, dict]:
        """
        Get the gen resource map.

        Returns
        -------
        dict[str, dict]
            The gen resource map.
        """
        gen_resource_map = self._connection_manager.gen_resource_map
        if gen_resource_map is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        return gen_resource_map

    @property
    def hash(self) -> int:
        """
        Get the hash of the system configuration.

        Returns
        -------
        int
            Hash of the system configuration.
        """
        return hash(self.qubecalib.system_config_database.asjson())

    def _make_backend_raw_result(
        self,
        *,
        status: dict,
        data: dict,
        config: dict,
    ) -> Quel1BackendRawResult:
        """Build canonical QuEL-1 raw result container."""
        return Quel1BackendRawResult(
            status=status,
            data=data,
            config=config,
        )

    def _check_box_availability(self, box_name: str):
        if box_name not in self.available_boxes:
            raise ValueError(
                f"Box {box_name} not in available boxes: {self.available_boxes}"
            )

    def get_resource_map(self, targets: list[str]) -> dict[str, list[dict]]:
        """Build a resource map for the requested targets."""
        db = self.qubecalib.system_config_database
        target_settings = db._target_settings
        box_settings = db._box_settings
        port_settings = db._port_settings
        result = {}
        for target in targets:
            if target not in target_settings:
                raise ValueError(f"Target {target} not in available targets.")

            channels = db.get_channels_by_target(target)
            bpc_list = [db.get_channel(channel) for channel in channels]
            result[target] = [
                {
                    "box": box_settings[box_name],
                    "port": port_settings[port_name],
                    "channel_number": channel_number,
                    "target": target_settings[target],
                }
                for box_name, port_name, channel_number in bpc_list
            ]
        return result

    def get_cap_resource_map(self, targets: Collection[str]) -> dict[str, dict]:
        """
        Get the resource map for the given targets.

        Parameters
        ----------
        targets : Collection[str]
            List of target names.
        """
        return {
            target: self.cap_resource_map[target]
            for target in targets
            if target in self.cap_resource_map
        }

    def get_gen_resource_map(self, targets: Collection[str]) -> dict[str, dict]:
        """
        Get the resource map for the given targets.

        Parameters
        ----------
        targets : Collection[str]
            List of target names.
        """
        return {
            target: self.gen_resource_map[target]
            for target in targets
            if target in self.gen_resource_map
        }

    def create_resource_map(
        self,
        type: Literal["cap", "gen"],
    ) -> dict[str, dict]:
        """Create a capture or generator resource map from configuration."""
        boxpool = self._connection_manager.boxpool
        if boxpool is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        db = self.qubecalib.system_config_database
        target_settings = db._target_settings
        box_settings = db._box_settings
        port_settings = db._port_settings
        pooled_boxes = boxpool._boxes
        result = {}
        for target in target_settings:
            channels = db.get_channels_by_target(target)
            bpc_list = [db.get_channel(channel) for channel in channels]
            for box_name, port_name, channel_number in bpc_list:
                if box_name not in pooled_boxes:
                    continue
                box = self.get_box(box_name)
                port_setting = port_settings[port_name]
                if (type == "cap" and port_setting.port in box.get_input_ports()) or (
                    type == "gen" and port_setting.port in box.get_output_ports()
                ):
                    result[target] = {
                        "box": box_settings[box_name],
                        "port": port_settings[port_name],
                        "channel_number": channel_number,
                        "target": target_settings[target],
                    }
        return result

    def clear_cache(self) -> None:
        """Clear cached box configuration data."""
        boxpool = self._connection_manager.boxpool
        if boxpool is not None:
            boxpool._box_config_cache.clear()
        self._clear_quel1system_box_cache()

    def get_box_config_cache(self) -> dict[str, Any]:
        """Return a snapshot of the box-config cache."""
        return deepcopy(self.box_config)

    def replace_box_config_cache(self, box_configs: dict[str, Any]) -> None:
        """Replace the box-config cache with the provided snapshot."""
        boxpool = self._connection_manager.boxpool
        if boxpool is None:
            if box_configs:
                raise ValueError("Boxes not connected. Call connect() method first.")
            return
        boxpool._box_config_cache = deepcopy(box_configs)
        self._replace_quel1system_box_cache(box_configs)

    def update_box_config_cache(self, box_configs: dict[str, Any]) -> None:
        """Update cached box configurations by box name."""
        boxpool = self._connection_manager.boxpool
        if boxpool is None:
            if box_configs:
                raise ValueError("Boxes not connected. Call connect() method first.")
            return
        for box_name, box_config in box_configs.items():
            boxpool._box_config_cache[box_name] = deepcopy(box_config)
        self._update_quel1system_box_cache(box_configs)

    def _clear_quel1system_box_cache(self) -> None:
        """Clear the Quel1System-side box cache."""
        system = self._connection_manager.quel1system
        if system is None:
            return
        system.config_cache.clear()
        system.config_fetched_at = None

    def _replace_quel1system_box_cache(self, box_configs: dict[str, Any]) -> None:
        """Replace the Quel1System-side box cache."""
        system = self._connection_manager.quel1system
        if system is None:
            return
        system.config_cache.clear()
        for box_name, box_config in box_configs.items():
            system.config_cache[box_name] = deepcopy(box_config)
        system.config_fetched_at = datetime.now() if system.config_cache else None

    def _update_quel1system_box_cache(self, box_configs: dict[str, Any]) -> None:
        """Update entries in the Quel1System-side box cache."""
        system = self._connection_manager.quel1system
        if system is None:
            return
        for box_name, box_config in box_configs.items():
            system.config_cache[box_name] = deepcopy(box_config)
        if system.config_cache:
            system.config_fetched_at = datetime.now()

    def set_box_options(self, box_options: dict[str, tuple[str, ...]]) -> None:
        """Set box option labels used for relinkup config options."""
        self._box_options = {
            box_name: tuple(option_labels)
            for box_name, option_labels in box_options.items()
        }

    def _resolve_config_options(
        self,
        *,
        box_name: str,
        boxtype: str,
    ) -> list[Quel1ConfigOption] | None:
        """Resolve config options for relinkup from optional per-box labels."""
        option_labels = list(self._box_options.get(box_name, ()))
        if boxtype == "quel1se-riken8":
            awg_option = _resolve_quel1se_r8_awg_option(option_labels)
            if awg_option not in option_labels:
                option_labels.insert(0, awg_option)
        if not option_labels:
            return None
        config_options: list[Quel1ConfigOption] = []
        option_map = self._driver.Quel1ConfigOption._value2member_map_
        for option_label in option_labels:
            option = option_map.get(option_label)
            if option is None:
                raise ValueError(
                    f"Unknown Quel1 config option `{option_label}` for box `{box_name}`."
                )
            config_options.append(cast(Any, option))
        return config_options

    def load_skew_yaml(self, file_path: str | Path) -> None:
        """
        Load skew calibration YAML into the system database.

        Parameters
        ----------
        file_path : str | Path
            Path to the skew calibration YAML file.
        """
        self.qubecalib.sysdb.load_skew_yaml(str(file_path))

    def run_skew_measurement(
        self,
        *,
        skew_yaml_path: str | Path,
        box_yaml_path: str | Path,
        clockmaster_ip: str,
        box_names: list[str],
        estimate: bool = True,
    ) -> tuple[Any, Any]:
        """
        Measure skew from YAML settings and return skew object and figure.

        Parameters
        ----------
        skew_yaml_path : str | Path
            Path to skew YAML.
        box_yaml_path : str | Path
            Path to box YAML.
        clockmaster_ip : str
            Clock master IP address.
        box_names : list[str]
            Boxes to include in the measurement.
        estimate : bool, optional
            Whether to run estimation after measurement.

        Returns
        -------
        tuple[Any, Any]
            A tuple of (skew object, plotly figure).
        """
        skew = self._driver.Skew.from_yaml(
            str(skew_yaml_path),
            box_yaml=str(box_yaml_path),
            clockmaster_ip=clockmaster_ip,
            boxes=box_names,
        )
        skew.system.resync()
        skew.measure()
        if estimate:
            skew.estimate()
        fig = skew.plot()
        return skew, fig

    def link_status(self, box_name: str) -> dict[int, bool]:
        """
        Get the link status of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.

        Returns
        -------
        dict[int, bool]
            Dictionary of link status.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        self._check_box_availability(box_name)
        box = self._get_existing_or_create_box(box_name, reconnect=False)
        return box.link_status()

    def _get_existing_or_create_box(
        self,
        box_name: str,
        *,
        reconnect: bool,
    ) -> Quel1Box:
        """Return an existing pooled box or create one when absent."""
        boxpool = self._connection_manager.boxpool
        if boxpool is not None and box_name in boxpool._boxes:
            return boxpool._boxes[box_name][0]
        return self._create_box(box_name, reconnect=reconnect)

    def _create_boxpool(
        self,
        box_names: list[str],
        *,
        parallel: bool | None = None,
    ) -> BoxPool:
        """
        Create a box pool and reconnect boxes.

        Parameters
        ----------
        box_names : list[str]
            Box names to add to the pool.
        parallel : bool, optional
            Whether to process box creation/reconnect in parallel.

        Returns
        -------
        BoxPool
            Created box pool with connected boxes.
        """
        if parallel is None:
            parallel = _DEFAULT_PARALLEL_MODE
        db = self.qubecalib.system_config_database
        boxpool = self._driver.BoxPool()
        clockmaster_setting = db._clockmaster_setting
        if clockmaster_setting is not None:
            boxpool.create_clock_master(ipaddr=str(clockmaster_setting.ipaddr))

        box_settings = db._box_settings
        settings_by_name: dict[str, BoxSetting] = {}
        for box_name in box_names:
            if box_name not in box_settings:
                raise ValueError(f"box({box_name}) is not defined")
            settings_by_name[box_name] = box_settings[box_name]

        boxes_to_reconnect: list[Quel1Box] = []
        if parallel and box_names:
            created_boxes = run_parallel_map(
                box_names,
                lambda box_name: db.create_box(box_name, reconnect=False),
                key=lambda box_name: box_name,
                max_workers=min(_MAX_BOX_PARALLEL_WORKERS, len(box_names)),
            )
            for box_name in box_names:
                setting = settings_by_name[box_name]
                box = created_boxes[box_name]
                sequencer = self._driver.SequencerClient(str(setting.ipaddr_sss))
                boxpool._boxes[box_name] = (box, sequencer)
                boxpool._linkstatus[box_name] = False
                boxes_to_reconnect.append(box)
        else:
            for box_name in box_names:
                setting = settings_by_name[box_name]
                box = boxpool.create(
                    box_name,
                    ipaddr_wss=str(setting.ipaddr_wss),
                    ipaddr_sss=str(setting.ipaddr_sss),
                    ipaddr_css=str(setting.ipaddr_css),
                    boxtype=setting.boxtype,
                )
                boxes_to_reconnect.append(box)

        if parallel and boxes_to_reconnect:
            max_workers = max(1, len(boxes_to_reconnect))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(box.reconnect) for box in boxes_to_reconnect]
                for future in futures:
                    future.result()
        else:
            for box in boxes_to_reconnect:
                box.reconnect()

        return boxpool

    def _create_box(self, box_name: str, *, reconnect: bool = True) -> Quel1Box:
        """
        Create a box from the system configuration.

        Parameters
        ----------
        box_name : str
            Box name to create.
        reconnect : bool, optional
            Whether to reconnect the box on creation.

        Returns
        -------
        Quel1Box
            Created box instance.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        self._check_box_availability(box_name)
        db = self.qubecalib.system_config_database
        return db.create_box(box_name, reconnect=reconnect)

    def _create_quel1system_from_boxpool(
        self,
        box_names: list[str],
    ) -> Quel1System:
        """
        Build a Quel1System from already-connected boxpool entries.

        Parameters
        ----------
        box_names : list[str]
            Box names to include in the system.

        Returns
        -------
        Quel1System
            Initialized system instance.
        """
        boxpool = self._connection_manager.boxpool
        if boxpool is None:
            raise ValueError("Boxes not connected. Call connect() method first.")

        db = self.qubecalib.system_config_database
        clockmaster_setting = db._clockmaster_setting
        if clockmaster_setting is None:
            raise ValueError("clock master is not found")

        pooled_boxes = boxpool._boxes
        boxes: list[Any] = [
            self._driver.NamedBox(name=box_name, box=pooled_boxes[box_name][0])
            for box_name in box_names
        ]
        system = self._driver.Quel1System.create(
            clockmaster=self._driver.QuBEMasterClient(str(clockmaster_setting.ipaddr)),
            boxes=boxes,
            update_copnfig_cache=False,
        )
        return system

    def connect(
        self,
        box_names: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Connect to the boxes.

        Parameters
        ----------
        box_names : str | list[str], optional
            List of box names to connect to. If None, connect to all available boxes.
        parallel : bool | None, optional
            If True, use parallel box reconnect implementation. If False, use
            legacy qubecalib implementation. If `None`, it follows
            `qubex.backend.quel1.DEFAULT_EXECUTION_MODE`.
        """
        self._connection_manager.connect(
            box_names=box_names,
            available_boxes=lambda: self.available_boxes,
            parallel=parallel,
            default_parallel_mode=_DEFAULT_PARALLEL_MODE,
            create_boxpool=lambda names, parallel_mode: self._create_boxpool(
                names,
                parallel=parallel_mode,
            ),
            create_quel1system_from_boxpool=self._create_quel1system_from_boxpool,
            create_resource_map=self.create_resource_map,
        )

    def _append_resource_if_new(
        self,
        resources: list[Any],
        seen: set[int],
        resource: Any | None,
    ) -> None:
        """Append one resource object once by identity."""
        if resource is None:
            return
        resource_id = id(resource)
        if resource_id in seen:
            return
        seen.add(resource_id)
        resources.append(resource)

    def _collect_held_resources(self) -> list[Any]:
        """Collect clockmaster and box objects currently held by the controller."""
        resources: list[Any] = []
        seen: set[int] = set()

        system = self._connection_manager.quel1system
        if system is not None:
            self._append_resource_if_new(
                resources,
                seen,
                getattr(system, "_clockmaster", None),
            )
            boxes = getattr(system, "boxes", None)
            if isinstance(boxes, Mapping):
                for box in boxes.values():
                    self._append_resource_if_new(resources, seen, box)

        boxpool = self._connection_manager.boxpool
        if boxpool is not None:
            self._append_resource_if_new(
                resources,
                seen,
                getattr(boxpool, "_clock_master", None),
            )
            pooled_boxes = getattr(boxpool, "_boxes", {})
            if isinstance(pooled_boxes, Mapping):
                for box, *_ in pooled_boxes.values():
                    self._append_resource_if_new(resources, seen, box)

        return resources

    def _disconnect_resource(self, resource: Any) -> None:
        """Disconnect one resource object via close/terminate if available."""
        for method_name in ("close", "terminate"):
            method = getattr(resource, method_name, None)
            if not callable(method):
                continue
            method()
            return

    def _disconnect_resource_safely(self, resource: Any) -> None:
        """Disconnect one resource and log errors without aborting cleanup."""
        try:
            self._disconnect_resource(resource)
        except Exception:
            logger.exception("Failed to disconnect backend resource: %r", resource)

    def disconnect(self) -> None:
        """Disconnect backend resources and reset connection-related state."""
        self._clear_quel1system_box_cache()
        self._connection_manager.disconnect(
            collect_held_resources=self._collect_held_resources,
            disconnect_resource_safely=self._disconnect_resource_safely,
        )

    def execute(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request using QuEL-1 execution defaults."""
        from .quel1_backend_executor import Quel1BackendExecutor

        factory = getattr(self, "create_measurement_backend_executor", None)
        manager_factory = None
        if callable(factory):
            manager_factory = lambda mode, clock: cast(
                Any,
                factory(
                    execution_mode=mode,
                    clock_health_checks=clock,
                ),
            )

        return self._execution_manager.execute(
            request=request,
            execution_mode=execution_mode,
            clock_health_checks=clock_health_checks,
            create_default_executor=lambda mode, clock: Quel1BackendExecutor(
                backend_controller=self,
                execution_mode=mode,
                clock_health_checks=clock,
            ),
            create_measurement_backend_executor=manager_factory,
        )

    def get_box(self, box_name: str) -> Quel1Box:
        """
        Get the box object.

        Parameters
        ----------
        box_name : str
            Name of the box.

        Returns
        -------
        Quel1Box
            The box object.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        self._check_box_availability(box_name)
        return self._get_existing_or_create_box(box_name, reconnect=True)

    def initialize_awg_and_capunits(
        self,
        box_names: str | Collection[str],
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Initialize all awg and capture units in the specified boxes.

        Parameters
        ----------
        box_names : str | list[str]
            List of box names to initialize.
        parallel : bool | None, optional
            Whether to initialize boxes in parallel. If `None`, it follows
            `qubex.backend.quel1.DEFAULT_EXECUTION_MODE`.
        """
        self._connection_manager.initialize_awg_and_capunits(
            is_connected=self.is_connected,
            box_names=box_names,
            parallel=parallel,
            default_parallel_mode=_DEFAULT_PARALLEL_MODE,
            max_parallel_workers=_MAX_BOX_PARALLEL_WORKERS,
            initialize_box_awg_and_capunits=self._initialize_box_awg_and_capunits,
        )

    def _initialize_box_awg_and_capunits(self, box_name: str) -> None:
        """Initialize AWG and capture units for one box."""
        self._check_box_availability(box_name)
        boxpool = self._connection_manager.boxpool
        if boxpool is None or box_name not in boxpool._boxes:
            raise ValueError(
                f"Box {box_name} is not connected. Call connect() method first."
            )
        box = adapt_quel1_box(boxpool._boxes[box_name][0])
        box.initialize_all_awgunits()
        box.initialize_all_capunits()

    def sync_box_to_hardware(self, box: Box) -> None:
        """Apply one experiment-system box configuration to hardware."""
        for port in box.ports:
            port_type = getattr(port.type, "value", None)
            if port_type in ("CTRL", "READ_OUT", "PUMP"):
                try:
                    self.config_port(
                        box.id,
                        port=port.number,
                        lo_freq=port.lo_freq,
                        cnco_freq=port.cnco_freq,
                        vatt=getattr(port, "vatt", None),
                        sideband=getattr(port, "sideband", None),
                        fullscale_current=getattr(port, "fullscale_current", None),
                        rfswitch=port.rfswitch,
                    )
                    for gen_channel in port.channels:
                        self.config_channel(
                            box.id,
                            port=port.number,
                            channel=gen_channel.number,
                            fnco_freq=gen_channel.fnco_freq,
                        )
                except Exception:
                    logger.exception("Failed to configure %s", port.id)
            elif port_type == "READ_IN":
                try:
                    self.config_port(
                        box.id,
                        port=port.number,
                        lo_freq=port.lo_freq,
                        cnco_freq=port.cnco_freq,
                        rfswitch=port.rfswitch,
                    )
                    for cap_channel in port.channels:
                        self.config_runit(
                            box.id,
                            port=port.number,
                            runit=cap_channel.number,
                            fnco_freq=cap_channel.fnco_freq,
                        )
                except Exception:
                    logger.exception("Failed to configure %s", port.id)

    def sync_experiment_system_to_backend_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> None:
        """Rebuild backend-controller topology from one experiment-system model."""
        control_system = experiment_system.control_system
        control_params = experiment_system.control_params
        self.define_clockmaster(
            ipaddr=control_system.clock_master_address,
            reset=True,
        )
        self.set_box_options({box.id: box.options for box in control_system.boxes})

        for box in control_system.boxes:
            self.define_box(
                box_name=box.id,
                ipaddr_wss=box.address,
                boxtype=box.type.value,
            )

            for port in box.ports:
                port_type = getattr(port.type, "value", None)
                if port_type == "NA":
                    continue
                self.define_port(
                    port_name=port.id,
                    box_name=box.id,
                    port_number=port.number,
                )

                for channel in port.channels:
                    if port_type == "READ_IN":
                        mux = experiment_system.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        ndelay_or_nwait = control_params.get_capture_delay(mux.index)
                    elif port_type == "MNTR_IN":
                        ndelay_or_nwait = DEFAULT_CAPTURE_DELAY
                    else:
                        ndelay_or_nwait = 0
                    self.define_channel(
                        channel_name=channel.id,
                        port_name=port.id,
                        channel_number=channel.number,
                        ndelay_or_nwait=ndelay_or_nwait,
                    )

                if port_type in ("PUMP", "MNTR_OUT", "MNTR_IN"):
                    self.add_channel_target_relation(
                        port.channels[0].id,
                        port.id,
                    )

        for target in experiment_system.all_targets:
            self.define_target(
                target_name=target.label,
                channel_name=target.channel.id,
                target_frequency=target.frequency,
            )

        self.clear_command_queue()
        self.clear_cache()

    def linkup(
        self,
        box_name: str,
        noise_threshold: int | None = None,
        **kwargs: Any,
    ) -> Quel1Box:
        """
        Linkup a box and return the box object.

        Parameters
        ----------
        box_name : str
            Name of the box to linkup.

        Returns
        -------
        Quel1Box
            The linked up box object.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        return self._connection_manager.linkup(
            box_name=box_name,
            noise_threshold=noise_threshold,
            relaxed_noise_threshold=_RELAXED_NOISE_THRESHOLD,
            check_box_availability=self._check_box_availability,
            get_existing_or_create_box=lambda name, reconnect: (
                self._get_existing_or_create_box(name, reconnect=reconnect)
            ),
            **kwargs,
        )

    def linkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> dict[str, Quel1Box]:
        """
        Linkup all the boxes in the list.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        noise_threshold : int | None, optional
            Threshold for linkup noise checks.
        parallel : bool | None, optional
            Whether to link up boxes in parallel. If `None`, it follows
            `qubex.backend.quel1.DEFAULT_EXECUTION_MODE`.

        Returns
        -------
        dict[str, Quel1Box]
            Dictionary of linked up boxes.
        """
        linked_boxes = self._connection_manager.linkup_boxes(
            box_list=box_list,
            noise_threshold=noise_threshold,
            parallel=parallel,
            default_parallel_mode=_DEFAULT_PARALLEL_MODE,
            max_parallel_workers=_MAX_BOX_PARALLEL_WORKERS,
            linkup_box=lambda name, threshold: self.linkup(
                name,
                noise_threshold=threshold,
            ),
        )
        return cast(Any, linked_boxes)

    def _safe_linkup_box(
        self,
        *,
        box_name: str,
        noise_threshold: int | None,
    ) -> Quel1Box | None:
        """Link up one box and log failures without raising."""
        try:
            linked_box = self.linkup(box_name, noise_threshold=noise_threshold)
            logger.info(f"{box_name:5} : Linked up")
        except Exception as exc:
            logger.exception(f"{box_name:5} : Error during linkup", exc_info=exc)
            return None
        else:
            return linked_box

    @staticmethod
    def _fallback_linkup_box_result(
        box_name: str,
        exc: BaseException,
    ) -> Quel1Box | None:
        """Log a linkup error and return no box for the failed item."""
        logger.exception(f"{box_name:5} : Error during linkup", exc_info=exc)
        return None

    def relinkup(self, box_name: str, noise_threshold: int | None = None) -> None:
        """
        Relink a box.

        Parameters
        ----------
        box_name : str
            Name of the box to relinkup.
        """
        self._connection_manager.relinkup(
            box_name=box_name,
            noise_threshold=noise_threshold,
            relaxed_noise_threshold=_RELAXED_NOISE_THRESHOLD,
            check_box_availability=self._check_box_availability,
            get_existing_or_create_box=lambda name, reconnect: (
                self._get_existing_or_create_box(name, reconnect=reconnect)
            ),
            resolve_config_options=lambda name, boxtype: self._resolve_config_options(
                box_name=name,
                boxtype=boxtype,
            ),
        )

    def relinkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Relink all the boxes in the list.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        noise_threshold : int | None, optional
            Threshold for relinkup noise checks.
        parallel : bool | None, optional
            Whether to relink boxes in parallel. If `None`, it follows
            `qubex.backend.quel1.DEFAULT_EXECUTION_MODE`.
        """
        self._connection_manager.relinkup_boxes(
            box_list=box_list,
            noise_threshold=noise_threshold,
            parallel=parallel,
            default_parallel_mode=_DEFAULT_PARALLEL_MODE,
            max_parallel_workers=_MAX_BOX_PARALLEL_WORKERS,
            relinkup_box=lambda name, threshold: self.relinkup(
                name,
                noise_threshold=threshold,
            ),
        )

    @staticmethod
    def _log_relinkup_error(box_name: str, exc: BaseException) -> None:
        """Log a relinkup error for one box."""
        logger.exception(f"{box_name:5} : Error during relinkup", exc_info=exc)

    def read_clocks(self, box_list: list[str]) -> list[tuple[bool, int, int]]:
        """
        Read the clocks of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.

        Returns
        -------
        list[tuple[bool, int, int]]
            List of clocks.
        """
        return self._clock_manager.read_clocks(
            box_list=box_list,
            check_box_availability=self._check_box_availability,
            resolve_box_sss_ip=lambda box_name: str(
                self.qubecalib.system_config_database._box_settings[box_name].ipaddr_sss
            ),
            read_clock_at_ip=lambda ipaddr_sss: self._driver.SequencerClient(
                target_ipaddr=ipaddr_sss
            ).read_clock(),
        )

    def check_clocks(self, box_list: list[str]) -> bool:
        """
        Check the clock of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.

        Returns
        -------
        bool
            True if the clocks are synchronized, False otherwise.
        """
        return self._clock_manager.check_clocks(
            read_clocks=lambda: self.read_clocks(box_list),
        )

    def _get_clockmaster_setting_ipaddr(self) -> str | None:
        """Return configured clockmaster IP address if available."""
        db = self.qubecalib.system_config_database
        clockmaster_setting = db._clockmaster_setting
        if clockmaster_setting is None:
            return None
        return str(clockmaster_setting.ipaddr)

    def resync_clocks(self, box_list: list[str]) -> bool:
        """
        Resync the clock of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        """
        return self._clock_manager.resync_clocks(
            box_list=box_list,
            get_clockmaster_setting_ipaddr=self._get_clockmaster_setting_ipaddr,
            create_clockmaster_client=lambda ipaddr: self._driver.QuBEMasterClient(
                master_ipaddr=ipaddr
            ),
            resolve_box_sss_ip=lambda box_name: str(
                self.qubecalib.system_config_database._box_settings[box_name].ipaddr_sss
            ),
            check_clocks=self.check_clocks,
        )

    def reset_clockmaster(self, ipaddr: str) -> bool:
        """
        Reset the clock master.

        Parameters
        ----------
        ipaddr : str
            Clock master IP address.

        Returns
        -------
        bool
            True if reset succeeds.
        """
        return self._clock_manager.reset_clockmaster(
            ipaddr=ipaddr,
            get_clockmaster_setting_ipaddr=self._get_clockmaster_setting_ipaddr,
            create_clockmaster_client=lambda addr: self._driver.QuBEMasterClient(
                master_ipaddr=addr
            ),
        )

    def sync_clocks(self, box_list: list[str]) -> bool:
        """
        Sync the clocks of the boxes if not synchronized.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        """
        return self._clock_manager.sync_clocks(
            resync_clocks=lambda: self.resync_clocks(box_list),
            box_count=len(box_list),
        )

    def dump_box(self, box_name: str) -> dict:
        """
        Dump the box configuration.

        Parameters
        ----------
        box_name : str
            Name of the box.

        Returns
        -------
        dict
            Dictionary of box configuration.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        return self._configuration_manager.dump_box(
            box_name=box_name,
            check_box_availability=self._check_box_availability,
            create_box=lambda name, reconnect: self._create_box(
                name, reconnect=reconnect
            ),
        )

    def dump_port(self, box_name: str, port_number: int | tuple[int, int]) -> dict:
        """
        Dump the port configuration.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port_number : int | tuple[int, int]
            Port number.

        Returns
        -------
        dict
            Dictionary of port configuration.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        return self._configuration_manager.dump_port(
            box_name=box_name,
            port_number=port_number,
            check_box_availability=self._check_box_availability,
            create_box=lambda name, reconnect: self._create_box(
                name, reconnect=reconnect
            ),
        )

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
        """
        Configure the port of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        lo_freq : float | None, optional
            Local oscillator frequency in GHz.
        cnco_freq : float | None, optional
            CNCO frequency in GHz.
        vatt : int | None, optional
            VATT value.
        sideband : str | None, optional
            Sideband value.
        fullscale_current : int | None, optional
            Fullscale current value.
        rfswitch : str | None, optional
            RF switch value.
        """
        self._configuration_manager.config_port(
            box_name=box_name,
            port=port,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
            check_box_availability=self._check_box_availability,
            create_box=lambda name, reconnect: self._create_box(
                name, reconnect=reconnect
            ),
        )

    def config_channel(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        channel: int,
        fnco_freq: float | None = None,
    ) -> None:
        """
        Configure the channel of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        channel : int
            Channel number.
        fnco_freq : float | None, optional
            FNCO frequency in GHz.
        """
        self._configuration_manager.config_channel(
            box_name=box_name,
            port=port,
            channel=channel,
            fnco_freq=fnco_freq,
            check_box_availability=self._check_box_availability,
            create_box=lambda name, reconnect: self._create_box(
                name, reconnect=reconnect
            ),
        )

    def config_runit(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        runit: int,
        fnco_freq: float | None = None,
    ) -> None:
        """
        Configure the runit of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        runit : int
            Runit number.
        fnco_freq : float | None, optional
            FNCO frequency in GHz.
        """
        self._configuration_manager.config_runit(
            box_name=box_name,
            port=port,
            runit=runit,
            fnco_freq=fnco_freq,
            check_box_availability=self._check_box_availability,
            create_box=lambda name, reconnect: self._create_box(
                name, reconnect=reconnect
            ),
        )

    def add_sequencer(self, sequencer: Sequencer) -> None:
        """
        Add a sequencer to the queue.

        Parameters
        ----------
        sequencer : Sequencer
            The sequencer to add to the queue.
        """
        self._configuration_manager.add_sequencer(
            qubecalib=self.qubecalib,
            sequencer=sequencer,
        )

    def show_command_queue(self) -> None:
        """Show the current command queue."""
        logger.info(
            self._configuration_manager.show_command_queue(qubecalib=self.qubecalib)
        )

    def clear_command_queue(self) -> None:
        """Clear the command queue."""
        self._configuration_manager.clear_command_queue(qubecalib=self.qubecalib)

    def define_clockmaster(self, *, ipaddr: str, reset: bool = True) -> None:
        """
        Define the clock master in qube-calib.

        Parameters
        ----------
        ipaddr : str
            Clock master IP address.
        reset : bool, optional
            Whether to reset clock master on define.
        """
        self._configuration_manager.define_clockmaster(
            qubecalib=self.qubecalib,
            ipaddr=ipaddr,
            reset=reset,
        )

    def define_box(
        self,
        *,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
    ) -> None:
        """
        Define a box in qube-calib.

        Parameters
        ----------
        box_name : str
            Box name.
        ipaddr_wss : str
            WSS IP address.
        boxtype : str
            Box type label.
        """
        self._configuration_manager.define_box(
            qubecalib=self.qubecalib,
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
        """
        Define a port in qube-calib.

        Parameters
        ----------
        port_name : str
            Port name.
        box_name : str
            Box name owning the port.
        port_number : int | tuple[int, int]
            Port number.
        """
        self._configuration_manager.define_port(
            qubecalib=self.qubecalib,
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
        """
        Define a channel in qube-calib.

        Parameters
        ----------
        channel_name : str
            Channel name.
        port_name : str
            Port name owning the channel.
        channel_number : int
            Channel number.
        ndelay_or_nwait : int, optional
            Capture delay or wait words.
        """
        self._configuration_manager.define_channel(
            qubecalib=self.qubecalib,
            channel_name=channel_name,
            port_name=port_name,
            channel_number=channel_number,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def add_channel_target_relation(self, channel_name: str, target_name: str) -> None:
        """
        Add a channel-target relation if it does not already exist.

        Parameters
        ----------
        channel_name : str
            Channel name.
        target_name : str
            Target name.
        """
        self._configuration_manager.add_channel_target_relation(
            qubecalib=self.qubecalib,
            channel_name=channel_name,
            target_name=target_name,
        )

    def define_target(
        self,
        target_name: str,
        channel_name: str,
        target_frequency: float | None = None,
    ) -> None:
        """
        Define a target.

        Parameters
        ----------
        target_name : str
            Name of the target.
        channel_name : str
            Name of the channel.
        target_frequency : float, optional
            Frequency of the target in GHz.
        """
        self._configuration_manager.define_target(
            qubecalib=self.qubecalib,
            target_name=target_name,
            channel_name=channel_name,
            target_frequency=target_frequency,
        )

    def modify_target_frequency(self, target: str, frequency: float) -> None:
        """
        Modify the target frequency.

        Parameters
        ----------
        target : str
            Name of the target.
        frequency : float
            Modified frequency in GHz.
        """
        self._configuration_manager.modify_target_frequency(
            qubecalib=self.qubecalib,
            target=target,
            frequency=frequency,
        )

    def modify_target_frequencies(self, frequencies: dict[str, float]) -> None:
        """
        Modify the target frequencies.

        Parameters
        ----------
        frequencies : dict[str, float]
            Dictionary of target frequencies.
        """
        self._configuration_manager.modify_target_frequencies(
            qubecalib=self.qubecalib,
            frequencies=frequencies,
        )

    def create_quel1_sequencer(
        self,
        *,
        gen_sampled_sequence: dict[str, Any],
        cap_sampled_sequence: dict[str, Any],
        resource_map: dict[str, list[dict]],
        interval: int,
    ) -> Sequencer:
        """
        Create a QuEL-1 sequencer wired to this controller's system resources.

        Parameters
        ----------
        gen_sampled_sequence : dict[str, Any]
            Generator sampled sequence map.
        cap_sampled_sequence : dict[str, Any]
            Capture sampled sequence map.
        resource_map : dict[str, list[dict]]
            Target resource map.
        interval : int
            Sequence interval in ns.

        Returns
        -------
        Sequencer
            Constructed sequencer.
        """
        from qubex.backend.quel1.compat.sequencer import Quel1Sequencer

        return Quel1Sequencer(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map=resource_map,  # type: ignore[arg-type]
            interval=interval,
            sysdb=self.qubecalib.sysdb,
            # TODO: Keep passing driver for constructor compatibility across
            # old/new driver packages. `Quel1Sequencer` itself must continue to
            # resolve port configs on boxpool path to avoid readout-sideband
            # mismatches on R8.
            driver=self.quel1system,
        )

    def create_gen_sampled_sequence(
        self,
        *,
        target_name: str,
        real: Any,
        imag: Any,
        modulation_frequency: float,
    ) -> Any:
        """
        Create a generator sampled sequence via qubecalib.neopulse.

        Parameters
        ----------
        target_name : str
            Target label.
        real : Any
            Real samples.
        imag : Any
            Imaginary samples.
        modulation_frequency : float
            Modulation frequency.

        Returns
        -------
        Any
            GenSampledSequence object.
        """
        return self._driver.GenSampledSequence(
            target_name=target_name,
            prev_blank=0,
            post_blank=None,
            original_prev_blank=0,
            original_post_blank=None,
            modulation_frequency=modulation_frequency,
            sub_sequences=[
                self._driver.GenSampledSubSequence(
                    real=real,
                    imag=imag,
                    repeats=1,
                    post_blank=None,
                    original_post_blank=None,
                )
            ],
        )

    def create_cap_sampled_sequence(
        self,
        *,
        target_name: str,
        modulation_frequency: float,
        capture_delay: int,
        capture_slots: list[tuple[int, int]],
    ) -> Any:
        """
        Create a capture sampled sequence via qubecalib.neopulse.

        Parameters
        ----------
        target_name : str
            Target label.
        modulation_frequency : float
            Modulation frequency.
        capture_delay : int
            Capture delay in samples.
        capture_slots : list[tuple[int, int]]
            List of `(duration, post_blank)` sample counts.

        Returns
        -------
        Any
            CapSampledSequence object.
        """
        cap_sub_sequence = self._driver.CapSampledSubSequence(
            capture_slots=[],
            repeats=None,
            prev_blank=capture_delay,
            post_blank=None,
            original_prev_blank=0,
            original_post_blank=None,
        )
        for duration, post_blank in capture_slots:
            cap_sub_sequence.capture_slots.append(
                self._driver.CaptureSlots(
                    duration=duration,
                    post_blank=post_blank,
                    original_duration=None,  # type: ignore[arg-type]
                    original_post_blank=None,  # type: ignore[arg-type]
                )
            )
        return self._driver.CapSampledSequence(
            target_name=target_name,
            repeats=None,
            prev_blank=0,
            post_blank=None,
            original_prev_blank=0,
            original_post_blank=None,
            modulation_frequency=modulation_frequency,
            sub_sequences=[cap_sub_sequence],
        )

    def execute_sequencer(
        self,
        sequencer: Sequencer,
        *,
        repeats: int,
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
        enable_sum: bool = False,
        enable_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> Quel1BackendRawResult:
        """
        Execute a single sequence and return the measurement result.

        Parameters
        ----------
        sequencer : Sequencer
            The sequencer to execute.
        repeats : int
            Number of repeats of the sequence.
        integral_mode : {"integral", "single"}, optional
            Integral mode.
        dsp_demodulation : bool, optional
            Enable DSP demodulation.
        software_demodulation : bool, optional
            Enable software demodulation.

        Returns
        -------
        RawResult
            Measurement result.
        """
        return cast(
            Quel1BackendRawResult,
            self._execution_manager.execute_sequencer(
                sequencer=sequencer,
                repeats=repeats,
                integral_mode=integral_mode,
                dsp_demodulation=dsp_demodulation,
                software_demodulation=software_demodulation,
                enable_sum=enable_sum,
                enable_classification=enable_classification,
                line_param0=line_param0,
                line_param1=line_param1,
                make_backend_raw_result=self._make_backend_raw_result,
            ),
        )

    def execute_sequencer_parallel(
        self,
        sequencer: Sequencer,
        *,
        repeats: int,
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
        enable_sum: bool = False,
        enable_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        clock_health_checks: bool = False,
    ) -> Quel1BackendRawResult:
        """
        Execute a single sequence with parallelized multi-box action build.

        Parameters
        ----------
        sequencer : Sequencer
            The sequencer to execute.
        repeats : int
            Number of repeats of the sequence.
        integral_mode : {"integral", "single"}, optional
            Integral mode.
        dsp_demodulation : bool, optional
            Enable DSP demodulation.
        software_demodulation : bool, optional
            Enable software demodulation.
        enable_sum : bool, optional
            Enable DSP summation.
        enable_classification : bool, optional
            Enable DSP classification.
        line_param0 : tuple[float, float, float] | None, optional
            Classifier line parameter 0.
        line_param1 : tuple[float, float, float] | None, optional
            Classifier line parameter 1.
        clock_health_checks : bool, optional
            Whether to enable additional clock-health diagnostics and
            inter-box timediff estimation.

        Returns
        -------
        Quel1BackendRawResult
            Measurement result with qubecalib-compatible parsed payload.
        """
        return cast(
            Quel1BackendRawResult,
            self._execution_manager.execute_sequencer_parallel(
                sequencer=sequencer,
                repeats=repeats,
                integral_mode=integral_mode,
                dsp_demodulation=dsp_demodulation,
                software_demodulation=software_demodulation,
                enable_sum=enable_sum,
                enable_classification=enable_classification,
                line_param0=line_param0,
                line_param1=line_param1,
                clock_health_checks=clock_health_checks,
                action_builder=self._driver.Action.build,
                runit_setting_factory=self._driver.RunitSetting,
                runit_id_factory=self._driver.RunitId,
                awg_setting_factory=self._driver.AwgSetting,
                awg_id_factory=self._driver.AwgId,
                make_backend_raw_result=self._make_backend_raw_result,
            ),
        )


# TODO: Remove this alias in future versions.
DeviceController = Quel1BackendController
