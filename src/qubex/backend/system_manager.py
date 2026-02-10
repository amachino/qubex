"""System management for experiment and device control."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.prompt import Confirm
from typing_extensions import Self, deprecated

from qubex.constants import DEFAULT_RAWDATA_DIR
from qubex.typing import ConfigurationMode

from .config_loader import ConfigLoader
from .control_system import Box, CapPort, GenPort, PortType
from .experiment_system import ExperimentSystem
from .quel1.quel1_backend_constants import DEFAULT_CAPTURE_DELAY
from .quel1.quel1_backend_controller import Quel1BackendController

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Hash values for system state components."""

    experiment_system: int
    backend_controller: int
    backend_settings: int

    def __eq__(self, other: Self) -> bool:
        """Return equality based on hash components."""
        if not isinstance(other, SystemState):
            return NotImplemented
        return (
            self.experiment_system == other.experiment_system
            and self.backend_controller == other.backend_controller
            and self.backend_settings == other.backend_settings
        )


class SystemManager:
    """
    Singleton class to manage the system state.

    Attributes
    ----------
    _instance : SystemManager
        Shared instance of the SystemManager.
    _initialized : bool
        Whether the SystemManager is initialized.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs) -> SystemManager:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def shared(cls) -> SystemManager:
        """
        Get the shared instance of the SystemManager.

        Returns
        -------
        SystemManager
            Shared instance of the SystemManager.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the SystemManager.

        Notes
        -----
        This class is a singleton. Use `SystemManager.shared()` to get the shared instance.
        """
        if self._initialized:
            return
        self._experiment_system = None
        self._backend_controller = Quel1BackendController()
        self._backend_settings: dict[str, Any] = {}
        self._backend_box_configs: dict[str, dict[str, Any]] = {}
        self._cached_state = SystemState(0, 0, 0)
        self._rawdata_dir = None
        self._initialized = True

    @property
    def rawdata_dir(self) -> Path | None:
        """Get the directory for raw data."""
        return self._rawdata_dir

    @property
    def config_loader(self) -> ConfigLoader:
        """Get the configuration loader."""
        return self._config_loader

    @property
    def is_loaded(self) -> bool:
        """Check if the experiment system is loaded."""
        return self._experiment_system is not None

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        if self._experiment_system is None:
            raise ValueError("Experiment system is not loaded.")
        return self._experiment_system

    @property
    def backend_controller(self) -> Quel1BackendController:
        """Get the backend controller."""
        return self._backend_controller

    @property
    @deprecated("Use `backend_controller` property instead.")
    def device_controller(self) -> Quel1BackendController:
        """Get the device controller (backward-compatible alias)."""
        return self.backend_controller

    @property
    def backend_settings(self) -> dict[str, Any]:
        """Get the backend settings."""
        return self._backend_settings or {}

    @property
    @deprecated("Use `backend_settings` property instead.")
    def device_settings(self) -> dict[str, Any]:
        """Get the device settings (backward-compatible alias)."""
        return self.backend_settings

    @property
    def current_state(self) -> SystemState:
        """Get the current synchronization state."""
        return SystemState(
            experiment_system=self.experiment_system.hash,
            backend_controller=self.backend_controller.hash,
            backend_settings=hash(str(self.backend_settings)),
        )

    @property
    @deprecated("Use `current_state` property instead.")
    def state(self) -> SystemState:
        """Get the current state (backward-compatible alias)."""
        return self.current_state

    @property
    def cached_state(self) -> SystemState:
        """Get the cached state."""
        return self._cached_state

    def set_rawdata_dir(self, value: Path | str | None) -> None:
        """
        Update the directory for raw data.

        Parameters
        ----------
        value : Path | str | None
            The directory path for raw data.
        """
        if value is None:
            self._rawdata_dir = None
        else:
            self._rawdata_dir = Path(value)
            self._rawdata_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        *,
        chip_id: str,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: ConfigurationMode | None = None,
        mock_mode: bool = False,
    ) -> None:
        """
        Load the experiment system and device controller.

        Parameters
        ----------
        chip_id : str
            Chip ID.
        config_dir : Path | str, optional
            Configuration directory.
        params_dir : Path | str, optional
            Parameters directory.
        targets_to_exclude : list[str], optional
            List of target labels to exclude, by default None.
        configuration_mode : ConfigurationMode, optional
            Configuration mode, by default "ge-cr-cr".
        """
        self._config_loader = ConfigLoader(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            targets_to_exclude=targets_to_exclude,
            configuration_mode=configuration_mode,
        )
        self._mock_mode = mock_mode
        self._experiment_system = self._config_loader.get_experiment_system(chip_id)
        if self._mock_mode:
            # skip updating backend controller in mock mode
            return
        # update backend controller to reflect the new experiment system
        self._sync_experiment_system_to_backend_controller()

    def pull(
        self,
        box_ids: Sequence[str],
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Pull the hardware state to the software state.

        This method updates the software state to reflect the hardware state.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to fetch the backend settings for.
        parallel : bool | None, optional
            Whether to fetch backend settings in parallel, by default True.
        """
        self._fetch_backend_settings_from_hardware(box_ids=box_ids, parallel=parallel)
        self._sync_backend_settings_to_device_controller()
        self._sync_backend_settings_to_experiment_system()

    def push(
        self,
        box_ids: Sequence[str],
        confirm: bool = True,
    ) -> None:
        """
        Push the software state to the hardware state.

        This method updates the hardware state to reflect the software state.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to configure.
        confirm : bool, optional
            Whether to confirm the operation, by default True.
        """
        boxes = [self.experiment_system.get_box(box_id) for box_id in box_ids]
        boxes_str = "\n".join([f"{box.id} ({box.name})" for box in boxes])

        if confirm:
            confirmed = Confirm.ask(
                f"""
You are going to configure the following boxes:

[bold bright_green]{boxes_str}[/bold bright_green]

This operation will overwrite the existing backend settings. Do you want to continue?
"""
            )
            if not confirmed:
                logger.info("Operation cancelled.")
                return

        self._sync_experiment_system_to_hardware(boxes=boxes)
        self._fetch_backend_settings_from_hardware(box_ids=box_ids)
        self._sync_backend_settings_to_device_controller()

    def is_synced(
        self,
        *,
        box_ids: Sequence[str],
    ) -> bool:
        """
        Check if the state is synced.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to check.

        Returns
        -------
        bool
            Whether the state is synced.
        """
        if self.current_state != self.cached_state:
            warnings.warn(
                "The current state is different from the cached state. ",
                category=UserWarning,
                stacklevel=2,
            )
            return False
        current_backend_settings = self._backend_settings
        current_backend_box_configs = self._backend_box_configs
        self._fetch_backend_settings_from_hardware(box_ids=box_ids)
        fetched_backend_settings = self._backend_settings
        self._backend_settings = current_backend_settings
        self._backend_box_configs = current_backend_box_configs
        if self.backend_settings != fetched_backend_settings:
            warnings.warn(
                "The current backend settings are different from the fetched backend settings. ",
                category=UserWarning,
                stacklevel=2,
            )
            return False
        return True

    def _sync_experiment_system_to_hardware(
        self,
        *,
        boxes: Sequence[Box],
    ) -> None:
        """Sync experiment-system port/channel settings to hardware."""
        for box in boxes:
            for port in box.ports:
                if isinstance(port, GenPort):
                    if port.type in (PortType.CTRL, PortType.READ_OUT, PortType.PUMP):
                        try:
                            self.backend_controller.config_port(
                                box_name=box.id,
                                port=port.number,
                                lo_freq=port.lo_freq,
                                cnco_freq=port.cnco_freq,
                                vatt=port.vatt,
                                sideband=port.sideband,
                                fullscale_current=port.fullscale_current,
                                rfswitch=port.rfswitch,
                            )
                            for gen_channel in port.channels:
                                self.backend_controller.config_channel(
                                    box_name=box.id,
                                    port=port.number,
                                    channel=gen_channel.number,
                                    fnco_freq=gen_channel.fnco_freq,
                                )
                        except Exception:
                            logger.exception("Failed to configure %s", port.id)
                elif isinstance(port, CapPort):
                    if port.type in (PortType.READ_IN,):
                        try:
                            self.backend_controller.config_port(
                                box_name=box.id,
                                port=port.number,
                                lo_freq=port.lo_freq,
                                cnco_freq=port.cnco_freq,
                                rfswitch=port.rfswitch,
                            )
                            for cap_channel in port.channels:
                                self.backend_controller.config_runit(
                                    box_name=box.id,
                                    port=port.number,
                                    runit=cap_channel.number,
                                    fnco_freq=cap_channel.fnco_freq,
                                )
                        except Exception:
                            logger.exception("Failed to configure %s", port.id)

    def _fetch_backend_settings_from_hardware(
        self,
        box_ids: Sequence[str],
        *,
        parallel: bool | None = None,
    ) -> None:
        if parallel is None:
            parallel = True
        boxes = [self.experiment_system.get_box(box_id) for box_id in box_ids]
        result: dict[str, Any] = {}
        box_configs: dict[str, dict[str, Any]] = {}
        if not boxes:
            self._backend_settings = result
            self._backend_box_configs = box_configs
            return

        def _dump_box(box: Box) -> dict[str, Any]:
            return self.backend_controller.dump_box(box.id)

        def _collect_ports(box: Box, box_config: dict[str, Any]) -> None:
            box_configs[box.id] = box_config
            result[box.id] = {"ports": {}}
            for port in box.ports:
                if port.type not in (PortType.NOT_AVAILABLE, PortType.MNTR_OUT):
                    try:
                        result[box.id]["ports"][port.number] = box_config["ports"][
                            port.number
                        ]
                    except Exception:
                        logger.exception(
                            "Failed to fetch port %s for box %s",
                            port.number,
                            box.id,
                        )

        if not parallel:
            for box in boxes:
                box_config = self.backend_controller.dump_box(box.id)
                _collect_ports(box, box_config)
            self._backend_settings = result
            self._backend_box_configs = box_configs
            return

        max_workers = min(32, len(boxes))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_box = {executor.submit(_dump_box, box): box for box in boxes}
            for future in as_completed(future_to_box):
                box = future_to_box[future]
                try:
                    box_config = future.result()
                except Exception:
                    logger.exception("Failed to dump box %s", box.id)
                    box_config = {}
                _collect_ports(box, box_config)
        self._backend_settings = result
        self._backend_box_configs = box_configs

    def _sync_backend_settings_to_device_controller(
        self,
    ) -> None:
        """Sync fetched hardware box configs to backend-controller cache."""
        for box_id, box_config in self._backend_box_configs.items():
            self.backend_controller.boxpool._box_config_cache[box_id] = box_config  # noqa: SLF001
        self._update_cached_state()

    def _sync_backend_settings_to_experiment_system(self) -> None:
        """
        Sync backend settings to the in-memory experiment system.

        Notes
        -----
        This method also updates the experiment system to reflect backend settings.
        """
        for box_id, box in self._backend_settings.items():
            for port_number, port in box["ports"].items():
                direction = port["direction"]
                lo_freq = port.get("lo_freq")
                lo_freq = int(lo_freq) if lo_freq is not None else None
                cnco_freq = int(port["cnco_freq"])
                if direction == "out":
                    sideband = port.get("sideband")
                    fullscale_current = int(port["fullscale_current"])
                    fnco_freqs = [
                        int(channel["fnco_freq"])
                        for channel in port["channels"].values()
                    ]
                elif direction == "in":
                    sideband = None
                    fullscale_current = None
                    fnco_freqs = [
                        int(channel["fnco_freq"]) for channel in port["runits"].values()
                    ]
                self.experiment_system.control_system.set_port_params(
                    box_id=box_id,
                    port_number=port_number,
                    sideband=sideband,
                    lo_freq=lo_freq,
                    cnco_freq=cnco_freq,
                    fnco_freqs=fnco_freqs,
                    fullscale_current=fullscale_current,
                )
        self._update_cached_state()

    def _sync_experiment_system_to_backend_controller(self) -> None:
        experiment_system = self.experiment_system
        control_system = experiment_system.control_system
        control_params = experiment_system.control_params

        self.backend_controller.define_clockmaster(
            ipaddr=control_system.clock_master_address,
            reset=True,  # this option has no effect and will be removed in the future
        )
        self.backend_controller.set_box_options(
            {box.id: box.options for box in control_system.boxes}
        )

        for box in control_system.boxes:
            self.backend_controller.define_box(
                box_name=box.id,
                ipaddr_wss=box.address,
                boxtype=box.type.value,
            )

            for port in box.ports:
                if port.type == PortType.NOT_AVAILABLE:
                    continue
                self.backend_controller.define_port(
                    port_name=port.id,
                    box_name=box.id,
                    port_number=port.number,  # type: ignore
                )

                for channel in port.channels:
                    if port.type == PortType.READ_IN:
                        mux = experiment_system.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        ndelay_or_nwait = control_params.get_capture_delay(mux.index)
                    elif port.type == PortType.MNTR_IN:
                        ndelay_or_nwait = DEFAULT_CAPTURE_DELAY
                    else:
                        ndelay_or_nwait = 0
                    self.backend_controller.define_channel(
                        channel_name=channel.id,
                        port_name=port.id,
                        channel_number=channel.number,
                        ndelay_or_nwait=ndelay_or_nwait,
                    )

                if port.type in (
                    PortType.PUMP,
                    PortType.MNTR_OUT,
                    PortType.MNTR_IN,
                ):
                    self.backend_controller.add_channel_target_relation(
                        port.channels[0].id,
                        port.id,
                    )

        for target in experiment_system.all_targets:
            self.backend_controller.define_target(
                target_name=target.label,
                channel_name=target.channel.id,
                target_frequency=target.frequency,
            )

        # reset the cache
        self.backend_controller.clear_command_queue()
        self.backend_controller.clear_cache()
        self._update_cached_state()

    def _update_cached_state(self) -> None:
        """Update cached state from the current system state."""
        self._cached_state = self.current_state

    @contextmanager
    def modified_frequencies(
        self, target_frequencies: dict[str, float]
    ) -> Iterator[None]:
        """
        Temporarily modify the target frequencies.

        Parameters
        ----------
        target_frequencies : dict[str, float]
            The target frequencies to be modified.
        """
        original_frequencies = {
            target.label: target.frequency
            for target in self.experiment_system.targets
            if target.label in target_frequencies
        }
        self.experiment_system.modify_target_frequencies(target_frequencies)
        self.backend_controller.modify_target_frequencies(target_frequencies)
        try:
            yield
        finally:
            self.experiment_system.modify_target_frequencies(original_frequencies)
            self.backend_controller.modify_target_frequencies(original_frequencies)

    @contextmanager
    def modified_backend_settings(
        self,
        label: str,
        *,
        lo_freq: int | None,
        cnco_freq: int,
        fnco_freq: int,
    ) -> Iterator[None]:
        """
        Temporarily modify the backend settings.

        Parameters
        ----------
        backend_settings : dict[str, Any]
            The backend settings to be modified.

        Examples
        --------
        >>> with system_manager.modified_backend_settings(
        ...     "Q00",
        ...     lo_freq=10_000_000_000,
        ...     cnco_freq=1_500,
        ...     fnco_freq=750,
        ... ):
        ...     ...
        """
        target = self.experiment_system.get_target(label)
        channel = target.channel
        port = channel.port
        box_cache = self.backend_controller.boxpool._box_config_cache  # noqa: SLF001

        original_lo_freq = port.lo_freq
        original_cnco_freq = port.cnco_freq
        original_fnco_freq = channel.fnco_freq
        original_box_cache = deepcopy(box_cache)

        self.backend_controller.config_port(
            box_name=port.box_id,
            port=port.number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
        )
        self.backend_controller.config_channel(
            box_name=port.box_id,
            port=port.number,
            channel=channel.number,
            fnco_freq=fnco_freq,
        )
        port_cache = box_cache[port.box_id]["ports"][port.number]
        port_cache["lo_freq"] = lo_freq
        port_cache["cnco_freq"] = cnco_freq
        port_cache["channels"][channel.number]["fnco_freq"] = fnco_freq
        self.backend_controller.initialize_awg_and_capunits(port.box_id)

        if target.is_read:
            cap_channel = self.experiment_system.get_cap_target(label).channel
            cap_port = cap_channel.port
            self.backend_controller.config_port(
                box_name=cap_port.box_id,
                port=cap_port.number,
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
            )
            self.backend_controller.config_runit(
                box_name=cap_port.box_id,
                port=cap_port.number,
                runit=cap_channel.number,
                fnco_freq=fnco_freq,
            )
            cap_port_cache = box_cache[cap_port.box_id]["ports"][cap_port.number]
            cap_port_cache["lo_freq"] = lo_freq
            cap_port_cache["cnco_freq"] = cnco_freq
            cap_port_cache["runits"][cap_channel.number]["fnco_freq"] = fnco_freq
            self.backend_controller.initialize_awg_and_capunits(cap_port.box_id)

        self.experiment_system.update_port_params(
            label,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            fnco_freq=fnco_freq,
        )
        try:
            yield
        finally:
            # restore the original backend settings
            self.experiment_system.update_port_params(
                label,
                lo_freq=original_lo_freq,
                cnco_freq=original_cnco_freq,
                fnco_freq=original_fnco_freq,
            )
            self.backend_controller.config_port(
                box_name=port.box_id,
                port=port.number,
                lo_freq=original_lo_freq,
                cnco_freq=original_cnco_freq,
            )
            self.backend_controller.config_channel(
                box_name=port.box_id,
                port=port.number,
                channel=channel.number,
                fnco_freq=original_fnco_freq,
            )
            if target.is_read:
                self.backend_controller.config_port(
                    box_name=cap_port.box_id,
                    port=cap_port.number,
                    lo_freq=original_lo_freq,
                    cnco_freq=original_cnco_freq,
                )
                self.backend_controller.config_runit(
                    box_name=cap_port.box_id,
                    port=cap_port.number,
                    runit=cap_channel.number,
                    fnco_freq=original_fnco_freq,
                )

            # restore the original box config
            self.backend_controller.boxpool._box_config_cache = original_box_cache  # noqa: SLF001

    @deprecated("This method will be removed in future versions.")
    @contextmanager
    def modified_device_settings(
        self,
        label: str,
        *,
        lo_freq: int | None,
        cnco_freq: int,
        fnco_freq: int,
    ) -> Iterator[None]:
        """Temporarily modify the device settings (backward-compatible alias)."""
        with self.modified_backend_settings(
            label,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            fnco_freq=fnco_freq,
        ):
            yield

    @contextmanager
    def save_rawdata(
        self,
        *,
        rawdata_dir: Path | str = DEFAULT_RAWDATA_DIR,
        tag: str | None = None,
    ) -> Iterator[None]:
        """
        Context manager to save raw data to a specified directory.

        Parameters
        ----------
        rawdata_dir : Path | str | None, optional
            Directory to save raw data.
        tag : str | None, optional
            Tag to append to the raw data file name, by default None.
        """
        original_rawdata_dir = self.rawdata_dir
        rawdata_dir = Path(rawdata_dir)
        if tag is not None:
            rawdata_dir = rawdata_dir / tag
        self.set_rawdata_dir(rawdata_dir)
        try:
            yield
        finally:
            self.set_rawdata_dir(original_rawdata_dir)
