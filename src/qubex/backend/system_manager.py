"""Coordinate synchronization between software and hardware states."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.prompt import Confirm
from typing_extensions import Self, deprecated

from qubex.constants import DEFAULT_RAWDATA_DIR
from qubex.typing import ConfigurationMode

from .config_loader import ConfigLoader
from .control_system import Box, CapPort, GenPort, PortType
from .experiment_system import ExperimentSystem
from .parallel_box_executor import run_parallel_each, run_parallel_map
from .quel1.quel1_backend_constants import DEFAULT_CAPTURE_DELAY
from .quel1.quel1_backend_controller import Quel1BackendController

logger = logging.getLogger(__name__)


class BackendSettings(dict[str, dict]):
    """Raw per-box settings returned by `backend_controller.dump_box`."""

    def __init__(self, initial: Mapping[str, dict] | None = None) -> None:
        """
        Initialize backend settings.

        Parameters
        ----------
        initial : Mapping[str, dict] | None, optional
            Initial per-box settings.
        """
        super().__init__(deepcopy(dict(initial or {})))

    @property
    def hash(self) -> int:
        """Return a stable hash of nested settings content."""
        return hash(self._freeze(self))

    @classmethod
    def _freeze(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(
                (key, cls._freeze(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        if isinstance(value, list):
            return tuple(cls._freeze(item) for item in value)
        return value


@dataclass
class SystemState:
    """Hash summary of synchronization-relevant state components."""

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
    Coordinate synchronization between software and hardware states.

    This singleton owns three related states:

    - experiment-system state (in-memory model used by application logic)
    - backend-controller state (qubecalib system model and caches)
    - backend-settings state (raw `dump_box` snapshots indexed by box ID)

    It exposes explicit pull/push operations and keeps a cached hash-based
    snapshot for consistency checks.

    Attributes
    ----------
    _instance : SystemManager
        Shared singleton instance.
    _initialized : bool
        Whether one-time initialization is complete.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs) -> SystemManager:
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def shared(cls) -> SystemManager:
        """
        Return the shared singleton instance.

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
        Initialize singleton-managed runtime state once.

        Notes
        -----
        This class is a singleton. Use `SystemManager.shared()` to access it.
        """
        if self._initialized:
            return
        self._experiment_system = None
        self._backend_controller = Quel1BackendController()
        self._backend_settings: BackendSettings = BackendSettings()
        self._cached_state = SystemState(0, 0, 0)
        self._rawdata_dir = None
        self._initialized = True

    @property
    def rawdata_dir(self) -> Path | None:
        """Return the current raw-data output directory."""
        return self._rawdata_dir

    @property
    def config_loader(self) -> ConfigLoader:
        """Return the active configuration loader."""
        return self._config_loader

    @property
    def is_loaded(self) -> bool:
        """Return whether an experiment system has been loaded."""
        return self._experiment_system is not None

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the loaded experiment system."""
        if self._experiment_system is None:
            raise ValueError("Experiment system is not loaded.")
        return self._experiment_system

    @property
    def backend_controller(self) -> Quel1BackendController:
        """Return the backend controller."""
        return self._backend_controller

    @property
    @deprecated("Use `backend_controller` property instead.")
    def device_controller(self) -> Quel1BackendController:
        """Backward-compatible alias of `backend_controller`."""
        return self.backend_controller

    @property
    def backend_settings(self) -> BackendSettings:
        """Return cached backend settings snapshots keyed by box ID."""
        if isinstance(self._backend_settings, BackendSettings):
            return self._backend_settings
        return BackendSettings(self._backend_settings)

    @property
    @deprecated("Use `backend_settings` property instead.")
    def device_settings(self) -> dict[str, dict]:
        """Backward-compatible alias of `backend_settings`."""
        return dict(self.backend_settings)

    @property
    def current_state(self) -> SystemState:
        """Return hash summary of current in-memory state."""
        return SystemState(
            experiment_system=self.experiment_system.hash,
            backend_controller=self.backend_controller.hash,
            backend_settings=self.backend_settings.hash,
        )

    @property
    @deprecated("Use `current_state` property instead.")
    def state(self) -> SystemState:
        """Backward-compatible alias of `current_state`."""
        return self.current_state

    @property
    def cached_state(self) -> SystemState:
        """Return hash summary captured at last successful sync."""
        return self._cached_state

    def set_rawdata_dir(self, value: Path | str | None) -> None:
        """
        Set the raw-data output directory.

        Parameters
        ----------
        value : Path | str | None
            Target directory path. If `None`, output is disabled.
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
        Load configuration and rebuild runtime software/controller model state.

        Parameters
        ----------
        chip_id : str
            Chip identifier.
        config_dir : Path | str, optional
            Directory containing configuration files.
        params_dir : Path | str, optional
            Directory containing parameter files.
        targets_to_exclude : list[str], optional
            Target labels to exclude from the loaded model.
        configuration_mode : ConfigurationMode, optional
            Configuration mode passed to `ConfigLoader`.
        mock_mode : bool, optional
            If `True`, skip backend-controller model synchronization.
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
        Pull selected hardware boxes into software-managed state.

        The method fetches raw backend settings from hardware for `box_ids`,
        merges them into cached settings, then applies the same subset to:

        - backend-controller cache
        - experiment-system control parameters

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to pull from hardware.
        parallel : bool | None, optional
            Whether to fetch per-box settings in parallel. If `None`, defaults
            to `True`.
        """
        fetched_backend_settings = self._fetch_backend_settings_from_hardware(
            box_ids=box_ids,
            parallel=parallel,
        )
        previous_backend_settings = BackendSettings(self.backend_settings)
        previous_box_cache = self.backend_controller.get_box_config_cache()
        merged_backend_settings = self._merge_backend_settings(
            base_settings=previous_backend_settings,
            patch_settings=fetched_backend_settings,
        )
        try:
            self._set_backend_settings(merged_backend_settings)
            self._sync_backend_settings_to_device_controller(
                backend_settings=fetched_backend_settings
            )
            self._sync_backend_settings_to_experiment_system(
                backend_settings=fetched_backend_settings
            )
        except Exception:
            self._set_backend_settings(previous_backend_settings)
            self.backend_controller.replace_box_config_cache(previous_box_cache)
            raise
        self._update_cached_state()

    def push(
        self,
        box_ids: Sequence[str],
        confirm: bool = True,
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Push software experiment-system settings to selected hardware boxes.

        After hardware writes, this method re-fetches raw backend settings for
        `box_ids` and merges them into cached backend settings to keep cache
        state consistent with hardware.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to push to hardware.
        confirm : bool, optional
            Whether to prompt before applying hardware writes.
        parallel : bool | None, optional
            Whether to configure selected boxes in parallel. If `None`,
            defaults to `True`.
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

        self._sync_experiment_system_to_hardware(
            boxes=boxes,
            parallel=parallel,
        )
        fetched_backend_settings = self._fetch_backend_settings_from_hardware(
            box_ids=box_ids,
            parallel=parallel,
        )
        previous_backend_settings = BackendSettings(self.backend_settings)
        previous_box_cache = self.backend_controller.get_box_config_cache()
        merged_backend_settings = self._merge_backend_settings(
            base_settings=previous_backend_settings,
            patch_settings=fetched_backend_settings,
        )
        try:
            self._set_backend_settings(merged_backend_settings)
            self._sync_backend_settings_to_device_controller(
                backend_settings=fetched_backend_settings
            )
        except Exception:
            self._set_backend_settings(previous_backend_settings)
            self.backend_controller.replace_box_config_cache(previous_box_cache)
            raise
        self._update_cached_state()

    def is_synced(
        self,
        *,
        box_ids: Sequence[str],
    ) -> bool:
        """
        Check synchronization status for selected boxes.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to validate against current hardware snapshots.

        Returns
        -------
        bool
            `True` if cached state hash matches and selected backend settings
            are identical to freshly fetched hardware snapshots.
        """
        if self.current_state != self.cached_state:
            warnings.warn(
                "The current state is different from the cached state. ",
                category=UserWarning,
                stacklevel=2,
            )
            return False
        fetched_backend_settings = self._fetch_backend_settings_from_hardware(
            box_ids=box_ids
        )
        current_backend_settings_subset = {
            box_id: self.backend_settings.get(box_id, {}) for box_id in box_ids
        }
        if current_backend_settings_subset != fetched_backend_settings:
            warnings.warn(
                "The current backend settings are different from the fetched backend settings. ",
                category=UserWarning,
                stacklevel=2,
            )
            return False
        return True

    @staticmethod
    def _merge_backend_settings(
        *,
        base_settings: BackendSettings,
        patch_settings: BackendSettings,
    ) -> BackendSettings:
        """Return merged backend settings where `patch_settings` overrides `base_settings`."""
        merged = BackendSettings(base_settings)
        for box_id, box_config in patch_settings.items():
            merged[box_id] = deepcopy(box_config)
        return merged

    def _set_backend_settings(self, backend_settings: Mapping[str, dict]) -> None:
        """Replace cached backend settings with normalized `BackendSettings`."""
        self._backend_settings = BackendSettings(backend_settings)

    def _sync_experiment_system_to_hardware(
        self,
        *,
        boxes: Sequence[Box],
        parallel: bool | None = None,
    ) -> None:
        """
        Apply experiment-system port/channel parameters to hardware boxes.

        Parameters
        ----------
        boxes : Sequence[Box]
            Target boxes to configure on hardware.
        parallel : bool | None, optional
            Whether to configure boxes in parallel. If `None`, defaults to
            `True`.
        """
        if parallel is None:
            parallel = True
        if not boxes:
            return
        if not parallel:
            for box in boxes:
                self._sync_box_to_hardware(box)
            return

        run_parallel_each(
            boxes,
            self._sync_box_to_hardware,
            on_error=self._log_box_sync_error,
        )

    @staticmethod
    def _log_box_sync_error(box: Box, exc: BaseException) -> None:
        """Log a failure during per-box hardware synchronization."""
        logger.exception("Failed to configure box %s", box.id, exc_info=exc)

    def _sync_box_to_hardware(self, box: Box) -> None:
        """Apply experiment-system port/channel parameters to one hardware box."""
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
    ) -> BackendSettings:
        """
        Fetch raw backend settings from hardware for selected boxes.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to read from hardware.
        parallel : bool | None, optional
            Whether to perform concurrent per-box reads. If `None`, defaults
            to `True`.

        Returns
        -------
        BackendSettings
            Raw `dump_box` snapshots keyed by box ID.

        Notes
        -----
        This method has no side effects on `SystemManager` state.
        """
        if parallel is None:
            parallel = True
        boxes = [self.experiment_system.get_box(box_id) for box_id in box_ids]
        result = BackendSettings()
        if not boxes:
            return result

        def _dump_box(box: Box) -> dict[str, Any]:
            return self.backend_controller.dump_box(box.id)

        if not parallel:
            for box in boxes:
                result[box.id] = self.backend_controller.dump_box(box.id)
            return result

        result.update(
            run_parallel_map(
                boxes,
                _dump_box,
                key=lambda box: box.id,
                as_completed_order=True,
                on_error=self._fallback_dump_box_result,
            )
        )
        return result

    @staticmethod
    def _fallback_dump_box_result(box: Box, exc: BaseException) -> dict[str, Any]:
        """Log a box dump failure and return an empty fallback config."""
        logger.exception("Failed to dump box %s", box.id, exc_info=exc)
        return {}

    def _sync_backend_settings_to_device_controller(
        self,
        *,
        backend_settings: BackendSettings | None = None,
    ) -> None:
        """
        Apply backend-settings snapshots to backend-controller cache.

        Parameters
        ----------
        backend_settings : BackendSettings | None, optional
            Settings to apply. If `None`, uses `self._backend_settings`.
        """
        if backend_settings is None:
            backend_settings = self._backend_settings
        self.backend_controller.update_box_config_cache(backend_settings)

    def _sync_backend_settings_to_experiment_system(
        self,
        *,
        backend_settings: BackendSettings | None = None,
    ) -> None:
        """
        Apply backend-settings snapshots to the in-memory experiment system.

        Parameters
        ----------
        backend_settings : BackendSettings | None, optional
            Settings to apply. If `None`, uses `self._backend_settings`.

        Notes
        -----
        Only synchronization-target ports are applied.
        """
        updates: list[
            tuple[
                str,
                int,
                Literal["U", "L"] | None,
                int | None,
                int,
                list[int],
                int | None,
            ]
        ] = []
        if backend_settings is None:
            backend_settings = self._backend_settings
        for box_id, box_config in backend_settings.items():
            ports_config = box_config.get("ports", {})
            try:
                box = self.experiment_system.get_box(box_id)
            except KeyError:
                logger.warning("Box %s is not found.", box_id)
                continue
            for experiment_port in box.ports:
                if experiment_port.type in (PortType.NOT_AVAILABLE, PortType.MNTR_OUT):
                    continue
                port_number = experiment_port.number
                if not isinstance(port_number, int):
                    continue
                port_config = ports_config.get(port_number)
                if not isinstance(port_config, dict):
                    continue
                direction = port_config.get("direction")
                lo_freq = port_config.get("lo_freq")
                lo_freq = int(lo_freq) if lo_freq is not None else None
                cnco_freq = int(port_config["cnco_freq"])
                if direction == "out":
                    raw_sideband = port_config.get("sideband")
                    sideband: Literal["U", "L"] | None = (
                        raw_sideband if raw_sideband in ("U", "L") else None
                    )
                    fullscale_current = port_config.get("fullscale_current")
                    fullscale_current = (
                        int(fullscale_current)
                        if fullscale_current is not None
                        else None
                    )
                    fnco_freqs = [
                        int(channel["fnco_freq"])
                        for channel in port_config.get("channels", {}).values()
                    ]
                elif direction == "in":
                    sideband = None
                    fullscale_current = None
                    fnco_freqs = [
                        int(channel["fnco_freq"])
                        for channel in port_config.get("runits", {}).values()
                    ]
                else:
                    continue
                channels = getattr(experiment_port, "channels", ())
                expected_fnco_count = (
                    len(channels)
                    if isinstance(channels, tuple) and len(channels) > 0
                    else None
                )
                if (
                    expected_fnco_count is not None
                    and len(fnco_freqs) != expected_fnco_count
                ):
                    logger.warning(
                        "Skipping backend port sync for %s:%s due to fnco count mismatch "
                        "(expected=%s, actual=%s).",
                        box_id,
                        port_number,
                        expected_fnco_count,
                        len(fnco_freqs),
                    )
                    continue
                updates.append(
                    (
                        box_id,
                        port_number,
                        sideband,
                        lo_freq,
                        cnco_freq,
                        fnco_freqs,
                        fullscale_current,
                    )
                )
        for (
            box_id,
            port_number,
            sideband,
            lo_freq,
            cnco_freq,
            fnco_freqs,
            fullscale_current,
        ) in updates:
            self.experiment_system.control_system.set_port_params(
                box_id=box_id,
                port_number=port_number,
                sideband=sideband,
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
                fnco_freqs=fnco_freqs,
                fullscale_current=fullscale_current,
            )

    def _sync_experiment_system_to_backend_controller(self) -> None:
        """Rebuild backend-controller model objects from experiment-system state."""
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
        """Refresh `cached_state` from current hash summaries."""
        self._cached_state = self.current_state

    @contextmanager
    def modified_frequencies(
        self, target_frequencies: dict[str, float]
    ) -> Iterator[None]:
        """
        Temporarily override target frequencies on software and controller.

        Parameters
        ----------
        target_frequencies : dict[str, float]
            Mapping from target label to temporary frequency.
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
        Temporarily override backend settings for one target channel path.

        Parameters
        ----------
        label : str
            Target label.
        lo_freq : int | None
            Temporary LO frequency.
        cnco_freq : int
            Temporary CNCO frequency.
        fnco_freq : int
            Temporary FNCO frequency.

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
        box_cache = self.backend_controller.get_box_config_cache()

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
        self.backend_controller.update_box_config_cache(
            {port.box_id: box_cache[port.box_id]}
        )
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
            self.backend_controller.update_box_config_cache(
                {cap_port.box_id: box_cache[cap_port.box_id]}
            )
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
            self.backend_controller.replace_box_config_cache(original_box_cache)

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
        """Backward-compatible alias of `modified_backend_settings`."""
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
        Temporarily set a raw-data output directory for a code block.

        Parameters
        ----------
        rawdata_dir : Path | str | None, optional
            Base output directory.
        tag : str | None, optional
            Optional subdirectory tag appended under `rawdata_dir`.
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
