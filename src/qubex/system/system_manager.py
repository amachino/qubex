"""Coordinate synchronization between software and hardware states."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rich.prompt import Confirm
from typing_extensions import Self, deprecated

from qubex.backend.backend_controller import (
    BACKEND_KIND_QUEL1,
    BACKEND_KIND_QUEL3,
    BackendKind,
    SystemBackendController,
)
from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import Quel3BackendController
from qubex.constants import (
    DEFAULT_RAWDATA_DIR,
)
from qubex.typing import ConfigurationMode

from .config_loader import ConfigLoader
from .control_system import Box
from .experiment_system import ExperimentSystem
from .quel1.quel1_system_synchronizer import Quel1SystemSynchronizer
from .quel3.quel3_system_synchronizer import Quel3SystemSynchronizer
from .system_synchronizer import SystemSynchronizer

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
    - backend controller state (qubecalib system model and caches)
    - backend-settings state (raw `dump_box` snapshots indexed by box ID)

    It exposes explicit pull/push operations and keeps a cached hash-based
    snapshot for consistency checks.

    Current runtime assumption:
    - one active experiment/measurement session per process
    - this class remains singleton-managed until a later refactor moves it to
      session/experiment-owned state

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
        self._backend_kind: BackendKind = BACKEND_KIND_QUEL1
        self._backend_controller = self._create_backend_controller(self._backend_kind)
        self._system_synchronizer = self._create_system_synchronizer(
            self._backend_controller,
            self._backend_kind,
        )
        self._backend_settings: BackendSettings = BackendSettings()
        self._cached_state = SystemState(0, 0, 0)
        self._rawdata_dir = None
        self._mock_mode = False
        self._initialized = True

    @staticmethod
    def _create_backend_controller(
        backend_kind: BackendKind,
    ) -> SystemBackendController:
        """Create a backend controller instance for one experiment session."""
        if backend_kind == BACKEND_KIND_QUEL3:
            return Quel3BackendController()
        return Quel1BackendController()

    @property
    def backend_kind(self) -> BackendKind:
        """Return backend family selected for the current experiment session."""
        return self._backend_kind

    def set_backend_kind(self, backend_kind: BackendKind) -> None:
        """
        Select backend family for the current experiment session.

        Parameters
        ----------
        backend_kind : BackendKind
            Backend family. One session uses either QuEL-1 or QuEL-3.
        """
        if backend_kind == self._backend_kind:
            return
        self._backend_kind = backend_kind
        self._backend_controller = self._create_backend_controller(backend_kind)
        self._system_synchronizer = self._create_system_synchronizer(
            self._backend_controller,
            self._backend_kind,
        )
        self._backend_settings = BackendSettings()

    def _create_system_synchronizer(
        self,
        backend_controller: SystemBackendController,
        backend_kind: BackendKind | None = None,
    ) -> SystemSynchronizer | None:
        """Create backend-specific system synchronizer when supported."""
        resolved_backend_kind = backend_kind or self._backend_kind
        if resolved_backend_kind == BACKEND_KIND_QUEL1:
            return Quel1SystemSynchronizer(
                backend_controller=cast(Quel1BackendController, backend_controller),
            )
        if resolved_backend_kind == BACKEND_KIND_QUEL3:
            return Quel3SystemSynchronizer(
                backend_controller=cast(Quel3BackendController, backend_controller),
            )
        if isinstance(backend_controller, Quel1BackendController):
            return Quel1SystemSynchronizer(backend_controller=backend_controller)
        if isinstance(backend_controller, Quel3BackendController):
            return Quel3SystemSynchronizer(backend_controller=backend_controller)
        return None

    def _resolve_system_synchronizer(
        self,
    ) -> SystemSynchronizer | None:
        """Return active system synchronizer, refreshing built-in synchronizers when needed."""
        system_synchronizer = self._system_synchronizer
        if system_synchronizer is None:
            return None
        if isinstance(system_synchronizer, Quel1SystemSynchronizer):
            if (
                not isinstance(self._backend_controller, Quel1BackendController)
                or system_synchronizer.backend_controller
                is not self._backend_controller
            ):
                system_synchronizer = self._create_system_synchronizer(
                    self._backend_controller,
                    self._backend_kind,
                )
                self._system_synchronizer = system_synchronizer
            return system_synchronizer
        if isinstance(system_synchronizer, Quel3SystemSynchronizer):
            if (
                not isinstance(self._backend_controller, Quel3BackendController)
                or system_synchronizer.backend_controller
                is not self._backend_controller
            ):
                system_synchronizer = self._create_system_synchronizer(
                    self._backend_controller,
                    self._backend_kind,
                )
                self._system_synchronizer = system_synchronizer
            return system_synchronizer
        return system_synchronizer

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
    def backend_controller(self) -> SystemBackendController:
        """Return the backend controller."""
        return self._backend_controller

    @property
    @deprecated("Use `backend_controller` property instead.")
    def device_controller(self) -> SystemBackendController:
        """Backward-compatible alias of `backend_controller`."""
        return self.backend_controller

    @property
    def backend_settings(self) -> BackendSettings:
        """Return cached backend settings snapshots keyed by box ID."""
        if isinstance(self._backend_settings, BackendSettings):
            return self._backend_settings
        return BackendSettings(self._backend_settings)

    @property
    def current_state(self) -> SystemState:
        """Return hash summary of current in-memory state."""
        return SystemState(
            experiment_system=self.experiment_system.hash,
            backend_controller=self.backend_controller.hash,
            backend_settings=self.backend_settings.hash,
        )

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
        chip_id: str | None = None,
        system_id: str | None = None,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: ConfigurationMode | None = None,
        backend_kind: BackendKind | None = None,
        backend_controller: SystemBackendController | None = None,
        mock_mode: bool = False,
    ) -> None:
        """
        Load configuration and rebuild runtime software/controller model state.

        Parameters
        ----------
        chip_id : str | None, optional
            Deprecated chip identifier compatibility input.
        system_id : str | None, optional
            Canonical system identifier.
        config_dir : Path | str, optional
            Directory containing configuration files.
        params_dir : Path | str, optional
            Directory containing parameter files.
        targets_to_exclude : list[str], optional
            Target labels to exclude from the loaded model.
        configuration_mode : ConfigurationMode, optional
            Configuration mode passed to `ConfigLoader`.
        backend_kind : BackendKind | None, optional
            Backend family used for this experiment session.
        backend_controller : SystemBackendController | None, optional
            Backend controller to install for the active experiment session.
        mock_mode : bool, optional
            If `True`, skip backend controller model synchronization.
        """
        next_config_loader = ConfigLoader(
            chip_id=chip_id,
            system_id=system_id,
            config_dir=config_dir,
            params_dir=params_dir,
            autoload=False,
        )
        next_config_loader.load(
            targets_to_exclude=targets_to_exclude,
            configuration_mode=configuration_mode,
            backend_kind=backend_kind,
        )
        next_experiment_system = next_config_loader.get_experiment_system()

        resolved_backend_kind = next_config_loader.backend_kind
        self.set_backend_kind(resolved_backend_kind)
        if backend_controller is not None:
            self._validate_backend_controller_kind(
                backend_controller=backend_controller,
                backend_kind=resolved_backend_kind,
            )
            self._backend_controller = backend_controller
            self._system_synchronizer = self._create_system_synchronizer(
                backend_controller,
                resolved_backend_kind,
            )
            self._backend_settings = BackendSettings()
        self._config_loader = next_config_loader
        self._mock_mode = mock_mode
        self._experiment_system = next_experiment_system
        if self._mock_mode:
            # skip updating backend controller in mock mode
            return
        # update backend controller to reflect the new experiment system
        self._sync_experiment_system_to_backend_controller()

    @staticmethod
    def _validate_backend_controller_kind(
        *,
        backend_controller: SystemBackendController,
        backend_kind: BackendKind,
    ) -> None:
        """Validate that one backend controller matches the selected backend kind."""
        if backend_kind == BACKEND_KIND_QUEL1 and not isinstance(
            backend_controller, Quel1BackendController
        ):
            raise TypeError(
                "Expected `Quel1BackendController` for `backend_kind='quel1'`."
            )
        if backend_kind == BACKEND_KIND_QUEL3 and not isinstance(
            backend_controller, Quel3BackendController
        ):
            raise TypeError(
                "Expected `Quel3BackendController` for `backend_kind='quel3'`."
            )

    def pull(
        self,
        box_ids: Sequence[str],
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Pull selected hardware boxes into software-managed state.

        This method fetches raw backend settings from hardware for `box_ids`,
        merges them into cached settings, then applies the same subset to:

        - backend controller cache
        - experiment-system control parameters

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to pull from hardware.
        parallel : bool | None, optional
            Whether to fetch per-box settings in parallel. If `None`, defaults
            to `True`.
        """
        if not self._supports_box_settings_cache_sync():
            logger.info(
                "Skipping pull because this backend does not expose box-settings cache APIs."
            )
            self._update_cached_state()
            return
        fetched_backend_settings = self._fetch_backend_settings_from_hardware(
            box_ids=box_ids,
            parallel=parallel,
        )
        previous_backend_settings = BackendSettings(self.backend_settings)
        previous_box_cache = self._get_box_config_cache_snapshot()
        merged_backend_settings = self._merge_backend_settings(
            base_settings=previous_backend_settings,
            patch_settings=fetched_backend_settings,
        )
        try:
            # Keep manager state and backend cache as a full snapshot.
            # `fetched_backend_settings` may contain only the requested boxes.
            self._set_backend_settings(merged_backend_settings)
            self._sync_backend_settings_to_backend_controller(
                backend_settings=merged_backend_settings
            )
            # Apply only fetched boxes to the in-memory model to avoid touching
            # unrelated boxes during partial pull.
            self._sync_backend_settings_to_experiment_system(
                backend_settings=fetched_backend_settings
            )
        except Exception:
            self._set_backend_settings(previous_backend_settings)
            self._replace_box_config_cache(previous_box_cache)
            raise
        self._update_cached_state()

    def push(
        self,
        box_ids: Sequence[str],
        confirm: bool = True,
        *,
        parallel: bool | None = None,
        target_labels: Sequence[str] | None = None,
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
        target_labels : Sequence[str] | None, optional
            Logical target labels to apply for backends that support
            target-scoped hardware configuration.
        """
        supports_cache_sync = self._supports_box_settings_cache_sync()
        if not supports_cache_sync:
            logger.info(
                "Running push without backend-settings cache synchronization for this backend."
            )
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
                # `load()` clears backend controller caches before `push()`.
                # When push is canceled, restore the previous cache snapshot from
                # backend settings so subsequent read paths can continue to work.
                self._sync_backend_settings_to_backend_controller()
                self._update_cached_state()
                logger.info("Operation cancelled.")
                return

        self._sync_experiment_system_to_hardware(
            boxes=boxes,
            parallel=parallel,
            target_labels=target_labels,
        )
        if not supports_cache_sync:
            self._update_cached_state()
            return
        fetched_backend_settings = self._fetch_backend_settings_from_hardware(
            box_ids=box_ids,
            parallel=parallel,
        )
        previous_backend_settings = BackendSettings(self.backend_settings)
        previous_box_cache = self._get_box_config_cache_snapshot()
        merged_backend_settings = self._merge_backend_settings(
            base_settings=previous_backend_settings,
            patch_settings=fetched_backend_settings,
        )
        try:
            # Rebuild backend cache from the merged full snapshot because push
            # re-fetches only the selected boxes.
            self._set_backend_settings(merged_backend_settings)
            self._sync_backend_settings_to_backend_controller(
                backend_settings=merged_backend_settings
            )
        except Exception:
            self._set_backend_settings(previous_backend_settings)
            self._replace_box_config_cache(previous_box_cache)
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

    def _supports_box_settings_cache_sync(self) -> bool:
        """Return whether backend supports hardware snapshot synchronization APIs."""
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            return False
        return bool(
            getattr(system_synchronizer, "supports_backend_settings_sync", False)
        )

    def _supports_mutable_backend_settings_cache(self) -> bool:
        """Return whether backend supports mutable backend-settings cache writes."""
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            return False
        return bool(
            getattr(
                system_synchronizer,
                "supports_mutable_backend_settings_cache",
                False,
            )
        )

    def _get_box_config_cache_snapshot(self) -> dict[str, dict]:
        """Return a snapshot of backend box-config cache when supported."""
        if not self._supports_mutable_backend_settings_cache():
            return {}
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            return {}
        return system_synchronizer.get_box_config_cache_snapshot()

    def _replace_box_config_cache(self, box_configs: Mapping[str, dict]) -> None:
        """Replace backend box-config cache when supported."""
        if not self._supports_mutable_backend_settings_cache():
            return
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            return
        system_synchronizer.replace_box_config_cache(dict(box_configs))

    def _sync_experiment_system_to_hardware(
        self,
        *,
        boxes: Sequence[Box],
        parallel: bool | None = None,
        target_labels: Sequence[str] | None = None,
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
        target_labels : Sequence[str] | None, optional
            Logical target labels to apply for backends that support
            target-scoped hardware configuration.
        """
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            logger.debug(
                "Skipping hardware sync because active backend has no system synchronizer.",
            )
            return
        system_synchronizer.sync_experiment_system_to_hardware(
            experiment_system=self.experiment_system,
            boxes=boxes,
            parallel=parallel,
            target_labels=target_labels,
        )

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
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            return BackendSettings()
        fetched = system_synchronizer.fetch_backend_settings_from_hardware(
            experiment_system=self.experiment_system,
            box_ids=box_ids,
            parallel=parallel,
        )
        return BackendSettings(fetched)

    def _sync_backend_settings_to_backend_controller(
        self,
        *,
        backend_settings: BackendSettings | None = None,
    ) -> None:
        """
        Apply backend-settings snapshots to backend controller cache.

        Parameters
        ----------
        backend_settings : BackendSettings | None, optional
            Settings to apply. If `None`, uses `self._backend_settings`.
        """
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            return
        if backend_settings is None:
            backend_settings = self._backend_settings
        system_synchronizer.sync_backend_settings_to_backend_controller(
            backend_settings=backend_settings,
        )

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
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            return
        if backend_settings is None:
            backend_settings = self._backend_settings
        system_synchronizer.sync_backend_settings_to_experiment_system(
            experiment_system=self.experiment_system,
            backend_settings=backend_settings,
        )

    def _sync_experiment_system_to_backend_controller(self) -> None:
        """Rebuild backend controller state via backend-specific synchronizer."""
        system_synchronizer = self._resolve_system_synchronizer()
        if system_synchronizer is None:
            logger.info(
                "Skipping backend controller model sync because this backend has no system synchronizer."
            )
            self._update_cached_state()
            return
        system_synchronizer.sync_experiment_system_to_backend_controller(
            self.experiment_system
        )
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
        modify_target_frequencies = getattr(
            self.backend_controller, "modify_target_frequencies", None
        )
        if callable(modify_target_frequencies):
            modify_target_frequencies(target_frequencies)
        try:
            yield
        finally:
            self.experiment_system.modify_target_frequencies(original_frequencies)
            if callable(modify_target_frequencies):
                modify_target_frequencies(original_frequencies)

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
        if self.backend_kind == BACKEND_KIND_QUEL3:
            # QuEL-3 does not expose the QuEL-1 mutable backend-settings cache
            # path. Keep this context manager as a compatibility no-op and rely
            # on direct per-target frequency overrides inside the caller.
            yield
            return
        if not self._supports_mutable_backend_settings_cache():
            raise NotImplementedError(
                "Active backend does not support backend-settings cache operations."
            )
        target = self.experiment_system.get_target(label)
        channel = target.channel
        port = channel.port
        box_cache = self._get_box_config_cache_snapshot()
        config_port = getattr(self.backend_controller, "config_port", None)
        config_channel = getattr(self.backend_controller, "config_channel", None)
        config_runit = getattr(self.backend_controller, "config_runit", None)
        initialize_awg_and_capunits = getattr(
            self.backend_controller, "initialize_awg_and_capunits", None
        )
        if (
            not callable(config_port)
            or not callable(config_channel)
            or not callable(config_runit)
            or not callable(initialize_awg_and_capunits)
        ):
            raise NotImplementedError(
                "Active backend does not support backend-settings cache operations."
            )

        original_lo_freq = port.lo_freq
        original_cnco_freq = port.cnco_freq
        original_fnco_freq = channel.fnco_freq
        original_box_cache = deepcopy(box_cache)

        config_port(
            box_name=port.box_id,
            port=port.number,
            lo_freq_hz=lo_freq,
            cnco_freq_hz=cnco_freq,
        )
        config_channel(
            box_name=port.box_id,
            port=port.number,
            channel=channel.number,
            fnco_freq_hz=fnco_freq,
        )
        port_cache = box_cache[port.box_id]["ports"][port.number]
        port_cache["lo_freq"] = lo_freq
        port_cache["cnco_freq"] = cnco_freq
        port_cache["channels"][channel.number]["fnco_freq"] = fnco_freq
        update_cache = getattr(self.backend_controller, "update_box_config_cache", None)
        if callable(update_cache):
            update_cache({port.box_id: box_cache[port.box_id]})
        initialized_box_ids = [port.box_id]

        if target.is_read:
            cap_channel = self.experiment_system.get_cap_target(label).channel
            cap_port = cap_channel.port
            config_port(
                box_name=cap_port.box_id,
                port=cap_port.number,
                lo_freq_hz=lo_freq,
                cnco_freq_hz=cnco_freq,
            )
            config_runit(
                box_name=cap_port.box_id,
                port=cap_port.number,
                runit=cap_channel.number,
                fnco_freq_hz=fnco_freq,
            )
            cap_port_cache = box_cache[cap_port.box_id]["ports"][cap_port.number]
            cap_port_cache["lo_freq"] = lo_freq
            cap_port_cache["cnco_freq"] = cnco_freq
            cap_port_cache["runits"][cap_channel.number]["fnco_freq"] = fnco_freq
            if callable(update_cache):
                update_cache({cap_port.box_id: box_cache[cap_port.box_id]})
            if cap_port.box_id not in initialized_box_ids:
                initialized_box_ids.append(cap_port.box_id)

        initialize_awg_and_capunits(initialized_box_ids)

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
            config_port(
                box_name=port.box_id,
                port=port.number,
                lo_freq_hz=original_lo_freq,
                cnco_freq_hz=original_cnco_freq,
            )
            config_channel(
                box_name=port.box_id,
                port=port.number,
                channel=channel.number,
                fnco_freq_hz=original_fnco_freq,
            )
            if target.is_read:
                config_port(
                    box_name=cap_port.box_id,
                    port=cap_port.number,
                    lo_freq_hz=original_lo_freq,
                    cnco_freq_hz=original_cnco_freq,
                )
                config_runit(
                    box_name=cap_port.box_id,
                    port=cap_port.number,
                    runit=cap_channel.number,
                    fnco_freq_hz=original_fnco_freq,
                )

            # restore the original box config
            self._replace_box_config_cache(original_box_cache)

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
