# ruff: noqa: SLF001

"""Connection and lifecycle manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from qubex.backend.parallel_box_executor import run_parallel_each, run_parallel_map
from qubex.backend.quel1.compat.box_adapter import adapt_quel1_box
from qubex.backend.quel1.quel1_backend_constants import DEFAULT_EXECUTION_MODE
from qubex.backend.quel1.quel1_runtime_context import (
    NOT_CONNECTED_ERROR_MESSAGE,
    Quel1RuntimeContext,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        BoxSettingProtocol as BoxSetting,
        Quel1BoxCommonProtocol as Quel1Box,
        Quel1ConfigOptionProtocol as Quel1ConfigOption,
        Quel1SystemProtocol as Quel1System,
    )

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


class Quel1ConnectionManager:
    """Handle connect/disconnect and box link-maintenance flows for QuEL-1."""

    def __init__(self, *, runtime_context: Quel1RuntimeContext) -> None:
        self._runtime_context = runtime_context

    @property
    def is_connected(self) -> bool:
        """Return whether runtime state currently has a connected system."""
        return self._runtime_context.is_connected

    @property
    def boxpool(self) -> BoxPool:
        """Return connected boxpool."""
        try:
            return self._runtime_context.boxpool
        except ValueError as exc:
            raise ValueError(NOT_CONNECTED_ERROR_MESSAGE) from exc

    @property
    def quel1system(self) -> Quel1System:
        """Return connected Quel1System."""
        try:
            return self._runtime_context.quel1system
        except ValueError as exc:
            raise ValueError(NOT_CONNECTED_ERROR_MESSAGE) from exc

    @property
    def cap_resource_map(self) -> dict[str, dict]:
        """Return capture resource map."""
        try:
            return self._runtime_context.cap_resource_map
        except ValueError as exc:
            raise ValueError(NOT_CONNECTED_ERROR_MESSAGE) from exc

    @property
    def gen_resource_map(self) -> dict[str, dict]:
        """Return generator resource map."""
        try:
            return self._runtime_context.gen_resource_map
        except ValueError as exc:
            raise ValueError(NOT_CONNECTED_ERROR_MESSAGE) from exc

    def set_connected_state(
        self,
        *,
        boxpool: BoxPool | None,
        quel1system: Quel1System | None,
        cap_resource_map: dict[str, dict] | None,
        gen_resource_map: dict[str, dict] | None,
    ) -> None:
        """Replace full connected runtime state."""
        self._runtime_context.set_connected_state(
            boxpool=boxpool,
            quel1system=quel1system,
            cap_resource_map=cap_resource_map,
            gen_resource_map=gen_resource_map,
        )

    def set_boxpool(self, boxpool: BoxPool | None) -> None:
        """Update only boxpool state."""
        self._runtime_context.set_boxpool(boxpool)

    def set_quel1system(self, quel1system: Quel1System | None) -> None:
        """Update only Quel1System state."""
        self._runtime_context.set_quel1system(quel1system)

    def set_cap_resource_map(self, resource_map: dict[str, dict] | None) -> None:
        """Update only capture resource map state."""
        self._runtime_context.set_cap_resource_map(resource_map)

    def set_gen_resource_map(self, resource_map: dict[str, dict] | None) -> None:
        """Update only generator resource map state."""
        self._runtime_context.set_gen_resource_map(resource_map)

    def clear_connected_state(self) -> None:
        """Clear connected runtime state."""
        self._runtime_context.clear_connected_state()

    def clear_cache(self) -> None:
        """Clear cached box configuration data."""
        if not self.is_connected:
            return
        boxpool = self.boxpool
        boxpool._box_config_cache.clear()
        quel1system = self.quel1system
        quel1system.config_cache.clear()
        quel1system.config_fetched_at = None

    def replace_box_config_cache(self, box_configs: dict[str, dict]) -> None:
        """Replace the box-config cache with the provided snapshot."""
        boxpool = self.boxpool
        boxpool._box_config_cache = deepcopy(box_configs)
        quel1system = self.quel1system
        quel1system.config_cache.clear()
        for box_name, box_config in box_configs.items():
            quel1system.config_cache[box_name] = deepcopy(box_config)
        quel1system.config_fetched_at = (
            datetime.now() if quel1system.config_cache else None
        )

    def update_box_config_cache(self, box_configs: dict[str, dict]) -> None:
        """Update cached box configurations by box name."""
        boxpool = self.boxpool
        for box_name, box_config in box_configs.items():
            boxpool._box_config_cache[box_name] = deepcopy(box_config)
        quel1system = self.quel1system
        for box_name, box_config in box_configs.items():
            quel1system.config_cache[box_name] = deepcopy(box_config)
        if quel1system.config_cache:
            quel1system.config_fetched_at = datetime.now()

    def connect(
        self,
        *,
        box_names: str | list[str] | None,
        parallel: bool | None,
    ) -> None:
        """Resolve and create connected runtime state for requested boxes."""
        if parallel is None:
            parallel = _DEFAULT_PARALLEL_MODE
        if self.is_connected:
            logger.info("Already connected. Skipping backend reconnect.")
            return

        resolved_box_names = self._resolve_box_names(box_names)
        boxpool = self.create_boxpool(resolved_box_names, parallel=parallel)
        self.set_connected_state(
            boxpool=boxpool,
            quel1system=None,
            cap_resource_map=None,
            gen_resource_map=None,
        )
        try:
            quel1system = self._create_quel1system_from_boxpool(resolved_box_names)
            self.set_quel1system(quel1system)
            cap_resource_map = self._create_resource_map("cap")
            gen_resource_map = self._create_resource_map("gen")
        except Exception:
            self.clear_connected_state()
            raise
        self.set_cap_resource_map(cap_resource_map)
        self.set_gen_resource_map(gen_resource_map)

    def disconnect(self) -> None:
        """Disconnect all currently held resources."""
        if not self.is_connected:
            return
        for resource in self._collect_held_resources():
            self._disconnect_resource_safely(resource)
        self.clear_connected_state()

    def initialize_awg_and_capunits(
        self,
        *,
        box_names: str | Collection[str],
        parallel: bool | None,
    ) -> None:
        """Initialize AWG and capture units for selected boxes."""
        self._require_connected()
        if isinstance(box_names, str):
            box_name_list = [box_names]
        else:
            box_name_list = list(box_names)
        unique_box_names = list(dict.fromkeys(box_name_list))
        if parallel is None:
            parallel = _DEFAULT_PARALLEL_MODE
        if not parallel:
            for box_name in unique_box_names:
                self._initialize_box_awg_and_capunits(box_name)
            return

        run_parallel_each(
            unique_box_names,
            self._initialize_box_awg_and_capunits,
            max_workers=_MAX_BOX_PARALLEL_WORKERS,
        )

    def linkup(
        self,
        *,
        box_name: str,
        noise_threshold: int | None,
        **kwargs: object,
    ) -> Quel1Box:
        """Linkup one box and return the connected box."""
        self._runtime_context.validate_box_availability(box_name)
        box = self._get_existing_or_create_box(box_name=box_name, reconnect=False)
        if noise_threshold is None:
            noise_threshold = _RELAXED_NOISE_THRESHOLD
        if not all(box.link_status().values()):
            raise ConnectionError(f"Box {box_name} has down links before linkup.")
        box.reconnect(background_noise_threshold=noise_threshold, **kwargs)
        status = box.link_status()
        if not all(status.values()):
            logger.warning(f"Failed to linkup box {box_name}. Status: {status}")
        return box

    def linkup_boxes(
        self,
        *,
        box_list: list[str],
        noise_threshold: int | None,
        parallel: bool | None,
    ) -> dict[str, Quel1Box]:
        """Linkup all requested boxes."""
        unique_box_list = list(dict.fromkeys(box_list))
        if parallel is None:
            parallel = _DEFAULT_PARALLEL_MODE
        if not parallel:
            boxes: dict[str, Quel1Box] = {}
            for box_name in unique_box_list:
                linked_box = self._safe_linkup_box(
                    box_name=box_name,
                    noise_threshold=noise_threshold,
                )
                if linked_box is not None:
                    boxes[box_name] = linked_box
            return boxes

        def _linkup_one(box_name: str) -> Quel1Box:
            return self.linkup(
                box_name=box_name,
                noise_threshold=noise_threshold,
            )

        results = run_parallel_map(
            unique_box_list,
            _linkup_one,
            key=self._box_name_key,
            max_workers=_MAX_BOX_PARALLEL_WORKERS,
            on_error=self._fallback_linkup_box_result,
        )
        boxes: dict[str, Quel1Box] = {}
        for box_name, linked_box in results.items():
            if linked_box is None:
                continue
            boxes[box_name] = linked_box
            logger.info(f"{box_name:5} : Linked up")
        return boxes

    def relinkup(
        self,
        *,
        box_name: str,
        noise_threshold: int | None,
    ) -> None:
        """Relink one box."""
        self._runtime_context.validate_box_availability(box_name)
        if noise_threshold is None:
            noise_threshold = _RELAXED_NOISE_THRESHOLD
        box = self._get_existing_or_create_box(box_name=box_name, reconnect=False)
        config_options = self._resolve_config_options(
            box_name=box_name, boxtype=box.boxtype
        )
        box.relinkup(
            use_204b=False,
            background_noise_threshold=noise_threshold,
            config_options=config_options,
        )
        box.reconnect(background_noise_threshold=noise_threshold)

    def relinkup_boxes(
        self,
        *,
        box_list: list[str],
        noise_threshold: int | None,
        parallel: bool | None,
    ) -> None:
        """Relink all requested boxes."""
        unique_box_list = list(dict.fromkeys(box_list))
        if parallel is None:
            parallel = _DEFAULT_PARALLEL_MODE
        if not parallel:
            for box_name in unique_box_list:
                self.relinkup(
                    box_name=box_name,
                    noise_threshold=noise_threshold,
                )
            return

        def _relinkup_one(box_name: str) -> None:
            self.relinkup(
                box_name=box_name,
                noise_threshold=noise_threshold,
            )

        run_parallel_each(
            unique_box_list,
            _relinkup_one,
            max_workers=_MAX_BOX_PARALLEL_WORKERS,
            on_error=self._log_relinkup_error,
        )

    def create_boxpool(
        self,
        box_names: list[str],
        *,
        parallel: bool | None = None,
    ) -> BoxPool:
        """Create a box pool and reconnect requested boxes."""
        if parallel is None:
            parallel = _DEFAULT_PARALLEL_MODE
        return self._create_boxpool(box_names, parallel=parallel)

    def create_box(
        self,
        *,
        box_name: str,
        reconnect: bool = True,
    ) -> Quel1Box:
        """Create one box from system configuration."""
        self._runtime_context.validate_box_availability(box_name)
        db = self._runtime_context.qubecalib.system_config_database
        return db.create_box(box_name, reconnect=reconnect)

    def get_existing_or_create_box(
        self,
        *,
        box_name: str,
        reconnect: bool,
    ) -> Quel1Box:
        """Return existing pooled box or create one from system configuration."""
        return self._get_existing_or_create_box(
            box_name=box_name,
            reconnect=reconnect,
        )

    def _resolve_box_names(self, box_names: str | list[str] | None) -> list[str]:
        """Resolve target box names from method input."""
        if box_names is None:
            return self._runtime_context.available_boxes
        if isinstance(box_names, str):
            return [box_names]
        return list(box_names)

    def _create_boxpool(self, box_names: list[str], *, parallel: bool) -> BoxPool:
        """Create a box pool and reconnect boxes."""
        qubecalib = self._runtime_context.qubecalib
        db = qubecalib.system_config_database
        driver = self._runtime_context.driver
        boxpool = driver.BoxPool()
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

            def _create_box_without_reconnect(box_name: str) -> Quel1Box:
                return db.create_box(box_name, reconnect=False)

            created_boxes = run_parallel_map(
                box_names,
                _create_box_without_reconnect,
                key=self._box_name_key,
                max_workers=min(_MAX_BOX_PARALLEL_WORKERS, len(box_names)),
            )
            for box_name in box_names:
                setting = settings_by_name[box_name]
                box = created_boxes[box_name]
                sequencer = driver.SequencerClient(str(setting.ipaddr_sss))
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
            with ThreadPoolExecutor(
                max_workers=max(1, len(boxes_to_reconnect))
            ) as executor:
                futures = [executor.submit(box.reconnect) for box in boxes_to_reconnect]
                for future in futures:
                    future.result()
        else:
            for box in boxes_to_reconnect:
                box.reconnect()
        return boxpool

    def _create_quel1system_from_boxpool(self, box_names: list[str]) -> Quel1System:
        """Build a Quel1System from already-connected boxpool entries."""
        boxpool = self.boxpool
        qubecalib = self._runtime_context.qubecalib
        db = qubecalib.system_config_database
        clockmaster_setting = db._clockmaster_setting
        if clockmaster_setting is None:
            raise ValueError("clock master is not found")
        driver = self._runtime_context.driver
        boxes = [
            driver.NamedBox(name=box_name, box=boxpool._boxes[box_name][0])
            for box_name in box_names
        ]
        return driver.Quel1System.create(
            clockmaster=driver.QuBEMasterClient(str(clockmaster_setting.ipaddr)),
            boxes=boxes,
            update_copnfig_cache=False,
        )

    def _create_resource_map(
        self,
        kind: Literal["cap", "gen"],
    ) -> dict[str, dict]:
        """Create capture or generator resource map from configuration."""
        boxpool = self.boxpool
        db = self._runtime_context.qubecalib.system_config_database
        target_settings = db._target_settings
        box_settings = db._box_settings
        port_settings = db._port_settings
        pooled_boxes = boxpool._boxes
        result: dict[str, dict] = {}
        for target in target_settings:
            channels = db.get_channels_by_target(target)
            bpc_list = [db.get_channel(channel) for channel in channels]
            for box_name, port_name, channel_number in bpc_list:
                if box_name not in pooled_boxes:
                    continue
                box = pooled_boxes[box_name][0]
                port_setting = port_settings[port_name]
                is_capture = (
                    kind == "cap" and port_setting.port in box.get_input_ports()
                )
                is_generator = (
                    kind == "gen" and port_setting.port in box.get_output_ports()
                )
                if not (is_capture or is_generator):
                    continue
                result[target] = {
                    "box": box_settings[box_name],
                    "port": port_settings[port_name],
                    "channel_number": channel_number,
                    "target": target_settings[target],
                }
        return result

    def _initialize_box_awg_and_capunits(self, box_name: str) -> None:
        """Initialize AWG and capture units for one box."""
        self._runtime_context.validate_box_availability(box_name)
        boxpool = self.boxpool
        if box_name not in boxpool._boxes:
            raise ValueError(
                f"Box {box_name} is not connected. Call connect() method first."
            )
        box = adapt_quel1_box(boxpool._boxes[box_name][0])
        box.initialize_all_awgunits()
        box.initialize_all_capunits()

    def _get_existing_or_create_box(
        self,
        *,
        box_name: str,
        reconnect: bool,
    ) -> Quel1Box:
        """Return existing pooled box or create one from system configuration."""
        boxpool = self.boxpool
        if box_name in boxpool._boxes:
            return boxpool._boxes[box_name][0]
        return self.create_box(box_name=box_name, reconnect=reconnect)

    def _resolve_config_options(
        self,
        *,
        box_name: str,
        boxtype: str,
    ) -> list[Quel1ConfigOption] | None:
        """Resolve config options for relinkup from optional per-box labels."""
        option_labels = list(self._runtime_context.box_options.get(box_name, ()))
        if boxtype == "quel1se-riken8":
            awg_option = _resolve_quel1se_r8_awg_option(option_labels)
            if awg_option not in option_labels:
                option_labels.insert(0, awg_option)
        if not option_labels:
            return None

        config_options: list[Quel1ConfigOption] = []
        option_map = self._runtime_context.driver.Quel1ConfigOption._value2member_map_
        for option_label in option_labels:
            option = option_map.get(option_label)
            if option is None:
                raise ValueError(
                    f"Unknown Quel1 config option `{option_label}` for box `{box_name}`."
                )
            config_options.append(option)
        return config_options

    def _collect_held_resources(self) -> list[object]:
        """Collect clockmaster and box objects currently held by runtime state."""
        resources: list[object] = []
        seen: set[int] = set()
        if not self.is_connected:
            return resources
        system = self.quel1system
        self._append_resource_if_new(
            resources, seen, getattr(system, "_clockmaster", None)
        )
        boxes = getattr(system, "boxes", None)
        if isinstance(boxes, Mapping):
            for box in boxes.values():
                self._append_resource_if_new(resources, seen, box)

        boxpool = self.boxpool
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

    def _append_resource_if_new(
        self,
        resources: list[object],
        seen: set[int],
        resource: object | None,
    ) -> None:
        """Append one resource object once by identity."""
        if resource is None:
            return
        resource_id = id(resource)
        if resource_id in seen:
            return
        seen.add(resource_id)
        resources.append(resource)

    def _disconnect_resource_safely(self, resource: object) -> None:
        """Disconnect one resource and log errors without aborting cleanup."""
        try:
            self._disconnect_resource(resource)
        except Exception:
            logger.exception("Failed to disconnect backend resource: %r", resource)

    def _disconnect_resource(self, resource: object) -> None:
        """Disconnect one resource object via close/terminate if available."""
        for method_name in ("close", "terminate"):
            method = getattr(resource, method_name, None)
            if not callable(method):
                continue
            method()
            return

    def _safe_linkup_box(
        self,
        *,
        box_name: str,
        noise_threshold: int | None,
    ) -> Quel1Box | None:
        """Link up one box and log failures without raising."""
        try:
            linked_box = self.linkup(
                box_name=box_name,
                noise_threshold=noise_threshold,
            )
            logger.info(f"{box_name:5} : Linked up")
        except Exception as exc:
            logger.exception(f"{box_name:5} : Error during linkup", exc_info=exc)
            return None
        else:
            return linked_box

    @staticmethod
    def _box_name_key(box_name: str) -> str:
        """Return box-name key for parallel map output ordering."""
        return box_name

    @staticmethod
    def _fallback_linkup_box_result(
        box_name: str, exc: BaseException
    ) -> Quel1Box | None:
        """Log a linkup error and return no box for the failed item."""
        logger.exception(f"{box_name:5} : Error during linkup", exc_info=exc)
        return None

    @staticmethod
    def _log_relinkup_error(box_name: str, exc: BaseException) -> None:
        """Log a relinkup error for one box."""
        logger.exception(f"{box_name:5} : Error during relinkup", exc_info=exc)

    def _require_connected(self) -> None:
        """Raise a consistent error when runtime is not connected."""
        if not self.is_connected:
            raise ValueError(NOT_CONNECTED_ERROR_MESSAGE)
