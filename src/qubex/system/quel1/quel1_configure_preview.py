# ruff: noqa: SLF001

"""QuEL-1 configure preview backed by temporary `Quel1Box` mocks."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeAlias, cast

from qubex.backend.backend_controller import BACKEND_KIND_QUEL1
from qubex.system.configure_preview import ConfigurePreview, ConfigureStateChange
from qubex.system.control_system import BoxType
from qubex.typing import ConfigurationMode

if TYPE_CHECKING:
    from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController

_PortNumber: TypeAlias = int | tuple[int, int]
_Snapshot: TypeAlias = Mapping[object, object]
_MutableSnapshot: TypeAlias = MutableMapping[object, object]

_GENERATOR_FIELDS: Final = (
    "lo_freq",
    "cnco_freq",
    "vatt",
    "sideband",
    "fullscale_current",
    "rfswitch",
)
_CAPTURE_FIELDS: Final = ("lo_freq", "cnco_freq", "rfswitch")
_FREQUENCY_FIELDS: Final = frozenset({"lo_freq", "cnco_freq", "fnco_freq"})

# These groups mirror quel_ic_config's port-to-line, LO, and RF-switch maps.
_SHARED_RESOURCE_PORT_GROUPS: Final[
    dict[str, dict[str, tuple[tuple[int, ...], ...]]]
] = {
    BoxType.QUEL1SE_A.value: {
        "lo_freq": ((0, 1), (3, 5), (7, 8), (10, 12)),
        "rfswitch": ((0, 1), (7, 8)),
    },
    BoxType.QUEL1SE_B.value: {
        "lo_freq": ((3, 5), (10, 12)),
    },
    BoxType.QUEL1SE_R8.value: {
        "lo_freq": ((0, 1), (4, 10)),
        "rfswitch": ((0, 1),),
    },
    BoxType.QUEL1_A.value: {
        "lo_freq": ((0, 1), (3, 5), (7, 8), (10, 12)),
        "rfswitch": ((0, 1), (7, 8)),
    },
    BoxType.QUEL1_B.value: {
        "lo_freq": ((2, 5), (9, 12)),
    },
    BoxType.QUBE_RIKEN_A.value: {
        "lo_freq": ((0, 1), (2, 4), (12, 13), (9, 11)),
        "rfswitch": ((0, 1), (12, 13)),
    },
    BoxType.QUBE_RIKEN_B.value: {
        "lo_freq": ((2, 4), (9, 11)),
    },
    BoxType.QUBE_OU_A.value: {
        "lo_freq": ((0, 1), (12, 13)),
    },
}


@dataclass(frozen=True)
class _WaveSubsystemMock:
    """Expose the address needed by driver-side box registration."""

    ipaddr_sss: str


class Quel1BoxMock:
    """Emulate the configuration subset of `Quel1Box` on a copied dump."""

    _instances_by_wss_ip: ClassVar[dict[str, Quel1BoxMock]] = {}

    def __init__(
        self,
        *,
        box_name: str,
        boxtype: str,
        ipaddr_sss: str,
        snapshot: Mapping[object, object],
    ) -> None:
        self.box_name = box_name
        self.boxtype = boxtype
        self.wss = _WaveSubsystemMock(ipaddr_sss=ipaddr_sss)
        self._initial = deepcopy(snapshot)
        self._configured = deepcopy(snapshot)

    @classmethod
    def create(cls, *, ipaddr_wss: object, **_: object) -> Quel1BoxMock:
        """Return the mock registered for one WSS address."""
        return cls._instances_by_wss_ip[str(ipaddr_wss)]

    @property
    def is_complete(self) -> bool:
        """Return whether the initial dump contains a ports mapping."""
        return isinstance(self._initial.get("ports"), Mapping)

    def reconnect(self, *_: Any, **__: Any) -> dict[int, bool]:
        """Emulate reconnect without touching hardware."""
        return {}

    def relinkup(self, *_: Any, **__: Any) -> dict[int, bool]:
        """Emulate relinkup without touching hardware."""
        return {}

    def link_status(self) -> dict[int, bool]:
        """Return an empty successful-enough link-status payload."""
        return {}

    def get_input_ports(self) -> list[Any]:
        """Return ports marked as inputs in the copied dump."""
        return self._ports_with_direction("in")

    def get_output_ports(self) -> list[Any]:
        """Return ports marked as outputs in the copied dump."""
        return self._ports_with_direction("out")

    def dump_box(self) -> dict[str, Any]:
        """Return the current simulated box state."""
        return cast("dict[str, Any]", deepcopy(dict(self._configured)))

    def dump_port(self, port: _PortNumber) -> dict[str, Any]:
        """Return one simulated port state."""
        port_config = _as_mapping(_mapping_item(self._ports(), port))
        return cast("dict[str, Any]", deepcopy(dict(port_config)))

    def config_port(
        self,
        *,
        port: _PortNumber,
        lo_freq: float | None = None,
        cnco_freq: float | None = None,
        vatt: int | None = None,
        sideband: str | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
    ) -> None:
        """Apply a port configuration to the simulated state."""
        values = (
            ("lo_freq", lo_freq),
            ("cnco_freq", cnco_freq),
            ("vatt", vatt),
            ("sideband", sideband),
            ("fullscale_current", fullscale_current),
            ("rfswitch", rfswitch),
        )
        for field, value in values:
            if value is not None:
                self._write_port_resource(
                    port=port,
                    field=field,
                    value=_normalize_value(value),
                )

    def config_channel(
        self,
        *,
        port: _PortNumber,
        channel: int,
        fnco_freq: float | None = None,
        awg_param: Any | None = None,
    ) -> None:
        """Apply a generator-channel configuration to simulated state."""
        del awg_param
        self._write_fnco(
            port=port,
            collection="channels",
            child=channel,
            fnco_freq=fnco_freq,
        )

    def config_runit(
        self,
        *,
        port: _PortNumber,
        runit: int,
        fnco_freq: float | None = None,
    ) -> None:
        """Apply a capture-unit configuration to simulated state."""
        self._write_fnco(
            port=port,
            collection="runits",
            child=runit,
            fnco_freq=fnco_freq,
        )

    def preview_entries(self) -> list[ConfigureStateChange]:
        """Return comparisons between the initial and simulated box states."""
        initial_ports = _as_mapping(self._initial.get("ports"))
        configured_ports = _as_mapping(self._configured.get("ports"))
        entries: list[ConfigureStateChange] = []
        for port_number, configured_value in configured_ports.items():
            configured_port = _as_mapping(configured_value)
            initial_port = _as_mapping(_mapping_item(initial_ports, port_number))
            direction = configured_port.get("direction")
            if direction == "out":
                fields = _GENERATOR_FIELDS
                child_collection = "channels"
                child_label = "channel"
            elif direction == "in":
                fields = _CAPTURE_FIELDS
                child_collection = "runits"
                child_label = "runit"
            else:
                continue
            entries.extend(
                _compare_fields(
                    box_id=self.box_name,
                    component=f"port {port_number}",
                    fields=fields,
                    initial=initial_port,
                    configured=configured_port,
                )
            )
            initial_children = _as_mapping(initial_port.get(child_collection))
            configured_children = _as_mapping(configured_port.get(child_collection))
            for child_number, configured_child in configured_children.items():
                entries.extend(
                    _compare_fields(
                        box_id=self.box_name,
                        component=(f"port {port_number} {child_label} {child_number}"),
                        fields=("fnco_freq",),
                        initial=_as_mapping(
                            _mapping_item(initial_children, child_number)
                        ),
                        configured=_as_mapping(configured_child),
                    )
                )
        return entries

    def _ports(self) -> _MutableSnapshot:
        """Return the mutable simulated ports mapping."""
        ports = self._configured.get("ports")
        return ports if isinstance(ports, MutableMapping) else {}

    def _ports_with_direction(self, direction: str) -> list[Any]:
        """Return normalized port keys having one direction."""
        return [
            _normalize_port_number(port_number)
            for port_number, config in self._ports().items()
            if _as_mapping(config).get("direction") == direction
        ]

    def _write_port_resource(
        self,
        *,
        port: _PortNumber,
        field: str,
        value: object,
    ) -> None:
        """Apply one value to every port sharing its physical resource."""
        # Shared LOs are approximated as one logical frequency; the preview does
        # not model independent per-output divider ratios of the physical LMX.
        physical_value = _encode_resource_value(field=field, value=value)
        for affected_port in _resource_ports(
            boxtype=self.boxtype,
            field=field,
            port=port,
        ):
            port_config = _mapping_item(self._ports(), affected_port)
            if not isinstance(port_config, MutableMapping):
                continue
            port_config[field] = _decode_resource_value(
                field=field,
                direction=port_config.get("direction"),
                value=physical_value,
            )

    def _write_fnco(
        self,
        *,
        port: _PortNumber,
        collection: str,
        child: int,
        fnco_freq: float | None,
    ) -> None:
        """Apply one FNCO value when the configure call specifies it."""
        if fnco_freq is None:
            return
        port_config = _mapping_item(self._ports(), port)
        if not isinstance(port_config, MutableMapping):
            return
        children = _ensure_mapping_item(port_config, collection)
        child_config = _ensure_mapping_item(children, child)
        child_config["fnco_freq"] = _normalize_value(fnco_freq)


class Quel1BoxPreviewContext:
    """Temporarily route controller configuration calls to `Quel1BoxMock`."""

    _lock: ClassVar[RLock] = RLock()

    def __init__(
        self,
        *,
        backend_controller: Quel1BackendController,
        backend_settings: Mapping[str, dict],
        box_types: Mapping[str, BoxType],
    ) -> None:
        """Initialize mock boxes from fetched hardware snapshots."""
        self._backend_controller = backend_controller
        self._database = backend_controller.qubecalib.system_config_database
        self._boxes = self._create_mock_boxes(
            backend_settings=backend_settings,
            box_types=box_types,
        )
        self._database_globals = self._resolve_database_globals()
        self._patched_globals: dict[str, object] = {}
        self._pooled_boxes: MutableMapping[str, tuple[Any, ...]] | None = None
        self._original_pool_entries: dict[str, tuple[Any, ...]] = {}
        self._previous_mock_instances: dict[str, Quel1BoxMock] = {}

    def __enter__(self) -> Quel1BoxPreviewContext:
        """Install the mock class and replace already-pooled box instances."""
        self._lock.acquire()
        try:
            self._previous_mock_instances = Quel1BoxMock._instances_by_wss_ip
            Quel1BoxMock._instances_by_wss_ip = dict(self._boxes_by_wss_ip())
            self._patch_database_globals()
            self._replace_pooled_boxes()
        except BaseException:
            try:
                self._restore()
            finally:
                self._lock.release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore real box classes and instances even after configuration errors."""
        del exc_type, exc_value, traceback
        try:
            self._restore()
        finally:
            self._lock.release()

    def build_preview(
        self,
        *,
        box_ids: Sequence[str],
        mode: ConfigurationMode | None,
    ) -> ConfigurePreview:
        """Build a preview from changes recorded inside the mock boxes."""
        entries: list[ConfigureStateChange] = []
        missing_box_ids: list[str] = []
        for box_id in box_ids:
            box = self._boxes.get(box_id)
            if box is None or not box.is_complete:
                missing_box_ids.append(box_id)
                continue
            entries.extend(box.preview_entries())
        return ConfigurePreview(
            backend_kind=BACKEND_KIND_QUEL1,
            box_ids=tuple(box_ids),
            mode=mode,
            entries=tuple(entries),
            missing_box_ids=tuple(missing_box_ids),
        )

    def _create_mock_boxes(
        self,
        *,
        backend_settings: Mapping[str, dict],
        box_types: Mapping[str, BoxType],
    ) -> dict[str, Quel1BoxMock]:
        """Create one stateful mock for each successfully dumped box."""
        box_settings = self._database._box_settings
        boxes: dict[str, Quel1BoxMock] = {}
        for box_name, snapshot in backend_settings.items():
            setting = box_settings.get(box_name)
            box_type = box_types.get(box_name)
            if setting is None or box_type is None:
                continue
            boxes[box_name] = Quel1BoxMock(
                box_name=box_name,
                boxtype=box_type.value,
                ipaddr_sss=str(setting.ipaddr_sss),
                snapshot=snapshot,
            )
        return boxes

    def _boxes_by_wss_ip(self) -> list[tuple[str, Quel1BoxMock]]:
        """Return mock boxes keyed by their configured WSS addresses."""
        box_settings = self._database._box_settings
        return [
            (str(box_settings[box_name].ipaddr_wss), box)
            for box_name, box in self._boxes.items()
        ]

    def _resolve_database_globals(self) -> dict[str, object]:
        """Return globals used by the database's `create_box()` implementation."""
        create_box = type(self._database).create_box
        globals_mapping = getattr(create_box, "__globals__", None)
        if not isinstance(globals_mapping, dict):
            raise TypeError("Quel1Box preview requires a Python create_box() method.")
        return globals_mapping

    def _patch_database_globals(self) -> None:
        """Replace the real driver class referenced by `create_box()`."""
        real_box_class = self._backend_controller.driver.Quel1Box
        create_box = type(self._database).create_box
        referenced_names = set(create_box.__code__.co_names)
        class_names = [
            name
            for name in referenced_names
            if self._database_globals.get(name) is real_box_class
        ]
        if not class_names:
            raise RuntimeError("Could not locate Quel1Box in create_box() globals.")
        for name in class_names:
            self._patched_globals[name] = self._database_globals[name]
            self._database_globals[name] = Quel1BoxMock
        if (
            "register_box" in referenced_names
            and "register_box" in self._database_globals
        ):
            self._patched_globals["register_box"] = self._database_globals[
                "register_box"
            ]
            self._database_globals["register_box"] = _ignore_box_registration

    def _replace_pooled_boxes(self) -> None:
        """Replace connected boxpool entries that bypass database creation."""
        if not self._backend_controller.is_connected:
            return
        pooled_boxes = self._backend_controller.boxpool._boxes
        if not isinstance(pooled_boxes, MutableMapping):
            raise TypeError("Connected boxpool does not expose mutable box entries.")
        self._pooled_boxes = pooled_boxes
        for box_name, mock_box in self._boxes.items():
            entry = pooled_boxes.get(box_name)
            if not isinstance(entry, tuple) or not entry:
                continue
            self._original_pool_entries[box_name] = entry
            pooled_boxes[box_name] = (mock_box, *entry[1:])

    def _restore(self) -> None:
        """Restore every global and pooled object replaced by this context."""
        try:
            if self._pooled_boxes is not None:
                self._pooled_boxes.update(self._original_pool_entries)
        finally:
            self._original_pool_entries.clear()
            self._pooled_boxes = None
            try:
                for name, value in self._patched_globals.items():
                    self._database_globals[name] = value
            finally:
                self._patched_globals.clear()
                Quel1BoxMock._instances_by_wss_ip = self._previous_mock_instances


def _ignore_box_registration(box: object) -> None:
    """Avoid leaking preview mocks into driver-global clock registries."""
    del box


def _resource_ports(
    *,
    boxtype: str,
    field: str,
    port: _PortNumber,
) -> tuple[_PortNumber, ...]:
    """Return logical ports backed by the same physical resource."""
    for port_group in _SHARED_RESOURCE_PORT_GROUPS.get(boxtype, {}).get(field, ()):
        if port in port_group:
            return port_group
    return (port,)


def _encode_resource_value(*, field: str, value: object) -> object:
    """Encode one logical value into shared physical state."""
    if field != "rfswitch":
        return value
    if value in ("open", "pass"):
        return False
    if value in ("loop", "block"):
        return True
    return value


def _decode_resource_value(
    *,
    field: str,
    direction: object,
    value: object,
) -> object:
    """Decode shared physical state for one logical port direction."""
    if field != "rfswitch" or not isinstance(value, bool):
        return value
    if direction == "in":
        return "loop" if value else "open"
    return "block" if value else "pass"


def _compare_fields(
    *,
    box_id: str,
    component: str,
    fields: Sequence[str],
    initial: _Snapshot,
    configured: _Snapshot,
) -> list[ConfigureStateChange]:
    """Return configured field comparisons for one component."""
    entries: list[ConfigureStateChange] = []
    for field in fields:
        if field == "rfswitch" and field not in initial:
            continue
        configured_value = _normalize_value(configured.get(field))
        if configured_value is None:
            continue
        entries.append(
            ConfigureStateChange(
                box_id=box_id,
                component=component,
                field=field,
                before=_normalize_value(initial.get(field)),
                after=configured_value,
                unit="Hz" if field in _FREQUENCY_FIELDS else None,
                is_frequency=field in _FREQUENCY_FIELDS,
            )
        )
    return entries


def _as_mapping(value: object) -> _Snapshot:
    """Return a value as a mapping or an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _mapping_item(mapping: _Snapshot, key: object) -> object:
    """Read a snapshot item that may use numeric or stringified keys."""
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key))


def _ensure_mapping_item(
    mapping: _MutableSnapshot,
    key: object,
) -> _MutableSnapshot:
    """Return a mutable nested snapshot, creating one when absent."""
    item = _mapping_item(mapping, key)
    if isinstance(item, MutableMapping):
        return item
    child: dict[object, object] = {}
    mapping[key] = child
    return child


def _normalize_port_number(port: object) -> object:
    """Convert integer-like string keys back to logical port numbers."""
    if isinstance(port, str) and port.isdecimal():
        return int(port)
    return port


def _normalize_value(value: object) -> object:
    """Normalize integral floating-point dump values for stable comparison."""
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
