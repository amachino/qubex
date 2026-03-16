"""Configuration manager for QuEL-3 backend instrument deployment."""

from __future__ import annotations

import asyncio
import importlib
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from qubex.backend.quel3.infra.quelware_imports import (
    Quel3ClientMode,
    load_quelware_client_factory,
    validate_quelware_client_runtime,
)
from qubex.backend.quel3.interfaces.client import (
    InstrumentDefinitionProtocol,
    InstrumentInfoProtocol,
    QuelwareClientFactory,
    QuelwareClientProtocol,
    ResourceInfoProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.models import InstrumentDeployRequest
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

T = TypeVar("T")


class _FixedTimelineProfileFactory(Protocol):
    """Factory protocol for fixed-timeline profile entities."""

    def __call__(
        self,
        *,
        frequency_range_min: float,
        frequency_range_max: float,
    ) -> object:
        """Create one fixed-timeline profile."""
        ...


class _InstrumentDefinitionFactory(Protocol):
    """Factory protocol for instrument-definition entities."""

    def __call__(
        self,
        *,
        alias: str,
        mode: object,
        role: object,
        profile: object,
    ) -> InstrumentDefinitionProtocol:
        """Create one instrument definition."""
        ...


class _EnumNamespace(Protocol):
    """Protocol for enum-like namespaces loaded from quelware."""

    def __getattr__(self, name: str) -> object:
        """Return one enum-like member by attribute name."""
        ...


@dataclass(frozen=True)
class _CachedInstrumentDefinition:
    """Cached instrument definition restored from hardware snapshot."""

    alias: str
    role: str
    mode: str | None = None
    profile: object | None = None


@dataclass(frozen=True)
class _CachedFixedTimelineProfile:
    """Cached fixed-timeline profile restored from hardware snapshot."""

    frequency_range_min: float | None = None
    frequency_range_max: float | None = None


@dataclass(frozen=True)
class _CachedInstrumentInfo:
    """Cached instrument info restored from hardware snapshot."""

    id: str
    port_id: str
    definition: _CachedInstrumentDefinition


@dataclass(frozen=True)
class _PortDeployResult:
    """Deployment result for one port batch."""

    deployed: dict[str, tuple[InstrumentInfoProtocol, ...]]
    target_alias_map: dict[str, str]


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-configuration")
    return bridge.run(factory, timeout=timeout)


class Quel3ConfigurationManager:
    """Deploy QuEL-3 instruments through quelware configuration APIs."""

    def __init__(
        self,
        *,
        quelware_endpoint: str,
        quelware_port: int,
        client_mode: Quel3ClientMode = "server",
        standalone_unit_label: str | None = None,
    ) -> None:
        normalized_client_mode = validate_quelware_client_runtime(
            client_mode=client_mode,
            standalone_unit_label=standalone_unit_label,
        )
        self._quelware_endpoint = quelware_endpoint
        self._quelware_port = quelware_port
        self._client_mode: Quel3ClientMode = normalized_client_mode
        self._standalone_unit_label = standalone_unit_label
        self._last_deployed_instrument_infos: dict[
            str, tuple[InstrumentInfoProtocol, ...]
        ] = {}
        self._target_alias_map: dict[str, str] = {}

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint used for deployment."""
        return self._quelware_endpoint

    @property
    def quelware_port(self) -> int:
        """Return quelware port used for deployment."""
        return self._quelware_port

    @property
    def client_mode(self) -> Quel3ClientMode:
        """Return configured quelware client mode."""
        return self._client_mode

    @property
    def standalone_unit_label(self) -> str | None:
        """Return configured standalone unit label."""
        return self._standalone_unit_label

    @property
    def last_deployed_instrument_infos(
        self,
    ) -> dict[str, tuple[InstrumentInfoProtocol, ...]]:
        """Return last deployed instrument infos keyed by alias."""
        return dict(self._last_deployed_instrument_infos)

    @property
    def target_alias_map(self) -> dict[str, str]:
        """Return last deployed target-to-alias mapping."""
        return dict(self._target_alias_map)

    def deploy_instruments(
        self,
        *,
        requests: Sequence[InstrumentDeployRequest],
        parallel: bool = True,
    ) -> dict[str, tuple[InstrumentInfoProtocol, ...]]:
        """Deploy instruments for the provided QuEL-3 requests."""
        return _run_async(
            lambda: self._deploy_instruments(
                requests=tuple(requests),
                parallel=parallel,
            )
        )

    def refresh_instrument_cache(self) -> dict[str, tuple[InstrumentInfoProtocol, ...]]:
        """Refresh cached alias mappings from existing quelware instruments."""
        return _run_async(self._refresh_instrument_cache)

    def fetch_backend_settings_from_hardware(
        self,
        *,
        unit_labels_by_box_id: Mapping[str, str],
        parallel: bool | None = None,
    ) -> dict[str, dict]:
        """Fetch normalized QuEL-3 instrument snapshots keyed by box ID."""
        return _run_async(
            lambda: self._fetch_backend_settings_from_hardware(
                unit_labels_by_box_id=dict(unit_labels_by_box_id),
                parallel=True if parallel is None else parallel,
            )
        )

    def sync_backend_settings_to_cache(
        self,
        *,
        backend_settings: Mapping[str, dict],
    ) -> None:
        """Restore instrument alias caches from normalized backend settings."""
        deployed: dict[str, tuple[InstrumentInfoProtocol, ...]] = {}
        target_alias_map: dict[str, str] = {}

        for box_config in backend_settings.values():
            instruments = box_config.get("instruments")
            if not isinstance(instruments, dict):
                continue
            for alias, instrument_config in instruments.items():
                if not isinstance(alias, str) or not isinstance(
                    instrument_config, dict
                ):
                    continue
                resource_id = instrument_config.get("resource_id")
                port_id = instrument_config.get("port_id")
                role = instrument_config.get("role")
                if not isinstance(resource_id, str) or not isinstance(port_id, str):
                    continue
                definition = self._build_cached_definition(
                    alias=alias,
                    role=role,
                    definition_config=instrument_config.get("definition"),
                )
                deployed[alias] = (
                    _CachedInstrumentInfo(
                        id=resource_id,
                        port_id=port_id,
                        definition=definition,
                    ),
                )
                target_alias_map[alias] = alias

        self._last_deployed_instrument_infos = deployed
        self._target_alias_map = target_alias_map

    async def _deploy_instruments(
        self,
        *,
        requests: tuple[InstrumentDeployRequest, ...],
        parallel: bool = True,
    ) -> dict[str, tuple[InstrumentInfoProtocol, ...]]:
        """Deploy instruments through quelware session APIs."""
        if len(requests) == 0:
            self._last_deployed_instrument_infos = {}
            self._target_alias_map = {}
            return {}

        client_factory = self._load_quelware_client_factory()
        (
            fixed_timeline_profile_cls,
            instrument_definition_cls,
            instrument_mode,
            instrument_role,
        ) = self._load_instrument_entities()

        deployed: dict[str, tuple[InstrumentInfoProtocol, ...]] = {}
        target_alias_map: dict[str, str] = {}
        requests_by_port: dict[str, list[InstrumentDeployRequest]] = defaultdict(list)
        for request in requests:
            requests_by_port[request.port_id].append(request)
        port_request_batches = tuple(
            (port_id, tuple(port_requests))
            for port_id, port_requests in requests_by_port.items()
        )
        async with client_factory(
            self._quelware_endpoint,
            self._quelware_port,
        ) as client:
            session_resource_ids = [port_id for port_id, _ in port_request_batches]
            async with client.create_session(session_resource_ids) as session:
                deploy_coroutines = [
                    self._deploy_port_batch(
                        session=session,
                        port_id=port_id,
                        port_requests=port_requests,
                        fixed_timeline_profile_cls=fixed_timeline_profile_cls,
                        instrument_definition_cls=instrument_definition_cls,
                        instrument_mode=instrument_mode,
                        instrument_role=instrument_role,
                    )
                    for port_id, port_requests in port_request_batches
                ]
                port_results = (
                    await asyncio.gather(*deploy_coroutines)
                    if parallel
                    else [await coro for coro in deploy_coroutines]
                )

        for port_result in port_results:
            deployed.update(port_result.deployed)
            target_alias_map.update(port_result.target_alias_map)

        # Temporary limitation:
        # quelware `append=True` is currently broken, so QuEL-3 deploy uses
        # per-request replacement semantics. Until quelware is fixed, keep the
        # local alias/instrument cache aligned with the explicit deploy subset.
        self._last_deployed_instrument_infos = dict(deployed)
        self._target_alias_map = target_alias_map
        return deployed

    async def _deploy_port_batch(
        self,
        *,
        session: SessionProtocol,
        port_id: str,
        port_requests: tuple[InstrumentDeployRequest, ...],
        fixed_timeline_profile_cls: _FixedTimelineProfileFactory,
        instrument_definition_cls: _InstrumentDefinitionFactory,
        instrument_mode: _EnumNamespace,
        instrument_role: _EnumNamespace,
    ) -> _PortDeployResult:
        """Deploy one port batch through the active quelware session."""
        definitions: list[InstrumentDefinitionProtocol] = []
        for request in port_requests:
            profile = fixed_timeline_profile_cls(
                frequency_range_min=request.frequency_range_min_hz,
                frequency_range_max=request.frequency_range_max_hz,
            )
            definitions.append(
                instrument_definition_cls(
                    alias=request.alias,
                    mode=instrument_mode.FIXED_TIMELINE,
                    role=getattr(instrument_role, request.role),
                    profile=profile,
                )
            )

        inst_infos = await session.deploy_instruments(
            port_id,
            definitions=definitions,
            append=True,
        )
        infos_by_alias: dict[str, list[InstrumentInfoProtocol]] = defaultdict(list)
        for inst_info in inst_infos:
            infos_by_alias[inst_info.definition.alias].append(inst_info)

        deployed: dict[str, tuple[InstrumentInfoProtocol, ...]] = {}
        target_alias_map: dict[str, str] = {}
        for request in port_requests:
            matched_infos = tuple(infos_by_alias.get(request.alias, ()))
            if len(matched_infos) != 1:
                raise ValueError(
                    "quelware did not return the deployed instrument info for one request."
                )
            deployed[request.alias] = matched_infos
            for target_label in request.target_labels:
                target_alias_map[target_label] = request.alias
        return _PortDeployResult(
            deployed=deployed,
            target_alias_map=target_alias_map,
        )

    async def _refresh_instrument_cache(
        self,
    ) -> dict[str, tuple[InstrumentInfoProtocol, ...]]:
        """Load existing fixed-timeline instruments into local alias caches."""
        client_factory = self._load_quelware_client_factory()
        async with client_factory(
            self._quelware_endpoint,
            self._quelware_port,
        ) as client:
            instrument_infos = await self._list_instrument_infos(
                client=client,
                parallel=True,
            )

        deployed: dict[str, tuple[InstrumentInfoProtocol, ...]] = {}
        target_alias_map: dict[str, str] = {}
        for instrument_info in instrument_infos:
            alias = instrument_info.definition.alias
            if len(alias.strip()) == 0:
                continue
            normalized_alias = alias.strip()
            deployed[normalized_alias] = (instrument_info,)
            target_alias_map[normalized_alias] = normalized_alias

        self._last_deployed_instrument_infos = deployed
        self._target_alias_map = target_alias_map
        return dict(deployed)

    async def _fetch_backend_settings_from_hardware(
        self,
        *,
        unit_labels_by_box_id: dict[str, str],
        parallel: bool,
    ) -> dict[str, dict]:
        """Fetch normalized instrument snapshots for selected QuEL-3 boxes."""
        box_id_by_unit_label = {
            unit_label: box_id for box_id, unit_label in unit_labels_by_box_id.items()
        }
        fetched: dict[str, dict] = {
            box_id: {"instruments": {}} for box_id in unit_labels_by_box_id
        }
        if len(unit_labels_by_box_id) == 0:
            return fetched

        client_factory = self._load_quelware_client_factory()
        async with client_factory(
            self._quelware_endpoint,
            self._quelware_port,
        ) as client:
            instrument_infos = await self._list_instrument_infos(
                client=client,
                parallel=parallel,
            )

        for instrument_info in instrument_infos:
            unit_label = self._extract_unit_label(str(instrument_info.port_id))
            box_id = box_id_by_unit_label.get(unit_label)
            if box_id is None:
                continue
            alias = instrument_info.definition.alias.strip()
            if len(alias) == 0:
                continue
            definition = self._serialize_instrument_definition(
                instrument_info.definition,
            )
            fetched[box_id]["instruments"][alias] = {
                "resource_id": str(instrument_info.id),
                "port_id": str(instrument_info.port_id),
                "role": self._normalize_role_name(instrument_info.definition.role),
                "definition": definition,
            }
        return fetched

    def _load_quelware_client_factory(self) -> QuelwareClientFactory:
        """Import quelware client factory lazily."""
        return load_quelware_client_factory(
            client_mode=self._client_mode,
            standalone_unit_label=self._standalone_unit_label,
        )

    @staticmethod
    def _is_instrument_resource(resource_info: ResourceInfoProtocol) -> bool:
        """Return whether one listed resource info represents an instrument."""
        category = resource_info.category
        category_name = getattr(category, "name", None)
        if isinstance(category_name, str):
            return category_name == "INSTRUMENT"
        return str(category) == "INSTRUMENT"

    async def _list_instrument_infos(
        self,
        *,
        client: QuelwareClientProtocol,
        parallel: bool,
    ) -> list[InstrumentInfoProtocol]:
        """List instrument infos from one quelware client session."""
        resource_infos = await client.list_resource_infos()
        instrument_resource_ids = [
            resource_info.id
            for resource_info in resource_infos
            if self._is_instrument_resource(resource_info)
        ]
        if parallel:
            return list(
                await asyncio.gather(
                    *(
                        client.get_instrument_info(resource_id)
                        for resource_id in instrument_resource_ids
                    )
                )
            )
        return [
            await client.get_instrument_info(resource_id)
            for resource_id in instrument_resource_ids
        ]

    @staticmethod
    def _extract_unit_label(resource_id: str) -> str:
        """Extract unit label prefix from one quelware resource ID."""
        return resource_id.split(":", maxsplit=1)[0]

    @staticmethod
    def _normalize_role_name(role: object) -> str:
        """Normalize one runtime instrument role value to a comparable string."""
        role_name = getattr(role, "name", role)
        if isinstance(role_name, str):
            return role_name
        return str(role_name)

    @classmethod
    def _normalize_enum_name(cls, value: object) -> str:
        """Normalize one enum-like runtime value to a comparable string."""
        return cls._normalize_role_name(value)

    @classmethod
    def _serialize_instrument_definition(cls, definition: object) -> dict[str, object]:
        """Serialize stable instrument-definition fields into plain data."""
        serialized = {
            "alias": getattr(definition, "alias", ""),
            "role": cls._normalize_enum_name(getattr(definition, "role", None)),
        }
        mode = getattr(definition, "mode", None)
        if mode is not None:
            serialized["mode"] = cls._normalize_enum_name(mode)
        profile = cls._serialize_profile(getattr(definition, "profile", None))
        if profile:
            serialized["profile"] = profile
        return serialized

    @staticmethod
    def _serialize_profile(profile: object) -> dict[str, object]:
        """Serialize stable fixed-timeline profile fields into plain data."""
        if profile is None:
            return {}
        serialized: dict[str, object] = {}
        for attr in ("frequency_range_min", "frequency_range_max"):
            value = getattr(profile, attr, None)
            if isinstance(value, int | float):
                serialized[attr] = float(value)
        return serialized

    @classmethod
    def _build_cached_definition(
        cls,
        *,
        alias: str,
        role: object,
        definition_config: object,
    ) -> _CachedInstrumentDefinition:
        """Build cached instrument-definition object from one backend snapshot."""
        role_name = cls._normalize_role_name(role)
        if not isinstance(definition_config, Mapping):
            return _CachedInstrumentDefinition(alias=alias, role=role_name)

        mode = definition_config.get("mode")
        mode_name = None
        if mode is not None:
            mode_name = cls._normalize_enum_name(mode)

        profile = None
        profile_config = definition_config.get("profile")
        if isinstance(profile_config, Mapping):
            freq_min = profile_config.get("frequency_range_min")
            freq_max = profile_config.get("frequency_range_max")
            profile = _CachedFixedTimelineProfile(
                frequency_range_min=(
                    float(freq_min) if isinstance(freq_min, int | float) else None
                ),
                frequency_range_max=(
                    float(freq_max) if isinstance(freq_max, int | float) else None
                ),
            )

        return _CachedInstrumentDefinition(
            alias=alias,
            role=role_name,
            mode=mode_name,
            profile=profile,
        )

    @staticmethod
    def _load_instrument_entities() -> tuple[
        _FixedTimelineProfileFactory,
        _InstrumentDefinitionFactory,
        _EnumNamespace,
        _EnumNamespace,
    ]:
        """Import instrument entities lazily from quelware core package."""
        instrument_module = importlib.import_module("quelware_core.entities.instrument")
        return (
            instrument_module.FixedTimelineProfile,
            instrument_module.InstrumentDefinition,
            instrument_module.InstrumentMode,
            instrument_module.InstrumentRole,
        )
