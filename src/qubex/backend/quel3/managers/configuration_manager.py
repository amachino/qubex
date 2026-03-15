"""Configuration manager for QuEL-3 backend instrument deployment."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from qubex.backend.quel3.infra.quelware_imports import (
    Quel3ClientMode,
    import_module_with_workspace_fallback,
    load_quelware_client_factory,
    validate_quelware_client_runtime,
)
from qubex.backend.quel3.interfaces.client import (
    InstrumentDefinitionProtocol,
    InstrumentInfoProtocol,
    QuelwareClientFactory,
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
        client_mode: str = "server",
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
            append=False,
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
            resource_infos = await client.list_resource_infos()
            instrument_resource_ids = [
                resource_info.id
                for resource_info in resource_infos
                if self._is_instrument_resource(resource_info)
            ]
            instrument_infos = await asyncio.gather(
                *(
                    client.get_instrument_info(resource_id)
                    for resource_id in instrument_resource_ids
                )
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

    @staticmethod
    def _load_instrument_entities() -> tuple[
        _FixedTimelineProfileFactory,
        _InstrumentDefinitionFactory,
        _EnumNamespace,
        _EnumNamespace,
    ]:
        """Import instrument entities lazily from quelware core package."""
        instrument_module = import_module_with_workspace_fallback(
            "quelware_core.entities.instrument"
        )
        return (
            instrument_module.FixedTimelineProfile,
            instrument_module.InstrumentDefinition,
            instrument_module.InstrumentMode,
            instrument_module.InstrumentRole,
        )
