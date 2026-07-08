"""Configuration manager for QuEL-3 backend instrument deployment."""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from qubex.backend.quel3.infra.quelware_imports import Quel3ClientMode
from qubex.backend.quel3.interfaces.client import (
    FixedTimelineProfileFactory,
    InstrumentDefinitionFactory,
    InstrumentDefinitionProtocol,
    InstrumentInfoProtocol,
    InstrumentModeNamespaceProtocol,
    InstrumentRoleNamespaceProtocol,
    InstrumentRoleProtocol,
    QuelwareClientFactory,
    QuelwareClientProtocol,
    ResourceInfoProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.managers.hardware_state_reader import (
    Quel3HardwareStateReader,
)
from qubex.backend.quel3.managers.runtime_config import Quel3RuntimeConfig
from qubex.backend.quel3.managers.session_workarounds import (
    QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS,
    QuelwareSessionError,
    enter_quelware_session_with_resource_retry,
    quelware_exception_summary,
    quelware_session_token,
)
from qubex.backend.quel3.models import InstrumentDeployRequest, RoleName
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

T = TypeVar("T")
TargetAliasKey = tuple[str, str]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QuelwareInstrumentEntities:
    """Lazy-loaded quelware instrument entities needed for deployment."""

    fixed_timeline_profile_factory: FixedTimelineProfileFactory
    instrument_definition_factory: InstrumentDefinitionFactory
    instrument_mode_namespace: InstrumentModeNamespaceProtocol
    instrument_role_namespace: InstrumentRoleNamespaceProtocol

    def role_value(self, role: RoleName) -> InstrumentRoleProtocol:
        """Return quelware instrument-role value for one deploy role name."""
        if role == "TRANSMITTER":
            return self.instrument_role_namespace.TRANSMITTER
        if role == "TRANSCEIVER":
            return self.instrument_role_namespace.TRANSCEIVER
        if role == "TRANSCEIVER_LOOPBACK":
            return self.instrument_role_namespace.TRANSCEIVER
        if role == "RECEIVER":
            return self.instrument_role_namespace.TRANSCEIVER
        raise ValueError(f"Unsupported QuEL-3 instrument role: {role!r}")


@dataclass(frozen=True)
class _CachedFixedTimelineProfile:
    """Cached fixed-timeline profile restored from hardware snapshot."""

    frequency_range_min: float | None = None
    frequency_range_max: float | None = None


@dataclass(frozen=True)
class _CachedInstrumentDefinition:
    """Cached instrument definition restored from hardware snapshot."""

    alias: str
    role: str
    mode: str | None = None
    profile: _CachedFixedTimelineProfile | None = None


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
    target_alias_map: dict[TargetAliasKey, str]


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
        runtime_config: Quel3RuntimeConfig | None = None,
    ) -> None:
        self._runtime_config = runtime_config or Quel3RuntimeConfig()
        self._last_deployed_instrument_infos: dict[
            str, tuple[InstrumentInfoProtocol, ...]
        ] = {}
        self._target_alias_map: dict[TargetAliasKey, str] = {}

    @property
    def runtime_config(self) -> Quel3RuntimeConfig:
        """Return the shared quelware runtime config."""
        return self._runtime_config

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint used for deployment."""
        return self._runtime_config.endpoint

    @property
    def quelware_port(self) -> int:
        """Return quelware port used for deployment."""
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
    def last_deployed_instrument_infos(
        self,
    ) -> dict[str, tuple[InstrumentInfoProtocol, ...]]:
        """Return last deployed instrument infos keyed by alias."""
        return dict(self._last_deployed_instrument_infos)

    @property
    def target_alias_map(self) -> dict[TargetAliasKey, str]:
        """Return last deployed box-and-target to runtime-alias mapping."""
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
        """Fetch backend settings from hardware through the state reader."""
        return Quel3HardwareStateReader(
            runtime_config=self._runtime_config,
        ).fetch_backend_settings_from_hardware(
            unit_labels_by_box_id=unit_labels_by_box_id,
            parallel=parallel,
        )

    def sync_backend_settings_to_cache(
        self,
        *,
        backend_settings: Mapping[str, dict],
    ) -> None:
        """Restore instrument alias caches from normalized backend settings."""
        deployed: dict[str, tuple[InstrumentInfoProtocol, ...]] = {}
        target_alias_map: dict[TargetAliasKey, str] = {}

        for box_id, box_config in backend_settings.items():
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
                local_alias, runtime_alias = self._split_alias_for_port(
                    alias=definition.alias,
                    port_id=port_id,
                )
                deployed[alias] = (
                    _CachedInstrumentInfo(
                        id=resource_id,
                        port_id=port_id,
                        definition=definition,
                    ),
                )
                target_alias_map[(box_id, local_alias)] = runtime_alias

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
        instrument_entities = self._load_instrument_entities()

        deployed: dict[str, tuple[InstrumentInfoProtocol, ...]] = {}
        target_alias_map: dict[TargetAliasKey, str] = {}
        requests_by_port: dict[str, list[InstrumentDeployRequest]] = defaultdict(list)
        for request in requests:
            requests_by_port[request.port_id].append(request)
        port_request_batches = tuple(
            (port_id, tuple(port_requests))
            for port_id, port_requests in requests_by_port.items()
        )
        port_results: list[_PortDeployResult]
        for attempt in range(QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS):
            try:
                port_results = await self._deploy_port_batches(
                    client_factory=client_factory,
                    port_request_batches=port_request_batches,
                    instrument_entities=instrument_entities,
                    parallel=parallel,
                    attempt=attempt + 1,
                    max_attempts=QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS,
                )
                break
            except Exception as exc:
                if attempt + 1 >= QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS:
                    if isinstance(exc, QuelwareSessionError):
                        raise
                    missing_session_id = "<unavailable>"
                    raise QuelwareSessionError(
                        "QuEL-3 quelware deploy request failed after retries",
                        session_token=missing_session_id,
                        cause=exc,
                    ) from exc
        else:
            raise RuntimeError("unreachable QuEL-3 deploy retry state")

        for port_result in port_results:
            deployed.update(port_result.deployed)
            target_alias_map.update(port_result.target_alias_map)

        self._last_deployed_instrument_infos = dict(deployed)
        # Keep local alias/instrument cache aligned with the explicitly deployed
        # subset so it matches this replacement-style call behavior.
        self._target_alias_map = target_alias_map
        return deployed

    async def _deploy_port_batches(
        self,
        *,
        client_factory: QuelwareClientFactory,
        port_request_batches: tuple[
            tuple[str, tuple[InstrumentDeployRequest, ...]], ...
        ],
        instrument_entities: _QuelwareInstrumentEntities,
        parallel: bool,
        attempt: int,
        max_attempts: int,
    ) -> list[_PortDeployResult]:
        """Deploy port batches in one quelware client/session context."""
        async with client_factory(
            self._runtime_config.endpoint,
            self._runtime_config.port,
        ) as client:
            session_resource_ids = [port_id for port_id, _ in port_request_batches]
            session_cm, session = await enter_quelware_session_with_resource_retry(
                client=client,
                resource_ids=session_resource_ids,
            )
            session_token = quelware_session_token(session)
            try:
                deploy_coroutines = [
                    self._deploy_port_batch(
                        session=session,
                        port_id=port_id,
                        port_requests=port_requests,
                        instrument_entities=instrument_entities,
                    )
                    for port_id, port_requests in port_request_batches
                ]
                return (
                    list(await asyncio.gather(*deploy_coroutines))
                    if parallel
                    else [await coro for coro in deploy_coroutines]
                )
            except Exception as exc:
                if attempt >= max_attempts:
                    raise QuelwareSessionError(
                        "QuEL-3 quelware deploy request failed after retries",
                        session_token=session_token,
                        cause=exc,
                    ) from exc
                logger.warning(
                    "QuEL-3 quelware deploy request failed; session_token=%s; "
                    "attempt=%d/%d; retrying with a fresh session; cause=%s",
                    session_token,
                    attempt,
                    max_attempts,
                    quelware_exception_summary(exc),
                )
                raise
            finally:
                try:
                    await session_cm.__aexit__(None, None, None)
                except Exception as exc:
                    logger.warning(
                        "QuEL-3 quelware deploy session cleanup failed; "
                        "session_token=%s; cause=%s",
                        session_token,
                        quelware_exception_summary(exc),
                    )

    async def _deploy_port_batch(
        self,
        *,
        session: SessionProtocol,
        port_id: str,
        port_requests: tuple[InstrumentDeployRequest, ...],
        instrument_entities: _QuelwareInstrumentEntities,
    ) -> _PortDeployResult:
        """Deploy one port batch through the active quelware session."""
        definitions: list[InstrumentDefinitionProtocol] = []
        for request in port_requests:
            profile = instrument_entities.fixed_timeline_profile_factory(
                frequency_range_min=request.frequency_range_min_hz,
                frequency_range_max=request.frequency_range_max_hz,
            )
            definitions.append(
                instrument_entities.instrument_definition_factory(
                    alias=request.alias,
                    mode=instrument_entities.instrument_mode_namespace.FIXED_TIMELINE,
                    role=instrument_entities.role_value(request.role),
                    profile=profile,
                )
            )

        instrument_infos = await session.deploy_instruments(
            port_id,
            definitions=definitions,
            # Qubex push treats the current target registry as the source of
            # truth for this port's selected instruments.
            append=False,
        )
        instrument_infos_by_alias: dict[str, list[InstrumentInfoProtocol]] = (
            defaultdict(list)
        )
        for instrument_info in instrument_infos:
            local_alias, _runtime_alias = self._split_alias_for_port(
                alias=instrument_info.definition.alias,
                port_id=instrument_info.port_id,
            )
            instrument_infos_by_alias[local_alias].append(instrument_info)

        deployed: dict[str, tuple[InstrumentInfoProtocol, ...]] = {}
        target_alias_map: dict[TargetAliasKey, str] = {}
        for request in port_requests:
            matched_instrument_infos = tuple(
                instrument_infos_by_alias.get(request.alias, ())
            )
            if len(matched_instrument_infos) != 1:
                raise ValueError(
                    "quelware did not return the deployed instrument info for one request."
                )
            deployed[request.alias] = matched_instrument_infos
            runtime_alias = self._runtime_alias_from_instrument_info(
                instrument_info=matched_instrument_infos[0],
                fallback_alias=request.alias,
            )
            for target_label in request.target_labels:
                target_alias_map[(request.box_id, target_label)] = runtime_alias
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
            self._runtime_config.endpoint,
            self._runtime_config.port,
        ) as client:
            instrument_infos = await self._list_instrument_infos(
                client=client,
                parallel=True,
                unit_labels=None,
            )

        deployed: dict[str, tuple[InstrumentInfoProtocol, ...]] = {}
        for instrument_info in instrument_infos:
            alias = instrument_info.definition.alias
            if len(alias.strip()) == 0:
                continue
            local_alias, _runtime_alias = self._split_alias_for_port(
                alias=alias,
                port_id=instrument_info.port_id,
            )
            deployed[local_alias] = (instrument_info,)

        self._last_deployed_instrument_infos = deployed
        self._target_alias_map = {}
        return dict(deployed)

    def _load_quelware_client_factory(self) -> QuelwareClientFactory:
        """Import quelware client factory lazily."""
        return self._runtime_config.load_client_factory()

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
        unit_labels: Collection[str] | None,
    ) -> list[InstrumentInfoProtocol]:
        """List instrument infos from one quelware client session."""
        resource_infos = await client.list_resource_infos()
        instrument_resource_infos = [
            resource_info
            for resource_info in resource_infos
            if self._is_instrument_resource(resource_info)
        ]
        if unit_labels is not None:
            selected_unit_labels = set(unit_labels)
            instrument_resource_infos = [
                resource_info
                for resource_info in instrument_resource_infos
                if self._is_resource_info_in_selected_units(
                    resource_info=resource_info,
                    unit_labels=selected_unit_labels,
                )
            ]
        instrument_resource_ids = [
            resource_info.id for resource_info in instrument_resource_infos
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

    @classmethod
    def _is_resource_info_in_selected_units(
        cls,
        *,
        resource_info: ResourceInfoProtocol,
        unit_labels: Collection[str],
    ) -> bool:
        """Return whether one resource can be matched to the selected units."""
        resource_id = str(resource_info.id)
        if ":" not in resource_id:
            return True
        return cls._extract_unit_label(resource_id) in unit_labels

    @staticmethod
    def _extract_unit_label(resource_id: str) -> str:
        """Extract unit label prefix from one quelware resource ID."""
        return resource_id.split(":", maxsplit=1)[0]

    @classmethod
    def _split_alias_for_port(cls, *, alias: str, port_id: str) -> tuple[str, str]:
        """Return local and runtime aliases for one instrument on a port."""
        runtime_alias = cls._runtime_alias_for_port(alias=alias, port_id=port_id)
        unit_label = cls._extract_unit_label(str(port_id))
        local_alias = cls._strip_unit_label_prefix(
            alias=runtime_alias,
            unit_label=unit_label,
        )
        return local_alias, runtime_alias

    @classmethod
    def _runtime_alias_for_port(cls, *, alias: str, port_id: str) -> str:
        """Return unit-qualified runtime alias for one port-local alias."""
        stripped_alias = alias.strip()
        if len(stripped_alias) == 0 or ":" not in str(port_id):
            return stripped_alias
        unit_label = cls._extract_unit_label(str(port_id))
        if stripped_alias.startswith(f"{unit_label}:") or ":" in stripped_alias:
            return stripped_alias
        return f"{unit_label}:{stripped_alias}"

    @staticmethod
    def _strip_unit_label_prefix(*, alias: str, unit_label: str) -> str:
        """Strip the quelware unit prefix from an alias when it matches the port."""
        prefix = f"{unit_label}:"
        if alias.startswith(prefix):
            return alias.removeprefix(prefix)
        return alias

    @staticmethod
    def _runtime_alias_from_instrument_info(
        *,
        instrument_info: InstrumentInfoProtocol,
        fallback_alias: str,
    ) -> str:
        """Return the runtime alias stored on one quelware instrument info."""
        alias = instrument_info.definition.alias.strip()
        if len(alias) == 0:
            alias = fallback_alias
        return Quel3ConfigurationManager._runtime_alias_for_port(
            alias=alias,
            port_id=str(instrument_info.port_id),
        )

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

        definition_alias = definition_config.get("alias")
        runtime_alias = (
            definition_alias.strip()
            if isinstance(definition_alias, str) and len(definition_alias.strip()) > 0
            else alias
        )

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
            alias=runtime_alias,
            role=role_name,
            mode=mode_name,
            profile=profile,
        )

    @staticmethod
    def _load_instrument_entities() -> _QuelwareInstrumentEntities:
        """Import instrument entities lazily from quelware core package."""
        instrument_module = importlib.import_module("quelware_core.entities.instrument")
        fixed_timeline_profile_factory: FixedTimelineProfileFactory = (
            instrument_module.FixedTimelineProfile
        )
        instrument_definition_factory: InstrumentDefinitionFactory = (
            instrument_module.InstrumentDefinition
        )
        instrument_mode_namespace: InstrumentModeNamespaceProtocol = (
            instrument_module.InstrumentMode
        )
        instrument_role_namespace: InstrumentRoleNamespaceProtocol = (
            instrument_module.InstrumentRole
        )
        return _QuelwareInstrumentEntities(
            fixed_timeline_profile_factory=fixed_timeline_profile_factory,
            instrument_definition_factory=instrument_definition_factory,
            instrument_mode_namespace=instrument_mode_namespace,
            instrument_role_namespace=instrument_role_namespace,
        )
