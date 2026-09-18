"""Configuration manager for QuEL-3 backend instrument deployment."""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from qubex.backend.quel3.infra.quelware_imports import Quel3ClientMode
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import (
    FixedTimelineProfileFactory,
    InstrumentDefinitionFactory,
    InstrumentDefinitionProtocol,
    InstrumentInfoProtocol,
    InstrumentModeNamespaceProtocol,
    InstrumentRoleNamespaceProtocol,
    InstrumentRoleProtocol,
    QuelwareClientFactory,
    SessionProtocol,
)
from qubex.backend.quel3.managers.hardware_state_reader import Quel3HardwareStateReader
from qubex.backend.quel3.managers.runtime_config import Quel3RuntimeConfig
from qubex.backend.quel3.managers.session_workarounds import (
    QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS,
    QuelwareSessionError,
    enter_quelware_session_with_resource_retry,
    quelware_exception_summary,
    quelware_session_token,
)
from qubex.backend.quel3.models import (
    InstrumentConfiguration,
    InstrumentRoleName,
    InstrumentSpec,
)
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QuelwareInstrumentEntities:
    """Lazy-loaded quelware instrument entities needed for deployment."""

    fixed_timeline_profile_factory: FixedTimelineProfileFactory
    instrument_definition_factory: InstrumentDefinitionFactory
    instrument_mode_namespace: InstrumentModeNamespaceProtocol
    instrument_role_namespace: InstrumentRoleNamespaceProtocol

    def role_value(self, role: InstrumentRoleName) -> InstrumentRoleProtocol:
        """Return quelware instrument-role value for one deploy role name."""
        if role == "TRANSMITTER":
            return self.instrument_role_namespace.TRANSMITTER
        if role == "TRANSCEIVER":
            return self.instrument_role_namespace.TRANSCEIVER
        if role == "TRANSCEIVER_LOOPBACK":
            return self.instrument_role_namespace.TRANSCEIVER_LOOPBACK
        if role == "RECEIVER":
            return self.instrument_role_namespace.RECEIVER
        raise ValueError(f"Unsupported QuEL-3 instrument role: {role!r}")


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-configuration")
    return bridge.run(factory, timeout=timeout)


class Quel3ConfigurationManager:
    """Deploy instrument configurations and refresh the supplied runtime cache."""

    def __init__(
        self,
        *,
        runtime_config: Quel3RuntimeConfig | None = None,
    ) -> None:
        self._runtime_config = runtime_config or Quel3RuntimeConfig()

    @property
    def runtime_config(self) -> Quel3RuntimeConfig:
        """Return the shared quelware runtime config."""
        return self._runtime_config

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint used for deployment."""
        return self._runtime_config.endpoint

    @property
    def quelware_port(self) -> int | None:
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

    def clear_instruments(
        self,
        *,
        unit_label: str,
        instrument_cache: InstrumentCache,
        parallel: bool = True,
    ) -> None:
        """
        Delete all instruments on one unit and invalidate that unit's cache.

        Parameters
        ----------
        unit_label : str
            Exact label of a discovered unit. Required; no all-unit default.
        instrument_cache : InstrumentCache
            Controller-owned cache whose selected unit should be invalidated.
        parallel : bool, default=True
            Whether to discard instruments on different ports concurrently.

        Raises
        ------
        ValueError
            The unit label is empty or was not discovered.

        Notes
        -----
        Discover ports independently of the cache. Invalidate only the selected
        unit before writing, even if it has no ports. Other units are retained.
        Session acquisition and deletion requests are not retried by this
        manager. Errors propagate unchanged and partial deletions are not
        rolled back. A failed write leaves the selected unit uncached.
        """
        if not unit_label.strip():
            raise ValueError("Unit label must not be empty.")
        _run_async(
            lambda: self._clear_instruments(
                unit_label=unit_label,
                instrument_cache=instrument_cache,
                parallel=parallel,
            )
        )

    async def _clear_instruments(
        self,
        *,
        unit_label: str,
        instrument_cache: InstrumentCache,
        parallel: bool,
    ) -> None:
        """Discover one unit's ports and discard their instruments in one session."""
        client_factory = self._load_quelware_client_factory()
        async with client_factory(
            self._runtime_config.endpoint, self._runtime_config.port
        ) as client:
            if unit_label not in client.list_unit_labels():
                raise ValueError(f"QuEL-3 unit was not discovered: {unit_label!r}.")
            resources = await client.list_resource_infos()
            port_ids = tuple(
                dict.fromkeys(
                    resource.id
                    for resource in resources
                    if str(
                        getattr(resource.category, "name", resource.category)
                    ).rsplit(".", maxsplit=1)[-1]
                    == "PORT"
                    and resource.id.startswith(f"{unit_label}:")
                )
            )
            instrument_cache.replace_units(
                unit_labels=(unit_label,), instrument_infos=()
            )
            if not port_ids:
                return
            async with client.create_session(port_ids) as session:
                if parallel:
                    results = await asyncio.gather(
                        *(session.discard_instruments(port_id) for port_id in port_ids),
                        return_exceptions=True,
                    )
                    for result in results:
                        if isinstance(result, BaseException):
                            raise result
                else:
                    for port_id in port_ids:
                        await session.discard_instruments(port_id)

    def deploy_instruments(
        self,
        *,
        configuration: InstrumentConfiguration,
        instrument_cache: InstrumentCache,
        hardware_state_reader: Quel3HardwareStateReader,
        parallel: bool = True,
    ) -> dict[str, InstrumentInfoProtocol]:
        """
        Replace configured ports and cache their complete hardware readback.

        Notes
        -----
        Old identities on listed ports are discarded before hardware writes.
        Failed deployment or readback leaves those ports absent from the cache.
        Other ports are retained, and an empty configuration performs no work.
        """
        return self._deploy_and_refresh(
            specifications=configuration.instruments,
            instrument_cache=instrument_cache,
            hardware_state_reader=hardware_state_reader,
            append=False,
            parallel=parallel,
        )

    def deploy_instrument(
        self,
        *,
        instrument: InstrumentSpec,
        instrument_cache: InstrumentCache,
        hardware_state_reader: Quel3HardwareStateReader,
        append: bool = True,
        parallel: bool = True,
    ) -> InstrumentInfoProtocol:
        """
        Deploy one instrument and return its complete hardware readback.

        Parameters
        ----------
        instrument : InstrumentSpec
            Instrument definition and its unit-qualified port ID.
        instrument_cache : InstrumentCache
            Controller-owned cache to update after deployment.
        hardware_state_reader : Quel3HardwareStateReader
            Reader for complete port readback after deployment.
        append : bool, default=True
            Add or replace this alias while preserving the port's other
            instruments. If false, replace every instrument on the port
            with this one.
        parallel : bool, default=True
            Whether hardware reads may run concurrently.

        Raises
        ------
        ValueError
            Hardware readback is invalid.

        Notes
        -----
        Alias replacement is handled by quelware without a pre-deployment read.
        Append does not retry the deployment request. After either operation,
        read the entire port into the cache. Write or readback failure leaves
        that port uncached.
        """
        instrument_infos = self._deploy_and_refresh(
            specifications=(instrument,),
            instrument_cache=instrument_cache,
            hardware_state_reader=hardware_state_reader,
            append=append,
            parallel=parallel,
        )
        return instrument_infos[instrument.alias]

    def _deploy_and_refresh(
        self,
        *,
        specifications: tuple[InstrumentSpec, ...],
        instrument_cache: InstrumentCache,
        hardware_state_reader: Quel3HardwareStateReader,
        append: bool,
        parallel: bool,
    ) -> dict[str, InstrumentInfoProtocol]:
        """Invalidate touched ports, deploy definitions, and publish complete readback."""
        if not specifications:
            return {}
        port_ids = tuple(dict.fromkeys(spec.port_id for spec in specifications))
        instrument_cache.replace_ports(port_ids=port_ids, instrument_infos=())
        _run_async(
            lambda: self._deploy_instruments(
                specifications=specifications,
                append=append,
                parallel=parallel,
            )
        )
        instrument_infos = hardware_state_reader.read_instrument_infos(
            port_ids=port_ids,
            parallel=parallel,
        )
        for specification in specifications:
            matches = [
                info
                for info in instrument_infos
                if InstrumentCache.alias_for(info) == specification.alias
                and info.port_id == specification.port_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Hardware readback did not return exactly one instrument "
                    f"for alias `{specification.alias}` on port `{specification.port_id}`."
                )
        instrument_cache.replace_ports(
            port_ids=port_ids, instrument_infos=instrument_infos
        )
        return {InstrumentCache.alias_for(info): info for info in instrument_infos}

    def refresh_instrument_cache(
        self,
        *,
        instrument_cache: InstrumentCache,
        hardware_state_reader: Quel3HardwareStateReader,
        unit_labels: Sequence[str] | None = None,
        parallel: bool = True,
    ) -> dict[str, InstrumentInfoProtocol]:
        """
        Refresh complete hardware information for all or selected units.

        `None` selects all units; an empty sequence performs no work. Replace
        the selected scope only after successful acquisition and validation.
        Return the instruments acquired by this call.
        """
        if unit_labels is not None and not unit_labels:
            return {}
        instrument_infos = hardware_state_reader.read_instrument_infos(
            unit_labels=() if unit_labels is None else tuple(unit_labels),
            parallel=parallel,
        )
        if unit_labels is None:
            instrument_cache.replace_all(instrument_infos=instrument_infos)
        else:
            instrument_cache.replace_units(
                unit_labels=unit_labels, instrument_infos=instrument_infos
            )
        return {InstrumentCache.alias_for(info): info for info in instrument_infos}

    @staticmethod
    def get_instrument_configuration(
        *, instrument_cache: InstrumentCache
    ) -> InstrumentConfiguration:
        """Export deployable specifications from the supplied live cache."""
        return instrument_cache.export_configuration()

    def save_instrument_configuration(
        self, path: str | Path, *, instrument_cache: InstrumentCache
    ) -> Path:
        """Save cached deployable specifications as YAML without reading hardware."""
        configuration = self.get_instrument_configuration(
            instrument_cache=instrument_cache
        )
        return configuration.save_yaml(path)

    @staticmethod
    def load_instrument_configuration(path: str | Path) -> InstrumentConfiguration:
        """Load deployable specifications from YAML without changing hardware or cache."""
        return InstrumentConfiguration.load_yaml(path)

    async def _deploy_instruments(
        self,
        *,
        specifications: tuple[InstrumentSpec, ...],
        append: bool = False,
        parallel: bool = True,
    ) -> None:
        """Deploy instruments through quelware session APIs."""
        if not specifications:
            return

        client_factory = self._load_quelware_client_factory()
        instrument_entities = self._load_instrument_entities()

        specifications_by_port: dict[str, list[InstrumentSpec]] = defaultdict(list)
        for specification in specifications:
            specifications_by_port[specification.port_id].append(specification)
        port_batches = tuple(
            (port_id, tuple(port_specifications))
            for port_id, port_specifications in specifications_by_port.items()
        )
        max_attempts = 1 if append else QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS
        for attempt in range(max_attempts):
            try:
                await self._deploy_port_batches(
                    client_factory=client_factory,
                    port_batches=port_batches,
                    instrument_entities=instrument_entities,
                    append=append,
                    parallel=parallel,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
                break
            except Exception as exc:
                if attempt + 1 >= max_attempts:
                    if isinstance(exc, QuelwareSessionError):
                        raise
                    missing_session_id = "<unavailable>"
                    raise QuelwareSessionError(
                        "QuEL-3 quelware append request failed without retry"
                        if append
                        else "QuEL-3 quelware deploy request failed after retries",
                        session_token=missing_session_id,
                        cause=exc,
                    ) from exc

    async def _deploy_port_batches(
        self,
        *,
        client_factory: QuelwareClientFactory,
        port_batches: tuple[tuple[str, tuple[InstrumentSpec, ...]], ...],
        instrument_entities: _QuelwareInstrumentEntities,
        append: bool,
        parallel: bool,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Deploy port batches in one quelware client/session context."""
        async with client_factory(
            self._runtime_config.endpoint,
            self._runtime_config.port,
        ) as client:
            session_resource_ids = [port_id for port_id, _ in port_batches]
            session_cm, session = await enter_quelware_session_with_resource_retry(
                client=client,
                resource_ids=session_resource_ids,
            )
            session_token = quelware_session_token(session)
            try:
                if parallel:
                    # Finish every port before closing or retrying this session.
                    results = await asyncio.gather(
                        *(
                            self._deploy_port_batch(
                                session=session,
                                port_id=port_id,
                                port_specifications=port_specifications,
                                instrument_entities=instrument_entities,
                                append=append,
                            )
                            for port_id, port_specifications in port_batches
                        ),
                        return_exceptions=True,
                    )
                    for result in results:
                        if isinstance(result, BaseException):
                            raise result  # noqa: TRY301
                else:
                    for port_id, port_specifications in port_batches:
                        await self._deploy_port_batch(
                            session=session,
                            port_id=port_id,
                            port_specifications=port_specifications,
                            instrument_entities=instrument_entities,
                            append=append,
                        )
            except Exception as exc:
                if attempt >= max_attempts:
                    raise QuelwareSessionError(
                        "QuEL-3 quelware append request failed without retry"
                        if append
                        else "QuEL-3 quelware deploy request failed after retries",
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
        port_specifications: tuple[InstrumentSpec, ...],
        instrument_entities: _QuelwareInstrumentEntities,
        append: bool,
    ) -> None:
        """Deploy one port batch through the active quelware session."""
        definitions: list[InstrumentDefinitionProtocol] = []
        for specification in port_specifications:
            profile = instrument_entities.fixed_timeline_profile_factory(
                frequency_range_min=specification.frequency_range_min_hz,
                frequency_range_max=specification.frequency_range_max_hz,
            )
            definitions.append(
                instrument_entities.instrument_definition_factory(
                    alias=specification.alias,
                    mode=instrument_entities.instrument_mode_namespace.FIXED_TIMELINE,
                    role=instrument_entities.role_value(specification.role),
                    profile=profile,
                )
            )

        await session.deploy_instruments(
            port_id,
            definitions=definitions,
            append=append,
        )

    def _load_quelware_client_factory(self) -> QuelwareClientFactory:
        """Import quelware client factory lazily."""
        return self._runtime_config.load_client_factory()

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
