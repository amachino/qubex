"""Configuration manager for QuEL-3 backend instrument deployment."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from qubex.backend.quel3.infra.quelware_imports import (
    Quel3ClientMode,
    QuelwareInstrumentEntities,
    load_quelware_instrument_entities,
)
from qubex.backend.quel3.infra.runtime_config import Quel3RuntimeConfig
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import (
    InstrumentDefinitionProtocol,
    InstrumentInfoProtocol,
    QuelwareClientFactory,
    SessionProtocol,
)
from qubex.backend.quel3.managers.session_workarounds import (
    QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS,
    QuelwareSessionAcquisitionError,
    QuelwareSessionError,
    enter_quelware_session,
    quelware_exception_summary,
    quelware_session_token,
)
from qubex.backend.quel3.models import (
    InstrumentConfiguration,
    InstrumentSpec,
)
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

if TYPE_CHECKING:
    from qubex.backend.quel3.infra.resource_reader import Quel3ResourceReader

T = TypeVar("T")

logger = logging.getLogger(__name__)
_MONITOR_MODE_CONTROL_KEY = "quel3.monitor.mode"


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

    def configure_monitor_mode(
        self,
        *,
        unit_label: str,
        instrument_cache: InstrumentCache,
        mode: Literal["open", "loopback"] = "open",
        clear_instruments: bool = False,
    ) -> str:
        """
        Configure one unit's monitor mode before instruments are deployed.

        Parameters
        ----------
        unit_label : str
            Exact label of the QuEL-3 unit.
        instrument_cache : InstrumentCache
            Shared cache to check and invalidate when instruments are deleted.
        mode : {"open", "loopback"}, default="open"
            Normal external output or monitor loopback mode.
        clear_instruments : bool, default=False
            Delete every instrument on this unit before configuring the mode.

        Returns
        -------
        str
            Applied monitor mode reported by quelware.

        Raises
        ------
        RuntimeError
            Cached instruments remain and `clear_instruments` is false.

        Notes
        -----
        Quelware requires a lock on every port of the unit and no deployed
        instruments when changing a unit control. Validate the live control
        before deleting instruments. Deletion and configuration share one
        all-port session. Deleted instruments are not restored; failures after
        deletion leave the unit's cache invalidated. Other units are retained.
        """
        if not unit_label.strip():
            raise ValueError("Unit label must not be empty.")
        if mode not in ("open", "loopback"):
            raise ValueError("Monitor mode must be 'open' or 'loopback'.")
        if not clear_instruments and any(
            info.port_id.partition(":")[0] == unit_label
            for info in instrument_cache.snapshot().values()
        ):
            raise RuntimeError(
                "Clear deployed instruments with clear_instruments() before "
                "changing QuEL-3 monitor mode, or pass clear_instruments=True."
            )
        return _run_async(
            lambda: self._configure_monitor_mode(
                unit_label=unit_label,
                instrument_cache=instrument_cache,
                mode=mode,
                clear_instruments=clear_instruments,
            )
        )

    def get_monitor_mode(self, *, unit_label: str) -> Literal["open", "loopback"]:
        """Read the current monitor mode of one discovered unit."""
        if not unit_label.strip():
            raise ValueError("Unit label must not be empty.")
        return _run_async(lambda: self._get_monitor_mode(unit_label=unit_label))

    async def _get_monitor_mode(
        self, *, unit_label: str
    ) -> Literal["open", "loopback"]:
        """Read the live monitor control before changing instrument state."""
        client_factory = self._load_quelware_client_factory()
        async with client_factory(
            self._runtime_config.endpoint, self._runtime_config.port
        ) as client:
            if unit_label not in client.list_unit_labels():
                raise ValueError(f"QuEL-3 unit was not discovered: {unit_label!r}.")
            configuration = await client.get_unit_configuration(unit_label)
            control = next(
                (
                    spec
                    for spec in configuration.supported
                    if spec.key == _MONITOR_MODE_CONTROL_KEY
                ),
                None,
            )
            if control is None:
                raise RuntimeError(
                    f"QuEL-3 unit {unit_label!r} does not support monitor mode."
                )
            mode = control.current_value
            if mode == "open" or mode == "loopback":
                return mode
            raise ValueError(f"Unsupported QuEL-3 monitor mode readback: {mode!r}.")

    async def _configure_monitor_mode(
        self,
        *,
        unit_label: str,
        instrument_cache: InstrumentCache,
        mode: Literal["open", "loopback"],
        clear_instruments: bool,
    ) -> str:
        """Validate the live control and apply it under an all-port session."""
        client_factory = self._load_quelware_client_factory()
        async with client_factory(
            self._runtime_config.endpoint, self._runtime_config.port
        ) as client:
            if unit_label not in client.list_unit_labels():
                raise ValueError(f"QuEL-3 unit was not discovered: {unit_label!r}.")
            configuration = await client.get_unit_configuration(unit_label)
            control = next(
                (
                    spec
                    for spec in configuration.supported
                    if spec.key == _MONITOR_MODE_CONTROL_KEY
                ),
                None,
            )
            if control is None:
                raise RuntimeError(
                    f"QuEL-3 unit {unit_label!r} does not support monitor mode."
                )
            if mode not in control.allowed_values:
                raise ValueError(
                    f"Monitor mode {mode!r} is not supported; "
                    f"allowed values: {control.allowed_values}."
                )
            if control.current_value == mode and not clear_instruments:
                return mode
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
            if not port_ids:
                raise RuntimeError(
                    f"QuEL-3 unit {unit_label!r} has no discovered ports to lock."
                )
            async with client.create_session(port_ids) as session:
                if clear_instruments:
                    instrument_cache.replace_units(
                        unit_labels=(unit_label,), instrument_infos=()
                    )
                    await self._discard_instruments(
                        session=session, port_ids=port_ids, parallel=True
                    )
                values = await session.configure_unit(
                    unit_label, {_MONITOR_MODE_CONTROL_KEY: mode}
                )
            applied = values.get(_MONITOR_MODE_CONTROL_KEY)
            if applied != mode:
                raise RuntimeError(
                    f"QuEL-3 monitor mode readback was {applied!r}; expected {mode!r}."
                )
            return mode

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
                await self._discard_instruments(
                    session=session, port_ids=port_ids, parallel=parallel
                )

    @staticmethod
    async def _discard_instruments(
        *, session: SessionProtocol, port_ids: tuple[str, ...], parallel: bool
    ) -> None:
        """Finish every discard request before leaving the shared port session."""
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
        append: bool = True,
        parallel: bool = True,
    ) -> dict[str, InstrumentInfoProtocol]:
        """
        Deploy instruments and cache the information returned by quelware.

        Parameters
        ----------
        configuration : InstrumentConfiguration
            Instrument definitions to deploy.
        instrument_cache : InstrumentCache
            Controller-owned cache to update after deployment.
        append : bool, default=True
            Preserve other aliases on each requested port. If false, replace
            all instruments on each requested port.
        parallel : bool, default=True
            Whether port deployments may run concurrently.

        Notes
        -----
        Return only the deployed instruments. With `append=True`, invalidate
        and refresh only the specified aliases, preserving other cached entries.
        With `append=False`, invalidate and replace the entire requested ports.
        Invalidation precedes hardware writes, so failure leaves the affected
        scope uncached. An empty configuration performs no work.
        """
        specifications = configuration.instruments
        if not specifications:
            return {}
        port_ids = tuple(dict.fromkeys(spec.port_id for spec in specifications))
        self._validate_deployment_targets(
            specifications=specifications,
            cached=instrument_cache.snapshot(),
            port_ids=port_ids,
            append=append,
        )
        aliases = tuple(spec.alias for spec in specifications)
        if append:
            instrument_cache.replace_aliases(aliases=aliases, instrument_infos=())
        else:
            instrument_cache.replace_ports(port_ids=port_ids, instrument_infos=())
        deployed = _run_async(
            lambda: self._request_instrument_deployment(
                specifications=specifications,
                append=append,
                parallel=parallel,
            )
        )
        if append:
            instrument_cache.replace_aliases(aliases=aliases, instrument_infos=deployed)
        else:
            instrument_cache.replace_ports(port_ids=port_ids, instrument_infos=deployed)
        return {InstrumentCache.alias_for(info): info for info in deployed}

    def deploy_instrument(
        self,
        *,
        instrument: InstrumentSpec,
        instrument_cache: InstrumentCache,
        append: bool = True,
        parallel: bool = True,
    ) -> InstrumentInfoProtocol:
        """
        Deploy one instrument and return its deployment information.

        Parameters
        ----------
        instrument : InstrumentSpec
            Instrument definition and its unit-qualified port ID.
        instrument_cache : InstrumentCache
            Controller-owned cache to update after deployment.
        append : bool, default=True
            Add or replace this alias while preserving the port's other
            instruments. If false, replace every instrument on the port
            with this one.
        parallel : bool, default=True
            Whether to use the parallel deployment path for this single port.

        Raises
        ------
        ValueError
            The deployment response is invalid, or the alias is already cached on
            another port. Known cross-port collisions fail before writing.

        Notes
        -----
        Alias replacement is handled by quelware without a pre-deployment read.
        Append requests run once and invalidate only the specified alias. With
        `append=False`, invalidate and replace the entire port. Cache the
        instrument information returned by a successful deployment.
        """
        instrument_infos = self.deploy_instruments(
            configuration=InstrumentConfiguration(instruments=(instrument,)),
            instrument_cache=instrument_cache,
            append=append,
            parallel=parallel,
        )
        return instrument_infos[instrument.alias]

    @staticmethod
    def _validate_deployment_targets(
        *,
        specifications: tuple[InstrumentSpec, ...],
        cached: Mapping[str, InstrumentInfoProtocol],
        port_ids: tuple[str, ...],
        append: bool,
    ) -> None:
        """Reject aliases on retained ports before invalidating the cache or writing."""
        for specification in specifications:
            previous = cached.get(specification.alias)
            if (
                previous is not None
                and previous.port_id != specification.port_id
                and (append or previous.port_id not in port_ids)
            ):
                raise ValueError(
                    f"Instrument alias `{specification.alias}` is already cached on port "
                    f"`{previous.port_id}`; cannot deploy it on `{specification.port_id}`."
                )

    def refresh_instrument_cache(
        self,
        *,
        instrument_cache: InstrumentCache,
        resource_reader: Quel3ResourceReader,
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
        instrument_infos = resource_reader.read_instrument_infos(
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

    async def _request_instrument_deployment(
        self,
        *,
        specifications: tuple[InstrumentSpec, ...],
        append: bool = True,
        parallel: bool = True,
    ) -> tuple[InstrumentInfoProtocol, ...]:
        """Deploy instruments through quelware session APIs."""
        if not specifications:
            return ()

        client_factory = self._load_quelware_client_factory()
        instrument_entities = self._load_instrument_entities()

        specifications_by_port: dict[str, list[InstrumentSpec]] = defaultdict(list)
        for specification in specifications:
            specifications_by_port[specification.port_id].append(specification)
        max_attempts = 1 if append else QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS
        responses: dict[str, list[InstrumentInfoProtocol]] = {}
        for attempt in range(max_attempts):
            try:
                responses = await self._deploy_port_batches(
                    client_factory=client_factory,
                    port_batches=specifications_by_port,
                    instrument_entities=instrument_entities,
                    append=append,
                    parallel=parallel,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
                break
            except QuelwareSessionAcquisitionError:
                raise
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
        deployed: list[InstrumentInfoProtocol] = []
        for port_id, infos in responses.items():
            deployed.extend(
                self._select_port_instruments(
                    port_id, specifications_by_port[port_id], infos, append=append
                )
            )
        return tuple(deployed)

    async def _deploy_port_batches(
        self,
        *,
        client_factory: QuelwareClientFactory,
        port_batches: Mapping[str, Sequence[InstrumentSpec]],
        instrument_entities: QuelwareInstrumentEntities,
        append: bool,
        parallel: bool,
        attempt: int,
        max_attempts: int,
    ) -> dict[str, list[InstrumentInfoProtocol]]:
        """Return the raw response for each port written in the session."""
        async with client_factory(
            self._runtime_config.endpoint,
            self._runtime_config.port,
        ) as client:
            session_cm, session = await enter_quelware_session(
                client=client,
                resource_ids=list(port_batches),
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
                            for port_id, port_specifications in port_batches.items()
                        ),
                        return_exceptions=True,
                    )
                    responses = {}
                    for port_id, result in zip(port_batches, results, strict=True):
                        if isinstance(result, BaseException):
                            raise result  # noqa: TRY301
                        responses[port_id] = result
                    return responses
                return {
                    port_id: await self._deploy_port_batch(
                        session=session,
                        port_id=port_id,
                        port_specifications=port_specifications,
                        instrument_entities=instrument_entities,
                        append=append,
                    )
                    for port_id, port_specifications in port_batches.items()
                }
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
        port_specifications: Sequence[InstrumentSpec],
        instrument_entities: QuelwareInstrumentEntities,
        append: bool,
    ) -> list[InstrumentInfoProtocol]:
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

        return await session.deploy_instruments(
            port_id,
            definitions=definitions,
            append=append,
        )

    @staticmethod
    def _select_port_instruments(
        port_id: str,
        specifications: Sequence[InstrumentSpec],
        infos: Sequence[InstrumentInfoProtocol],
        *,
        append: bool,
    ) -> tuple[InstrumentInfoProtocol, ...]:
        """Validate one completed write and select the instruments requested on its port."""
        requested = {spec.alias for spec in specifications}
        deployed: dict[str, InstrumentInfoProtocol] = {}
        seen_ids: set[str] = set()
        for info in infos:
            if info.port_id != port_id:
                raise ValueError(
                    f"Deployment on {port_id!r} returned unexpected port {info.port_id!r}."
                )
            if info.id in seen_ids:
                raise ValueError(
                    f"Duplicate resource ID {info.id!r} in deployment response on {port_id!r}."
                )
            seen_ids.add(info.id)
            try:
                alias = InstrumentCache.alias_for(info)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid instrument in deployment response on {port_id!r}: {exc}"
                ) from exc
            if alias not in requested:
                if append:
                    # quelware may include existing instruments on this port.
                    continue
                raise ValueError(
                    f"Unexpected alias {alias!r} in replacement response on {port_id!r}."
                )
            if alias in deployed:
                raise ValueError(
                    f"Duplicate alias {alias!r} in deployment response on {port_id!r}."
                )
            deployed[alias] = info
        if missing := requested - deployed.keys():
            raise ValueError(
                f"Missing instruments in deployment response on {port_id!r}: {sorted(missing)}."
            )
        return tuple(deployed.values())

    def _load_quelware_client_factory(self) -> QuelwareClientFactory:
        """Import quelware client factory lazily."""
        return self._runtime_config.load_client_factory()

    @staticmethod
    def _load_instrument_entities() -> QuelwareInstrumentEntities:
        """Load the quelware deployment dependencies."""
        return load_quelware_instrument_entities()
