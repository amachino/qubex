"""Hardware-state reader for QuEL-3 runtime inspection."""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar, cast

from qubex.backend.quel3.infra.quelware_imports import Quel3ClientMode
from qubex.backend.quel3.interfaces import (
    QuelwareClientFactory,
    QuelwareClientProtocol,
)
from qubex.backend.quel3.managers.runtime_config import Quel3RuntimeConfig
from qubex.backend.quel3.models import (
    Quel3HardwareState,
    Quel3HardwareStateIssue,
    Quel3HardwareStateView,
    Quel3InstrumentState,
    Quel3PortDiagnostic,
    Quel3PortState,
    Quel3UnitState,
)
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

T = TypeVar("T")


@dataclass(frozen=True)
class _HardwareStateCollectionPlan:
    """Describe hardware-state sections required for one collection."""

    collect_ports: bool
    collect_instruments: bool
    collect_diagnostics: bool

    @classmethod
    def build(
        cls,
        *,
        view: Quel3HardwareStateView | None,
        include_diagnostics: bool,
    ) -> _HardwareStateCollectionPlan:
        """Build a collection plan for one optional rendered view."""
        if view is not None and view not in {
            "summary",
            "units",
            "ports",
            "instruments",
            "diagnostics",
            "all",
        }:
            raise ValueError(f"Unsupported QuEL-3 hardware-state view: {view!r}")
        collect_ports = view in (None, "summary", "ports", "diagnostics", "all")
        collect_instruments = view in (None, "summary", "instruments", "all")
        return cls(
            collect_ports=collect_ports or include_diagnostics,
            collect_instruments=collect_instruments,
            collect_diagnostics=include_diagnostics,
        )


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-hardware-state")
    return bridge.run(factory, timeout=timeout)


async def _resolve(value: T | Awaitable[T]) -> T:
    """Resolve a value that may be awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


class Quel3HardwareStateReader:
    """Collect read-only QuEL-3 hardware state through quelware APIs."""

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
        """Return quelware endpoint used for hardware state reads."""
        return self._runtime_config.endpoint

    @property
    def quelware_port(self) -> int:
        """Return quelware port used for hardware state reads."""
        return self._runtime_config.port

    @property
    def client_mode(self) -> Quel3ClientMode:
        """Return configured quelware client mode."""
        return self._runtime_config.client_mode_value

    @property
    def quelware_pat_path(self) -> str | None:
        """Return configured quelware personal access token path."""
        return self._runtime_config.pat_path

    def collect_state(
        self,
        *,
        unit_labels: Sequence[str] = (),
        port_ids: Sequence[str] = (),
        instrument_aliases: Sequence[str] = (),
        include_diagnostics: bool = False,
        parallel: bool = True,
        timeout_seconds: float | None = None,
        view: Quel3HardwareStateView | None = None,
    ) -> Quel3HardwareState:
        """
        Collect one structured QuEL-3 hardware-state snapshot.

        Filters are applied in order: `unit_labels`, then `port_ids`, then
        `instrument_aliases`. Local port IDs and aliases match every currently
        selected unit. Unit-qualified aliases narrow instruments, their related
        ports, and diagnostics to the qualified unit.

        Parameters
        ----------
        unit_labels : Sequence[str], optional
            Unit labels to inspect. Empty means all discovered units.
        port_ids : Sequence[str], optional
            Full port IDs such as `unit-a:tx_p01` or local IDs such as
            `tx_p01`.
        instrument_aliases : Sequence[str], optional
            Unit-qualified aliases such as `unit-a:Q00` or local aliases such
            as `Q00`.
        include_diagnostics : bool, optional
            Whether to collect diagnostic dumps for the final visible ports.
        parallel : bool, optional
            Whether resource reads should run concurrently.
        timeout_seconds : float | None, optional
            Timeout for the synchronous collection call.
        view : Quel3HardwareStateView | None, optional
            Rendered view whose unused hardware sections may be skipped. `None`
            collects the complete structured state.
        """
        timeout = (
            DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        return _run_async(
            lambda: self._collect_state(
                unit_labels=tuple(unit_labels),
                port_ids=tuple(port_ids),
                instrument_aliases=tuple(instrument_aliases),
                include_diagnostics=include_diagnostics,
                parallel=parallel,
                view=view,
            ),
            timeout=timeout,
        )

    def fetch_backend_settings_from_hardware(
        self,
        *,
        unit_labels_by_box_id: Mapping[str, str],
        parallel: bool | None = None,
    ) -> dict[str, dict]:
        """Fetch QuEL-3 backend settings by projecting hardware state."""
        if len(unit_labels_by_box_id) == 0:
            return {}
        state = self.collect_state(
            unit_labels=tuple(unit_labels_by_box_id.values()),
            include_diagnostics=False,
            parallel=True if parallel is None else parallel,
        )
        return self.project_backend_settings(
            state=state,
            unit_labels_by_box_id=unit_labels_by_box_id,
        )

    async def _collect_state(
        self,
        *,
        unit_labels: tuple[str, ...],
        port_ids: tuple[str, ...],
        instrument_aliases: tuple[str, ...],
        include_diagnostics: bool,
        parallel: bool,
        view: Quel3HardwareStateView | None,
    ) -> Quel3HardwareState:
        """Collect hardware state from one quelware client context."""
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        collection_plan = _HardwareStateCollectionPlan.build(
            view=view,
            include_diagnostics=include_diagnostics,
        )
        needs_instrument_lookup = collection_plan.collect_instruments or (
            collection_plan.collect_ports and bool(instrument_aliases)
        )
        client_factory = self._load_quelware_client_factory()
        async with client_factory(
            self._runtime_config.endpoint,
            self._runtime_config.port,
        ) as client:
            discovered_unit_labels = tuple(
                str(label) for label in await _resolve(client.list_unit_labels())
            )
            selected_unit_labels = tuple(unit_labels)
            visible_unit_labels = self._visible_unit_labels(
                discovered_unit_labels=discovered_unit_labels,
                selected_unit_labels=selected_unit_labels,
            )
            units = tuple(Quel3UnitState(label=label) for label in visible_unit_labels)
            resource_infos: list[object] = []
            if collection_plan.collect_ports or needs_instrument_lookup:
                resource_infos = [
                    resource_info
                    for resource_info in await _resolve(client.list_resource_infos())
                    if self._is_visible_resource(
                        resource_info=resource_info,
                        selected_unit_labels=selected_unit_labels,
                    )
                ]

            resolved_instruments: tuple[Quel3InstrumentState, ...] = ()
            instrument_issues: tuple[Quel3HardwareStateIssue, ...] = ()
            if needs_instrument_lookup:
                instrument_resource_infos = self._filter_instrument_resource_infos(
                    resource_infos=resource_infos,
                    selected_unit_labels=selected_unit_labels,
                    port_ids=port_ids,
                    instrument_aliases=instrument_aliases,
                )
                (
                    resolved_instruments,
                    instrument_issues,
                ) = await self._collect_instruments(
                    client=client,
                    resource_infos=instrument_resource_infos,
                    selected_unit_labels=selected_unit_labels,
                    parallel=parallel,
                )
            visible_instruments = self._filter_visible_instruments(
                instruments=resolved_instruments,
                port_ids=port_ids,
                instrument_aliases=instrument_aliases,
            )

            ports: tuple[Quel3PortState, ...] = ()
            port_issues: tuple[Quel3HardwareStateIssue, ...] = ()
            if collection_plan.collect_ports:
                port_resource_infos = self._filter_port_resource_infos(
                    resource_infos=resource_infos,
                    port_ids=port_ids,
                    instrument_aliases=instrument_aliases,
                    visible_instruments=visible_instruments,
                )
                ports, port_issues = await self._collect_ports(
                    client=client,
                    resource_infos=port_resource_infos,
                    selected_unit_labels=selected_unit_labels,
                    parallel=parallel,
                )
                ports = self._filter_visible_ports(
                    ports=ports,
                    visible_instruments=visible_instruments,
                    port_ids=port_ids,
                    instrument_aliases=instrument_aliases,
                )

            instruments = (
                visible_instruments if collection_plan.collect_instruments else ()
            )
            diagnostics, diagnostic_issues = await self._collect_diagnostics(
                client=client,
                ports=ports,
                include_diagnostics=collection_plan.collect_diagnostics,
                parallel=parallel,
            )

        issues = (
            *port_issues,
            *instrument_issues,
            *diagnostic_issues,
            *self._evaluate_state(
                selected_unit_labels=selected_unit_labels,
                discovered_unit_labels=discovered_unit_labels,
                units=units,
                ports=ports,
                instruments=instruments,
                evaluate_ports=collection_plan.collect_ports,
                evaluate_instruments=collection_plan.collect_instruments,
            ),
        )
        return Quel3HardwareState(
            generated_at=generated_at,
            endpoint=self._runtime_config.endpoint,
            port=self._runtime_config.port,
            selected_unit_labels=selected_unit_labels,
            units=units,
            ports=tuple(sorted(ports, key=lambda port: port.id)),
            instruments=tuple(
                sorted(
                    instruments,
                    key=lambda instrument: (
                        instrument.port_id,
                        instrument.normalized_alias or instrument.alias or "",
                        instrument.id,
                    ),
                )
            ),
            diagnostics=tuple(sorted(diagnostics, key=lambda item: item.port_id)),
            issues=issues,
        )

    async def _collect_ports(
        self,
        *,
        client: QuelwareClientProtocol,
        resource_infos: Sequence[object],
        selected_unit_labels: tuple[str, ...],
        parallel: bool,
    ) -> tuple[tuple[Quel3PortState, ...], tuple[Quel3HardwareStateIssue, ...]]:
        """Collect port states and preserve per-resource failures as issues."""
        port_resource_ids = tuple(
            self._resource_id(resource_info)
            for resource_info in resource_infos
            if self._category_name(getattr(resource_info, "category", None)) == "PORT"
        )

        async def _fetch(resource_id: str) -> Quel3PortState:
            return self._port_state(await _resolve(client.get_port_info(resource_id)))

        results = await self._collect_resource_results(
            resource_ids=port_resource_ids,
            fetch=_fetch,
            parallel=parallel,
        )
        ports: list[Quel3PortState] = []
        issues: list[Quel3HardwareStateIssue] = []
        for resource_id, result in zip(port_resource_ids, results, strict=True):
            if isinstance(result, BaseException):
                issues.append(
                    self._resource_issue(
                        operation="get_port_info",
                        resource_id=resource_id,
                        exc=result,
                    )
                )
                fallback = Quel3PortState(
                    id=resource_id,
                    unit_label=self._unit_label(resource_id),
                    role=None,
                )
                if self._is_selected_unit_label(
                    unit_label=fallback.unit_label,
                    selected_unit_labels=selected_unit_labels,
                ):
                    ports.append(fallback)
                continue
            if not self._is_selected_unit_label(
                unit_label=result.unit_label,
                selected_unit_labels=selected_unit_labels,
            ):
                continue
            ports.append(result)
        return tuple(ports), tuple(issues)

    async def _collect_instruments(
        self,
        *,
        client: QuelwareClientProtocol,
        resource_infos: Sequence[object],
        selected_unit_labels: tuple[str, ...],
        parallel: bool,
    ) -> tuple[tuple[Quel3InstrumentState, ...], tuple[Quel3HardwareStateIssue, ...]]:
        """Collect instrument states and preserve per-resource failures as issues."""
        instrument_resource_ids = tuple(
            self._resource_id(resource_info)
            for resource_info in resource_infos
            if self._category_name(getattr(resource_info, "category", None))
            == "INSTRUMENT"
        )

        async def _fetch(resource_id: str) -> Quel3InstrumentState:
            return self._instrument_state(
                await _resolve(client.get_instrument_info(resource_id))
            )

        results = await self._collect_resource_results(
            resource_ids=instrument_resource_ids,
            fetch=_fetch,
            parallel=parallel,
        )
        instruments: list[Quel3InstrumentState] = []
        issues: list[Quel3HardwareStateIssue] = []
        for resource_id, result in zip(instrument_resource_ids, results, strict=True):
            if isinstance(result, BaseException):
                issues.append(
                    self._resource_issue(
                        operation="get_instrument_info",
                        resource_id=resource_id,
                        exc=result,
                    )
                )
                continue
            if not self._is_selected_unit_label(
                unit_label=result.unit_label,
                selected_unit_labels=selected_unit_labels,
            ):
                continue
            instruments.append(result)
        return tuple(instruments), tuple(issues)

    async def _collect_diagnostics(
        self,
        *,
        client: QuelwareClientProtocol,
        ports: Sequence[Quel3PortState],
        include_diagnostics: bool,
        parallel: bool,
    ) -> tuple[tuple[Quel3PortDiagnostic, ...], tuple[Quel3HardwareStateIssue, ...]]:
        """Collect optional port diagnostic dumps."""
        if not include_diagnostics:
            return (), ()
        port_ids = tuple(port.id for port in ports)

        async def _fetch(port_id: str) -> Quel3PortDiagnostic:
            dump_port_state = client.dump_port_state
            text = await _resolve(dump_port_state(port_id))
            return Quel3PortDiagnostic(
                port_id=port_id,
                unit_label=self._unit_label(port_id),
                text=str(text),
            )

        results = await self._collect_resource_results(
            resource_ids=tuple(port_ids),
            fetch=_fetch,
            parallel=parallel,
        )
        diagnostics: list[Quel3PortDiagnostic] = []
        issues: list[Quel3HardwareStateIssue] = []
        for port_id, result in zip(port_ids, results, strict=True):
            if isinstance(result, BaseException):
                issues.append(
                    self._resource_issue(
                        operation="dump_port_state",
                        resource_id=port_id,
                        exc=result,
                    )
                )
                continue
            diagnostics.append(result)
        return tuple(diagnostics), tuple(issues)

    @classmethod
    def _filter_instrument_resource_infos(
        cls,
        *,
        resource_infos: Sequence[object],
        selected_unit_labels: tuple[str, ...],
        port_ids: tuple[str, ...],
        instrument_aliases: tuple[str, ...],
    ) -> tuple[object, ...]:
        """Filter known instrument resources to selector-qualified units."""
        unit_scopes: list[set[str]] = []
        if selected_unit_labels:
            unit_scopes.append(set(selected_unit_labels))
        unit_scopes.extend(
            {selector.split(":", maxsplit=1)[0] for selector in selectors}
            for selectors in (port_ids, instrument_aliases)
            if selectors and all(":" in selector for selector in selectors)
        )

        scoped_unit_labels: set[str] | None = None
        for unit_scope in unit_scopes:
            scoped_unit_labels = (
                unit_scope
                if scoped_unit_labels is None
                else scoped_unit_labels & unit_scope
            )
        if scoped_unit_labels is not None and not scoped_unit_labels:
            return ()

        candidates: list[object] = []
        for resource_info in resource_infos:
            if (
                cls._category_name(getattr(resource_info, "category", None))
                != "INSTRUMENT"
            ):
                continue
            resource_id = cls._resource_id(resource_info)
            if (
                scoped_unit_labels is not None
                and ":" in resource_id
                and cls._unit_label(resource_id) not in scoped_unit_labels
            ):
                continue
            candidates.append(resource_info)
        return tuple(candidates)

    @classmethod
    def _filter_port_resource_infos(
        cls,
        *,
        resource_infos: Sequence[object],
        port_ids: tuple[str, ...],
        instrument_aliases: tuple[str, ...],
        visible_instruments: Sequence[Quel3InstrumentState],
    ) -> tuple[object, ...]:
        """Filter known port resources before requesting their details."""
        related_port_ids = {instrument.port_id for instrument in visible_instruments}
        candidates: list[object] = []
        for resource_info in resource_infos:
            if cls._category_name(getattr(resource_info, "category", None)) != "PORT":
                continue
            resource_id = cls._resource_id(resource_info)
            is_qualified = ":" in resource_id
            if (
                port_ids
                and is_qualified
                and not cls._matches_port_filters(
                    port_id=resource_id,
                    port_ids=port_ids,
                )
            ):
                continue
            if instrument_aliases:
                if not related_port_ids:
                    continue
                if is_qualified and resource_id not in related_port_ids:
                    continue
            candidates.append(resource_info)
        return tuple(candidates)

    @classmethod
    def _filter_visible_instruments(
        cls,
        *,
        instruments: Sequence[Quel3InstrumentState],
        port_ids: tuple[str, ...],
        instrument_aliases: tuple[str, ...],
    ) -> tuple[Quel3InstrumentState, ...]:
        """Filter resolved instruments by selected port IDs and aliases."""
        visible_instruments = tuple(
            instrument
            for instrument in instruments
            if cls._matches_port_filters(
                port_id=instrument.port_id,
                port_ids=port_ids,
            )
        )
        visible_instruments = tuple(
            instrument
            for instrument in visible_instruments
            if cls._matches_alias_filters(
                instrument=instrument,
                instrument_aliases=instrument_aliases,
            )
        )
        return visible_instruments

    @classmethod
    def _filter_visible_ports(
        cls,
        *,
        ports: Sequence[Quel3PortState],
        visible_instruments: Sequence[Quel3InstrumentState],
        port_ids: tuple[str, ...],
        instrument_aliases: tuple[str, ...],
    ) -> tuple[Quel3PortState, ...]:
        """Filter resolved ports by selected IDs and related instruments."""
        visible_ports = tuple(
            port
            for port in ports
            if cls._matches_port_filters(port_id=port.id, port_ids=port_ids)
        )
        if instrument_aliases:
            visible_instrument_port_ids = {
                instrument.port_id for instrument in visible_instruments
            }
            visible_ports = tuple(
                port for port in visible_ports if port.id in visible_instrument_port_ids
            )
        return visible_ports

    @classmethod
    def _matches_port_filters(
        cls,
        *,
        port_id: str,
        port_ids: tuple[str, ...],
    ) -> bool:
        """Return whether one port ID matches local or full selected IDs."""
        if not port_ids:
            return True
        return any(
            cls._matches_port_id(port_id=port_id, selected_port_id=selected_port_id)
            for selected_port_id in port_ids
        )

    @classmethod
    def _matches_port_id(
        cls,
        *,
        port_id: str,
        selected_port_id: str,
    ) -> bool:
        """Return whether one port ID matches a local or full ID filter."""
        if ":" in selected_port_id:
            return port_id == selected_port_id
        return cls._local_resource_id(port_id) == selected_port_id

    @classmethod
    def _matches_alias_filters(
        cls,
        *,
        instrument: Quel3InstrumentState,
        instrument_aliases: tuple[str, ...],
    ) -> bool:
        """Return whether one instrument matches selected aliases."""
        if not instrument_aliases:
            return True
        return any(
            cls._matches_instrument_alias(
                instrument=instrument,
                selected_alias=selected_alias,
            )
            for selected_alias in instrument_aliases
        )

    @classmethod
    def _matches_instrument_alias(
        cls,
        *,
        instrument: Quel3InstrumentState,
        selected_alias: str,
    ) -> bool:
        """Return whether one instrument matches a local or unit-qualified alias."""
        if ":" in selected_alias:
            unit_label, local_alias = selected_alias.split(":", maxsplit=1)
            return (
                instrument.unit_label == unit_label
                and local_alias in cls._local_instrument_aliases(instrument)
            )
        return selected_alias in cls._local_instrument_aliases(instrument)

    @classmethod
    def _local_instrument_aliases(
        cls,
        instrument: Quel3InstrumentState,
    ) -> tuple[str, ...]:
        """Return local aliases that can match one instrument."""
        aliases: list[str] = []
        if instrument.normalized_alias is not None:
            aliases.append(instrument.normalized_alias)
        if instrument.alias is not None:
            aliases.append(instrument.alias)
            prefix = f"{instrument.unit_label}:"
            if instrument.alias.startswith(prefix):
                aliases.append(instrument.alias.removeprefix(prefix))
        return tuple(dict.fromkeys(aliases))

    @staticmethod
    async def _collect_resource_results(
        *,
        resource_ids: tuple[str, ...],
        fetch: Callable[[str], Awaitable[T]],
        parallel: bool,
    ) -> tuple[T | BaseException, ...]:
        """Collect resource results in parallel or serial order."""

        async def _fetch_result(resource_id: str) -> T | BaseException:
            try:
                return await fetch(resource_id)
            except Exception as exc:
                return exc

        if parallel:
            return tuple(
                await asyncio.gather(
                    *(_fetch_result(resource_id) for resource_id in resource_ids),
                    return_exceptions=True,
                )
            )

        return tuple([await _fetch_result(resource_id) for resource_id in resource_ids])

    @classmethod
    def project_backend_settings(
        cls,
        *,
        state: Quel3HardwareState,
        unit_labels_by_box_id: Mapping[str, str],
    ) -> dict[str, dict]:
        """Project hardware state into QuEL-3 backend-settings cache data."""
        settings: dict[str, dict] = {
            box_id: {"instruments": {}} for box_id in unit_labels_by_box_id
        }
        box_ids_by_unit_label: dict[str, list[str]] = defaultdict(list)
        for box_id, unit_label in unit_labels_by_box_id.items():
            box_ids_by_unit_label[unit_label].append(box_id)

        for instrument in state.instruments:
            alias = instrument.normalized_alias or instrument.alias
            if alias is None:
                continue
            for box_id in box_ids_by_unit_label.get(instrument.unit_label, ()):
                settings[box_id]["instruments"][alias] = (
                    cls._backend_settings_instrument(instrument)
                )
        return settings

    @staticmethod
    def _backend_settings_instrument(instrument: Quel3InstrumentState) -> dict:
        """Return backend-settings data for one instrument state."""
        definition: dict[str, object] = {
            "alias": instrument.alias or instrument.normalized_alias or "",
            "role": instrument.role,
        }
        if instrument.mode is not None:
            definition["mode"] = instrument.mode
        profile: dict[str, float] = {}
        if instrument.frequency_range_min_hz is not None:
            profile["frequency_range_min"] = instrument.frequency_range_min_hz
        if instrument.frequency_range_max_hz is not None:
            profile["frequency_range_max"] = instrument.frequency_range_max_hz
        if profile:
            definition["profile"] = profile
        return {
            "resource_id": instrument.id,
            "port_id": instrument.port_id,
            "role": instrument.role,
            "definition": definition,
        }

    @classmethod
    def _evaluate_state(
        cls,
        *,
        selected_unit_labels: tuple[str, ...],
        discovered_unit_labels: tuple[str, ...],
        units: Sequence[Quel3UnitState],
        ports: Sequence[Quel3PortState],
        instruments: Sequence[Quel3InstrumentState],
        evaluate_ports: bool,
        evaluate_instruments: bool,
    ) -> tuple[Quel3HardwareStateIssue, ...]:
        """Evaluate derived health issues for a hardware-state snapshot."""
        issues: list[Quel3HardwareStateIssue] = []
        discovered = set(discovered_unit_labels)
        missing_units = sorted(set(selected_unit_labels) - discovered)
        if missing_units:
            issues.append(
                Quel3HardwareStateIssue(
                    severity="error",
                    code="UNIT_NOT_FOUND",
                    message="Selected QuEL-3 units were not discovered.",
                    detail=", ".join(missing_units),
                )
            )
        if len(units) == 0:
            issues.append(
                Quel3HardwareStateIssue(
                    severity="error",
                    code="NO_UNITS",
                    message="No QuEL-3 units were discovered.",
                )
            )
        if evaluate_ports and len(ports) == 0:
            issues.append(
                Quel3HardwareStateIssue(
                    severity="warning",
                    code="NO_PORTS",
                    message="No QuEL-3 port resources were found.",
                )
            )
        if evaluate_instruments and len(instruments) == 0:
            issues.append(
                Quel3HardwareStateIssue(
                    severity="warning",
                    code="NO_INSTRUMENTS",
                    message="No QuEL-3 instrument resources were found.",
                )
            )
        if evaluate_ports:
            issues.extend(cls._port_dependency_issues(ports))
        if evaluate_instruments:
            issues.extend(
                cls._instrument_issues(
                    instruments=instruments,
                    ports=ports,
                    check_port_references=evaluate_ports,
                )
            )
        return tuple(issues)

    @staticmethod
    def _port_dependency_issues(
        ports: Sequence[Quel3PortState],
    ) -> list[Quel3HardwareStateIssue]:
        """Return issues for missing port dependency references."""
        issues: list[Quel3HardwareStateIssue] = []
        port_ids = {port.id for port in ports}
        for port in ports:
            missing = [
                resource_id
                for resource_id in port.depends_on
                if resource_id not in port_ids
            ]
            if missing:
                issues.append(
                    Quel3HardwareStateIssue(
                        severity="warning",
                        code="UNKNOWN_PORT_DEPENDENCY",
                        message="Port references resources not listed as ports.",
                        detail=", ".join(missing),
                        resource_id=port.id,
                    )
                )
        return issues

    @classmethod
    def _instrument_issues(
        cls,
        *,
        instruments: Sequence[Quel3InstrumentState],
        ports: Sequence[Quel3PortState],
        check_port_references: bool,
    ) -> list[Quel3HardwareStateIssue]:
        """Return issues for instrument-port and definition consistency."""
        issues: list[Quel3HardwareStateIssue] = []
        port_ids = {port.id for port in ports}
        aliases = Counter(
            (instrument.unit_label, instrument.normalized_alias or instrument.alias)
            for instrument in instruments
            if instrument.normalized_alias or instrument.alias
        )
        for (unit_label, alias), count in sorted(aliases.items()):
            if count > 1:
                issues.append(
                    Quel3HardwareStateIssue(
                        severity="warning",
                        code="DUPLICATE_INSTRUMENT_ALIAS",
                        message=(
                            f"Instrument alias {alias} appears {count} times "
                            f"in unit {unit_label}."
                        ),
                    )
                )

        for instrument in instruments:
            if check_port_references and instrument.port_id not in port_ids:
                issues.append(
                    Quel3HardwareStateIssue(
                        severity="error",
                        code="ORPHAN_INSTRUMENT",
                        message="Instrument points to an unknown port.",
                        detail=instrument.port_id,
                        resource_id=instrument.id,
                    )
                )
            if not instrument.alias:
                issues.append(
                    Quel3HardwareStateIssue(
                        severity="warning",
                        code="EMPTY_INSTRUMENT_ALIAS",
                        message="Instrument has no alias.",
                        resource_id=instrument.id,
                    )
                )
            issues.extend(cls._frequency_issues(instrument))
        return issues

    @staticmethod
    def _frequency_issues(
        instrument: Quel3InstrumentState,
    ) -> list[Quel3HardwareStateIssue]:
        """Return frequency-range issues for one instrument."""
        lower = instrument.frequency_range_min_hz
        upper = instrument.frequency_range_max_hz
        if lower is None or upper is None:
            return [
                Quel3HardwareStateIssue(
                    severity="warning",
                    code="MISSING_FREQUENCY_RANGE",
                    message="Instrument has no complete frequency range.",
                    resource_id=instrument.id,
                )
            ]
        if lower >= upper:
            return [
                Quel3HardwareStateIssue(
                    severity="error",
                    code="INVALID_FREQUENCY_RANGE",
                    message="Instrument has an invalid frequency range.",
                    detail=f"{lower} >= {upper}",
                    resource_id=instrument.id,
                )
            ]
        return []

    @classmethod
    def _port_state(cls, port_info: object) -> Quel3PortState:
        """Build one port state from a quelware port-info object."""
        port_info_obj = cast(Any, port_info)
        port_id = str(port_info_obj.id)
        depends_on = tuple(str(item) for item in getattr(port_info, "depends_on", ()))
        return Quel3PortState(
            id=port_id,
            unit_label=cls._unit_label(port_id),
            role=cls._enum_name(getattr(port_info, "role", None)),
            depends_on=depends_on,
        )

    @classmethod
    def _instrument_state(cls, instrument_info: object) -> Quel3InstrumentState:
        """Build one instrument state from a quelware instrument-info object."""
        instrument_info_obj = cast(Any, instrument_info)
        instrument_id = str(instrument_info_obj.id)
        port_id = str(instrument_info_obj.port_id)
        definition = instrument_info_obj.definition
        config = getattr(instrument_info, "config", None)
        profile = getattr(definition, "profile", None)
        alias = cls._string_or_none(getattr(definition, "alias", None))
        return Quel3InstrumentState(
            id=instrument_id,
            unit_label=cls._unit_label(port_id),
            port_id=port_id,
            alias=alias,
            normalized_alias=cls._normalize_alias(alias=alias, port_id=port_id),
            role=cls._enum_name(getattr(definition, "role", None)),
            mode=cls._enum_name(getattr(definition, "mode", None)),
            frequency_range_min_hz=cls._float_or_none(
                getattr(profile, "frequency_range_min", None)
            ),
            frequency_range_max_hz=cls._float_or_none(
                getattr(profile, "frequency_range_max", None)
            ),
            sampling_period_fs=cls._int_or_none(
                getattr(config, "sampling_period_fs", None)
            ),
            bitdepth=cls._int_or_none(getattr(config, "bitdepth", None)),
            timeline_step_samples=cls._int_or_none(
                getattr(config, "timeline_step_samples", None)
            ),
            samples_per_tick=cls._int_or_none(
                getattr(config, "samples_per_tick", None)
            ),
        )

    @staticmethod
    def _resource_issue(
        *,
        operation: str,
        resource_id: str,
        exc: BaseException,
    ) -> Quel3HardwareStateIssue:
        """Return an issue describing one failed resource fetch."""
        return Quel3HardwareStateIssue(
            severity="error",
            code="RESOURCE_FETCH_ERROR",
            message=f"{operation} failed.",
            detail=str(exc),
            resource_id=resource_id,
        )

    @classmethod
    def _is_visible_resource(
        cls,
        *,
        resource_info: object,
        selected_unit_labels: tuple[str, ...],
    ) -> bool:
        """Return whether a resource belongs to selected units."""
        if not selected_unit_labels:
            return True
        resource_id = cls._resource_id(resource_info)
        if ":" not in resource_id:
            return True
        return cls._unit_label(resource_id) in set(selected_unit_labels)

    @staticmethod
    def _is_selected_unit_label(
        *,
        unit_label: str,
        selected_unit_labels: tuple[str, ...],
    ) -> bool:
        """Return whether a resolved state belongs to selected units."""
        return not selected_unit_labels or unit_label in set(selected_unit_labels)

    @staticmethod
    def _resource_id(resource_info: object) -> str:
        """Return string resource ID from a resource-info object."""
        return str(cast(Any, resource_info).id)

    @staticmethod
    def _visible_unit_labels(
        *,
        discovered_unit_labels: tuple[str, ...],
        selected_unit_labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return discovered units visible under one selection."""
        if not selected_unit_labels:
            return discovered_unit_labels
        selected = set(selected_unit_labels)
        return tuple(label for label in discovered_unit_labels if label in selected)

    @staticmethod
    def _unit_label(resource_id: str) -> str:
        """Return the unit-label prefix from one resource ID."""
        return resource_id.split(":", maxsplit=1)[0]

    @staticmethod
    def _local_resource_id(resource_id: str) -> str:
        """Return the local suffix from a unit-qualified resource ID."""
        if ":" not in resource_id:
            return resource_id
        return resource_id.split(":", maxsplit=1)[1]

    @staticmethod
    def _category_name(category: object) -> str:
        """Normalize one category enum-like value to its name."""
        category_name = getattr(category, "name", None)
        text = category_name if isinstance(category_name, str) else str(category)
        return text.rsplit(".", maxsplit=1)[-1]

    @staticmethod
    def _enum_name(value: object) -> str | None:
        """Normalize one enum-like value to its name."""
        if value is None:
            return None
        enum_name = getattr(value, "name", None)
        if isinstance(enum_name, str):
            return enum_name
        return str(value)

    @classmethod
    def _normalize_alias(cls, *, alias: str | None, port_id: str) -> str | None:
        """Strip the unit-label prefix from an instrument alias when present."""
        if alias is None:
            return None
        stripped = alias.strip()
        prefix = f"{cls._unit_label(port_id)}:"
        if stripped.startswith(prefix):
            stripped = stripped.removeprefix(prefix).strip()
        return stripped or None

    @staticmethod
    def _string_or_none(value: object) -> str | None:
        """Return stripped text or `None`."""
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        """Return float for numeric values except booleans."""
        if isinstance(value, bool):
            return None
        return float(value) if isinstance(value, int | float) else None

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        """Return int for integer values except booleans."""
        if isinstance(value, bool):
            return None
        return value if isinstance(value, int) else None

    def _load_quelware_client_factory(self) -> QuelwareClientFactory:
        """Import quelware client factory lazily."""
        return self._runtime_config.load_client_factory()
