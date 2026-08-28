"""Execution manager for QuEL-3 backend controller."""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass, replace
from typing import TypeGuard, TypeVar, cast

import numpy as np

from qubex.backend.quel3.builders.sequencer_builder import Quel3SequencerBuilder
from qubex.backend.quel3.infra.quelware_imports import Quel3ClientMode
from qubex.backend.quel3.instrument_groups import (
    build_transmitter_aliases,
    is_transmitter_role,
    split_transmitter_alias,
)
from qubex.backend.quel3.interfaces import (
    CaptureModeNamespaceProtocol,
    CaptureModeProtocol,
    DirectiveProtocol,
    InstrumentDriverFactory,
    InstrumentDriverProtocol,
    InstrumentInfoProtocol,
    InstrumentResolverFactory,
    InstrumentResolverProtocol,
    IqWaveformResultProtocol,
    QuelwareClientFactory,
    ResourceIdProtocol,
    ResultContainerProtocol,
    SequencerProtocol,
    SessionProtocol,
    SetCaptureModeFactory,
    SetFrequencyFactory,
)
from qubex.backend.quel3.managers.runtime_config import Quel3RuntimeConfig
from qubex.backend.quel3.managers.session_manager import Quel3SessionManager
from qubex.backend.quel3.managers.session_workarounds import (
    QuelwareSessionError,
    quelware_exception_summary,
)
from qubex.backend.quel3.models import (
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3WaveformEvent,
)
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

T = TypeVar("T")

QUEL3_SESSION_REQUEST_MAX_ATTEMPTS = 4
QUEL3_SESSION_TRIGGER_WAIT_MS: int | None = None

logger = logging.getLogger(__name__)


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-execution")
    return bridge.run(factory, timeout=timeout)


def _has_iq_array(value: object) -> TypeGuard[IqWaveformResultProtocol]:
    """Return whether one runtime value exposes waveform IQ samples."""
    return hasattr(value, "iq_array")


@dataclass(frozen=True)
class _PayloadExecutionSession:
    """Session-bound QuEL-3 state for one resolved payload."""

    session: SessionProtocol
    alias_to_resource_id: dict[str, ResourceIdProtocol]
    alias_to_driver: dict[str, InstrumentDriverProtocol]
    capture_sampling_period_ns: float | None


@dataclass(frozen=True)
class _PayloadExecutionPlan:
    """Resolved payload and runtime aliases required for one execution."""

    resolved_payload: Quel3ExecutionPayload


@dataclass(frozen=True)
class _QuelwareExecutionApi:
    """Lazy-loaded quelware API symbols needed for fixed-timeline execution."""

    client_factory: QuelwareClientFactory
    instrument_resolver_factory: InstrumentResolverFactory
    sequencer_factory: Callable[..., SequencerProtocol]
    fixed_timeline_driver_factory: InstrumentDriverFactory
    set_frequency_directive_factory: SetFrequencyFactory
    set_capture_mode_directive_factory: SetCaptureModeFactory
    capture_mode_namespace: CaptureModeNamespaceProtocol

    def build_capture_mode_directive(
        self,
        capture_mode: Quel3CaptureMode,
    ) -> DirectiveProtocol:
        """Build one capture-mode directive from payload capture mode."""
        if capture_mode is Quel3CaptureMode.UNSPECIFIED:
            raise ValueError(f"Unsupported capture mode: {capture_mode}.")
        try:
            mode = cast(
                CaptureModeProtocol,
                getattr(self.capture_mode_namespace, capture_mode.name),
            )
        except AttributeError as exc:
            raise RuntimeError(
                "quelware runtime does not expose required "
                f"`CaptureMode.{capture_mode.name}`."
            ) from exc
        return self.set_capture_mode_directive_factory(mode=mode)


class Quel3ExecutionManager:
    """Handle backend execution entrypoints for QuEL-3 controller."""

    def __init__(
        self,
        *,
        runtime_config: Quel3RuntimeConfig | None = None,
        sampling_period_ns: float,
        capture_decimation_factor: int,
        session_manager: Quel3SessionManager | None = None,
    ) -> None:
        self._runtime_config = runtime_config or Quel3RuntimeConfig()
        self._sampling_period_ns = sampling_period_ns
        self._capture_decimation_factor = capture_decimation_factor
        self._sequencer_builder = Quel3SequencerBuilder()
        self._session_manager = (
            session_manager
            if session_manager is not None
            else Quel3SessionManager(
                runtime_config=self._runtime_config,
            )
        )
        self._instrument_resolver: InstrumentResolverProtocol | None = None

    @property
    def runtime_config(self) -> Quel3RuntimeConfig:
        """Return the shared quelware runtime config."""
        return self._runtime_config

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint used for execution."""
        return self._runtime_config.endpoint

    @property
    def quelware_port(self) -> int | None:
        """Return quelware port used for execution."""
        return self._runtime_config.port

    @property
    def sampling_period_ns(self) -> float:
        """Return backend sampling period in ns."""
        return self._sampling_period_ns

    @property
    def client_mode(self) -> Quel3ClientMode:
        """Return configured quelware client mode."""
        return self._runtime_config.client_mode_value

    @property
    def quelware_pat_path(self) -> str | None:
        """Return configured quelware personal access token path."""
        return self._runtime_config.pat_path

    def invalidate_instrument_resolver(self) -> None:
        """Discard cached instrument resolution state."""
        self._instrument_resolver = None

    def execute_sync(
        self,
        *,
        request: object,
        parallel: bool = True,
    ) -> Quel3BackendExecutionResult:
        """Execute a QuEL-3 backend request synchronously."""
        return _run_async(
            lambda: self.execute_async(request=request, parallel=parallel)
        )

    async def execute_async(
        self,
        *,
        request: object,
        parallel: bool = True,
    ) -> Quel3BackendExecutionResult:
        """Execute a QuEL-3 backend request asynchronously."""
        return await self.execute(request=request, parallel=parallel)

    async def execute_batch_async(
        self,
        *,
        requests: list[object] | tuple[object, ...],
        parallel: bool = True,
    ) -> list[Quel3BackendExecutionResult]:
        """Execute multiple QuEL-3 backend requests as one resolved batch."""
        payloads: list[Quel3ExecutionPayload] = []
        for request in requests:
            payload = getattr(request, "payload", None)
            if not isinstance(payload, Quel3ExecutionPayload):
                raise TypeError(
                    "Quel3ExecutionManager expects request payload to be `Quel3ExecutionPayload`."
                )
            payloads.append(payload)
        if len(payloads) == 0:
            return []

        try:
            quelware_api = self._load_quelware_api()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc
        payload_plans = [
            self._prepare_payload_execution_plan(payload=payload)
            for payload in payloads
        ]

        return await self._execute_batch_once(
            payload_plans=payload_plans,
            quelware_api=quelware_api,
            parallel=parallel,
        )

    async def execute(
        self,
        *,
        request: object,
        parallel: bool = True,
    ) -> Quel3BackendExecutionResult:
        """
        Execute a QuEL-3 backend request asynchronously.

        Parameters
        ----------
        request : object
            Backend execution request with `payload`.
        parallel : bool, optional
            Whether to parallelize per-instrument phases, by default `True`.
        """
        payload = getattr(request, "payload", None)
        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3ExecutionManager expects request payload to be `Quel3ExecutionPayload`."
            )
        if len(payload.fixed_timelines) == 0:
            raise ValueError("Quel3ExecutionPayload must include fixed timelines.")

        try:
            quelware_api = self._load_quelware_api()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc
        payload_plan = self._prepare_payload_execution_plan(payload=payload)

        results = await self._execute_batch_once(
            payload_plans=[payload_plan],
            quelware_api=quelware_api,
            parallel=parallel,
        )
        return results[0]

    async def _close_session_manager_after_attempt(self) -> None:
        """Close session state without masking request success or failure."""
        session_token = self._active_session_token()
        try:
            await self._session_manager.close()
        except Exception as exc:
            logger.warning(
                "QuEL-3 quelware session cleanup failed; session_token=%s; cause=%s",
                session_token,
                quelware_exception_summary(exc),
            )

    def _active_session_token(self) -> str:
        """Return the currently open session token for diagnostics."""
        return self._session_manager.session_token or "<unavailable>"

    async def _execute_batch_once(
        self,
        *,
        payload_plans: list[_PayloadExecutionPlan],
        quelware_api: _QuelwareExecutionApi,
        parallel: bool,
    ) -> list[Quel3BackendExecutionResult]:
        """Execute one payload batch with per-payload quelware sessions."""
        # Batch flow:
        # 1. Lazily open one quelware client and refresh the resolver when invalidated.
        # 2. Resolve instrument info for each precomputed payload plan.
        # 3. Reopen a per-payload session for the resolved instrument resource IDs.
        # 4. Execute the resolved payload and collect results.
        # 5. Retry only the failed payload after transient session request failures.
        resolver = self._instrument_resolver
        if resolver is None:
            resolver = quelware_api.instrument_resolver_factory()

        try:
            return [
                await self._execute_payload_plan_with_session_request_retry(
                    payload_plan=payload_plan,
                    resolver=resolver,
                    quelware_api=quelware_api,
                    parallel=parallel,
                )
                for payload_plan in payload_plans
            ]
        finally:
            await self._close_session_manager_after_attempt()

    async def _execute_payload_plan_with_session_request_retry(
        self,
        *,
        payload_plan: _PayloadExecutionPlan,
        resolver: InstrumentResolverProtocol,
        quelware_api: _QuelwareExecutionApi,
        parallel: bool,
    ) -> Quel3BackendExecutionResult:
        """Execute one payload plan, recreating the session after failures."""
        max_attempts = max(1, int(QUEL3_SESSION_REQUEST_MAX_ATTEMPTS))
        for attempt in range(max_attempts):
            attempt_number = attempt + 1
            try:
                await self._ensure_batch_client_ready(
                    resolver=resolver,
                    quelware_api=quelware_api,
                    force_resolver_refresh=attempt > 0,
                )
                resolved_payload, alias_to_instrument_info = (
                    self._resolve_payload_instruments(
                        payload=payload_plan.resolved_payload,
                        resolver=resolver,
                    )
                )
                aliases = tuple(sorted(resolved_payload.fixed_timelines))
                aliases_with_captures = frozenset(
                    alias
                    for alias, timeline in resolved_payload.fixed_timelines.items()
                    if len(timeline.capture_windows) > 0
                )
                session_state = await self._open_payload_execution_session(
                    alias_to_instrument_info=alias_to_instrument_info,
                    aliases=aliases,
                    aliases_with_captures=aliases_with_captures,
                    quelware_api=quelware_api,
                )
                return await self._execute_resolved_payload(
                    payload=resolved_payload,
                    session_state=session_state,
                    quelware_api=quelware_api,
                    parallel=parallel,
                )
            except Exception as exc:
                session_token = (
                    exc.session_token
                    if isinstance(exc, QuelwareSessionError)
                    else self._active_session_token()
                )
                await self._close_session_manager_after_attempt()
                if attempt_number >= max_attempts:
                    if isinstance(exc, QuelwareSessionError):
                        raise
                    raise QuelwareSessionError(
                        "QuEL-3 quelware session request failed after retries",
                        session_token=session_token,
                        cause=exc,
                    ) from exc
                logger.warning(
                    "QuEL-3 quelware session request failed; session_token=%s; "
                    "attempt=%d/%d; retrying with a fresh session; cause=%s",
                    session_token,
                    attempt_number,
                    max_attempts,
                    quelware_exception_summary(exc),
                )
        raise RuntimeError("unreachable QuEL-3 session request retry state")

    async def _ensure_batch_client_ready(
        self,
        *,
        resolver: InstrumentResolverProtocol,
        quelware_api: _QuelwareExecutionApi,
        force_resolver_refresh: bool,
    ) -> None:
        """Open the batch client and refresh resolver state when needed."""
        if self._session_manager.is_open:
            return
        await self._session_manager.open(client_factory=quelware_api.client_factory)
        if self._instrument_resolver is not resolver or force_resolver_refresh:
            await resolver.refresh(self._session_manager.client)
            self._instrument_resolver = resolver

    async def _open_payload_execution_session(
        self,
        *,
        alias_to_instrument_info: dict[str, InstrumentInfoProtocol],
        aliases: Sequence[str],
        aliases_with_captures: Collection[str],
        quelware_api: _QuelwareExecutionApi,
    ) -> _PayloadExecutionSession:
        """Open a payload session and rebuild session-bound drivers."""
        instrument_resource_ids: list[ResourceIdProtocol] = []
        for alias in aliases:
            try:
                resource_id = alias_to_instrument_info[alias].id
            except KeyError as exc:
                raise ValueError(
                    f"Instrument resource ID is not resolved for alias `{alias}`."
                ) from exc
            if len(resource_id) == 0:
                raise ValueError(
                    f"Instrument resource ID is not resolved for alias `{alias}`."
                )
            instrument_resource_ids.append(resource_id)
        alias_to_resource_id = dict(zip(aliases, instrument_resource_ids, strict=True))
        session_token = self._active_session_token()
        try:
            session = await self._session_manager.reopen_session(
                tuple(instrument_resource_ids),
            )
        except QuelwareSessionError:
            raise
        except Exception as exc:
            raise QuelwareSessionError(
                "QuEL-3 quelware session reopen failed",
                session_token=session_token,
                cause=exc,
            ) from exc
        if session is None:
            raise RuntimeError(
                "QuEL-3 session reopen did not return an execution session."
            )

        alias_to_driver: dict[str, InstrumentDriverProtocol] = {}
        for alias in aliases:
            instrument_info = alias_to_instrument_info[alias]
            try:
                driver = quelware_api.fixed_timeline_driver_factory(
                    session,
                    instrument_info,
                )
            except Exception as exc:
                raise RuntimeError(
                    "QuEL-3 fixed-timeline driver creation failed "
                    f"for instrument alias `{alias}`."
                ) from exc
            alias_to_driver[alias] = driver
        capture_sampling_period_ns = self._resolve_capture_sampling_period_ns(
            aliases_with_captures=aliases_with_captures,
            alias_to_driver=alias_to_driver,
            aliases=aliases,
        )
        return _PayloadExecutionSession(
            session=session,
            alias_to_resource_id=alias_to_resource_id,
            alias_to_driver=alias_to_driver,
            capture_sampling_period_ns=capture_sampling_period_ns,
        )

    @staticmethod
    def _resolve_capture_sampling_period_ns(
        *,
        aliases_with_captures: Collection[str],
        alias_to_driver: dict[str, InstrumentDriverProtocol],
        aliases: Sequence[str],
    ) -> float | None:
        """Resolve the capture sampling period for one resolved payload."""
        capture_sampling_period_ns: float | None = None
        for alias in aliases:
            if alias not in aliases_with_captures:
                continue
            driver = alias_to_driver[alias]
            sampling_period_fs = driver.instrument_config.sampling_period_fs
            alias_sampling_period_ns = sampling_period_fs / 1e6
            if capture_sampling_period_ns is None:
                capture_sampling_period_ns = alias_sampling_period_ns
            elif not np.isclose(
                capture_sampling_period_ns,
                alias_sampling_period_ns,
            ):
                raise ValueError("Capture aliases must agree on sampling period.")
        return capture_sampling_period_ns

    async def _execute_resolved_payload(
        self,
        *,
        payload: Quel3ExecutionPayload,
        session_state: _PayloadExecutionSession,
        quelware_api: _QuelwareExecutionApi,
        parallel: bool,
    ) -> Quel3BackendExecutionResult:
        """Execute one payload using an already-open payload session."""
        aliases = sorted(payload.fixed_timelines.keys())
        alias_bindings: dict[str, tuple[int, int]] = {}
        instrument_resource_ids: list[ResourceIdProtocol] = []
        for alias in aliases:
            driver = session_state.alias_to_driver[alias]
            sampling_period_fs = driver.instrument_config.sampling_period_fs
            timeline_step_samples = driver.instrument_config.timeline_step_samples
            alias_bindings[alias] = (
                sampling_period_fs,
                timeline_step_samples,
            )
            instrument_resource_ids.append(session_state.alias_to_resource_id[alias])

        sequencer = self._sequencer_builder.build(
            payload=payload,
            sequencer_factory=quelware_api.sequencer_factory,
            default_sampling_period_ns=self._sampling_period_ns,
            alias_bindings=alias_bindings,
        )

        alias_to_directives: dict[str, list[DirectiveProtocol]] = {}
        for alias in aliases:
            directives: list[DirectiveProtocol] = []
            frequency_hz = payload.fixed_timelines[alias].frequency_hz
            if frequency_hz is not None:
                directives.append(
                    quelware_api.set_frequency_directive_factory(hz=frequency_hz)
                )
            capture_mode_directive = quelware_api.build_capture_mode_directive(
                payload.capture_mode
            )
            if capture_mode_directive is not None:
                directives.append(capture_mode_directive)
            directives.append(sequencer.export_set_fixed_timeline_directive(alias))
            alias_to_directives[alias] = directives

        drivers = tuple(session_state.alias_to_driver.values())
        # Initializing drivers in parallel is currently unreliable, so initialize
        # them serially as a workaround.
        for driver in drivers:
            await driver.initialize()

        if parallel:
            await asyncio.gather(
                *(
                    session_state.alias_to_driver[alias].apply(
                        alias_to_directives[alias]
                    )
                    for alias in aliases
                )
            )
        else:
            for alias in aliases:
                await session_state.alias_to_driver[alias].apply(
                    alias_to_directives[alias]
                )

        shot_samples = {
            alias: {window.name: [] for window in timeline.capture_windows}
            for alias, timeline in payload.fixed_timelines.items()
        }
        await session_state.session.trigger(
            instrument_ids=instrument_resource_ids,
            wait_ms=QUEL3_SESSION_TRIGGER_WAIT_MS,
        )
        if parallel:
            results = await asyncio.gather(
                *(
                    session_state.alias_to_driver[alias].wait_for_result()
                    for alias in aliases
                )
            )
            alias_results = dict(zip(aliases, results, strict=True))
        else:
            alias_results: dict[str, ResultContainerProtocol] = {}
            for alias in aliases:
                alias_results[alias] = await session_state.alias_to_driver[
                    alias
                ].wait_for_result()

        for alias, timeline in payload.fixed_timelines.items():
            result = alias_results[alias]
            for window in timeline.capture_windows:
                window_key = window.name
                capture_samples = self._extract_capture_samples(
                    result,
                    window_key,
                    capture_mode=payload.capture_mode,
                )
                if capture_samples is None:
                    continue
                shot_samples[alias][window.name].append(capture_samples)

        return self._build_measurement_result(
            payload=payload,
            shot_samples=shot_samples,
            capture_sampling_period_ns=session_state.capture_sampling_period_ns,
            backend_sampling_period_ns=self._sampling_period_ns,
            capture_decimation_factor=self._capture_decimation_factor,
        )

    @classmethod
    def _prepare_payload_execution_plan(
        cls,
        *,
        payload: Quel3ExecutionPayload,
    ) -> _PayloadExecutionPlan:
        """Resolve and validate one payload before opening a session."""
        runnable_payload = cls._filter_runnable_payload(payload)
        resolved_payload = cls._resolve_payload(payload=runnable_payload)
        return _PayloadExecutionPlan(
            resolved_payload=resolved_payload,
        )

    @classmethod
    def _resolve_payload_instruments(
        cls,
        *,
        payload: Quel3ExecutionPayload,
        resolver: InstrumentResolverProtocol,
    ) -> tuple[Quel3ExecutionPayload, dict[str, InstrumentInfoProtocol]]:
        """Expand resolved transmitter timelines to four physical aliases."""
        physical_timelines: dict[str, Quel3FixedTimeline] = {}
        alias_to_instrument_info: dict[str, InstrumentInfoProtocol] = {}
        for requested_alias, timeline in payload.fixed_timelines.items():
            resolved_group = cls._resolve_instrument_group(
                resolver=resolver,
                requested_alias=requested_alias,
            )
            for physical_alias, instrument_info in resolved_group.items():
                physical_timelines[physical_alias] = timeline
                alias_to_instrument_info[physical_alias] = instrument_info
        return (
            replace(payload, fixed_timelines=physical_timelines),
            alias_to_instrument_info,
        )

    @classmethod
    def _resolve_instrument_group(
        cls,
        *,
        resolver: InstrumentResolverProtocol,
        requested_alias: str,
    ) -> dict[str, InstrumentInfoProtocol]:
        """Resolve one binding to a transmitter quartet or one other instrument."""
        instrument_info = cls._find_instrument_info_by_alias(
            resolver=resolver,
            alias=requested_alias,
        )
        if not is_transmitter_role(instrument_info.definition.role):
            return {requested_alias: instrument_info}

        base_alias, suffix_index = split_transmitter_alias(requested_alias)
        if suffix_index is None:
            raise ValueError(
                "QuEL-3 transmitter must use aliases ending in `-0` through `-3`."
            )

        return {
            alias: (
                instrument_info
                if alias == requested_alias
                else cls._find_instrument_info_by_alias(
                    resolver=resolver,
                    alias=alias,
                )
            )
            for alias in build_transmitter_aliases(base_alias)
        }

    @classmethod
    def _resolve_payload(
        cls,
        *,
        payload: Quel3ExecutionPayload,
    ) -> Quel3ExecutionPayload:
        """Resolve timeline bindings to concrete instrument aliases."""
        bindings = payload.instrument_bindings

        alias_to_events: dict[str, list[tuple[int, Quel3WaveformEvent]]] = defaultdict(
            list
        )
        alias_to_captures: dict[
            str,
            list[tuple[float, float, int, str]],
        ] = defaultdict(list)
        alias_to_length_ns: dict[str, float] = {}
        alias_to_frequency_hz: dict[str, float] = {}
        sequence_index = 0

        for target, timeline in payload.fixed_timelines.items():
            binding = bindings.get(target)
            if binding is None and len(bindings) > 0:
                raise ValueError(
                    f"Instrument binding is not configured for target `{target}`."
                )
            alias = (
                target
                if binding is None
                else cls._resolve_alias_from_binding(binding=binding)
            )
            alias_to_length_ns[alias] = max(
                alias_to_length_ns.get(alias, 0.0),
                timeline.length_ns,
            )
            if timeline.frequency_hz is not None:
                current_frequency_hz = alias_to_frequency_hz.get(alias)
                if (
                    current_frequency_hz is not None
                    and current_frequency_hz != timeline.frequency_hz
                ):
                    raise ValueError(
                        "Conflicting frequency directives resolved to the same "
                        f"instrument alias `{alias}`."
                    )
                alias_to_frequency_hz[alias] = timeline.frequency_hz
            for event in timeline.events:
                alias_to_events[alias].append((sequence_index, event))
                sequence_index += 1
            for capture_window in timeline.capture_windows:
                alias_to_captures[alias].append(
                    (
                        capture_window.start_offset_ns,
                        capture_window.length_ns,
                        sequence_index,
                        target,
                    )
                )
                sequence_index += 1

        resolved_timelines = {}
        for alias, length_ns in alias_to_length_ns.items():
            event_entries = sorted(
                alias_to_events.get(alias, []),
                key=lambda item: (item[1].start_offset_ns, item[0]),
            )
            events = tuple(event for _, event in event_entries)
            capture_entries = sorted(
                alias_to_captures.get(alias, []),
                key=lambda item: (item[0], item[1], item[2]),
            )
            capture_windows = []
            for index, (
                start_offset_ns,
                length_ns_window,
                _order,
                _target,
            ) in enumerate(capture_entries):
                capture_windows.append(
                    Quel3CaptureWindow(
                        name=f"{alias}:{index}",
                        start_offset_ns=start_offset_ns,
                        length_ns=length_ns_window,
                    )
                )
            resolved_timelines[alias] = Quel3FixedTimeline(
                events=events,
                capture_windows=tuple(capture_windows),
                length_ns=length_ns,
                frequency_hz=alias_to_frequency_hz.get(alias),
            )

        return Quel3ExecutionPayload(
            waveform_library=payload.waveform_library,
            fixed_timelines=resolved_timelines,
            n_iterations=payload.n_iterations,
            shot_interval_ns=payload.shot_interval_ns,
            capture_mode=payload.capture_mode,
            instrument_bindings={},
            capture_port_bindings={},
        )

    @staticmethod
    def _filter_runnable_payload(
        payload: Quel3ExecutionPayload,
    ) -> Quel3ExecutionPayload:
        """Drop fixed timelines that would export an empty hardware directive."""
        runnable_timelines = {
            alias: timeline
            for alias, timeline in payload.fixed_timelines.items()
            if len(timeline.events) > 0 or len(timeline.capture_windows) > 0
        }
        if len(runnable_timelines) == 0:
            raise ValueError(
                "Quel3ExecutionPayload has no waveform events or capture windows to execute."
            )
        return replace(payload, fixed_timelines=runnable_timelines)

    @classmethod
    def _resolve_alias_from_binding(
        cls,
        *,
        binding: str,
    ) -> str:
        """Resolve one target binding to one instrument alias with fail-fast rules."""
        if binding.startswith("alias:"):
            alias = binding.removeprefix("alias:").strip()
            if len(alias) == 0:
                raise ValueError("Empty alias binding is not allowed.")
            _local_alias, unit_label = cls._split_unit_qualified_alias(alias)
            if unit_label is None:
                raise ValueError(
                    f"QuEL-3 alias binding must include a unit label: `{binding}`."
                )
            return alias
        raise ValueError(f"Unsupported instrument binding: `{binding}`.")

    @classmethod
    def _find_instrument_info_by_alias(
        cls,
        *,
        resolver: InstrumentResolverProtocol,
        alias: str,
    ) -> InstrumentInfoProtocol:
        """Find one instrument info, passing unit when the alias is unit-qualified."""
        local_alias, alias_unit_label = cls._split_unit_qualified_alias(alias)
        if alias_unit_label is not None:
            return resolver.find_inst_info_by_alias(
                local_alias,
                unit=alias_unit_label,
            )
        return resolver.find_inst_info_by_alias(alias)

    @staticmethod
    def _split_unit_qualified_alias(alias: str) -> tuple[str, str | None]:
        """Split `unit_label:alias` into local alias and unit label."""
        stripped_alias = alias.strip()
        unit_label, separator, local_alias = stripped_alias.partition(":")
        if separator and len(unit_label) > 0 and len(local_alias) > 0:
            return local_alias, unit_label
        return stripped_alias, None

    @staticmethod
    def _extract_capture_samples(
        result: ResultContainerProtocol,
        window_key: str,
        *,
        capture_mode: Quel3CaptureMode,
    ) -> np.ndarray | None:
        """Extract one capture sample-array from a result container entry."""
        if capture_mode in (
            Quel3CaptureMode.AVERAGED_WAVEFORM,
            Quel3CaptureMode.RAW_WAVEFORMS,
        ):
            values = result.iq_waveform_result.get(window_key)
            if values is None or len(values) == 0:
                return None
            if capture_mode is Quel3CaptureMode.RAW_WAVEFORMS:
                waveforms = []
                for value in values:
                    if not _has_iq_array(value):
                        return None
                    waveforms.append(np.asarray(value.iq_array, dtype=np.complex128))
                return np.stack(waveforms, axis=0)
            latest = values[-1]
            if not _has_iq_array(latest):
                return None
            return np.asarray(latest.iq_array, dtype=np.complex128)

        if capture_mode in (
            Quel3CaptureMode.AVERAGED_VALUE,
            Quel3CaptureMode.VALUES_PER_ITER,
        ):
            values = result.iq_point_result.get(window_key)
            if values is None or len(values) == 0:
                return None
            return np.asarray(values, dtype=np.complex128)

        raise ValueError(f"Unsupported capture mode: {capture_mode}.")

    @staticmethod
    def _build_measurement_result(
        *,
        payload: Quel3ExecutionPayload,
        shot_samples: dict[str, dict[str, list[np.ndarray]]],
        capture_sampling_period_ns: float | None,
        backend_sampling_period_ns: float,
        capture_decimation_factor: int,
    ) -> Quel3BackendExecutionResult:
        """Build canonical measurement result from per-shot capture samples."""
        if payload.capture_mode in (
            Quel3CaptureMode.AVERAGED_VALUE,
            Quel3CaptureMode.AVERAGED_WAVEFORM,
        ):
            is_averaged = True
        elif payload.capture_mode in (
            Quel3CaptureMode.VALUES_PER_ITER,
            Quel3CaptureMode.RAW_WAVEFORMS,
        ):
            is_averaged = False
        else:
            raise ValueError(f"Unsupported capture mode: {payload.capture_mode}")

        base_sampling_period_ns = capture_sampling_period_ns
        if base_sampling_period_ns is None:
            base_sampling_period_ns = backend_sampling_period_ns
        effective_sampling_period_ns = (
            base_sampling_period_ns * capture_decimation_factor
            if is_averaged
            else base_sampling_period_ns
        )

        measurement_data: dict[str, list[np.ndarray]] = defaultdict(list)
        for alias, timeline in payload.fixed_timelines.items():
            for window in timeline.capture_windows:
                samples = shot_samples.get(alias, {}).get(window.name, [])
                if len(samples) == 0:
                    measurement_data[alias].append(np.array([], dtype=np.complex128))
                    continue
                if is_averaged:
                    stacked_samples = np.stack(samples, axis=0)
                    capture_data = np.mean(stacked_samples, axis=0)
                else:
                    capture_data = samples[0]
                if payload.capture_mode in (
                    Quel3CaptureMode.AVERAGED_VALUE,
                    Quel3CaptureMode.VALUES_PER_ITER,
                ):
                    capture_data = capture_data * (
                        Quel3ExecutionManager._resolve_capture_sample_count(
                            window=window,
                            sampling_period_ns=effective_sampling_period_ns,
                        )
                    )
                measurement_data[alias].append(capture_data)

        return Quel3BackendExecutionResult(
            status={},
            data=dict(measurement_data),
            config={"sampling_period_ns": effective_sampling_period_ns},
        )

    @staticmethod
    def _resolve_capture_sample_count(
        *,
        window: Quel3CaptureWindow,
        sampling_period_ns: float,
    ) -> int:
        """Resolve the number of time samples after capture-grid ceiling."""
        samples = window.length_ns / sampling_period_ns
        rounded_samples = round(samples)
        if np.isclose(samples, rounded_samples, rtol=0.0, atol=1e-3):
            return max(1, rounded_samples)
        return max(1, int(np.ceil(samples)))

    def _load_quelware_api(
        self,
    ) -> _QuelwareExecutionApi:
        """Import quelware helpers lazily and return required symbols."""
        resolver_module = importlib.import_module(
            "quelware_client.client.helpers.instrument_resolver"
        )
        sequencer_module = importlib.import_module(
            "quelware_client.client.helpers.sequencer"
        )
        directive_module = importlib.import_module("quelware_core.entities.directives")
        driver_module = importlib.import_module(
            "quelware_client.core.instrument_driver"
        )
        client_factory: QuelwareClientFactory = (
            self._runtime_config.load_client_factory()
        )
        instrument_resolver_factory: InstrumentResolverFactory = (
            resolver_module.InstrumentResolver
        )
        sequencer_factory: Callable[..., SequencerProtocol] = sequencer_module.Sequencer
        fixed_timeline_driver_factory: InstrumentDriverFactory = (
            driver_module.create_instrument_driver_fixed_timeline
        )
        capture_mode_namespace: CaptureModeNamespaceProtocol = (
            directive_module.CaptureMode
        )
        set_frequency_directive_factory: SetFrequencyFactory = (
            directive_module.SetFrequency
        )
        set_capture_mode_directive_factory: SetCaptureModeFactory = (
            directive_module.SetCaptureMode
        )
        return _QuelwareExecutionApi(
            client_factory=client_factory,
            instrument_resolver_factory=instrument_resolver_factory,
            sequencer_factory=sequencer_factory,
            fixed_timeline_driver_factory=fixed_timeline_driver_factory,
            capture_mode_namespace=capture_mode_namespace,
            set_frequency_directive_factory=set_frequency_directive_factory,
            set_capture_mode_directive_factory=set_capture_mode_directive_factory,
        )
