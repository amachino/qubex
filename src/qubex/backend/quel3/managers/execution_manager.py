"""Execution manager for QuEL-3 backend controller."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import TypeGuard, TypeVar

import numpy as np

from qubex.backend.quel3.builders.sequencer_builder import Quel3SequencerBuilder
from qubex.backend.quel3.infra.quelware_imports import (
    Quel3ClientMode,
    QuelwareExecutionApi,
    load_quelware_execution_api,
)
from qubex.backend.quel3.infra.runtime_config import Quel3RuntimeConfig
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces import (
    DirectiveProtocol,
    InstrumentDriverProtocol,
    InstrumentInfoProtocol,
    IqWaveformResultProtocol,
    ResourceIdProtocol,
    ResultContainerProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.managers.session_manager import Quel3SessionManager
from qubex.backend.quel3.managers.session_workarounds import (
    QUELWARE_SESSION_EXTEND_TTL_MS,
    run_with_session_request_retry,
)
from qubex.backend.quel3.models import (
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
)
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

T = TypeVar("T")

QUEL3_SESSION_TRIGGER_WAIT_MS: int | None = None


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
    """Session-bound QuEL-3 drivers and resource IDs for one payload."""

    session: SessionProtocol
    alias_to_resource_id: dict[str, ResourceIdProtocol]
    alias_to_driver: dict[str, InstrumentDriverProtocol]
    capture_sampling_period_ns: float | None


@dataclass(frozen=True)
class _PayloadExecutionPlan:
    """Runnable payload and cached instrument information for one execution."""

    payload: Quel3ExecutionPayload
    aliases: tuple[str, ...]
    aliases_with_captures: frozenset[str]
    alias_to_instrument_info: dict[str, InstrumentInfoProtocol]


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

    def execute_sync(
        self,
        *,
        request: object,
        instrument_cache: InstrumentCache,
        parallel: bool = True,
    ) -> Quel3BackendExecutionResult:
        """Execute a QuEL-3 backend request synchronously."""
        return _run_async(
            lambda: self.execute_async(
                request=request, instrument_cache=instrument_cache, parallel=parallel
            )
        )

    async def execute_async(
        self,
        *,
        request: object,
        instrument_cache: InstrumentCache,
        parallel: bool = True,
    ) -> Quel3BackendExecutionResult:
        """Execute a QuEL-3 backend request asynchronously."""
        return await self.execute(
            request=request, instrument_cache=instrument_cache, parallel=parallel
        )

    async def execute_batch_async(
        self,
        *,
        requests: list[object] | tuple[object, ...],
        instrument_cache: InstrumentCache,
        parallel: bool = True,
    ) -> list[Quel3BackendExecutionResult]:
        """Validate all cached instruments and execute a batch of QuEL-3 requests."""
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

        payload_plans = [
            self._prepare_payload_execution_plan(
                payload=payload, instrument_cache=instrument_cache
            )
            for payload in payloads
        ]

        try:
            quelware_api = self._load_quelware_api()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc

        return await self._execute_payload_plans(
            payload_plans=payload_plans,
            quelware_api=quelware_api,
            parallel=parallel,
        )

    async def execute(
        self,
        *,
        request: object,
        instrument_cache: InstrumentCache,
        parallel: bool = True,
    ) -> Quel3BackendExecutionResult:
        """
        Execute a QuEL-3 backend request asynchronously.

        Parameters
        ----------
        request : object
            Backend execution request with `payload`.
        instrument_cache : InstrumentCache
            Explicitly refreshed instrument information owned by the controller.
        parallel : bool, optional
            Whether to parallelize per-instrument phases, by default `True`.
        """
        results = await self.execute_batch_async(
            requests=(request,),
            instrument_cache=instrument_cache,
            parallel=parallel,
        )
        return results[0]

    async def _execute_payload_plans(
        self,
        *,
        payload_plans: list[_PayloadExecutionPlan],
        quelware_api: QuelwareExecutionApi,
        parallel: bool,
    ) -> list[Quel3BackendExecutionResult]:
        """Execute one payload batch with per-payload quelware sessions."""
        try:
            return [
                await run_with_session_request_retry(
                    manager=self._session_manager,
                    client_factory=quelware_api.client_factory,
                    resource_ids=tuple(
                        payload_plan.alias_to_instrument_info[alias].id
                        for alias in payload_plan.aliases
                    ),
                    operation=partial(
                        self._execute_payload_plan,
                        payload_plan=payload_plan,
                        quelware_api=quelware_api,
                        parallel=parallel,
                    ),
                )
                for payload_plan in payload_plans
            ]
        finally:
            await self._session_manager.close_safely()

    async def _execute_payload_plan(
        self,
        session: SessionProtocol,
        *,
        payload_plan: _PayloadExecutionPlan,
        quelware_api: QuelwareExecutionApi,
        parallel: bool,
    ) -> Quel3BackendExecutionResult:
        """Build session-bound drivers and execute one payload plan."""
        session_state = self._build_payload_execution_session(
            session=session,
            alias_to_instrument_info=payload_plan.alias_to_instrument_info,
            aliases=payload_plan.aliases,
            aliases_with_captures=payload_plan.aliases_with_captures,
            quelware_api=quelware_api,
        )
        return await self._execute_payload(
            payload=payload_plan.payload,
            session_state=session_state,
            quelware_api=quelware_api,
            parallel=parallel,
        )

    def _build_payload_execution_session(
        self,
        *,
        session: SessionProtocol,
        alias_to_instrument_info: dict[str, InstrumentInfoProtocol],
        aliases: Sequence[str],
        aliases_with_captures: Collection[str],
        quelware_api: QuelwareExecutionApi,
    ) -> _PayloadExecutionSession:
        """Build drivers bound to the supplied payload session."""
        instrument_resource_ids = [
            alias_to_instrument_info[alias].id for alias in aliases
        ]
        alias_to_resource_id = dict(zip(aliases, instrument_resource_ids, strict=True))
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
        """Resolve the capture sampling period for one payload."""
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

    async def _execute_payload(
        self,
        *,
        payload: Quel3ExecutionPayload,
        session_state: _PayloadExecutionSession,
        quelware_api: QuelwareExecutionApi,
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
        await session_state.session.extend(QUELWARE_SESSION_EXTEND_TTL_MS)
        await session_state.session.trigger(
            instrument_ids=instrument_resource_ids,
            wait_ms=QUEL3_SESSION_TRIGGER_WAIT_MS,
        )
        results = await session_state.session.wait_for_results(instrument_resource_ids)

        for alias, timeline in payload.fixed_timelines.items():
            result = results[session_state.alias_to_resource_id[alias]]
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
        instrument_cache: InstrumentCache,
    ) -> _PayloadExecutionPlan:
        """Validate cached instruments and prepare runnable timelines for execution."""
        timelines = {
            alias: cls._prepare_timeline(alias=alias, timeline=timeline)
            for alias, timeline in payload.fixed_timelines.items()
            if timeline.events or timeline.capture_windows
        }
        if not timelines:
            raise ValueError(
                "Quel3ExecutionPayload has no waveform events or capture windows to execute."
            )
        aliases = tuple(sorted(timelines))
        alias_to_instrument_info = {
            alias: instrument_cache.get(alias) for alias in aliases
        }
        return _PayloadExecutionPlan(
            payload=replace(payload, fixed_timelines=timelines),
            aliases=aliases,
            aliases_with_captures=frozenset(
                alias
                for alias, timeline in timelines.items()
                if timeline.capture_windows
            ),
            alias_to_instrument_info=alias_to_instrument_info,
        )

    @staticmethod
    def _prepare_timeline(
        *, alias: str, timeline: Quel3FixedTimeline
    ) -> Quel3FixedTimeline:
        """Validate unique capture names and stably order events and captures by time."""
        capture_names: set[str] = set()
        for window in timeline.capture_windows:
            if window.name in capture_names:
                raise ValueError(
                    f"Duplicate capture window name `{window.name}` for alias `{alias}`."
                )
            capture_names.add(window.name)
        return replace(
            timeline,
            events=tuple(
                sorted(timeline.events, key=lambda event: event.start_offset_ns)
            ),
            capture_windows=tuple(
                sorted(
                    timeline.capture_windows,
                    key=lambda window: (window.start_offset_ns, window.length_ns),
                )
            ),
        )

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

    def _load_quelware_api(self) -> QuelwareExecutionApi:
        """Load execution dependencies with the configured client factory."""
        return load_quelware_execution_api(
            client_factory=self._runtime_config.load_client_factory()
        )
