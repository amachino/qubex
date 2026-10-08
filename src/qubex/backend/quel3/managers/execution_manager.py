"""Execution manager for QuEL-3 backend controller."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TypeVar

import numpy as np

from qubex.backend.backend_controller import BackendExecutionRequest
from qubex.backend.quel3.builders.sequencer_builder import Quel3SequencerBuilder
from qubex.backend.quel3.execution.payload_builder import (
    BuiltExecution,
    Quel3ExecutionPayloadBuilder,
)
from qubex.backend.quel3.execution.planner import Quel3ExecutionPlanner
from qubex.backend.quel3.execution.resources import (
    PayloadAnalysis,
    Quel3PayloadAnalyzer,
)
from qubex.backend.quel3.execution.results import Quel3ResultAssembler
from qubex.backend.quel3.execution.timing import sample_count
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
    Quel3ExecutionOptions,
    Quel3ExecutionPayload,
    Quel3ExecutionPlan,
)
from qubex.core.async_bridge import get_shared_async_bridge

T = TypeVar("T")

logger = logging.getLogger(__name__)

QUEL3_SESSION_TRIGGER_WAIT_MS: int | None = None


@contextmanager
def _log_phase_duration(phase: str) -> Iterator[None]:
    """Measure an execution phase only when DEBUG diagnostics are enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        yield
        return
    started = time.monotonic()
    try:
        yield
    finally:
        logger.debug(
            "QuEL-3 phase=%s elapsed_seconds=%.6f", phase, time.monotonic() - started
        )


def _run_async(factory: Callable[[], Awaitable[T]]) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-execution")
    return bridge.run_without_timeout(factory)


@dataclass(frozen=True)
class _PayloadExecutionSession:
    """Session-bound QuEL-3 drivers and resource IDs for one payload."""

    session: SessionProtocol
    alias_to_resource_id: dict[str, ResourceIdProtocol]
    alias_to_driver: dict[str, InstrumentDriverProtocol]
    capture_sampling_period_ns: float | None


class Quel3ExecutionManager:
    """Handle backend execution entrypoints for QuEL-3 controller."""

    def __init__(
        self,
        *,
        runtime_config: Quel3RuntimeConfig | None = None,
        sampling_period_ns: float,
        capture_decimation_factor: int,
        session_manager: Quel3SessionManager | None = None,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> None:
        self.execution_options = execution_options or Quel3ExecutionOptions()
        self._analyzer = Quel3PayloadAnalyzer(
            default_sampling_period_ns=sampling_period_ns
        )
        self._planner = Quel3ExecutionPlanner()
        self._payload_builder = Quel3ExecutionPayloadBuilder()
        self._runtime_config = runtime_config or Quel3RuntimeConfig()
        self._sampling_period_ns = sampling_period_ns
        self._capture_decimation_factor = capture_decimation_factor
        self._cable_delay_ns: dict[str, dict[str, float]] | None = None
        self._sequencer_builder = Quel3SequencerBuilder()
        self._session_manager = (
            session_manager
            if session_manager is not None
            else Quel3SessionManager(
                runtime_config=self._runtime_config,
            )
        )

    @property
    def cable_delay_ns(self) -> dict[str, dict[str, float]]:
        """Return a copy of per-port cable delays in ns."""
        return {
            unit: dict(ports) for unit, ports in (self._cable_delay_ns or {}).items()
        }

    @cable_delay_ns.setter
    def cable_delay_ns(self, delays: Mapping[str, Mapping[str, float]]) -> None:
        """Copy default delays; effective settings are validated before execution."""
        self._cable_delay_ns = {unit: dict(ports) for unit, ports in delays.items()}

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
        request: BackendExecutionRequest,
        instrument_cache: InstrumentCache,
        parallel: bool = True,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> Quel3BackendExecutionResult:
        """Execute a QuEL-3 backend request synchronously."""
        return _run_async(
            lambda: self.execute_async(
                request=request,
                instrument_cache=instrument_cache,
                parallel=parallel,
                execution_options=execution_options,
            )
        )

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
        instrument_cache: InstrumentCache,
        parallel: bool = True,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> Quel3BackendExecutionResult:
        """Execute a QuEL-3 backend request asynchronously."""
        return await self.execute(
            request=request,
            instrument_cache=instrument_cache,
            parallel=parallel,
            execution_options=execution_options,
        )

    async def execute_batch_async(
        self,
        *,
        requests: Sequence[BackendExecutionRequest],
        instrument_cache: InstrumentCache,
        parallel: bool = True,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> list[Quel3BackendExecutionResult]:
        """Validate all cached instruments and execute a batch of QuEL-3 requests."""
        analyses = self._analyze_requests(requests, instrument_cache)
        plan = self._planner.plan(
            tuple(analysis.planning for analysis in analyses),
            options=execution_options or self.execution_options,
        )
        executions = self._payload_builder.build_all(analyses, plan)
        if not executions:
            return []
        instruments = instrument_cache.snapshot()

        try:
            quelware_api = self._load_quelware_api()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc

        return await self._execute_built_payloads(
            analyses=analyses,
            executions=executions,
            instruments=instruments,
            quelware_api=quelware_api,
            parallel=parallel,
        )

    async def execute(
        self,
        *,
        request: BackendExecutionRequest,
        instrument_cache: InstrumentCache,
        parallel: bool = True,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> Quel3BackendExecutionResult:
        """
        Execute a QuEL-3 backend request asynchronously.

        Parameters
        ----------
        request : BackendExecutionRequest
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
            execution_options=execution_options,
        )
        return results[0]

    def plan_execution(
        self,
        *,
        requests: Sequence[BackendExecutionRequest],
        instrument_cache: InstrumentCache,
        execution_options: Quel3ExecutionOptions | None = None,
    ) -> Quel3ExecutionPlan:
        """Plan a batch using cached instruments without loading or contacting quelware."""
        analyses = self._analyze_requests(requests, instrument_cache)
        return self._planner.plan(
            tuple(analysis.planning for analysis in analyses),
            options=execution_options or self.execution_options,
        )

    def _analyze_requests(
        self,
        requests: Sequence[BackendExecutionRequest],
        instrument_cache: InstrumentCache,
    ) -> tuple[PayloadAnalysis, ...]:
        """Resolve request defaults and analyze all payloads before hardware IO."""
        payloads = []
        for request in requests:
            payload = request.payload
            if not isinstance(payload, Quel3ExecutionPayload):
                raise TypeError(
                    "Quel3ExecutionManager expects request payload to be `Quel3ExecutionPayload`."
                )
            delays = (
                payload.cable_delay_ns
                if payload.cable_delay_ns is not None
                else self._cable_delay_ns
            )
            if delays is not None:
                delays = {unit: dict(ports) for unit, ports in delays.items()}
                for unit, ports in delays.items():
                    for port, delay in ports.items():
                        if not math.isfinite(delay) or delay < 0:
                            raise ValueError(
                                f"Cable delay for {unit}:{port} must be finite and nonnegative."
                            )
            payloads.append(replace(payload, cable_delay_ns=delays))
        return self._analyzer.analyze_all(tuple(payloads), instrument_cache)

    async def _execute_built_payloads(
        self,
        *,
        analyses: tuple[PayloadAnalysis, ...],
        executions: tuple[BuiltExecution, ...],
        instruments: dict[str, InstrumentInfoProtocol],
        quelware_api: QuelwareExecutionApi,
        parallel: bool,
    ) -> list[Quel3BackendExecutionResult]:
        """Execute physical runs and restore complete logical jobs in input order."""
        logger.debug(
            "QuEL-3 jobs=%d executions=%d",
            len(analyses),
            len(executions),
        )
        assembler = Quel3ResultAssembler(
            tuple(analysis.payload for analysis in analyses)
        )
        try:
            for index, run in enumerate(executions):
                if logger.isEnabledFor(logging.DEBUG):
                    sources = ", ".join(
                        f"job={f.job_index} shots=[{f.shot_start},{f.shot_stop})"
                        for f in run.result_mapping.fragments
                    )
                    logger.debug(
                        "QuEL-3 execution=%d sources=(%s) iterations=%d",
                        index,
                        sources,
                        run.payload.n_iterations,
                    )
                alias_to_instrument_info = {
                    alias: instruments[alias]
                    for alias in sorted(run.payload.fixed_timelines)
                }
                started = time.monotonic()
                result = await run_with_session_request_retry(
                    manager=self._session_manager,
                    client_factory=quelware_api.client_factory,
                    resource_ids=tuple(
                        info.id for info in alias_to_instrument_info.values()
                    ),
                    operation=partial(
                        self._execute_payload_in_session,
                        payload=run.payload,
                        alias_to_instrument_info=alias_to_instrument_info,
                        quelware_api=quelware_api,
                        parallel=parallel,
                    ),
                )
                with _log_phase_duration("restore_results"):
                    assembler.add(run.result_mapping, result)
                logger.debug(
                    "QuEL-3 execution=%d completed elapsed_seconds=%.6f",
                    index,
                    time.monotonic() - started,
                )
            with _log_phase_duration("finish_results"):
                return assembler.finish()
        finally:
            with _log_phase_duration("cleanup"):
                await self._session_manager.close_safely()

    async def _execute_payload_in_session(
        self,
        session: SessionProtocol,
        *,
        payload: Quel3ExecutionPayload,
        alias_to_instrument_info: dict[str, InstrumentInfoProtocol],
        quelware_api: QuelwareExecutionApi,
        parallel: bool,
    ) -> Quel3BackendExecutionResult:
        """Build session-bound drivers and execute a validated payload."""
        with _log_phase_duration("bind_instruments"):
            session_state = self._build_payload_execution_session(
                session=session,
                alias_to_instrument_info=alias_to_instrument_info,
                aliases=tuple(alias_to_instrument_info),
                aliases_with_captures={
                    alias
                    for alias, timeline in payload.fixed_timelines.items()
                    if timeline.capture_windows
                },
                quelware_api=quelware_api,
            )
        return await self._execute_payload(
            alias_to_port={
                alias: str(info.port_id)
                for alias, info in alias_to_instrument_info.items()
            },
            payload=payload,
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
        alias_to_port: dict[str, str],
    ) -> Quel3BackendExecutionResult:
        """Execute one payload using an already-open payload session."""
        cable_delay_ns = payload.cable_delay_ns
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

        with _log_phase_duration("build_sequencer"):
            sequencer = self._sequencer_builder.build_prepared(
                payload=payload,
                sequencer_factory=quelware_api.sequencer_factory,
                default_sampling_period_ns=self._sampling_period_ns,
                alias_bindings=alias_bindings,
            )

        reference_delay_ns = max(
            (
                delay
                for ports in (cable_delay_ns or {}).values()
                for delay in ports.values()
            ),
            default=0.0,
        )
        alias_to_directives: dict[str, list[DirectiveProtocol]] = {}
        for alias in aliases:
            directives: list[DirectiveProtocol] = []
            if cable_delay_ns is not None:
                unit, _, port = alias_to_port[alias].partition(":")
                offset_ns = reference_delay_ns - cable_delay_ns.get(unit, {}).get(
                    port, 0.0
                )
                samples = offset_ns / (alias_bindings[alias][0] / 1e6)
                if not math.isfinite(samples) or not math.isclose(
                    samples, round(samples), rel_tol=0.0, abs_tol=1e-6
                ):
                    raise ValueError(
                        f"Timing offset for {alias!r} ({offset_ns} ns) must match the instrument sampling grid."
                    )
                factory = quelware_api.set_timing_offset_directive_factory
                if factory is None:
                    raise RuntimeError(
                        "quelware runtime does not expose required SetTimingOffset."
                    )
                directives.append(factory(offset_samples=round(samples)))
            frequency_hz = payload.fixed_timelines[alias].frequency_hz
            if frequency_hz is not None:
                directives.append(
                    quelware_api.set_frequency_directive_factory(hz=frequency_hz)
                )
            if payload.fixed_timelines[alias].capture_windows:
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
        with _log_phase_duration("initialize"):
            for driver in drivers:
                await driver.initialize()

        with _log_phase_duration("apply"):
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
        with _log_phase_duration("trigger"):
            await session_state.session.extend(QUELWARE_SESSION_EXTEND_TTL_MS)
            await session_state.session.trigger(
                instrument_ids=instrument_resource_ids,
                wait_ms=QUEL3_SESSION_TRIGGER_WAIT_MS,
            )
        with _log_phase_duration("fetch_results"):
            results = await session_state.session.wait_for_results(
                instrument_resource_ids
            )

        with _log_phase_duration("convert_result"):
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
                return np.stack(
                    [
                        np.asarray(value.iq_array, dtype=np.complex128)
                        for value in values
                    ],
                    axis=0,
                )
            return np.asarray(values[-1].iq_array, dtype=np.complex128)

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
        return max(1, sample_count(window.length_ns, sampling_period_ns))

    def _load_quelware_api(self) -> QuelwareExecutionApi:
        """Load execution dependencies with the configured client factory."""
        return load_quelware_execution_api(
            client_factory=self._runtime_config.load_client_factory()
        )
