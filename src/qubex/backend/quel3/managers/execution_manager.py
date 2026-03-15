"""Execution manager for QuEL-3 backend controller."""

from __future__ import annotations

import asyncio
import importlib
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TypeGuard, TypeVar

import numpy as np

from qubex.backend.quel3.builders.sequencer_builder import Quel3SequencerBuilder
from qubex.backend.quel3.infra.quelware_imports import (
    Quel3ClientMode,
    load_quelware_client_factory,
    validate_quelware_client_runtime,
)
from qubex.backend.quel3.interfaces import (
    CaptureModeNamespaceProtocol,
    CaptureModeValue,
    DirectiveProtocol,
    InstrumentDriverFactory,
    InstrumentDriverProtocol,
    InstrumentResolverFactory,
    InstrumentResolverProtocol,
    IqWaveformResultProtocol,
    QuelwareClientFactory,
    ResourceIdProtocol,
    ResultContainerProtocol,
    SequencerProtocol,
    SetCaptureModeFactory,
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


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-execution")
    return bridge.run(factory, timeout=timeout)


def _is_capture_mode_value(value: object) -> TypeGuard[CaptureModeValue]:
    """Return whether one runtime value behaves like quelware capture mode."""
    return isinstance(value, str) or isinstance(getattr(value, "name", None), str)


def _has_iq_array(value: object) -> TypeGuard[IqWaveformResultProtocol]:
    """Return whether one runtime value exposes waveform IQ samples."""
    return hasattr(value, "iq_array")


class Quel3ExecutionManager:
    """Handle backend execution entrypoints for QuEL-3 controller."""

    def __init__(
        self,
        *,
        quelware_endpoint: str,
        quelware_port: int,
        sampling_period_ns: float,
        capture_decimation_factor: int,
        client_mode: Quel3ClientMode = "server",
        standalone_unit_label: str | None = None,
    ) -> None:
        normalized_client_mode = validate_quelware_client_runtime(
            client_mode=client_mode,
            standalone_unit_label=standalone_unit_label,
        )
        self._quelware_endpoint = quelware_endpoint
        self._quelware_port = quelware_port
        self._sampling_period_ns = sampling_period_ns
        self._capture_decimation_factor = capture_decimation_factor
        self._client_mode: Quel3ClientMode = normalized_client_mode
        self._standalone_unit_label = standalone_unit_label
        self._sequencer_builder = Quel3SequencerBuilder()

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint used for execution."""
        return self._quelware_endpoint

    @property
    def quelware_port(self) -> int:
        """Return quelware port used for execution."""
        return self._quelware_port

    @property
    def sampling_period_ns(self) -> float:
        """Return backend sampling period in ns."""
        return self._sampling_period_ns

    @property
    def client_mode(self) -> Quel3ClientMode:
        """Return configured quelware client mode."""
        return self._client_mode

    @property
    def standalone_unit_label(self) -> str | None:
        """Return configured standalone unit label."""
        return self._standalone_unit_label

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
        return await self._execute(payload, parallel=parallel)

    async def _execute(
        self,
        payload: Quel3ExecutionPayload,
        *,
        parallel: bool,
    ) -> Quel3BackendExecutionResult:
        """Execute one fixed-timeline measurement flow via quelware."""
        if len(payload.fixed_timelines) == 0:
            raise ValueError("Quel3ExecutionPayload must include fixed timelines.")

        try:
            (
                create_quelware_client,
                instrument_resolver_factory,
                sequencer_factory,
                create_instrument_driver_fixed_timeline,
                capture_mode_enum,
                set_capture_mode_factory,
            ) = self._load_quelware_api()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc

        async with create_quelware_client(
            self._quelware_endpoint,
            self._quelware_port,
        ) as client:
            resolver = instrument_resolver_factory()
            await resolver.refresh(client)

            resolved_payload = self._resolve_payload(
                payload=payload,
                resolver=resolver,
            )
            resolved_payload = self._filter_runnable_payload(resolved_payload)
            aliases = sorted(resolved_payload.fixed_timelines.keys())
            alias_to_id = self._resolve_alias_to_id_map(
                resolver=resolver,
                aliases=aliases,
            )
            instrument_ids = [alias_to_id[alias] for alias in aliases]

            async with client.create_session(instrument_ids) as session:
                alias_to_driver: dict[str, InstrumentDriverProtocol] = {}
                alias_bindings: dict[str, tuple[int, int]] = {}
                sampling_period_ns: float | None = None
                for alias in aliases:
                    instrument_info = resolver.find_inst_info_by_alias(alias)
                    driver = create_instrument_driver_fixed_timeline(
                        session,
                        instrument_info,
                    )
                    alias_to_driver[alias] = driver
                    sampling_period_fs = getattr(
                        driver.instrument_config,
                        "sampling_period_fs",
                        None,
                    )
                    timeline_step_samples = getattr(
                        driver.instrument_config,
                        "timeline_step_samples",
                        None,
                    )
                    if not isinstance(sampling_period_fs, int):
                        raise TypeError(
                            "Instrument config must expose integer `sampling_period_fs`."
                        )
                    if not isinstance(timeline_step_samples, int):
                        raise TypeError(
                            "Instrument config must expose integer `timeline_step_samples`."
                        )
                    alias_bindings[alias] = (
                        sampling_period_fs,
                        timeline_step_samples,
                    )
                    if isinstance(sampling_period_fs, int):
                        sampling_period_ns = sampling_period_fs / 1e6

                timeline_iterations = self._resolve_timeline_iterations(
                    capture_mode=payload.capture_mode,
                    repeats=payload.repeats,
                )
                sequencer = self._sequencer_builder.build(
                    payload=resolved_payload,
                    sequencer_factory=sequencer_factory,
                    default_sampling_period_ns=self._sampling_period_ns,
                    alias_bindings=alias_bindings,
                    iterations=timeline_iterations,
                )

                alias_to_directives = {
                    alias: self._build_driver_directives(
                        alias=alias,
                        sequencer=sequencer,
                        capture_mode=payload.capture_mode,
                        capture_mode_enum=capture_mode_enum,
                        set_capture_mode_factory=set_capture_mode_factory,
                    )
                    for alias in alias_to_driver
                }
                await self._initialize_drivers(
                    drivers=tuple(alias_to_driver.values()),
                    parallel=parallel,
                )
                await self._apply_drivers(
                    alias_to_driver=alias_to_driver,
                    alias_to_directives=alias_to_directives,
                    parallel=parallel,
                )

                shot_samples = self._initialize_shot_samples(resolved_payload)
                await session.trigger(instrument_ids=instrument_ids)
                alias_results = await self._fetch_alias_results(
                    aliases=aliases,
                    alias_to_driver=alias_to_driver,
                    parallel=parallel,
                )
                for alias, timeline in resolved_payload.fixed_timelines.items():
                    result = alias_results[alias]
                    for window in timeline.capture_windows:
                        window_key = window.name
                        capture_samples = self._extract_capture_samples(
                            result,
                            window_key,
                        )
                        if capture_samples is None:
                            continue
                        shot_samples[alias][window.name].append(capture_samples)

        return self._build_measurement_result(
            payload=resolved_payload,
            shot_samples=shot_samples,
            sampling_period_ns=sampling_period_ns,
            backend_sampling_period_ns=self._sampling_period_ns,
            capture_decimation_factor=self._capture_decimation_factor,
        )

    @classmethod
    def _build_driver_directives(
        cls,
        *,
        alias: str,
        sequencer: SequencerProtocol,
        capture_mode: Quel3CaptureMode,
        capture_mode_enum: CaptureModeNamespaceProtocol,
        set_capture_mode_factory: SetCaptureModeFactory,
    ) -> list[DirectiveProtocol]:
        """Build the directives applied to one instrument driver."""
        directives: list[DirectiveProtocol] = []
        capture_mode_directive = cls._build_capture_mode_directive(
            capture_mode=capture_mode,
            capture_mode_enum=capture_mode_enum,
            set_capture_mode_factory=set_capture_mode_factory,
        )
        if capture_mode_directive is not None:
            directives.append(capture_mode_directive)
        directives.append(sequencer.export_set_fixed_timeline_directive(alias))
        return directives

    @staticmethod
    async def _initialize_drivers(
        *,
        drivers: tuple[InstrumentDriverProtocol, ...],
        parallel: bool,
    ) -> None:
        """Initialize drivers with optional per-driver parallelism."""
        if parallel:
            await asyncio.gather(*(driver.initialize() for driver in drivers))
            return
        for driver in drivers:
            await driver.initialize()

    @staticmethod
    async def _apply_drivers(
        *,
        alias_to_driver: dict[str, InstrumentDriverProtocol],
        alias_to_directives: dict[str, list[DirectiveProtocol]],
        parallel: bool,
    ) -> None:
        """Apply directives with optional per-driver parallelism."""
        if parallel:
            await asyncio.gather(
                *(
                    driver.apply(alias_to_directives[alias])
                    for alias, driver in alias_to_driver.items()
                )
            )
            return
        for alias, driver in alias_to_driver.items():
            await driver.apply(alias_to_directives[alias])

    @staticmethod
    async def _fetch_alias_results(
        *,
        aliases: list[str],
        alias_to_driver: dict[str, InstrumentDriverProtocol],
        parallel: bool,
    ) -> dict[str, ResultContainerProtocol]:
        """Fetch results with optional per-driver parallelism."""
        if parallel:
            results = await asyncio.gather(
                *(alias_to_driver[alias].fetch_result() for alias in aliases)
            )
            return dict(zip(aliases, results, strict=True))
        alias_results: dict[str, ResultContainerProtocol] = {}
        for alias in aliases:
            alias_results[alias] = await alias_to_driver[alias].fetch_result()
        return alias_results

    @staticmethod
    def _resolve_alias_to_id_map(
        *,
        resolver: InstrumentResolverProtocol,
        aliases: list[str],
    ) -> dict[str, ResourceIdProtocol]:
        """Resolve alias-to-resource-id mapping using InstrumentResolver."""
        resource_ids = resolver.resolve(aliases)
        if len(resource_ids) != len(aliases):
            raise ValueError(
                "InstrumentResolver returned inconsistent alias resolution length."
            )
        return dict(zip(aliases, resource_ids, strict=True))

    @classmethod
    def _resolve_payload(
        cls,
        *,
        payload: Quel3ExecutionPayload,
        resolver: InstrumentResolverProtocol,
    ) -> Quel3ExecutionPayload:
        """Resolve timeline bindings to concrete instrument aliases."""
        bindings = (
            payload.instrument_bindings
            if len(payload.instrument_bindings) > 0
            else {key: f"alias:{key}" for key in payload.fixed_timelines}
        )

        alias_to_events: dict[str, list[tuple[int, Quel3WaveformEvent]]] = defaultdict(
            list
        )
        alias_to_captures: dict[
            str,
            list[tuple[float, float, int, str]],
        ] = defaultdict(list)
        alias_to_length_ns: dict[str, float] = {}
        sequence_index = 0

        for target, timeline in payload.fixed_timelines.items():
            binding = bindings.get(target)
            if binding is None:
                raise ValueError(
                    f"Instrument binding is not configured for target `{target}`."
                )
            capture_port_binding = payload.capture_port_bindings.get(target)
            alias = cls._resolve_alias_from_binding(
                target=target,
                resolver=resolver,
                binding=binding,
                capture_port_binding=capture_port_binding,
                has_events=len(timeline.events) > 0,
                has_captures=len(timeline.capture_windows) > 0,
            )
            alias_to_length_ns[alias] = max(
                alias_to_length_ns.get(alias, 0.0),
                timeline.length_ns,
            )
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
            )

        return Quel3ExecutionPayload(
            waveform_library=payload.waveform_library,
            fixed_timelines=resolved_timelines,
            interval_ns=payload.interval_ns,
            repeats=payload.repeats,
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
        target: str,
        resolver: InstrumentResolverProtocol,
        binding: str,
        capture_port_binding: str | None,
        has_events: bool,
        has_captures: bool,
    ) -> str:
        """Resolve one target binding to one instrument alias with fail-fast rules."""
        if binding.startswith("alias:"):
            alias = binding.removeprefix("alias:").strip()
            if len(alias) == 0:
                raise ValueError("Empty alias binding is not allowed.")
            try:
                resolver.find_inst_info_by_alias(alias)
            except ValueError as exc:
                raise ValueError(
                    f"Instrument alias `{alias}` could not be resolved."
                ) from exc
            return alias
        raise ValueError(f"Unsupported instrument binding: `{binding}`.")

    @staticmethod
    def _build_capture_mode_directive(
        *,
        capture_mode: Quel3CaptureMode,
        capture_mode_enum: CaptureModeNamespaceProtocol,
        set_capture_mode_factory: SetCaptureModeFactory,
    ) -> DirectiveProtocol:
        """Build one capture-mode directive from payload capture mode."""
        candidates_by_mode: dict[Quel3CaptureMode, tuple[str, ...]] = {
            Quel3CaptureMode.RAW_WAVEFORMS: ("RAW_WAVEFORMS",),
            Quel3CaptureMode.AVERAGED_WAVEFORM: ("AVERAGED_WAVEFORM",),
            Quel3CaptureMode.AVERAGED_VALUE: ("AVERAGED_VALUE",),
            Quel3CaptureMode.VALUES_PER_ITER: ("VALUES_PER_ITER",),
        }
        candidates = candidates_by_mode.get(capture_mode)
        if candidates is None:
            raise ValueError(f"Unsupported capture mode: {capture_mode}.")
        mode: CaptureModeValue | None = None
        for candidate in candidates:
            resolved_mode = getattr(capture_mode_enum, candidate, None)
            if _is_capture_mode_value(resolved_mode):
                mode = resolved_mode
                break
        if mode is None:
            raise RuntimeError(
                "quelware runtime does not expose required "
                f"`CaptureMode.{candidates[0]}`."
            )
        return set_capture_mode_factory(mode=mode)

    @staticmethod
    def _resolve_timeline_iterations(
        *,
        capture_mode: Quel3CaptureMode,
        repeats: int,
    ) -> int:
        """Resolve timeline iteration count from capture mode and repeats."""
        if repeats <= 1:
            return 1
        if capture_mode in (
            Quel3CaptureMode.AVERAGED_VALUE,
            Quel3CaptureMode.AVERAGED_WAVEFORM,
            Quel3CaptureMode.VALUES_PER_ITER,
        ):
            return repeats
        return 1

    @staticmethod
    def _initialize_shot_samples(
        payload: Quel3ExecutionPayload,
    ) -> dict[str, dict[str, list[np.ndarray]]]:
        """Initialize nested shot-sample container by alias/capture window."""
        return {
            alias: {window.name: [] for window in timeline.capture_windows}
            for alias, timeline in payload.fixed_timelines.items()
        }

    @staticmethod
    def _extract_capture_samples(
        result: ResultContainerProtocol,
        window_key: str,
    ) -> np.ndarray | None:
        """Extract one capture sample-array from a result container entry."""
        values = result.iq_result.get(window_key)
        if values is None or len(values) == 0:
            return None
        first = values[0]
        if _has_iq_array(first):
            latest = values[-1]
            if not _has_iq_array(latest):
                return None
            return np.asarray(latest.iq_array, dtype=np.complex128)
        return np.asarray(values, dtype=np.complex128)

    @staticmethod
    def _build_measurement_result(
        *,
        payload: Quel3ExecutionPayload,
        shot_samples: dict[str, dict[str, list[np.ndarray]]],
        sampling_period_ns: float | None,
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

        measurement_data: dict[str, list[np.ndarray]] = defaultdict(list)
        for alias, timeline in payload.fixed_timelines.items():
            for window in timeline.capture_windows:
                samples = shot_samples.get(alias, {}).get(window.name, [])
                if len(samples) == 0:
                    measurement_data[alias].append(np.array([], dtype=np.complex128))
                    continue
                if is_averaged:
                    values = np.stack(samples, axis=0)
                    measurement_data[alias].append(np.mean(values, axis=0))
                else:
                    measurement_data[alias].append(samples[0])

        base_sampling_period_ns = (
            sampling_period_ns
            if sampling_period_ns is not None
            else backend_sampling_period_ns
        )
        effective_sampling_period_ns = (
            base_sampling_period_ns * capture_decimation_factor
            if is_averaged
            else base_sampling_period_ns
        )

        return Quel3BackendExecutionResult(
            status={},
            data=dict(measurement_data),
            config={"sampling_period_ns": effective_sampling_period_ns},
        )

    def _load_quelware_api(
        self,
    ) -> tuple[
        QuelwareClientFactory,
        InstrumentResolverFactory,
        Callable[..., SequencerProtocol],
        InstrumentDriverFactory,
        CaptureModeNamespaceProtocol,
        SetCaptureModeFactory,
    ]:
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
        create_quelware_client: QuelwareClientFactory = load_quelware_client_factory(
            client_mode=self._client_mode,
            standalone_unit_label=self._standalone_unit_label,
        )
        instrument_resolver_factory: InstrumentResolverFactory = (
            resolver_module.InstrumentResolver
        )
        sequencer_factory: Callable[..., SequencerProtocol] = sequencer_module.Sequencer
        create_instrument_driver_fixed_timeline: InstrumentDriverFactory = (
            driver_module.create_instrument_driver_fixed_timeline
        )
        capture_mode_enum: CaptureModeNamespaceProtocol = directive_module.CaptureMode
        set_capture_mode_factory: SetCaptureModeFactory = (
            directive_module.SetCaptureMode
        )
        return (
            create_quelware_client,
            instrument_resolver_factory,
            sequencer_factory,
            create_instrument_driver_fixed_timeline,
            capture_mode_enum,
            set_capture_mode_factory,
        )
