"""Execution manager for QuEL-3 backend controller."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from qubex.backend.quel3.builders.sequencer_builder import Quel3SequencerBuilder
from qubex.backend.quel3.infra.quelware_imports import (
    import_module_with_workspace_fallback,
)
from qubex.backend.quel3.interfaces import (
    DirectiveProtocol,
    InstrumentDriverFactory,
    InstrumentDriverProtocol,
    InstrumentInfoProtocol,
    InstrumentResolverFactory,
    InstrumentResolverProtocol,
    QuelwareClientFactory,
    ResourceIdProtocol,
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


@dataclass(frozen=True)
class _PortBinding:
    """Parsed port binding for matching target ports to instrument resources."""

    unit: str
    out_port: int | None
    in_port: int | None


class Quel3ExecutionManager:
    """Handle backend execution entrypoints for QuEL-3 controller."""

    def __init__(
        self,
        *,
        quelware_endpoint: str,
        quelware_port: int,
        sampling_period: float,
        capture_decimation_factor: int,
    ) -> None:
        self._quelware_endpoint = quelware_endpoint
        self._quelware_port = quelware_port
        self._sampling_period = sampling_period
        self._capture_decimation_factor = capture_decimation_factor
        self._sequencer_builder = Quel3SequencerBuilder()

    async def execute(self, *, request: object) -> Quel3BackendExecutionResult:
        """
        Execute a QuEL-3 backend request asynchronously.

        Parameters
        ----------
        request : object
            Backend execution request with `payload`.
        """
        payload = getattr(request, "payload", None)
        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3ExecutionManager expects request payload to be `Quel3ExecutionPayload`."
            )
        return await self._execute(payload)

    async def _execute(
        self,
        payload: Quel3ExecutionPayload,
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
            aliases = sorted(resolved_payload.fixed_timelines.keys())
            alias_to_id = self._resolve_alias_to_id_map(
                resolver=resolver,
                aliases=aliases,
            )
            instrument_ids = [alias_to_id[alias] for alias in aliases]

            sequencer = self._sequencer_builder.build(
                payload=resolved_payload,
                sequencer_factory=sequencer_factory,
                default_sampling_period_ns=self._sampling_period,
            )

            async with client.create_session(instrument_ids) as session:
                alias_to_driver: dict[str, InstrumentDriverProtocol] = {}
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
                    if isinstance(sampling_period_fs, int):
                        sampling_period_ns = sampling_period_fs / 1e6

                for alias, driver in alias_to_driver.items():
                    capture_mode_directive = self._build_capture_mode_directive(
                        capture_mode=payload.capture_mode,
                        capture_mode_enum=capture_mode_enum,
                        set_capture_mode_factory=set_capture_mode_factory,
                    )
                    if capture_mode_directive is not None:
                        await driver.apply(capture_mode_directive)
                    directive = sequencer.export_set_fixed_timeline_directive(
                        alias,
                        driver.instrument_config.sampling_period_fs,
                    )
                    await driver.apply(directive)

                shot_count = (
                    payload.repeats
                    if (
                        payload.capture_mode
                        in (
                            Quel3CaptureMode.AVERAGED_VALUE,
                            Quel3CaptureMode.AVERAGED_WAVEFORM,
                        )
                        and payload.repeats > 1
                    )
                    else 1
                )
                shot_samples = self._initialize_shot_samples(resolved_payload)
                for _ in range(shot_count):
                    await session.trigger(instrument_ids=instrument_ids)
                    alias_results = {
                        alias: await driver.fetch_result()
                        for alias, driver in alias_to_driver.items()
                    }
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
            backend_sampling_period=self._sampling_period,
            capture_decimation_factor=self._capture_decimation_factor,
        )

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

    @classmethod
    def _resolve_alias_from_binding(
        cls,
        *,
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
        if binding.startswith("port:"):
            port_binding = cls._parse_box_port_binding(
                binding.removeprefix("port:").strip()
            )
            capture_binding = (
                cls._parse_box_port_binding(capture_port_binding)
                if (
                    capture_port_binding is not None
                    and len(capture_port_binding.strip()) > 0
                )
                else None
            )
            return cls._resolve_alias_from_port_binding(
                resolver=resolver,
                port_binding=port_binding,
                capture_port_binding=capture_binding,
                has_events=has_events,
                has_captures=has_captures,
            )
        raise ValueError(f"Unsupported instrument binding: `{binding}`.")

    @classmethod
    def _resolve_alias_from_port_binding(
        cls,
        *,
        resolver: InstrumentResolverProtocol,
        port_binding: _PortBinding,
        capture_port_binding: _PortBinding | None,
        has_events: bool,
        has_captures: bool,
    ) -> str:
        """Resolve one alias from one box-port binding with fail-fast behavior."""
        instrument_infos = cls._list_instrument_infos_by_alias(resolver)
        candidates: list[str] = []
        for alias, instrument_info in instrument_infos.items():
            parsed = cls._parse_instrument_port_binding(str(instrument_info.port_id))
            if parsed is None or parsed.unit != port_binding.unit:
                continue
            role_name = str(instrument_info.definition.role)
            supports_tx = any(
                token in role_name for token in ("TRANSMITTER", "TRANSCEIVER")
            )
            supports_rx = any(
                token in role_name for token in ("RECEIVER", "TRANSCEIVER")
            )

            if has_events:
                if not supports_tx:
                    continue
                if (
                    parsed.out_port is not None
                    and parsed.out_port != port_binding.out_port
                ):
                    continue
            if has_captures:
                if not supports_rx:
                    continue
                expected_capture_port = (
                    capture_port_binding.out_port
                    if capture_port_binding is not None
                    else port_binding.out_port
                )
                if parsed.in_port is not None:
                    if parsed.in_port != expected_capture_port:
                        continue
                elif (
                    parsed.out_port is not None
                    and parsed.out_port != expected_capture_port
                ):
                    continue
            candidates.append(alias)

        if len(candidates) == 0:
            available = sorted(instrument_infos.keys())
            raise ValueError(
                "No instrument alias is compatible with binding "
                f"`{port_binding.unit}-{port_binding.out_port}`. available={available}"
            )
        unique_candidates = sorted(dict.fromkeys(candidates))
        if len(unique_candidates) > 1:
            raise ValueError(
                "Ambiguous instrument aliases for binding "
                f"`{port_binding.unit}-{port_binding.out_port}`: {unique_candidates}"
            )
        return unique_candidates[0]

    @staticmethod
    def _list_instrument_infos_by_alias(
        resolver: InstrumentResolverProtocol,
    ) -> dict[str, InstrumentInfoProtocol]:
        """List instrument infos keyed by alias from resolver state."""
        alias_to_id = getattr(resolver, "_alias_to_id", None)
        if not isinstance(alias_to_id, dict):
            raise TypeError("InstrumentResolver does not expose alias mapping state.")
        infos: dict[str, InstrumentInfoProtocol] = {}
        for alias in sorted(alias_to_id.keys()):
            alias_str = str(alias)
            infos[alias_str] = resolver.find_inst_info_by_alias(alias_str)
        return infos

    @staticmethod
    def _parse_box_port_binding(binding: str) -> _PortBinding:
        """Parse `<box>-<port>` binding used by Qubex runtime wiring."""
        box_id, separator, port_text = binding.rpartition("-")
        if separator == "" or len(box_id) == 0 or len(port_text) == 0:
            raise ValueError(f"Invalid port binding: `{binding}`.")
        try:
            port_number = int(port_text)
        except ValueError as exc:
            raise ValueError(f"Invalid port number in binding: `{binding}`.") from exc
        return _PortBinding(unit=box_id, out_port=port_number, in_port=port_number)

    @staticmethod
    def _parse_instrument_port_binding(resource_id: str) -> _PortBinding | None:
        """Parse quelware instrument `port_id` into comparable port binding."""
        match = re.fullmatch(
            r"(?P<unit>[^:]+):p(?P<first>\d+)(?:p(?P<second>\d+))?(?P<kind>tx|rx|trx)?",
            resource_id,
        )
        if match is None:
            return None
        unit = str(match.group("unit"))
        first = int(match.group("first"))
        second_raw = match.group("second")
        second = int(second_raw) if second_raw is not None else None
        kind = match.group("kind")
        if kind == "tx":
            return _PortBinding(unit=unit, out_port=first, in_port=None)
        if kind == "rx":
            return _PortBinding(unit=unit, out_port=None, in_port=first)
        if kind == "trx":
            if second is None:
                return _PortBinding(unit=unit, out_port=first, in_port=first)
            return _PortBinding(unit=unit, out_port=second, in_port=first)
        return _PortBinding(unit=unit, out_port=first, in_port=first)

    @staticmethod
    def _build_capture_mode_directive(
        *,
        capture_mode: Quel3CaptureMode,
        capture_mode_enum: Any,
        set_capture_mode_factory: SetCaptureModeFactory,
    ) -> DirectiveProtocol | None:
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
        mode = None
        for candidate in candidates:
            if hasattr(capture_mode_enum, candidate):
                mode = getattr(capture_mode_enum, candidate)
                break
        if mode is None:
            return None
        return set_capture_mode_factory(mode=mode)

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
        result: object,
        window_key: str,
    ) -> np.ndarray | None:
        """Extract one capture sample-array from a result container entry."""
        iq_result = getattr(result, "iq_result", None)
        if not isinstance(iq_result, dict):
            return None
        values = iq_result.get(window_key, [])
        if not isinstance(values, list) or len(values) == 0:
            return None
        first = values[0]
        if hasattr(first, "iq_array"):
            latest = values[-1]
            iq_array = getattr(latest, "iq_array", None)
            if iq_array is None:
                return None
            return np.asarray(iq_array, dtype=np.complex128)
        return np.asarray(values, dtype=np.complex128)

    @staticmethod
    def _build_measurement_result(
        *,
        payload: Quel3ExecutionPayload,
        shot_samples: dict[str, dict[str, list[np.ndarray]]],
        sampling_period_ns: float | None,
        backend_sampling_period: float,
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

        base_sampling_period = (
            sampling_period_ns
            if sampling_period_ns is not None
            else backend_sampling_period
        )
        effective_sampling_period = (
            base_sampling_period * capture_decimation_factor
            if is_averaged
            else base_sampling_period
        )

        return Quel3BackendExecutionResult(
            status={},
            data=dict(measurement_data),
            config={"sampling_period_ns": effective_sampling_period},
        )

    @staticmethod
    def _load_quelware_api() -> tuple[
        QuelwareClientFactory,
        InstrumentResolverFactory,
        Callable[..., SequencerProtocol],
        InstrumentDriverFactory,
        Any,
        SetCaptureModeFactory,
    ]:
        """Import quelware helpers lazily and return required symbols."""
        client_module = import_module_with_workspace_fallback("quelware_client.client")
        resolver_module = import_module_with_workspace_fallback(
            "quelware_client.client.helpers.instrument_resolver"
        )
        sequencer_module = import_module_with_workspace_fallback(
            "quelware_client.client.helpers.sequencer"
        )
        directive_module = import_module_with_workspace_fallback(
            "quelware_core.entities.directives"
        )
        driver_module = import_module_with_workspace_fallback(
            "quelware_client.core.instrument_driver"
        )
        create_quelware_client: QuelwareClientFactory = (
            client_module.create_quelware_client
        )
        instrument_resolver_factory: InstrumentResolverFactory = (
            resolver_module.InstrumentResolver
        )
        sequencer_factory: Callable[..., SequencerProtocol] = sequencer_module.Sequencer
        create_instrument_driver_fixed_timeline: InstrumentDriverFactory = (
            driver_module.create_instrument_driver_fixed_timeline
        )
        capture_mode_enum = directive_module.CaptureMode
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
