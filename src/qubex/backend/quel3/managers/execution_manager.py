"""Execution manager for QuEL-3 backend controller."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from collections import defaultdict
from collections.abc import Coroutine, Iterable
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import numpy.typing as npt

from qubex.backend.quel3.managers.sequencer_builder import Quel3SequencerBuilder
from qubex.backend.quel3.quel3_execution_payload import Quel3ExecutionPayload
from qubex.backend.quel3.quel3_runtime_context import Quel3RuntimeContextReader
from qubex.backend.target_registry import TargetRegistry

if TYPE_CHECKING:
    from qubex.measurement.models.measurement_result import MeasurementResult

ExecutionMode = Literal["serial", "parallel"]


class _CompileSequencerProtocol(Protocol):
    """Sequencer protocol extended with directive export for runtime execution."""

    def register_waveform(
        self,
        name: str,
        waveform: npt.ArrayLike,
        sampling_period_ns: float | None = None,
    ) -> None:
        """Register one waveform in the sequencer library."""
        ...

    def add_event(
        self,
        instrument_alias: str,
        waveform_name: str,
        start_offset_ns: float,
        gain: float = 1.0,
        phase_offset_deg: float = 0.0,
    ) -> None:
        """Append one waveform event to the timeline."""
        ...

    def add_capture_window(
        self,
        instrument_alias: str,
        window_name: str,
        start_offset_ns: float,
        length_ns: float,
    ) -> None:
        """Append one capture window to the timeline."""
        ...

    def export_set_fixed_timeline_directive(
        self,
        instrument_alias: str,
        sampling_period_fs: int,
    ) -> object:
        """Export fixed-timeline directive for one instrument alias."""
        ...


class _IqDataProtocol(Protocol):
    """Minimal IQ-data protocol returned by quelware drivers."""

    @property
    def iq_array(self) -> npt.ArrayLike:
        """Return raw IQ samples."""
        ...


class _FixedTimelineResultProtocol(Protocol):
    """Minimal fixed-timeline result protocol."""

    @property
    def iq_datas(self) -> dict[str, list[_IqDataProtocol]]:
        """Return capture-window IQ data mapping."""
        ...


class _InstrumentConfigProtocol(Protocol):
    """Minimal instrument-config protocol."""

    @property
    def sampling_period_fs(self) -> int:
        """Return sampling period in femtoseconds."""
        ...


class _InstrumentDriverProtocol(Protocol):
    """Minimal instrument driver protocol used in execution."""

    @property
    def instrument_config(self) -> _InstrumentConfigProtocol:
        """Return instrument runtime configuration."""
        ...

    async def apply(self, directive: object) -> None:
        """Apply one fixed-timeline directive."""
        ...

    async def setup(self) -> None:
        """Set up the instrument for trigger."""
        ...

    async def fetch_result(self) -> _FixedTimelineResultProtocol:
        """Fetch one fixed-timeline execution result."""
        ...


class _SessionProtocol(Protocol):
    """Minimal quelware session protocol."""

    async def trigger(self) -> None:
        """Trigger one fixed-timeline session run."""
        ...


class _SessionContextManagerProtocol(Protocol):
    """Async context manager protocol for sessions."""

    async def __aenter__(self) -> _SessionProtocol:
        """Enter session context."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Exit session context."""
        ...


class _QuelwareClientProtocol(Protocol):
    """Minimal quelware client protocol for execution."""

    async def list_resource_infos(self) -> object:
        """List available resources."""
        ...

    async def get_instrument_info(self, resource_id: object) -> object:
        """Get instrument info for one resource ID."""
        ...

    def create_session(
        self,
        resource_id: object,
    ) -> _SessionContextManagerProtocol:
        """Create one execution session for one selected resource."""
        ...


class _QuelwareClientContextManager(Protocol):
    """Async context manager protocol for quelware clients."""

    async def __aenter__(self) -> _QuelwareClientProtocol:
        """Enter client context."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Exit client context."""
        ...


class _QuelwareClientFactory(Protocol):
    """Factory protocol for quelware clients."""

    def __call__(self, endpoint: str, port: int) -> _QuelwareClientContextManager:
        """Create one quelware client context manager."""
        ...


class _InstrumentMapperProtocol(Protocol):
    """Minimal instrument-mapper protocol."""

    def add_instrument_info(self, instrument_info: object) -> None:
        """Add one instrument info object."""
        ...

    def build_alias_to_id_map(self) -> dict[str, object]:
        """Build mapping from alias to resource ID."""
        ...

    def get_instrument_info(self, resource_id: object) -> object:
        """Get instrument info by resource ID."""
        ...


class _InstrumentMapperFactory(Protocol):
    """Factory protocol for instrument mapper instances."""

    def __call__(self) -> _InstrumentMapperProtocol:
        """Create one instrument mapper instance."""
        ...


class _SequencerFactory(Protocol):
    """Factory protocol for sequencer instances."""

    def __call__(
        self,
        *,
        default_sampling_period_ns: float,
    ) -> _CompileSequencerProtocol:
        """Create one sequencer instance."""
        ...


class _InstrumentDriverFactory(Protocol):
    """Factory protocol for fixed-timeline instrument drivers."""

    def __call__(
        self,
        session: _SessionProtocol,
        instrument_info: object,
    ) -> _InstrumentDriverProtocol:
        """Create one instrument driver for a resource."""
        ...


class _ResourceIdResolver(Protocol):
    """Protocol for resolving resource IDs from resource-info payload."""

    def __call__(self, resource_infos: object) -> Iterable[object]:
        """Resolve all instrument resource IDs."""
        ...


class Quel3ExecutionManager:
    """Handle backend execution entrypoints for QuEL-3 controller."""

    def __init__(self, *, runtime_context: Quel3RuntimeContextReader) -> None:
        self._runtime_context = runtime_context
        self._sequencer_builder = Quel3SequencerBuilder()

    def resolve_instrument_alias(self, target: str) -> str:
        """Resolve quelware instrument alias for a measurement target."""
        return self._runtime_context.alias_map.get(target, target)

    def execute_payload(self, *, payload: Quel3ExecutionPayload) -> object:
        """
        Execute a QuEL-3 measurement payload.

        Parameters
        ----------
        payload : Quel3ExecutionPayload
            Backend execution payload produced by measurement adapter.
        """
        return self._run_coroutine(self._execute_measurement_async(payload))

    @staticmethod
    def _run_coroutine(coroutine: Coroutine[object, object, object]) -> object:
        """Run async quelware workflow in sync measurement entrypoint."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        result_holder: dict[str, object] = {}
        error_holder: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result_holder["value"] = asyncio.run(coroutine)
            except BaseException as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder["value"]

    async def _execute_measurement_async(
        self,
        payload: Quel3ExecutionPayload,
    ) -> object:
        """Execute one fixed-timeline measurement flow via quelware."""
        aliases = sorted(set(payload.instrument_aliases.values()))
        if len(aliases) == 0:
            raise ValueError("Quel3ExecutionPayload must include instrument aliases.")
        if len(aliases) > 1:
            raise NotImplementedError(
                "Quel3 execution currently supports a single instrument alias. "
                "This keeps behavior within quelware-client examples."
            )

        try:
            (
                create_quelware_client,
                instrument_mapper_factory,
                sequencer_factory,
                create_instrument_driver_fixed_timeline,
                get_all_instrument_ids_from_resource_infos,
            ) = self.load_quelware_api()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc

        sequencer = self._sequencer_builder.build(
            payload=payload,
            sequencer_factory=sequencer_factory,
            default_sampling_period_ns=self._runtime_context.default_sampling_period,
        )

        async with create_quelware_client(
            self._runtime_context.quelware_endpoint,
            self._runtime_context.quelware_port,
        ) as client:
            mapper = instrument_mapper_factory()
            for resource_id in get_all_instrument_ids_from_resource_infos(
                await client.list_resource_infos()
            ):
                mapper.add_instrument_info(
                    await client.get_instrument_info(resource_id)
                )
            alias_to_id = mapper.build_alias_to_id_map()
            missing_aliases = [alias for alias in aliases if alias not in alias_to_id]
            if missing_aliases:
                available = sorted(alias_to_id.keys())
                raise ValueError(
                    f"Quelware aliases are missing: {missing_aliases}. available={available}"
                )

            alias = aliases[0]
            async with client.create_session(alias_to_id[alias]) as session:
                alias_to_driver: dict[str, _InstrumentDriverProtocol] = {}
                sampling_period_ns: float | None = None
                for alias in aliases:
                    instrument_info = mapper.get_instrument_info(alias_to_id[alias])
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
                    directive = sequencer.export_set_fixed_timeline_directive(
                        alias,
                        driver.instrument_config.sampling_period_fs,
                    )
                    await driver.apply(directive)
                    await driver.setup()

                shot_count = (
                    payload.repeats
                    if payload.mode == "avg" and payload.repeats > 1
                    else 1
                )
                shot_samples = self._initialize_shot_samples(payload)
                for _ in range(shot_count):
                    await session.trigger()
                    alias_results = {
                        alias: await driver.fetch_result()
                        for alias, driver in alias_to_driver.items()
                    }
                    for target, timeline in payload.timelines.items():
                        alias = payload.instrument_aliases[target]
                        result = alias_results[alias]
                        for window in timeline.capture_windows:
                            window_key = self._sequencer_builder.capture_window_key(
                                target, window.name
                            )
                            iq_datas = result.iq_datas.get(window_key, [])
                            if len(iq_datas) == 0:
                                continue
                            shot_samples[target][window.name].append(
                                np.asarray(iq_datas[-1].iq_array, dtype=np.complex128)
                            )

        return self.build_measurement_result(
            payload=payload,
            shot_samples=shot_samples,
            sampling_period_ns=sampling_period_ns,
            default_sampling_period=self._runtime_context.default_sampling_period,
            avg_sample_stride=self._runtime_context.measurement_result_avg_sample_stride,
        )

    @staticmethod
    def _initialize_shot_samples(
        payload: object,
    ) -> dict[str, dict[str, list[np.ndarray]]]:
        """Initialize nested shot-sample container by target/capture window."""
        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError("Expected Quel3ExecutionPayload.")
        return {
            target: {window.name: [] for window in timeline.capture_windows}
            for target, timeline in payload.timelines.items()
        }

    @staticmethod
    def build_measurement_result(
        *,
        payload: object,
        shot_samples: dict[str, dict[str, list[np.ndarray]]],
        sampling_period_ns: float | None,
        default_sampling_period: float,
        avg_sample_stride: int,
    ) -> MeasurementResult:
        """Build canonical measurement result from per-shot capture samples."""
        from qubex.measurement.models.measurement_result import MeasurementResult

        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError("Expected Quel3ExecutionPayload.")
        measurement_data: dict[str, list[np.ndarray]] = defaultdict(list)
        output_target_labels = payload.output_target_labels
        for target, timeline in payload.timelines.items():
            output_target = output_target_labels.get(
                target,
                Quel3ExecutionManager._measurement_target_label(target),
            )
            for window in timeline.capture_windows:
                samples = shot_samples.get(target, {}).get(window.name, [])
                if len(samples) == 0:
                    measurement_data[output_target].append(
                        np.array([], dtype=np.complex128)
                    )
                    continue
                if payload.mode == "avg":
                    values = np.stack(samples, axis=0)
                    measurement_data[output_target].append(np.mean(values, axis=0))
                else:
                    measurement_data[output_target].append(samples[0])

        mode = payload.mode
        if mode == "single":
            result_mode: Literal["single", "avg"] = "single"
        elif mode == "avg":
            result_mode = "avg"
        else:
            raise ValueError(f"Unsupported measurement mode: {mode}")

        return MeasurementResult(
            mode=result_mode,
            data=dict(measurement_data),
            device_config={},
            measurement_config={
                "mode": mode,
                "shots": payload.repeats,
                "interval_ns": payload.interval_ns,
                "dsp_demodulation": payload.dsp_demodulation,
                "enable_sum": payload.enable_sum,
                "enable_classification": payload.enable_classification,
                "line_param0": payload.line_param0,
                "line_param1": payload.line_param1,
            },
            sampling_period_ns=(
                sampling_period_ns
                if sampling_period_ns is not None
                else default_sampling_period
            ),
            avg_sample_stride=avg_sample_stride,
        )

    @staticmethod
    def _measurement_target_label(target: str) -> str:
        """Return canonical measurement target label used by legacy APIs."""
        try:
            return TargetRegistry().resolve_qubit_label(target, allow_legacy=True)
        except Exception:
            return target

    @staticmethod
    def load_quelware_api() -> tuple[
        _QuelwareClientFactory,
        _InstrumentMapperFactory,
        _SequencerFactory,
        _InstrumentDriverFactory,
        _ResourceIdResolver,
    ]:
        """Import quelware helpers lazily and return required symbols."""

        def _import_api() -> tuple[
            _QuelwareClientFactory,
            _InstrumentMapperFactory,
            _SequencerFactory,
            _InstrumentDriverFactory,
            _ResourceIdResolver,
        ]:
            resource_module = importlib.import_module("quelware_core.entities.resource")
            client_module = importlib.import_module("quelware_client.client")
            mapper_module = importlib.import_module(
                "quelware_client.client.helpers.instrument_mapper"
            )
            sequencer_module = importlib.import_module(
                "quelware_client.client.helpers.sequencer"
            )
            driver_module = importlib.import_module(
                "quelware_client.core.instrument_driver"
            )
            create_quelware_client: _QuelwareClientFactory = (
                client_module.create_quelware_client
            )
            instrument_mapper_factory: _InstrumentMapperFactory = (
                mapper_module.InstrumentMapper
            )
            sequencer_factory: _SequencerFactory = sequencer_module.Sequencer
            create_instrument_driver_fixed_timeline: _InstrumentDriverFactory = (
                driver_module.create_instrument_driver_fixed_timeline
            )
            get_all_instrument_ids_from_resource_infos: _ResourceIdResolver = (
                resource_module.get_all_instrument_ids_from_resource_infos
            )
            return (
                create_quelware_client,
                instrument_mapper_factory,
                sequencer_factory,
                create_instrument_driver_fixed_timeline,
                get_all_instrument_ids_from_resource_infos,
            )

        try:
            return _import_api()
        except (ModuleNotFoundError, SyntaxError):
            Quel3ExecutionManager.append_local_quelware_paths()
            return _import_api()

    @staticmethod
    def append_local_quelware_paths() -> None:
        """Append local quelware source paths when present in the workspace."""
        root = Path(__file__).resolve().parents[5]
        candidates = (
            root / "packages" / "quelware-client" / "quelware-client" / "src",
            root / "packages" / "quelware-client" / "quelware-core" / "python" / "src",
        )
        for path in candidates:
            path_str = str(path)
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)
