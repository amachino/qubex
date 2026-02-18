"""Quel3 backend controller scaffold."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import threading
from collections import defaultdict
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from qubex.backend.quel1 import SAMPLING_PERIOD, Quel1BackendController


class Quel3BackendController(Quel1BackendController):
    """
    Quel3 controller scaffold with measurement-layer capability hints.

    Notes
    -----
    This class intentionally reuses the existing QuEL-1 control-plane
    implementation for shared configuration operations while exposing
    QuEL-3-specific measurement capability metadata.
    """

    MEASUREMENT_BACKEND_KIND: Literal["quel3"] = "quel3"
    MEASUREMENT_CONSTRAINT_MODE: Literal["relaxed"] = "relaxed"
    MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE: int = 4
    DEFAULT_SAMPLING_PERIOD: float = float(SAMPLING_PERIOD)

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        sampling_period_ns: float | None = None,
        alias_map: dict[str, str] | None = None,
        quelware_endpoint: str | None = None,
        quelware_port: int | None = None,
        trigger_wait: int | None = None,
    ) -> None:
        """
        Initialize a Quel3 controller scaffold.

        Parameters
        ----------
        config_path : str | Path | None, optional
            Optional config path passed to the shared base controller.
        sampling_period_ns : float | None, optional
            Session sampling period used by measurement-layer adapters.
        alias_map : dict[str, str] | None, optional
            Optional target-label to instrument-alias mapping.
        quelware_endpoint : str | None, optional
            Quelware API endpoint. Falls back to environment variable
            `QUBEX_QUELWARE_ENDPOINT` then `"localhost"`.
        quelware_port : int | None, optional
            Quelware API port. Falls back to environment variable
            `QUBEX_QUELWARE_PORT` then `50051`.
        trigger_wait : int | None, optional
            Trigger wait count passed to quelware session trigger. Falls back
            to environment variable `QUBEX_QUELWARE_TRIGGER_WAIT` then
            `1_000_000`.
        """
        super().__init__(config_path=config_path)
        if sampling_period_ns is not None:
            self.DEFAULT_SAMPLING_PERIOD = float(sampling_period_ns)
        self._alias_map = dict(alias_map or {})
        self._quelware_endpoint = (
            quelware_endpoint
            if quelware_endpoint is not None
            else os.getenv("QUBEX_QUELWARE_ENDPOINT", "localhost")
        )
        self._quelware_port = (
            int(quelware_port)
            if quelware_port is not None
            else int(os.getenv("QUBEX_QUELWARE_PORT", "50051"))
        )
        self._trigger_wait = (
            int(trigger_wait)
            if trigger_wait is not None
            else int(os.getenv("QUBEX_QUELWARE_TRIGGER_WAIT", "1000000"))
        )

    def set_instrument_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Replace full target-to-alias mapping for quelware execution."""
        self._alias_map = dict(alias_map)

    def update_instrument_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Update target-to-alias mapping for quelware execution."""
        self._alias_map.update(alias_map)

    def resolve_instrument_alias(self, target: str) -> str:
        """Resolve quelware instrument alias for a measurement target."""
        return self._alias_map.get(target, target)

    def execute_measurement(self, *, payload: object) -> object:
        """
        Execute a Quel3 measurement payload.

        Parameters
        ----------
        payload : object
            Backend execution payload produced by the measurement adapter.
        """
        from qubex.measurement.adapters.backend_adapter import Quel3ExecutionPayload

        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3BackendController.execute_measurement expects Quel3ExecutionPayload."
            )
        return self._run_coroutine(self._execute_measurement_async(payload))

    def _run_coroutine(self, coroutine: Coroutine[Any, Any, object]) -> object:
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
        payload: object,
    ) -> object:
        """Execute one fixed-timeline measurement flow via quelware."""
        from qubex.measurement.adapters.backend_adapter import Quel3ExecutionPayload

        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError(
                "Quel3BackendController.execute_measurement expects Quel3ExecutionPayload."
            )
        try:
            (
                create_quelware_client,
                InstrumentMapper,
                Sequencer,
                create_instrument_driver_fixed_timeline,
                get_all_instrument_ids_from_resource_infos,
            ) = self._load_quelware_api()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc

        aliases = sorted(set(payload.instrument_aliases.values()))
        if len(aliases) == 0:
            raise ValueError("Quel3ExecutionPayload must include instrument aliases.")

        sequencer = Sequencer(default_sampling_period_ns=self.DEFAULT_SAMPLING_PERIOD)
        for target, timeline in payload.timelines.items():
            alias = payload.instrument_aliases[target]
            waveform_name = self._waveform_name(target)
            sequencer.register_waveform(
                waveform_name,
                np.asarray(timeline.waveform, dtype=np.complex128),
                sampling_period_ns=float(timeline.sampling_period_ns),
            )
            sequencer.add_event(
                alias,
                waveform_name,
                start_offset_ns=0.0,
                gain=1.0,
                phase_offset_deg=0.0,
            )
            for window in timeline.capture_windows:
                sequencer.add_capture_window(
                    alias,
                    self._capture_window_key(target, window.name),
                    start_offset_ns=float(window.start_offset_ns),
                    length_ns=float(window.length_ns),
                )

        async with create_quelware_client(
            self._quelware_endpoint, self._quelware_port
        ) as client:
            mapper = InstrumentMapper()
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

            session_resources = [alias_to_id[alias] for alias in aliases]
            async with client.create_session(session_resources) as session:
                alias_to_driver: dict[str, Any] = {}
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
                        sampling_period_ns = float(sampling_period_fs) / 1e6

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
                    await session.trigger(wait=self._trigger_wait)
                    alias_results = {
                        alias: await driver.fetch_result()
                        for alias, driver in alias_to_driver.items()
                    }
                    for target, timeline in payload.timelines.items():
                        alias = payload.instrument_aliases[target]
                        result = alias_results[alias]
                        for window in timeline.capture_windows:
                            window_key = self._capture_window_key(target, window.name)
                            iq_datas = result.iq_datas.get(window_key, [])
                            if len(iq_datas) == 0:
                                continue
                            shot_samples[target][window.name].append(
                                np.asarray(iq_datas[-1].iq_array, dtype=np.complex128)
                            )

        return self._build_measurement_result(
            payload=payload,
            shot_samples=shot_samples,
            sampling_period_ns=sampling_period_ns,
        )

    @staticmethod
    def _waveform_name(target: str) -> str:
        """Return deterministic waveform name for one target."""
        return f"wf_{target}"

    @staticmethod
    def _capture_window_key(target: str, window_name: str) -> str:
        """Return deterministic capture-window key for one target/window pair."""
        return f"{target}:{window_name}"

    @staticmethod
    def _initialize_shot_samples(
        payload: object,
    ) -> dict[str, dict[str, list[np.ndarray]]]:
        """Initialize nested shot-sample container by target/capture window."""
        from qubex.measurement.adapters.backend_adapter import Quel3ExecutionPayload

        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError("Expected Quel3ExecutionPayload.")
        return {
            target: {window.name: [] for window in timeline.capture_windows}
            for target, timeline in payload.timelines.items()
        }

    @classmethod
    def _build_measurement_result(
        cls,
        *,
        payload: object,
        shot_samples: dict[str, dict[str, list[np.ndarray]]],
        sampling_period_ns: float | None,
    ) -> object:
        """Build canonical measurement result from per-shot capture samples."""
        from qubex.measurement.adapters.backend_adapter import Quel3ExecutionPayload
        from qubex.measurement.models.measurement_result import MeasurementResult

        if not isinstance(payload, Quel3ExecutionPayload):
            raise TypeError("Expected Quel3ExecutionPayload.")
        measurement_data: dict[str, list[np.ndarray]] = defaultdict(list)
        for target, timeline in payload.timelines.items():
            output_target = cls._measurement_target_label(target)
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
        if mode not in {"single", "avg"}:
            raise ValueError(f"Unsupported measurement mode: {mode}")

        return MeasurementResult(
            mode=cast(Literal["single", "avg"], mode),
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
                float(sampling_period_ns)
                if sampling_period_ns is not None
                else float(cls.DEFAULT_SAMPLING_PERIOD)
            ),
            avg_sample_stride=cls.MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE,
        )

    @staticmethod
    def _measurement_target_label(target: str) -> str:
        """Return canonical measurement target label used by legacy APIs."""
        from qubex.backend.target import Target

        try:
            return Target.qubit_label(target)
        except Exception:
            return target

    @staticmethod
    def _load_quelware_api() -> tuple[Any, Any, Any, Any, Any]:
        """Import quelware helpers lazily and return required symbols."""

        def _import_api() -> tuple[Any, Any, Any, Any, Any]:
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
            return (
                client_module.create_quelware_client,
                mapper_module.InstrumentMapper,
                sequencer_module.Sequencer,
                driver_module.create_instrument_driver_fixed_timeline,
                resource_module.get_all_instrument_ids_from_resource_infos,
            )

        try:
            return _import_api()
        except (ModuleNotFoundError, SyntaxError):
            Quel3BackendController._append_local_quelware_paths()
            return _import_api()

    @staticmethod
    def _append_local_quelware_paths() -> None:
        """Append local quelware source paths when present in the workspace."""
        root = Path(__file__).resolve().parents[4]
        candidates = (
            root / "packages" / "quelware-client" / "quelware-client" / "src",
            root / "packages" / "quelware-client" / "quelware-core" / "python" / "src",
        )
        for path in candidates:
            path_str = str(path)
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)
