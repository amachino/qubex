"""Monitor workflows built from QuEL-3 backend components."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from qubex.backend.backend_controller import BackendExecutionRequest
from qubex.backend.quel3.builders import Quel3PulseEventBuilder
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    Quel3ExecutionManager,
    Quel3HardwareStateReader,
)
from qubex.backend.quel3.models import (
    InstrumentSpec,
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
)

if TYPE_CHECKING:
    from qxpulse import PulseSchedule


class Quel3MonitorTool:
    """
    Own QuEL-3 monitor capture workflows and temporary instrument restoration.

    Use the supplied managers and reader with one shared instrument cache.
    """

    def __init__(
        self,
        *,
        configuration_manager: Quel3ConfigurationManager,
        execution_manager: Quel3ExecutionManager,
        hardware_state_reader: Quel3HardwareStateReader,
        instrument_cache: InstrumentCache,
    ) -> None:
        """
        Bind monitor operations to existing backend components.

        Parameters
        ----------
        configuration_manager : Quel3ConfigurationManager
            Manager for instrument deployment and unit configuration.
        execution_manager : Quel3ExecutionManager
            Manager for executing monitor capture payloads.
        hardware_state_reader : Quel3HardwareStateReader
            Reader for the original instrument configuration.
        instrument_cache : InstrumentCache
            Shared cache updated by deployment and used for execution.
        """
        self._configuration_manager = configuration_manager
        self._execution_manager = execution_manager
        self._hardware_state_reader = hardware_state_reader
        self._instrument_cache = instrument_cache

    def configure_mode(self, *, unit_label: str, mode: str = "loopback") -> str:
        """
        Set a unit's monitor mode before deploying output and monitor instruments.

        Parameters
        ----------
        unit_label : str
            Exact QuEL-3 unit label.
        mode : str, default="loopback"
            Value supported by the unit's `quel3.monitor.mode` control.

        Returns
        -------
        str
            Applied mode reported by quelware.

        Raises
        ------
        RuntimeError
            If the supplied cache contains instruments on this unit. Clear
            the unit through the configuration manager before changing mode.

        Notes
        -----
        Quelware also rejects a mode change if uncached instruments remain
        deployed on the unit. Re-deploy instruments after changing the mode.
        """
        if any(
            info.port_id.startswith(f"{unit_label}:")
            for info in self._instrument_cache.snapshot().values()
        ):
            raise RuntimeError(
                "Clear deployed instruments with clear_instruments() before "
                "changing QuEL-3 monitor mode."
            )
        return self._configuration_manager.configure_monitor_mode(
            unit_label=unit_label, mode=mode
        )

    def run_schedule(
        self,
        *,
        unit_label: str,
        pulse_schedule: PulseSchedule,
        capture_start_ns: float = 0.0,
        capture_length_ns: float | None = None,
        n_iterations: int = 1,
        shot_interval_ns: float = 0.0,
        parallel: bool = True,
    ) -> dict[str, npt.NDArray[np.complex128]]:
        """
        Run each schedule target on the monitor port and restore the unit state.

        Parameters
        ----------
        unit_label : str
            Unit containing every scheduled output instrument.
        pulse_schedule : PulseSchedule
            Valid schedule whose target names are output instrument aliases.
        capture_start_ns : float, default=0.0
            Capture start time relative to the output trigger, in ns.
        capture_length_ns : float | None, optional
            Capture duration in ns. Defaults to the schedule duration.
        n_iterations : int, default=1
            Number of unaveraged captures.
        shot_interval_ns : float, default=0.0
            Idle interval between iterations, in ns.
        parallel : bool, default=True
            Whether to parallelize instrument execution phases.

        Returns
        -------
        dict[str, NDArray[np.complex128]]
            Raw IQ arrays keyed by schedule target, each with shape
            `(n_iterations, samples)`.

        Notes
        -----
        Schedule frequencies are in GHz. A target without a frequency uses the
        center of its live instrument's frequency range. The same frequency is
        applied to the output and monitor instruments. The method reads the
        live instruments, clears the selected unit, enters loopback mode, and
        executes targets sequentially. It clears temporary instruments and
        restores the original monitor mode and instrument configuration even
        if deployment or execution fails. Blanks advance event offsets without
        allocating zero-filled waveform samples.
        """
        if not unit_label.strip():
            raise ValueError("Unit label must not be empty.")
        labels = pulse_schedule.labels
        if not labels:
            raise ValueError("Monitor PulseSchedule must have at least one channel.")
        if not pulse_schedule.is_valid():
            raise ValueError("Monitor PulseSchedule is invalid.")
        resolved_capture_length_ns = (
            pulse_schedule.duration if capture_length_ns is None else capture_length_ns
        )
        self._validate_monitor_capture_settings(
            capture_start_ns=capture_start_ns,
            capture_length_ns=resolved_capture_length_ns,
            n_iterations=n_iterations,
            shot_interval_ns=shot_interval_ns,
        )
        original_cache = InstrumentCache()
        original_cache.replace_all(
            instrument_infos=self._hardware_state_reader.read_instrument_infos(
                unit_labels=(unit_label,), parallel=parallel
            )
        )
        original_configuration = original_cache.export_configuration()
        original_specs = {
            spec.alias: spec for spec in original_configuration.instruments
        }
        occupied_aliases = set(self._instrument_cache.snapshot()) | set(original_specs)
        monitor_alias = "_qubex_monitor"
        suffix = 0
        while monitor_alias in occupied_aliases:
            suffix += 1
            monitor_alias = f"_qubex_monitor_{suffix}"
        prepared: dict[str, tuple[Quel3FixedTimeline, dict[str, Quel3Waveform]]] = {}
        for label in labels:
            try:
                spec = original_specs[label]
            except KeyError as exc:
                raise ValueError(
                    f"Monitor PulseSchedule target {label!r} has no instrument "
                    f"on unit {unit_label!r}."
                ) from exc
            if spec.port_id.partition(":")[0] != unit_label:
                raise ValueError(
                    f"Monitor PulseSchedule target {label!r} belongs to another unit."
                )
            if spec.role == "RECEIVER" or spec.port_id.endswith(":mon"):
                raise ValueError(
                    f"Monitor target {label!r} is not an output instrument."
                )
            frequency_ghz = pulse_schedule.get_frequency(label)
            frequency_hz = (
                spec.frequency_range_min_hz / 2 + spec.frequency_range_max_hz / 2
                if frequency_ghz is None
                else frequency_ghz * 1e9
            )
            if not math.isfinite(frequency_hz):
                raise ValueError(
                    f"Monitor PulseSchedule target {label!r} requires a finite frequency."
                )
            if not (
                spec.frequency_range_min_hz
                <= frequency_hz
                <= spec.frequency_range_max_hz
            ):
                raise ValueError(
                    f"Monitor PulseSchedule frequency for {label!r} is outside "
                    f"the instrument range."
                )
            waveform_library: dict[str, Quel3Waveform] = {}
            events, _ = Quel3PulseEventBuilder.build(
                sequence=pulse_schedule.get_sequence(label, copy=False),
                waveform_name_by_shape_key={},
                waveform_library=waveform_library,
                waveform_index=0,
            )
            prepared[label] = (
                Quel3FixedTimeline(
                    events=events,
                    capture_windows=(),
                    length_ns=pulse_schedule.duration,
                    frequency_hz=frequency_hz,
                ),
                waveform_library,
            )

        original_mode = self._configuration_manager.get_monitor_mode(
            unit_label=unit_label
        )
        captured: dict[str, npt.NDArray[np.complex128]] = {}
        try:
            self._configuration_manager.clear_instruments(
                unit_label=unit_label,
                instrument_cache=self._instrument_cache,
                parallel=parallel,
            )
            self.configure_mode(unit_label=unit_label, mode="loopback")
            for label in labels:
                spec = original_specs[label]
                timeline, waveform_library = prepared[label]
                self._configuration_manager.deploy_instrument(
                    instrument=spec,
                    instrument_cache=self._instrument_cache,
                    hardware_state_reader=self._hardware_state_reader,
                    append=False,
                    parallel=parallel,
                )
                self._configuration_manager.deploy_instrument(
                    instrument_cache=self._instrument_cache,
                    hardware_state_reader=self._hardware_state_reader,
                    instrument=InstrumentSpec(
                        port_id=f"{unit_label}:mon",
                        alias=monitor_alias,
                        role="RECEIVER",
                        frequency_range_min_hz=spec.frequency_range_min_hz,
                        frequency_range_max_hz=spec.frequency_range_max_hz,
                    ),
                    append=False,
                    parallel=parallel,
                )
                captured[label] = self._execute_target(
                    target=label,
                    timeline=timeline,
                    waveform_library=waveform_library,
                    monitor_alias=monitor_alias,
                    capture_start_ns=capture_start_ns,
                    capture_length_ns=resolved_capture_length_ns,
                    n_iterations=n_iterations,
                    shot_interval_ns=shot_interval_ns,
                    parallel=parallel,
                )
        finally:
            self._configuration_manager.clear_instruments(
                unit_label=unit_label,
                instrument_cache=self._instrument_cache,
                parallel=parallel,
            )
            self.configure_mode(unit_label=unit_label, mode=original_mode)
            self._configuration_manager.deploy_instruments(
                configuration=original_configuration,
                instrument_cache=self._instrument_cache,
                hardware_state_reader=self._hardware_state_reader,
                parallel=parallel,
            )
        return captured

    def _execute_target(
        self,
        *,
        target: str,
        timeline: Quel3FixedTimeline,
        waveform_library: dict[str, Quel3Waveform],
        monitor_alias: str,
        capture_start_ns: float,
        capture_length_ns: float,
        n_iterations: int,
        shot_interval_ns: float,
        parallel: bool,
    ) -> npt.NDArray[np.complex128]:
        """Execute one prepared output timeline and return its monitor IQ."""
        timeline_length_ns = max(
            timeline.length_ns, capture_start_ns + capture_length_ns
        )
        payload = Quel3ExecutionPayload(
            waveform_library=waveform_library,
            fixed_timelines={
                target: timeline,
                monitor_alias: Quel3FixedTimeline(
                    events=(),
                    capture_windows=(
                        Quel3CaptureWindow(
                            "monitor_iq", capture_start_ns, capture_length_ns
                        ),
                    ),
                    length_ns=timeline_length_ns,
                    frequency_hz=timeline.frequency_hz,
                ),
            },
            n_iterations=n_iterations,
            shot_interval_ns=shot_interval_ns,
            capture_mode=Quel3CaptureMode.RAW_WAVEFORMS,
        )
        result = self._execution_manager.execute_sync(
            request=BackendExecutionRequest(payload=payload),
            instrument_cache=self._instrument_cache,
            parallel=parallel,
        )
        if not isinstance(result, Quel3BackendExecutionResult):
            raise TypeError("QuEL-3 execution did not return a backend result.")
        try:
            captured = np.asarray(result.data[monitor_alias][0], dtype=np.complex128)
        except (KeyError, IndexError) as exc:
            raise RuntimeError("QuEL-3 monitor execution returned no IQ data.") from exc
        if captured.size == 0:
            raise RuntimeError("QuEL-3 monitor execution returned no IQ data.")
        return captured

    @staticmethod
    def _validate_monitor_capture_settings(
        *,
        capture_start_ns: float,
        capture_length_ns: float,
        n_iterations: int,
        shot_interval_ns: float,
    ) -> None:
        """Reject invalid capture settings before a managed run changes hardware."""
        if n_iterations < 1:
            raise ValueError("n_iterations must be positive.")
        if not math.isfinite(shot_interval_ns) or shot_interval_ns < 0:
            raise ValueError("shot_interval_ns must be finite and nonnegative.")
        if not math.isfinite(capture_start_ns) or capture_start_ns < 0:
            raise ValueError("capture_start_ns must be finite and nonnegative.")
        if not math.isfinite(capture_length_ns) or capture_length_ns <= 0:
            raise ValueError("capture_length_ns must be finite and positive.")
