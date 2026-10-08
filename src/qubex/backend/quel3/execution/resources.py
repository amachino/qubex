"""Validate payloads and extract the metadata needed for execution planning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces import InstrumentInfoProtocol
from qubex.backend.quel3.models.payload import (
    Quel3CaptureMode,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
)

from .timing import normalize_timeline, sample_count


@dataclass(frozen=True)
class ResourceRequirements:
    """Waveform counts per alias, RAW counts per receiver per shot, and timing."""

    waveform_samples: dict[str, int]
    capture_samples_per_shot: dict[str, int]
    timeline_duration_ns: float
    sampling_grid_fs: int


@dataclass(frozen=True)
class ExecutionConditions:
    """Settings that must agree when adjacent payloads share an execution."""

    n_iterations: int
    capture_mode: Quel3CaptureMode
    shot_interval_ns: float
    cable_delay_ns: dict[str, dict[str, float]] | None
    frequencies_hz: dict[str, float | None]
    capture_sampling_period_fs: int | None


@dataclass(frozen=True)
class PayloadPlanningInfo:
    """Planning input containing only resource counts, timing, and settings."""

    requirements: ResourceRequirements
    conditions: ExecutionConditions


@dataclass(frozen=True)
class PayloadAnalysis:
    """Normalized payload for construction and separate metadata for planning."""

    payload: Quel3ExecutionPayload
    planning: PayloadPlanningInfo


def capture_resource(port_id: str) -> str:
    """Identify a unit's physical receiver shared by transceiver instruments."""
    unit, separator, local = port_id.partition(":")
    if separator and unit:
        if local == "mon":
            return port_id
        match = re.fullmatch(r"(?:trx|rx)_p(\d+)(?:p\d+)?", local)
        if match:
            return f"{unit}:rx_p{int(match[1]):02d}"
    raise ValueError(f"Cannot identify capture resource for port {port_id!r}.")


def _finite_nonnegative(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative.")


class Quel3PayloadAnalyzer:
    """Prepare each payload once, without loading quelware or contacting hardware."""

    def __init__(self, *, default_sampling_period_ns: float) -> None:
        self._default_sampling_period_ns = default_sampling_period_ns

    def analyze_all(
        self,
        payloads: tuple[Quel3ExecutionPayload, ...],
        instrument_cache: InstrumentCache,
    ) -> tuple[PayloadAnalysis, ...]:
        """Analyze the complete batch against explicitly cached instruments."""
        instruments = {
            alias: instrument_cache.get(alias)
            for payload in payloads
            for alias, timeline in payload.fixed_timelines.items()
            if timeline.events or timeline.capture_windows
        }
        return tuple(self.analyze(payload, instruments) for payload in payloads)

    def analyze(
        self,
        payload: Quel3ExecutionPayload,
        instruments: dict[str, InstrumentInfoProtocol],
    ) -> PayloadAnalysis:
        """Validate one payload and extract independent planning metadata."""
        self._validate_execution_settings(payload)
        library = self._analyze_waveforms(payload)
        waveform_durations = {
            name: len(waveform.iq_array) * cast(float, waveform.sampling_period_ns)
            for name, waveform in library.items()
        }
        timelines = {
            alias: self._analyze_timeline(
                alias, timeline, instruments[alias], waveform_durations
            )
            for alias, timeline in payload.fixed_timelines.items()
            if timeline.events or timeline.capture_windows
        }
        if not timelines:
            raise ValueError(
                "Quel3ExecutionPayload has no waveform events or capture windows to execute."
            )
        normalized = replace(
            payload, fixed_timelines=timelines, waveform_library=library
        )
        return PayloadAnalysis(normalized, self._planning_info(normalized, instruments))

    def _analyze_timeline(
        self,
        alias: str,
        timeline: Quel3FixedTimeline,
        instrument: InstrumentInfoProtocol,
        waveform_durations_ns: dict[str, float],
    ) -> Quel3FixedTimeline:
        sampling_fs = instrument.config.sampling_period_fs
        if sampling_fs <= 0 or instrument.config.timeline_step_samples <= 0:
            raise ValueError(f"Invalid cached sampling configuration for {alias!r}.")
        self._validate_timeline(alias, timeline)
        normalized = normalize_timeline(timeline, sampling_fs, waveform_durations_ns)
        return replace(
            normalized,
            events=tuple(
                sorted(normalized.events, key=lambda event: event.start_offset_ns)
            ),
            capture_windows=tuple(
                sorted(
                    normalized.capture_windows,
                    key=lambda window: (window.start_offset_ns, window.length_ns),
                )
            ),
        )

    @staticmethod
    def _planning_info(
        payload: Quel3ExecutionPayload,
        instruments: dict[str, InstrumentInfoProtocol],
    ) -> PayloadPlanningInfo:
        waveform_samples: dict[str, int] = {}
        captures: dict[str, int] = {}
        sampling_periods = []
        capture_periods: set[int] = set()
        for alias, timeline in payload.fixed_timelines.items():
            info = instruments[alias]
            sampling_fs = info.config.sampling_period_fs
            sampling_periods.append(sampling_fs)
            waveform_samples[alias] = sum(
                len(payload.waveform_library[name].iq_array)
                for name in {event.waveform_name for event in timeline.events}
            )
            if timeline.capture_windows:
                capture_periods.add(sampling_fs)
                receiver = capture_resource(info.port_id)
                count = sum(
                    sample_count(window.length_ns, sampling_fs / 1e6)
                    for window in timeline.capture_windows
                )
                captures[receiver] = captures.get(receiver, 0) + count
        if len(capture_periods) > 1:
            raise ValueError("Capture aliases must agree on sampling period.")
        requirements = ResourceRequirements(
            waveform_samples=waveform_samples,
            capture_samples_per_shot=captures,
            timeline_duration_ns=max(
                t.length_ns for t in payload.fixed_timelines.values()
            ),
            sampling_grid_fs=math.lcm(*sampling_periods),
        )
        conditions = ExecutionConditions(
            n_iterations=payload.n_iterations,
            capture_mode=payload.capture_mode,
            shot_interval_ns=payload.shot_interval_ns,
            cable_delay_ns=payload.cable_delay_ns,
            frequencies_hz={
                alias: t.frequency_hz for alias, t in payload.fixed_timelines.items()
            },
            capture_sampling_period_fs=next(iter(capture_periods), None),
        )
        return PayloadPlanningInfo(requirements, conditions)

    @staticmethod
    def _validate_execution_settings(payload: Quel3ExecutionPayload) -> None:
        if (
            isinstance(payload.n_iterations, bool)
            or not isinstance(payload.n_iterations, int)
            or payload.n_iterations < 1
        ):
            raise ValueError("n_iterations must be a positive integer.")
        _finite_nonnegative(payload.shot_interval_ns, "shot_interval_ns")
        if payload.capture_mode not in tuple(
            mode for mode in Quel3CaptureMode if mode != Quel3CaptureMode.UNSPECIFIED
        ):
            raise ValueError(f"Unsupported capture mode: {payload.capture_mode}.")

    def _analyze_waveforms(
        self, payload: Quel3ExecutionPayload
    ) -> dict[str, Quel3Waveform]:
        referenced = {
            event.waveform_name
            for timeline in payload.fixed_timelines.values()
            for event in timeline.events
        }
        library = {}
        for name in sorted(referenced):
            if name not in payload.waveform_library:
                raise ValueError(f"Unknown waveform name in event: {name}.")
            waveform = payload.waveform_library[name]
            array = waveform.iq_array
            period = (
                waveform.sampling_period_ns
                if waveform.sampling_period_ns is not None
                else self._default_sampling_period_ns
            )
            if not math.isfinite(period) or period <= 0:
                raise ValueError(
                    "Waveform sampling period must be finite and positive."
                )
            if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Waveform {name!r} must contain finite one-dimensional IQ samples."
                )
            if np.max(np.abs(array)) > 1 + 1e-7:
                raise ValueError(f"Waveform {name!r} magnitude exceeds 1 + 1e-7.")
            library[name] = replace(waveform, sampling_period_ns=period)
        return library

    @staticmethod
    def _validate_timeline(alias: str, timeline: Quel3FixedTimeline) -> None:
        _finite_nonnegative(timeline.length_ns, "timeline length")
        if timeline.frequency_hz is not None and not math.isfinite(
            timeline.frequency_hz
        ):
            raise ValueError(f"Nonfinite frequency for {alias!r}.")
        for event in timeline.events:
            _finite_nonnegative(event.start_offset_ns, "event offset")
            if not math.isfinite(event.gain) or not math.isfinite(
                event.phase_offset_deg
            ):
                raise ValueError("Event gain and phase must be finite.")
        names: set[str] = set()
        for window in timeline.capture_windows:
            if window.name in names:
                raise ValueError(
                    f"Duplicate capture window name `{window.name}` for alias `{alias}`."
                )
            names.add(window.name)
            _finite_nonnegative(window.start_offset_ns, "capture offset")
            if not math.isfinite(window.length_ns) or window.length_ns <= 0:
                raise ValueError("Capture length must be finite and positive.")
