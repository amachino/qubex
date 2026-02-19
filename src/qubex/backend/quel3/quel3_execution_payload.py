"""Execution payload models for QuEL-3 backend requests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Quel3CaptureWindow:
    """Capture window definition for QuEL-3 fixed-timeline execution."""

    name: str
    start_offset_ns: float
    length_ns: float


@dataclass(frozen=True)
class Quel3WaveformEvent:
    """Waveform event definition for one target in QuEL-3 execution."""

    start_offset_ns: float
    waveform: npt.NDArray[np.complex128]
    sampling_period_ns: float | None = None


@dataclass(frozen=True)
class Quel3TargetTimeline:
    """Timeline definition for one target in QuEL-3 execution."""

    sampling_period_ns: float
    events: tuple[Quel3WaveformEvent, ...]
    capture_windows: tuple[Quel3CaptureWindow, ...]
    length_ns: float
    modulation_frequency_hz: float | None = None


@dataclass(frozen=True)
class Quel3ExecutionPayload:
    """Execution payload for translating measurement requests to fixed timeline."""

    timelines: dict[str, Quel3TargetTimeline]
    instrument_aliases: dict[str, str]
    output_target_labels: dict[str, str]
    interval_ns: float
    repeats: int
    mode: str
    dsp_demodulation: bool
    enable_sum: bool
    enable_classification: bool
    line_param0: tuple[float, float, float]
    line_param1: tuple[float, float, float]
