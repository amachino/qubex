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

    waveform_name: str
    start_offset_ns: float
    gain: float = 1.0
    phase_offset_deg: float = 0.0


@dataclass(frozen=True)
class Quel3Waveform:
    """Registered waveform definition for one name in QuEL-3 execution."""

    iq_array: npt.NDArray[np.complex128]
    sampling_period_ns: float | None = None


@dataclass(frozen=True)
class Quel3FixedTimeline:
    """Fixed-timeline definition for one target in QuEL-3 execution."""

    events: tuple[Quel3WaveformEvent, ...]
    capture_windows: tuple[Quel3CaptureWindow, ...]
    length_ns: float


@dataclass(frozen=True)
class Quel3ExecutionPayload:
    """Execution payload for translating measurement requests to fixed timeline."""

    waveform_library: dict[str, Quel3Waveform]
    fixed_timelines: dict[str, Quel3FixedTimeline]
    interval_ns: float
    repeats: int
    mode: str
