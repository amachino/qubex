"""Compile QuEL-3 waveform events from a pulse sequence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np
from qxpulse import Blank, Pulse, PulseArray

from qubex.backend.quel3.models import Quel3Waveform, Quel3WaveformEvent
from qubex.backend.quel3.quel3_backend_constants import READOUT_SAMPLING_PERIOD_NS


class _WaveformLibrary:
    def __init__(self) -> None:
        self.waveforms: dict[str, Quel3Waveform] = {}
        self._names: dict[tuple[float, bytes], str] = {}

    def register(self, samples: np.ndarray, period: float) -> str:
        key = (period, hashlib.blake2b(samples.tobytes(), digest_size=16).digest())
        name = self._names.get(key)
        if name is None:
            name = f"waveform_{len(self.waveforms):04d}"
            # A small slice must not retain the entire uncompressed IQ buffer.
            self.waveforms[name] = Quel3Waveform(
                samples.copy() if samples.base is not None else samples, period
            )
            self._names[key] = name
        return name


class Quel3PulseEventBuilder:
    """Build sparse events and shared waveform entries from `PulseArray`."""

    def build(
        self, targets: dict[str, tuple[PulseArray, bool]]
    ) -> tuple[dict[str, tuple[Quel3WaveformEvent, ...]], dict[str, Quel3Waveform]]:
        """
        Compile the targets of one payload into events and a shared waveform library.

        Each target maps to its pulse sequence and a flag selecting the readout grid.
        Event offsets are relative to the start of each target's sequence.
        """
        library = _WaveformLibrary()
        templates: dict[
            tuple[str, bool, type[Pulse]], tuple[Quel3WaveformEvent, ...]
        ] = {}
        events_by_target: dict[str, tuple[Quel3WaveformEvent, ...]] = {}
        for label, (sequence, target_is_read) in targets.items():
            events: list[Quel3WaveformEvent] = []
            offset_ns = 0.0
            for waveform in sequence.get_flattened_waveforms(apply_frame_shifts=True):
                if isinstance(waveform, Blank):
                    offset_ns += waveform.duration
                    continue
                if not isinstance(waveform, Pulse):
                    raise TypeError(
                        f"Unsupported waveform type in PulseArray: {type(waveform).__name__}."
                    )
                if waveform.length:
                    key = (waveform.shape_hash, target_is_read, type(waveform))
                    if key not in templates:
                        templates[key] = self._compile(
                            waveform, target_is_read, library
                        )
                    events.extend(
                        replace(
                            part,
                            start_offset_ns=offset_ns + part.start_offset_ns,
                            gain=waveform.scale,
                            phase_offset_deg=math.degrees(waveform.phase)
                            + part.phase_offset_deg,
                        )
                        for part in templates[key]
                    )
                offset_ns += waveform.duration
            events_by_target[label] = tuple(events)
        return events_by_target, library.waveforms

    def _compile(
        self, pulse: Pulse, target_is_read: bool, library: _WaveformLibrary
    ) -> tuple[Quel3WaveformEvent, ...]:
        shape, period = self._normalize_waveform(
            target_is_read=target_is_read,
            shape=pulse.shape_values,
            sampling_period_ns=pulse.sampling_period,
        )
        if not np.all(np.isfinite(shape)):
            raise ValueError("Waveform IQ values must be finite.")
        return (Quel3WaveformEvent(library.register(shape, period), 0.0),)

    @staticmethod
    def _normalize_waveform(
        *,
        target_is_read: bool,
        shape: np.ndarray,
        sampling_period_ns: float,
    ) -> tuple[np.ndarray, float]:
        """
        Normalize waveform sampling periods for QuEL-3 target classes.

        This is a temporary QuEL-3 workaround while Qubex still carries one
        backend-level sampling period instead of per-channel `dt`.
        Control waveforms stay on the shared QuEL-3 control grid (0.4 ns).
        Readout waveforms are normalized here to the readout grid
        (`READOUT_SAMPLING_PERIOD_NS`)
        before registration so mixed control/readout schedules can still be
        executed through the current single-`dt` stack.
        """
        if not target_is_read:
            return shape, sampling_period_ns

        readout_sampling_period_ns = READOUT_SAMPLING_PERIOD_NS
        if np.isclose(sampling_period_ns, readout_sampling_period_ns):
            return shape, sampling_period_ns

        ratio = readout_sampling_period_ns / sampling_period_ns
        rounded_ratio = round(ratio)
        if rounded_ratio <= 0 or not np.isclose(ratio, rounded_ratio):
            raise ValueError(
                "Readout waveform sampling period must divide the QuEL-3 readout "
                "sampling period exactly: "
                f"sampling_period_ns={sampling_period_ns}."
            )
        remainder = shape.size % rounded_ratio
        if remainder != 0:
            pad_width = rounded_ratio - remainder
            shape = np.pad(shape, (0, pad_width), mode="edge")
        reshaped = shape.reshape(-1, rounded_ratio)
        return reshaped.mean(axis=1), readout_sampling_period_ns
