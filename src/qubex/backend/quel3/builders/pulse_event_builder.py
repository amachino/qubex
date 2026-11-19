"""Compile QuEL-3 waveform events from a pulse sequence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np
from qxpulse import Blank, FlatTop, Pulse, PulseArray, Rect

from qubex.backend.quel3.models import (
    Quel3Waveform,
    Quel3WaveformCompression,
    Quel3WaveformEvent,
)
from qubex.backend.quel3.quel3_backend_constants import (
    READOUT_SAMPLING_PERIOD_FS,
    READOUT_SAMPLING_PERIOD_NS,
)


@dataclass(frozen=True)
class _RepeatedBlock:
    """Start, size and repeat count on the registered waveform's sample grid."""

    start: int
    size: int
    count: int

    @property
    def end(self) -> int:
        """Return the sample following the last copy."""
        return self.start + self.size * self.count


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

    def __init__(self, *, compression: Quel3WaveformCompression | None = None) -> None:
        self._compression = compression or Quel3WaveformCompression()

    def build(
        self, targets: dict[str, PulseArray], *, instrument_periods_fs: dict[str, int]
    ) -> tuple[dict[str, tuple[Quel3WaveformEvent, ...]], dict[str, Quel3Waveform]]:
        """
        Compile the targets of one payload into events and a shared waveform library.

        Each target with a nonempty pulse needs its instrument sampling period in fs.
        Event offsets are relative to the start of each target's sequence.
        """
        library = _WaveformLibrary()
        templates: dict[
            tuple[str, int, type[Pulse]], tuple[Quel3WaveformEvent, ...]
        ] = {}
        events_by_target: dict[str, tuple[Quel3WaveformEvent, ...]] = {}

        for label, sequence in targets.items():
            instrument_period_fs = instrument_periods_fs.get(label)
            events: list[Quel3WaveformEvent] = []
            offset_ns = 0.0
            for waveform in sequence.get_flattened_waveforms(apply_frame_shifts=True):
                if not isinstance(waveform, Pulse):
                    raise TypeError(
                        f"Unsupported waveform type in PulseArray: {type(waveform).__name__}."
                    )

                if waveform.length == 0:
                    continue
                if isinstance(waveform, Blank):
                    offset_ns += waveform.duration
                    continue

                if instrument_period_fs is None or instrument_period_fs <= 0:
                    raise ValueError(
                        f"Instrument {label!r} requires a positive cached sampling period. "
                        "Deploy or refresh instruments before building the payload."
                    )

                key = (waveform.shape_hash, instrument_period_fs, type(waveform))
                if key not in templates:
                    templates[key] = self._compile(
                        waveform, instrument_period_fs, library
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
        self, pulse: Pulse, instrument_period_fs: int, library: _WaveformLibrary
    ) -> tuple[Quel3WaveformEvent, ...]:
        shape, period = self._normalize_waveform(
            instrument_period_fs=instrument_period_fs,
            shape=pulse.shape_values,
            sampling_period_ns=pulse.sampling_period,
        )
        if not np.all(np.isfinite(shape)):
            raise ValueError("Waveform IQ values must be finite.")
        block = self._find_repeated_block(
            pulse,
            length=len(shape),
            period=period,
            instrument_period_fs=instrument_period_fs,
        )
        if block is None:
            return (Quel3WaveformEvent(library.register(shape, period), 0.0),)

        parts = []
        if block.start:
            parts.append(
                Quel3WaveformEvent(library.register(shape[: block.start], period), 0.0)
            )
        name = library.register(shape[block.start : block.start + block.size], period)
        for index in range(block.count):
            offset = index * block.size
            phase = math.remainder(-360 * pulse.detuning * (offset * period), 360)
            parts.append(
                Quel3WaveformEvent(
                    name, (block.start + offset) * period, phase_offset_deg=phase
                )
            )
        if block.end < len(shape):
            parts.append(
                Quel3WaveformEvent(
                    library.register(shape[block.end :], period), block.end * period
                )
            )
        return tuple(parts)

    def _find_repeated_block(
        self,
        pulse: Pulse,
        *,
        length: int,
        period: float,
        instrument_period_fs: int,
    ) -> _RepeatedBlock | None:
        """Select a block on the registered grid and verify its source samples."""
        if not self._compression.enabled or type(pulse) not in (Rect, FlatTop):
            return None
        stride = round(period / pulse.sampling_period)
        if stride * pulse.sampling_period != period:
            # Approximate readout ratios are valid for averaging, but drift in phase.
            return None
        # Only split at boundaries that are also hardware sample boundaries.
        period_fs = round(period * 1e6)
        if not math.isclose(period * 1e6, period_fs, rel_tol=0, abs_tol=1e-8):
            return None
        quantum = instrument_period_fs // math.gcd(period_fs, instrument_period_fs)
        # Keep both ramps, their derivative neighbours and partial readout groups.
        margin = (
            math.ceil(pulse.tau / pulse.sampling_period) + 2
            if isinstance(pulse, FlatTop)
            else 0
        )
        block = self._choose_repeated_block(
            length=length,
            first=(margin + stride - 1) // stride,
            stop=(pulse.length - margin) // stride,
            alignment=quantum,
            target=self._compression.target_block_samples,
            minimum=self._compression.min_chunk_samples,
        )
        if block is None:
            return None
        # Sample transforms such as padded() preserve the original pulse type.
        envelope = pulse.envelope_values[block.start * stride : block.end * stride]
        if not np.allclose(envelope, envelope[0], rtol=1e-14, atol=0):
            return None
        return block

    @staticmethod
    def _choose_repeated_block(
        *, length: int, first: int, stop: int, alignment: int, target: int, minimum: int
    ) -> _RepeatedBlock | None:
        """Fit fixed blocks in [first, stop) on the registered sample grid."""
        size = target // alignment * alignment
        if size < minimum:
            return None

        start = max(first, minimum) if first else 0
        start = (start + alignment - 1) // alignment * alignment
        count = (stop - start) // size
        tail = length - (start + count * size)
        if 0 < tail < minimum:
            count -= 1
        if count < 2:
            return None
        return _RepeatedBlock(start, size, count)

    @staticmethod
    def _normalize_waveform(
        *,
        instrument_period_fs: int,
        shape: np.ndarray,
        sampling_period_ns: float,
    ) -> tuple[np.ndarray, float]:
        """Average onto the 800,000 fs grid while PulseSchedule uses a common period."""
        if instrument_period_fs != READOUT_SAMPLING_PERIOD_FS:
            return shape, sampling_period_ns

        if np.isclose(sampling_period_ns, READOUT_SAMPLING_PERIOD_NS):
            return shape, sampling_period_ns

        ratio = READOUT_SAMPLING_PERIOD_NS / sampling_period_ns
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
        return reshaped.mean(axis=1), READOUT_SAMPLING_PERIOD_NS
