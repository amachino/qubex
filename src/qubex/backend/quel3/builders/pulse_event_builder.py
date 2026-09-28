"""Compile QuEL-3 waveform events from a pulse sequence."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from qxpulse import Blank, Pulse, PulseArray

from qubex.backend.quel3.models import Quel3Waveform, Quel3WaveformEvent

WaveformNormalizer = Callable[[np.ndarray, float], tuple[np.ndarray, float]]


class Quel3PulseEventBuilder:
    """Build sparse events and shared waveform entries from `PulseArray`."""

    @staticmethod
    def build(
        *,
        sequence: PulseArray,
        waveform_name_by_shape_key: dict[tuple[str, int], str],
        waveform_library: dict[str, Quel3Waveform],
        waveform_index: int,
        normalize_waveform: WaveformNormalizer | None = None,
    ) -> tuple[tuple[Quel3WaveformEvent, ...], int]:
        """Register pulse shapes and preserve blank timing, gain, and phase."""
        events: list[Quel3WaveformEvent] = []
        current_offset_ns = 0.0
        for waveform in sequence.get_flattened_waveforms(apply_frame_shifts=True):
            duration_ns = waveform.duration
            if isinstance(waveform, Blank):
                current_offset_ns += duration_ns
                continue

            if isinstance(waveform, Pulse):
                sampling_period_ns = waveform.sampling_period
                shape = np.asarray(waveform.shape_values, dtype=np.complex128)
                if shape.size == 0:
                    current_offset_ns += duration_ns
                    continue
                if normalize_waveform is not None:
                    shape, sampling_period_ns = normalize_waveform(
                        shape, sampling_period_ns
                    )
                if not np.all(np.isfinite(shape.real) & np.isfinite(shape.imag)):
                    raise ValueError("Waveform IQ values must be finite.")
                shape_key = (
                    waveform.shape_hash,
                    round(float(sampling_period_ns) * 1e6),
                )
                waveform_name = waveform_name_by_shape_key.get(shape_key)
                if waveform_name is None:
                    waveform_name = f"waveform_{waveform_index:04d}"
                    waveform_index += 1
                    waveform_library[waveform_name] = Quel3Waveform(
                        iq_array=shape,
                        sampling_period_ns=sampling_period_ns,
                    )
                    waveform_name_by_shape_key[shape_key] = waveform_name
                events.append(
                    Quel3WaveformEvent(
                        waveform_name=waveform_name,
                        start_offset_ns=current_offset_ns,
                        gain=waveform.scale,
                        phase_offset_deg=math.degrees(waveform.phase),
                    )
                )
                current_offset_ns += duration_ns
                continue

            raise TypeError(
                f"Unsupported waveform type in PulseArray: {type(waveform).__name__}."
            )
        return tuple(events), waveform_index
