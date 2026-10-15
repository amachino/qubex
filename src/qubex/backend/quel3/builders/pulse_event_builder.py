"""Compile QuEL-3 waveform events from a pulse sequence."""

from __future__ import annotations

import math

import numpy as np
from qxpulse import Blank, Pulse, PulseArray

from qubex.backend.quel3.models import Quel3Waveform, Quel3WaveformEvent
from qubex.backend.quel3.quel3_backend_constants import READOUT_SAMPLING_PERIOD_NS


class Quel3PulseEventBuilder:
    """Build sparse events and shared waveform entries from `PulseArray`."""

    @staticmethod
    def build(
        *,
        target_is_read: bool,
        sequence: PulseArray,
        waveform_name_by_shape_key: dict[tuple[str, int], str],
        waveform_library: dict[str, Quel3Waveform],
        waveform_index: int,
    ) -> tuple[tuple[Quel3WaveformEvent, ...], int]:
        """
        Normalize pulse shapes and preserve blank timing, gain, and phase.

        Parameters
        ----------
        target_is_read : bool
            Whether to normalize shapes to the QuEL-3 readout sampling grid.
        sequence : PulseArray
            Pulse sequence with logical offsets in ns.
        waveform_name_by_shape_key : dict
            Shared index used to reuse registered pulse shapes.
        waveform_library : dict[str, Quel3Waveform]
            Shared waveform library to update.
        waveform_index : int
            Next waveform name index.

        Returns
        -------
        tuple[tuple[Quel3WaveformEvent, ...], int]
            Sparse events and the next unused waveform index.

        Notes
        -----
        Update the supplied shape index and waveform library without modifying
        the pulse sequence. Readout padding preserves logical event offsets.
        """
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
                shape, sampling_period_ns = Quel3PulseEventBuilder._normalize_waveform(
                    target_is_read=target_is_read,
                    shape=shape,
                    sampling_period_ns=sampling_period_ns,
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
