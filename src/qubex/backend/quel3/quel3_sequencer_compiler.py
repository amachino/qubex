"""Compile QuEL-3 execution payloads into sequencer directives."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from .quel3_execution_payload import Quel3ExecutionPayload


class _SequencerProtocol(Protocol):
    """Minimal protocol required by the sequencer compiler."""

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


T = TypeVar("T", bound=_SequencerProtocol)


@dataclass(frozen=True)
class _WaveformSegment:
    """One contiguous non-blank waveform segment."""

    start_index: int
    values: npt.NDArray[np.complex128]


class Quel3SequencerCompiler:
    """Compile `Quel3ExecutionPayload` to sequencer events and waveforms."""

    def __init__(
        self,
        *,
        amplitude_epsilon: float = 1e-12,
        shape_quantization: float = 1e-9,
        waveform_name_prefix: str = "wf_shared",
    ) -> None:
        if amplitude_epsilon <= 0:
            raise ValueError("amplitude_epsilon must be positive.")
        if shape_quantization <= 0:
            raise ValueError("shape_quantization must be positive.")
        self._amplitude_epsilon = float(amplitude_epsilon)
        self._shape_quantization = float(shape_quantization)
        self._waveform_name_prefix = waveform_name_prefix

    def compile(
        self,
        *,
        payload: Quel3ExecutionPayload,
        sequencer_factory: Callable[..., T],
        default_sampling_period_ns: float,
    ) -> T:
        """
        Build one sequencer instance from a QuEL-3 execution payload.

        Parameters
        ----------
        payload : Quel3ExecutionPayload
            QuEL-3 execution payload from measurement adapter.
        sequencer_factory : Callable[..., T]
            Sequencer class or factory compatible with quelware `Sequencer`.
        default_sampling_period_ns : float
            Sequencer default sampling period in ns.

        Returns
        -------
        T
            Compiled sequencer instance.
        """
        sequencer = sequencer_factory(
            default_sampling_period_ns=float(default_sampling_period_ns)
        )
        waveform_name_by_shape_key: dict[str, str] = {}
        waveform_index = 0

        for target, timeline in payload.timelines.items():
            alias = payload.instrument_aliases[target]
            default_event_sampling_period_ns = float(timeline.sampling_period_ns)
            for event in timeline.events:
                sampling_period_ns = (
                    default_event_sampling_period_ns
                    if event.sampling_period_ns is None
                    else float(event.sampling_period_ns)
                )
                start_offset_ns = float(event.start_offset_ns)
                waveform = np.asarray(event.waveform, dtype=np.complex128)
                for segment in self._iter_non_blank_segments(waveform):
                    shape, gain, phase_offset_deg = self._factor_shape(segment.values)
                    shape_key = self._shape_key(
                        shape=shape, sampling_period_ns=sampling_period_ns
                    )
                    waveform_name = waveform_name_by_shape_key.get(shape_key)
                    if waveform_name is None:
                        waveform_name = (
                            f"{self._waveform_name_prefix}_{waveform_index:04d}"
                        )
                        waveform_index += 1
                        sequencer.register_waveform(
                            waveform_name,
                            shape,
                            sampling_period_ns=sampling_period_ns,
                        )
                        waveform_name_by_shape_key[shape_key] = waveform_name

                    sequencer.add_event(
                        alias,
                        waveform_name,
                        start_offset_ns=(
                            start_offset_ns
                            + float(segment.start_index) * sampling_period_ns
                        ),
                        gain=gain,
                        phase_offset_deg=phase_offset_deg,
                    )

            for capture_window in timeline.capture_windows:
                sequencer.add_capture_window(
                    alias,
                    self.capture_window_key(target, capture_window.name),
                    start_offset_ns=float(capture_window.start_offset_ns),
                    length_ns=float(capture_window.length_ns),
                )

        return sequencer

    @staticmethod
    def capture_window_key(target: str, window_name: str) -> str:
        """Return deterministic capture-window key for one target/window pair."""
        return f"{target}:{window_name}"

    def _iter_non_blank_segments(
        self,
        waveform: npt.NDArray[np.complex128],
    ) -> Iterator[_WaveformSegment]:
        """Yield contiguous non-blank waveform segments."""
        non_blank_indices = np.flatnonzero(np.abs(waveform) > self._amplitude_epsilon)
        if non_blank_indices.size == 0:
            return

        start = int(non_blank_indices[0])
        previous = start
        for index in non_blank_indices[1:]:
            index_int = int(index)
            if index_int == previous + 1:
                previous = index_int
                continue
            yield _WaveformSegment(
                start_index=start,
                values=waveform[start : previous + 1],
            )
            start = index_int
            previous = index_int

        yield _WaveformSegment(
            start_index=start,
            values=waveform[start : previous + 1],
        )

    def _factor_shape(
        self,
        values: npt.NDArray[np.complex128],
    ) -> tuple[npt.NDArray[np.complex128], float, float]:
        """Factor one segment into normalized shape and complex scalar."""
        amplitudes = np.abs(values)
        peak_index = int(np.argmax(amplitudes))
        peak_value = values[peak_index]
        gain = float(amplitudes[peak_index])
        if gain <= self._amplitude_epsilon:
            raise ValueError("Non-blank segment peak amplitude must be positive.")
        shape = np.asarray(values / peak_value, dtype=np.complex128)
        phase_offset_deg = float(np.rad2deg(np.angle(peak_value)))
        return shape, gain, phase_offset_deg

    def _shape_key(
        self,
        *,
        shape: npt.NDArray[np.complex128],
        sampling_period_ns: float,
    ) -> str:
        """Create deterministic deduplication key for one normalized shape."""
        quantized_real = np.rint(shape.real / self._shape_quantization).astype(np.int64)
        quantized_imag = np.rint(shape.imag / self._shape_quantization).astype(np.int64)
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(np.asarray(shape.size, dtype=np.int64).tobytes())
        hasher.update(np.asarray(sampling_period_ns, dtype=np.float64).tobytes())
        hasher.update(quantized_real.tobytes())
        hasher.update(quantized_imag.tobytes())
        return hasher.hexdigest()
