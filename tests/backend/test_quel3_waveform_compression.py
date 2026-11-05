"""Numerical regression checks for QuEL-3 waveform compression."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import FlatTop, PulseArray, Rect, VirtualZ

from qubex.backend.quel3 import Quel3BackendController
from qubex.backend.quel3.builders import Quel3PulseEventBuilder
from qubex.backend.quel3.execution.compression import compress_payload
from qubex.backend.quel3.execution.resources import Quel3PayloadAnalyzer
from qubex.backend.quel3.models import (
    Quel3CaptureMode,
    Quel3ExecutionOptions,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
)
from qubex.backend.quel3.models.payload import (
    Quel3CompressionHint,
    Quel3Waveform,
    Quel3WaveformEvent,
)


@pytest.mark.parametrize("enabled", [False, True])
def test_execution_option_controls_compression(enabled):
    """Configured compression affects analysis before resource counting."""
    controller = Quel3BackendController.from_config_mapping(
        {"execution": {"compress_waveforms": enabled, "max_waveform_samples": 50000}}
    )
    waveform = Quel3Waveform(
        np.full(100000, 0.25 + 0j), 0.4, Quel3CompressionHint(0, 100000, 0.0)
    )
    payload = Quel3ExecutionPayload(
        {"wave": waveform},
        {"Q": Quel3FixedTimeline((Quel3WaveformEvent("wave", 0),), (), 40000, 5e9)},
        1,
        0,
        Quel3CaptureMode.RAW_WAVEFORMS,
    )
    instrument = SimpleNamespace(
        config=SimpleNamespace(sampling_period_fs=400000, timeline_step_samples=64)
    )
    analyzed = Quel3PayloadAnalyzer(default_sampling_period_ns=0.4).analyze(
        payload, {"Q": cast(Any, instrument)}, options=controller.execution_options
    )
    count = analyzed.planning.requirements.waveform_samples["Q"]
    assert (count < 50000) if enabled else count == 100000
    assert Quel3ExecutionOptions().compress_waveforms


def test_modified_rect_falls_back_without_changing_samples():
    """A stale hint must not compress a padded or otherwise modified envelope."""
    samples = np.concatenate([np.full(60000, 0.2 + 0j), np.zeros(10000)]).astype(
        np.complex128
    )
    payload = Quel3ExecutionPayload(
        {"wave": Quel3Waveform(samples, 0.4, Quel3CompressionHint(0, 70000, 0))},
        {"Q": Quel3FixedTimeline((Quel3WaveformEvent("wave", 0),), (), 28000)},
        1,
        0,
        Quel3CaptureMode.RAW_WAVEFORMS,
    )
    assert compress_payload(payload, {"Q": 400000}, 50000) is payload


@pytest.mark.parametrize("read", [False, True])
@pytest.mark.parametrize("frequency", [0.0, 0.0137123, -0.027431])
@pytest.mark.parametrize("shape", [Rect, FlatTop])
def test_compression_preserves_complex_samples(read, frequency, shape):
    """Compression preserves IQ, frame shifts, timing, and repeated references."""
    kwargs: dict[str, Any] = {"tau": 45.2} if shape is FlatTop else {}
    pulse = shape(
        duration=40014.8,
        amplitude=0.3,
        sampling_period=0.4,
        detuning=frequency,
        **kwargs,
    )
    sequence = PulseArray([VirtualZ(0.73), pulse, pulse.scaled(-0.6)])
    library = {}
    events, _ = Quel3PulseEventBuilder.build(
        target_is_read=read,
        sequence=sequence,
        waveform_name_by_shape_key={},
        waveform_library=library,
        waveform_index=0,
    )
    payload = Quel3ExecutionPayload(
        library,
        {"Q": Quel3FixedTimeline(events, (), sequence.duration)},
        1,
        0,
        Quel3CaptureMode.RAW_WAVEFORMS,
    )
    dt = 0.8 if read else 0.4

    def render(value):
        result = np.zeros(round(sequence.duration / dt) + 4, dtype=complex)
        for event in value.fixed_timelines["Q"].events:
            samples = value.waveform_library[event.waveform_name].iq_array
            start = round(event.start_offset_ns / dt)
            result[start : start + len(samples)] += (
                samples * event.gain * np.exp(1j * np.deg2rad(event.phase_offset_deg))
            )
        return result

    payload = replace(
        payload,
        fixed_timelines={
            "Q": replace(
                payload.fixed_timelines["Q"],
                events=tuple(
                    replace(e, start_offset_ns=np.ceil(e.start_offset_ns / dt) * dt)
                    for e in events
                ),
            )
        },
    )
    compressed = compress_payload(payload, {"Q": round(dt * 1e6)}, 50000)
    actual = render(compressed)
    expected = render(payload)
    assert actual.shape == expected.shape
    # Compare finite IQ directly, avoiding NumPy 1.26's NaN/Inf masking path.
    assert np.max(np.abs(actual - expected)) == pytest.approx(0.0, rel=0, abs=1e-10)
    assert sum(len(w.iq_array) for w in compressed.waveform_library.values()) < 50000
    assert len(compressed.fixed_timelines["Q"].events) > len(events)
    assert compressed.fixed_timelines["Q"].length_ns == sequence.duration
    without_hints = replace(
        payload,
        waveform_library={
            name: replace(waveform, compression_hint=None)
            for name, waveform in library.items()
        },
    )
    assert (
        compress_payload(without_hints, {"Q": round(dt * 1e6)}, 50000) is without_hints
    )
