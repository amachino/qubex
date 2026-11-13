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
    assert compress_payload(payload, {"Q": 400000}) is payload


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
    compressed = compress_payload(payload, {"Q": round(dt * 1e6)})
    actual = render(compressed)
    expected = render(payload)
    assert actual.shape == expected.shape
    # Compare finite IQ directly, avoiding NumPy 1.26's NaN/Inf masking path.
    assert np.max(np.abs(actual - expected)) == pytest.approx(0.0, rel=0, abs=1e-10)
    assert sum(len(w.iq_array) for w in compressed.waveform_library.values()) < 50000
    assert len(compressed.fixed_timelines["Q"].events) > len(events)
    assert compressed.fixed_timelines["Q"].length_ns == sequence.duration
    assert all(len(w.iq_array) >= 1000 for w in compressed.waveform_library.values())
    without_hints = replace(
        payload,
        waveform_library={
            name: replace(waveform, compression_hint=None)
            for name, waveform in library.items()
        },
    )
    assert compress_payload(without_hints, {"Q": round(dt * 1e6)}) is without_hints


@pytest.mark.parametrize(
    ("length", "sizes"),
    [
        (999, [999]),
        (5001, [5001]),
        (7499, [7499]),
        (7500, [3750, 3750]),
        (9999, [4499, 4499, 1001]),
        (10000, [5000, 5000]),
        (10001, [4500, 4500, 1001]),
        (14999, [4666, 4666, 4666, 1001]),
        (19999, [4749, 4749, 4749, 4749, 1003]),
    ],
)
def test_compression_avoids_short_fragments(length, sizes):
    """Near-target blocks preserve every sample without creating short tails."""
    waveform = Quel3Waveform(
        np.full(length, 0.2 + 0j), 0.4, Quel3CompressionHint(0, length, 0.0)
    )
    payload = Quel3ExecutionPayload(
        {"wave": waveform},
        {"Q": Quel3FixedTimeline((Quel3WaveformEvent("wave", 0),), (), length * 0.4)},
        1,
        0,
        Quel3CaptureMode.RAW_WAVEFORMS,
    )
    result = compress_payload(payload, {"Q": 400000})
    events = result.fixed_timelines["Q"].events
    assert [
        len(result.waveform_library[e.waveform_name].iq_array) for e in events
    ] == sizes
    if len(sizes) == 1:
        assert result is payload
    else:
        assert all(size >= 1000 for size in sizes)
    assert sum(sizes) == length
    assert len(payload.waveform_library["wave"].iq_array) == length


@pytest.mark.parametrize(
    "values",
    [
        {"compression_target_block_samples": 0},
        {"compression_min_chunk_samples": -1},
        {"compression_min_chunk_samples": 5001},
        {"compression_target_block_samples": True},
    ],
)
def test_compression_options_reject_invalid_lengths(values):
    """Compression lengths must be positive integers with minimum below target."""
    with pytest.raises(ValueError, match=r"compression_|Compression"):
        Quel3ExecutionOptions(**values)


def test_compression_lengths_load_from_config():
    """System execution configuration controls block and minimum chunk lengths."""
    controller = Quel3BackendController.from_config_mapping(
        {
            "execution": {
                "compression_target_block_samples": 4000,
                "compression_min_chunk_samples": 1500,
            }
        }
    )
    assert controller.execution_options.compression_target_block_samples == 4000
    assert controller.execution_options.compression_min_chunk_samples == 1500


@pytest.mark.parametrize("sampling_fs", [400000, 1200000])
@pytest.mark.parametrize("length", [5001, 9999, 10001, 14999, 19999])
def test_flat_top_fragments_preserve_iq_and_sample_grid(length, sampling_fs):
    """Short ramps join adjacent samples and all generated events stay on grid."""
    samples = np.full(length, 0.2 + 0j)
    samples[:113] = np.linspace(0, 0.2, 113)
    samples[-113:] = np.linspace(0.2, 0, 113)
    payload = Quel3ExecutionPayload(
        {
            "wave": Quel3Waveform(
                samples, 0.4, Quel3CompressionHint(115, length - 115, 0)
            )
        },
        {"Q": Quel3FixedTimeline((Quel3WaveformEvent("wave", 0),), (), length * 0.4)},
        1,
        0,
        Quel3CaptureMode.RAW_WAVEFORMS,
    )
    result = compress_payload(payload, {"Q": sampling_fs})
    reconstructed = np.zeros_like(samples)
    for event in result.fixed_timelines["Q"].events:
        chunk = result.waveform_library[event.waveform_name].iq_array
        assert len(chunk) >= 1000
        start = round(event.start_offset_ns / 0.4)
        assert start % (sampling_fs // 400000) == 0
        reconstructed[start : start + len(chunk)] = chunk
    np.testing.assert_array_equal(reconstructed, samples)


def test_custom_lengths_control_actual_fragments():
    """Custom target and minimum lengths change compression through the analyzer."""
    waveform = Quel3Waveform(
        np.full(12001, 0.2 + 0j), 0.4, Quel3CompressionHint(0, 12001, 0.0)
    )
    payload = Quel3ExecutionPayload(
        {"wave": waveform},
        {"Q": Quel3FixedTimeline((Quel3WaveformEvent("wave", 0),), (), 4800.4, 5e9)},
        1,
        0,
        Quel3CaptureMode.RAW_WAVEFORMS,
    )
    instrument = SimpleNamespace(
        config=SimpleNamespace(sampling_period_fs=400000, timeline_step_samples=64)
    )
    result = Quel3PayloadAnalyzer(default_sampling_period_ns=0.4).analyze(
        payload,
        {"Q": cast(Any, instrument)},
        options=Quel3ExecutionOptions(
            compression_target_block_samples=4000, compression_min_chunk_samples=1500
        ),
    )
    sizes = [
        len(result.payload.waveform_library[e.waveform_name].iq_array)
        for e in result.payload.fixed_timelines["Q"].events
    ]
    assert sizes == [3500, 3500, 3500, 1501]
