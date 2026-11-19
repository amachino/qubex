"""Tests for block sharing while compiling QuEL-3 pulse schedules."""

from functools import partial

import numpy as np
import pytest
from qxpulse import Arbitrary, FlatTop, PulseArray, Rect, VirtualZ

from qubex.backend.quel3.builders import Quel3PulseEventBuilder
from qubex.backend.quel3.execution.timing import sample_count
from qubex.backend.quel3.models import Quel3WaveformCompression


def _compile(pulse, *, instrument_period_fs=400_000, options=None):
    builder = Quel3PulseEventBuilder(compression=options)
    events, library = builder.build(
        {"target": PulseArray([pulse])},
        instrument_periods_fs={"target": instrument_period_fs},
    )
    return library, events["target"]


@pytest.mark.parametrize(
    ("length", "sizes"),
    [
        (999, [999]),
        (7499, [7499]),
        (7500, [7500]),
        (9999, [9999]),
        (10000, [5000, 5000]),
        (10001, [10001]),
        (11000, [5000, 5000, 1000]),
        (15001, [5000, 5000, 5001]),
    ],
)
def test_rect_block_sizes(length, sizes):
    """Fixed-size blocks preserve all samples without introducing a short tail."""
    pulse = Rect(duration=length * 0.4, sampling_period=0.4, amplitude=0.25)
    library, events = _compile(pulse)
    assert [len(library[e.waveform_name].iq_array) for e in events] == sizes
    assert [e.start_offset_ns for e in events] == pytest.approx(
        np.cumsum([0, *sizes[:-1]]) * 0.4
    )
    assert len(library) == len(set(sizes))


def test_long_detuned_rect_preserves_complex_iq():
    """A millisecond detuned rectangle fits in one shared 5000-sample block."""
    pulse = Rect(duration=1e6, sampling_period=0.4, amplitude=0.5, detuning=0.2)
    library, events = _compile(pulse)
    assert len(library) == 1
    assert len(events) == 500
    block = library[events[0].waveform_name].iq_array
    assert len(block) == 5000
    phases = np.exp(1j * np.deg2rad([e.phase_offset_deg for e in events]))
    actual = phases[:, None] * block
    assert np.max(np.abs(actual.ravel() - pulse.shape_values)) < 1e-10


def test_identical_iq_shares_entries_across_pulse_types():
    """Whole waveforms and blocks share IQ while arbitrary pulses retain one event."""
    builder = Quel3PulseEventBuilder()
    pulse = Rect(duration=8000, sampling_period=0.4, amplitude=0.3)
    full = Arbitrary(pulse.shape_values, sampling_period=0.4)
    block = Arbitrary(pulse.shape_values[:5000], sampling_period=0.4)
    sequence = PulseArray([full, block, pulse, VirtualZ(0.73), pulse.scaled(-0.6)])
    by_target, library = builder.build(
        {"control": sequence}, instrument_periods_fs={"control": 400_000}
    )
    events = by_target["control"]
    assert len(events) == 10
    assert len(library) == 2
    assert events[0].waveform_name != events[1].waveform_name
    assert {part.waveform_name for part in events[1:]} == {events[1].waveform_name}
    assert all(part.gain == -0.6 for part in events[-4:])
    assert all(
        part.phase_offset_deg == pytest.approx(np.degrees(-0.73))
        for part in events[-4:]
    )


def test_readout_normalization_preserves_the_source_detuning_clock():
    """A nearly matching input grid retains the normalized IQ of the whole pulse."""
    pulse = Rect(
        duration=20000 * 0.4000002,
        sampling_period=0.4000002,
        amplitude=0.3,
        detuning=0.2,
    )
    library, events = _compile(pulse, instrument_period_fs=800_000)
    expected = pulse.shape_values.reshape(-1, 2).mean(axis=1)
    actual = np.concatenate(
        [
            library[part.waveform_name].iq_array
            * np.exp(1j * np.deg2rad(part.phase_offset_deg))
            for part in events
        ]
    )
    assert np.max(np.abs(actual - expected)) < 1e-10


@pytest.mark.parametrize("side", ["left", "right"])
def test_padded_rect_preserves_modified_envelope(side):
    """A padded rectangle retains its actual zeros in the registered waveform."""
    pulse = Rect(duration=4000, sampling_period=0.4, amplitude=0.25).padded(
        8000, pad_side=side
    )
    library, events = _compile(pulse)
    assert len(events) == 1
    np.testing.assert_array_equal(
        library[events[0].waveform_name].iq_array, pulse.shape_values
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"target_block_samples": 0},
        {"min_chunk_samples": -1},
        {"min_chunk_samples": 5001},
        {"enabled": 1},
    ],
)
def test_invalid_compression_settings(settings):
    """Block lengths must be positive, ordered integers and enabled must be boolean."""
    with pytest.raises(ValueError, match=r"enabled|block_samples|chunk_samples"):
        Quel3WaveformCompression(**settings)


def test_generation_settings_choose_block_sizes():
    """Building the same pulse with different options fixes different event lists."""
    pulse = Rect(duration=12001 * 0.4, sampling_period=0.4, amplitude=0.2)
    for options, sizes in [
        (Quel3WaveformCompression(enabled=False), [12001]),
        (
            Quel3WaveformCompression(target_block_samples=4000, min_chunk_samples=1500),
            [4000, 4000, 4001],
        ),
    ]:
        library, events = _compile(pulse, options=options)
        assert [len(library[e.waveform_name].iq_array) for e in events] == sizes


@pytest.mark.parametrize("instrument_period_fs", [400_000, 800_000])
@pytest.mark.parametrize("frequency", [0.0, 0.0137123, -0.027431])
@pytest.mark.parametrize(
    "shape", [Rect, pytest.param(partial(FlatTop, tau=45.2, beta=0.25), id="FlatTop")]
)
def test_block_playback_preserves_normalized_complex_iq(
    instrument_period_fs, frequency, shape
):
    """Playback preserves ramps, odd readout padding, detuning, gain and frame shifts."""
    pulse = shape(
        duration=40014.8,
        amplitude=0.3,
        sampling_period=0.4,
        detuning=frequency,
    )
    sequence = PulseArray([VirtualZ(0.73), pulse, pulse.scaled(-0.6)])

    def render(enabled):
        builder = Quel3PulseEventBuilder(
            compression=Quel3WaveformCompression(enabled=enabled)
        )
        by_target, library = builder.build(
            {"target": sequence}, instrument_periods_fs={"target": instrument_period_fs}
        )
        events = by_target["target"]
        period = instrument_period_fs / 1e6
        signal = np.zeros(round(sequence.duration / period) + 2, dtype=complex)
        for event in events:
            start = sample_count(event.start_offset_ns, period)
            samples = library[event.waveform_name].iq_array
            assert len(samples) >= 1000
            signal[start : start + len(samples)] += (
                event.gain * np.exp(1j * np.deg2rad(event.phase_offset_deg)) * samples
            )
        return signal, len(events)

    actual, count = render(True)
    expected, _ = render(False)
    assert count > 2
    assert np.max(np.abs(actual - expected)) < 1e-10


@pytest.mark.parametrize("instrument_period_fs", [400_000, 600_000])
def test_flat_top_chunks_align_to_the_control_grid(instrument_period_fs):
    """Fractional control samples produce full fragments at hardware sample boundaries."""
    pulse = FlatTop(duration=8000.2, amplitude=0.2, tau=40.2, sampling_period=0.2)
    library, events = _compile(pulse, instrument_period_fs=instrument_period_fs)
    assert len(events) > 1
    period_ns = instrument_period_fs / 1e6
    for event in events:
        assert event.start_offset_ns / period_ns == pytest.approx(
            round(event.start_offset_ns / period_ns), abs=1e-8
        )
    actual = np.concatenate([library[e.waveform_name].iq_array for e in events])
    np.testing.assert_array_equal(actual, pulse.shape_values)


@pytest.mark.parametrize(
    ("length", "first", "stop", "alignment", "target", "minimum", "expected"),
    [
        (31, 1, 30, 3, 10, 4, (6, 9, 2)),
        (30, 0, 30, 6, 5, 3, None),
        (30, 0, 30, 6, 10, 7, None),
        (11, 0, 10, 1, 5, 2, None),
    ],
)
def test_integer_block_policy(
    length, first, stop, alignment, target, minimum, expected
):
    """Fixed blocks respect alignment, plateau margins and minimum fragment length."""
    block = Quel3PulseEventBuilder._choose_repeated_block(  # noqa: SLF001
        length=length,
        first=first,
        stop=stop,
        alignment=alignment,
        target=target,
        minimum=minimum,
    )
    assert ((block.start, block.size, block.count) if block else None) == expected


def test_repeated_flat_top_preserves_internal_ramps():
    """A repeated flat-top retains the ramps between copies."""
    pulse = FlatTop(duration=8000, sampling_period=0.4, amplitude=0.2, tau=40).repeated(
        3
    )
    library, events = _compile(pulse)
    assert len(events) == 1
    np.testing.assert_array_equal(
        library[events[0].waveform_name].iq_array, pulse.shape_values
    )
