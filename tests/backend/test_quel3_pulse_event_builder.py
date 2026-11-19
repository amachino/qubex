"""Tests for QuEL-3 pulse-event compilation and waveform normalization."""

from __future__ import annotations

import numpy as np
import pytest
from qxpulse import Arbitrary, Blank, PhaseShift, PulseArray

from qubex.backend.quel3.builders import Quel3PulseEventBuilder


@pytest.mark.parametrize(
    ("target_is_read", "sampling_period_ns", "values", "expected", "expected_period"),
    [
        (False, 0.4, [1, 3, 5], [1, 3, 5], 0.4),
        (True, 0.8, [1, 3, 5], [1, 3, 5], 0.8),
        (True, 0.4, [1, 3, 5, 7], [2, 6], 0.8),
        (True, 0.4, [1, 3, 5], [2, 5], 0.8),
        (True, 0.4, [1 + 3j, 3 + 1j], [2 + 2j], 0.8),
    ],
)
def test_builder_normalizes_readout_waveforms(
    target_is_read: bool,
    sampling_period_ns: float,
    values: list[complex],
    expected: list[complex],
    expected_period: float,
) -> None:
    """Readout normalization should average samples and pad with the final value."""
    pulse = Arbitrary(values, sampling_period=sampling_period_ns)
    original_shape = pulse.shape_values.copy()
    builder = Quel3PulseEventBuilder()

    events, library = builder.build({"target": (PulseArray([pulse]), target_is_read)})

    assert len(library) == 1
    waveform = library[events["target"][0].waveform_name]
    np.testing.assert_allclose(waveform.iq_array, expected, rtol=1e-12, atol=1e-12)
    assert waveform.sampling_period_ns == pytest.approx(expected_period)
    np.testing.assert_array_equal(pulse.shape_values, original_shape)


@pytest.mark.parametrize("sampling_period_ns", [0.3, 1.6])
def test_builder_rejects_incompatible_readout_sampling_periods(
    sampling_period_ns: float,
) -> None:
    """Readout waveforms should require a period dividing the hardware grid."""
    with pytest.raises(ValueError, match="must divide"):
        Quel3PulseEventBuilder().build(
            {
                "target": (
                    PulseArray([Arbitrary([1, 3], sampling_period=sampling_period_ns)]),
                    True,
                )
            }
        )


def test_builder_preserves_sparse_readout_events_and_modifiers() -> None:
    """Normalized readout shapes should be reused with logical timing and modifiers."""
    sequence = PulseArray(
        [
            Arbitrary([0.1, 0.3], sampling_period=0.4, scale=0.5),
            Blank(0.4, sampling_period=0.4),
            PhaseShift(np.pi / 2),
            Arbitrary([0.1, 0.3], sampling_period=0.4, scale=0.7),
        ]
    )
    builder = Quel3PulseEventBuilder()

    by_target, library = builder.build({"readout": (sequence, True)})
    events = by_target["readout"]

    assert len(library) == 1
    assert events[0].waveform_name == events[1].waveform_name
    assert [event.start_offset_ns for event in events] == pytest.approx([0, 1.2])
    assert [event.gain for event in events] == pytest.approx([0.5, 0.7])
    assert [event.phase_offset_deg for event in events] == pytest.approx([0, 90])


def test_builder_shares_waveforms_only_within_one_payload() -> None:
    """Targets in a payload share waveforms and later builds have independent libraries."""
    sequence = PulseArray([Arbitrary([0.1, 0.3], sampling_period=0.4)])
    builder = Quel3PulseEventBuilder()

    events, first = builder.build({"a": (sequence, False), "b": (sequence, False)})
    readout, second = builder.build({"r": (sequence, True)})

    assert len(first) == len(second) == 1
    assert events["a"][0].waveform_name == events["b"][0].waveform_name
    np.testing.assert_array_equal(
        first[events["a"][0].waveform_name].iq_array, [0.1, 0.3]
    )
    np.testing.assert_allclose(
        second[readout["r"][0].waveform_name].iq_array, [0.2], rtol=1e-12, atol=1e-12
    )


def test_builder_distinguishes_control_and_readout_shapes() -> None:
    """Shared source shapes should remain distinct after readout normalization."""
    sequence = PulseArray([Arbitrary([0.1, 0.3], sampling_period=0.4)])
    builder = Quel3PulseEventBuilder()
    events, library = builder.build(
        {"control": (sequence, False), "readout": (sequence, True)}
    )
    event_names = [events[label][0].waveform_name for label in ("control", "readout")]

    assert len(library) == 2
    assert len(set(event_names)) == 2
    np.testing.assert_allclose(
        library[event_names[0]].iq_array,
        [0.1, 0.3],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        library[event_names[1]].iq_array, [0.2], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("target_is_read", [False, True])
def test_builder_rejects_nonfinite_iq(target_is_read: bool) -> None:
    """Nonfinite IQ should be rejected for control and readout waveforms."""
    with pytest.raises(ValueError, match="IQ values must be finite"):
        Quel3PulseEventBuilder().build(
            {
                "target": (
                    PulseArray([Arbitrary([0, np.nan], sampling_period=0.4)]),
                    target_is_read,
                )
            }
        )
