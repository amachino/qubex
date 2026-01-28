"""Tests for the XY4 dynamical decoupling sequence."""

import pytest

from qubex.pulse.blank import Blank
from qubex.pulse.library.xy4 import XY4
from qubex.pulse.pulse import Pulse


def test_xy4_requires_multiple_of_sampling_period():
    """Given tau not divisible by the sampling period, then XY4 rejects the value."""
    with pytest.raises(
        ValueError,
        match=r"Tau must be a multiple of the sampling period",
    ):
        XY4(
            tau=Pulse.SAMPLING_PERIOD + 1,
            pi_x=Blank(duration=Pulse.SAMPLING_PERIOD),
            pi_y=Blank(duration=Pulse.SAMPLING_PERIOD),
        )


def test_xy4_requires_positive_cycle_count():
    """Given a non-positive cycle count, then XY4 raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"The number of XY4 cycles must be greater than 0\.",
    ):
        XY4(
            tau=Pulse.SAMPLING_PERIOD,
            pi_x=Blank(duration=Pulse.SAMPLING_PERIOD),
            pi_y=Blank(duration=Pulse.SAMPLING_PERIOD),
            n=0,
        )


def test_xy4_constructs_expected_sequence_and_duration():
    """Given valid inputs, then XY4 builds the expected waveform sequence."""
    tau = Pulse.SAMPLING_PERIOD
    pi_x = Blank(duration=2 * tau)
    pi_y = Blank(duration=3 * tau)
    cycles = 2
    xy4 = XY4(tau=tau, pi_x=pi_x, pi_y=pi_y, n=cycles)
    assert len(xy4.waveforms) == 12 * cycles
    expected_duration = cycles * (8 * tau + 2 * pi_x.duration + 2 * pi_y.duration)
    assert xy4.duration == expected_duration
    assert xy4.waveforms[0].duration == tau
    assert xy4.waveforms[1].duration == pi_x.duration
    assert xy4.waveforms[4].duration == pi_y.duration


def test_xy4_zero_tau_has_no_waveforms():
    """Given tau zero, then XY4 produces an empty waveform sequence."""
    xy4 = XY4(
        tau=0,
        pi_x=Blank(duration=Pulse.SAMPLING_PERIOD),
        pi_y=Blank(duration=Pulse.SAMPLING_PERIOD),
    )
    assert xy4.waveforms == []
