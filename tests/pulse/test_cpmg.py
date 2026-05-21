"""Tests for the CPMG dynamical decoupling sequence."""

import pytest
from qxpulse.blank import Blank
from qxpulse.library.cpmg import CPMG
from qxpulse.pulse import Pulse
from qxpulse.waveform import Waveform


def test_cpmg_requires_multiple_of_sampling_period():
    """Given tau not divisible by the sampling period, then CPMG rejects the value."""
    with pytest.raises(
        ValueError,
        match=r"Tau must be a multiple of the sampling period",
    ):
        CPMG(
            tau=Pulse.SAMPLING_PERIOD + 1,
            pi=Blank(duration=Pulse.SAMPLING_PERIOD),
        )


def test_cpmg_accepts_float_multiple_of_sampling_period(monkeypatch):
    """Given a float multiple of dt, then CPMG accepts tau despite float remainder noise."""
    monkeypatch.setattr(Waveform, "SAMPLING_PERIOD", 0.4)

    cpmg = CPMG(
        tau=1.2,
        pi=Blank(duration=0.4),
    )

    # Default n=2 creates two (tau, pi, tau) blocks: 2 * (3 + 1 + 3) samples.
    assert cpmg.length == 14
