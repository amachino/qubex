"""Tests for the Blank pulse."""

import numpy as np
import pytest
import qxpulse.blank as blank_module
from qxpulse import Blank, Pulse

import qubex as qx

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """Blank should inherit from Pulse."""
    assert issubclass(Blank, Pulse)


def test_empty_init():
    """Blank should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Blank()  # type: ignore


def test_init():
    """Blank should be initialized with a duration."""
    pulse = Blank(duration=5 * dt)
    assert pulse.name == "Blank"
    assert pulse.length == 5
    assert pulse.duration == 5 * dt
    assert (pulse.values == [0, 0, 0, 0, 0]).all()


def test_values_are_sampled_lazily(monkeypatch):
    """Blank should delay sampling until values are requested."""
    calls = {"count": 0}
    original_zeros = blank_module.np.zeros

    def counting_zeros(*args, **kwargs):
        calls["count"] += 1
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(blank_module.np, "zeros", counting_zeros)

    pulse = Blank(duration=5 * dt)
    assert pulse.length == 5
    assert calls["count"] == 0

    _ = pulse.values
    assert calls["count"] == 1

    _ = pulse.values
    assert calls["count"] == 1


def test_invalid_duration():
    """Blank should raise a ValueError if duration is not a multiple of the sampling period."""
    with pytest.raises(
        ValueError, match=r"Duration must be a multiple of the sampling period"
    ):
        Blank(duration=5 * dt + np.pi)
