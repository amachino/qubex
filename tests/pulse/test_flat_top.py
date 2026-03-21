"""Tests for the FlatTop pulse."""

import pytest
from qxpulse import FlatTop, Pulse

import qubex as qx

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """FlatTop should inherit from Pulse."""
    assert issubclass(FlatTop, Pulse)


def test_empty_init():
    """FlatTop should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        FlatTop()  # type: ignore


def test_init():
    """FlatTop should be initialized with valid parameters."""
    pulse = FlatTop(duration=5 * dt, amplitude=1, tau=2 * dt)
    assert pulse.name == "FlatTop"
    assert pulse.length == 5
    assert pulse.duration == 5 * dt
    assert pulse.amplitude == 1
    assert pulse.tau == 2 * dt
    assert pulse.values == pytest.approx(
        [
            0.14644661 + 0.0j,
            0.85355339 + 0.0j,
            1.0 + 0.0j,
            0.85355339 + 0.0j,
            0.14644661 + 0.0j,
        ]
    )


def test_zero_duration():
    """FlatTop should be initialized with zero duration."""
    pulse = FlatTop(duration=0, amplitude=1, tau=0)
    assert pulse.duration == 0
    assert pulse.values == pytest.approx([])


def test_invalid_duration():
    """FlatTop should raise a ValueError if duration is not greater than `2 * tau`."""
    with pytest.raises(ValueError, match=r"duration must be greater than `2 \* tau`\."):
        FlatTop(duration=5 * dt, amplitude=1, tau=3 * dt)


def test_shape_kwargs_do_not_leak_pulse_init_parameters(monkeypatch):
    """FlatTop should not forward pulse-init kwargs into the shape sampler."""
    captured: dict[str, object] = {}
    original_func = FlatTop.func

    def recording_func(*args, **kwargs):
        captured.update(kwargs)
        return original_func(*args, **kwargs)

    monkeypatch.setattr(FlatTop, "func", staticmethod(recording_func))

    pulse = FlatTop(
        duration=5 * dt,
        amplitude=1,
        tau=2 * dt,
        sampling_period=dt,
        lazy=False,
        window="hann",
    )

    assert pulse.length == 5
    assert captured["window"] == "hann"
    assert "sampling_period" not in captured
    assert "lazy" not in captured
