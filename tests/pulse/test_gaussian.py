"""Tests for the Gaussian pulse."""

import numpy as np
import pytest
from qxpulse import Gaussian, Pulse

import qubex as qx

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """Gaussian should inherit from Pulse."""
    assert issubclass(Gaussian, Pulse)


def test_empty_init():
    """Gaussian should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Gaussian()  # type: ignore


def test_init():
    """Gaussian should be initialized with valid parameters."""
    pulse = Gaussian(duration=5 * dt, amplitude=1, sigma=2 * dt, beta=1)
    assert pulse.name == "Gaussian"
    assert pulse.length == 5
    assert pulse.duration == 5 * dt
    assert pulse.amplitude == 1
    assert pulse.sigma == 2 * dt
    assert pulse.beta == 1
    assert pulse.values == pytest.approx(
        [
            0.27426494 + 0.27967908j,
            0.78327125 + 0.20346533j,
            1.0 + 0.0j,
            0.78327125 - 0.20346533j,
            0.27426494 - 0.27967908j,
        ]
    )


def test_zero_duration():
    """Gaussian should be initialized with zero duration."""
    pulse = Gaussian(duration=0, amplitude=1, sigma=1, beta=1)
    assert pulse.duration == 0
    assert pulse.values == pytest.approx([])


def test_invalid_parameter():
    """Gaussian should raise a ValueError if sigma is zero."""
    with pytest.raises(ValueError, match=r"Sigma must be greater than zero\."):
        Gaussian(duration=5 * dt, amplitude=1, sigma=0, beta=1)


def test_values_are_sampled_lazily(monkeypatch):
    """Gaussian should delay sampling until values are requested."""
    calls = {"count": 0}
    original_func = Gaussian.func

    def counting_func(*args, **kwargs):
        calls["count"] += 1
        return original_func(*args, **kwargs)

    monkeypatch.setattr(Gaussian, "func", staticmethod(counting_func))

    pulse = Gaussian(duration=5 * dt, amplitude=1, sigma=2 * dt, beta=1)
    assert pulse.length == 5
    assert calls["count"] == 0

    sampled = pulse.values
    assert calls["count"] == 1
    assert isinstance(sampled, np.ndarray)
    assert sampled.dtype == np.complex128

    _ = pulse.values
    assert calls["count"] == 1
