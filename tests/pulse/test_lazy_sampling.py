"""Tests for lazy sampling behavior across pulse shapes."""

import numpy as np
import pytest
from qxpulse import Bump, Drag, FlatTop, Gaussian, RaisedCosine, Rect, Sintegral
from qxpulse.library.flat_top import MultiDerivativeFlatTop
from qxpulse.library.multi_derivative import MultiDerivative
from qxpulse.library.sintegral import MultiDerivativeSintegral
from qxpulse.library.squad import Squad
from qxpulse.library.vert_ramp import VertRamp

import qubex as qx

dt = qx.pulse.get_sampling_period()

PULSE_CASES = [
    (Bump, {"amplitude": 1.0}),
    (Drag, {"amplitude": 1.0, "beta": 0.1}),
    (FlatTop, {"amplitude": 1.0, "tau": 2 * dt}),
    (Gaussian, {"amplitude": 1.0, "sigma": 2 * dt}),
    (MultiDerivative, {"amplitude": 1.0, "betas": {1: 0.1}}),
    (MultiDerivativeFlatTop, {"amplitude": 1.0, "tau": 2 * dt, "betas": {1: 0.1}}),
    (MultiDerivativeSintegral, {"amplitude": 1.0, "betas": {1: 0.1}}),
    (RaisedCosine, {"amplitude": 1.0}),
    (Rect, {"amplitude": 1.0}),
    (Sintegral, {"amplitude": 1.0}),
    (Squad, {"amplitude": 1.0, "delta": 1.0, "tau": 2 * dt}),
    pytest.param(
        VertRamp,
        {"amplitude": 1.0},
        marks=pytest.mark.filterwarnings(
            "ignore:The 'VertRamp' class is deprecated and will be removed in a future release."
        ),
    ),
]


@pytest.mark.parametrize(("pulse_cls", "kwargs"), PULSE_CASES)
def test_pulse_shape_samples_lazily(monkeypatch, pulse_cls, kwargs):
    """Pulse shape should sample only when values are first requested."""
    calls = {"count": 0}

    def counting_func(*args, **func_kwargs):
        calls["count"] += 1
        if args:
            t = args[0]
        else:
            t = func_kwargs["t"]
        t = np.asarray(t)
        return np.zeros_like(t, dtype=np.complex128)

    monkeypatch.setattr(pulse_cls, "func", staticmethod(counting_func))

    pulse = pulse_cls(duration=6 * dt, **kwargs)
    assert calls["count"] == 0

    sampled = pulse.values
    assert calls["count"] == 1
    assert isinstance(sampled, np.ndarray)
    assert sampled.dtype == np.complex128

    _ = pulse.values
    assert calls["count"] == 1


def test_pulse_shape_samples_on_init_when_lazy_is_false(monkeypatch):
    """Pulse shape should sample during init when lazy is False."""
    calls = {"count": 0}
    original_func = Gaussian.func

    def counting_func(*args, **kwargs):
        calls["count"] += 1
        return original_func(*args, **kwargs)

    monkeypatch.setattr(Gaussian, "func", staticmethod(counting_func))

    pulse = Gaussian(duration=6 * dt, amplitude=1.0, sigma=2 * dt, lazy=False)
    assert pulse.length == 6
    assert calls["count"] == 1

    _ = pulse.values
    assert calls["count"] == 1
