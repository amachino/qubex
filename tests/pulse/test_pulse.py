"""Tests for the Pulse base class."""

import copy as pycopy

import numpy as np
import pytest

import qubex as qx
from qubex.pulse import (
    Arbitrary,
    Blank,
    Bump,
    Drag,
    FlatTop,
    Gaussian,
    Pulse,
    RaisedCosine,
    Rect,
    Sintegral,
    Waveform,
)

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """Pulse should inherit from Waveform."""
    assert issubclass(Pulse, Waveform)


def test_direct_pulse_instantiation_is_deprecated():
    """Pulse should emit a deprecation warning on direct instantiation."""
    with pytest.warns(DeprecationWarning, match="Arbitrary"):
        pulse = Pulse([0.1])

    assert pulse.values == pytest.approx([0.1])


@pytest.mark.parametrize(
    "pulse_cls",
    [
        Arbitrary,
        Blank,
        Bump,
        Drag,
        FlatTop,
        Gaussian,
        RaisedCosine,
        Rect,
        Sintegral,
    ],
)
def test_builtin_pulse_subclasses_define_sample_values(pulse_cls):
    """Built-in Pulse subclasses should define _sample_values explicitly."""
    assert "_sample_values" in pulse_cls.__dict__


def test_pulse_lazily_materializes_values():
    """Pulse should materialize sampled values lazily on first access only."""

    class _ConstantPulse(Pulse):
        def __init__(self, *, duration: float):
            self.sample_call_count = 0
            super().__init__(duration=duration)

        def _sample_values(self):
            self.sample_call_count += 1
            return np.ones(self.length, dtype=np.complex128)

    pulse = _ConstantPulse(duration=3 * dt)
    assert pulse.length == 3
    assert pulse.sample_call_count == 0

    assert pulse.values == pytest.approx([1, 1, 1])
    assert pulse.sample_call_count == 1

    assert pulse.values == pytest.approx([1, 1, 1])
    assert pulse.sample_call_count == 1


def test_pulse_materializes_values_on_init_when_lazy_is_false():
    """Pulse should materialize sampled values at init when lazy is False."""

    class _ConstantPulse(Pulse):
        def __init__(self, *, duration: float, lazy: bool):
            self.sample_call_count = 0
            super().__init__(duration=duration, lazy=lazy)
            self._finalize_initialization()

        def _sample_values(self):
            self.sample_call_count += 1
            return np.ones(self.length, dtype=np.complex128)

    pulse = _ConstantPulse(duration=3 * dt, lazy=False)
    assert pulse.length == 3
    assert pulse.sample_call_count == 1

    assert pulse.values == pytest.approx([1, 1, 1])
    assert pulse.sample_call_count == 1


def test_pulse_with_values_ignores_lazy_flag():
    """Pulse should not call sampler when explicit values are provided."""

    class _ConstantPulse(Pulse):
        def __init__(self, *, values, lazy: bool):
            self.sample_call_count = 0
            super().__init__(values=values, lazy=lazy)

        def _sample_values(self):
            self.sample_call_count += 1
            return np.ones(self.length, dtype=np.complex128)

    pulse = _ConstantPulse(values=[1, 2, 3], lazy=False)
    assert pulse.length == 3
    assert pulse.sample_call_count == 0
    assert pulse.values == pytest.approx([1, 2, 3])


def test_pulse_samples_on_values_access_when_values_is_none(monkeypatch):
    """Pulse should sample on values access when values is None."""
    del monkeypatch

    class _SampledPulse(Pulse):
        def __init__(self, *, duration: float):
            self.calls = 0
            super().__init__(duration=duration)

        def _sample_values(self):
            self.calls += 1
            return np.ones(self.length, dtype=np.complex128)

    pulse = _SampledPulse(duration=3 * dt)
    assert pulse.length == 3
    assert pulse.calls == 0

    _ = pulse.values
    assert pulse.calls == 1


def test_pulse_subclass_without_sampler_override_is_deprecated():
    """Pulse subclass without _sample_values override should be deprecated."""

    class _NoSamplerPulse(Pulse):
        def __init__(self, *, duration: float):
            super().__init__(duration=duration)

    with pytest.warns(DeprecationWarning, match="_sample_values"):
        pulse = _NoSamplerPulse(duration=3 * dt)

    assert pulse.values == pytest.approx([0, 0, 0])


def test_copy_materializes_lazy_pulse_once():
    """Copy should materialize lazy pulse values once and reuse cached samples."""
    calls = {"count": 0}

    class _ConstantPulse(Pulse):
        def __init__(self, *, duration: float):
            super().__init__(duration=duration)

        def _sample_values(self):
            calls["count"] += 1
            return np.ones(self.length, dtype=np.complex128)

    pulse = _ConstantPulse(duration=3 * dt)
    copied = pulse.copy()

    assert calls["count"] == 1
    assert copied.values == pytest.approx([1, 1, 1])
    assert calls["count"] == 1


def test_shallow_copy_materializes_lazy_pulse_once():
    """Shallow copy should materialize lazy pulse values once and reuse cache."""
    calls = {"count": 0}

    class _ConstantPulse(Pulse):
        def __init__(self, *, duration: float):
            super().__init__(duration=duration)

        def _sample_values(self):
            calls["count"] += 1
            return np.ones(self.length, dtype=np.complex128)

    pulse = _ConstantPulse(duration=3 * dt)
    copied = pycopy.copy(pulse)

    assert calls["count"] == 1
    assert copied.values == pytest.approx([1, 1, 1])
    assert calls["count"] == 1


def test_scalar_transforms_reuse_materialized_samples():
    """Scalar transforms should reuse sampled values via shallow copy."""

    class _ConstantPulse(Pulse):
        def __init__(self, *, duration: float):
            super().__init__(duration=duration)

        def _sample_values(self):
            return np.ones(self.length, dtype=np.complex128)

    pulse = _ConstantPulse(duration=3 * dt)
    scaled = pulse.scaled(2.0)
    detuned = pulse.detuned(0.001)
    shifted = pulse.shifted(np.pi / 2)

    source_values = pulse.__dict__.get("_values")
    assert source_values is not None
    assert scaled.__dict__.get("_values") is source_values
    assert detuned.__dict__.get("_values") is source_values
    assert shifted.__dict__.get("_values") is source_values
