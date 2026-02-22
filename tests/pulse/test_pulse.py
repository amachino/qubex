"""Tests for the Pulse base class."""

import copy as pycopy
import warnings

import numpy as np
import pytest
from qxpulse import (
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

import qubex as qx

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """Pulse should inherit from Waveform."""
    assert issubclass(Pulse, Waveform)


def test_arbitrary_inheritance():
    """Arbitrary should inherit from Pulse."""
    assert issubclass(Arbitrary, Pulse)


def test_direct_pulse_instantiation_is_deprecated():
    """Pulse should emit a deprecation warning on direct instantiation."""
    with pytest.warns(DeprecationWarning, match="Arbitrary"):
        pulse = Pulse([0.1])

    assert pulse.values == pytest.approx([0.1])


def test_arbitrary_instantiation_is_not_deprecated():
    """Arbitrary should instantiate without a deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pulse = Arbitrary([0.1, 0.2j])

    assert isinstance(pulse, Pulse)
    assert pulse.values == pytest.approx([0.1, 0.2j])


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
    calls = {"count": 0}
    original_sample_values = Pulse._sample_values  # noqa: SLF001

    def counting_sample_values(self):
        calls["count"] += 1
        return original_sample_values(self)

    monkeypatch.setattr(Pulse, "_sample_values", counting_sample_values)

    pulse = Pulse(duration=3 * dt, lazy=False)
    assert pulse.length == 3
    assert calls["count"] == 0

    _ = pulse.values
    assert calls["count"] == 1


def test_pulse_subclass_without_sampler_override_is_deprecated():
    """Pulse subclass without _sample_values override should be deprecated."""

    class _NoSamplerPulse(Pulse):
        def __init__(self, *, duration: float):
            super().__init__(duration=duration)

    with pytest.warns(DeprecationWarning, match="_sample_values"):
        pulse = _NoSamplerPulse(duration=3 * dt)

    assert pulse.values == pytest.approx([0, 0, 0])


def test_empty_init():
    """Pulse should be initialized with no parameters."""
    pulse = Pulse()
    assert pulse.name == "Pulse"
    assert pulse.length == 0
    assert pulse.duration == 0.0
    assert pulse.phase == 0.0
    assert pulse.scale == 1.0
    assert pulse.detuning == 0.0
    assert pulse.values == pytest.approx([])
    assert pulse.times == pytest.approx([])


def test_empty_list():
    """Pulse should be initialized with an empty list."""
    pulse = Pulse([])
    assert pulse.name == "Pulse"
    assert pulse.length == 0
    assert pulse.duration == 0.0
    assert pulse.phase == 0.0
    assert pulse.scale == 1.0
    assert pulse.detuning == 0.0
    assert pulse.values == pytest.approx([])
    assert pulse.times == pytest.approx([])


def test_init():
    """Pulse should be initialized with valid parameters."""
    pulse = Pulse([0, -0.5, +0.5j])
    assert pulse.length == 3
    assert pulse.duration == 3 * dt
    assert pulse.times == pytest.approx(np.arange(3) * dt)
    assert pulse.values == pytest.approx([0, -0.5, +0.5j])
    assert pulse.real == pytest.approx([0, -0.5, 0])
    assert pulse.imag == pytest.approx([0, 0, 0.5])
    assert pulse.abs == pytest.approx([0, 0.5, 0.5])
    assert pulse.angle == pytest.approx([0, np.pi, np.pi / 2])


def test_sampling_period_default():
    """Pulse should use the global default sampling period when not provided."""
    pulse = Pulse([0, 1])
    assert pulse.sampling_period == dt
    assert pulse.duration == 2 * dt
    assert pulse.times == pytest.approx(np.arange(2) * dt)


def test_sampling_period_override():
    """Pulse should use the provided sampling period per instance."""
    custom_dt = dt / 2
    pulse = Pulse([0, 1, 2], sampling_period=custom_dt)
    assert pulse.sampling_period == custom_dt
    assert pulse.duration == 3 * custom_dt
    assert pulse.times == pytest.approx(np.arange(3) * custom_dt)


def test_copy():
    """Pulse should be copied."""
    pulse = Pulse([1, 2, 3])
    copy = pulse.copy()
    assert isinstance(copy, Pulse)
    assert copy is not pulse
    assert copy.values == pytest.approx(pulse.values)


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


def test_paddded():
    """Pulse should be padded with zeros."""
    pulse = Pulse([1, 1, 1])
    padded = pulse.padded(10 * dt)
    assert padded != pulse
    assert padded.values == pytest.approx([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    padded_right = pulse.padded(10 * dt, "right")
    assert padded_right.values == pytest.approx([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    padded_left = pulse.padded(10 * dt, "left")
    assert padded_left.values == pytest.approx([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])


def test_scaled():
    """Pulse should be scaled by a given parameter."""
    pulse = Pulse([0.1, 0.2, 0.3])
    scaled = pulse.scaled(2)
    assert scaled.scale == 2
    assert scaled != pulse
    assert scaled.values == pytest.approx([0.2, 0.4, 0.6])
    assert scaled.scaled(1.5).values == pytest.approx([0.3, 0.6, 0.9])


def test_detuned():
    """Pulse should be detuned by a given parameter."""
    pulse = Pulse([0.1, 0.1, 0.1])
    detuned = pulse.detuned(0.001)
    assert detuned.detuning == 0.001
    assert detuned != pulse
    assert detuned.values == pytest.approx(
        [
            0.1,
            0.1 * np.exp(-1j * 0.001 * 2 * np.pi * dt),
            0.1 * np.exp(-2j * 0.001 * 2 * np.pi * dt),
        ]
    )


def test_shifted():
    """Pulse should be phase shifted by a given parameter."""
    pulse = Pulse([1, -1, 1j])
    shifted = pulse.shifted(np.pi / 2)
    assert shifted.phase == np.pi / 2
    assert shifted != pulse
    assert shifted.values == pytest.approx([1j, -1j, -1])


def test_repeated():
    """Pulse should be repeated a given number of times."""
    pulse = Pulse([1, 2, 3])
    repeated = pulse.repeated(3)
    assert repeated != pulse
    assert repeated.values == pytest.approx([1, 2, 3, 1, 2, 3, 1, 2, 3])


def test_inverted():
    """Pulse should be inverted."""
    pulse = Pulse([1, 2, 3])
    inverted = pulse.inverted()
    assert inverted != pulse
    assert inverted.values == pytest.approx([-3, -2, -1])
    assert inverted.inverted().values == pytest.approx(pulse.values)


def test_shape_values_preserves_internal_zeros():
    """Shape values should keep zero regions inside one pulse waveform."""
    pulse = Pulse(
        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        scale=2.0,
        sampling_period=0.4,
    )

    assert pulse.shape_values == pytest.approx(
        np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    )


def test_shape_values_removes_scale_and_phase():
    """Shape values should remove scale and phase from pulse values."""
    pulse = Pulse([1.0 + 0.0j], scale=2.0, phase=np.pi / 2)

    assert pulse.shape_values == pytest.approx(
        np.array([1.0 + 0.0j], dtype=np.complex128)
    )


def test_shape_values_of_all_zero_pulse():
    """Shape values should stay all-zero for all-zero pulses."""
    pulse = Pulse([0.0 + 0.0j, 0.0 + 0.0j])

    assert pulse.shape_values == pytest.approx(
        np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    )


def test_shape_hash_ignores_scale_and_phase():
    """Shape hash should be identical for pulses differing only in scale and phase."""
    values = [0.1 + 0.2j, 0.3 - 0.4j]
    pulse_a = Pulse(values, scale=0.5, phase=0.1)
    pulse_b = Pulse(values, scale=1.7, phase=-0.9)

    shape_hash_a = pulse_a.shape_hash
    shape_hash_b = pulse_b.shape_hash
    assert shape_hash_a == shape_hash_b


def test_shape_hash_depends_on_detuning():
    """Shape hash should differ when detuning changes."""
    values = [0.1 + 0.2j, 0.3 - 0.4j]
    pulse_a = Pulse(values, detuning=0.0)
    pulse_b = Pulse(values, detuning=0.02)

    shape_hash_a = pulse_a.shape_hash
    shape_hash_b = pulse_b.shape_hash
    assert shape_hash_a != shape_hash_b
