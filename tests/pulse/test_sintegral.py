"""Tests for sine-integral pulses and helpers."""

from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qxpulse.library.sintegral import (
    MultiDerivativeSintegral,
    Sintegral,
    sin_pow_derivative,
    sin_pow_integral,
)
from qxpulse.pulse import Pulse


def test_sin_pow_integral_derivative_matches():
    """Given n, when differentiating the integral, then sin^n is recovered."""
    x = np.linspace(0.1, np.pi - 0.1, 200)
    integral = sin_pow_integral(x, n=5)
    derivative = np.gradient(integral, x, edge_order=2)
    mask = slice(2, -2)
    assert_allclose(
        derivative[mask],
        np.sin(x[mask]) ** 5,
        rtol=1e-4,
        atol=5e-4,
    )


def test_sin_pow_integral_invalid_power():
    """Given an invalid power, then sin_pow_integral raises ValueError."""
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        sin_pow_integral([0.0], n=-1)
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        sin_pow_integral([0.0], n=cast(int, 2.5))


def test_sin_pow_derivative_orders():
    """Given n and m, then sin_pow_derivative matches expected derivatives."""
    x = np.linspace(0.1, 2.0, 120)
    assert_allclose(sin_pow_derivative(x, n=3, m=0), np.sin(x) ** 3, rtol=1e-8)
    expected_first = 3 * (np.sin(x) ** 2) * np.cos(x)
    assert_allclose(sin_pow_derivative(x, n=3, m=1), expected_first, rtol=1e-8)


def test_sin_pow_derivative_invalid_arguments():
    """Given invalid arguments, then sin_pow_derivative raises ValueError."""
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        sin_pow_derivative([0.0], n=-1, m=0)
    with pytest.raises(ValueError, match="m must be a non-negative integer"):
        sin_pow_derivative([0.0], n=2, m=cast(int, 1.5))


def test_sintegral_func_beta_zero_matches_base():
    """Given beta zero, then Sintegral.func equals the uncorrected waveform."""
    duration = 6 * Pulse.SAMPLING_PERIOD
    t = np.linspace(0, duration, 8)
    base = Sintegral.func(t, duration=duration, amplitude=1.1, power=2)
    beta_zero = Sintegral.func(
        t,
        duration=duration,
        amplitude=1.1,
        power=2,
        beta=0.0,
    )
    assert_allclose(beta_zero, base)


def test_sintegral_func_outside_range_zeroed():
    """Given t outside [0, duration], then Sintegral.func returns zeros."""
    duration = 4 * Pulse.SAMPLING_PERIOD
    t = np.array([-1.0, duration + 0.5])
    values = Sintegral.func(t, duration=duration, amplitude=1.0, power=3)
    assert values[0] == 0
    assert values[-1] == 0


def test_sintegral_func_requires_positive_duration():
    """Given a zero duration, then Sintegral.func raises ValueError."""
    with pytest.raises(ValueError, match=r"Duration cannot be zero\."):
        Sintegral.func([0.0], duration=0, amplitude=1.0, power=2)


def test_multiderivative_sintegral_even_odd_contributions():
    """Given betas, then even orders affect the real part and odd orders the imaginary."""
    duration = 6 * Pulse.SAMPLING_PERIOD
    betas: dict[int, float] = {1: 0.35, 2: 0.15}
    amplitude = 1.0
    power: int = 2
    t = np.linspace(0, duration, int(duration / Pulse.SAMPLING_PERIOD) + 1)
    values = MultiDerivativeSintegral.func(
        t,
        duration=duration,
        amplitude=amplitude,
        power=power,
        betas=betas,
    )
    scale = amplitude / (
        sin_pow_integral(np.pi, n=power) - sin_pow_integral(0, n=power)
    )
    omega = sin_pow_integral(2 * np.pi * t / duration, n=power)
    omega -= sin_pow_integral(0, n=power)
    omega *= scale

    expected_real = omega.copy()
    expected_imag = np.zeros_like(omega)
    for order, beta in betas.items():
        derivative = sin_pow_derivative(
            2 * np.pi * t / duration,
            n=power,
            m=order - 1,
        )
        derivative *= scale * (2 * np.pi / duration) ** order
        term = beta * derivative
        if order % 2 == 0:
            expected_real += term
        else:
            expected_imag += term
    half_threshold = (duration * 0.5) // Pulse.SAMPLING_PERIOD * Pulse.SAMPLING_PERIOD
    mask = t <= half_threshold
    assert_allclose(values.real[mask], expected_real[mask], rtol=1e-7)
    assert_allclose(values.imag[mask], expected_imag[mask], rtol=1e-7)


def test_multiderivative_without_betas_matches_sintegral():
    """Given no betas, then MultiDerivativeSintegral matches Sintegral."""
    duration = 6 * Pulse.SAMPLING_PERIOD
    t = np.linspace(0, duration, 15)
    power: int = 3
    baseline = Sintegral.func(t, duration=duration, amplitude=1.2, power=power)
    multi = MultiDerivativeSintegral.func(
        t,
        duration=duration,
        amplitude=1.2,
        power=power,
        betas=None,
    )
    assert_allclose(multi, baseline)
