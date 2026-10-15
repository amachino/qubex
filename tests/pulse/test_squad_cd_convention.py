"""Public CD convention, migration, and physical two-level regression tests."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qxpulse import FlatTop
from qxpulse.library.squad import Squad
from scipy.linalg import expm


@pytest.mark.parametrize("delta", [-0.8, 0.8])
@pytest.mark.parametrize("coefficient", [None, 0.0, 0.5, 1.0, -1.0])
@pytest.mark.parametrize("lazy", [False, True])
def test_public_cd_apis_have_identical_sign(delta, coefficient, lazy):
    """Both public constructors produce identical I/Q for equal CD settings."""
    kwargs: dict[str, Any] = dict(
        duration=40, amplitude=0.6, tau=12, delta=delta, sampling_period=0.1
    )
    direct = Squad(**kwargs, correction_factor=coefficient, lazy=lazy)
    flat = FlatTop(
        **kwargs,
        type="Squad",
        correction_type="CD",
        correction_factor=coefficient,
        lazy=lazy,
    )
    assert_allclose(direct.values, flat.values, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("api", [Squad, Squad.func])
@pytest.mark.parametrize("legacy", [None, 0.0, 1.0, -0.5])
def test_legacy_factor_is_not_silently_reinterpreted(api, legacy):
    """Removed factor keywords fail instead of silently reversing an existing pulse."""
    args = () if api is Squad else (np.arange(0.5, 40, 1),)
    with pytest.raises(TypeError, match="factor"):
        api(*args, duration=40, amplitude=0.6, delta=-0.8, tau=12, factor=legacy)


def test_migrating_legacy_factor_preserves_iq():
    """Negating the old factor reproduces the released direct-SQUAD quadrature."""
    t = np.arange(0.05, 40, 0.1)
    legacy_factor = 0.7
    pulse = Squad.func(
        t,
        duration=40,
        amplitude=0.6,
        delta=-0.8,
        tau=12,
        correction_factor=-legacy_factor,
    )
    expected_q = (
        legacy_factor * (-0.8) * np.gradient(pulse.real, t) / (0.8**2 + pulse.real**2)
    )
    assert_allclose(pulse.imag, expected_q, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("amplitude", [0.6, 0.01])
def test_flat_top_cd_keeps_released_formula(amplitude):
    """FlatTop retains its existing negative CD sign and sampled derivative."""
    t = np.arange(0.05, 40, 0.1)
    pulse = FlatTop.func(
        t,
        duration=40,
        amplitude=amplitude,
        delta=-0.8,
        tau=12,
        type="Squad",
        correction_type="CD",
        correction_factor=0.5,
    )
    expected_q = -0.5 * (-0.8) * np.gradient(pulse.real, t) / (0.8**2 + pulse.real**2)
    assert_allclose(pulse.imag, expected_q, rtol=1e-12, atol=1e-14)


def test_weak_drive_cd_approaches_drag():
    """CD and DRAG share sign and strength in the weak-drive limit."""
    t = np.arange(0.05, 40, 0.1)
    kwargs: dict[str, Any] = dict(
        duration=40,
        amplitude=1e-4,
        delta=-0.8,
        tau=12,
        type="Squad",
        correction_factor=0.5,
    )
    cd = FlatTop.func(t, **kwargs, correction_type="CD")
    drag = FlatTop.func(t, **kwargs, correction_type="DRAG")
    assert_allclose(cd, drag, rtol=2e-8, atol=1e-14)


@pytest.mark.parametrize("api", [Squad, FlatTop])
def test_dimensionless_strength_and_output_scale(api):
    """Angular-rate construction with final scaling preserves command-unit I/Q."""
    k = 4.0
    kwargs: dict[str, Any] = dict(duration=40, tau=12, sampling_period=0.1)
    if api is FlatTop:
        kwargs.update(type="Squad", correction_type="CD")
    angular = api(
        **kwargs,
        amplitude=k * 0.6,
        delta=k * (-0.8),
        correction_factor=0.5,
        scale=1 / k,
    )
    command_reference = FlatTop(
        duration=40,
        tau=12,
        sampling_period=0.1,
        amplitude=0.6,
        delta=-0.8,
        type="Squad",
        correction_type="CD",
        correction_factor=0.5 / k,
    )
    assert_allclose(angular.values, command_reference.values, rtol=1e-12, atol=1e-14)


def test_cd_sign_follows_transition_minus_drive_hamiltonian():
    """CD returns a two-level eigenstate under H=(-delta Z+I X+Q Y)/2."""
    dt = 0.01
    t = np.arange(dt / 2, 12, dt)
    delta = -0.6
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    errors = []
    for strength in (0.0, 1.0):
        pulse = Squad.func(
            t,
            duration=12,
            tau=4,
            amplitude=0.9,
            delta=delta,
            correction_factor=strength,
        )
        state = np.array([1, 0], dtype=complex)
        for sample in pulse:
            h = (-delta * z + sample.real * x + sample.imag * y) / 2
            state = expm(-1j * dt * h) @ state
        errors.append(abs(state[1]) ** 2)
    assert errors[1] < 1e-5
    assert errors[0] > 1e-3
