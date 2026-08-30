"""Tests for SQUAD ramp windows and sampled CD conventions."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qxpulse import FlatTop
from qxpulse.library.squad import Squad, TukeyWindowConfig
from scipy.integrate import quad


@pytest.mark.parametrize("sampling_period", [0.1, 2.0])
@pytest.mark.parametrize("delta", [-0.8, 0.8])
@pytest.mark.parametrize("factor", [None, 0.0, 1.0, -1.0])
def test_tukey_midpoints_match_hann(sampling_period, delta, factor):
    """Tukey positions (0.5, 0.5) reproduce the full sampled Hann I/Q pulse."""
    kwargs: dict[str, Any] = dict(
        duration=40.0,
        amplitude=0.6,
        delta=delta,
        tau=12.0,
        factor=factor,
    )
    hann = Squad(**kwargs, window="hann", sampling_period=sampling_period)
    tukey = Squad(
        **kwargs,
        window={"type": "tukey", "rise_end": 0.5, "fall_start": 0.5},
        sampling_period=sampling_period,
    )

    assert_allclose(tukey.values, hann.values, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("window", ["tukey", {"type": "tukey"}])
def test_tukey_defaults_match_hann(window):
    """Selecting Tukey without positions defaults to the Hann limit."""
    kwargs: dict[str, Any] = dict(duration=40.0, amplitude=0.6, delta=0.8, tau=12.0)
    assert_allclose(
        Squad(**kwargs, window=window).values,
        Squad(**kwargs).values,
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("rise_end", "fall_start"),
    [
        (0.2, 0.7),
        (0.1, 0.4),
        (0.7, 0.9),
        (0.3, 0.3),
        (0, 0.7),
        (0.2, 1),
        (0, 0),
        (1, 1),
        (0, 1),
    ],
)
def test_tukey_matches_independent_window_integral(rise_end, fall_start):
    """SQUAD uses the normalized integral of the specified asymmetric Tukey window."""

    def window(u):
        if u < rise_end:
            return 0.5 * (1 - np.cos(np.pi * u / rise_end))
        if u > fall_start:
            return 0.5 * (1 - np.cos(np.pi * (1 - u) / (1 - fall_start)))
        return 1.0

    u = np.unique(np.r_[np.linspace(0, 1, 17), rise_end, fall_start])
    area = quad(window, 0, 1, points=[rise_end, fall_start], epsabs=1e-13)[0]
    g = np.array(
        [
            quad(
                window,
                0,
                x,
                points=[p for p in (rise_end, fall_start) if p < x],
                epsabs=1e-13,
            )[0]
            / area
            for x in u
        ]
    )
    s = np.sin(np.arctan(0.6 / 0.8)) * g
    expected = 0.8 * s / np.sqrt(1 - s**2)

    actual = Squad.func(
        12 * u,
        duration=40,
        amplitude=0.6,
        delta=0.8,
        tau=12,
        factor=0,
        window={"type": "tukey", "rise_end": rise_end, "fall_start": fall_start},
    )

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert np.all(np.diff(actual.real) >= -1e-14)


def test_tukey_rectangle_matches_constant_adiabatic_ramp():
    """Tukey positions (0, 1) reproduce the existing constant-adiabatic window."""
    kwargs: dict[str, Any] = dict(duration=40, amplitude=0.6, delta=0.8, tau=12)
    assert_allclose(
        Squad(
            **kwargs, window={"type": "tukey", "rise_end": 0, "fall_start": 1}
        ).values,
        Squad(**kwargs, window="none").values,
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_tukey_constructor_matches_func_and_preserves_pulse_symmetry(lazy):
    """An asymmetric ramp window is reversed on the falling side of the pulse."""
    kwargs: dict[str, Any] = dict(
        duration=40,
        amplitude=0.6,
        delta=0.8,
        tau=12,
        factor=0.7,
        window={"type": "tukey", "rise_end": 0.2, "fall_start": 0.6},
    )
    pulse = Squad(**kwargs, sampling_period=0.1, lazy=lazy)
    t = (np.arange(pulse.length) + 0.5) * pulse.sampling_period
    values = pulse.values

    assert_allclose(values, Squad.func(t, **kwargs), rtol=1e-12, atol=1e-14)
    assert_allclose(values.real, values.real[::-1], rtol=1e-12, atol=1e-14)
    assert_allclose(values.imag, -values.imag[::-1], rtol=1e-12, atol=1e-14)
    assert_allclose(values[(t >= 12) & (t <= 28)].real, 0.6, rtol=0, atol=1e-14)


@pytest.mark.parametrize("delta", [-0.8, 0.8])
@pytest.mark.parametrize("factor", [-1.0, 0.0, 1.0])
def test_tukey_preserves_direct_cd_sign(delta, factor):
    """Direct SQUAD keeps Q = factor * delta * dI/dt / (delta**2 + I**2)."""
    t = np.linspace(0, 40, 401)
    values = Squad.func(
        t,
        duration=40,
        amplitude=0.6,
        delta=delta,
        tau=12,
        factor=factor,
        window={"type": "tukey", "rise_end": 0.2, "fall_start": 0.7},
    )
    expected_q = (
        factor * delta * np.gradient(values.real, t) / (delta**2 + values.real**2)
    )
    assert_allclose(values.imag, expected_q, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("sampling_period", [0.1, 2.0])
def test_flat_top_forwards_tukey_positions_with_existing_cd_mapping(sampling_period):
    """FlatTop forwards the window positions and retains its opposite CD factor convention."""
    kwargs: dict[str, Any] = dict(
        duration=40,
        amplitude=0.6,
        delta=0.8,
        tau=12,
        window={"type": "tukey", "rise_end": 0.2, "fall_start": 0.7},
    )
    direct = Squad(**kwargs, factor=1, sampling_period=sampling_period)
    flat = FlatTop(
        **kwargs,
        type="Squad",
        correction_type="CD",
        correction_factor=-1,
        sampling_period=sampling_period,
    )
    assert_allclose(flat.values, direct.values, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    ("rise_end", "fall_start"),
    [
        (-0.1, 0.5),
        (0.2, 1.1),
        (0.8, 0.2),
        (np.nan, 0.5),
        (0.5, np.nan),
        (np.inf, 1),
        (0, np.inf),
        (-np.inf, 1),
    ],
)
@pytest.mark.parametrize("duration", [0, 40])
def test_tukey_rejects_invalid_positions_at_public_entry_points(
    rise_end, fall_start, duration
):
    """Nonfinite, out-of-range, or reversed Tukey positions fail even for lazy or empty pulses."""
    kwargs: dict[str, Any] = dict(
        duration=duration,
        amplitude=0.6,
        delta=0.8,
        tau=0,
        factor=0,
        window={"type": "tukey", "rise_end": rise_end, "fall_start": fall_start},
    )
    message = r"0 <= rise_end <= fall_start <= 1"
    with pytest.raises(ValueError, match=message):
        Squad(**kwargs)
    with pytest.raises(ValueError, match=message):
        Squad.func([0.0], **kwargs)


@pytest.mark.parametrize("window", ["none", "hann", "beta"])
def test_window_dict_matches_existing_string(window):
    """A dictionary with only type reproduces the corresponding string defaults."""
    kwargs: dict[str, Any] = dict(duration=40, amplitude=0.6, delta=0.8, tau=12)
    assert_allclose(
        Squad(**kwargs, window={"type": window}).values,
        Squad(**kwargs, window=window).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("wrapper", [False, True])
def test_window_dict_is_detached_from_caller(lazy, wrapper):
    """Changing the caller's dictionary cannot change either pulse's sampled waveform."""
    window = {"type": "tukey", "rise_end": 0.2, "fall_start": 0.7}
    kwargs: dict[str, Any] = dict(
        duration=40, amplitude=0.6, delta=0.8, tau=12, window=window, lazy=lazy
    )
    if wrapper:
        pulse = FlatTop(**kwargs, type="Squad")
        expected = FlatTop(**kwargs, type="Squad").values.copy()
    else:
        pulse = Squad(**kwargs)
        expected = Squad(**kwargs).values.copy()
    assert window == {"type": "tukey", "rise_end": 0.2, "fall_start": 0.7}
    window["rise_end"] = 0.6
    window["fall_start"] = 0.9
    assert_allclose(pulse.values, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "window",
    [
        {},
        {"rise_end": 0.2},
        {"type": "unknown"},
        {"type": []},
        {"type": "hann", "rise_end": 0.2},
        {"type": "tukey", "rise_edn": 0.2},
        {"type": "tukey", "mode": 0.2},
        {"type": "beta", "rise_end": 0.2},
    ],
)
def test_window_dict_rejects_invalid_schema(window):
    """Missing types, unknown windows, misspellings, and unrelated keys are rejected."""
    kwargs: dict[str, Any] = dict(
        duration=0, amplitude=0.6, delta=0.8, tau=0, window=window
    )
    with pytest.raises(ValueError, match="window"):
        Squad(**kwargs)
    with pytest.raises(ValueError, match="window"):
        Squad.func([], **kwargs)
    with pytest.raises(ValueError, match="window"):
        FlatTop(**kwargs, type="Squad")


@pytest.mark.parametrize("bad_value", ["0.2", None, True, [0.2], 0.2j])
def test_tukey_dict_requires_real_positions(bad_value):
    """Position values must be real numbers rather than strings, booleans, or containers."""
    with pytest.raises(TypeError, match="rise_end"):
        Squad(
            duration=40,
            amplitude=0.6,
            delta=0.8,
            tau=12,
            window={"type": "tukey", "rise_end": bad_value},
        )


@pytest.mark.parametrize("position", ["rise_end", "fall_start"])
def test_single_tukey_position_defaults_the_other_to_half(position):
    """Either omitted Tukey position independently defaults to 0.5."""
    value = 0.2 if position == "rise_end" else 0.7
    kwargs: dict[str, Any] = dict(duration=40, amplitude=0.6, delta=0.8, tau=12)
    supplied: TukeyWindowConfig = {"type": "tukey"}
    expected: TukeyWindowConfig = {"type": "tukey", "rise_end": 0.5, "fall_start": 0.5}
    if position == "rise_end":
        supplied["rise_end"] = expected["rise_end"] = value
    else:
        supplied["fall_start"] = expected["fall_start"] = value
    assert_allclose(
        Squad(**kwargs, window=supplied).values,
        Squad(**kwargs, window=expected).values,
        rtol=0,
        atol=0,
    )


def test_beta_dict_matches_legacy_parameters():
    """Beta dictionary settings reproduce the existing beta_mode and beta_sum API."""
    kwargs: dict[str, Any] = dict(duration=40, amplitude=0.6, delta=0.8, tau=12)
    assert_allclose(
        Squad(**kwargs, window={"type": "beta", "mode": 0.4, "sum": 6.0}).values,
        Squad(**kwargs, window="beta", beta_mode=0.4, beta_sum=6.0).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("options", [{"beta_mode": 0.4}, {"beta_sum": 6.0}])
def test_dictionary_rejects_conflicting_legacy_beta_options(options):
    """Dictionary settings cannot silently override separate nondefault beta arguments."""
    kwargs: dict[str, Any] = dict(
        duration=40,
        amplitude=0.6,
        delta=0.8,
        tau=12,
        window={"type": "beta", "mode": 0.2},
        **options,
    )
    with pytest.raises(ValueError, match=r"beta_mode.*beta_sum"):
        Squad(**kwargs)


@pytest.mark.parametrize(
    "window",
    [
        {"type": "beta", "mode": np.nan},
        {"type": "beta", "sum": np.inf},
        {"type": "beta", "mode": -0.1},
        {"type": "beta", "mode": 1.1},
        {"type": "beta", "sum": 2.0},
    ],
)
def test_beta_dict_rejects_invalid_shape(window):
    """Beta dictionaries require a finite mode in [0, 1] and a finite sum above two."""
    with pytest.raises(ValueError, match="window"):
        Squad(duration=40, amplitude=0.6, delta=0.8, tau=12, window=window)


@pytest.mark.parametrize("option", ["tukey_rise_end", "tukey_fall_start"])
def test_tukey_positions_are_not_separate_pulse_options(option):
    """The withdrawn unpublished Tukey keywords must not be silently accepted."""
    kwargs: dict[str, Any] = dict(
        duration=40,
        amplitude=0.6,
        delta=0.8,
        tau=12,
        window={"type": "tukey"},
        **{option: 0.2},
    )
    with pytest.raises(TypeError, match=option):
        Squad(**kwargs)
    with pytest.raises(TypeError, match=option):
        Squad.func([0.0], **kwargs)
