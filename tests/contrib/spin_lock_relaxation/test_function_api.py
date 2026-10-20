"""Tests for tilted-axis spin-lock relaxation helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from plotly.graph_objects import Figure

from qubex.contrib import spin_lock_relaxation
from qubex.contrib.experiment import spin_lock_relaxation as experiment_api
from qubex.contrib.experiment.spin_lock_relaxation import (
    _build_sequence,
    _fit_global_model,
    _normalization_noise,
    _predict_expectations,
    _resolve_drive,
    _resolve_duration_range,
)
from qubex.pulse import FlatTop, Rect

spin_lock_module = importlib.import_module(
    "qubex.contrib.experiment.spin_lock_relaxation"
)


def test_spin_lock_relaxation_is_exported_from_contrib() -> None:
    """The new experiment should be available from both contrib namespaces."""
    assert experiment_api is spin_lock_relaxation


@pytest.mark.parametrize(
    ("t1", "t2_echo", "expected_start", "expected_stop", "expected_size"),
    [
        (10_000.0, 20_000.0, 1_000.0, 100_000.0, 21),
        (10_000.0, None, 1_000.0, 50_000.0, 21),
        (None, 500.0, 100.0, 2_500.0, 21),
        (None, None, 100.0, 200_000.0, 31),
    ],
)
def test_default_duration_range_uses_available_coherence_times(
    t1: float | None,
    t2_echo: float | None,
    expected_start: float,
    expected_stop: float,
    expected_size: int,
) -> None:
    """Default durations should use any available T1 or T2 echo value."""

    class _Loader:
        @staticmethod
        def load_param_data(name: str) -> dict[str, float | None]:
            return {"Q00": t1 if name == "t1" else t2_echo}

    class _Util:
        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            array = np.asarray(values, dtype=float)
            return np.round(array / sampling_period) * sampling_period

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=1.0),
            system_manager=SimpleNamespace(config_loader=_Loader()),
            util=_Util(),
        )
    )

    durations = _resolve_duration_range(exp, "Q00", None)

    assert durations.size == expected_size
    assert durations[0] == pytest.approx(expected_start)
    assert durations[-1] == pytest.approx(expected_stop)


def test_default_duration_range_rejects_unmeasurably_short_coherence() -> None:
    """A stored coherence time below the minimum grid should fail clearly."""

    class _Loader:
        @staticmethod
        def load_param_data(name: str) -> dict[str, float | None]:
            return {"Q00": 10.0 if name == "t1" else None}

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=1.0),
            system_manager=SimpleNamespace(config_loader=_Loader()),
            util=SimpleNamespace(),
        )
    )

    with pytest.raises(ValueError, match="too short"):
        _resolve_duration_range(exp, "Q00", None)


def test_drive_amplitude_takes_precedence_over_rabi_frequency() -> None:
    """An explicit amplitude should determine both drive and reported Rabi rate."""

    class _Pulse:
        @staticmethod
        def calc_control_amplitude(_target: str, _frequency: float) -> float:
            raise AssertionError("Rabi input must be ignored when amplitude is set.")

        @staticmethod
        def calc_rabi_rate(_target: str, amplitude: float) -> float:
            return 0.1 * amplitude

    exp: Any = SimpleNamespace(pulse=_Pulse())

    amplitude, rabi_frequency = _resolve_drive(
        exp,
        target="Q00",
        drive_amplitude={"Q00": 0.3},
        rabi_frequency={"different_target": 0.08},
    )

    assert amplitude == pytest.approx(0.3)
    assert rabi_frequency == pytest.approx(0.03)


def test_sequence_uses_flattop_duration_excluding_both_ramps() -> None:
    """The requested duration should describe only the pulse flat-top."""

    class _Pulse:
        @staticmethod
        def get_pulse_for_state(_target: str, _state: str) -> Rect:
            return Rect(duration=4.0, amplitude=0.1)

        @staticmethod
        def y90m(_target: str) -> Rect:
            return Rect(duration=4.0, amplitude=0.1)

        @staticmethod
        def x90(_target: str) -> Rect:
            return Rect(duration=4.0, amplitude=0.1)

    exp: Any = SimpleNamespace(pulse=_Pulse())

    schedule = _build_sequence(
        exp,
        target="Q00",
        flat_duration=100.0,
        drive_amplitude=0.2,
        ramp_duration=16.0,
        initial_state="+",
        basis="X",
    )

    pulses = schedule.get_sequences()["Q00"].elements
    spin_lock_pulse = next(pulse for pulse in pulses if isinstance(pulse, FlatTop))
    assert spin_lock_pulse.duration == pytest.approx(132.0)
    assert spin_lock_pulse.tau == pytest.approx(16.0)


def test_multiple_targets_are_measured_independently_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple targets should run as separate measurements in input order."""
    measured_targets: list[str] = []

    class _Pulse:
        @staticmethod
        def validate_rabi_params(targets: list[str]) -> None:
            assert targets == ["Q01", "Q00"]

    def measure_target(_exp: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        measured_targets.append(kwargs["target"])
        shape = (3, 7)
        return np.zeros(shape, dtype=complex), np.zeros(shape)

    def analyze_target(_exp: Any, **kwargs: Any) -> tuple[dict[str, str], Figure]:
        return {"target": kwargs["target"]}, Figure()

    monkeypatch.setattr(
        spin_lock_module,
        "_resolve_duration_range",
        lambda *_args: np.asarray([100.0, 200.0, 300.0]),
    )
    monkeypatch.setattr(
        spin_lock_module,
        "_resolve_drive",
        lambda *_args, **_kwargs: (0.2, 0.02),
    )
    monkeypatch.setattr(spin_lock_module, "_measure_target", measure_target)
    monkeypatch.setattr(spin_lock_module, "_analyze_target", analyze_target)
    exp: Any = SimpleNamespace(pulse=_Pulse())

    result = spin_lock_relaxation(exp, ["Q01", "Q00"], plot=False)

    assert measured_targets == ["Q01", "Q00"]
    assert result.data["targets"] == ["Q01", "Q00"]


def test_global_fit_recovers_synthetic_relaxation_parameters() -> None:
    """The seven-observable fit should recover a noiseless tilted-axis model."""
    durations = np.geomspace(100.0, 50_000.0, 21)
    phases = np.linspace(-1.0, 2.0, durations.size)
    expected = {
        "theta": 0.18,
        "t1_rho": 11_000.0,
        "t2_rho": 7_000.0,
        "p_inf": -0.22,
        "a0": 0.96,
        "b0": 0.91,
        "p0": 0.04,
    }
    observations = _predict_expectations(
        durations,
        phases=phases,
        **expected,
    )

    fit = _fit_global_model(durations, observations, sigma=0.01)

    assert fit["success"] is True
    for name, value in expected.items():
        assert fit["parameters"][name] == pytest.approx(
            value,
            rel=2e-3,
            abs=2e-4,
        )


def test_global_fit_reports_covariance_based_uncertainties() -> None:
    """A successful global fit should expose finite covariance uncertainties."""
    durations = np.geomspace(100.0, 20_000.0, 15)
    observations = _predict_expectations(
        durations,
        theta=0.1,
        t1_rho=8_000.0,
        t2_rho=5_000.0,
        p_inf=0.15,
        a0=0.95,
        b0=0.9,
        p0=0.02,
        phases=np.linspace(0.2, 1.7, durations.size),
    )
    rng = np.random.default_rng(7)
    observations = observations + rng.normal(0.0, 0.003, observations.shape)

    fit = _fit_global_model(durations, observations, sigma=0.01)

    errors = fit["errors"]
    assert all(
        np.isfinite(errors[name]) for name in ("theta", "t1_rho", "t2_rho", "p_inf")
    )
    assert all(errors[name] >= 0 for name in ("theta", "t1_rho", "t2_rho", "p_inf"))


def test_global_fit_rejects_nonfinite_expectations() -> None:
    """Nonfinite expectation values should fail before numerical optimization."""
    durations = np.asarray([100.0, 200.0, 300.0])
    observations = np.zeros((3, 7))
    observations[1, 2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        _fit_global_model(durations, observations, sigma=0.01)


def test_global_fit_rejects_nonincreasing_durations() -> None:
    """The decay fit should reject an unordered or repeated time axis."""
    observations = np.zeros((3, 7))

    with pytest.raises(ValueError, match="strictly increasing"):
        _fit_global_model([100.0, 300.0, 200.0], observations, sigma=0.01)


def test_normalization_noise_rejects_invalid_calibration() -> None:
    """Invalid Rabi normalization noise should not become a fabricated error bar."""
    exp: Any = SimpleNamespace(
        pulse=SimpleNamespace(
            rabi_params={
                "Q00": SimpleNamespace(noise=np.nan, amplitude=0.5),
            }
        )
    )

    with pytest.raises(ValueError, match="normalization noise"):
        _normalization_noise(exp, "Q00")
