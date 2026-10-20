"""Tests for tilted-axis spin-lock relaxation helpers."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from plotly.graph_objects import Figure

from qubex.contrib import spin_lock_relaxation, spin_lock_relaxation_spectroscopy
from qubex.contrib.experiment import (
    spin_lock_relaxation as experiment_api,
    spin_lock_relaxation_spectroscopy as experiment_spectroscopy_api,
)
from qubex.contrib.experiment.spin_lock_relaxation import (
    MEASUREMENT_SETTINGS,
    _analyze_spectroscopy_target,
    _build_sequence,
    _fit_global_model,
    _make_figure,
    _measure_target_grid,
    _normalization_noise,
    _predict_expectations,
    _print_fit_summary,
    _resolve_drive,
    _resolve_duration_range,
    _resolve_spin_lock_frequency_range,
)
from qubex.pulse import FlatTop, Rect

spin_lock_module = importlib.import_module(
    "qubex.contrib.experiment.spin_lock_relaxation"
)


def test_spin_lock_relaxation_is_exported_from_contrib() -> None:
    """The new experiment should be available from both contrib namespaces."""
    assert experiment_api is spin_lock_relaxation


def test_spin_lock_relaxation_spectroscopy_is_exported_from_contrib() -> None:
    """The spectroscopy experiment should be available from both namespaces."""
    assert experiment_spectroscopy_api is spin_lock_relaxation_spectroscopy


def test_default_spin_lock_frequency_range_is_positive() -> None:
    """The default spectroscopy sweep should exclude the singular zero drive."""
    frequencies = _resolve_spin_lock_frequency_range(None)

    assert frequencies.size == 14
    assert np.all(frequencies > 0)
    assert frequencies[0] == pytest.approx(0.00519197056)
    assert frequencies[-1] == pytest.approx(0.12)


def test_default_spin_lock_frequency_range_is_returned_as_a_copy() -> None:
    """Mutating one resolved sweep must not change later default sweeps."""
    frequencies = _resolve_spin_lock_frequency_range(None)

    frequencies[0] = -1.0

    assert _resolve_spin_lock_frequency_range(None)[0] > 0


@pytest.mark.parametrize("frequencies", [[0.0, 0.01], [-0.01, 0.01]])
def test_spin_lock_frequency_range_rejects_nonpositive_values(
    frequencies: list[float],
) -> None:
    """The spectroscopy sweep should reject frequencies that cannot define shift."""
    with pytest.raises(ValueError, match="positive"):
        _resolve_spin_lock_frequency_range(frequencies)


@pytest.mark.parametrize(
    ("t1", "t2_echo", "expected_start", "expected_stop", "expected_size"),
    [
        (10_000.0, 20_000.0, 250.0, 400_000.0, 21),
        (10_000.0, None, 250.0, 200_000.0, 21),
        (None, 500.0, 100.0, 10_000.0, 21),
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
            return {"Q00": 1.0 if name == "t1" else None}

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=1.0),
            system_manager=SimpleNamespace(config_loader=_Loader()),
            util=SimpleNamespace(),
        )
    )

    with pytest.raises(ValueError, match="too short"):
        _resolve_duration_range(exp, "Q00", None)


def test_duration_range_rejects_zero_after_discretization() -> None:
    """Durations rounded to zero should fail before measurement."""

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
            measurement=SimpleNamespace(sampling_period=4.0),
            util=_Util(),
        )
    )

    with pytest.raises(ValueError, match="sampling grid"):
        _resolve_duration_range(exp, "Q00", [1.0, 4.0, 8.0, 12.0])


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


def test_frequency_duration_grid_is_measured_in_a_single_flattened_sweep() -> None:
    """The grid should preserve frequency-major, duration, then setting order."""
    sampled_pulses: list[tuple[float, float]] = []

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

    class _MeasurementService:
        @staticmethod
        def sweep_parameter(*, sequence: Any, sweep_range: Any, **_kwargs: Any) -> Any:
            indices = np.asarray(sweep_range)
            for index in indices:
                schedule = sequence(int(index))
                pulse = next(
                    element
                    for element in schedule.get_sequences()["Q00"].elements
                    if isinstance(element, FlatTop)
                )
                sampled_pulses.append((pulse.amplitude, pulse.duration))
            values = np.arange(indices.size, dtype=float)
            return SimpleNamespace(
                data={
                    "Q00": SimpleNamespace(
                        data=values.astype(complex),
                        normalized=values,
                    )
                }
            )

    exp: Any = SimpleNamespace(
        pulse=_Pulse(),
        measurement_service=_MeasurementService(),
        ctx=SimpleNamespace(util=SimpleNamespace(no_output=nullcontext)),
    )

    raw_data, expectations = _measure_target_grid(
        exp,
        target="Q00",
        durations=np.asarray([100.0, 200.0]),
        drive_amplitudes=np.asarray([0.1, 0.2]),
        ramp_duration=16.0,
        n_shots=100,
        shot_interval=1e-4,
    )

    assert raw_data.shape == (2, 2, 7)
    assert expectations.shape == (2, 2, 7)
    assert sampled_pulses[:14] == [(0.1, 132.0)] * 7 + [(0.1, 232.0)] * 7
    assert sampled_pulses[14:] == [(0.2, 132.0)] * 7 + [(0.2, 232.0)] * 7


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


def test_spectroscopy_measures_multiple_targets_independently_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spectroscopy targets should run as separate flattened sweeps in order."""
    measured_targets: list[str] = []

    class _Pulse:
        @staticmethod
        def validate_rabi_params(targets: list[str]) -> None:
            assert targets == ["Q01", "Q00"]

    def measure_grid(_exp: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        measured_targets.append(kwargs["target"])
        shape = (2, 3, 7)
        return np.zeros(shape, dtype=complex), np.zeros(shape)

    def analyze_target(
        _exp: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Figure]]:
        target = kwargs["target"]
        return (
            {
                "target": target,
                "fit_success": np.asarray([True, True]),
                "fit_messages": ["ok", "ok"],
            },
            {
                f"{target}_parameter_heatmaps": Figure(),
                f"{target}_relaxation_spectroscopy": Figure(),
            },
        )

    monkeypatch.setattr(
        spin_lock_module,
        "_resolve_duration_range",
        lambda *_args: np.asarray([100.0, 200.0, 300.0]),
    )
    monkeypatch.setattr(
        spin_lock_module,
        "_resolve_frequency_sweep",
        lambda *_args, **_kwargs: (
            np.asarray([0.1, 0.2]),
            np.asarray([0.01, 0.02]),
        ),
    )
    monkeypatch.setattr(spin_lock_module, "_measure_target_grid", measure_grid)
    monkeypatch.setattr(
        spin_lock_module,
        "_analyze_spectroscopy_target",
        analyze_target,
    )
    exp: Any = SimpleNamespace(pulse=_Pulse())

    result = spin_lock_relaxation_spectroscopy(
        exp,
        ["Q01", "Q00"],
        spin_lock_frequency_range=[0.01, 0.02],
        plot=False,
    )

    assert measured_targets == ["Q01", "Q00"]
    assert result.data["targets"] == ["Q01", "Q00"]
    assert set(result.figures or {}) == {
        "Q01_parameter_heatmaps",
        "Q01_relaxation_spectroscopy",
        "Q00_parameter_heatmaps",
        "Q00_relaxation_spectroscopy",
    }


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


def test_global_fit_rejects_an_unconverged_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconverged optimizer should not produce a reported fit."""
    monkeypatch.setattr(
        "scipy.optimize.least_squares",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            message="maximum evaluations exceeded",
        ),
    )
    durations = np.asarray([100.0, 200.0, 300.0])
    observations = np.zeros((3, 7))

    with pytest.raises(RuntimeError, match="maximum evaluations exceeded"):
        _fit_global_model(durations, observations, sigma=0.01)


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


def test_fit_summary_is_printed_separately_from_figure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fit and derived parameters should be written to standard output."""
    _print_fit_summary(
        target="Q00",
        fit={
            "parameters": {
                "theta": 0.1,
                "t1_rho": 10_000.0,
                "t2_rho": 8_000.0,
                "p_inf": -0.2,
            },
            "errors": {
                "theta": 0.01,
                "t1_rho": 500.0,
                "t2_rho": 400.0,
                "p_inf": 0.03,
            },
        },
        rabi_frequency=0.02,
        shift=0.002,
        shift_error=0.0002,
        axis_frequency=0.0201,
        axis_frequency_error=0.0001,
    )

    output = capsys.readouterr().out
    assert "Spin-lock relaxation fit: Q00" in output
    assert "θ = 0.1 ± 0.01 rad" in output
    assert "T1ρ = 10 ± 0.5 μs" in output
    assert "shift = 2 ± 0.2 MHz" in output
    assert "axis rotation frequency = 20.1 ± 0.1 MHz" in output


def test_relaxation_figure_uses_a_wide_three_panel_layout() -> None:
    """The three-panel figure should give each subplot a balanced aspect ratio."""
    durations = np.asarray([100.0, 200.0, 300.0])
    points = {
        name: np.ones(durations.size)
        for name in ("a", "a_error", "b", "b_error", "p", "p_error")
    }
    fit = {
        "parameters": {
            "a0": 1.0,
            "b0": 1.0,
            "p0": 0.0,
            "p_inf": 0.0,
            "t1_rho": 1_000.0,
            "t2_rho": 1_000.0,
        }
    }

    figure = _make_figure(
        target="Q00",
        durations=durations,
        points=points,
        fit=fit,
    )

    assert figure.layout.width == 1200
    assert figure.layout.height == 360


def test_spectroscopy_analysis_recovers_each_frequency_independently() -> None:
    """Each Rabi frequency should produce its own fit and derived frequency."""
    durations = np.geomspace(100.0, 30_000.0, 15)
    rabi_frequencies = np.asarray([0.01, 0.02])
    theta = np.asarray([0.1, -0.15])
    expectations = np.stack(
        [
            _predict_expectations(
                durations,
                theta=float(angle),
                t1_rho=10_000.0 + index * 2_000.0,
                t2_rho=7_000.0 + index * 1_000.0,
                p_inf=-0.2,
                a0=0.95,
                b0=0.9,
                p0=0.02,
                phases=np.linspace(0.2, 2.0, durations.size),
            )
            for index, angle in enumerate(theta)
        ]
    )
    exp: Any = SimpleNamespace(
        pulse=SimpleNamespace(
            rabi_params={
                "Q00": SimpleNamespace(noise=0.005, amplitude=0.5),
            }
        )
    )

    result, figures = _analyze_spectroscopy_target(
        exp,
        target="Q00",
        durations=durations,
        raw_data=np.zeros(expectations.shape, dtype=complex),
        expectations=expectations,
        requested_frequencies=rabi_frequencies,
        drive_amplitudes=np.asarray([0.1, 0.2]),
        rabi_frequencies=rabi_frequencies,
        ramp_duration=16.0,
    )

    np.testing.assert_allclose(result["t1_rho"], [10_000.0, 12_000.0], rtol=2e-3)
    np.testing.assert_allclose(result["t2_rho"], [7_000.0, 8_000.0], rtol=2e-3)
    np.testing.assert_allclose(
        result["shift"],
        rabi_frequencies * np.tan(theta),
        rtol=2e-3,
    )
    np.testing.assert_allclose(
        result["axis_rotation_frequency"],
        np.hypot(rabi_frequencies, result["shift"]),
        rtol=1e-7,
    )
    assert result["point_estimates"]["a"].shape == (2, durations.size)
    assert np.all(result["fit_success"])
    assert set(figures) == {
        "Q00_parameter_heatmaps",
        "Q00_relaxation_spectroscopy",
    }
    assert len(figures["Q00_parameter_heatmaps"].data) == 3
    assert len(figures["Q00_relaxation_spectroscopy"].data) == 3
    heatmaps = figures["Q00_parameter_heatmaps"]
    assert heatmaps.layout.yaxis.type == "log"
    assert heatmaps.layout.yaxis2.type == "log"
    assert heatmaps.layout.yaxis3.type == "log"
    summary = figures["Q00_relaxation_spectroscopy"]
    assert summary.layout.yaxis.rangemode == "tozero"
    assert summary.layout.yaxis.minallowed == 0
    assert summary.layout.yaxis2.rangemode == "tozero"
    assert summary.layout.yaxis2.minallowed == 0


def test_spectroscopy_analysis_continues_after_one_fit_failure() -> None:
    """A failed frequency fit should leave NaNs without hiding later results."""
    durations = np.geomspace(100.0, 20_000.0, 12)
    valid = _predict_expectations(
        durations,
        theta=0.1,
        t1_rho=8_000.0,
        t2_rho=5_000.0,
        p_inf=0.1,
        a0=0.95,
        b0=0.9,
        p0=0.0,
        phases=np.linspace(0.1, 1.5, durations.size),
    )
    invalid = valid.copy()
    invalid[0, 0] = np.nan
    expectations = np.stack((invalid, valid))
    exp: Any = SimpleNamespace(
        pulse=SimpleNamespace(
            rabi_params={
                "Q00": SimpleNamespace(noise=0.005, amplitude=0.5),
            }
        )
    )

    result, _ = _analyze_spectroscopy_target(
        exp,
        target="Q00",
        durations=durations,
        raw_data=np.zeros(expectations.shape, dtype=complex),
        expectations=expectations,
        requested_frequencies=np.asarray([0.01, 0.02]),
        drive_amplitudes=np.asarray([0.1, 0.2]),
        rabi_frequencies=np.asarray([0.01, 0.02]),
        ramp_duration=16.0,
    )

    assert result["fit_success"].tolist() == [False, True]
    assert np.isnan(result["t1_rho"][0])
    assert np.isfinite(result["t1_rho"][1])
    assert "finite" in result["fit_messages"][0]


def test_spectroscopy_analysis_handles_all_fit_failures() -> None:
    """All-failed sweeps should still return inspectable data and figures."""
    durations = np.geomspace(100.0, 20_000.0, 12)
    expectations = np.full((2, durations.size, len(MEASUREMENT_SETTINGS)), np.nan)
    exp: Any = SimpleNamespace(
        pulse=SimpleNamespace(
            rabi_params={
                "Q00": SimpleNamespace(noise=0.005, amplitude=0.5),
            }
        )
    )

    result, figures = _analyze_spectroscopy_target(
        exp,
        target="Q00",
        durations=durations,
        raw_data=np.zeros(expectations.shape, dtype=complex),
        expectations=expectations,
        requested_frequencies=np.asarray([0.01, 0.02]),
        drive_amplitudes=np.asarray([0.1, 0.2]),
        rabi_frequencies=np.asarray([0.01, 0.02]),
        ramp_duration=16.0,
    )

    assert result["fit_success"].tolist() == [False, False]
    assert np.all(np.isnan(result["axis_rotation_frequency"]))
    assert len(figures["Q00_parameter_heatmaps"].data) == 3
    assert len(figures["Q00_relaxation_spectroscopy"].data) == 3
