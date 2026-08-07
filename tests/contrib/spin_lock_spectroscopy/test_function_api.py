"""Tests for functional APIs in `qubex.contrib.experiment.spin_lock_spectroscopy`."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qubex.analysis import FitStatus
from qubex.contrib import spin_lock_sequence, spin_lock_spectroscopy
from qubex.contrib.experiment import (
    spin_lock_sequence as experiment_spin_lock_sequence,
    spin_lock_spectroscopy as experiment_spin_lock_spectroscopy,
)
from qubex.contrib.experiment.spin_lock_spectroscopy import (
    _analyze_spin_lock_target,
    _chevron_detuning_half_width_limit,
    _estimate_spin_lock_parameters_with_chevron,
    _final_projection_phase,
    _final_projection_phase_maps,
    _fit_data_without_figure,
    _frequency_axis_from_calibration,
    _frequency_sort_indices,
    _make_spin_lock_heatmap_figure,
    _make_spin_lock_relaxation_figure,
    _resolve_bool,
    _resolve_drive_detuning,
    _resolve_duration_range,
    _resolve_frequency_range,
    _resolve_nonnegative_finite_float,
    _resolve_positive_integer,
    _resolve_spin_lock_calibration,
    _resolve_spin_lock_parameters_from_calibration,
    _scaled_chevron_ranges,
    _validate_hpi_pulses,
)

spin_lock_module = importlib.import_module(
    "qubex.contrib.experiment.spin_lock_spectroscopy"
)


def test_all_spin_lock_spectroscopy_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then spin-lock helpers are available."""
    assert callable(spin_lock_sequence)
    assert callable(spin_lock_spectroscopy)


def test_all_spin_lock_spectroscopy_functions_are_exported_from_experiment() -> None:
    """Given experiment package, when imported, then spin-lock helpers are available."""
    assert experiment_spin_lock_sequence is spin_lock_sequence
    assert experiment_spin_lock_spectroscopy is spin_lock_spectroscopy


def test_spin_lock_spectroscopy_checks_hpi_pulse_before_measurement() -> None:
    """Given unavailable half-pi pulse, then spectroscopy fails before calibration."""

    class _Pulse:
        def get_hpi_pulse(self, target: str) -> None:
            raise ValueError(f"hpi pulse for {target} is unavailable.")

        def validate_rabi_params(self, _targets: list[str]) -> None:
            raise AssertionError("Rabi validation should not run after hpi failure.")

    exp: Any = SimpleNamespace(pulse=_Pulse())

    with pytest.raises(ValueError, match="hpi pulse"):
        spin_lock_spectroscopy(
            exp,
            targets=["Q00"],
            spin_lock_calibration={
                "Q00": {
                    "drive_amplitudes": np.array([0.0]),
                    "qubit_frequencies": np.array([5.0]),
                    "rabi_frequencies": np.array([0.0]),
                }
            },
        )


def test_spin_lock_spectroscopy_requires_rabi_params_with_calibration() -> None:
    """Given reusable calibration, then Rabi params are still checked for normalization."""

    class _Pulse:
        @staticmethod
        def get_hpi_pulse(_target: str) -> object:
            return object()

        @staticmethod
        def validate_rabi_params(_targets: list[str]) -> None:
            raise ValueError("Rabi parameters are not stored.")

    exp: Any = SimpleNamespace(pulse=_Pulse())

    with pytest.raises(ValueError, match="Rabi parameters"):
        spin_lock_spectroscopy(
            exp,
            targets=["Q00"],
            spin_lock_calibration={
                "Q00": {
                    "drive_amplitudes": np.array([0.0]),
                    "qubit_frequencies": np.array([5.0]),
                    "rabi_frequencies": np.array([0.0]),
                }
            },
        )


def test_validate_hpi_pulses_checks_each_target() -> None:
    """Given multiple targets, then half-pi pulse availability is checked for all."""

    class _Pulse:
        def __init__(self) -> None:
            self.targets: list[str] = []

        def get_hpi_pulse(self, target: str) -> object:
            self.targets.append(target)
            return object()

    pulse = _Pulse()
    exp: Any = SimpleNamespace(pulse=pulse)

    _validate_hpi_pulses(exp, ["Q00", "Q01"])

    assert pulse.targets == ["Q00", "Q01"]


def test_default_spin_lock_frequency_range_uses_tangent_spacing_to_120_mhz() -> None:
    """Given defaults, when inspected, then the Rabi-frequency range uses tangent spacing."""
    frequency_range = spin_lock_module.DEFAULT_SPIN_LOCK_FREQUENCY_RANGE
    expected = 0.12 * np.tan(np.pi / 3 * np.linspace(0, 1, 15)) / np.sqrt(3)

    np.testing.assert_allclose(frequency_range, expected)
    assert frequency_range[0] == pytest.approx(0.0)
    assert frequency_range[-1] == pytest.approx(0.12)


def test_resolve_frequency_range_accepts_zero_and_rejects_negative() -> None:
    """Given frequency inputs, then zero is allowed but negative values are rejected."""
    np.testing.assert_allclose(_resolve_frequency_range([0.0, 0.01]), [0.0, 0.01])

    with pytest.raises(ValueError, match="nonnegative"):
        _resolve_frequency_range([-0.01, 0.01])


def test_scaled_chevron_ranges_match_reference_at_12p5_mhz() -> None:
    """Given 12.5 MHz expected Rabi rate, then chevron ranges use reference spans."""

    class _Util:
        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            array = np.asarray(values, dtype=np.float64)
            return np.round(array / sampling_period) * sampling_period

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=_Util(),
            measurement=SimpleNamespace(sampling_period=2.0),
        )
    )

    detuning_range, time_range, omega_rabi_range = _scaled_chevron_ranges(
        exp,
        expected_rabi_frequency=0.0125,
    )

    np.testing.assert_allclose(detuning_range, np.linspace(-0.05, 0.05, 41))
    np.testing.assert_allclose(time_range, np.linspace(0, 250, 26))
    assert omega_rabi_range[-1] == pytest.approx(0.025)


def test_scaled_chevron_ranges_shorten_time_for_larger_rabi_rate() -> None:
    """Given larger drive, then chevron time span is shortened."""

    class _Util:
        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            return np.asarray(values, dtype=np.float64)

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=_Util(),
            measurement=SimpleNamespace(sampling_period=1.0),
        )
    )

    detuning_range, time_range, omega_rabi_range = _scaled_chevron_ranges(
        exp,
        expected_rabi_frequency=0.025,
    )

    assert detuning_range[0] == pytest.approx(-0.1)
    assert detuning_range[-1] == pytest.approx(0.1)
    assert time_range[-1] == pytest.approx(125.0)
    assert omega_rabi_range[-1] == pytest.approx(0.05)


def test_scaled_chevron_ranges_caps_detuning_span_by_sampling_nyquist() -> None:
    """Given strong drive, then chevron detuning span is capped by sampling Nyquist."""

    class _Util:
        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            return np.asarray(values, dtype=np.float64)

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=_Util(),
            measurement=SimpleNamespace(sampling_period=1.0),
        )
    )

    detuning_range, time_range, omega_rabi_range = _scaled_chevron_ranges(
        exp,
        expected_rabi_frequency=0.2,
    )

    assert detuning_range[0] == pytest.approx(-0.4)
    assert detuning_range[-1] == pytest.approx(0.4)
    assert time_range.size == spin_lock_module.CHEVRON_TIME_POINTS
    assert time_range[-1] == pytest.approx(25.0)
    assert omega_rabi_range[-1] == pytest.approx(0.4)


def test_scaled_chevron_ranges_rounds_time_to_sampling_grid() -> None:
    """Given a coarse sampling grid, then only chevron time points are rounded."""

    class _Util:
        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            array = np.asarray(values, dtype=np.float64)
            return np.round(array / sampling_period) * sampling_period

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=_Util(),
            measurement=SimpleNamespace(sampling_period=7.0),
        )
    )

    detuning_range, time_range, _ = _scaled_chevron_ranges(
        exp,
        expected_rabi_frequency=0.0125,
    )

    np.testing.assert_allclose(detuning_range, np.linspace(-0.05, 0.05, 41))
    np.testing.assert_allclose(time_range / 7.0, np.round(time_range / 7.0))


def test_scaled_chevron_ranges_keeps_time_point_count_on_coarse_grid() -> None:
    """Given a coarse sampling grid, then chevron time point count is preserved."""

    class _Util:
        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            array = np.asarray(values, dtype=np.float64)
            return np.round(array / sampling_period) * sampling_period

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=_Util(),
            measurement=SimpleNamespace(sampling_period=1000.0),
        )
    )

    _, time_range, _ = _scaled_chevron_ranges(
        exp,
        expected_rabi_frequency=0.2,
    )

    assert time_range.size == spin_lock_module.CHEVRON_TIME_POINTS
    np.testing.assert_allclose(np.diff(time_range), 1000.0)


def test_chevron_detuning_half_width_limit_uses_sampling_period_nyquist() -> None:
    """Given a sampling grid, then the detuning cap is 80% of Nyquist."""
    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(measurement=SimpleNamespace(sampling_period=2.0))
    )

    assert _chevron_detuning_half_width_limit(exp) == pytest.approx(0.2)


def test_resolve_spin_lock_parameters_reuses_supplied_calibration() -> None:
    """Given previous calibration data, then measured axes are recovered from it."""
    spin_lock_calibration = {
        "Q00": {
            "requested_rabi_frequencies": np.array([0.01, 0.02]),
            "drive_amplitudes": np.array([0.1, 0.2]),
            "qubit_frequencies": np.array([5.001, 5.004]),
            "rabi_frequencies": np.array([0.011, 0.021]),
        }
    }

    drive_amplitude_map, qubit_frequency_map, rabi_frequency_map, resolved = (
        _resolve_spin_lock_parameters_from_calibration(
            targets=["Q00"],
            requested_rabi_frequency_range=np.array([0.01, 0.02]),
            spin_lock_calibration=spin_lock_calibration,
            base_frequencies={"Q00": 5.0},
        )
    )

    np.testing.assert_allclose(drive_amplitude_map["Q00"], [0.1, 0.2])
    np.testing.assert_allclose(qubit_frequency_map["Q00"], [5.001, 5.004])
    np.testing.assert_allclose(rabi_frequency_map["Q00"], [0.011, 0.021])
    np.testing.assert_allclose(resolved["Q00"]["ac_stark_shifts"], [0.001, 0.004])


def test_resolve_spin_lock_parameters_accepts_ac_stark_shifts() -> None:
    """Given AC Stark shifts, then qubit frequencies are reconstructed."""
    spin_lock_calibration = {
        "Q00": {
            "drive_amplitudes": np.array([0.1, 0.2]),
            "ac_stark_shifts": np.array([0.001, 0.004]),
            "rabi_frequencies": np.array([0.011, 0.021]),
        }
    }

    _, qubit_frequency_map, rabi_frequency_map, _ = (
        _resolve_spin_lock_parameters_from_calibration(
            targets=["Q00"],
            requested_rabi_frequency_range=np.array([0.01, 0.02]),
            spin_lock_calibration=spin_lock_calibration,
            base_frequencies={"Q00": 5.0},
        )
    )

    np.testing.assert_allclose(qubit_frequency_map["Q00"], [5.001, 5.004])
    np.testing.assert_allclose(rabi_frequency_map["Q00"], [0.011, 0.021])


def test_frequency_axis_from_calibration_prefers_requested_axis() -> None:
    """Given calibration data, then the stored requested axis is reused if present."""
    spin_lock_calibration = {
        "Q00": {
            "requested_rabi_frequencies": np.array([0.01, 0.02]),
            "rabi_frequencies": np.array([0.011, 0.021]),
        }
    }

    frequency_axis = _frequency_axis_from_calibration(
        targets=["Q00"],
        spin_lock_calibration=spin_lock_calibration,
    )

    np.testing.assert_allclose(frequency_axis, [0.01, 0.02])


def test_frequency_axis_from_calibration_falls_back_to_rabi_axis() -> None:
    """Given manual calibration data, then the Rabi axis can define the sweep."""
    spin_lock_calibration = {
        "Q00": {
            "rabi_frequencies": np.array([0.011, 0.021]),
        }
    }

    frequency_axis = _frequency_axis_from_calibration(
        targets=["Q00"],
        spin_lock_calibration=spin_lock_calibration,
    )

    np.testing.assert_allclose(frequency_axis, [0.011, 0.021])


def test_resolve_spin_lock_parameters_reuses_zero_rabi_result() -> None:
    """Given zero-frequency calibration, then a zero measured Rabi rate is accepted."""
    spin_lock_calibration = {
        "Q00": {
            "requested_rabi_frequencies": np.array([0.0]),
            "drive_amplitudes": np.array([0.0]),
            "qubit_frequencies": np.array([5.0]),
            "rabi_frequencies": np.array([0.0]),
        }
    }

    drive_amplitude_map, qubit_frequency_map, rabi_frequency_map, _ = (
        _resolve_spin_lock_parameters_from_calibration(
            targets=["Q00"],
            requested_rabi_frequency_range=np.array([0.0]),
            spin_lock_calibration=spin_lock_calibration,
            base_frequencies={"Q00": 5.0},
        )
    )

    np.testing.assert_allclose(drive_amplitude_map["Q00"], [0.0])
    np.testing.assert_allclose(qubit_frequency_map["Q00"], [5.0])
    np.testing.assert_allclose(rabi_frequency_map["Q00"], [0.0])


def test_estimate_spin_lock_parameters_skips_zero_rabi_chevron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a zero-frequency point, then no chevron measurement is taken for it."""

    class _Util:
        @staticmethod
        def no_output() -> Any:
            return nullcontext()

        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            array = np.asarray(values, dtype=np.float64)
            return np.round(array / sampling_period) * sampling_period

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=_Util(),
            measurement=SimpleNamespace(sampling_period=2.0),
        )
    )
    calls: list[dict[str, Any]] = []
    progress_iterables: list[list[int]] = []

    def _estimate_chevron(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            data={
                "results": {
                    "Q00": {
                        "omega_q": 5.001,
                        "omega_rabi": 0.012,
                        "peak_background_rms_ratio": 10.0,
                    }
                }
            },
            figures=None,
        )

    def _fake_tqdm(iterable: Any, **_kwargs: Any) -> list[int]:
        values = list(iterable)
        progress_iterables.append(values)
        return values

    monkeypatch.setattr(
        spin_lock_module,
        "estimate_qubit_frequency_from_chevron",
        _estimate_chevron,
    )
    monkeypatch.setattr(spin_lock_module, "tqdm", _fake_tqdm)

    qubit_frequency_map, rabi_frequency_map, spin_lock_calibration, _ = (
        _estimate_spin_lock_parameters_with_chevron(
            exp,
            targets=["Q00"],
            requested_rabi_frequency_range=np.array([0.0, 0.0125]),
            amplitude_map={"Q00": np.array([0.0, 0.1])},
            base_frequencies={"Q00": 5.0},
            chevron_n_shots=10,
            shot_interval=0.0,
            return_chevron_figures=False,
        )
    )

    assert len(calls) == 1
    assert progress_iterables == [[1]]
    np.testing.assert_allclose(qubit_frequency_map["Q00"], [5.0, 5.001])
    np.testing.assert_allclose(rabi_frequency_map["Q00"], [0.0, 0.012])
    np.testing.assert_allclose(
        spin_lock_calibration["Q00"]["drive_amplitudes"],
        [0.0, 0.1],
    )
    np.testing.assert_allclose(
        spin_lock_calibration["Q00"]["rabi_frequencies"],
        [0.0, 0.012],
    )
    np.testing.assert_array_equal(
        spin_lock_calibration["Q00"]["chevron_measured"],
        [False, True],
    )


def test_resolve_spin_lock_parameters_rejects_mismatched_calibration() -> None:
    """Given stale calibration metadata, then reuse is rejected before measurement."""
    spin_lock_calibration = {
        "Q00": {
            "requested_rabi_frequencies": np.array([0.03]),
            "drive_amplitudes": np.array([0.1]),
            "qubit_frequencies": np.array([5.001]),
            "rabi_frequencies": np.array([0.011]),
        }
    }

    with pytest.raises(ValueError, match="spin_lock_frequency_range"):
        _resolve_spin_lock_parameters_from_calibration(
            targets=["Q00"],
            requested_rabi_frequency_range=np.array([0.01]),
            spin_lock_calibration=spin_lock_calibration,
            base_frequencies={"Q00": 5.0},
        )


def test_resolve_spin_lock_parameters_requires_frequency_calibration() -> None:
    """Given incomplete previous calibration, then reuse is rejected."""
    spin_lock_calibration = {
        "Q00": {
            "drive_amplitudes": np.array([0.1]),
            "rabi_frequencies": np.array([0.011]),
        }
    }

    with pytest.raises(ValueError, match="ac_stark_shifts"):
        _resolve_spin_lock_parameters_from_calibration(
            targets=["Q00"],
            requested_rabi_frequency_range=np.array([0.01]),
            spin_lock_calibration=spin_lock_calibration,
            base_frequencies={"Q00": 5.0},
        )


def test_resolve_spin_lock_parameters_rejects_non_bool_chevron_measured() -> None:
    """Given non-bool chevron flags, then manual calibration is rejected."""
    spin_lock_calibration = {
        "Q00": {
            "requested_rabi_frequencies": np.array([0.01]),
            "drive_amplitudes": np.array([0.1]),
            "qubit_frequencies": np.array([5.001]),
            "rabi_frequencies": np.array([0.011]),
            "chevron_measured": np.array([1]),
        }
    }

    with pytest.raises(TypeError, match="chevron_measured"):
        _resolve_spin_lock_parameters_from_calibration(
            targets=["Q00"],
            requested_rabi_frequency_range=np.array([0.01]),
            spin_lock_calibration=spin_lock_calibration,
            base_frequencies={"Q00": 5.0},
        )


def test_resolve_spin_lock_calibration_reuses_provided_dataset_without_measuring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given previous calibration data, then calibration skips new chevron sweeps."""

    def _unexpected_chevron_measurement(**_kwargs: Any) -> None:
        raise AssertionError("Chevron measurement should not be called.")

    monkeypatch.setattr(
        spin_lock_module,
        "_estimate_spin_lock_parameters_with_chevron",
        _unexpected_chevron_measurement,
    )

    calibration = _resolve_spin_lock_calibration(
        object(),  # type: ignore[arg-type]
        targets=["Q00"],
        requested_rabi_frequency_range=np.array([0.01]),
        base_frequencies={"Q00": 5.0},
        spin_lock_calibration={
            "Q00": {
                "requested_rabi_frequencies": np.array([0.01]),
                "drive_amplitudes": np.array([0.1]),
                "qubit_frequencies": np.array([5.001]),
                "rabi_frequencies": np.array([0.011]),
            }
        },
        estimate_with_chevron=True,
        chevron_n_shots=10,
        n_shots=100,
        shot_interval=0.0,
        return_chevron_figures=True,
    )

    assert calibration.calibration_source == "provided"
    assert calibration.chevron_n_shots is None
    assert calibration.chevron_figures == {}
    np.testing.assert_allclose(calibration.drive_amplitude_map["Q00"], [0.1])
    np.testing.assert_allclose(calibration.qubit_frequency_map["Q00"], [5.001])
    np.testing.assert_allclose(calibration.rabi_frequency_map["Q00"], [0.011])


def test_resolve_spin_lock_calibration_marks_zero_only_axis_as_not_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given only zero Rabi points, then chevron calibration is not marked measured."""

    def _unexpected_chevron_measurement(**_kwargs: Any) -> None:
        raise AssertionError("Chevron measurement should not be called.")

    exp: Any = SimpleNamespace(
        pulse=SimpleNamespace(
            calc_control_amplitude=lambda _target, _frequency: 0.0,
        )
    )
    monkeypatch.setattr(
        spin_lock_module,
        "_estimate_spin_lock_parameters_with_chevron",
        _unexpected_chevron_measurement,
    )

    calibration = _resolve_spin_lock_calibration(
        exp,
        targets=["Q00"],
        requested_rabi_frequency_range=np.array([0.0]),
        base_frequencies={"Q00": 5.0},
        spin_lock_calibration=None,
        estimate_with_chevron=True,
        chevron_n_shots=0,
        n_shots=100,
        shot_interval=0.0,
        return_chevron_figures=True,
    )

    assert calibration.calibration_source == "none"
    assert calibration.chevron_n_shots is None
    assert calibration.chevron_figures == {}
    np.testing.assert_allclose(calibration.drive_amplitude_map["Q00"], [0.0])
    np.testing.assert_allclose(calibration.qubit_frequency_map["Q00"], [5.0])
    np.testing.assert_allclose(calibration.rabi_frequency_map["Q00"], [0.0])
    np.testing.assert_array_equal(
        calibration.calibration_dataset["Q00"]["chevron_measured"],
        [False],
    )


def test_frequency_sort_indices_sorts_nonmonotonic_axis() -> None:
    """Given a nonmonotonic measured Rabi axis, then sort indices reorder it."""
    frequency_range = np.array([0.02, 0.01, 0.03])

    indices = _frequency_sort_indices(frequency_range)

    np.testing.assert_allclose(frequency_range[indices], [0.01, 0.02, 0.03])


def test_final_projection_phase_tracks_detuned_drive_frame() -> None:
    """Given detuned spin-lock drive, then final projection tracks accumulated phase."""
    final_phase = _final_projection_phase(
        final_phase=np.pi / 2,
        drive_detuning=0.001,
        duration=100.0,
    )

    assert final_phase == pytest.approx(np.pi / 2 - 2.0 * np.pi * 0.001 * 100.0)


def test_final_projection_phase_is_unchanged_without_detuning() -> None:
    """Given no spin-lock detuning, then final projection phase is unchanged."""
    final_phase = _final_projection_phase(
        final_phase=np.pi / 2,
        drive_detuning=None,
        duration=100.0,
    )

    assert final_phase == pytest.approx(np.pi / 2)


def test_final_projection_phase_maps_return_frequency_duration_grids() -> None:
    """Given offsets and durations, then final projection metadata matches sweep grid."""
    corrections, phases = _final_projection_phase_maps(
        targets=["Q00"],
        frequency_offset_map={"Q00": np.array([0.001, -0.002])},
        durations=np.array([100.0, 200.0, 300.0]),
        final_phase=np.pi / 2,
    )

    expected_corrections = (
        -2.0
        * np.pi
        * np.array(
            [
                [0.001 * 100.0, 0.001 * 200.0, 0.001 * 300.0],
                [-0.002 * 100.0, -0.002 * 200.0, -0.002 * 300.0],
            ]
        )
    )
    np.testing.assert_allclose(corrections["Q00"], expected_corrections)
    np.testing.assert_allclose(phases["Q00"], np.pi / 2 + expected_corrections)


def test_spin_lock_relaxation_figure_yaxis_starts_at_zero() -> None:
    """Given relaxation data, then the summary figure lower y-axis bound is zero."""
    fig = _make_spin_lock_relaxation_figure(
        target="Q00",
        spin_lock_rabi_frequency_range=np.array([0.01, 0.02]),
        relaxation_times=np.array([1000.0, 2000.0]),
        relaxation_time_errors=np.array([500.0, 800.0]),
    )

    assert fig.layout.yaxis.range[0] == 0
    assert fig.layout.yaxis.title.text == "Relaxation time (μs)"
    assert fig.layout.xaxis.type == "linear"
    assert fig.layout.xaxis.range[0] == 0


def test_spin_lock_heatmap_figure_uses_linear_frequency_axis() -> None:
    """Given heatmap data, then the spin-lock frequency axis is linear."""
    fig = _make_spin_lock_heatmap_figure(
        target="Q00",
        spin_lock_rabi_frequency_range=np.array([0.01, 0.02]),
        duration_range=np.array([100.0, 200.0, 300.0]),
        population=np.ones((3, 2)),
    )

    assert fig.layout.xaxis.type == "linear"
    assert fig.layout.xaxis.range[0] == 0
    assert fig.layout.yaxis.type == "log"
    heatmap_trace: Any = fig.data[0]
    assert heatmap_trace.zmin == 0
    assert heatmap_trace.zmax == 1


def test_analyze_spin_lock_target_returns_expected_shapes_and_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given sweep data, then per-target analysis returns arrays and figures."""

    class _FitResult(dict[str, float]):
        status = FitStatus.SUCCESS
        message = "ok"
        figure = None

        def __init__(self, **kwargs: float) -> None:
            super().__init__(**kwargs)
            self.data = dict(self)

    fit_kwargs: list[dict[str, Any]] = []

    def _fit_exp_decay(**kwargs: Any) -> _FitResult:
        fit_kwargs.append(kwargs)
        return _FitResult(tau=1000.0, tau_err=100.0, r2=0.99)

    monkeypatch.setattr(spin_lock_module.fitting, "fit_exp_decay", _fit_exp_decay)
    sweep_data = SimpleNamespace(
        data=np.arange(6, dtype=np.float64),
        normalized=np.linspace(-1.0, 1.0, 6),
    )

    analysis = _analyze_spin_lock_target(
        target="Q00",
        sweep_data=sweep_data,
        durations=np.array([100.0, 200.0, 300.0]),
        requested_rabi_frequency_range=np.array([0.02, 0.01]),
        rabi_frequency_range=np.array([0.02, 0.01]),
        effective_frequency_range=np.array([0.02, 0.01]),
        drive_frequencies=np.array([5.0, 5.0]),
        qubit_frequencies=np.array([5.0, 5.0]),
    )

    assert analysis.raw_data.shape == (2, 3)
    assert analysis.normalized_signal.shape == (3, 2)
    assert analysis.population.shape == (3, 2)
    np.testing.assert_allclose(analysis.relaxation_times, [1000.0, 1000.0])
    assert {kwargs["xlabel"] for kwargs in fit_kwargs} == {"Duration (μs)"}
    assert "Q00_heatmap" in analysis.figures
    assert "Q00_relaxation" in analysis.figures
    heatmap_figure: Any = analysis.figures["Q00_heatmap"]
    heatmap_trace: Any = heatmap_figure.data[0]
    np.testing.assert_allclose(heatmap_trace.x, [10.0, 20.0])


def test_analyze_spin_lock_target_fits_zero_rabi_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a zero Rabi point, then decay fitting is still applied."""

    class _FitResult(dict[str, float]):
        status = FitStatus.SUCCESS
        message = "ok"
        figure = None

        def __init__(self, **kwargs: float) -> None:
            super().__init__(**kwargs)
            self.data = dict(self)

    fit_kwargs: list[dict[str, Any]] = []

    def _fit_exp_decay(**kwargs: Any) -> _FitResult:
        fit_kwargs.append(kwargs)
        return _FitResult(tau=1000.0, tau_err=100.0, r2=0.99)

    monkeypatch.setattr(spin_lock_module.fitting, "fit_exp_decay", _fit_exp_decay)
    sweep_data = SimpleNamespace(
        data=np.arange(6, dtype=np.float64),
        normalized=np.linspace(-1.0, 1.0, 6),
    )

    analysis = _analyze_spin_lock_target(
        target="Q00",
        sweep_data=sweep_data,
        durations=np.array([100.0, 200.0, 300.0]),
        requested_rabi_frequency_range=np.array([0.0, 0.01]),
        rabi_frequency_range=np.array([0.0, 0.01]),
        effective_frequency_range=np.array([0.0, 0.01]),
        drive_frequencies=np.array([5.0, 5.0]),
        qubit_frequencies=np.array([5.0, 5.0]),
    )

    assert [kwargs["target"] for kwargs in fit_kwargs] == [
        "Q00_0_MHz",
        "Q00_10_MHz",
    ]
    np.testing.assert_allclose(analysis.relaxation_times, [1000.0, 1000.0])
    assert [fit_result["status"] for fit_result in analysis.fit_results] == [
        "success",
        "success",
    ]
    assert "rabi_frequency" in analysis.fit_results[0]
    assert "frequency" not in analysis.fit_results[0]


def test_resolve_duration_range_removes_duplicate_discretized_points() -> None:
    """Given close durations, then discretization removes duplicate time points."""

    class _Util:
        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            array = np.asarray(values, dtype=np.float64)
            return np.round(array / sampling_period) * sampling_period

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=_Util(),
            measurement=SimpleNamespace(sampling_period=10.0),
        )
    )

    durations = _resolve_duration_range(exp, [9.0, 11.0, 21.0, 31.0])

    np.testing.assert_allclose(durations, [10.0, 20.0, 30.0])


def test_resolve_duration_range_rejects_too_few_fit_points() -> None:
    """Given too few unique durations, then decay fitting is rejected early."""

    class _Util:
        @staticmethod
        def discretize_time_range(
            values: Any,
            *,
            sampling_period: float,
        ) -> np.ndarray:
            array = np.asarray(values, dtype=np.float64)
            return np.round(array / sampling_period) * sampling_period

    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=_Util(),
            measurement=SimpleNamespace(sampling_period=10.0),
        )
    )

    with pytest.raises(ValueError, match="duration_range"):
        _resolve_duration_range(exp, [9.0, 11.0, 21.0])


def test_resolve_duration_range_rejects_invalid_sampling_period() -> None:
    """Given an invalid sampling period, then duration discretization is rejected."""
    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            util=SimpleNamespace(),
            measurement=SimpleNamespace(sampling_period=0.0),
        )
    )

    with pytest.raises(ValueError, match="sampling_period"):
        _resolve_duration_range(exp, [10.0, 20.0, 30.0])


def test_fit_data_without_figure_tolerates_missing_data_attribute() -> None:
    """Given a nonstandard fit result, then no figure payload is returned."""
    assert _fit_data_without_figure(SimpleNamespace()) == {}


def test_fit_data_without_figure_drops_embedded_figure() -> None:
    """Given fit result data, then embedded figure payload is omitted."""
    fit_result = SimpleNamespace(data={"tau": 100.0, "fig": object()})

    assert _fit_data_without_figure(fit_result) == {"tau": 100.0}


def test_resolve_drive_detuning_rejects_nonfinite_values() -> None:
    """Given invalid detuning, then validation rejects it before measurement."""
    with pytest.raises(ValueError, match="drive_detuning"):
        _resolve_drive_detuning(np.nan)


def test_resolve_positive_integer_rejects_invalid_values() -> None:
    """Given invalid shot counts, then validation rejects them early."""
    with pytest.raises(ValueError, match="n_shots"):
        _resolve_positive_integer(0, name="n_shots", default=1024)

    with pytest.raises(TypeError, match="n_shots"):
        _resolve_positive_integer(np.nan, name="n_shots", default=1024)  # type: ignore[arg-type]


def test_resolve_nonnegative_finite_float_rejects_invalid_values() -> None:
    """Given invalid shot intervals, then validation rejects them early."""
    with pytest.raises(ValueError, match="shot_interval"):
        _resolve_nonnegative_finite_float(
            np.nan,
            name="shot_interval",
            default=0.0,
        )

    with pytest.raises(ValueError, match="shot_interval"):
        _resolve_nonnegative_finite_float(
            -1.0,
            name="shot_interval",
            default=0.0,
        )


def test_resolve_bool_rejects_non_bool_values() -> None:
    """Given non-bool value, then boolean option validation rejects it."""
    with pytest.raises(TypeError, match="Boolean options"):
        _resolve_bool("False", default=False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration": 0.0, "drive_amplitude": 0.1}, "duration"),
        ({"duration": np.nan, "drive_amplitude": 0.1}, "duration"),
        ({"duration": 100.0, "drive_amplitude": np.nan}, "drive_amplitude"),
        ({"duration": 100.0, "drive_amplitude": 1.1}, "drive_amplitude"),
        (
            {
                "duration": 100.0,
                "drive_amplitude": 0.1,
                "drive_detuning": np.nan,
            },
            "drive_detuning",
        ),
        (
            {
                "duration": 100.0,
                "drive_amplitude": 0.1,
                "drive_phase": np.nan,
            },
            "phase",
        ),
    ],
)
def test_spin_lock_sequence_rejects_invalid_scalar_inputs(
    kwargs: dict[str, float],
    match: str,
) -> None:
    """Given invalid scalar inputs, when building a sequence, then validation fails."""
    with pytest.raises(ValueError, match=match):
        spin_lock_sequence(
            object(),  # type: ignore[arg-type]
            target="Q00",
            **kwargs,
        )
