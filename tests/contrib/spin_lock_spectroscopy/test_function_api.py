"""Tests for functional APIs in `qubex.contrib.experiment.spin_lock_spectroscopy`."""

from __future__ import annotations

import importlib
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
    _final_projection_phase,
    _final_projection_phase_maps,
    _frequency_sort_indices,
    _make_spin_lock_relaxation_figure,
    _resolve_awg_max_frequency,
    _resolve_bool,
    _resolve_drive_detuning,
    _resolve_duration_range,
    _resolve_nonnegative_finite_float,
    _resolve_positive_integer,
    _scaled_chevron_ranges,
    _validate_awg_frequency_ranges,
    _validate_chevron_awg_frequency_range,
)

spin_lock_module = importlib.import_module(
    "qubex.contrib.experiment.spin_lock_spectroscopy"
)


def _make_awg_validation_exp(
    *,
    sideband: str,
    nco_frequency: float,
    awg_max_frequency_hz: float | None = 250_000_000,
) -> Any:
    class _ExperimentSystem:
        @staticmethod
        def get_target(_target: str) -> SimpleNamespace:
            return SimpleNamespace(sideband=sideband)

        @staticmethod
        def get_nco_frequency(_target: str) -> float:
            return nco_frequency

    return SimpleNamespace(
        ctx=SimpleNamespace(
            experiment_system=_ExperimentSystem(),
            measurement=SimpleNamespace(
                constraint_profile=SimpleNamespace(
                    awg_max_frequency_hz=awg_max_frequency_hz
                )
            ),
        )
    )


def test_all_spin_lock_spectroscopy_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then spin-lock helpers are available."""
    assert callable(spin_lock_sequence)
    assert callable(spin_lock_spectroscopy)


def test_all_spin_lock_spectroscopy_functions_are_exported_from_experiment() -> None:
    """Given experiment package, when imported, then spin-lock helpers are available."""
    assert experiment_spin_lock_sequence is spin_lock_sequence
    assert experiment_spin_lock_spectroscopy is spin_lock_spectroscopy


def test_default_spin_lock_frequency_range_is_log_spaced_to_150_mhz() -> None:
    """Given defaults, when inspected, then the Rabi-frequency range reaches 150 MHz."""
    frequency_range = spin_lock_module.DEFAULT_SPIN_LOCK_FREQUENCY_RANGE

    assert frequency_range[0] == pytest.approx(0.01)
    assert frequency_range[-1] == pytest.approx(0.15)
    np.testing.assert_allclose(
        np.diff(np.log(frequency_range)),
        np.diff(np.log(frequency_range))[0],
    )


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
            measurement=SimpleNamespace(sampling_period=10.0),
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


def test_scaled_chevron_ranges_caps_detuning_span_at_200_mhz() -> None:
    """Given strong drive, then chevron detuning span is capped at +/-200 MHz."""

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

    assert detuning_range[0] == pytest.approx(-0.2)
    assert detuning_range[-1] == pytest.approx(0.2)
    assert time_range[-1] == pytest.approx(15.625)
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


def test_scaled_chevron_ranges_rejects_too_few_time_points() -> None:
    """Given a coarse sampling grid, then an unusable chevron time range is rejected."""

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

    with pytest.raises(ValueError, match="chevron time_range"):
        _scaled_chevron_ranges(
            exp,
            expected_rabi_frequency=0.2,
        )


def test_validate_chevron_awg_frequency_range_accepts_reachable_span() -> None:
    """Given reachable drive frequencies, then current NCO validation passes."""
    exp = _make_awg_validation_exp(sideband="U", nco_frequency=5.0)

    _validate_chevron_awg_frequency_range(
        exp,
        targets=["Q00"],
        base_frequencies={"Q00": 5.0},
        detuning_range=np.array([-0.2, 0.0, 0.2]),
    )


def test_validate_chevron_awg_frequency_range_rejects_unreachable_span() -> None:
    """Given unreachable drive frequencies, then current NCO validation fails."""
    exp = _make_awg_validation_exp(sideband="U", nco_frequency=5.0)

    with pytest.raises(ValueError, match="AWG modulation"):
        _validate_chevron_awg_frequency_range(
            exp,
            targets=["Q00"],
            base_frequencies={"Q00": 5.1},
            detuning_range=np.array([-0.2, 0.0, 0.2]),
        )


def test_validate_awg_frequency_range_accepts_lower_sideband_span() -> None:
    """Given lower sideband, then AWG validation uses NCO minus drive frequency."""
    exp = _make_awg_validation_exp(sideband="L", nco_frequency=5.0)

    _validate_awg_frequency_ranges(
        exp,
        targets=["Q00"],
        drive_frequency_ranges={"Q00": np.array([4.8, 4.9, 5.0])},
        description="test frequencies",
    )


def test_validate_awg_frequency_range_skips_unknown_backend_limit() -> None:
    """Given no AWG limit in profile, then validation is left to the backend."""
    exp = _make_awg_validation_exp(
        sideband="U",
        nco_frequency=5.0,
        awg_max_frequency_hz=None,
    )

    _validate_awg_frequency_ranges(
        exp,
        targets=["Q00"],
        drive_frequency_ranges={"Q00": np.array([5.0, 6.0])},
        description="test frequencies",
    )


def test_resolve_awg_max_frequency_skips_legacy_context_without_profile() -> None:
    """Given old fake measurement context, then AWG validation is skipped."""
    exp: Any = SimpleNamespace(ctx=SimpleNamespace(measurement=SimpleNamespace()))

    assert _resolve_awg_max_frequency(exp) is None


def test_resolve_awg_max_frequency_uses_quel1_like_profile() -> None:
    """Given QuEL-1-like constraints, then the contrib fallback uses QuEL-1 AWG limit."""
    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            measurement=SimpleNamespace(
                constraint_profile=SimpleNamespace(
                    sampling_period_ns=2.0,
                    word_length_samples=4,
                    block_length_samples=64,
                    require_workaround_capture=True,
                    enforce_word_alignment=True,
                    enforce_block_alignment=True,
                    enforce_capture_spacing=True,
                )
            )
        )
    )

    assert _resolve_awg_max_frequency(exp) == pytest.approx(0.25)


def test_resolve_awg_max_frequency_skips_non_quel1_strict_profile() -> None:
    """Given strict non-QuEL-1 constraints, then no QuEL-1 AWG limit is assumed."""
    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(
            measurement=SimpleNamespace(
                constraint_profile=SimpleNamespace(
                    sampling_period_ns=0.4,
                    word_length_samples=4,
                    block_length_samples=64,
                    require_workaround_capture=True,
                    enforce_word_alignment=True,
                    enforce_block_alignment=True,
                    enforce_capture_spacing=True,
                )
            )
        )
    )

    assert _resolve_awg_max_frequency(exp) is None


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

    def _fit_exp_decay(**_kwargs: Any) -> _FitResult:
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
    np.testing.assert_allclose(analysis.sort_indices, [1, 0])
    assert "Q00_heatmap" in analysis.figures
    assert "Q00_relaxation" in analysis.figures


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
