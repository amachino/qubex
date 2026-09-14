"""Tests for high-level paired IRB orchestration."""

from __future__ import annotations

import importlib
from collections.abc import Collection
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule
from scipy.optimize import curve_fit

from qubex.clifford.clifford import Clifford
from qubex.contrib import paired_interleaved_randomized_benchmarking


class _RabiParam:
    """Map an encoded IQ value to a normalized Bloch-Z value."""

    @staticmethod
    def normalize(iq: Any) -> float:
        """Decode a synthetic ground-state probability."""
        return 2.0 * float(np.real(np.mean(np.asarray(iq)))) - 1.0


class _MeasurementService:
    """Execute synthetic serial or parallel sweep schedules."""

    def __init__(
        self,
        sequence_calls: list[Any],
        *,
        decays: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.sequence_calls = sequence_calls
        self.decays = decays or {}
        self.calls: list[dict[str, Any]] = []

    async def run_sweep_measurement(
        self,
        schedule: Any,
        *,
        sweep_values: Any,
        **kwargs: Any,
    ) -> Any:
        """Return one result containing every target scheduled at each point."""
        values = np.asarray(sweep_values)
        call: dict[str, Any] = {
            "sweep_values": values.copy(),
            "point_durations": [],
            "point_markers": [],
            **kwargs,
        }
        self.calls.append(call)
        results = []
        for value in values:
            call_start = len(self.sequence_calls)
            combined_schedule = schedule(int(value))
            markers = self.sequence_calls[call_start:]
            call["point_durations"].append(combined_schedule.duration)
            call["point_markers"].append(markers)
            point_data = {}
            for marker in markers:
                reference_decay, interleaved_decay = self.decays.get(
                    marker.target,
                    (0.98, 0.95),
                )
                decay = (
                    reference_decay
                    if marker.interleaved_clifford is None
                    else interleaved_decay
                )
                is_two_qubit = marker.target.startswith("CR")
                amplitude = 0.73 if is_two_qubit else 0.48
                offset = 0.25 if is_two_qubit else 0.50
                probability = amplitude * decay**marker.n_cliffords + offset
                capture_config = SimpleNamespace(
                    shot_averaging=kwargs["shot_averaging"],
                    time_integration=kwargs["time_integration"],
                )
                point_data[marker.target] = [
                    SimpleNamespace(
                        data=np.asarray(probability, dtype=np.complex128),
                        config=capture_config,
                    )
                ]
            results.append(SimpleNamespace(data=point_data))
        return SimpleNamespace(results=results)


class _Experiment:
    """Provide the experiment surface used by high-level paired IRB tests."""

    def __init__(
        self,
        targets: tuple[str, ...] = ("Q0", "Q1"),
        *,
        cr_pairs: dict[str, tuple[str, str]] | None = None,
        decays: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.sequence_calls: list[Any] = []
        self.measurement_service = _MeasurementService(
            self.sequence_calls,
            decays=decays,
        )
        cr_pairs = cr_pairs or {}
        target_names = (*targets, *cr_pairs)
        self.pulse = SimpleNamespace(
            rabi_params={target: _RabiParam() for target in targets},
            validate_rabi_params=lambda selected: None,
        )
        self.experiment_system = SimpleNamespace(
            get_target=lambda target: SimpleNamespace(is_cr=target in cr_pairs),
            resolve_qubit_label=lambda target: target,
        )
        physical_qubits = {qubit for pair in cr_pairs.values() for qubit in pair}
        self.ctx = SimpleNamespace(
            reset_awg_and_capunits=lambda *, qubits: None,
            classifiers={qubit: object() for qubit in physical_qubits},
            calib_note=SimpleNamespace(
                cr_params={target: object() for target in cr_pairs}
            ),
            cr_pair=lambda target: cr_pairs[target],
        )
        self.interleaved_1q = Clifford.X90()
        self.interleaved_2q = Clifford.ZX90()
        self.benchmarking_service = SimpleNamespace(
            clifford={
                "X90": self.interleaved_1q,
                "ZX90": self.interleaved_2q,
            }
        )
        self._target_duration_offsets = {
            target: 2.0 * index for index, target in enumerate(target_names)
        }

    def rb_sequence(self, target: str, **kwargs: Any) -> PulseSchedule:
        """Record target-specific options and return a valid pulse schedule."""
        marker = SimpleNamespace(
            target=target,
            n_cliffords=kwargs["n"],
            seed=kwargs["seed"],
            interleaved_clifford=kwargs["interleaved_clifford"],
            interleaved_waveform=kwargs["interleaved_waveform"],
            x90=kwargs["x90"],
            zx90=kwargs["zx90"],
        )
        duration = (
            2.0 * (marker.n_cliffords + 1) + self._target_duration_offsets[target]
        )
        marker.duration = duration
        self.sequence_calls.append(marker)
        with PulseSchedule([target]) as schedule:
            schedule.add(target, Blank(duration=duration))
        return schedule


def test_serial_multi_target_combines_independent_results_and_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serial targets should retain independent seeds in one combined result."""
    exp: Any = _Experiment()
    monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            ["Q0", "Q1"],
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=4,
            sequence_seed=7,
            n_bootstrap=0,
            pairs_per_sweep=24,
            plot=True,
            save_image=False,
        )

    assert list(result.data) == ["Q0", "Q1"]
    assert result.figures is not None
    assert set(result.figures) == {"Q0", "Q1"}
    assert not np.array_equal(
        result["Q0"]["acquisition"]["seeds"],
        result["Q1"]["acquisition"]["seeds"],
    )
    assert result["Q0"]["grid_selection"]["pilot_stop_reason"] == "explicit_range"
    assert result["Q1"]["grid_selection"]["pilot_stop_reason"] == "explicit_range"
    np.testing.assert_array_equal(
        result["Q0"]["grid_selection"]["selected_main_grid"],
        [0, 1, 2, 4, 8, 16],
    )


def test_ordered_target_sequences_preserve_deterministic_seed_assignment() -> None:
    """Equivalent target lists and tuples should assign identical child seeds."""
    list_experiment: Any = _Experiment()
    tuple_experiment: Any = _Experiment()
    options = {
        "interleaved_clifford": "X90",
        "n_cliffords_range": [0, 1, 2, 4, 8, 16],
        "n_trials": 2,
        "sequence_seed": 101,
        "acquisition_seed": 103,
        "n_bootstrap": 0,
        "pairs_per_sweep": 100,
        "plot": False,
        "save_image": False,
    }

    with pytest.warns(RuntimeWarning):
        list_result = paired_interleaved_randomized_benchmarking(
            list_experiment,
            ["Q0", "Q1"],
            **options,
        )
    with pytest.warns(RuntimeWarning):
        tuple_result = paired_interleaved_randomized_benchmarking(
            tuple_experiment,
            ("Q0", "Q1"),
            **options,
        )

    assert list(list_result.data) == ["Q0", "Q1"]
    assert list(tuple_result.data) == ["Q0", "Q1"]
    for target in ("Q0", "Q1"):
        np.testing.assert_array_equal(
            list_result[target]["acquisition"]["seeds"],
            tuple_result[target]["acquisition"]["seeds"],
        )


def test_unordered_target_collection_is_rejected_before_measurement() -> None:
    """Target collections without stable order should not assign RNG streams."""
    exp: Any = _Experiment()

    with pytest.raises(TypeError, match="must preserve order"):
        paired_interleaved_randomized_benchmarking(
            exp,
            {"Q0", "Q1"},  # type: ignore[arg-type]
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_serial_multi_target_auto_range_selects_each_target_independently() -> None:
    """Serial targets should use independent pilot decisions and main grids."""
    exp: Any = _Experiment(
        decays={"Q0": (0.90, 0.85), "Q1": (0.98, 0.97)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            ["Q0", "Q1"],
            interleaved_clifford="X90",
            auto_range=True,
            max_n_cliffords=128,
            n_trials=3,
            sequence_seed=11,
            acquisition_seed=13,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    assert result["Q0"]["grid_selection"]["selected_main_grid"][-1] == 37
    assert result["Q1"]["grid_selection"]["selected_main_grid"][-1] == 128
    assert result["Q0"]["acquisition"]["n_cliffords"][-1] == 37
    assert result["Q1"]["acquisition"]["n_cliffords"][-1] == 128


def test_serial_multi_target_validates_every_explicit_seed_matrix_first() -> None:
    """An invalid later seed matrix should fail before measuring any target."""
    exp: Any = _Experiment()
    seeds = {
        "Q0": np.arange(18, dtype=np.int64).reshape(6, 3),
        "Q1": np.arange(15, dtype=np.int64).reshape(5, 3),
    }

    with pytest.raises(ValueError, match=r"shape \(6, 3\)"):
        paired_interleaved_randomized_benchmarking(
            exp,
            ["Q0", "Q1"],
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=3,
            seeds=seeds,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_multi_target_seed_mapping_rejects_none_before_measurement() -> None:
    """Every entry of an explicit multi-target seed mapping must be a matrix."""
    exp: Any = _Experiment()
    seeds = {
        "Q0": np.arange(8, dtype=np.int64).reshape(4, 2),
        "Q1": None,
    }

    with pytest.raises(TypeError, match=r"explicit matrix.*Q1"):
        paired_interleaved_randomized_benchmarking(
            exp,
            ("Q0", "Q1"),
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            seeds=seeds,  # type: ignore[arg-type]
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_high_level_rejects_conflicting_range_controls_before_measurement() -> None:
    """An explicit grid and maximum must remain mutually exclusive."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(ValueError, match="Specify only one"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            max_n_cliffords=8,
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x90": Blank(duration=2.0)}, "physical-qubit waveform mapping"),
        (
            {"interleaved_waveform": Blank(duration=2.0)},
            "PulseSchedule",
        ),
    ],
)
def test_two_qubit_waveform_overrides_are_rejected_before_measurement(
    overrides: dict[str, Any],
    message: str,
) -> None:
    """2Q waveform-shape errors should not reach the measurement service."""
    exp: Any = _Experiment(targets=(), cr_pairs={"CR0": ("Q0", "Q1")})

    with pytest.raises(TypeError, match=message):
        paired_interleaved_randomized_benchmarking(
            exp,
            "CR0",
            interleaved_clifford="ZX90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
            **overrides,
        )

    assert exp.measurement_service.calls == []


def test_missing_target_override_is_rejected_before_measurement() -> None:
    """A partial high-level TargetMap must not silently fall back to defaults."""
    exp: Any = _Experiment()

    with pytest.raises(ValueError, match=r"interleaved_waveform.*Q1"):
        paired_interleaved_randomized_benchmarking(
            exp,
            ("Q0", "Q1"),
            interleaved_clifford="X90",
            interleaved_waveform={"Q0": Blank(duration=2.0)},
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_incomplete_two_qubit_x90_map_is_rejected_before_measurement() -> None:
    """A CR X90 override must provide waveforms for both physical qubits."""
    exp: Any = _Experiment(targets=(), cr_pairs={"CR0": ("Q0", "Q1")})

    with pytest.raises(ValueError, match=r"x90.*Q1"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "CR0",
            interleaved_clifford="ZX90",
            x90={"Q0": Blank(duration=2.0)},
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_multi_target_schedule_reuse_is_rejected_before_measurement() -> None:
    """One target-containing schedule must not be reused across target labels."""
    exp: Any = _Experiment()

    with pytest.raises(ValueError, match="cannot reuse"):
        paired_interleaved_randomized_benchmarking(
            exp,
            ("Q0", "Q1"),
            interleaved_clifford="X90",
            interleaved_waveform=PulseSchedule(),
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


@pytest.mark.parametrize(
    ("decays", "expected_maximum"),
    [((0.90, 0.85), 37), ((0.98, 0.97), 128)],
)
def test_auto_range_selects_from_paired_pilot_decay(
    decays: tuple[float, float],
    expected_maximum: int,
) -> None:
    """Fast and slow paired decays should select correspondingly sized ranges."""
    exp: Any = _Experiment(targets=("Q0",), decays={"Q0": decays})

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            auto_range=True,
            pilot_n_trials=6,
            auto_range_remaining_fraction=0.2,
            max_n_cliffords=128,
            n_trials=4,
            sequence_seed=17,
            acquisition_seed=19,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    target = result["Q0"]
    grid_selection = target["grid_selection"]
    pilot = grid_selection["pilot_raw_result"]
    assert grid_selection["pilot_stop_reason"] == "decay_threshold_reached"
    assert grid_selection["selected_main_grid"][-1] == expected_maximum
    assert grid_selection["pilot_reference_fit"]["model"] == "A * p**m + 1/d"
    assert grid_selection["pilot_reference_fit"]["C"] == pytest.approx(0.5)
    assert grid_selection["pilot_reference_fit"]["weighting"] == "unweighted"
    assert grid_selection["pilot_fit_quality"]["reference"]["valid"] is True
    assert grid_selection["pilot_fit_quality"]["reference"]["decay_significance"] >= 3.0
    assert pilot["reference"]["trials"].shape[1] == 6
    assert target["reference"]["trials"].shape[1] == 4
    assert not np.array_equal(
        pilot["acquisition"]["seeds"],
        target["acquisition"]["seeds"],
    )
    assert target["reference"]["trials"].shape[0] == len(
        grid_selection["selected_main_grid"]
    )


def test_high_fidelity_auto_range_repositions_the_main_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-fidelity decay should move main points into informative long lengths."""
    monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.9995, 0.9990)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            auto_range=True,
            max_n_cliffords=2_048,
            n_trials=4,
            sequence_seed=71,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=True,
            save_image=False,
        )

    grid_selection = result["Q0"]["grid_selection"]
    main_grid = grid_selection["selected_main_grid"]
    np.testing.assert_array_equal(
        grid_selection["pilot_grid"],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048],
    )
    assert main_grid[0] == 0
    assert main_grid[-1] <= 2_048
    assert any(length >= 100 for length in main_grid)
    assert not {2, 4, 8, 16}.intersection(main_grid)
    assert not np.array_equal(main_grid, grid_selection["pilot_grid"])
    assert grid_selection["maximum_anchor_added"] is True
    assert grid_selection["fallback_used"] is False
    np.testing.assert_array_equal(
        result["Q0"]["acquisition"]["n_cliffords"],
        main_grid,
    )
    assert result.figure is not None
    figure: Any = result.figure
    np.testing.assert_array_equal(figure.data[1].x, main_grid)
    assert figure.layout.xaxis.type == "linear"


def test_fixed_offset_pilot_prevents_high_fidelity_premature_stopping() -> None:
    """A noisy slow 1Q decay should not be mistaken for a short decay scale."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16], dtype=np.int64)
    rng = np.random.default_rng(20_260_911)
    legacy_premature_stops = 0
    fixed_offset_premature_stops = 0
    fixed_offset_estimates: tuple[list[float], list[float]] = ([], [])
    n_experiments = 1_000

    for _ in range(n_experiments):
        paired_noise = rng.normal(0.0, 0.008, size=(len(n_cliffords), 6))
        reference_trials = (
            0.48 * 0.999 ** n_cliffords[:, None]
            + 0.5
            + paired_noise
            + rng.normal(0.0, 0.006, size=(len(n_cliffords), 6))
        )
        interleaved_trials = (
            0.48 * 0.998 ** n_cliffords[:, None]
            + 0.5
            + paired_noise
            + rng.normal(0.0, 0.006, size=(len(n_cliffords), 6))
        )
        legacy_fits = []
        pilot_fits = []
        significances = []
        for arm_index, trials in enumerate((reference_trials, interleaved_trials)):
            mean = np.mean(trials, axis=1)
            sem = np.std(trials, axis=1, ddof=1) / np.sqrt(trials.shape[1])
            legacy_fits.append(
                module._fit_rb_decay(  # noqa: SLF001
                    n_cliffords,
                    mean,
                    np.maximum(sem, 1e-3),
                )
            )
            pilot_fit = module._fit_pilot_decay(  # noqa: SLF001
                n_cliffords,
                mean,
                dimension=2,
            )
            pilot_fits.append(pilot_fit)
            fixed_offset_estimates[arm_index].append(pilot_fit.decay_parameter)
            significances.append(
                module._observed_decay_significance(mean, sem)  # noqa: SLF001
            )

        legacy_premature_stops += int(
            all(
                not fit.parameters_at_bound
                and fit.r_squared is not None
                and fit.r_squared >= 0.5
                for fit in legacy_fits
            )
            and max(fit.decay_parameter**16 for fit in legacy_fits) <= 0.2
        )
        fixed_offset_premature_stops += int(
            all(not fit.parameters_at_bound for fit in pilot_fits)
            and max(fit.decay_parameter**16 for fit in pilot_fits) <= 0.2
            and min(significances) >= 3.0
        )

    assert legacy_premature_stops / n_experiments >= 0.05
    assert fixed_offset_premature_stops / n_experiments <= 0.01
    assert np.mean(fixed_offset_estimates[0]) == pytest.approx(0.999, abs=1e-4)
    assert np.mean(fixed_offset_estimates[1]) == pytest.approx(0.998, abs=1e-4)


@pytest.mark.parametrize(
    ("dimension", "amplitude", "offset", "decay_parameter"),
    [(2, 0.5, 0.5, 0.99), (4, 0.75, 0.25, 0.95)],
)
def test_ideal_pilot_accepts_the_physical_amplitude_upper_bound(
    dimension: int,
    amplitude: float,
    offset: float,
    decay_parameter: float,
) -> None:
    """Ideal 1Q and 2Q contrast maxima should produce valid pilot fits."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16], dtype=np.int64)
    mean = amplitude * decay_parameter**n_cliffords + offset
    trials = mean[:, None] + np.linspace(-1e-4, 1e-4, 6)[None, :]

    assessment = module._assess_pilot_arm(  # noqa: SLF001
        n_cliffords,
        module._trial_moments(trials),  # noqa: SLF001
        dimension=dimension,
    )

    assert assessment.valid is True
    assert assessment.fit is not None
    assert "A" in assessment.fit.parameters_at_bound
    assert "pathological_amplitude" not in assessment.invalid_reasons


def test_pilot_rejects_a_small_decay_amplitude() -> None:
    """A fitted contrast too small to identify decay should remain invalid."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16], dtype=np.int64)
    mean = 0.01 * 0.8**n_cliffords + 0.5
    trials = mean[:, None] + np.linspace(-1e-5, 1e-5, 6)[None, :]

    assessment = module._assess_pilot_arm(  # noqa: SLF001
        n_cliffords,
        module._trial_moments(trials),  # noqa: SLF001
        dimension=2,
    )

    assert assessment.valid is False
    assert "pathological_amplitude" in assessment.invalid_reasons


@pytest.mark.parametrize("decay_parameter", [0.0, 1.0])
def test_pilot_rejects_decay_parameters_at_either_bound(
    decay_parameter: float,
) -> None:
    """A pilot decay scale at either fit bound should remain invalid."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16], dtype=np.int64)
    mean = 0.4 * decay_parameter**n_cliffords + 0.5
    trials = mean[:, None] + np.linspace(-1e-5, 1e-5, 6)[None, :]

    assessment = module._assess_pilot_arm(  # noqa: SLF001
        n_cliffords,
        module._trial_moments(trials),  # noqa: SLF001
        dimension=2,
    )

    assert assessment.valid is False
    assert "decay_parameter_at_bound" in assessment.invalid_reasons


def test_non_exponential_pilot_shape_is_invalid_despite_endpoint_decay() -> None:
    """Significant endpoint decay should not rescue a poor exponential shape."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16], dtype=np.int64)
    mean = np.asarray([0.99, 0.80, 0.97, 0.70, 0.92, 0.60])
    trials = mean[:, None] + np.linspace(-0.002, 0.002, 6)[None, :]

    assessment = module._assess_pilot_arm(  # noqa: SLF001
        n_cliffords,
        module._trial_moments(trials),  # noqa: SLF001
        dimension=2,
    )

    assert assessment.decay_significance > 3.0
    assert assessment.fit is not None
    assert assessment.fit.r_squared is not None
    assert assessment.fit.r_squared < 0.5
    assert assessment.valid is False
    assert "poor_decay_shape" in assessment.invalid_reasons


def test_upper_bound_contrast_does_not_enable_high_fidelity_premature_stop() -> None:
    """A noisy ideal-contrast slow decay should not stop at 16 Cliffords."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16], dtype=np.int64)
    rng = np.random.default_rng(20_260_912)
    premature_stops = 0
    upper_bound_hits = 0
    n_experiments = 1_000

    for _ in range(n_experiments):
        paired_noise = rng.normal(0.0, 0.008, size=(len(n_cliffords), 6))
        assessments = []
        for decay_parameter in (0.999, 0.998):
            trials = (
                0.5 * decay_parameter ** n_cliffords[:, None]
                + 0.5
                + paired_noise
                + rng.normal(0.0, 0.006, size=(len(n_cliffords), 6))
            )
            assessment = module._assess_pilot_arm(  # noqa: SLF001
                n_cliffords,
                module._trial_moments(trials),  # noqa: SLF001
                dimension=2,
            )
            assessments.append(assessment)
            upper_bound_hits += int(
                assessment.fit is not None and "A" in assessment.fit.parameters_at_bound
            )
        fits = tuple(assessment.fit for assessment in assessments)
        premature_stops += int(
            all(assessment.valid for assessment in assessments)
            and all(fit is not None for fit in fits)
            and max(fit.decay_parameter**16 for fit in fits if fit is not None) <= 0.2
        )

    assert upper_bound_hits > 0
    assert premature_stops / n_experiments <= 0.01


def test_unweighted_pilot_is_stabler_than_noisy_sem_weighting() -> None:
    """Noisy six-trial SEM weights should not control the pilot decay scale."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    n_cliffords = np.asarray([0, 1, 2, 4, 8, 16], dtype=np.int64)
    rng = np.random.default_rng(20_260_911)
    weighted_estimates = []
    unweighted_estimates = []

    def fixed_offset_decay(
        lengths: np.ndarray,
        amplitude: float,
        decay_parameter: float,
    ) -> np.ndarray:
        return amplitude * decay_parameter**lengths + 0.5

    for _ in range(1_000):
        trials = (
            0.48 * 0.999 ** n_cliffords[:, None]
            + 0.5
            + rng.normal(0.0, 0.01, size=(len(n_cliffords), 6))
        )
        mean = np.mean(trials, axis=1)
        sem = np.std(trials, axis=1, ddof=1) / np.sqrt(trials.shape[1])
        weighted_parameters, _ = curve_fit(
            fixed_offset_decay,
            n_cliffords.astype(np.float64),
            mean,
            p0=(0.48, 0.98),
            sigma=np.maximum(sem, 1e-3),
            absolute_sigma=True,
            bounds=((0.0, 0.0), (0.5, 1.0)),
            maxfev=20_000,
        )
        weighted_estimates.append(float(weighted_parameters[1]))
        unweighted_estimates.append(
            module._fit_pilot_decay(  # noqa: SLF001
                n_cliffords,
                mean,
                dimension=2,
            ).decay_parameter
        )

    assert abs(np.mean(unweighted_estimates) - 0.999) < 1e-4
    assert np.std(unweighted_estimates) < 0.95 * np.std(weighted_estimates)


def test_observed_decay_safeguard_overrides_a_spurious_fast_pilot_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low-significance first stage should continue despite a small fitted p."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    original_fit = module._fit_pilot_decay  # noqa: SLF001
    original_significance = module._observed_decay_significance  # noqa: SLF001

    def biased_first_fit(
        n_cliffords: np.ndarray,
        mean: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        fit = original_fit(n_cliffords, mean, **kwargs)
        if len(n_cliffords) == 6:
            return replace(fit, decay_parameter=0.80)
        return fit

    def low_first_significance(mean: np.ndarray, sem: np.ndarray) -> float:
        if len(mean) == 6:
            return 0.0
        return original_significance(mean, sem)

    monkeypatch.setattr(module, "_fit_pilot_decay", biased_first_fit)
    monkeypatch.setattr(
        module,
        "_observed_decay_significance",
        low_first_significance,
    )
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.999, 0.998)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=64,
            n_trials=3,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    selection = result["Q0"]["grid_selection"]
    assert selection["pilot_grid"][-1] == 64
    assert selection["pilot_stage_diagnostics"][0]["decision"] == (
        "insufficient_observed_decay"
    )


def test_moderate_fidelity_auto_range_uses_a_shorter_adaptive_grid() -> None:
    """Moderate 1Q decay should place the main grid below the high-fidelity scale."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.995, 0.990)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=2_048,
            n_trials=4,
            sequence_seed=73,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    main_grid = result["Q0"]["grid_selection"]["selected_main_grid"]
    assert 750 <= main_grid[-1] <= 850
    assert not {2, 4, 8, 16}.intersection(main_grid)
    assert result["Q0"]["grid_selection"]["pilot_remaining_fraction"] == 0.10
    assert result["Q0"]["grid_selection"]["main_remaining_fractions"] == (
        0.90,
        0.70,
        0.50,
        0.30,
        0.15,
        0.05,
        0.02,
    )


def test_two_qubit_auto_range_samples_the_short_decay_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2Q-like decay should allocate several main points at short lengths."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    exp: Any = _Experiment(
        targets=(),
        cr_pairs={"CR0": ("Q0", "Q1")},
        decays={"CR0": (0.98, 0.95)},
    )

    def convert(point_result: Any, **kwargs: Any) -> Any:
        """Expose the synthetic marker probability as a 2Q `P(00)` value."""
        probability = float(np.real(point_result.data["CR0"][-1].data))
        return SimpleNamespace(
            get_mitigated_probabilities=lambda qubits: {"00": probability},
            get_probabilities=lambda qubits: {"00": probability},
        )

    monkeypatch.setattr(
        module.MeasurementResultConverter,
        "to_measure_result",
        Mock(side_effect=convert),
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "CR0",
            interleaved_clifford="ZX90",
            max_n_cliffords=256,
            n_trials=4,
            sequence_seed=79,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            mitigate_readout=False,
            plot=False,
            save_image=False,
        )

    main_grid = result["CR0"]["grid_selection"]["selected_main_grid"]
    assert main_grid[-1] == 194
    assert sum(length <= 16 for length in main_grid) >= 5
    assert result["CR0"]["metadata"]["dimension"] == 4
    selection = result["CR0"]["grid_selection"]
    assert selection["pilot_reference_fit"]["C"] == pytest.approx(0.25)
    assert selection["pilot_interleaved_fit"]["C"] == pytest.approx(0.25)
    assert selection["pilot_model"]["used_for_main_fit"] is False
    assert selection["pilot_model"]["unmitigated_2q_readout_caveat"] is not None
    assert selection["pilot_stop_reason"] == "decay_threshold_reached"
    assert selection["pilot_grid"][-1] < 256


def test_very_fast_decay_adds_only_the_reference_tail() -> None:
    """The 2% reference point should extend the interleaved-first grid."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.50, 0.40)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=128,
            n_trials=4,
            sequence_seed=81,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    selection = result["Q0"]["grid_selection"]
    np.testing.assert_array_equal(selection["selected_main_grid"], [0, 1, 2, 3, 4, 6])
    np.testing.assert_array_equal(selection["supplemental_integer_grid"], [])
    np.testing.assert_array_equal(selection["candidate_interleaved_grid"], [1, 2, 3, 4])
    np.testing.assert_array_equal(selection["reference_tail_grid"], [6])
    assert selection["fallback_used"] is False


def test_short_grid_fill_does_not_enumerate_a_large_clifford_range() -> None:
    """Sparse-grid completion should remain bounded for a very large maximum."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )

    grid, added = module._fill_short_main_grid(  # noqa: SLF001
        np.asarray([0, 1, 10**12], dtype=np.int64),
        maximum=10**12,
    )

    assert len(grid) == 6
    assert len(added) == 3
    assert grid[0] == 0
    assert grid[-1] == 10**12
    assert np.all(np.diff(grid) > 0)


def test_default_grid_rejects_a_maximum_larger_than_the_storage_dtype() -> None:
    """An out-of-range maximum should fail clearly instead of overflowing."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )

    with pytest.raises(ValueError, match="signed 64-bit integer"):
        module._default_n_cliffords(np.iinfo(np.int64).max + 1)  # noqa: SLF001


def test_auto_range_includes_a_non_power_of_two_maximum_candidate() -> None:
    """Auto-range should measure its exact configured exploration ceiling."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.999, 0.998)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=100,
            n_trials=3,
            sequence_seed=107,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    selection = result["Q0"]["grid_selection"]
    np.testing.assert_array_equal(
        selection["candidate_pilot_grid"],
        [0, 1, 2, 4, 8, 16, 32, 64, 100],
    )
    assert selection["pilot_grid"][-1] == 100


def test_fixed_power_of_two_grid_does_not_add_a_non_power_of_two_maximum() -> None:
    """Disabling auto-range should retain the legacy fixed-grid behavior."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            auto_range=False,
            max_n_cliffords=100,
            n_trials=3,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    np.testing.assert_array_equal(
        result["Q0"]["acquisition"]["n_cliffords"],
        [0, 1, 2, 4, 8, 16, 32, 64],
    )


def test_faster_reference_adds_no_tail_to_the_interleaved_grid() -> None:
    """A covered reference range should not add internal reference candidates."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.979, 0.980)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=512,
            n_trials=4,
            sequence_seed=83,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    selection = result["Q0"]["grid_selection"]
    np.testing.assert_array_equal(selection["reference_tail_grid"], [])
    np.testing.assert_array_equal(
        selection["selected_main_grid"],
        [0, 1, *selection["candidate_interleaved_grid"]],
    )
    assert selection["maximum_anchor_added"] is False


def test_different_decay_parameters_contribute_both_candidate_regions() -> None:
    """Different arm decays should both contribute informative main lengths."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.995, 0.950)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=512,
            n_trials=4,
            sequence_seed=89,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    selection = result["Q0"]["grid_selection"]
    assert selection["candidate_reference_grid"][-1] == 512
    assert selection["candidate_interleaved_grid"][-1] == 76
    np.testing.assert_array_equal(
        selection["reference_tail_grid"], [138, 240, 378, 512]
    )
    np.testing.assert_array_equal(
        selection["selected_main_grid"],
        sorted(
            {
                0,
                1,
                *selection["candidate_interleaved_grid"],
                *selection["reference_tail_grid"],
            }
        ),
    )
    assert {21, 71}.isdisjoint(selection["selected_main_grid"])


@pytest.mark.parametrize(
    ("fractions", "expected_maximum", "expected_points", "expected_sequence_cost"),
    [
        ((0.9, 0.7, 0.5, 0.3, 0.15), 378, 9, 8_336),
        ((0.9, 0.75, 0.6, 0.45, 0.3, 0.2), 321, 10, 8_088),
        ((0.8, 0.6, 0.4, 0.2, 0.1), 459, 9, 10_672),
    ],
)
def test_main_contrast_fraction_designs_produce_stable_bounded_grids(
    fractions: tuple[float, ...],
    expected_maximum: int,
    expected_points: int,
    expected_sequence_cost: int,
) -> None:
    """Supported contrast designs should retain exact noiseless fit estimates."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.995, 0.990)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=512,
            main_remaining_fractions=fractions,
            n_trials=4,
            sequence_seed=101,
            n_bootstrap=5,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    target = result["Q0"]
    selection = target["grid_selection"]
    main_grid = selection["selected_main_grid"]
    expected_fidelity = 1.0 - 0.5 * (1.0 - 0.990 / 0.995)
    assert selection["main_remaining_fractions"] == fractions
    assert len(main_grid) == expected_points
    assert main_grid[-1] == expected_maximum
    assert target["reference"]["fit"]["p"] == pytest.approx(0.995, abs=1e-8)
    assert target["interleaved"]["fit"]["p"] == pytest.approx(0.990, abs=1e-8)
    assert target["gate_fidelity"] == pytest.approx(expected_fidelity, abs=1e-8)
    assert target["gate_fidelity_err"] == pytest.approx(0.0, abs=1e-12)
    assert target["gate_fidelity_ci95"] == pytest.approx(
        (expected_fidelity, expected_fidelity),
        abs=1e-8,
    )
    assert target["acquisition"]["n_measurement_points"] == 8 * len(main_grid)
    assert 8 * int(np.sum(main_grid)) == expected_sequence_cost


def test_main_contrast_fractions_are_validated_before_measurement() -> None:
    """Invalid contrast ordering should fail without making a hardware call."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(ValueError, match="strictly decreasing"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            main_remaining_fractions=(0.5, 0.7),
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


@pytest.mark.parametrize(
    "fractions",
    [
        {0.90, 0.70, 0.50, 0.30, 0.15},
        frozenset({0.90, 0.70, 0.50, 0.30, 0.15}),
    ],
)
def test_unordered_main_contrast_fractions_are_rejected_before_measurement(
    fractions: Collection[float],
) -> None:
    """Unordered contrast targets should fail before assigning a main grid."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(TypeError, match="must preserve order"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            main_remaining_fractions=fractions,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_main_contrast_fraction_mapping_is_rejected_before_measurement() -> None:
    """Mapping keys must not be mistaken for an ordered contrast sequence."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(TypeError, match="not a mapping"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            main_remaining_fractions={0.9: "high", 0.5: "middle", 0.1: "low"},
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_serial_auto_range_validates_main_trials_before_pilot() -> None:
    """Invalid main trial count should fail before any pilot hardware call."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(ValueError, match=r"n_trials.*at least 2"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_trials=1,
            max_n_cliffords=32,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_unit_decay_uses_an_auditable_power_of_two_fallback() -> None:
    """An unidentifiable `p=1` pilot should safely retain the bounded pilot grid."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (1.0, 1.0)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=32,
            n_trials=4,
            sequence_seed=97,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    selection = result["Q0"]["grid_selection"]
    assert selection["fallback_used"] is True
    assert selection["fallback_reason"] == "pilot_fit_unusable"
    assert selection["pilot_stop_reason"] == "max_n_cliffords_reached"
    assert selection["pilot_stop_detail"] == "fit_unusable"
    assert selection["pilot_fit_quality"]["reference"]["valid"] is False
    assert (
        "insufficient_observed_decay"
        in selection["pilot_fit_quality"]["reference"]["invalid_reasons"]
    )
    np.testing.assert_array_equal(
        selection["selected_main_grid"],
        [0, 1, 2, 4, 8, 16, 32],
    )


def test_invalid_pilot_decay_uses_the_power_of_two_fallback() -> None:
    """Non-finite pilot decay parameters must not enter logarithmic mapping."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    fallback_grid = module._default_n_cliffords(32)  # noqa: SLF001

    selection = module._select_decay_adapted_main_grid(  # noqa: SLF001
        (
            SimpleNamespace(decay_parameter=np.nan),
            SimpleNamespace(decay_parameter=0.95),
        ),
        remaining_fractions=(0.9, 0.7, 0.5, 0.3, 0.15),
        maximum=32,
        fallback_grid=fallback_grid,
    )

    assert selection.fallback_used is True
    assert selection.fallback_reason is not None
    assert selection.fallback_reason.startswith("invalid_pilot_decay")
    np.testing.assert_array_equal(selection.selected_grid, fallback_grid)


def test_auto_range_rejects_explicit_seed_matrix_before_measurement() -> None:
    """An unknown main range must not accept a fixed-shape seed matrix."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(ValueError, match=r"auto_range.*seeds"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            auto_range=True,
            max_n_cliffords=32,
            seeds=np.arange(24, dtype=np.int64).reshape(6, 4),
            n_trials=4,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_auto_range_rejects_a_ceiling_with_too_few_pilot_lengths() -> None:
    """Auto-range should fail before hardware when six pilot points cannot fit."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(ValueError, match="at least six pilot lengths"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            max_n_cliffords=8,
            n_trials=2,
            n_bootstrap=0,
            pairs_per_sweep=100,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_auto_range_rejects_fewer_than_four_pilot_trials() -> None:
    """An unstable pilot trial count should fail before hardware acquisition."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(ValueError, match=r"pilot_n_trials.*at least 4"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            pilot_n_trials=3,
            max_n_cliffords=32,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_disabled_auto_range_measures_the_complete_default_grid_without_pilot() -> None:
    """Disabling auto-range should run one fresh acquisition on the full grid."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            auto_range=False,
            max_n_cliffords=32,
            n_trials=3,
            sequence_seed=21,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    grid_selection = result["Q0"]["grid_selection"]
    np.testing.assert_array_equal(
        result["Q0"]["acquisition"]["n_cliffords"],
        [0, 1, 2, 4, 8, 16, 32],
    )
    assert grid_selection["enabled"] is False
    assert grid_selection["pilot_stop_reason"] == "auto_range_disabled"
    assert grid_selection["pilot_raw_result"] is None
    assert len(exp.measurement_service.calls) == 1


def test_auto_range_records_when_the_decay_threshold_is_not_reached(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A slow pilot decay should use the full allowed range and record why."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.999, 0.998)},
    )

    caplog.set_level(
        "INFO",
        logger="qubex.contrib.experiment.paired_interleaved_randomized_benchmarking",
    )
    with pytest.warns(RuntimeWarning) as warning_records:
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            auto_range=True,
            auto_range_remaining_fraction=0.1,
            max_n_cliffords=32,
            n_trials=4,
            sequence_seed=23,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    assert any(
        "maximum Clifford length" in str(record.message) for record in warning_records
    )
    grid_selection = result["Q0"]["grid_selection"]
    assert grid_selection["pilot_stop_reason"] == "max_n_cliffords_reached"
    assert grid_selection["selected_main_grid"][-1] == 32
    assert grid_selection["pilot_grid"][-1] == 32
    np.testing.assert_array_equal(
        grid_selection["supplemental_integer_grid"],
        [3, 7, 15],
    )
    assert grid_selection["fallback_used"] is False
    assert "max_n_cliffords_reached" in caplog.text
    assert "maximum reached" in caplog.text


def test_pilot_threshold_controls_pilot_cost_but_not_the_adaptive_main_grid() -> None:
    """Pilot thresholds may change pilot cost without changing a fitted main grid."""
    selected_maxima = []
    main_measurement_points = []
    pilot_measurement_points = []
    p_reference = []
    p_interleaved = []
    fidelities = []
    bootstrap_sigmas = []
    for threshold in (0.1, 0.2, 0.3):
        exp: Any = _Experiment(
            targets=("Q0",),
            decays={"Q0": (0.92, 0.90)},
        )
        with pytest.warns(RuntimeWarning):
            result = paired_interleaved_randomized_benchmarking(
                exp,
                "Q0",
                interleaved_clifford="X90",
                auto_range=True,
                auto_range_remaining_fraction=threshold,
                max_n_cliffords=128,
                n_trials=4,
                sequence_seed=29,
                n_bootstrap=10,
                pairs_per_sweep=1_000,
                plot=False,
                save_image=False,
            )
        grid_selection = result["Q0"]["grid_selection"]
        selected_maxima.append(grid_selection["selected_main_grid"][-1])
        main_measurement_points.append(
            result["Q0"]["acquisition"]["n_measurement_points"]
        )
        pilot_measurement_points.append(
            grid_selection["pilot_raw_result"]["acquisition"]["n_measurement_points"]
        )
        p_reference.append(result["Q0"]["reference"]["fit"]["p"])
        p_interleaved.append(result["Q0"]["interleaved"]["fit"]["p"])
        fidelities.append(result["Q0"]["gate_fidelity"])
        bootstrap_sigmas.append(result["Q0"]["gate_fidelity_err"])

    assert selected_maxima == [47, 47, 47]
    assert main_measurement_points == [72, 72, 72]
    assert pilot_measurement_points == [84, 84, 72]
    assert p_reference == pytest.approx([0.92, 0.92, 0.92], abs=1e-8)
    assert p_interleaved == pytest.approx([0.90, 0.90, 0.90], abs=1e-8)
    assert fidelities == pytest.approx(
        [1.0 - 0.5 * (1.0 - 0.90 / 0.92)] * 3,
        abs=1e-8,
    )
    assert bootstrap_sigmas == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_auto_range_extends_when_the_first_pilot_shape_is_poor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poor first decay shape should force another length before selection."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    original_fit = module._fit_pilot_decay  # noqa: SLF001

    def mark_six_point_fit_as_poor_shape(
        n_cliffords: np.ndarray,
        mean: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        """Lower only the minimum-size fit's shape diagnostic."""
        fit = original_fit(n_cliffords, mean, **kwargs)
        if len(n_cliffords) == 6:
            return replace(fit, r_squared=0.1)
        return fit

    monkeypatch.setattr(
        module,
        "_fit_pilot_decay",
        mark_six_point_fit_as_poor_shape,
    )
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.90, 0.85)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            auto_range=True,
            max_n_cliffords=64,
            n_trials=3,
            sequence_seed=61,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            plot=False,
            save_image=False,
        )

    grid_selection = result["Q0"]["grid_selection"]
    assert grid_selection["pilot_stop_reason"] == "decay_threshold_reached"
    assert grid_selection["pilot_grid"][-1] == 32
    assert grid_selection["selected_main_grid"][-1] == 37
    assert (
        "poor_decay_shape"
        in (
            grid_selection["pilot_stage_diagnostics"][0]["fit_quality"]["reference"][
                "invalid_reasons"
            ]
        )
    )


def test_parallel_targets_share_points_but_keep_independent_sequence_seeds() -> None:
    """Parallel acquisition should align schedules and preserve per-target pairs."""
    exp: Any = _Experiment()

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            ["Q0", "Q1"],
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=3,
            sequence_seed=31,
            acquisition_seed=37,
            n_bootstrap=0,
            pairs_per_sweep=100,
            in_parallel=True,
            plot=False,
            save_image=False,
        )

    q0_acquisition = result["Q0"]["acquisition"]
    q1_acquisition = result["Q1"]["acquisition"]
    assert not np.array_equal(q0_acquisition["seeds"], q1_acquisition["seeds"])
    assert q0_acquisition["pair_chunks"][0] is not q1_acquisition["pair_chunks"][0]
    assert [pair["first_protocol"] for pair in q0_acquisition["planned_order"]] == [
        pair["first_protocol"] for pair in q1_acquisition["planned_order"]
    ]
    assert (
        sum(len(call["sweep_values"]) for call in exp.measurement_service.calls)
        == (q0_acquisition["n_measurement_points"])
    )
    for call in exp.measurement_service.calls:
        for markers, duration in zip(
            call["point_markers"],
            call["point_durations"],
            strict=True,
        ):
            assert {marker.target for marker in markers} == {"Q0", "Q1"}
            assert len({marker.interleaved_clifford is None for marker in markers}) == 1
            assert duration == max(marker.duration for marker in markers)


def test_parallel_rejects_overlapping_two_qubit_targets_before_measurement() -> None:
    """CR targets sharing a physical qubit must not be scheduled together."""
    exp: Any = _Experiment(
        targets=(),
        cr_pairs={"CR0": ("Q0", "Q1"), "CR1": ("Q1", "Q2")},
    )

    with pytest.raises(ValueError, match="share physical qubits"):
        paired_interleaved_randomized_benchmarking(
            exp,
            ["CR0", "CR1"],
            interleaved_clifford="ZX90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=3,
            n_bootstrap=0,
            in_parallel=True,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_parallel_rejects_mixed_one_and_two_qubit_targets() -> None:
    """The initial parallel implementation should require one target kind."""
    exp: Any = _Experiment(
        targets=("Q0",),
        cr_pairs={"CR0": ("Q1", "Q2")},
    )

    with pytest.raises(ValueError, match="cannot mix 1Q and 2Q"):
        paired_interleaved_randomized_benchmarking(
            exp,
            ["Q0", "CR0"],
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=3,
            n_bootstrap=0,
            in_parallel=True,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls == []


def test_parallel_two_qubit_targets_use_unaveraged_shared_measurement_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disjoint CR targets should share points and retain separate `P(00)` data."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    exp: Any = _Experiment(
        targets=(),
        cr_pairs={"CR0": ("Q0", "Q1"), "CR1": ("Q2", "Q3")},
    )
    x90 = {qubit: Blank(duration=2.0) for qubit in ("Q0", "Q1", "Q2", "Q3")}
    zx90 = {target: PulseSchedule() for target in ("CR0", "CR1")}
    interleaved = {target: PulseSchedule() for target in ("CR0", "CR1")}
    converted = SimpleNamespace(
        get_mitigated_probabilities=lambda qubits: {"00": 0.91},
        get_probabilities=lambda qubits: {"00": 0.89},
    )
    monkeypatch.setattr(
        module.MeasurementResultConverter,
        "to_measure_result",
        Mock(return_value=converted),
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            ["CR0", "CR1"],
            interleaved_clifford="ZX90",
            interleaved_waveform=interleaved,
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=3,
            sequence_seed=39,
            n_bootstrap=0,
            pairs_per_sweep=100,
            x90=x90,
            zx90=zx90,
            in_parallel=True,
            plot=False,
            save_image=False,
        )

    assert exp.measurement_service.calls[0]["shot_averaging"] is False
    np.testing.assert_allclose(result["CR0"]["reference"]["trials"], 0.91)
    np.testing.assert_allclose(result["CR1"]["interleaved"]["trials"], 0.91)
    assert result["CR0"]["metadata"]["dimension"] == 4
    assert result["CR1"]["metadata"]["dimension"] == 4
    for marker in exp.sequence_calls:
        assert marker.x90 is x90
        assert marker.zx90 is zx90[marker.target]
        if marker.interleaved_clifford is not None:
            assert marker.interleaved_waveform is interleaved[marker.target]


def test_parallel_auto_range_uses_the_slowest_targets_shared_grid() -> None:
    """Parallel main acquisition should use the longest range needed by the group."""
    exp: Any = _Experiment(
        decays={"Q0": (0.90, 0.85), "Q1": (0.98, 0.97)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            ["Q0", "Q1"],
            interleaved_clifford="X90",
            auto_range=True,
            max_n_cliffords=128,
            n_trials=3,
            sequence_seed=41,
            acquisition_seed=43,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            in_parallel=True,
            plot=False,
            save_image=False,
        )

    assert result["Q0"]["grid_selection"]["target_candidate_main_grid"][-1] == 37
    assert result["Q1"]["grid_selection"]["target_candidate_main_grid"][-1] == 128
    np.testing.assert_array_equal(
        result["Q0"]["grid_selection"]["selected_main_grid"],
        result["Q1"]["grid_selection"]["selected_main_grid"],
    )
    assert (
        result["Q0"]["grid_selection"]["selected_main_grid"]
        is not result["Q1"]["grid_selection"]["selected_main_grid"]
    )
    shared_grid = result["Q0"]["grid_selection"]["selected_main_grid"]
    assert len(shared_grid) == 16
    assert 2 in shared_grid
    assert shared_grid[-1] == 128


def test_parallel_shared_grid_retains_each_targets_complete_grid() -> None:
    """The parallel union should preserve every target's selected grid."""
    decays = {
        "Q0": (0.9995, 0.9990),
        "Q1": (0.9950, 0.9900),
        "Q2": (0.9800, 0.9500),
    }
    parallel_exp: Any = _Experiment(targets=tuple(decays), decays=decays)
    serial_exp: Any = _Experiment(targets=tuple(decays), decays=decays)
    options = {
        "interleaved_clifford": "X90",
        "max_n_cliffords": 2_048,
        "n_trials": 4,
        "sequence_seed": 109,
        "acquisition_seed": 113,
        "n_bootstrap": 10,
        "pairs_per_sweep": 1_000,
        "plot": False,
        "save_image": False,
    }

    with pytest.warns(RuntimeWarning):
        parallel = paired_interleaved_randomized_benchmarking(
            parallel_exp,
            tuple(decays),
            in_parallel=True,
            **options,
        )
    with pytest.warns(RuntimeWarning):
        serial = paired_interleaved_randomized_benchmarking(
            serial_exp,
            tuple(decays),
            **options,
        )

    shared_grid = parallel["Q0"]["grid_selection"]["selected_main_grid"]
    assert len(shared_grid) > 14
    target_grids = []
    for target, (p_reference, p_interleaved) in decays.items():
        selection = parallel[target]["grid_selection"]
        target_grid = selection["target_candidate_main_grid"]
        target_grids.append(target_grid)
        assert len(target_grid) >= 6
        assert set(target_grid).issubset(shared_grid)
        assert parallel[target]["reference"]["fit"]["p"] == pytest.approx(
            p_reference,
            abs=1e-8,
        )
        assert parallel[target]["interleaved"]["fit"]["p"] == pytest.approx(
            p_interleaved,
            abs=1e-8,
        )
        assert parallel[target]["gate_fidelity"] == pytest.approx(
            serial[target]["gate_fidelity"],
            abs=1e-10,
        )
        assert parallel[target]["gate_fidelity_err"] == pytest.approx(
            serial[target]["gate_fidelity_err"],
            abs=1e-12,
        )
    np.testing.assert_array_equal(
        shared_grid,
        sorted({int(length) for grid in target_grids for length in grid}),
    )


def test_parallel_shared_grid_preserves_fallback_target_anchors() -> None:
    """The shared union should retain every fallback-grid point."""
    exp: Any = _Experiment(
        decays={"Q0": (1.0, 1.0), "Q1": (0.98, 0.95)},
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            ("Q0", "Q1"),
            interleaved_clifford="X90",
            max_n_cliffords=128,
            n_trials=3,
            sequence_seed=127,
            n_bootstrap=0,
            pairs_per_sweep=1_000,
            in_parallel=True,
            plot=False,
            save_image=False,
        )

    selection = result["Q0"]["grid_selection"]
    assert selection["fallback_used"] is True
    target_grid = selection["target_candidate_main_grid"]
    assert len(target_grid) == 9
    assert {0, 1, 128}.issubset(target_grid)
    assert set(target_grid).issubset(selection["selected_main_grid"])


def test_target_maps_resolve_waveforms_and_ignore_zx90_for_one_qubit_targets() -> None:
    """Resolve 1Q waveforms while keeping the 2Q-only ZX90 override out of RB."""
    exp: Any = _Experiment()
    x90 = {"Q0": Blank(duration=2), "Q1": Blank(duration=4)}
    interleaved = {"Q0": Blank(duration=6), "Q1": Blank(duration=8)}
    zx90 = {"Q0": PulseSchedule(), "Q1": PulseSchedule()}

    with pytest.warns(RuntimeWarning):
        paired_interleaved_randomized_benchmarking(
            exp,
            ["Q0", "Q1"],
            interleaved_clifford="X90",
            interleaved_waveform=interleaved,
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=3,
            sequence_seed=47,
            n_bootstrap=0,
            pairs_per_sweep=100,
            x90=x90,
            zx90=zx90,
            plot=False,
            save_image=False,
        )

    for target in ("Q0", "Q1"):
        markers = [marker for marker in exp.sequence_calls if marker.target == target]
        assert all(marker.x90 is x90[target] for marker in markers)
        assert all(marker.zx90 is None for marker in markers)
        interleaved_markers = [
            marker for marker in markers if marker.interleaved_clifford is not None
        ]
        assert all(
            marker.interleaved_waveform is interleaved[target]
            for marker in interleaved_markers
        )


def test_high_level_defaults_plot_save_and_log_bootstrap_summary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Legacy-facing defaults should produce a figure, save it, and log statistics."""
    exp: Any = _Experiment(targets=("Q0",))
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    show = Mock()
    save = Mock()
    monkeypatch.setattr("plotly.graph_objects.Figure.show", show)
    monkeypatch.setattr(module.viz, "save_figure", save)
    caplog.set_level(
        "INFO",
        logger="qubex.contrib.experiment.paired_interleaved_randomized_benchmarking",
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=4,
            sequence_seed=53,
            n_bootstrap=5,
            pairs_per_sweep=100,
        )

    assert result.figure is not None
    assert result.figures is not None
    assert set(result.figures) == {"Q0"}
    show.assert_called_once()
    save.assert_called_once()
    assert "Paired IRB: Q0" in caplog.text
    assert "p_ref =" in caplog.text
    assert "p_irb =" in caplog.text
    assert "Gate fidelity =" in caplog.text
    assert "95% bootstrap CI" in caplog.text
    assert "decay_threshold_reached" not in caplog.text
    assert "explicit_range" in caplog.text


def test_high_level_explicit_false_disables_plot_and_save_and_logs_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit false flags should remain side-effect free and log invalid uncertainty."""
    exp: Any = _Experiment(targets=("Q0",))
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    show = Mock()
    save = Mock()
    monkeypatch.setattr("plotly.graph_objects.Figure.show", show)
    monkeypatch.setattr(module.viz, "save_figure", save)
    caplog.set_level(
        "INFO",
        logger="qubex.contrib.experiment.paired_interleaved_randomized_benchmarking",
    )

    with pytest.warns(RuntimeWarning):
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=4,
            sequence_seed=59,
            n_bootstrap=0,
            pairs_per_sweep=100,
            plot=False,
            save_image=False,
        )

    assert result.figure is None
    assert result.figures is None
    show.assert_not_called()
    save.assert_not_called()
    assert "Statistical uncertainty unavailable" in caplog.text
    assert "optimizer = disabled" in caplog.text
