"""Planning and fitting tests for CPMG noise spectroscopy."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.typing import ArrayLike

import qubex.visualization as viz
from qubex.contrib.experiment.cpmg_noise_spectroscopy import (
    _build_cpmg_sequence,
    _fit_cpmg_decay,
    _load_payload,
    _make_frequency_plan,
    _n_pi_from_time_range,
    _plot_cpmg_decay,
    _resolve_pi_pulse,
    _save_payload,
    cpmg_noise_spectroscopy,
)
from qubex.pulse import Rect

_TEST_TIME_RANGE_NS = [8_000.0, 10_000.0, 12_000.0, 14_000.0, 16_000.0, 20_000.0]


class _PulseService:
    def __init__(self, *, sampling_period: float = 0.4) -> None:
        self.hpi = Rect(
            duration=10.0,
            amplitude=0.5,
            sampling_period=sampling_period,
        )
        self.pi = Rect(
            duration=20.0,
            amplitude=1.0,
            sampling_period=sampling_period,
        )

    def get_hpi_pulse(self, _target: str) -> Rect:
        return self.hpi

    def x180(self, _target: str) -> Rect:
        return self.pi


class _Experiment:
    def __init__(self) -> None:
        self.pulse = _PulseService()


def test_pi_pulse_resolution_does_not_hide_provider_failures() -> None:
    """A callable pulse provider failure should propagate instead of using a fallback."""

    class FailingPulseService(_PulseService):
        def x180(self, _target: str) -> Rect:
            raise RuntimeError("calibration unavailable")

    exp = _Experiment()
    exp.pulse = FailingPulseService()

    with pytest.raises(RuntimeError, match="calibration unavailable"):
        _resolve_pi_pulse(exp, "Q00", None)  # type: ignore[arg-type]


def test_frequency_plan_includes_pi_duration_and_sampling_grid() -> None:
    """A 5 MHz request with a 20 ns pi pulse should realize a 100 ns center spacing."""
    plan = _make_frequency_plan(
        [0.005],
        sampling_period=0.4,
        pi_duration=20.0,
        max_pulse_duty_ratio=0.25,
    )

    point = plan[0]
    assert point.status == "valid"
    assert point.half_tau == pytest.approx(40.0)
    assert point.center_spacing == pytest.approx(100.0)
    assert point.frequency == pytest.approx(0.005)
    assert point.pulse_duty_ratio == pytest.approx(0.2)


def test_frequency_plan_marks_duty_limit_and_duplicate_ticks() -> None:
    """Duty-limit failures and quantized duplicates should retain their requested indices."""
    plan = _make_frequency_plan(
        [0.001, 0.00100001, 0.004],
        sampling_period=2.0,
        pi_duration=40.0,
        max_pulse_duty_ratio=0.25,
    )

    assert [point.requested_index for point in plan] == [0, 1, 2]
    assert plan[0].status == "valid"
    assert plan[1].status == "duplicate"
    assert plan[1].reason == "duplicate_realized_frequency"
    assert plan[2].status == "skipped"
    assert plan[2].reason == "pulse_duty_ratio_exceeded"


def test_frequency_plan_depends_on_each_targets_pi_duration() -> None:
    """Odd tick differences in pi duration should produce target-specific frequencies."""
    short = _make_frequency_plan(
        [0.0033],
        sampling_period=2.0,
        pi_duration=20.0,
        max_pulse_duty_ratio=0.25,
    )
    long = _make_frequency_plan(
        [0.0033],
        sampling_period=2.0,
        pi_duration=22.0,
        max_pulse_duty_ratio=0.25,
    )

    assert short[0].frequency != long[0].frequency


def test_frequency_plan_rejects_nonfinite_sampling_period() -> None:
    """A nonfinite sampling period should fail with a clear validation error."""
    with pytest.raises(ValueError, match="sampling_period must be finite and positive"):
        _make_frequency_plan(
            [0.001],
            sampling_period=np.nan,
            pi_duration=20.0,
            max_pulse_duty_ratio=0.25,
        )


def test_schedule_duration_contains_exact_cpmg_evolution_time() -> None:
    """The schedule between endpoint rotations should equal n_pi times center spacing."""
    exp = _Experiment()
    schedule = _build_cpmg_sequence(
        exp,  # type: ignore[arg-type]
        target="Q00",
        half_tau=40.0,
        n_pi=16,
        pattern="++++",
        pi_pulses=None,
        sampling_period=0.4,
    )

    assert schedule.duration == pytest.approx(10.0 + 16 * 100.0 + 10.0)


@pytest.mark.parametrize(
    ("pattern", "period", "times_ns", "expected"),
    [
        ("+--+", 4, [1_790.0, 1_810.0], [16, 20]),
        ("+-+-", 2, [1_680.0, 1_720.0], [16, 18]),
    ],
)
def test_time_plan_rounds_to_nearest_complete_pattern_cycle(
    pattern: str,
    period: int,
    times_ns: list[float],
    expected: list[int],
) -> None:
    """Time-derived counts should round to the nearest complete pattern cycle."""
    counts = _n_pi_from_time_range(
        times_ns,
        center_spacing_ns=100.0,
        pattern=pattern,  # type: ignore[arg-type]
        min_pi_pulses=16,
    )

    assert all(counts % period == 0)
    np.testing.assert_array_equal(counts, expected)


@pytest.mark.parametrize("center_spacing_ns", [0.0, -1.0, np.nan, np.inf])
def test_time_plan_rejects_invalid_center_spacing(center_spacing_ns: float) -> None:
    """Time-derived planning should reject nonpositive or nonfinite spacing."""
    with pytest.raises(ValueError, match="center_spacing_ns"):
        _n_pi_from_time_range(
            [1_000.0],
            center_spacing_ns=center_spacing_ns,
            pattern="++++",
        )


def test_cpmg_fit_recovers_signed_amplitude_and_gamma() -> None:
    """Synthetic decay data should recover a negative amplitude and the known decay rate."""
    times_ns = np.linspace(2_000.0, 100_000.0, 40)
    gamma_per_ns = 1.0 / 35_000.0
    values = -0.42 * np.exp(-gamma_per_ns * times_ns) + 0.55

    result = _fit_cpmg_decay(times_ns, values, min_fit_points=6)

    assert result.fit_status == "ok"
    assert result.amplitude == pytest.approx(-0.42, rel=1e-4)
    assert result.gamma2_per_ns == pytest.approx(gamma_per_ns, rel=1e-4)
    assert result.t2_ns == pytest.approx(35_000.0, rel=1e-4)


def test_cpmg_fit_reports_unresolved_constant_data() -> None:
    """A constant signal should not produce an apparently finite T2 estimate."""
    result = _fit_cpmg_decay(
        np.linspace(1_000.0, 100_000.0, 20),
        np.full(20, 0.4),
        min_fit_points=6,
    )

    assert result.fit_status == "unresolved_decay"
    assert np.isnan(result.t2_ns)
    assert "weak_decay" in result.quality_flags


def test_cpmg_fit_uses_distinct_status_for_quality_warnings() -> None:
    """A usable but poor-model fit should not be labeled as fully reliable."""
    times_ns = np.linspace(2_000.0, 100_000.0, 40)
    values = (
        0.42 * np.exp(-times_ns / 35_000.0)
        + 0.5
        + 0.08 * np.sin(np.arange(len(times_ns)) * 2.3)
    )

    result = _fit_cpmg_decay(times_ns, values, min_fit_points=6)

    assert result.fit_status == "quality_warning"
    assert "poor_model_fit" in result.quality_flags
    assert np.isfinite(result.t2_ns)


def test_cpmg_fit_uses_valid_measurement_sem_as_absolute_weights() -> None:
    """Valid pointwise SEM should select weighted fitting and reduced chi-square."""
    times_ns = np.linspace(2_000.0, 80_000.0, 30)
    values = 0.45 * np.exp(-times_ns / 30_000.0) + 0.48
    sem = np.linspace(0.005, 0.02, len(times_ns))

    result = _fit_cpmg_decay(
        times_ns,
        values,
        sem_values=sem,
        min_fit_points=6,
    )

    assert result.fit_method_used == "weighted_least_squares"
    assert result.uncertainty_source == "measurement_sem"
    assert np.isfinite(result.reduced_chi2)


def test_cpmg_fit_falls_back_without_discarding_points_for_invalid_sem() -> None:
    """Invalid SEM should select an unweighted fit without dropping signal points."""
    times_ns = np.linspace(2_000.0, 80_000.0, 30)
    values = 0.45 * np.exp(-times_ns / 30_000.0) + 0.48
    sem = np.full(len(times_ns), 0.01)
    sem[4] = 0.0

    result = _fit_cpmg_decay(
        times_ns,
        values,
        sem_values=sem,
        min_fit_points=6,
    )

    assert result.fit_method_used == "unweighted_least_squares"
    assert result.uncertainty_source == "residual_covariance_invalid_sem"
    assert result.n_points == len(times_ns)


def test_cpmg_fit_ignores_sem_for_an_excluded_signal_point() -> None:
    """SEM at a nonfinite signal point should not disable weighting elsewhere."""
    times_ns = np.linspace(2_000.0, 80_000.0, 30)
    values = 0.45 * np.exp(-times_ns / 30_000.0) + 0.48
    sem = np.full(len(times_ns), 0.01)
    values[4] = np.nan
    sem[4] = np.nan

    result = _fit_cpmg_decay(
        times_ns,
        values,
        sem_values=sem,
        min_fit_points=6,
    )

    assert result.fit_method_used == "weighted_least_squares"
    assert result.uncertainty_source == "measurement_sem"
    assert result.n_points == len(times_ns) - 1


def test_cpmg_fit_rejects_too_few_finite_points() -> None:
    """A curve with too few finite points should return an explicit fit status."""
    result = _fit_cpmg_decay(
        [1_000.0, 2_000.0, 3_000.0, 4_000.0],
        [0.8, np.nan, 0.6, 0.5],
        min_fit_points=4,
    )

    assert result.fit_status == "insufficient_points"
    assert np.isnan(result.gamma2_per_ns)


class _FakeMeasurementService:
    def __init__(self) -> None:
        self.orders: list[float] = []

    def sweep_parameter(self, **kwargs: object) -> SimpleNamespace:
        sequence = kwargs["sequence"]
        sweep_range = np.asarray(kwargs["sweep_range"], dtype=np.int64)
        first_schedule = sequence(int(sweep_range[0]))  # type: ignore[operator]
        target = first_schedule.labels[0]
        endpoint_duration = 20.0
        spacing = (first_schedule.duration - endpoint_duration) / sweep_range[0]
        self.orders.append(float(1.0 / (2.0 * spacing)))
        times = sweep_range.astype(np.float64) * spacing
        probability = 0.42 * np.exp(-times / 40_000.0) + 0.5
        return SimpleNamespace(
            data={target: SimpleNamespace(normalized=2.0 * probability - 1.0)}
        )


class _FakeExperiment(_Experiment):
    def __init__(self) -> None:
        super().__init__()
        self.measurement_service = _FakeMeasurementService()
        self.ctx = SimpleNamespace(
            qubit_labels=["Q00"],
            measurement=SimpleNamespace(sampling_period=0.4),
        )


def test_experiment_randomizes_acquisition_but_stores_fixed_frequency_indices() -> None:
    """Seeded acquisition should be reproducible while matrices retain requested order."""
    kwargs = {
        "targets": ["Q00"],
        "frequency_range": [0.0005, 0.001, 0.002, 0.003],
        "time_range": _TEST_TIME_RANGE_NS,
        "random_seed": 1234,
        "plot": False,
        "save_data": False,
        "enable_tqdm": False,
    }
    first = cpmg_noise_spectroscopy(_FakeExperiment(), **kwargs)  # type: ignore[arg-type]
    second = cpmg_noise_spectroscopy(_FakeExperiment(), **kwargs)  # type: ignore[arg-type]

    assert first.figure is None
    assert first.figures is None
    assert first.data["acquisition_orders"] == second.data["acquisition_orders"]
    assert first.data["acquisition_orders"][0][0] != [0, 1, 2, 3]
    matrix = np.asarray(first.data["gamma2_matrix"])
    assert matrix.shape == (1, 1, 4)
    assert np.all(np.isfinite(matrix))


def test_public_experiment_uses_time_based_defaults() -> None:
    """Omitted ranges should use the configured frequency and geometric time grids."""
    result = cpmg_noise_spectroscopy(
        _FakeExperiment(),  # type: ignore[arg-type]
        targets=["Q00"],
        plot=False,
        save_data=False,
        enable_tqdm=False,
    )

    np.testing.assert_allclose(
        result.data["frequency_range"],
        np.linspace(0.0005, 0.004, 20),
    )
    realized_time_range = result.data["time_range"]
    assert realized_time_range is not None
    np.testing.assert_allclose(
        realized_time_range,
        np.geomspace(200.0, 200_000.0, 31),
    )
    assert "requested_frequency_range" not in result.data
    assert "n_pi_range" not in result.data
    assert result.data["metadata"]["min_pi_pulses"] == 4
    assert result.data["metadata"]["max_pulse_duty_ratio"] == 0.25
    assert all("P_sem_values" not in record for record in result.data["fit_records"])
    assert all(min(record["n_pi_values"]) >= 4 for record in result.data["fit_records"])


@pytest.mark.parametrize(
    ("pattern", "period"),
    [("+--+", 4), ("+-+-", 2)],
)
def test_time_based_defaults_keep_complete_sign_cycles(
    pattern: str,
    period: int,
) -> None:
    """Time-derived pulse counts should retain only complete sign-pattern cycles."""
    result = cpmg_noise_spectroscopy(
        _FakeExperiment(),  # type: ignore[arg-type]
        targets=["Q00"],
        frequency_range=[0.001],
        pi_pulse_pattern=pattern,  # type: ignore[arg-type]
        plot=False,
        save_data=False,
        enable_tqdm=False,
    )

    counts = cast(list[int], result.data["fit_records"][0]["n_pi_values"])
    assert len(counts) >= 6
    assert all(count % period == 0 for count in counts)


def test_public_experiment_rejects_zero_frequency() -> None:
    """The public workflow should reject the removed zero-frequency Ramsey path."""
    with pytest.raises(ValueError, match="positive"):
        cpmg_noise_spectroscopy(
            _FakeExperiment(),  # type: ignore[arg-type]
            targets=["Q00"],
            frequency_range=[0.0],
            plot=False,
            save_data=False,
            enable_tqdm=False,
        )


def test_numerical_only_experiment_does_not_create_plotly_figures(monkeypatch) -> None:
    """Disabling every plot output should avoid the Plotly figure factory entirely."""

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("figure creation is disabled")

    monkeypatch.setattr(viz, "make_figure", fail_if_called)

    result = cpmg_noise_spectroscopy(
        _FakeExperiment(),  # type: ignore[arg-type]
        targets=["Q00"],
        frequency_range=[0.001],
        time_range=_TEST_TIME_RANGE_NS,
        plot=False,
        save_image=False,
        save_fit_images=False,
        save_data=False,
        enable_tqdm=False,
    )

    assert result.figure is None
    assert result.figures is None


def test_plotting_experiment_returns_prefit_heatmap(monkeypatch) -> None:
    """The raw-data heatmap should use exact measured times and a fixed color scale."""
    monkeypatch.setattr(go.Figure, "show", lambda _self: None)

    result = cpmg_noise_spectroscopy(
        _FakeExperiment(),  # type: ignore[arg-type]
        targets=["Q00"],
        frequency_range=[0.001, 0.002],
        time_range=_TEST_TIME_RANGE_NS,
        n_repeats=2,
        plot=True,
        show_t1=False,
        save_data=False,
        enable_tqdm=False,
    )

    assert result.figures is not None
    heatmap = result.figures["Q00_raw_data_heatmap"]
    assert isinstance(heatmap, go.Figure)
    heatmap_data = cast(tuple[object, ...], heatmap.data)
    assert len(heatmap_data) == 1
    trace = cast(go.Scatter, heatmap_data[0])
    assert trace.type == "scatter"
    expected_coordinates = {
        (float(record["f_real_GHz"]) * 1e3, float(time_ns) * 1e-3)
        for record in result.data["fit_records"]
        for time_ns in record["actual_decay_times_ns"]
    }
    trace_x = np.asarray(cast(ArrayLike, trace.x), dtype=np.float64)
    trace_y = np.asarray(cast(ArrayLike, trace.y), dtype=np.float64)
    actual_coordinates = set(zip(trace_x, trace_y, strict=True))
    assert actual_coordinates == expected_coordinates
    trace_data = cast(dict[str, Any], trace.to_plotly_json())
    assert trace_data["marker"]["cmin"] == 0
    assert trace_data["marker"]["cmax"] == 1
    assert heatmap.layout.xaxis.rangemode is None
    assert heatmap.layout.xaxis.title.text == "Effective CPMG Frequency (MHz)"
    assert heatmap.layout.yaxis.title.text == "Effective decay time (us)"


def test_ascending_acquisition_sorts_realized_frequency_not_input_index() -> None:
    """Ascending acquisition should sort valid realized frequencies without moving result columns."""
    result = cpmg_noise_spectroscopy(
        _FakeExperiment(),  # type: ignore[arg-type]
        targets=["Q00"],
        frequency_range=[0.003, 0.001, 0.002],
        time_range=_TEST_TIME_RANGE_NS,
        frequency_order="ascending",
        plot=False,
        save_data=False,
        enable_tqdm=False,
    )

    assert result.data["acquisition_orders"] == [[[1, 2, 0]]]
    frequencies = np.asarray(result.data["realized_frequency_matrix"])[0]
    assert frequencies[0] > frequencies[1]
    assert np.all(np.isfinite(np.asarray(result.data["gamma2_matrix"])))


def test_schema_v2_round_trip_preserves_curves_and_frequency_axes(tmp_path) -> None:
    """Saving and loading schema v2 should preserve detailed fits and target axes."""
    result = cpmg_noise_spectroscopy(
        _FakeExperiment(),  # type: ignore[arg-type]
        targets=["Q00"],
        frequency_range=[0.001],
        time_range=_TEST_TIME_RANGE_NS,
        frequency_order="ascending",
        plot=False,
        save_data=False,
        enable_tqdm=False,
    )
    payload = result.data

    _save_payload(payload=payload, save_dir=tmp_path)  # type: ignore[arg-type]
    loaded = _load_payload(tmp_path)

    assert loaded["schema_version"] == 2
    assert loaded["fit_records"][0]["n_pi_values"] == [16, 20, 24, 28, 32, 40]
    np.testing.assert_allclose(
        loaded["realized_frequency_matrix"],
        payload["realized_frequency_matrix"],
    )
    np.testing.assert_allclose(loaded["gamma2_matrix"], payload["gamma2_matrix"])
    fit_record = loaded["fit_records"][0]
    times_ns = np.asarray(fit_record["actual_decay_times_ns"], dtype=np.float64)
    values = np.asarray(fit_record["P_values"], dtype=np.float64)
    fit = _fit_cpmg_decay(
        times_ns,
        values,
        min_fit_points=6,
    )
    figure = _plot_cpmg_decay(
        times_ns=times_ns,
        values=values,
        fit=fit,
        target="Q00",
        frequency_ghz=cast(float, fit_record["f_real_GHz"]),
        xaxis_type="log",
    )
    figure_data = cast(list[dict[str, Any]], figure.to_dict()["data"])
    assert len(figure_data) == 2


def test_schema_v1_loader_derives_gamma_and_common_frequency_axis(tmp_path) -> None:
    """A legacy save directory should remain loadable with derived schema-v2 fields."""
    params = {
        "timestamp": "20260609_000000",
        "targets": ["Q00"],
        "n_repeats": 1,
        "frequency_range": [0.0, 0.001],
        "half_tau_range": [0.0, 250.0],
        "time_range": [1000.0, 2000.0],
    }
    (tmp_path / "params.json").write_text(json.dumps(params), encoding="utf-8")
    np.save(tmp_path / "t2_matrix.npy", np.asarray([[[1000.0, 2000.0]]]))
    np.save(tmp_path / "t2_error_matrix.npy", np.asarray([[[100.0, 400.0]]]))

    loaded = _load_payload(tmp_path)

    assert loaded["schema_version"] == 1
    np.testing.assert_allclose(
        loaded["realized_frequency_matrix"],
        [[0.0, 0.001]],
    )
    np.testing.assert_allclose(loaded["gamma2_matrix"], [[[0.001, 0.0005]]])
    np.testing.assert_allclose(
        loaded["gamma2_error_matrix"],
        [[[0.0001, 0.0001]]],
    )


def test_schema_loader_rejects_malformed_fit_records(tmp_path) -> None:
    """A malformed fit-results document should fail with a clear validation error."""
    params = {
        "timestamp": "20260609_000000",
        "targets": ["Q00"],
        "n_repeats": 1,
        "frequency_range": [0.001],
        "half_tau_range": [250.0],
    }
    (tmp_path / "params.json").write_text(json.dumps(params), encoding="utf-8")
    (tmp_path / "fit_results.json").write_text(
        json.dumps({"not": "a list"}),
        encoding="utf-8",
    )
    np.save(tmp_path / "t2_matrix.npy", np.asarray([[[2000.0]]]))

    with pytest.raises(ValueError, match="list of objects"):
        _load_payload(tmp_path)
