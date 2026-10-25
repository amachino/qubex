"""Tests for `qubex.contrib.experiment.quantum_efficiency_measurement` analysis helpers."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest
from qxpulse import Blank

from qubex.contrib.experiment.quantum_efficiency_measurement import (
    _fit_ramsey_fringe,
    _fit_snr_sweep,
    _measure_ramsey_fringe,
    analyze_quantum_efficiency,
    compute_readout_snr,
    measurement_induced_dephasing,
    measurement_induced_dephasing_experiment,
    quantum_efficiency_measurement,
    readout_snr,
    sweep_readout_snr,
)
from qubex.experiment import Experiment
from qubex.experiment.models import Result


def _build_analysis_inputs() -> dict[str, Any]:
    """Build one synthetic dataset with known SNR and dephasing parameters."""
    amplitudes = np.linspace(0.01, 0.20, 10)
    phases = np.linspace(0.0, 4.0 * np.pi, 41)
    expected_sigma_m = 0.05
    expected_slope = 10.0
    b = 0.45
    probability_rows = []
    ground_raw = []
    excited_raw = []

    for amplitude in amplitudes:
        rho01 = b * np.exp(-(amplitude**2) / (2.0 * expected_sigma_m**2))
        sigma_z = 2.0 * rho01 * np.cos(phases + 0.1)
        probability_rows.append(0.5 * (1.0 - sigma_z))

        projected_ground = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        projected_excited = projected_ground + expected_slope * amplitude
        ground_raw.append(projected_ground.astype(np.complex128))
        excited_raw.append(projected_excited.astype(np.complex128))

    return {
        "amplitudes": amplitudes,
        "phases": phases,
        "ramsey_probabilities": np.asarray(probability_rows, dtype=np.float64),
        "ground_raw": ground_raw,
        "excited_raw": excited_raw,
        "expected_sigma_m": expected_sigma_m,
        "expected_slope": expected_slope,
    }


def _error_y_array(figure: go.Figure, index: int = 0) -> Any:
    """Return one trace's error-bar array without relying on Plotly stub unions."""
    trace = cast(go.Scatter, figure.data[index])
    payload = cast(dict[str, Any], trace.to_plotly_json())
    error_y = cast(dict[str, Any], payload["error_y"])
    return error_y["array"]


class _RecordingProgress:
    """Record progress operations without rendering a terminal bar."""

    def __init__(self, **kwargs: object) -> None:
        self.initial_total = int(cast(int, kwargs["total"]))
        self.total = self.initial_total
        self.desc = kwargs["desc"]
        self.unit = kwargs["unit"]
        self.disable = kwargs["disable"]
        self.n = 0
        self.refresh_count = 0
        self.closed = False

    def update(self, increment: int = 1) -> None:
        self.n += increment

    def refresh(self) -> None:
        self.refresh_count += 1

    def close(self) -> None:
        self.closed = True


def test_analyze_quantum_efficiency_should_recover_fit_based_eta() -> None:
    """Given synthetic amplitude data, when analyzed, then the fit-based QE matches the known model."""
    dataset = _build_analysis_inputs()
    result = analyze_quantum_efficiency(
        exp=object(),
        target="Q00",
        readout_amplitudes=dataset["amplitudes"],
        ramsey_phases=dataset["phases"],
        ramsey_excited_probabilities=dataset["ramsey_probabilities"],
        ground_state_raw=dataset["ground_raw"],
        excited_state_raw=dataset["excited_raw"],
        plot=False,
    )

    expected_eta = (
        0.5 * dataset["expected_slope"] ** 2 * dataset["expected_sigma_m"] ** 2
    )
    np.testing.assert_allclose(result["quantum_efficiency"], expected_eta, rtol=5e-2)
    assert result["snr_fit"]["a"] == pytest.approx(dataset["expected_slope"], rel=5e-2)
    assert result["snr_fit"]["a_err"] > 0.0
    assert np.all(np.asarray(result["snr"]["standard_error"]) > 0.0)
    assert result["quantum_efficiency_error"] > 0.0
    assert result["snr_fit"]["fit_amplitudes"][0] == pytest.approx(0.0, abs=1e-12)
    assert result["snr_fit"]["y_fit"][0] == pytest.approx(0.0, abs=1e-12)
    assert result["measurement_induced_dephasing_fit"]["sigma_m"] == pytest.approx(
        dataset["expected_sigma_m"], rel=5e-2
    )
    assert result["measurement_induced_dephasing_fit"]["weighted"] is True
    assert result["measurement_induced_dephasing_fit"]["sigma_m_err"] > 0.0
    assert np.all(
        np.asarray(result["measurement_induced_dephasing"]["rho01_standard_error"])
        > 0.0
    )
    assert result["measurement_induced_dephasing_fit"]["fit_amplitudes"][
        0
    ] == pytest.approx(0.0, abs=1e-12)
    assert isinstance(result.get_figure(), go.Figure)
    assert isinstance(result.get_figure("snr"), go.Figure)
    assert isinstance(result.get_figure("dephasing"), go.Figure)
    snr_figure = cast(go.Figure, result.get_figure("snr"))
    dephasing_figure = cast(go.Figure, result.get_figure("dephasing"))
    assert _error_y_array(snr_figure) is not None
    assert _error_y_array(dephasing_figure) is not None
    sigma_m_annotation = result.get_figure("dephasing").layout.annotations[0]
    assert sigma_m_annotation.text is not None
    assert "±" in sigma_m_annotation.text

    overview = result.get_figure()
    assert overview.layout.xaxis.range is not None
    assert overview.layout.xaxis.range[0] == pytest.approx(0.0)
    assert overview.layout.yaxis.range is not None
    assert overview.layout.yaxis2.range is not None
    assert overview.layout.yaxis.range[0] == pytest.approx(0.0)
    assert overview.layout.yaxis2.range[0] == pytest.approx(0.0)
    assert overview.layout.yaxis.range[1] > np.max(
        np.asarray(result["measurement_induced_dephasing"]["rho01"])
        + np.asarray(result["measurement_induced_dephasing"]["rho01_standard_error"])
    )
    assert overview.layout.yaxis2.range[1] > np.max(
        np.asarray(result["snr"]["values"])
        + np.asarray(result["snr"]["standard_error"])
    )
    assert overview.layout.legend.orientation == "h"
    assert overview.layout.legend.y is not None
    assert overview.layout.legend.y < 0.0
    assert overview.layout.shapes
    assert overview.layout.shapes[0].y0 == pytest.approx(0.0)
    assert overview.layout.shapes[0].y1 == pytest.approx(0.0)
    assert _error_y_array(overview) is not None
    assert _error_y_array(overview, 2) is not None
    eta_annotation = next(
        annotation
        for annotation in overview.layout.annotations
        if annotation.text is not None and "η" in annotation.text
    )
    assert eta_annotation.y is not None
    assert eta_annotation.y > 1.0


def test_compute_readout_snr_should_support_waveform_raw_iq() -> None:
    """Given waveform IQ shots, when computing SNR, then the projected SNR is positive and stable."""
    ground = np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j],
            [0.1 + 0.0j, 1.1 + 0.0j],
            [-0.1 + 0.0j, 0.9 + 0.0j],
        ],
        dtype=np.complex128,
    )
    excited = np.array(
        [
            [1.0 + 0.0j, 2.0 + 0.0j],
            [1.1 + 0.0j, 2.1 + 0.0j],
            [0.9 + 0.0j, 1.9 + 0.0j],
        ],
        dtype=np.complex128,
    )

    summary = compute_readout_snr(ground, excited)

    assert summary["snr"] > 0
    assert summary["snr_standard_error"] > 0
    assert summary["snr_uncertainty_method"] == "delta_fallback"
    assert np.asarray(summary["projected_ground"]).shape == (3,)
    assert np.asarray(summary["projected_excited"]).shape == (3,)


def test_compute_readout_snr_should_use_explicit_projection_weights() -> None:
    """Explicit weights should fix the projection independently of state means."""
    ground = np.array([-1.0, 0.0, 1.0], dtype=np.complex128)
    excited = ground + 1.0j

    summary = compute_readout_snr(
        ground,
        excited,
        projection_weights=1.0 + 0.0j,
    )

    assert complex(summary["weights"]) == pytest.approx(1.0 + 0.0j)
    np.testing.assert_allclose(summary["projected_ground"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(summary["projected_excited"], [-1.0, 0.0, 1.0])


def test_compute_readout_snr_should_propagate_fit_covariance() -> None:
    """A resolved coupled fit should provide a model-based SNR error bar."""
    random = np.random.default_rng(0)
    ground = random.normal(0.0, 1.0, 256).astype(np.complex128)
    excited = random.normal(2.0, 1.0, 256).astype(np.complex128)

    summary = compute_readout_snr(ground, excited)

    assert summary["snr_uncertainty_method"] == "fit_covariance"
    assert 0.0 < summary["snr_standard_error"] < summary["snr"]


def test_fit_snr_sweep_should_use_inverse_variance_weights() -> None:
    """The origin-constrained slope should use the supplied absolute errors."""
    amplitudes = np.array([1.0, 2.0, 3.0])
    snr = np.array([1.0, 4.0, 3.0])
    standard_errors = np.array([1.0, 0.1, 1.0])
    weights = 1.0 / standard_errors**2
    expected_slope = np.sum(weights * amplitudes * snr) / np.sum(
        weights * amplitudes**2
    )

    result = _fit_snr_sweep(
        target="Q00",
        readout_amplitudes=amplitudes,
        snr_values=snr,
        snr_standard_errors=standard_errors,
    )

    assert result["a"] == pytest.approx(expected_slope)
    assert result["a_err"] == pytest.approx(
        1.0 / np.sqrt(np.sum(weights * amplitudes**2))
    )
    np.testing.assert_allclose(
        _error_y_array(cast(go.Figure, result.get_figure())),
        standard_errors,
    )


@pytest.mark.parametrize("zero_tolerance", [0.0, -1.0, np.nan, np.inf])
def test_compute_readout_snr_should_reject_invalid_zero_tolerance(
    zero_tolerance: float,
) -> None:
    """The numerical threshold must not permit a zero-axis division."""
    ground = np.array([-1.0, 0.0, 1.0], dtype=np.complex128)

    with pytest.raises(ValueError, match="positive and finite"):
        compute_readout_snr(
            ground,
            ground + 1.0,
            zero_tolerance=zero_tolerance,
        )


def test_analyze_quantum_efficiency_should_share_max_amplitude_projection() -> None:
    """The default sweep analysis should reuse the largest-amplitude IQ axis."""
    dataset = _build_analysis_inputs()
    ground = np.array([-1.0, 0.0, 1.0], dtype=np.complex128)
    dataset["ground_raw"][0] = ground
    dataset["excited_raw"][0] = ground + 1.0j

    shared = analyze_quantum_efficiency(
        exp=object(),
        target="Q00",
        readout_amplitudes=dataset["amplitudes"],
        ramsey_phases=dataset["phases"],
        ramsey_excited_probabilities=dataset["ramsey_probabilities"],
        ground_state_raw=dataset["ground_raw"],
        excited_state_raw=dataset["excited_raw"],
        plot=False,
    )
    per_amplitude = analyze_quantum_efficiency(
        exp=object(),
        target="Q00",
        readout_amplitudes=dataset["amplitudes"],
        ramsey_phases=dataset["phases"],
        ramsey_excited_probabilities=dataset["ramsey_probabilities"],
        ground_state_raw=dataset["ground_raw"],
        excited_state_raw=dataset["excited_raw"],
        projection_method="per_amplitude",
        plot=False,
    )

    assert shared["projection_method"] == "max_amplitude"
    assert shared["projection_reference_amplitude"] == pytest.approx(
        np.max(dataset["amplitudes"])
    )
    for summary in shared["snr"]["projection"].values():
        assert complex(summary["weights"]) == pytest.approx(1.0 + 0.0j)
    first_amplitude = float(dataset["amplitudes"][0])
    assert complex(
        per_amplitude["snr"]["projection"][first_amplitude]["weights"]
    ) == pytest.approx(0.0 + 1.0j)


def test_analyze_quantum_efficiency_should_reject_invalid_projection_method() -> None:
    """An unknown projection strategy should fail before fitting."""
    dataset = _build_analysis_inputs()

    with pytest.raises(ValueError, match="projection_method"):
        analyze_quantum_efficiency(
            exp=object(),
            target="Q00",
            readout_amplitudes=dataset["amplitudes"],
            ramsey_phases=dataset["phases"],
            ramsey_excited_probabilities=dataset["ramsey_probabilities"],
            ground_state_raw=dataset["ground_raw"],
            excited_state_raw=dataset["excited_raw"],
            projection_method=cast(Any, "invalid"),
            plot=False,
        )


def test_analyze_quantum_efficiency_should_reject_zero_reference_axis() -> None:
    """A largest-amplitude distribution with no state separation cannot set an axis."""
    dataset = _build_analysis_inputs()
    dataset["excited_raw"][-1] = np.asarray(
        dataset["ground_raw"][-1],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="nonzero norm"):
        analyze_quantum_efficiency(
            exp=object(),
            target="Q00",
            readout_amplitudes=dataset["amplitudes"],
            ramsey_phases=dataset["phases"],
            ramsey_excited_probabilities=dataset["ramsey_probabilities"],
            ground_state_raw=dataset["ground_raw"],
            excited_state_raw=dataset["excited_raw"],
            plot=False,
        )


def test_measure_ramsey_fringe_should_use_final_capture_from_one_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One amplitude should use the final capture from one phase sweep."""
    phases = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    expected_rho01 = 0.3
    calls: dict[str, object] = {"count": 0, "validated_targets": None}

    async def _run_sweep_measurement(
        *,
        schedule: Any,
        sweep_values: np.ndarray,
        **options: object,
    ) -> object:
        calls["count"] = cast(int, calls["count"]) + 1
        calls["sweep_range"] = np.asarray(sweep_values, dtype=np.float64)
        calls["options"] = options
        sigma_z = 2.0 * expected_rho01 * np.cos(sweep_values + 0.2)
        results = []
        for phase, final_iq in zip(sweep_values, sigma_z, strict=True):
            schedule(float(phase))
            results.append(
                SimpleNamespace(
                    data={
                        "Q00": [
                            SimpleNamespace(kerneled=np.asarray(0.125 + 0.0j)),
                            SimpleNamespace(kerneled=np.asarray(complex(final_iq))),
                        ]
                    }
                )
            )
        return SimpleNamespace(results=results)

    class _Bridge:
        def run(self, factory: Any, **_options: object) -> object:
            return asyncio.run(factory())

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(
        qe_module,
        "get_shared_async_bridge",
        lambda *, key: _Bridge(),
        raising=False,
    )

    experiment = cast(
        Experiment,
        SimpleNamespace(
            ctx=SimpleNamespace(
                resolve_qubit_label=lambda target: target,
                resolve_read_label=lambda target: target,
            ),
            pulse=SimpleNamespace(
                x90=lambda _target: Blank(8),
                readout=lambda _target, **_kwargs: Blank(8),
                rabi_params={
                    "Q00": SimpleNamespace(
                        normalize=lambda values: np.real(np.asarray(values))
                    )
                },
                validate_rabi_params=lambda targets: calls.update(
                    validated_targets=list(targets)
                ),
            ),
            measurement_service=SimpleNamespace(
                run_sweep_measurement=_run_sweep_measurement
            ),
        ),
    )

    summary = _measure_ramsey_fringe(
        experiment,
        "Q00",
        readout_amplitude=0.1,
        phase_range=phases,
        n_shots=1024,
        shot_interval=100_000.0,
    )

    assert calls["count"] == 1
    assert calls["validated_targets"] == ["Q00"]
    np.testing.assert_allclose(cast(Any, calls["sweep_range"]), phases)
    options = cast(dict[str, object], calls["options"])
    assert options["shot_averaging"] is True
    assert options["time_integration"] is True
    assert options["state_classification"] is False
    assert options["final_measurement"] is True
    assert summary["rho01"] == pytest.approx(expected_rho01)


def test_measure_ramsey_fringe_should_reject_invalid_phases_before_sweep() -> None:
    """A rank-deficient phase range should fail before hardware acquisition."""
    experiment = cast(
        Experiment,
        SimpleNamespace(
            ctx=SimpleNamespace(
                resolve_qubit_label=lambda target: target,
                resolve_read_label=lambda target: target,
            ),
            pulse=SimpleNamespace(
                validate_rabi_params=lambda _targets: None,
            ),
            measurement_service=SimpleNamespace(
                run_sweep_measurement=lambda **_kwargs: pytest.fail(
                    "sweep should not run for invalid phases"
                )
            ),
        ),
    )

    with pytest.raises(ValueError, match="independent phase points"):
        _measure_ramsey_fringe(
            experiment,
            "Q00",
            readout_amplitude=0.1,
            phase_range=np.array([0.0, 0.0, np.pi]),
            n_shots=1024,
            shot_interval=100_000.0,
        )


def test_fit_ramsey_fringe_should_reject_rank_deficient_phases() -> None:
    """A three-parameter fringe fit requires three independent phase rows."""
    with pytest.raises(ValueError, match="independent phase points"):
        _fit_ramsey_fringe([0.0, 0.0, np.pi], [0.2, 0.2, 0.8])


def test_fit_ramsey_fringe_should_estimate_coherence_error_from_residuals() -> None:
    """Ramsey residuals should produce a finite propagated rho01 error."""
    phases = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    sigma_z = 0.6 * np.cos(phases + 0.2)
    residual_pattern = np.array(
        [1, -1, 0, 1, 0, -1, 1, 0, -1, 1, -1, 0, 1, 0, -1, 1, -1],
        dtype=np.float64,
    )

    low_noise = _fit_ramsey_fringe(
        phases,
        0.5 * (1.0 - sigma_z - 0.002 * residual_pattern),
    )
    high_noise = _fit_ramsey_fringe(
        phases,
        0.5 * (1.0 - sigma_z - 0.010 * residual_pattern),
    )

    assert low_noise["rho01"] == pytest.approx(0.3, abs=2e-3)
    assert low_noise["rho01_standard_error"] > 0.0
    assert high_noise["rho01_standard_error"] > low_noise["rho01_standard_error"]
    assert low_noise["coefficient_covariance"].shape == (3, 3)
    assert low_noise["degrees_of_freedom"] == len(phases) - 3


def test_analyze_quantum_efficiency_should_reject_shape_mismatch() -> None:
    """Given mismatched Ramsey shapes, when analyzing, then a clear ValueError is raised."""
    dataset = _build_analysis_inputs()

    with pytest.raises(ValueError, match="ramsey_excited_probabilities"):
        analyze_quantum_efficiency(
            exp=object(),
            target="Q00",
            readout_amplitudes=dataset["amplitudes"],
            ramsey_phases=dataset["phases"],
            ramsey_excited_probabilities=dataset["ramsey_probabilities"][:-1],
            ground_state_raw=dataset["ground_raw"],
            excited_state_raw=dataset["excited_raw"],
            plot=False,
        )


def test_analyze_quantum_efficiency_should_reject_invalid_raw_shape() -> None:
    """Given invalid raw IQ arrays, when analyzing, then a clear ValueError is raised."""
    dataset = _build_analysis_inputs()
    invalid_ground = [np.zeros((2, 2, 2), dtype=np.float64)] * len(
        dataset["amplitudes"]
    )

    with pytest.raises(ValueError, match="ground_state_raw"):
        analyze_quantum_efficiency(
            exp=object(),
            target="Q00",
            readout_amplitudes=dataset["amplitudes"],
            ramsey_phases=dataset["phases"],
            ramsey_excited_probabilities=dataset["ramsey_probabilities"],
            ground_state_raw=invalid_ground,
            excited_state_raw=dataset["excited_raw"],
            plot=False,
        )


def test_analyze_quantum_efficiency_should_reject_scalar_raw_series() -> None:
    """A scalar raw payload should produce a clear validation error."""
    dataset = _build_analysis_inputs()
    with pytest.raises(ValueError, match="one entry for each readout amplitude"):
        analyze_quantum_efficiency(
            exp=object(),
            target="Q00",
            readout_amplitudes=dataset["amplitudes"],
            ramsey_phases=dataset["phases"],
            ramsey_excited_probabilities=dataset["ramsey_probabilities"],
            ground_state_raw=np.asarray(1.0 + 1.0j),
            excited_state_raw=dataset["excited_raw"],
            plot=False,
        )


def test_analyze_quantum_efficiency_should_accept_ragged_object_raw_series() -> None:
    """Object arrays should support a different shot count at each amplitude."""
    dataset = _build_analysis_inputs()
    n_amplitudes = len(dataset["amplitudes"])
    ground_raw = np.empty(n_amplitudes, dtype=object)
    excited_raw = np.empty(n_amplitudes, dtype=object)
    for index, (ground, excited) in enumerate(
        zip(dataset["ground_raw"], dataset["excited_raw"], strict=True)
    ):
        ground_raw[index] = np.resize(ground, len(ground) + index)
        excited_raw[index] = np.resize(excited, len(excited) + index)

    result = analyze_quantum_efficiency(
        exp=object(),
        target="Q00",
        readout_amplitudes=dataset["amplitudes"],
        ramsey_phases=dataset["phases"],
        ramsey_excited_probabilities=dataset["ramsey_probabilities"],
        ground_state_raw=ground_raw,
        excited_state_raw=excited_raw,
        plot=False,
    )

    assert np.isfinite(result["quantum_efficiency"])


def test_public_experiment_helpers_should_run_with_stubbed_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a stub experiment, when calling the public helpers, then each returns a Result with figures."""
    dataset = _build_analysis_inputs()
    amplitudes = dataset["amplitudes"]
    phases = dataset["phases"]
    probability_map = {
        float(amplitude): np.asarray(
            dataset["ramsey_probabilities"][index], dtype=np.float64
        )
        for index, amplitude in enumerate(amplitudes)
    }
    probability_map[0.0] = 0.5 * (1.0 - 2.0 * 0.45 * np.cos(phases + 0.1))
    raw_map = {
        float(amplitude): (
            np.asarray(dataset["ground_raw"][index], dtype=np.complex128),
            np.asarray(dataset["excited_raw"][index], dtype=np.complex128),
        )
        for index, amplitude in enumerate(amplitudes)
    }

    def _fake_measure_ramsey_fringe(
        _exp: Experiment,
        _target: str,
        *,
        readout_amplitude: float,
        phase_range: np.ndarray,
        n_shots: int | None,
        shot_interval: float | None,
    ) -> dict[str, object]:
        del n_shots, shot_interval
        phase_index = np.array(
            [int(np.argmin(np.abs(phases - phase))) for phase in phase_range],
            dtype=np.int64,
        )
        probabilities = probability_map[float(readout_amplitude)][phase_index]
        return dict(_fit_ramsey_fringe(phase_range, probabilities))

    def _fake_measure_readout_snr(
        _exp: Experiment,
        _target: str,
        *,
        readout_amplitude: float,
        n_shots: int,
        shot_interval: float | None,
        readout_duration: float | None,
    ) -> Any:
        del n_shots, shot_interval, readout_duration
        ground_raw, excited_raw = raw_map[float(readout_amplitude)]
        return compute_readout_snr(ground_raw, excited_raw)

    def _fake_measure_readout_iq(
        _exp: Experiment,
        _target: str,
        *,
        readout_amplitude: float,
        n_shots: int,
        shot_interval: float | None,
        readout_duration: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del n_shots, shot_interval, readout_duration
        return raw_map[float(readout_amplitude)]

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(
        qe_module, "_measure_ramsey_fringe", _fake_measure_ramsey_fringe
    )
    monkeypatch.setattr(qe_module, "_measure_readout_snr", _fake_measure_readout_snr)
    monkeypatch.setattr(qe_module, "_measure_readout_iq", _fake_measure_readout_iq)

    experiment = cast(Experiment, object())
    single_dephasing_result = measurement_induced_dephasing(
        experiment,
        "Q00",
        readout_amplitude=float(amplitudes[3]),
        use_reference=True,
        phase_range=phases,
        plot=False,
    )
    dephasing_result = measurement_induced_dephasing_experiment(
        experiment,
        "Q00",
        amplitude_range=amplitudes,
        phase_range=phases,
        plot=False,
    )
    single_snr_result = readout_snr(
        experiment,
        "Q00",
        readout_amplitude=float(amplitudes[3]),
        n_shots=1024,
        plot=False,
    )
    snr_result = sweep_readout_snr(
        experiment,
        "Q00",
        amplitude_range=amplitudes,
        n_shots=1024,
        plot=False,
    )
    qe_result = quantum_efficiency_measurement(
        experiment,
        "Q00",
        amplitude_range=amplitudes,
        phase_range=phases,
        n_shots=1024,
        plot=False,
    )

    assert single_dephasing_result["measurement_induced_dephasing"] >= 0
    expected_beta_m_error = np.hypot(
        single_dephasing_result["rho01_standard_error"]
        / single_dephasing_result["rho01"],
        single_dephasing_result["reference_rho01_standard_error"]
        / single_dephasing_result["reference_rho01"],
    )
    assert single_dephasing_result["beta_m_standard_error"] == pytest.approx(
        expected_beta_m_error
    )
    assert single_dephasing_result[
        "measurement_induced_dephasing_standard_error"
    ] == pytest.approx(expected_beta_m_error)
    assert isinstance(single_dephasing_result.get_figure(), go.Figure)
    beta_m_annotation = single_dephasing_result.get_figure().layout.annotations[0]
    assert beta_m_annotation.text is not None
    assert "±" in beta_m_annotation.text
    assert isinstance(dephasing_result.get_figure(), go.Figure)
    assert single_snr_result["snr"] > 0
    assert single_snr_result["snr_standard_error"] > 0
    assert isinstance(single_snr_result.get_figure(), go.Figure)
    assert isinstance(snr_result.get_figure(), go.Figure)
    assert np.all(np.asarray(snr_result["snr_standard_error"]) > 0.0)
    assert _error_y_array(cast(go.Figure, snr_result.get_figure())) is not None
    assert isinstance(qe_result.get_figure(), go.Figure)
    assert qe_result["quantum_efficiency"] > 0
    assert qe_result["quantum_efficiency_error"] > 0
    assert qe_result["early_stop_enabled"] is False
    assert qe_result["early_stopped"] is False


def test_quantum_efficiency_measurement_should_resolve_default_sweep_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ranges should use 17 phases and the target readout amplitude."""
    captured: dict[str, object] = {}

    def _fake_quantum_efficiency_measurement(
        _exp: Experiment,
        target: str,
        **options: object,
    ) -> Result:
        captured["target"] = target
        captured.update(options)
        return Result(data={"target": target})

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(
        qe_module,
        "_quantum_efficiency_measurement",
        _fake_quantum_efficiency_measurement,
    )
    experiment = cast(
        Experiment,
        SimpleNamespace(
            ctx=SimpleNamespace(
                resolve_qubit_label=lambda target: target,
                params=SimpleNamespace(get_readout_amplitude=lambda target: 0.2),
            )
        ),
    )

    quantum_efficiency_measurement(experiment, "Q00", plot=False)

    np.testing.assert_allclose(
        cast(Any, captured["phase_range"]),
        np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False),
    )
    np.testing.assert_allclose(
        cast(Any, captured["amplitude_range"]),
        np.geomspace(0.002, 0.2, 17),
    )
    assert captured["projection_method"] == "max_amplitude"
    assert captured["early_stop"] is False
    assert captured["enable_tqdm"] is True


def test_quantum_efficiency_measurement_should_prefer_explicit_readout_amplitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit readout amplitude should set the default sweep upper bound."""
    captured: dict[str, object] = {}

    def _fake_quantum_efficiency_measurement(
        _exp: Experiment,
        target: str,
        **options: object,
    ) -> Result:
        captured.update(options)
        return Result(data={"target": target})

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(
        qe_module,
        "_quantum_efficiency_measurement",
        _fake_quantum_efficiency_measurement,
    )
    experiment = cast(
        Experiment,
        SimpleNamespace(
            ctx=SimpleNamespace(
                resolve_qubit_label=lambda target: target,
                params=SimpleNamespace(get_readout_amplitude=lambda target: 0.2),
            )
        ),
    )

    quantum_efficiency_measurement(
        experiment,
        "Q00",
        readout_amplitude=0.1,
        plot=False,
        enable_tqdm=False,
    )

    np.testing.assert_allclose(
        cast(Any, captured["amplitude_range"]),
        np.geomspace(0.001, 0.1, 17),
    )
    assert captured["enable_tqdm"] is False


def test_dephasing_sweep_should_keep_nine_points_before_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three low-coherence points should stop only after nine total points."""
    amplitudes = np.geomspace(0.001, 0.1, 17)
    measured_amplitudes: list[float] = []

    def _fake_measure_ramsey_fringe(
        _exp: Experiment,
        _target: str,
        *,
        readout_amplitude: float,
        phase_range: np.ndarray,
        n_shots: int | None,
        shot_interval: float | None,
    ) -> dict[str, object]:
        del phase_range, n_shots, shot_interval
        measured_amplitudes.append(readout_amplitude)
        rho01 = 0.45 if len(measured_amplitudes) == 1 else 0.05
        return {
            "rho01": rho01,
            "rho01_standard_error": np.nan,
        }

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(
        qe_module,
        "_measure_ramsey_fringe",
        _fake_measure_ramsey_fringe,
    )

    result = measurement_induced_dephasing_experiment(
        cast(Experiment, object()),
        "Q00",
        amplitude_range=amplitudes,
        phase_range=np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False),
        early_stop=True,
        plot=False,
    )

    assert len(measured_amplitudes) == 9
    assert result["early_stop_enabled"] is True
    assert result["early_stopped"] is True
    assert result["fit_result"]["weighted"] is False
    assert result["rho01_reference"] == pytest.approx(0.45)
    np.testing.assert_allclose(result["readout_amplitudes"], amplitudes[:9])

    measured_amplitudes.clear()
    full_result = measurement_induced_dephasing_experiment(
        cast(Experiment, object()),
        "Q00",
        amplitude_range=amplitudes,
        phase_range=np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False),
        plot=False,
    )

    assert len(measured_amplitudes) == len(amplitudes)
    assert full_result["early_stop_enabled"] is False
    assert full_result["early_stopped"] is False
    np.testing.assert_allclose(full_result["readout_amplitudes"], amplitudes)


def test_dephasing_sweep_should_reject_non_increasing_amplitudes() -> None:
    """Adaptive stopping is only well-defined for a small-to-large sweep."""
    with pytest.raises(ValueError, match="strictly increasing"):
        measurement_induced_dephasing_experiment(
            cast(Experiment, object()),
            "Q00",
            amplitude_range=[0.01, 0.03, 0.02],
            phase_range=np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False),
            plot=False,
        )


def test_dephasing_sweep_should_reject_a_single_amplitude() -> None:
    """A two-parameter dephasing fit should reject one amplitude before acquisition."""
    with pytest.raises(ValueError, match="at least two"):
        measurement_induced_dephasing_experiment(
            cast(Experiment, object()),
            "Q00",
            amplitude_range=[0.01],
            phase_range=np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False),
            plot=False,
        )


def test_quantum_efficiency_should_use_early_stopped_range_for_snr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SNR sweep should use only amplitudes completed by dephasing."""
    requested_amplitudes = np.geomspace(0.001, 0.1, 17)
    completed_amplitudes = requested_amplitudes[:9]
    captured: dict[str, object] = {}

    def _fake_dephasing(*args: object, **kwargs: object) -> Result:
        del args, kwargs
        return Result(data={"readout_amplitudes": completed_amplitudes})

    def _fake_snr(*args: object, **kwargs: object) -> Result:
        del args
        captured["amplitude_range"] = kwargs["amplitude_range"]
        captured["projection_method"] = kwargs["projection_method"]
        return Result(data={})

    def _fake_build(**kwargs: object) -> Result:
        captured["build_amplitudes"] = kwargs["readout_amplitudes"]
        return Result(data={})

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(
        qe_module,
        "_measurement_induced_dephasing_experiment",
        _fake_dephasing,
    )
    monkeypatch.setattr(qe_module, "_sweep_readout_snr", _fake_snr)
    monkeypatch.setattr(qe_module, "_build_quantum_efficiency_result", _fake_build)

    qe_module._quantum_efficiency_measurement(  # noqa: SLF001
        cast(Experiment, object()),
        "Q00",
        amplitude_range=requested_amplitudes,
        phase_range=np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False),
        n_shots=1024,
        shot_interval=None,
        readout_duration=None,
        projection_method="max_amplitude",
        early_stop=True,
        plot=False,
        save_image=False,
        enable_tqdm=False,
    )

    np.testing.assert_allclose(
        cast(Any, captured["amplitude_range"]), completed_amplitudes
    )
    np.testing.assert_allclose(
        cast(Any, captured["build_amplitudes"]), completed_amplitudes
    )
    assert captured["projection_method"] == "max_amplitude"


def test_quantum_efficiency_progress_should_track_settings_and_adjust_for_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress should cover Ramsey phases and state distributions at measured amplitudes."""
    requested_amplitudes = np.asarray([0.01, 0.02, 0.03, 0.04])
    completed_amplitudes = requested_amplitudes[:2]
    phases = np.asarray([0.0, np.pi / 2.0, np.pi])
    progress_bars: list[_RecordingProgress] = []

    def _fake_tqdm(**kwargs: object) -> _RecordingProgress:
        progress = _RecordingProgress(**kwargs)
        progress_bars.append(progress)
        return progress

    def _fake_dephasing(*args: object, **kwargs: object) -> Result:
        del args
        update = cast(Any, kwargs["progress_callback"])
        for _ in completed_amplitudes:
            update(len(cast(np.ndarray, kwargs["phase_range"])))
        return Result(data={"readout_amplitudes": completed_amplitudes})

    def _fake_snr(*args: object, **kwargs: object) -> Result:
        del args
        update = cast(Any, kwargs["progress_callback"])
        for _ in cast(np.ndarray, kwargs["amplitude_range"]):
            update(2)
        return Result(data={})

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(qe_module, "tqdm", _fake_tqdm)
    monkeypatch.setattr(
        qe_module,
        "_measurement_induced_dephasing_experiment",
        _fake_dephasing,
    )
    monkeypatch.setattr(qe_module, "_sweep_readout_snr", _fake_snr)
    monkeypatch.setattr(
        qe_module,
        "_build_quantum_efficiency_result",
        lambda **_kwargs: Result(data={}),
    )

    qe_module._quantum_efficiency_measurement(  # noqa: SLF001
        cast(Experiment, object()),
        "Q00",
        amplitude_range=requested_amplitudes,
        phase_range=phases,
        n_shots=1024,
        shot_interval=None,
        readout_duration=None,
        projection_method="max_amplitude",
        early_stop=True,
        plot=False,
        save_image=False,
        enable_tqdm=False,
    )

    assert len(progress_bars) == 1
    progress = progress_bars[0]
    assert progress.initial_total == len(requested_amplitudes) * (len(phases) + 2)
    assert progress.total == len(completed_amplitudes) * (len(phases) + 2)
    assert progress.n == progress.total
    assert progress.desc == "Quantum efficiency: Q00"
    assert progress.unit == "setting"
    assert progress.disable is True
    assert progress.refresh_count == 1
    assert progress.closed is True


def test_quantum_efficiency_progress_should_close_when_measurement_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress should close when acquisition raises an exception."""
    progress_bars: list[_RecordingProgress] = []

    def _fake_tqdm(**kwargs: object) -> _RecordingProgress:
        progress = _RecordingProgress(**kwargs)
        progress_bars.append(progress)
        return progress

    def _raise_during_dephasing(*args: object, **kwargs: object) -> Result:
        del args
        cast(Any, kwargs["progress_callback"])(3)
        raise RuntimeError("measurement failed")

    qe_module = importlib.import_module(
        "qubex.contrib.experiment.quantum_efficiency_measurement"
    )
    monkeypatch.setattr(qe_module, "tqdm", _fake_tqdm)
    monkeypatch.setattr(
        qe_module,
        "_measurement_induced_dephasing_experiment",
        _raise_during_dephasing,
    )

    with pytest.raises(RuntimeError, match="measurement failed"):
        qe_module._quantum_efficiency_measurement(  # noqa: SLF001
            cast(Experiment, object()),
            "Q00",
            amplitude_range=np.asarray([0.01, 0.02]),
            phase_range=np.asarray([0.0, np.pi / 2.0, np.pi]),
            n_shots=1024,
            shot_interval=None,
            readout_duration=None,
            projection_method="max_amplitude",
            early_stop=False,
            plot=False,
            save_image=False,
            enable_tqdm=True,
        )

    assert progress_bars[0].n == 3
    assert progress_bars[0].closed is True
