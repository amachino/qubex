"""Tests for continuous-time GEF relaxation characterization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.linalg import expm

import qubex.contrib.experiment.gef_relaxation as relaxation_module
from qubex import contrib
from qubex.contrib import experiment as experiment_contrib
from qubex.contrib.experiment.gef_relaxation import (
    GefRelaxationMeasurements,
    analyze_gef_relaxation,
    characterize_gef_relaxation,
)
from qubex.contrib.experiment.readout_mist import (
    GefPopulationFitDiagnostic,
    GefPopulationSeries,
)
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.pulse import FlatTop

RATE_NAMES = (
    "g_to_e",
    "e_to_g",
    "g_to_f",
    "e_to_f",
    "f_to_g",
    "f_to_e",
)


def _generator(rates: dict[str, float]) -> np.ndarray:
    generator = np.zeros((3, 3), dtype=np.float64)
    indices = (
        ("g_to_e", 1, 0),
        ("e_to_g", 0, 1),
        ("g_to_f", 2, 0),
        ("e_to_f", 2, 1),
        ("f_to_g", 0, 2),
        ("f_to_e", 1, 2),
    )
    for name, final_state, initial_state in indices:
        generator[final_state, initial_state] = rates[name]
    for initial_state in range(3):
        generator[initial_state, initial_state] = -np.sum(generator[:, initial_state])
    return generator


def _series(populations: np.ndarray) -> GefPopulationSeries:
    covariance = np.repeat(
        (1e-7 * np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]))[
            None, :, :
        ],
        populations.shape[0],
        axis=0,
    )
    diagnostics = tuple(
        GefPopulationFitDiagnostic(
            objective=0.0,
            design_rank=2,
            design_condition_number=1.0,
            residual=np.zeros(15),
            message="synthetic",
        )
        for _ in populations
    )
    return GefPopulationSeries(
        population=populations,
        covariance=covariance,
        standard_error=np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)),
        population_unconstrained=populations,
        fit_diagnostics=diagnostics,
    )


def _synthetic_measurements() -> tuple[GefRelaxationMeasurements, dict[str, float]]:
    rates = {
        "g_to_e": 0.002,
        "e_to_g": 0.045,
        "g_to_f": 0.001,
        "e_to_f": 0.004,
        "f_to_g": 0.006,
        "f_to_e": 0.080,
    }
    initial = {
        "initial_g": np.array([0.97, 0.02, 0.01]),
        "initial_e": np.array([0.03, 0.94, 0.03]),
        "initial_f": np.array([0.02, 0.06, 0.92]),
    }
    time_ns = np.array([0.0, 250.0, 750.0, 2_000.0, 6_000.0, 20_000.0, 60_000.0])
    generator = _generator(rates)
    protocols = {
        protocol: _series(
            np.stack([expm(generator * (time * 1e-3)) @ p0 for time in time_ns])
        )
        for protocol, p0 in initial.items()
    }
    return GefRelaxationMeasurements(time_ns=time_ns, protocols=protocols), rates


def test_analysis_recovers_six_continuous_transition_rates() -> None:
    """Exact GEF trajectories should recover all six generator rates."""
    measurements, expected_rates = _synthetic_measurements()

    analysis = analyze_gef_relaxation(measurements)

    assert analysis.success
    assert analysis.jacobian_rank == 12
    np.testing.assert_allclose(
        [analysis.transition_rates[name].value_per_us for name in RATE_NAMES],
        [expected_rates[name] for name in RATE_NAMES],
        rtol=2e-4,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.sum(analysis.generator_matrix_per_us, axis=0),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_analysis_reports_directional_and_total_f_lifetimes() -> None:
    """Derived times should distinguish f-to-e decay from total f-state exit."""
    measurements, rates = _synthetic_measurements()

    analysis = analyze_gef_relaxation(measurements)

    assert analysis.derived_times_ns["t1_f_to_e"].value == pytest.approx(
        1_000.0 / rates["f_to_e"], rel=2e-4
    )
    assert analysis.derived_times_ns["f_state_exit_lifetime"].value == pytest.approx(
        1_000.0 / (rates["f_to_e"] + rates["f_to_g"]), rel=2e-4
    )
    assert analysis.derived_times_ns["ef_pair_relaxation_time"].value == pytest.approx(
        1_000.0 / (rates["f_to_e"] + rates["e_to_f"]), rel=2e-4
    )


class _Context:
    qubit_labels = ("Q0",)
    measurement = SimpleNamespace(sampling_period=2.0)
    experiment_system = SimpleNamespace(
        measurement_defaults={"execution": {"shot_interval_ns": 500_000.0}}
    )
    util = SimpleNamespace(
        discretize_time_range=lambda values, *, sampling_period: (
            np.round(np.asarray(values) / sampling_period) * sampling_period
        )
    )

    @staticmethod
    def resolve_qubit_label(label: str) -> str:
        return label

    @staticmethod
    def resolve_ge_label(label: str) -> str:
        return label

    @staticmethod
    def resolve_ef_label(label: str) -> str:
        return f"{label}:ef"


def _experiment() -> Experiment:
    pulse = SimpleNamespace(
        x180=lambda label: FlatTop(
            duration=32.0 if label.endswith(":ef") else 24.0,
            amplitude=0.1,
            tau=4.0,
        )
    )
    return cast(Experiment, SimpleNamespace(ctx=_Context(), pulse=pulse))


def test_workflow_builds_three_preparations_and_returns_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workflow should acquire G, E, and F trajectories in one GEF batch."""
    calls: list[dict[str, Any]] = []
    measurements, _ = _synthetic_measurements()

    def measure(*_args: Any, **kwargs: Any) -> Result:
        calls.append(kwargs)
        fits: dict[str, dict[str, Any]] = {}
        for name in kwargs["sequences"]:
            protocol, time_text = name.split("/")
            time_ns = float(time_text.removeprefix("t_ns="))
            time_index = int(np.nonzero(measurements.time_ns == time_ns)[0][0])
            series = measurements.protocols[protocol]
            fits[name] = {
                "Q0": SimpleNamespace(
                    population=series.population[time_index],
                    population_covariance=series.covariance[time_index],
                    population_standard_error=series.standard_error[time_index],
                    population_unconstrained=series.population_unconstrained[
                        time_index
                    ],
                    objective=0.0,
                    residual=np.zeros(15),
                    design_rank=2,
                    design_condition_number=1.0,
                    message="synthetic",
                )
            }
        return Result(
            data={
                "fits": fits,
                "calibration": {"Q0": object()},
                "raw_iq": {name: object() for name in kwargs["sequences"]},
                "moment_summaries": {name: object() for name in kwargs["sequences"]},
            }
        )

    monkeypatch.setattr(relaxation_module, "measure_gef_populations", measure)

    result = characterize_gef_relaxation(
        _experiment(),
        "Q0",
        time_range=measurements.time_ns,
        acquisition_order="interleaved",
        enable_tqdm=False,
        plot=False,
    )

    assert len(calls) == 1
    assert len(calls[0]["sequences"]) == 3 * len(measurements.time_ns)
    assert tuple(calls[0]["sequences"])[:9] == (
        "initial_g/t_ns=0",
        "initial_e/t_ns=0",
        "initial_f/t_ns=0",
        "initial_e/t_ns=250",
        "initial_f/t_ns=250",
        "initial_g/t_ns=250",
        "initial_f/t_ns=750",
        "initial_g/t_ns=750",
        "initial_e/t_ns=750",
    )
    assert calls[0]["n_bootstrap"] == 0
    submitted = calls[0]["sequences"]
    assert submitted["initial_g/t_ns=250"].duration == pytest.approx(250.0)
    assert submitted["initial_e/t_ns=250"].duration == pytest.approx(274.0)
    assert submitted["initial_f/t_ns=250"].duration == pytest.approx(306.0)
    analysis = result.data["analysis"]["Q0"]
    assert analysis.success
    assert result.data["measurements"]["Q0"].time_ns[0] == 0.0


def test_supplied_calibration_is_not_reported_as_new_calibration_shots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reused calibration should record that no calibration shots were acquired."""
    measurements, _ = _synthetic_measurements()
    calibration = {"Q0": cast(Any, object())}

    def measure(*_args: Any, **kwargs: Any) -> Result:
        fits = {}
        for name in kwargs["sequences"]:
            protocol, time_text = name.split("/")
            time_ns = float(time_text.removeprefix("t_ns="))
            index = int(np.nonzero(measurements.time_ns == time_ns)[0][0])
            series = measurements.protocols[protocol]
            fits[name] = {
                "Q0": SimpleNamespace(
                    population=series.population[index],
                    population_covariance=series.covariance[index],
                    population_standard_error=series.standard_error[index],
                    population_unconstrained=series.population_unconstrained[index],
                    objective=0.0,
                    residual=np.zeros(15),
                    design_rank=2,
                    design_condition_number=1.0,
                    message="synthetic",
                )
            }
        return Result(
            data={
                "fits": fits,
                "calibration": calibration,
                "raw_iq": {},
                "moment_summaries": {},
            }
        )

    monkeypatch.setattr(relaxation_module, "measure_gef_populations", measure)

    result = characterize_gef_relaxation(
        _experiment(),
        "Q0",
        time_range=measurements.time_ns,
        calibration=calibration,
        gef_calibration_n_shots=False,  # type: ignore[arg-type]
        enable_tqdm=False,
        plot=False,
    )

    assert result.data["measurement_options"]["gef_calibration_n_shots"] is None


@pytest.mark.parametrize(
    ("option", "value", "error_type"),
    [
        ("shot_interval", True, TypeError),
        ("acquisition_seed", True, TypeError),
        ("acquisition_seed", 1.5, TypeError),
    ],
)
def test_invalid_scalar_options_fail_before_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: object,
    error_type: type[Exception],
) -> None:
    """Invalid scalar options should be rejected before hardware acquisition."""
    monkeypatch.setattr(
        relaxation_module,
        "measure_gef_populations",
        lambda *_args, **_kwargs: pytest.fail("acquisition should not run"),
    )
    options: dict[str, Any] = {option: value}

    with pytest.raises(error_type):
        characterize_gef_relaxation(
            _experiment(),
            "Q0",
            time_range=[0, 100, 200, 300, 400],
            enable_tqdm=False,
            plot=False,
            **options,
        )


def test_boolean_time_point_fails_before_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boolean delay points should not be interpreted as numeric times."""
    monkeypatch.setattr(
        relaxation_module,
        "measure_gef_populations",
        lambda *_args, **_kwargs: pytest.fail("acquisition should not run"),
    )

    with pytest.raises(TypeError, match="time_range"):
        characterize_gef_relaxation(
            _experiment(),
            "Q0",
            time_range=[0, True, 100, 200, 300],
            enable_tqdm=False,
            plot=False,
        )


def test_schedule_names_distinguish_close_large_delay_values() -> None:
    """Distinct delay points should never collide in the schedule mapping."""
    # The collision is observable only after acquisition has already lost a point.
    first = relaxation_module._schedule_name("initial_g", 1_000_000.0)  # noqa: SLF001
    second = relaxation_module._schedule_name("initial_g", 1_000_002.0)  # noqa: SLF001

    assert first != second


def test_public_namespaces_export_gef_relaxation_workflow() -> None:
    """The continuous-time characterization should be available from contrib APIs."""
    assert contrib.characterize_gef_relaxation is characterize_gef_relaxation
    assert experiment_contrib.characterize_gef_relaxation is characterize_gef_relaxation
    assert experiment_contrib.analyze_gef_relaxation is analyze_gef_relaxation
