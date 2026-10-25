"""Public API tests for readout-MIST characterization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest

import qubex.contrib.experiment.readout_mist as mist_module
from qubex import contrib
from qubex.contrib import experiment as experiment_contrib
from qubex.contrib.experiment import analyze_readout_mist, characterize_readout_mist
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.pulse import FlatTop, PulseSchedule


def test_readout_mist_is_exported_from_public_contrib_namespaces() -> None:
    """Measurement and postprocessing should be public in contrib namespaces."""
    assert mist_module.__all__ == [
        "analyze_readout_mist",
        "characterize_readout_mist",
    ]
    assert contrib.characterize_readout_mist is characterize_readout_mist
    assert experiment_contrib.characterize_readout_mist is characterize_readout_mist
    assert contrib.analyze_readout_mist is analyze_readout_mist
    assert experiment_contrib.analyze_readout_mist is analyze_readout_mist


class _Context:
    experiment_system = SimpleNamespace(
        measurement_defaults={"execution": {"shot_interval_ns": 500_000.0}}
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

    @staticmethod
    def resolve_read_label(label: str) -> str:
        return f"R{label}"


def _experiment() -> Experiment:
    pulse = SimpleNamespace(
        readout=lambda _target: FlatTop(duration=80.0, amplitude=0.2, tau=8.0),
        x180=lambda label: FlatTop(duration=24.0, amplitude=0.1, tau=4.0),
    )
    return cast(Experiment, SimpleNamespace(ctx=_Context(), pulse=pulse))


@pytest.mark.parametrize(
    "counts",
    [
        (0, 1, 2, 4),
        (1, 2, 4, 8, 16),
        (0, 2, 4, 8, 16),
        (0, 1, 1, 4, 8),
        (0, 1, 2, 4, 3),
        (0, 1, 2, 4, 8.0),
    ],
)
def test_invalid_repetition_counts_fail_before_acquisition(counts: Any) -> None:
    """Invalid grids should fail before any hardware acquisition."""
    with pytest.raises((TypeError, ValueError)):
        characterize_readout_mist(
            _experiment(),
            "Q0",
            repetition_counts=counts,
            plot=False,
        )


def test_custom_cycle_cannot_also_request_an_implicit_gap() -> None:
    """A complete cycle override should not accept a second trailing gap."""
    with PulseSchedule() as cycle:
        cycle.add("RQ0", FlatTop(duration=80.0, amplitude=0.2, tau=8.0))

    with pytest.raises(ValueError, match="cycle_gap_ns"):
        characterize_readout_mist(
            _experiment(),
            "Q0",
            readout_cycle=cycle,
            cycle_gap_ns=20.0,
            plot=False,
        )


def test_invalid_gef_covariance_cutoff_fails_before_acquisition() -> None:
    """The GEF covariance cutoff should be validated before hardware acquisition."""
    with pytest.raises((TypeError, ValueError), match="gef_covariance_rcond"):
        characterize_readout_mist(
            _experiment(),
            "Q0",
            gef_covariance_rcond=-1.0,
            plot=False,
        )


def test_public_measurement_api_excludes_advanced_analysis_options() -> None:
    """The daily measurement API should expose only the full-model workflow."""
    defaults = characterize_readout_mist.__kwdefaults__

    assert defaults is not None
    assert mist_module.DEFAULT_REPETITION_COUNTS == (
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    )
    assert defaults["measure_reference"] is True
    assert defaults["enable_tqdm"] is True
    assert "fit_reduced_models" not in defaults
    assert "compute_profile_upper_bounds" not in defaults
    assert "profile_confidence_level" not in defaults
    assert "gef_bootstrap_n_resamples" not in defaults


@pytest.mark.parametrize(
    ("acquisition_order", "acquisition_seed"),
    [("interleaved", 0), ("randomized", None)],
)
def test_workflow_interleaves_actual_and_complete_blank_reference(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    acquisition_order: str,
    acquisition_seed: int | None,
) -> None:
    """Actual and blank-reference points should remain adjacent in one sweep."""
    calls: list[dict[str, Any]] = []
    transition = np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.02], [0.01, 0.01, 0.97]])
    initial = {
        "initial_g": np.array([0.98, 0.015, 0.005]),
        "initial_e": np.array([0.015, 0.975, 0.01]),
        "initial_f": np.array([0.01, 0.02, 0.97]),
    }
    reference_transition = np.array(
        [[0.995, 0.003, 0.002], [0.003, 0.994, 0.004], [0.002, 0.003, 0.994]]
    )
    covariance = 1e-7 * np.array(
        [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]
    )
    calibration = {"Q0": object()}

    def measure(*_args: Any, **kwargs: Any) -> Result:
        calls.append(kwargs)
        fits = {}
        for name in kwargs["sequences"]:
            protocol = next(value for value in initial if f"/{value}/" in name)
            count = int(name.rsplit("=", 1)[1])
            fitted_transition = (
                reference_transition if name.startswith("reference/") else transition
            )
            population = (
                np.linalg.matrix_power(fitted_transition, count) @ initial[protocol]
            )
            fits[name] = {
                "Q0": SimpleNamespace(
                    population=population,
                    population_covariance=covariance,
                    population_standard_error=np.sqrt(np.diag(covariance)),
                    population_unconstrained=population,
                    objective=0.0,
                    residual=np.zeros(15),
                    design_rank=2,
                    design_condition_number=2.0,
                    message="synthetic",
                )
            }
        return Result(
            data={
                "fits": fits,
                "calibration": calibration,
                "raw_iq": {name: f"iq:{name}" for name in kwargs["sequences"]},
                "moment_summaries": {
                    name: f"moments:{name}" for name in kwargs["sequences"]
                },
            }
        )

    monkeypatch.setattr(mist_module, "measure_gef_populations", measure)
    monkeypatch.setattr(go.Figure, "show", lambda self: None)
    with PulseSchedule() as cycle:
        cycle.add("RQ0", FlatTop(duration=80.0, amplitude=0.25, tau=8.0))
        cycle.add("RQ0", FlatTop(duration=20.0, amplitude=0.05, tau=4.0))

    result = characterize_readout_mist(
        _experiment(),
        target="Q0",
        repetition_counts=(0, 1, 2, 4, 8),
        readout_cycle=cycle,
        acquisition_order=acquisition_order,  # type: ignore[arg-type]
        acquisition_seed=acquisition_seed,
        plot=True,
        enable_tqdm=False,
    )

    assert len(calls) == 1
    assert calls[0]["calibration_n_shots"] == 8192
    assert calls[0]["capture_index"] == -1
    assert "calibration" not in calls[0]
    submitted_names = tuple(calls[0]["sequences"])
    if acquisition_order == "interleaved":
        assert submitted_names[:6] == (
            "actual/initial_g/n=0",
            "reference/initial_g/n=0",
            "actual/initial_e/n=0",
            "reference/initial_e/n=0",
            "actual/initial_f/n=0",
            "reference/initial_f/n=0",
        )
    for actual_name, reference_name in zip(
        submitted_names[::2], submitted_names[1::2], strict=True
    ):
        assert actual_name.removeprefix("actual/") == reference_name.removeprefix(
            "reference/"
        )
    actual_n1 = calls[0]["sequences"]["actual/initial_g/n=1"]
    reference_n1 = calls[0]["sequences"]["reference/initial_g/n=1"]
    assert np.any(actual_n1.values["RQ0"] != 0.0)
    np.testing.assert_allclose(reference_n1.values["RQ0"], 0.0, rtol=0.0, atol=0.0)
    assert actual_n1.duration == reference_n1.duration
    raw_data = result.data["raw_data"]
    assert tuple(raw_data["actual_raw_iq"]) == tuple(
        name for name in submitted_names if name.startswith("actual/")
    )
    assert tuple(raw_data["reference_raw_iq"]) == tuple(
        name for name in submitted_names if name.startswith("reference/")
    )
    assert tuple(raw_data["actual_moment_summaries"]) == tuple(
        name for name in submitted_names if name.startswith("actual/")
    )
    assert tuple(raw_data["reference_population_fits"]) == tuple(
        name for name in submitted_names if name.startswith("reference/")
    )
    measurements = result.data["measurements"]
    np.testing.assert_allclose(
        measurements.protocols["initial_g"].actual.population[1],
        transition @ initial["initial_g"],
    )
    reference_series = measurements.protocols["initial_g"].reference
    assert reference_series is not None
    np.testing.assert_allclose(
        reference_series.population[1],
        reference_transition @ initial["initial_g"],
    )
    assert result.data["analysis"].actual.state_order == ("g", "e", "f")
    assert result.data["analysis"].actual_model_comparison is None
    assert result.data["analysis"].reference_model_comparison is None
    assert "profile_upper_bounds" not in result.data["analysis_options"]
    assert "reduced_models" not in result.data["analysis_options"]
    assert result.data["readout_cycle_timing"]["trailing_gap_in_final_cycle"] is True
    assert result.data["measurement_options"]["diagnostic_capture_index"] == -1
    assert result.data["measurement_options"]["submitted_sequence_order"] == (
        submitted_names
    )
    assert result.data["measurement_options"]["reference_acquisition"] == (
        "same_batch_adjacent_pairs"
    )
    assert capsys.readouterr().out == ""
