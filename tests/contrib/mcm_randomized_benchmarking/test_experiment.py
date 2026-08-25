"""Tests for the MCM randomized benchmarking experiment workflow."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from plotly import graph_objects as go

from qubex.analysis import FitResult, FitStatus
from qubex.contrib.experiment.mcm_randomized_benchmarking import (
    mcm_randomized_benchmarking,
)

mcm_module = importlib.import_module(
    "qubex.contrib.experiment.mcm_randomized_benchmarking"
)


class FakeMeasurementService:
    """Return distinct intermediate and final capture values."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def execute(self, sequence: Any, **options: Any) -> Any:
        """Encode the terminal probability in the last capture."""
        self.calls.append((sequence, options))
        call_index = len(self.calls) - 1
        n_cliffords = (0, 1, 2)[call_index // 6]
        seed = (3, 5)[call_index % 6 // 3]
        protocol_offset = (0.00, 0.05, 0.10)[call_index % 3]
        control_probability = 0.90 - 0.02 * n_cliffords + 0.001 * seed
        ancilla_probability = control_probability - protocol_offset

        def capture(value: float) -> Any:
            return SimpleNamespace(kerneled=np.asarray([value + 0.0j]))

        return SimpleNamespace(
            data={
                "RQ0": [capture(-10.0), capture(control_probability)],
                "Q1": [capture(-20.0), capture(ancilla_probability)],
            }
        )


class ConstantMeasurementService:
    """Return fixed terminal captures for real pulse schedules."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def execute(self, sequence: Any, **options: Any) -> Any:
        """Return one terminal capture for each test qubit."""
        self.calls.append((sequence, options))

        def capture(value: float) -> Any:
            return SimpleNamespace(kerneled=np.asarray([value + 0.0j]))

        return SimpleNamespace(
            data={f"RQ{index}": [capture(0.9 - 0.05 * index)] for index in range(5)}
        )


def test_experiment_uses_terminal_captures_and_reports_induced_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """The workflow should analyze final captures and compare MCM-RB with delay-RB."""
    measurement_service = FakeMeasurementService()
    fake_experiment.measurement_service = measurement_service

    def fake_fit_rb(*, target: str, title: str, **options: Any) -> FitResult:
        del options
        protocol = title.split()[0]
        p = {
            ("mcm-rb", "Q0"): 0.96,
            ("delay-rb", "Q0"): 0.98,
        }.get((protocol, target), 0.90)
        return FitResult(
            status=FitStatus.SUCCESS,
            data={"p": p, "p_err": 0.01},
            figure=go.Figure(),
        )

    monkeypatch.setattr(mcm_module.fitting, "fit_rb", fake_fit_rb)

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=2,
        seeds=[3, 5],
        n_shots=64,
        shot_interval=1_000.0,
        plot=False,
        save_image=False,
    )

    mcm_control = result.data["protocols"]["mcm-rb"]["Q0"]
    induced = result.data["measurement_induced_control_error"]

    np.testing.assert_allclose(
        mcm_control["trials"],
        [[0.903, 0.905], [0.883, 0.885], [0.863, 0.865]],
        rtol=0.0,
        atol=1e-12,
    )
    assert induced["value"] == pytest.approx(
        0.5 * (1.0 - 0.96 / 0.98),
        rel=1e-12,
        abs=1e-12,
    )
    assert induced["error"] == pytest.approx(
        0.5 * np.sqrt((0.01 / 0.98) ** 2 + (0.96 * 0.01 / 0.98**2) ** 2),
        rel=1e-12,
        abs=1e-12,
    )
    assert result.data["measurement_induced_ancilla_population_error"] is None
    assert (
        result.data["measurement_induced_ancilla_population_error_with_idle_control"]
        is None
    )
    assert fake_experiment.ctx.reset_calls == [{"Q0", "Q1"}]
    assert len(measurement_service.calls) == 18
    assert all(
        call[1]["final_measurement"] is True for call in measurement_service.calls
    )
    assert all(
        call[1]["reset_awg_and_capunits"] is False for call in measurement_service.calls
    )
    assert set(result.figures or {}) == {
        "mcm-rb:Q0",
        "mcm-rb:Q1",
        "delay-rb:Q0",
        "delay-rb:Q1",
        "mcm-rep:Q0",
        "mcm-rep:Q1",
    }


def test_randomized_ancilla_analyzes_both_targets_with_matched_references(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Randomized mode should fit both targets and compare matched MCM and delays."""
    measurement_service = ConstantMeasurementService()
    fake_experiment.measurement_service = measurement_service

    def fake_fit_rb(*, target: str, title: str, **options: Any) -> FitResult:
        del options
        protocol = title.split()[0]
        p = {
            ("mcm-rb", "Q0"): 0.96,
            ("delay-rb", "Q0"): 0.98,
            ("mcm-rb", "Q1"): 0.90,
            ("delay-rb", "Q1"): 0.95,
        }[(protocol, target)]
        return FitResult(
            status=FitStatus.SUCCESS,
            data={"p": p, "p_err": 0.01},
            figure=go.Figure(),
        )

    monkeypatch.setattr(mcm_module.fitting, "fit_rb", fake_fit_rb)

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        ancilla_mode="randomized",
        plot=False,
        save_image=False,
    )

    assert tuple(result.data["protocols"]) == ("mcm-rb", "delay-rb")
    assert set(result.data["protocols"]["mcm-rb"]) == {"Q0", "Q1"}
    assert result.data["measurement_induced_control_error"]["value"] == pytest.approx(
        0.5 * (1.0 - 0.96 / 0.98),
        rel=1e-12,
        abs=1e-12,
    )
    assert result.data["measurement_induced_ancilla_population_error"][
        "value"
    ] == pytest.approx(
        0.5 * (1.0 - 0.90 / 0.95),
        rel=1e-12,
        abs=1e-12,
    )
    assert result.data["metadata"]["ancilla_mode"] == "randomized"
    assert result.data["metadata"]["ancilla_x180_duration"] == 16.0
    assert len(measurement_service.calls) == 6
    for mcm_call, delay_call in zip(
        measurement_service.calls[::2],
        measurement_service.calls[1::2],
        strict=True,
    ):
        assert (
            mcm_call[0].get_sequence("Q1").values.tolist()
            == delay_call[0].get_sequence("Q1").values.tolist()
        )


def test_multiple_controls_and_ancillas_report_per_target_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Multiple controls and ancillas should be acquired and analyzed together."""
    measurement_service = ConstantMeasurementService()
    fake_experiment.measurement_service = measurement_service

    def fake_fit_rb(*, target: str, title: str, **options: Any) -> FitResult:
        del options
        protocol = title.split()[0]
        base = {"Q0": 0.96, "Q1": 0.92, "Q2": 0.94, "Q3": 0.90}[target]
        p = base if protocol == "mcm-rb" else base + 0.02
        return FitResult(
            status=FitStatus.SUCCESS,
            data={"p": p, "p_err": 0.01},
            figure=go.Figure(),
        )

    monkeypatch.setattr(mcm_module.fitting, "fit_rb", fake_fit_rb)

    result = mcm_randomized_benchmarking(
        fake_experiment,
        ["Q0", "Q2"],
        ["Q1", "Q3"],
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        ancilla_mode="randomized",
        plot=False,
        save_image=False,
    )

    assert set(result.data["protocols"]["mcm-rb"]) == {"Q0", "Q1", "Q2", "Q3"}
    assert set(result.data["measurement_induced_control_error"]) == {"Q0", "Q2"}
    assert set(result.data["measurement_induced_ancilla_population_error"]) == {
        "Q1",
        "Q3",
    }
    assert result.data["measurement_induced_control_error"]["Q0"][
        "value"
    ] == pytest.approx(0.5 * (1.0 - 0.96 / 0.98), rel=1e-12, abs=1e-12)
    assert result.data["metadata"]["controls"] == ("Q0", "Q2")
    assert result.data["metadata"]["ancillas"] == ("Q1", "Q3")
    assert result.data["metadata"]["control"] is None
    assert result.data["metadata"]["ancilla"] is None
    assert fake_experiment.pulse.validated_targets == ["Q0", "Q2", "Q1", "Q3"]
    assert fake_experiment.ctx.reset_calls == [{"Q0", "Q1", "Q2", "Q3"}]
    assert len(measurement_service.calls) == 6
    assert set(result.figures or {}) == {
        f"{protocol}:{target}"
        for protocol in ("mcm-rb", "delay-rb")
        for target in ("Q0", "Q2", "Q1", "Q3")
    }


def test_randomized_ancilla_reports_active_and_idle_control_induced_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Matched RB and repetition pairs should report their ancilla errors."""
    measurement_service = ConstantMeasurementService()
    fake_experiment.measurement_service = measurement_service

    def fake_fit_rb(*, target: str, title: str, **options: Any) -> FitResult:
        del options
        protocol = title.split()[0]
        p = {
            ("mcm-rb", "Q0"): 0.96,
            ("delay-rb", "Q0"): 0.98,
            ("mcm-rb", "Q1"): 0.90,
            ("delay-rb", "Q1"): 0.95,
            ("mcm-rep", "Q0"): 0.97,
            ("delay-rep", "Q0"): 0.99,
            ("mcm-rep", "Q1"): 0.88,
            ("delay-rep", "Q1"): 0.92,
        }[(protocol, target)]
        return FitResult(
            status=FitStatus.SUCCESS,
            data={"p": p, "p_err": 0.01},
            figure=go.Figure(),
        )

    monkeypatch.setattr(mcm_module.fitting, "fit_rb", fake_fit_rb)

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols=("mcm-rb", "delay-rb", "mcm-rep", "delay-rep"),
        ancilla_mode="randomized",
        plot=False,
        save_image=False,
    )

    assert tuple(result.data["protocols"]) == (
        "mcm-rb",
        "delay-rb",
        "mcm-rep",
        "delay-rep",
    )
    assert result.data["measurement_induced_control_error"]["value"] == pytest.approx(
        0.5 * (1.0 - 0.96 / 0.98), rel=1e-12, abs=1e-12
    )
    assert result.data["measurement_induced_ancilla_population_error"][
        "value"
    ] == pytest.approx(0.5 * (1.0 - 0.90 / 0.95), rel=1e-12, abs=1e-12)
    assert result.data[
        "measurement_induced_ancilla_population_error_with_idle_control"
    ]["value"] == pytest.approx(
        0.5 * (1.0 - 0.88 / 0.92),
        rel=1e-12,
        abs=1e-12,
    )
    assert len(measurement_service.calls) == 12


def test_randomized_mcm_repetition_does_not_report_unmatched_induced_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """MCM repetition alone should expose its fit without an induced error."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **options: FitResult(
            status=FitStatus.SUCCESS,
            data={"p": 0.95, "p_err": 0.01},
            figure=go.Figure(),
        ),
    )

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols="mcm-rep",
        ancilla_mode="randomized",
        plot=False,
        save_image=False,
    )

    assert result.data["protocols"]["mcm-rep"]["Q1"]["decay_parameter"] == 0.95
    assert result.data["measurement_induced_control_error"] is None
    assert result.data["measurement_induced_ancilla_population_error"] is None
    assert (
        result.data["measurement_induced_ancilla_population_error_with_idle_control"]
        is None
    )


def test_multiple_targets_do_not_report_unmatched_induced_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Unmatched protocols should return no induced-error field for any group size."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **options: FitResult(
            status=FitStatus.SUCCESS,
            data={"p": 0.95, "p_err": 0.01},
            figure=go.Figure(),
        ),
    )

    result = mcm_randomized_benchmarking(
        fake_experiment,
        ["Q0", "Q2"],
        ["Q1", "Q3"],
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols="mcm-rep",
        ancilla_mode="randomized",
        plot=False,
        save_image=False,
    )

    assert result.data["measurement_induced_control_error"] is None
    assert result.data["measurement_induced_ancilla_population_error"] is None
    assert (
        result.data["measurement_induced_ancilla_population_error_with_idle_control"]
        is None
    )


def test_experiment_rejects_mismatched_trial_and_seed_counts(
    fake_experiment: Any,
) -> None:
    """The workflow should require exactly one seed per trial."""
    with pytest.raises(ValueError, match="number of seeds"):
        mcm_randomized_benchmarking(
            fake_experiment,
            "Q0",
            "Q1",
            n_cliffords_range=[0, 1, 2],
            n_trials=2,
            seeds=[1],
            plot=False,
            save_image=False,
        )


def test_experiment_rejects_invalid_xaxis_type(fake_experiment: Any) -> None:
    """The workflow should reject unsupported fit-axis scales."""
    with pytest.raises(ValueError, match="xaxis_type"):
        mcm_randomized_benchmarking(
            fake_experiment,
            "Q0",
            "Q1",
            n_cliffords_range=[0, 1, 2],
            n_trials=1,
            seeds=[1],
            xaxis_type="symlog",  # type: ignore[arg-type]
            plot=False,
            save_image=False,
        )


@pytest.mark.parametrize("seed", [1.0, "1", 1 + 0j])
def test_experiment_rejects_noninteger_seeds(
    fake_experiment: Any,
    seed: object,
) -> None:
    """Seed arrays should contain integer values, not coercible values."""
    with pytest.raises(TypeError, match=r"seeds.*integers"):
        mcm_randomized_benchmarking(
            fake_experiment,
            "Q0",
            "Q1",
            n_cliffords_range=[0, 1, 2],
            n_trials=1,
            seeds=[seed],  # type: ignore[arg-type]
            protocols="mcm-rep",
            plot=False,
            save_image=False,
        )


def test_experiment_preserves_large_integer_seed(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Seed validation should not lose precision through floating-point conversion."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **options: FitResult(
            status=FitStatus.ERROR,
            message=f"Skipped fit for {options['target']}",
        ),
    )
    seed = 2**53 + 1

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[seed],
        protocols="mcm-rep",
        plot=False,
        save_image=False,
    )

    assert result.data["seeds"].tolist() == [seed]


@pytest.mark.parametrize("shot_interval", [0.0, -1.0, np.nan, np.inf])
def test_experiment_rejects_nonpositive_or_nonfinite_shot_interval(
    fake_experiment: Any,
    shot_interval: float,
) -> None:
    """The workflow should reject a nonpositive or nonfinite shot interval."""
    with pytest.raises(ValueError, match="shot_interval"):
        mcm_randomized_benchmarking(
            fake_experiment,
            "Q0",
            "Q1",
            n_cliffords_range=[0, 1, 2],
            n_trials=1,
            seeds=[3],
            protocols="mcm-rep",
            shot_interval=shot_interval,
            plot=False,
            save_image=False,
        )


def test_experiment_rejects_boolean_shot_interval(fake_experiment: Any) -> None:
    """A boolean shot interval should not be interpreted as a number."""
    with pytest.raises(TypeError, match="shot_interval"):
        mcm_randomized_benchmarking(
            fake_experiment,
            "Q0",
            "Q1",
            n_cliffords_range=[0, 1, 2],
            n_trials=1,
            seeds=[3],
            protocols="mcm-rep",
            shot_interval=True,
            plot=False,
            save_image=False,
        )


def test_experiment_generates_each_random_clifford_sequence_once(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """All selected protocols should share one generated sequence per length and seed."""
    fake_experiment.measurement_service = ConstantMeasurementService()

    def fake_fit_rb(**options: Any) -> FitResult:
        del options
        return FitResult(
            status=FitStatus.SUCCESS,
            data={"p": 0.95, "p_err": 0.01},
            figure=go.Figure(),
        )

    monkeypatch.setattr(mcm_module.fitting, "fit_rb", fake_fit_rb)

    mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=2,
        seeds=[3, 5],
        plot=False,
        save_image=False,
    )

    generator = fake_experiment.benchmarking_service.clifford_generator
    assert generator.calls == [
        (0, "1Q", 3),
        (0, "1Q", 5),
        (1, "1Q", 3),
        (1, "1Q", 5),
        (2, "1Q", 3),
        (2, "1Q", 5),
    ]


def test_single_protocol_fit_failure_returns_raw_data_without_figures(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """A failed single-protocol fit should retain data without an IRB estimate."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **options: FitResult(
            status=FitStatus.ERROR,
            message=f"Failed fit for {options['target']}",
        ),
    )

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols="mcm-rep",
        plot=False,
        save_image=False,
    )

    target_result = result.data["protocols"]["mcm-rep"]["Q0"]
    assert target_result["trials"].shape == (3, 1)
    assert target_result["decay_parameter"] is None
    assert result.data["measurement_induced_control_error"] is None
    assert result.figures is None
