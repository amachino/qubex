"""Tests for the MCM randomized benchmarking experiment workflow."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from io import StringIO
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from plotly import graph_objects as go
from rich.console import Console

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


class PairedBootstrapMeasurementService:
    """Return protocol-paired trial fluctuations for bootstrap tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def execute(self, sequence: Any, **options: Any) -> Any:
        """Encode a shared trial offset in both matched protocols."""
        self.calls.append((sequence, options))
        call_index = len(self.calls) - 1
        n_index = call_index // 6
        trial_index = call_index % 6 // 2
        protocol_index = call_index % 2
        shared_trial_offset = (-0.06, 0.0, 0.09)[trial_index]
        protocol_base = (0.72, 0.84)[protocol_index]
        probability = protocol_base + 0.01 * n_index + shared_trial_offset

        def capture(value: float) -> Any:
            return SimpleNamespace(kerneled=np.asarray([value + 0.0j]))

        return SimpleNamespace(
            data={"RQ0": [capture(probability)], "RQ1": [capture(probability)]}
        )


class ExponentialMeasurementService:
    """Return exact trial-dependent exponential decay curves."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def execute(self, sequence: Any, **options: Any) -> Any:
        """Encode a bounded exponential for each protocol and trial."""
        self.calls.append((sequence, options))
        call_index = len(self.calls) - 1
        n_cliffords = (0, 1, 2, 4, 8)[call_index // 6]
        trial_index = call_index % 6 // 2
        protocol_index = call_index % 2
        central_decay = (0.94, 0.97)[protocol_index]
        decay_parameter = central_decay + (-0.01, 0.0, 0.01)[trial_index]
        probability = 0.4 * decay_parameter**n_cliffords + 0.5

        def capture(value: float) -> Any:
            return SimpleNamespace(kerneled=np.asarray([value + 0.0j]))

        return SimpleNamespace(
            data={"RQ0": [capture(probability)], "RQ1": [capture(probability)]}
        )


def _run_analysis_option_validation(
    fake_experiment: Any,
    *,
    n_bootstrap: int = 0,
    bootstrap_seed: int | None = 0,
    bootstrap_confidence_level: float = 0.95,
    min_fit_r_squared: float | None = 0.9,
) -> Any:
    """Call the public workflow with explicit analysis-option types."""
    return mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols="mcm-rep",
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence_level=bootstrap_confidence_level,
        min_fit_r_squared=min_fit_r_squared,
        plot=False,
        save_image=False,
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

    mcm_control = result.data["protocol_results"]["mcm-rb"]["Q0"]
    induced_errors = result.data["measurement_induced_errors"]
    induced = induced_errors["control"]["Q0"]

    np.testing.assert_allclose(
        mcm_control["trials"],
        [[0.903, 0.905], [0.883, 0.885], [0.863, 0.865]],
        rtol=0.0,
        atol=1e-12,
    )
    assert set(mcm_control) == {
        "trials",
        "mean",
        "std",
        "fit",
        "decay_parameter",
        "decay_parameter_uncertainty",
        "uncertainty_method",
        "error_per_cycle",
        "error_per_cycle_uncertainty",
        "bootstrap",
        "fit_validity",
    }
    assert induced["value"] == pytest.approx(
        0.5 * (1.0 - 0.96 / 0.98),
        rel=1e-12,
        abs=1e-12,
    )
    assert induced["uncertainty"] == pytest.approx(
        0.5 * np.sqrt((0.01 / 0.98) ** 2 + (0.96 * 0.01 / 0.98**2) ** 2),
        rel=1e-12,
        abs=1e-12,
    )
    assert induced["uncertainty_method"] == "independent_fit_propagation"
    assert induced["bootstrap"]["confidence_interval"] is None
    assert induced["bootstrap"]["unavailable_reason"] == (
        "at_least_four_sequence_lengths_required"
    )
    assert set(induced) == {
        "value",
        "uncertainty",
        "uncertainty_method",
        "bootstrap",
        "fit_validity",
    }
    assert set(induced["bootstrap"]) == {
        "successful_resamples",
        "success_rate",
        "standard_error",
        "confidence_interval",
        "unavailable_reason",
    }
    assert induced_errors["ancilla_population_with_cliffords"] == {"Q1": None}
    assert induced_errors["ancilla_population_with_control_delay"] == {"Q1": None}
    assert set(result.data) == {
        "n_cliffords",
        "seeds",
        "protocol_results",
        "measurement_induced_errors",
        "metadata",
    }
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


def test_paired_bootstrap_preserves_trial_pairing_in_induced_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Bootstrap ratios should resample matched protocol trial columns together."""
    fake_experiment.measurement_service = PairedBootstrapMeasurementService()

    def fake_fit_rb(*, target: str, title: str, **options: Any) -> FitResult:
        del target, options
        protocol = title.split()[0]
        p = {"mcm-rb": 0.75, "delay-rb": 0.87}[protocol]
        return FitResult(
            status=FitStatus.SUCCESS,
            data={
                "A": 0.4,
                "A_err": 0.01,
                "p": p,
                "p_err": 0.01,
                "C": 0.5,
                "C_err": 0.01,
                "r2": 0.99,
            },
            figure=go.Figure(),
        )

    monkeypatch.setattr(mcm_module.fitting, "fit_rb", fake_fit_rb)
    monkeypatch.setattr(
        mcm_module,
        "_fit_decay_parameter",
        lambda n_cliffords, probabilities: float(probabilities[-1]),
    )

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=3,
        seeds=[3, 5, 7],
        protocols=("mcm-rb", "delay-rb"),
        n_bootstrap=20,
        bootstrap_seed=7,
        bootstrap_confidence_level=0.8,
        plot=False,
        save_image=False,
    )

    bootstrap_indices = np.random.default_rng(7).integers(0, 3, size=(20, 3))
    trial_offsets = np.asarray([-0.06, 0.0, 0.09])
    offset_means = np.mean(trial_offsets[bootstrap_indices], axis=1)
    p_measurement = 0.75 + offset_means
    p_reference = 0.87 + offset_means
    bootstrap_errors = 0.5 * (1.0 - p_measurement / p_reference)
    expected_interval = tuple(np.quantile(bootstrap_errors, [0.1, 0.9]))

    estimate = result.data["measurement_induced_errors"]["control"]["Q0"]
    assert estimate["uncertainty_method"] == "paired_bootstrap"
    assert estimate["uncertainty"] == pytest.approx(
        np.std(bootstrap_errors, ddof=1), rel=1e-12, abs=1e-12
    )
    assert estimate["bootstrap"]["confidence_interval"] == pytest.approx(
        expected_interval, rel=1e-12, abs=1e-12
    )
    assert estimate["bootstrap"]["successful_resamples"] == 20
    assert estimate["fit_validity"]["is_valid"] is True
    target_result = result.data["protocol_results"]["mcm-rb"]["Q0"]
    assert target_result["uncertainty_method"] == "paired_bootstrap"
    assert target_result["bootstrap"]["successful_resamples"] == 20


def test_induced_error_retains_value_when_uncertainty_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """A matched decay ratio should not disappear only because uncertainty is absent."""
    fake_experiment.measurement_service = ConstantMeasurementService()

    def fake_fit_rb(*, title: str, **_: Any) -> FitResult:
        protocol = title.split()[0]
        return FitResult(
            status=FitStatus.SUCCESS,
            data={"p": {"mcm-rb": 0.96, "delay-rb": 0.98}[protocol]},
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
        protocols=("mcm-rb", "delay-rb"),
        n_bootstrap=0,
        plot=False,
        save_image=False,
    )

    estimate = result.data["measurement_induced_errors"]["control"]["Q0"]
    assert estimate["value"] == pytest.approx(
        0.5 * (1.0 - 0.96 / 0.98), rel=1e-12, abs=1e-12
    )
    assert estimate["uncertainty"] is None
    assert estimate["uncertainty_method"] == "unavailable"


def test_bootstrap_fits_exponential_resamples_through_the_public_workflow(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Bootstrap should fit every well-conditioned exponential resample."""
    fake_experiment.measurement_service = ExponentialMeasurementService()

    def fake_fit_rb(*, target: str, title: str, **options: Any) -> FitResult:
        del target, options
        protocol = title.split()[0]
        p = {"mcm-rb": 0.94, "delay-rb": 0.97}[protocol]
        return FitResult(
            status=FitStatus.SUCCESS,
            data={
                "A": 0.4,
                "A_err": 0.01,
                "p": p,
                "p_err": 0.01,
                "C": 0.5,
                "C_err": 0.01,
                "r2": 0.999,
            },
            figure=go.Figure(),
        )

    monkeypatch.setattr(mcm_module.fitting, "fit_rb", fake_fit_rb)

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2, 4, 8],
        n_trials=3,
        seeds=[3, 5, 7],
        protocols=("mcm-rb", "delay-rb"),
        n_bootstrap=12,
        bootstrap_seed=11,
        plot=False,
        save_image=False,
    )

    bootstrap = result.data["protocol_results"]["mcm-rb"]["Q0"]["bootstrap"]
    assert bootstrap["successful_resamples"] == 12
    assert bootstrap["success_rate"] == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert bootstrap["unavailable_reason"] is None
    assert bootstrap["confidence_interval"] is not None


def test_failed_primary_fit_does_not_report_an_orphaned_bootstrap_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """A failed primary fit should not expose uncertainty without a fitted value."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **_: FitResult(status=FitStatus.ERROR),
    )
    monkeypatch.setattr(
        mcm_module,
        "_fit_decay_parameter",
        lambda *_: 0.9,
    )

    result = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=2,
        seeds=[3, 5],
        protocols="mcm-rep",
        n_bootstrap=8,
        plot=False,
        print_summary=False,
        enable_tqdm=False,
    )

    target_result = result.data["protocol_results"]["mcm-rep"]["Q0"]
    assert target_result["decay_parameter"] is None
    assert target_result["decay_parameter_uncertainty"] is None
    assert target_result["uncertainty_method"] == "unavailable"
    assert target_result["bootstrap"]["successful_resamples"] == 8


def test_fit_validity_reports_insufficient_points_and_low_r_squared(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Fit diagnostics should distinguish convergence from a valid decay fit."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **_: FitResult(
            status=FitStatus.SUCCESS,
            data={"p": 0.95, "p_err": 0.01, "r2": 0.2},
            figure=go.Figure(),
        ),
    )

    insufficient = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols="mcm-rep",
        n_bootstrap=0,
        plot=False,
        save_image=False,
    )
    low_r_squared = mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=1,
        seeds=[3],
        protocols="mcm-rep",
        n_bootstrap=0,
        min_fit_r_squared=0.9,
        plot=False,
        save_image=False,
    )

    insufficient_validity = insufficient.data["protocol_results"]["mcm-rep"]["Q0"][
        "fit_validity"
    ]
    low_r_squared_validity = low_r_squared.data["protocol_results"]["mcm-rep"]["Q0"][
        "fit_validity"
    ]
    assert insufficient_validity["is_valid"] is False
    assert "insufficient_sequence_lengths" in insufficient_validity["reasons"]
    assert low_r_squared_validity["is_valid"] is False
    assert "r_squared_below_threshold" in low_r_squared_validity["reasons"]


def test_measurement_scale_metadata_records_the_applied_intermediate_readout(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Metadata should identify scaled intermediate and calibrated terminal pulses."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **_: FitResult(
            status=FitStatus.SUCCESS,
            data={"p": 0.95, "p_err": 0.01, "r2": 0.99},
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
        protocols="mcm-rb",
        measurement_scale=2.0,
        n_bootstrap=0,
        plot=False,
        save_image=False,
    )

    readout_metadata = result.data["metadata"]["pulses"]["intermediate_measurements"][
        "Q1"
    ]
    assert readout_metadata == {
        "source": "scaled_calibrated",
        "scale": 2.0,
        "duration_ns": 64.0,
        "peak_amplitude": pytest.approx(0.4, rel=0.0, abs=1e-12),
        "integrated_power": pytest.approx(10.24, rel=0.0, abs=1e-12),
        "integrated_power_units": "amplitude_squared_ns",
        "ramp_trimmed_active_interval_ns": (0.0, 64.0),
    }
    assert result.data["metadata"]["pulses"]["terminal_measurements"] == {
        "targets": ("Q0", "Q1"),
        "source": "calibrated_default",
    }
    assert result.data["metadata"]["pulses"]["ancilla_x180_durations_ns"] == {}
    assert set(result.data["metadata"]["pulses"]) == {
        "intermediate_measurements",
        "terminal_measurements",
        "ancilla_x180_durations_ns",
    }
    assert set(result.data["metadata"]) == {
        "controls",
        "ancillas",
        "control_echo",
        "ancilla_mode",
        "acquisition",
        "analysis",
        "pulses",
    }
    assert result.data["metadata"]["acquisition"] == {
        "n_trials": 1,
        "n_shots": 1024,
        "shot_interval_ns": 153_600.0,
        "time_integration": True,
    }
    assert result.data["metadata"]["analysis"] == {
        "n_bootstrap": 0,
        "bootstrap_seed": 0,
        "bootstrap_confidence_level": 0.95,
        "min_bootstrap_success_rate": 0.8,
        "min_fit_r_squared": 0.9,
    }


def test_experiment_prints_summary_by_default_independently_of_plot(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """The workflow should print fit and induced-error tables by default."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    output = StringIO()
    monkeypatch.setattr(
        mcm_module,
        "console",
        Console(file=output, force_terminal=False, width=160),
        raising=False,
    )

    def fake_fit_rb(*, title: str, **_: Any) -> FitResult:
        protocol = title.split()[0]
        return FitResult(
            status=FitStatus.SUCCESS,
            data={
                "A": 0.4,
                "A_err": 0.01,
                "p": {"mcm-rb": 0.96, "delay-rb": 0.98}[protocol],
                "p_err": 0.01,
                "C": 0.5,
                "C_err": 0.01,
                "r2": 0.99,
            },
            figure=go.Figure(),
        )

    monkeypatch.setattr(mcm_module.fitting, "fit_rb", fake_fit_rb)

    mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols=("mcm-rb", "delay-rb"),
        n_bootstrap=0,
        plot=False,
        enable_tqdm=False,
    )

    rendered = output.getvalue()
    assert "MCM randomized benchmarking: protocol fits" in rendered
    assert "MCM randomized benchmarking: measurement-induced errors" in rendered
    assert "mcm-rb" in rendered
    assert "control" in rendered


def test_experiment_can_disable_summary_printing(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """An explicit false value should suppress all summary-table output."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    output = StringIO()
    monkeypatch.setattr(
        mcm_module,
        "console",
        Console(file=output, force_terminal=False),
        raising=False,
    )
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **_: FitResult(status=FitStatus.ERROR),
    )

    mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols="mcm-rep",
        n_bootstrap=0,
        plot=False,
        print_summary=False,
        enable_tqdm=False,
    )

    assert output.getvalue() == ""


def test_experiment_enables_progress_bar_by_default(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """The workflow should pass an enabled default to its progress bar."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    progress_options: list[dict[str, Any]] = []

    class FakeProgress:
        def update(self) -> None:
            """Accept one simulated completed schedule."""

        def close(self) -> None:
            """Accept progress cleanup."""

    def fake_tqdm(**options: Any) -> FakeProgress:
        progress_options.append(options)
        return FakeProgress()

    monkeypatch.setattr(mcm_module, "tqdm", fake_tqdm)
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **_: FitResult(status=FitStatus.ERROR),
    )

    mcm_randomized_benchmarking(
        fake_experiment,
        "Q0",
        "Q1",
        n_cliffords_range=[0, 1, 2],
        n_trials=1,
        seeds=[3],
        protocols="mcm-rep",
        n_bootstrap=0,
        plot=False,
        print_summary=False,
    )

    assert len(progress_options) == 1
    assert progress_options[0]["disable"] is False


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

    assert tuple(result.data["protocol_results"]) == ("mcm-rb", "delay-rb")
    assert set(result.data["protocol_results"]["mcm-rb"]) == {"Q0", "Q1"}
    induced_errors = result.data["measurement_induced_errors"]
    assert induced_errors["control"]["Q0"]["value"] == pytest.approx(
        0.5 * (1.0 - 0.96 / 0.98),
        rel=1e-12,
        abs=1e-12,
    )
    assert induced_errors["ancilla_population_with_cliffords"]["Q1"][
        "value"
    ] == pytest.approx(
        0.5 * (1.0 - 0.90 / 0.95),
        rel=1e-12,
        abs=1e-12,
    )
    assert result.data["metadata"]["ancilla_mode"] == "randomized"
    assert result.data["metadata"]["pulses"]["ancilla_x180_durations_ns"] == {
        "Q1": 16.0
    }
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

    assert set(result.data["protocol_results"]["mcm-rb"]) == {
        "Q0",
        "Q1",
        "Q2",
        "Q3",
    }
    induced_errors = result.data["measurement_induced_errors"]
    assert set(induced_errors["control"]) == {"Q0", "Q2"}
    assert set(induced_errors["ancilla_population_with_cliffords"]) == {
        "Q1",
        "Q3",
    }
    assert induced_errors["control"]["Q0"]["value"] == pytest.approx(
        0.5 * (1.0 - 0.96 / 0.98), rel=1e-12, abs=1e-12
    )
    assert result.data["metadata"]["controls"] == ("Q0", "Q2")
    assert result.data["metadata"]["ancillas"] == ("Q1", "Q3")
    assert fake_experiment.pulse.validated_targets == ["Q0", "Q2", "Q1", "Q3"]
    assert fake_experiment.ctx.reset_calls == [{"Q0", "Q1", "Q2", "Q3"}]
    assert len(measurement_service.calls) == 6
    assert set(result.figures or {}) == {
        f"{protocol}:{target}"
        for protocol in ("mcm-rb", "delay-rb")
        for target in ("Q0", "Q2", "Q1", "Q3")
    }


def test_randomized_ancilla_reports_clifford_and_control_delay_induced_errors(
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
        control_echo=True,
        ancilla_mode="randomized",
        plot=False,
        save_image=False,
    )

    assert tuple(result.data["protocol_results"]) == (
        "mcm-rb",
        "delay-rb",
        "mcm-rep",
        "delay-rep",
    )
    induced_errors = result.data["measurement_induced_errors"]
    assert induced_errors["control"]["Q0"]["value"] == pytest.approx(
        0.5 * (1.0 - 0.96 / 0.98), rel=1e-12, abs=1e-12
    )
    assert induced_errors["ancilla_population_with_cliffords"]["Q1"][
        "value"
    ] == pytest.approx(0.5 * (1.0 - 0.90 / 0.95), rel=1e-12, abs=1e-12)
    assert induced_errors["ancilla_population_with_control_delay"]["Q1"][
        "value"
    ] == pytest.approx(
        0.5 * (1.0 - 0.88 / 0.92),
        rel=1e-12,
        abs=1e-12,
    )
    assert set(induced_errors) == {
        "control",
        "ancilla_population_with_cliffords",
        "ancilla_population_with_control_delay",
    }
    assert len(measurement_service.calls) == 12


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (
            lambda exp: _run_analysis_option_validation(exp, n_bootstrap=-1),
            "n_bootstrap",
        ),
        (
            lambda exp: _run_analysis_option_validation(exp, bootstrap_seed=-1),
            "bootstrap_seed",
        ),
        (
            lambda exp: _run_analysis_option_validation(
                exp,
                bootstrap_confidence_level=1.0,
            ),
            "bootstrap_confidence_level",
        ),
        (
            lambda exp: _run_analysis_option_validation(
                exp,
                min_fit_r_squared=1.1,
            ),
            "min_fit_r_squared",
        ),
    ],
)
def test_experiment_rejects_invalid_bootstrap_and_fit_diagnostic_options(
    fake_experiment: Any,
    run: Callable[[Any], Any],
    message: str,
) -> None:
    """Bootstrap and fit-diagnostic options should reject invalid bounds."""
    with pytest.raises((TypeError, ValueError), match=message):
        run(fake_experiment)


def test_randomized_mcm_repetition_does_not_report_unmatched_induced_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """MCM repetition alone should expose its fit without an induced error."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    output = StringIO()
    monkeypatch.setattr(
        mcm_module,
        "console",
        Console(file=output, force_terminal=False),
    )
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **_: FitResult(
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

    assert result.data["protocol_results"]["mcm-rep"]["Q1"]["decay_parameter"] == 0.95
    assert result.data["measurement_induced_errors"] == {
        "control": {"Q0": None},
        "ancilla_population_with_cliffords": {"Q1": None},
        "ancilla_population_with_control_delay": {"Q1": None},
    }
    assert "measurement-induced errors" not in output.getvalue()


def test_multiple_targets_do_not_report_unmatched_induced_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_experiment: Any,
) -> None:
    """Unmatched protocols should retain target keys with unavailable estimates."""
    fake_experiment.measurement_service = ConstantMeasurementService()
    monkeypatch.setattr(
        mcm_module.fitting,
        "fit_rb",
        lambda **_: FitResult(
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

    assert result.data["measurement_induced_errors"] == {
        "control": {"Q0": None, "Q2": None},
        "ancilla_population_with_cliffords": {"Q1": None, "Q3": None},
        "ancilla_population_with_control_delay": {"Q1": None, "Q3": None},
    }


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

    target_result = result.data["protocol_results"]["mcm-rep"]["Q0"]
    assert target_result["trials"].shape == (3, 1)
    assert target_result["decay_parameter"] is None
    assert result.data["measurement_induced_errors"]["control"] == {"Q0": None}
    assert result.figures is None
