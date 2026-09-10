"""Tests for CR-pulse coherence measurement orchestration."""

# ruff: noqa: SLF001

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from qubex.contrib.experiment import (
    cr_pulse_coherence_analysis as analysis_module,
    cr_pulse_coherence_characterization as module,
)
from qubex.contrib.experiment.cr_pulse_coherence_analysis import (
    CrOnDephasingFit,
)
from qubex.contrib.experiment.cr_pulse_fidelity_simulation import (
    CrPulseFidelitySimulationResult,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, PulseSchedule, Rect, VirtualZ


class _DummyPulseService:
    """Provide deterministic pulse durations for schedule construction."""

    def __init__(self) -> None:
        self.rabi_params = {
            "Q0": SimpleNamespace(normalize=lambda values: np.asarray(values).real),
            "Q1": SimpleNamespace(normalize=lambda values: np.asarray(values).real),
        }

    def x90(self, _target: str) -> Rect:
        """Return a two-nanosecond X90 stand-in."""
        return Rect(duration=2.0, amplitude=1.0)

    def x90m(self, _target: str) -> Rect:
        """Return a two-nanosecond negative X90 stand-in."""
        return Rect(duration=2.0, amplitude=-1.0)

    def x180(self, _target: str) -> Blank:
        """Return a four-nanosecond X180 stand-in."""
        return Blank(4.0)

    def y90(self, _target: str) -> Blank:
        """Return a two-nanosecond Y90 stand-in."""
        return Blank(2.0)

    def y90m(self, _target: str) -> Blank:
        """Return a two-nanosecond negative Y90 stand-in."""
        return Blank(2.0)

    def z180(self) -> VirtualZ:
        """Return a virtual Z180 pulse."""
        return VirtualZ(np.pi)

    def get_pulse_for_state(self, target: str, state: str) -> Blank:
        """Return state-preparation stand-ins."""
        durations = {"0": 0.0, "1": 4.0, "+": 2.0}
        del target
        return Blank(durations[state])


class _DummyContext:
    """Resolve dummy qubit labels."""

    def resolve_qubit_label(self, target: str) -> str:
        """Return one of the two supported labels."""
        if target not in {"Q0", "Q1"}:
            raise ValueError(target)
        return target


class _DummyExperiment:
    """Provide only the interfaces needed by the workflow."""

    def __init__(self) -> None:
        self.pulse = _DummyPulseService()
        self.ctx = _DummyContext()
        self.measurement_service = SimpleNamespace()


def _schedule(duration: float) -> PulseSchedule:
    """Return a three-channel active stand-in for a ZX90 schedule."""
    with PulseSchedule(["Q0", "Q0-Q1", "Q1"]) as schedule:
        schedule.add("Q0-Q1", Rect(duration=duration, amplitude=1.0))
    return schedule


def test_fidelity_uncertainty_api_has_no_outer_bootstrap_controls() -> None:
    """The public workflow should expose only local covariance propagation."""
    parameters = inspect.signature(module.characterize_cr_pulse_coherence).parameters
    plot_parameters = inspect.signature(
        analysis_module.plot_cr_pulse_coherence
    ).parameters

    assert "propagate_fidelity_uncertainty" in parameters
    assert parameters["propagate_fidelity_uncertainty"].default is None
    assert "n_fidelity_bootstrap" not in parameters
    assert "fidelity_bootstrap_seed" not in parameters
    assert "leakage_ylim" not in parameters
    assert "leakage_ylim" not in plot_parameters


def test_fidelity_output_prints_local_standard_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Successful delta-method errors should be displayed with point estimates."""
    simulation = CrPulseFidelitySimulationResult(
        idle_coherence_limited_fidelity=0.999,
        cr_on_coherence_limited_fidelity=0.982,
        cr_on_dissipative_limited_fidelity=0.973,
        average_leakage=0.008,
        model_metadata={},
    )
    uncertainty = analysis_module.CrPulseFidelityLinearUncertainty(
        success=True,
        message="ok",
        idle_coherence_limited_fidelity_standard_error=0.0,
        cr_on_coherence_limited_fidelity_standard_error=0.0018,
        cr_on_dissipative_limited_fidelity_standard_error=0.0024,
        average_leakage_standard_error=0.0015,
        output_covariance=np.zeros((4, 4)),
        parameter_covariance=np.zeros((9, 9)),
        jacobian=np.zeros((4, 9)),
        parameter_names=tuple(f"p{index}" for index in range(9)),
        output_names=tuple(f"y{index}" for index in range(4)),
        finite_difference_steps=np.zeros(9),
        finite_difference_schemes=("central",) * 9,
        fixed_zero_parameters=(),
        metadata={},
    )
    analysis = analysis_module.CrPulseFidelityAnalysis(
        success=True,
        message="ok",
        cr_on_noise=None,
        control_dephasing_fit=None,
        target_dephasing_fit=None,
        simulation=simulation,
        linear_uncertainty=uncertainty,
    )

    module._print_fidelity_analysis(analysis, ())

    output = capsys.readouterr().out
    assert "98.200000% ± 0.180000%" in output
    assert "97.300000% ± 0.240000%" in output
    assert "0.800000% ± 0.150000%" in output
    assert "Idle T1/T2 uncertainty was not propagated" in output


def test_default_un_echoed_zx90_repeats_two_same_sign_cr_lobes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generated un-echoed ZX90 should join two identical CR lobes."""
    exp = _DummyExperiment()
    primitive = _schedule(8.0)
    primitive.cr_duration = 8.0  # type: ignore[attr-defined]
    primitive.echo = False  # type: ignore[attr-defined]
    calls: list[tuple[str, str, bool]] = []

    def zx90(control: str, target: str, *, echo: bool) -> PulseSchedule:
        calls.append((control, target, echo))
        return primitive

    monkeypatch.setattr(exp.pulse, "zx90", zx90, raising=False)

    gate = module._build_un_echoed_zx90(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
    )

    assert calls == [("Q0", "Q1", False)]
    assert gate.duration == pytest.approx(16.0)
    assert vars(gate)["cr_duration"] == pytest.approx(16.0)
    assert vars(gate)["echo"] is False
    np.testing.assert_array_equal(
        gate.get_sampled_sequence("Q0-Q1"),
        np.tile(primitive.get_sampled_sequence("Q0-Q1"), 2),
    )


def test_control_t2_echo_block_uses_requested_x180_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control T2 echo should apply XI, XI+IX, and XI between four ZX90s."""
    exp = _DummyExperiment()
    x180_targets: list[str] = []
    original_x180 = exp.pulse.x180

    def record_x180(target: str) -> Blank:
        x180_targets.append(target)
        return original_x180(target)

    monkeypatch.setattr(exp.pulse, "x180", record_x180)

    block = module._control_t2_echo_block(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        _schedule(20.0),
    )

    assert x180_targets == ["Q0", "Q0", "Q1", "Q0"]
    assert block.duration == pytest.approx(4 * 20.0 + 3 * 4.0)


def test_target_t2rho_echo_block_places_z180_between_two_zx90s() -> None:
    """Target T2rho echo should place one target Z180 between two ZX90s."""
    block = module._target_t2rho_echo_block(
        _DummyExperiment(),  # type: ignore[arg-type]
        "Q1",
        _schedule(20.0),
    )

    target_elements = block.get_sequences(copy=True)["Q1"].flattened_elements
    z_rotations = [
        element for element in target_elements if isinstance(element, VirtualZ)
    ]
    cr_elements = block.get_sequences(copy=True)["Q0-Q1"].flattened_elements

    assert [abs(rotation.theta) for rotation in z_rotations] == pytest.approx([np.pi])
    assert sum(isinstance(element, Rect) for element in cr_elements) == 2
    assert block.duration == pytest.approx(2 * 20.0)


def test_protocol_references_match_actual_evolution_durations() -> None:
    """Every reference should preserve its protocol's actual evolution time."""
    exp = _DummyExperiment()
    no_echo = _schedule(8.0)
    echo = _schedule(20.0)

    point = module._build_protocol_sequences(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        n=3,
        zx90_no_echo=no_echo,
        zx90_echo=echo,
    )

    for protocol in (
        "control_ground",
        "control_excited",
        "control_t2_echo",
        "target_t2rho_echo",
    ):
        assert point.sequences[f"{protocol}_reference"].duration == pytest.approx(
            point.sequences[protocol].duration
        )
        assert point.evolution_durations[f"{protocol}_reference"] == pytest.approx(
            point.evolution_durations[protocol]
        )
        for condition in (protocol, f"{protocol}_reference"):
            sequence = point.sequences[condition]
            assert sequence.is_valid()
    assert point.evolution_durations["control_ground"] == pytest.approx(4 * 3 * 8.0)
    assert point.evolution_durations["control_excited"] == pytest.approx(4 * 3 * 8.0)
    assert point.evolution_durations["control_t2_echo"] == pytest.approx(
        3 * (4 * 20.0 + 3 * 4.0)
    )
    assert point.evolution_durations["target_t2rho_echo"] == pytest.approx(
        2 * 3 * (2 * 20.0)
    )
    for protocol in ("control_ground", "control_excited"):
        actual = point.sequences[protocol].get_sampled_sequence("Q0-Q1")
        reference = point.sequences[f"{protocol}_reference"].get_sampled_sequence(
            "Q0-Q1"
        )
        unit_nonzero = np.count_nonzero(no_echo.get_sampled_sequence("Q0-Q1"))
        assert np.count_nonzero(actual) == 12 * unit_nonzero
        assert np.count_nonzero(reference) == 0
    ground_reference_target = point.sequences[
        "control_ground_reference"
    ].get_sampled_sequence("Q1")
    excited_reference_target = point.sequences[
        "control_excited_reference"
    ].get_sampled_sequence("Q1")
    assert np.max(ground_reference_target.real) > 0
    assert np.min(ground_reference_target.real) >= 0
    assert np.min(excited_reference_target.real) < 0
    assert np.max(excited_reference_target.real) <= 0
    for protocol in ("control_t2_echo", "target_t2rho_echo"):
        actual = point.sequences[protocol].get_sampled_sequence("Q0-Q1")
        reference = point.sequences[f"{protocol}_reference"].get_sampled_sequence(
            "Q0-Q1"
        )
        unit_nonzero = np.count_nonzero(echo.get_sampled_sequence("Q0-Q1"))
        assert np.count_nonzero(actual) == 12 * unit_nonzero
        assert np.count_nonzero(reference) == 0
    assert (
        np.count_nonzero(
            point.sequences["target_t2rho_echo"].get_sampled_sequence("Q1")
        )
        == 0
    )


def test_characterization_runs_requested_hardware_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration and selected conditions should run in canonical order."""
    exp = _DummyExperiment()
    events: list[object] = []
    calibration = {"Q0": object(), "Q1": object()}

    def fake_calibrate(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("calibration")
        return calibration

    def fake_measure_gef(
        _exp: object,
        targets: object,
        sequences: dict[str, PulseSchedule],
        **kwargs: object,
    ) -> Result:
        del targets, kwargs
        events.append(("gef", tuple(sequences)))
        populations = {
            name: {
                "Q0": np.array([0.8, 0.18, 0.02]),
                "Q1": np.array([0.2, 0.79, 0.01]),
            }
            for name in sequences
        }
        raw_iq = {
            name: {
                analyzer: {
                    "Q0": np.ones(8, dtype=complex),
                    "Q1": np.ones(8, dtype=complex),
                }
                for analyzer in ("s1", "s4", "s5")
            }
            for name in sequences
        }
        fits = {name: {"Q0": object(), "Q1": object()} for name in sequences}
        moment_summaries = {
            name: {
                analyzer: {"Q0": object(), "Q1": object()}
                for analyzer in ("s1", "s4", "s5")
            }
            for name in sequences
        }
        return Result(
            data={
                "populations": populations,
                "raw_iq": raw_iq,
                "fits": fits,
                "moment_summaries": moment_summaries,
            }
        )

    pauli_values = iter([0.9, 0.8, 0.7, 0.6] * 3)

    def fake_measure_pauli(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        events.append("pauli")
        value = next(pauli_values)
        return module._PauliMeasurement(
            expectation=value,
            standard_error=0.01,
            raw_iq=np.array([value + 0j]),
        )

    def fake_bootstrap(
        _calibration: object,
        raw_iq: dict[str, object],
        **kwargs: object,
    ) -> dict[str, dict[str, Any]]:
        del kwargs
        events.append("bootstrap")
        return {
            name: {
                target: SimpleNamespace(
                    samples=np.tile(
                        np.array([0.8, 0.18, 0.02])
                        if target == "Q0"
                        else np.array([0.2, 0.79, 0.01]),
                        (8, 1),
                    ),
                    standard_error=np.full(3, 0.01),
                    unavailable_reason=None,
                )
                for target in ("Q0", "Q1")
            }
            for name in raw_iq
        }

    monkeypatch.setattr(module, "calibrate_gef_population", fake_calibrate)
    monkeypatch.setattr(module, "measure_gef_populations", fake_measure_gef)
    monkeypatch.setattr(module, "_measure_pauli_expectation", fake_measure_pauli)
    monkeypatch.setattr(module, "bootstrap_gef_populations", fake_bootstrap)

    with pytest.warns(RuntimeWarning, match="target F-state population"):
        result = module.characterize_cr_pulse_coherence(
            exp,  # type: ignore[arg-type]
            "Q0",
            "Q1",
            measure_orthogonal_components=False,
            n_values=[0, 1, 2],
            zx90_no_echo=_schedule(8.0),
            zx90_echo=_schedule(20.0),
            n_bootstrap=8,
            target_leakage_warning_threshold=0.005,
            propagate_fidelity_uncertainty=False,
            enable_tqdm=False,
            plot=False,
        )

    expected_per_n: list[object] = [
        (
            "gef",
            (
                "control_ground_reference",
                "control_ground",
                "control_excited_reference",
                "control_excited",
            ),
        ),
        "pauli",
        "pauli",
        "pauli",
        "pauli",
    ]
    assert events == [
        "calibration",
        *expected_per_n,
        *expected_per_n,
        *expected_per_n,
        "bootstrap",
    ]
    measurements = result.data["measurements"]
    analysis = result.data["analysis"]
    assert measurements.n_values == (0, 1, 2)
    assert measurements.protocols == (
        "control_ground",
        "control_excited",
        "control_t2_echo",
        "target_t2rho_echo",
    )
    assert measurements.cr_pulse_counts == (0, 4, 8)
    assert_allclose(
        measurements.target_polarizations["control_ground"]["actual"],
        (0.79 - 0.2) / (0.79 + 0.2),
    )
    assert analysis.transition_rates["unit"] == "1/ns"
    assert analysis.decay_times["unit"] == "ns"
    assert analysis.fidelity_analysis is None
    assert result.data["measurement_options"]["n_shots"] == 4096
    assert result.data["measurement_options"]["calibration_n_shots"] == 8192
    assert result.data["analysis_options"]["target_leakage_warning_triggered"]
    assert result.data["analysis_options"]["maximum_target_leakage"] == pytest.approx(
        0.01
    )
    assert "leakage_ylim" not in result.data["analysis_options"]
    assert "cr_duration" in result.data["analysis_options"]["forward_model_error"]
    assert set(analysis.fits["target_t1rho"]) == {
        "control_ground",
        "control_excited",
    }
    assert set(analysis.fits["target_leakage"]) == {
        "control_ground",
        "control_excited",
    }
    assert all(
        protocol_fits["actual"].model == "control_transition_forward"
        and protocol_fits["reference"].model == "exponential"
        for protocol_fits in analysis.fits["target_t1rho"].values()
    )
    assert analysis.fit_status["target_t1rho_model"] == {
        "control_ground": "control_transition_forward",
        "control_excited": "control_transition_forward",
    }
    assert set(result.figures or {}) == {
        "control_ground_control_populations",
        "control_ground_target_polarization",
        "control_excited_control_populations",
        "control_excited_target_polarization",
        "control_t2_echo",
        "target_t2rho_echo",
    }
    ground_control = result.get_figure("control_ground_control_populations")
    ground_control_data: Any = ground_control.data
    assert ground_control_data[0].name == "reference Pg"
    assert ground_control_data[-1].name == "actual fit Pf"
    ground_control_layout: Any = ground_control.layout
    assert tuple(ground_control_layout.yaxis.range) == (0.0, 1.0)
    assert ground_control_layout.xaxis.range[0] == 0.0
    assert tuple(ground_control_layout.xaxis2.ticktext) == ("0", "4", "8")
    ground_target = result.get_figure("control_ground_target_polarization")
    ground_target_layout: Any = ground_target.layout
    assert tuple(ground_target_layout.yaxis.range) == (-1.05, 1.05)
    assert tuple(ground_target_layout.yaxis2.range) == (0.0, 0.05)
    assert tuple(ground_target_layout.xaxis3.ticktext) == ("0", "4", "8")


def test_pauli_measurement_uses_only_requested_basis_analyzer() -> None:
    """X, Y, and Z measurements should append -Y90, +X90, and no analyzer."""
    exp = _DummyExperiment()
    calls: list[dict[str, object]] = []

    def measure(sequence: PulseSchedule, **kwargs: object) -> SimpleNamespace:
        calls.append({"sequence": sequence, **kwargs})
        return SimpleNamespace(
            data={"Q0": SimpleNamespace(kerneled=np.array([1.0, 3.0]) + 0j)}
        )

    exp.measurement_service.measure = measure
    preparation = _schedule(8.0)

    measured_x = module._measure_pauli_expectation(
        exp,  # type: ignore[arg-type]
        preparation,
        "Q0",
        "X",
        n_shots=2,
        shot_interval=1.0,
    )
    measured_y = module._measure_pauli_expectation(
        exp,  # type: ignore[arg-type]
        preparation,
        "Q0",
        "Y",
        n_shots=2,
        shot_interval=1.0,
    )
    measured_z = module._measure_pauli_expectation(
        exp,  # type: ignore[arg-type]
        preparation,
        "Q0",
        "Z",
        n_shots=2,
        shot_interval=1.0,
    )

    assert measured_x.expectation == pytest.approx(2.0)
    assert measured_x.standard_error == pytest.approx(1.0)
    assert measured_y.expectation == pytest.approx(2.0)
    assert measured_z.expectation == pytest.approx(2.0)
    assert calls[0]["sequence"].duration == pytest.approx(10.0)  # type: ignore[union-attr]
    assert calls[1]["sequence"].duration == pytest.approx(10.0)  # type: ignore[union-attr]
    assert calls[2]["sequence"].duration == pytest.approx(8.0)  # type: ignore[union-attr]
    y_sequence: Any = calls[1]["sequence"]
    assert np.max(y_sequence.get_sampled_sequence("Q0").real) > 0
    for call in calls:
        assert call["mode"] == "single"
        assert call["state_classification"] is False


def test_unavailable_bootstrap_uses_analytic_polarization_error() -> None:
    """An unavailable bootstrap should fall back to the analytic covariance."""
    bootstrap: Any = SimpleNamespace(
        unavailable_reason="success_rate_below_threshold",
        samples=np.array(
            [
                [0.8, 0.18, 0.02],
                [0.7, 0.27, 0.03],
            ]
        ),
    )
    analytic_fit: Any = SimpleNamespace(
        population=np.array([0.8, 0.18, 0.02]),
        population_covariance=np.diag([4e-4, 1e-4, 1e-4]),
    )

    standard_error, source = module._polarization_standard_error(
        bootstrap,
        analytic_fit,
    )

    assert np.isfinite(standard_error)
    assert source == "analytic"


def test_population_error_prefers_bootstrap_then_analytic_fallback() -> None:
    """Population errors should follow the documented source priority."""
    analytic_fit: Any = SimpleNamespace(
        population_standard_error=np.array([0.02, 0.03, 0.04]),
    )
    available_bootstrap: Any = SimpleNamespace(
        unavailable_reason=None,
        standard_error=np.array([0.01, 0.01, 0.01]),
    )
    unavailable_bootstrap: Any = SimpleNamespace(
        unavailable_reason="disabled",
        standard_error=None,
    )

    bootstrap_error, bootstrap_source = module._population_standard_error(
        available_bootstrap,
        analytic_fit,
    )
    analytic_error, analytic_source = module._population_standard_error(
        unavailable_bootstrap,
        analytic_fit,
    )

    assert_allclose(bootstrap_error, 0.01)
    assert bootstrap_source == "bootstrap"
    assert_allclose(analytic_error, [0.02, 0.03, 0.04])
    assert analytic_source == "analytic"


def test_error_bars_do_not_present_missing_uncertainty_as_zero() -> None:
    """Unavailable uncertainties should remain missing and hide the error bars."""
    error_config = analysis_module._error_array(np.array([np.nan, np.nan]))

    assert error_config["visible"] is False
    assert np.all(np.isnan(np.asarray(error_config["array"], dtype=float)))


@pytest.mark.parametrize(
    ("n_shots", "calibration_n_shots", "option_name"),
    [
        (1, None, "n_shots"),
        (None, 1, "calibration_n_shots"),
    ],
)
def test_characterization_rejects_too_few_shots_before_calibration(
    monkeypatch: pytest.MonkeyPatch,
    n_shots: int | None,
    calibration_n_shots: int | None,
    option_name: str,
) -> None:
    """Shot counts below two should fail before any calibration is attempted."""

    def unexpected_calibration(*args: object, **kwargs: object) -> None:
        raise AssertionError("calibration should not run")

    monkeypatch.setattr(module, "calibrate_gef_population", unexpected_calibration)

    with pytest.raises(ValueError, match=rf"{option_name}.*at least two"):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            n_values=[0, 1, 2],
            zx90_no_echo=_schedule(8.0),
            zx90_echo=_schedule(20.0),
            n_shots=n_shots,
            calibration_n_shots=calibration_n_shots,
            enable_tqdm=False,
            plot=False,
        )


def test_characterization_rejects_boolean_covariance_cutoff_before_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A boolean covariance cutoff should fail before hardware calibration."""

    def unexpected_calibration(*args: object, **kwargs: object) -> None:
        raise AssertionError("calibration should not run")

    monkeypatch.setattr(module, "calibrate_gef_population", unexpected_calibration)

    with pytest.raises(TypeError, match="covariance_rcond must be a real number"):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            n_values=[0, 1, 2],
            zx90_no_echo=_schedule(8.0),
            zx90_echo=_schedule(20.0),
            covariance_rcond=False,
            enable_tqdm=False,
            plot=False,
        )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("measure_orthogonal_components", 1),
        ("propagate_fidelity_uncertainty", 1),
        ("enable_tqdm", 1),
        ("plot", 1),
    ],
)
def test_characterization_rejects_nonboolean_flags_before_measurement(
    option: str,
    value: Any,
) -> None:
    """Boolean workflow flags should not rely on Python truthiness."""
    with pytest.raises(TypeError, match=option):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            **{option: value},
        )


@pytest.mark.parametrize(
    "protocols",
    [
        ["control_ground"],
        ["control_excited"],
        ["control_ground", "control_t2_echo"],
        ["control_excited", "target_t2rho_echo"],
    ],
)
def test_characterization_requires_ground_and_excited_as_a_pair(
    protocols: list[str],
) -> None:
    """Ground and excited protocols should be paired for the joint rate fit."""
    with pytest.raises(
        ValueError,
        match="control_ground and control_excited must be selected together",
    ):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            protocols=protocols,
            n_values=[0, 1, 2],
            zx90_no_echo=_schedule(8.0),
            zx90_echo=_schedule(20.0),
            enable_tqdm=False,
            plot=False,
        )


def test_fidelity_simulation_requires_all_protocols_before_measurement() -> None:
    """An incomplete protocol set should be rejected before hardware work."""
    with pytest.raises(ValueError, match="requires all four protocols"):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            protocols=["control_t2_echo", "target_t2rho_echo"],
            n_values=[0, 1, 2],
            zx90_echo=_schedule(20.0),
            run_fidelity_simulation=True,
            idle_t1={"Q0": 50_000.0, "Q1": 40_000.0},
            idle_t2_echo={"Q0": 70_000.0, "Q1": 60_000.0},
            enable_tqdm=False,
            plot=False,
        )


def test_explicit_uncertainty_requires_all_protocols_before_measurement() -> None:
    """Explicit uncertainty must reject a partial protocol selection."""
    with pytest.raises(ValueError, match="requires all four protocols"):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            protocols="control_t2_echo",
            propagate_fidelity_uncertainty=True,
        )


def test_explicit_uncertainty_conflicts_with_disabled_simulation() -> None:
    """Uncertainty cannot be explicitly enabled while simulation is disabled."""
    with pytest.raises(ValueError, match="requires fidelity simulation"):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            run_fidelity_simulation=False,
            propagate_fidelity_uncertainty=True,
        )


def test_fidelity_gate_metadata_is_validated_before_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic schedule should fail before starting GEF calibration."""

    def unexpected_calibration(*args: object, **kwargs: object) -> None:
        raise AssertionError("calibration should not run")

    monkeypatch.setattr(module, "calibrate_gef_population", unexpected_calibration)

    echo_gate = _schedule(20.0)

    with pytest.raises(ValueError, match=r"cr_duration.*echo"):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            n_values=[0, 1, 2],
            zx90_no_echo=_schedule(8.0),
            zx90_echo=echo_gate,
            run_fidelity_simulation=True,
            idle_t1={"Q0": 50_000.0, "Q1": 40_000.0},
            idle_t2_echo={"Q0": 70_000.0, "Q1": 60_000.0},
            enable_tqdm=False,
            plot=False,
        )


@pytest.mark.parametrize(
    ("parameter_name", "declared_echo"),
    [("zx90_no_echo", True), ("zx90_echo", False)],
)
def test_characterization_rejects_contradictory_gate_echo_metadata(
    monkeypatch: pytest.MonkeyPatch,
    parameter_name: str,
    declared_echo: bool,
) -> None:
    """A gate override must not contradict its requested protocol role."""

    def unexpected_calibration(*args: object, **kwargs: object) -> None:
        raise AssertionError("calibration should not run")

    no_echo = _schedule(8.0)
    echo = _schedule(20.0)
    gate = no_echo if parameter_name == "zx90_no_echo" else echo
    gate.echo = declared_echo  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "calibrate_gef_population", unexpected_calibration)

    with pytest.raises(ValueError, match=parameter_name):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            n_values=[0, 1, 2],
            zx90_no_echo=no_echo,
            zx90_echo=echo,
            run_fidelity_simulation=False,
            propagate_fidelity_uncertainty=False,
            enable_tqdm=False,
            plot=False,
        )


@pytest.mark.parametrize(
    "protocols",
    [[], ["control_t2_echo", "control_t2_echo"], ["C"]],
)
def test_characterization_rejects_invalid_protocol_selection(
    protocols: list[str],
) -> None:
    """Protocol selection should be nonempty, unique, and use descriptive names."""
    with pytest.raises(ValueError, match="protocols"):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            protocols=protocols,
            n_values=[0, 1, 2],
            zx90_echo=_schedule(20.0),
            enable_tqdm=False,
            plot=False,
        )


@pytest.mark.parametrize(
    ("protocol", "measured_qubit", "primary_basis"),
    [
        ("control_t2_echo", "Q0", "X"),
        ("target_t2rho_echo", "Q1", "Z"),
    ],
)
def test_characterization_can_measure_one_pauli_protocol_without_gef_calibration(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    measured_qubit: str,
    primary_basis: str,
) -> None:
    """A Pauli-only run should skip GEF calibration and unrelated protocols."""
    calls: list[tuple[str, str]] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("GEF work should not run")

    def fake_measure_pauli(
        _exp: object,
        _sequence: PulseSchedule,
        target: str,
        basis: str,
        **kwargs: object,
    ) -> Any:
        del kwargs
        calls.append((target, basis))
        value = 0.9 * np.exp(-0.1 * len(calls))
        return module._PauliMeasurement(
            expectation=value,
            standard_error=0.01,
            raw_iq=np.array([value + 0j]),
        )

    monkeypatch.setattr(module, "calibrate_gef_population", forbidden)
    monkeypatch.setattr(module, "measure_gef_populations", forbidden)
    monkeypatch.setattr(module, "bootstrap_gef_populations", forbidden)
    monkeypatch.setattr(module, "_measure_pauli_expectation", fake_measure_pauli)

    result = module.characterize_cr_pulse_coherence(
        _DummyExperiment(),  # type: ignore[arg-type]
        "Q0",
        "Q1",
        protocols=protocol,
        measure_orthogonal_components=False,
        n_values=[0, 1, 2],
        zx90_echo=_schedule(20.0),
        run_fidelity_simulation=False,
        enable_tqdm=False,
        plot=False,
    )

    measurements = result.data["measurements"]
    analysis = result.data["analysis"]
    assert calls == [(measured_qubit, primary_basis)] * 6
    assert measurements.protocols == (protocol,)
    assert measurements.pauli_components == {protocol: (primary_basis,)}
    assert analysis.fidelity_analysis is None
    assert result.data["raw_data"]["calibration"] is None
    assert measurements.populations == {}
    assert set(result.figures or {}) == {protocol}


def test_characterization_measures_and_plots_orthogonal_components_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Orthogonal components should be measured and plotted without fitting."""
    calls: list[tuple[str, str]] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("GEF work should not run")

    def fake_measure_pauli(
        _exp: object,
        _sequence: PulseSchedule,
        target: str,
        basis: str,
        **kwargs: object,
    ) -> Any:
        del kwargs
        calls.append((target, basis))
        value = 0.8 * np.exp(-0.02 * len(calls))
        return module._PauliMeasurement(
            expectation=value,
            standard_error=0.01,
            raw_iq=np.array([value + 0j]),
        )

    monkeypatch.setattr(module, "calibrate_gef_population", forbidden)
    monkeypatch.setattr(module, "measure_gef_populations", forbidden)
    monkeypatch.setattr(module, "bootstrap_gef_populations", forbidden)
    monkeypatch.setattr(module, "_measure_pauli_expectation", fake_measure_pauli)

    result = module.characterize_cr_pulse_coherence(
        _DummyExperiment(),  # type: ignore[arg-type]
        "Q0",
        "Q1",
        protocols=["control_t2_echo", "target_t2rho_echo"],
        n_values=[0, 1, 2],
        zx90_echo=_schedule(20.0),
        propagate_fidelity_uncertainty=False,
        enable_tqdm=False,
        plot=False,
    )

    expected_per_n = [
        *(("Q0", basis) for _ in range(2) for basis in ("X", "Y", "Z")),
        *(("Q1", basis) for _ in range(2) for basis in ("Z", "X", "Y")),
    ]
    assert calls == expected_per_n * 3
    measurements = result.data["measurements"]
    analysis = result.data["analysis"]
    assert measurements.pauli_components == {
        "control_t2_echo": ("X", "Y", "Z"),
        "target_t2rho_echo": ("Z", "X", "Y"),
    }
    pauli_expectations: Any = measurements.pauli_expectations
    assert set(pauli_expectations["control_t2_echo"]) == {"X", "Y", "Z"}
    assert "pauli_components" not in analysis.fits
    assert analysis.fits["phenomenological_echo_decay"]["control_t2_echo"][
        "actual"
    ].success
    assert "pauli_components" not in analysis.decay_times
    assert analysis.fit_status["forward_fit_success"] == {
        "control_t2_echo": False,
        "target_t2rho_echo": False,
    }
    control_t2_traces: Any = result.get_figure("control_t2_echo").data
    assert len(control_t2_traces) == 8
    trace_names = {trace.name for trace in control_t2_traces}
    assert trace_names >= {
        "reference <Y> data",
        "actual <Z> data",
    }
    assert not any(
        name.startswith("actual <Y>") and not name.endswith("data")
        for name in trace_names
    )
    assert not any(
        name.startswith("reference <Z>") and not name.endswith("data")
        for name in trace_names
    )
    actual_markers = {
        trace.marker.symbol
        for trace in control_t2_traces
        if trace.name.startswith("actual") and trace.name.endswith("data")
    }
    reference_markers = {
        trace.marker.symbol
        for trace in control_t2_traces
        if trace.name.startswith("reference") and trace.name.endswith("data")
    }
    assert len(actual_markers) == 3
    assert len(reference_markers) == 3
    assert actual_markers.isdisjoint(reference_markers)
    target_t2rho_traces: Any = result.get_figure("target_t2rho_echo").data
    assert len(target_t2rho_traces) == 8
    target_trace_names = {trace.name for trace in target_t2rho_traces}
    assert not any(
        name.startswith("actual <X>") and not name.endswith("data")
        for name in target_trace_names
    )
    assert not any(
        name.startswith("reference <Y>") and not name.endswith("data")
        for name in target_trace_names
    )


@pytest.mark.parametrize("n_values", [[1, 2], [0, 2, 1], [0, 1, 1], [0, -1]])
def test_characterization_rejects_invalid_n_values(n_values: list[int]) -> None:
    """The sweep should require unique increasing nonnegative n values starting at zero."""
    with pytest.raises(ValueError, match="n_values"):
        module.characterize_cr_pulse_coherence(
            _DummyExperiment(),  # type: ignore[arg-type]
            "Q0",
            "Q1",
            n_values=n_values,
            zx90_no_echo=_schedule(8.0),
            zx90_echo=_schedule(20.0),
            enable_tqdm=False,
            plot=False,
        )


@pytest.mark.parametrize(
    ("run_fidelity_simulation", "simulation_fails", "control_fit_fails"),
    [
        (None, False, False),
        (False, False, False),
        (None, True, False),
        (None, False, True),
    ],
)
def test_characterization_integrates_fidelity_simulation_and_forward_curves(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_fidelity_simulation: bool | None,
    simulation_fails: bool,
    control_fit_fails: bool,
) -> None:
    """Forward fitting should be independent of the final fidelity calculation."""
    exp = _DummyExperiment()
    calibration = {"Q0": object(), "Q1": object()}

    def fake_calibrate(*args: object, **kwargs: object) -> dict[str, object]:
        return calibration

    def fake_measure_gef(
        _exp: object,
        targets: object,
        sequences: dict[str, PulseSchedule],
        **kwargs: object,
    ) -> Result:
        del targets, kwargs
        populations = {
            name: {
                "Q0": np.array([0.8, 0.18, 0.02]),
                "Q1": np.array([0.2, 0.79, 0.01]),
            }
            for name in sequences
        }
        raw_iq = {
            name: {
                analyzer: {
                    "Q0": np.ones(4, dtype=complex),
                    "Q1": np.ones(4, dtype=complex),
                }
                for analyzer in ("s1", "s4", "s5")
            }
            for name in sequences
        }
        return Result(
            data={
                "populations": populations,
                "raw_iq": raw_iq,
                "fits": {name: {"Q0": object(), "Q1": object()} for name in sequences},
                "moment_summaries": {
                    name: {
                        analyzer: {"Q0": object(), "Q1": object()}
                        for analyzer in ("s1", "s4", "s5")
                    }
                    for name in sequences
                },
            }
        )

    def fake_bootstrap(
        _calibration: object,
        raw_iq: dict[str, object],
        **kwargs: object,
    ) -> dict[str, dict[str, Any]]:
        del kwargs
        return {
            name: {
                target: SimpleNamespace(
                    samples=np.tile(np.array([0.2, 0.79, 0.01]), (4, 1)),
                    standard_error=np.full(3, 0.01),
                    unavailable_reason=None,
                )
                for target in ("Q0", "Q1")
            }
            for name in raw_iq
        }

    pauli_values = iter([0.95, 0.9, 0.92, 0.85] * 3)

    def fake_measure_pauli(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        value = next(pauli_values)
        return module._PauliMeasurement(
            expectation=value,
            standard_error=0.01,
            raw_iq=np.array([value + 0j, value + 0j]),
        )

    fit_times = np.array([0.0, 1.0, 2.0])
    rates = (0.2, 0.05, 0.03, 0.01)
    rate_matrix = np.array(
        [
            [-rates[1], rates[0], 0.0],
            [rates[1], -(rates[0] + rates[3]), rates[2]],
            [0.0, rates[3], -rates[2]],
        ]
    )
    ground_initial = np.array([0.98, 0.01, 0.01])
    excited_initial = np.array([0.02, 0.96, 0.02])
    ground_population = np.stack(
        [expm(rate_matrix * time) @ ground_initial for time in fit_times]
    )
    excited_population = np.stack(
        [expm(rate_matrix * time) @ excited_initial for time in fit_times]
    )
    rate_fit = analysis_module.fit_three_level_rate_model(
        fit_times,
        ground_population,
        excited_population,
    )
    ground_t1rho_fit = analysis_module.fit_target_t1rho(
        fit_times,
        np.exp(-fit_times / 3),
    )
    excited_t1rho_fit = analysis_module.fit_target_t1rho(
        fit_times,
        -0.8 * np.exp(-fit_times / 4),
    )
    ground_leakage_fit = analysis_module.fit_target_leakage(
        fit_times,
        np.array([0.01, 0.03, 0.05]),
        relative_uncertainty_threshold=100.0,
    )
    excited_leakage_fit = analysis_module.fit_target_leakage(
        fit_times,
        np.array([0.02, 0.04, 0.06]),
        relative_uncertainty_threshold=100.0,
    )
    pauli_fit = analysis_module.fit_exponential_decay(
        fit_times,
        0.1 + 0.9 * np.exp(-fit_times / 3),
    )

    def fake_fit_all(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return {
            "control_rate_model": {"actual": rate_fit, "reference": rate_fit},
            "target_t1rho": {
                "control_ground": {
                    "actual": ground_t1rho_fit,
                    "reference": ground_t1rho_fit,
                },
                "control_excited": {
                    "actual": excited_t1rho_fit,
                    "reference": excited_t1rho_fit,
                },
            },
            "target_leakage": {
                "control_ground": {
                    "actual": ground_leakage_fit,
                    "reference": ground_leakage_fit,
                },
                "control_excited": {
                    "actual": excited_leakage_fit,
                    "reference": excited_leakage_fit,
                },
            },
            "phenomenological_echo_decay": {
                "control_t2_echo": {
                    "actual": pauli_fit,
                    "reference": pauli_fit,
                },
                "target_t2rho_echo": {
                    "actual": pauli_fit,
                    "reference": pauli_fit,
                },
            },
        }

    control_dephasing_fit = CrOnDephasingFit(
        success=True,
        message="ok",
        gamma_phi=1e-5,
        gamma_phi_error=1e-6,
        covariance=np.array([[1e-12]]),
        amplitude=0.9,
        offset=0.05,
        fitted_values=np.array([0.95, 0.9, 0.85]),
        curve_n_values=np.array([0, 1, 2]),
        curve_values=np.array([0.95, 0.9, 0.85]),
        r_squared=0.99,
    )
    target_dephasing_fit = CrOnDephasingFit(
        success=True,
        message="ok",
        gamma_phi=2e-5,
        gamma_phi_error=2e-6,
        covariance=np.array([[4e-12]]),
        amplitude=0.8,
        offset=0.1,
        fitted_values=np.array([0.9, 0.85, 0.8]),
        curve_n_values=np.array([0, 1, 2]),
        curve_values=np.array([0.9, 0.85, 0.8]),
        r_squared=0.99,
    )
    simulation = CrPulseFidelitySimulationResult(
        idle_coherence_limited_fidelity=0.999,
        cr_on_coherence_limited_fidelity=0.995,
        cr_on_dissipative_limited_fidelity=0.99,
        average_leakage=0.004,
        model_metadata={},
    )
    captured_base_noise: list[Any] = []
    captured_fitted_noise: list[Any] = []

    def fake_prepare(*args: object, **kwargs: object) -> object:
        del kwargs
        captured_base_noise.append(args[3])
        return object()

    def fake_control_forward_fit(*args: object, **kwargs: object) -> CrOnDephasingFit:
        del args, kwargs
        if control_fit_fails:
            return replace(
                control_dephasing_fit,
                success=False,
                message="control rate is not identifiable",
            )
        return control_dephasing_fit

    def fake_target_forward_fit(*args: object, **kwargs: object) -> CrOnDephasingFit:
        del args, kwargs
        return target_dephasing_fit

    def fake_simulate(
        *args: object, **kwargs: object
    ) -> CrPulseFidelitySimulationResult:
        del kwargs
        captured_fitted_noise.append(args[3])
        if simulation_fails:
            raise RuntimeError("nonphysical channel")
        return simulation

    monkeypatch.setattr(module, "calibrate_gef_population", fake_calibrate)
    monkeypatch.setattr(module, "measure_gef_populations", fake_measure_gef)
    monkeypatch.setattr(module, "bootstrap_gef_populations", fake_bootstrap)
    monkeypatch.setattr(module, "_measure_pauli_expectation", fake_measure_pauli)
    monkeypatch.setattr(analysis_module, "_fit_measured_observables", fake_fit_all)
    monkeypatch.setattr(analysis_module, "prepare_cr_echo_decay_model", fake_prepare)
    monkeypatch.setattr(
        analysis_module,
        "fit_cr_on_control_dephasing",
        fake_control_forward_fit,
    )
    monkeypatch.setattr(
        analysis_module,
        "fit_cr_on_target_rotating_frame_dephasing",
        fake_target_forward_fit,
    )
    monkeypatch.setattr(analysis_module, "simulate_cr_pulse_fidelity", fake_simulate)

    echo_gate = _schedule(20.0)
    echo_gate.cr_duration = 8.0  # type: ignore[attr-defined]
    echo_gate.echo = True  # type: ignore[attr-defined]
    echo_gate.pi_pulse = Blank(2.0)  # type: ignore[attr-defined]
    no_echo_gate = _schedule(8.0)
    no_echo_gate.cr_duration = 8.0  # type: ignore[attr-defined]
    no_echo_gate.echo = False  # type: ignore[attr-defined]
    result = module.characterize_cr_pulse_coherence(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        measure_orthogonal_components=False,
        n_values=[0, 1, 2],
        zx90_no_echo=no_echo_gate,
        zx90_echo=echo_gate,
        n_bootstrap=4,
        idle_t1={"Q0": 50_000.0, "Q1": 40_000.0},
        idle_t2_echo={"Q0": 70_000.0, "Q1": 60_000.0},
        run_fidelity_simulation=run_fidelity_simulation,
        propagate_fidelity_uncertainty=False,
        enable_tqdm=False,
        plot=False,
    )

    analysis = result.data["analysis"]
    fidelity_analysis = analysis.fidelity_analysis
    assert fidelity_analysis is not None
    noise = captured_base_noise[0]
    assert noise.gamma_control_g_to_e == pytest.approx(rate_fit.gamma_ge_up)
    assert noise.gamma_control_e_to_g == pytest.approx(rate_fit.gamma_ge_down)
    assert noise.gamma_control_e_to_f == pytest.approx(rate_fit.gamma_ef_up)
    assert noise.gamma_control_f_to_e == pytest.approx(rate_fit.gamma_ef_down)
    assert noise.target_t1rho == pytest.approx(24 / 7)
    assert noise.target_leakage_rate == pytest.approx(
        0.5 * (ground_leakage_fit.leakage_rate + excited_leakage_fit.leakage_rate)
    )
    t1rho_summary = analysis.decay_times["T1rho"]
    assert t1rho_summary["control_ground"]["actual"]["value"] == pytest.approx(3.0)
    assert t1rho_summary["control_excited"]["actual"]["value"] == pytest.approx(4.0)
    assert t1rho_summary["effective"]["actual"]["value"] == pytest.approx(24 / 7)
    leakage_summary = analysis.transition_rates["target_leakage"]
    assert leakage_summary["effective"]["actual"]["leakage_rate"][
        "value"
    ] == pytest.approx(noise.target_leakage_rate)
    assert analysis.fit_status["target_effective_rate_aggregation"] == {
        "target_t1rho": "arithmetic_mean_of_inverse_lifetimes",
        "target_leakage_rate": "arithmetic_mean",
        "target_seepage_rate": "arithmetic_mean",
    }
    returned_control_fit = analysis.fits["cr_on_dephasing"]["control_t2_echo"]
    assert analysis.fits["cr_on_dephasing"]["target_t2rho_echo"] is (
        target_dephasing_fit
    )
    assert returned_control_fit.success is not control_fit_fails
    assert analysis.fit_status["forward_fit_success"] == {
        "control_t2_echo": not control_fit_fails,
        "target_t2rho_echo": True,
    }
    assert (
        result.data["analysis_options"]["run_fidelity_simulation"]
        is run_fidelity_simulation
    )
    output = capsys.readouterr().out
    if control_fit_fails:
        assert captured_fitted_noise == []
        assert not fidelity_analysis.success
        assert fidelity_analysis.target_dephasing_fit is (target_dephasing_fit)
        assert fidelity_analysis.cr_on_noise is not None
        assert fidelity_analysis.cr_on_noise.gamma_phi_control == 0.0
        assert fidelity_analysis.cr_on_noise.gamma_phi_rho_target == pytest.approx(2e-5)
        assert "control rate is not identifiable" in output
        assert "gamma_phi_control" not in analysis.transition_rates["cr_on_dephasing"]
        assert "gamma_phi_rho_target" in analysis.transition_rates["cr_on_dephasing"]
    else:
        assert fidelity_analysis.cr_on_noise is not None
        assert fidelity_analysis.cr_on_noise.gamma_phi_control == pytest.approx(1e-5)
        assert fidelity_analysis.cr_on_noise.gamma_phi_rho_target == pytest.approx(2e-5)
    if (
        run_fidelity_simulation is None
        and not simulation_fails
        and not control_fit_fails
    ):
        assert captured_fitted_noise[0].gamma_phi_control == pytest.approx(1e-5)
        assert captured_fitted_noise[0].gamma_phi_rho_target == pytest.approx(2e-5)
        assert fidelity_analysis.simulation is simulation
        assert analysis.fit_status["fixed_zero_unresolved_rates"] == (
            "target_seepage_rate_ground",
            "target_seepage_rate_excited",
        )
        assert "CR-pulse fidelity simulation" in output
        assert "Idle coherence limit:" in output
        assert "99.900000%" in output
        assert "CR-on coherence limit:" in output
        assert "99.500000%" in output
        assert "CR-on dissipative limit:" in output
        assert "99.000000%" in output
        assert "Average leakage:" in output
        assert "0.400000%" in output
        assert "unresolved rates were fixed to zero" in output
        assert "target_seepage_rate" in output
    elif simulation_fails and not control_fit_fails:
        assert captured_fitted_noise[0].gamma_phi_control == pytest.approx(1e-5)
        assert fidelity_analysis.control_dephasing_fit is control_dephasing_fit
        assert fidelity_analysis.target_dephasing_fit is target_dephasing_fit
        assert not fidelity_analysis.success
        assert "nonphysical channel" in fidelity_analysis.message
        assert "CR-pulse fidelity analysis failed" in output
    elif not control_fit_fails:
        assert captured_fitted_noise == []
        assert fidelity_analysis.simulation is None
        assert "CR-pulse fidelity simulation" not in output
    control_traces: Any = result.get_figure("control_t2_echo").data
    actual_fit = next(
        trace
        for trace in control_traces
        if trace.name
        == (
            "actual <X> exponential diagnostic"
            if control_fit_fails
            else "actual <X> forward fit"
        )
    )
    if not control_fit_fails:
        assert np.asarray(actual_fit.y) == pytest.approx(
            control_dephasing_fit.curve_values
        )
    target_traces: Any = result.get_figure("target_t2rho_echo").data
    target_actual_fit = next(
        trace for trace in target_traces if trace.name == "actual <Z> forward fit"
    )
    assert np.asarray(target_actual_fit.y) == pytest.approx(
        target_dephasing_fit.curve_values
    )
