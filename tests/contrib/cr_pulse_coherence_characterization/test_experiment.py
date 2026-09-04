"""Tests for CR-pulse coherence measurement orchestration."""

# ruff: noqa: SLF001

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qubex.contrib.experiment import cr_pulse_coherence_characterization as module
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

    def z90(self) -> VirtualZ:
        """Return a virtual Z90 pulse."""
        return VirtualZ(np.pi / 2)

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

    for protocol in ("A", "B", "C", "D"):
        assert point.sequences[f"{protocol}_reference"].duration == pytest.approx(
            point.sequences[protocol].duration
        )
        assert point.evolution_durations[f"{protocol}_reference"] == pytest.approx(
            point.evolution_durations[protocol]
        )
        for condition in (protocol, f"{protocol}_reference"):
            sequence = point.sequences[condition]
            assert sequence.is_valid()
    assert point.evolution_durations["A"] == pytest.approx(4 * 3 * 8.0)
    assert point.evolution_durations["B"] == pytest.approx(4 * 3 * 8.0)
    assert point.cr_pulse_count == 12
    for protocol in ("A", "B"):
        actual = point.sequences[protocol].get_sampled_sequence("Q0-Q1")
        reference = point.sequences[f"{protocol}_reference"].get_sampled_sequence(
            "Q0-Q1"
        )
        unit_nonzero = np.count_nonzero(no_echo.get_sampled_sequence("Q0-Q1"))
        assert np.count_nonzero(actual) == 12 * unit_nonzero
        assert np.count_nonzero(reference) == 0
    a_reference_target = point.sequences["A_reference"].get_sampled_sequence("Q1")
    b_reference_target = point.sequences["B_reference"].get_sampled_sequence("Q1")
    assert np.max(a_reference_target.real) > 0
    assert np.min(a_reference_target.real) >= 0
    assert np.min(b_reference_target.real) < 0
    assert np.max(b_reference_target.real) <= 0
    for protocol in ("C", "D"):
        actual = point.sequences[protocol].get_sampled_sequence("Q0-Q1")
        reference = point.sequences[f"{protocol}_reference"].get_sampled_sequence(
            "Q0-Q1"
        )
        unit_nonzero = np.count_nonzero(echo.get_sampled_sequence("Q0-Q1"))
        assert np.count_nonzero(actual) == 12 * unit_nonzero
        assert np.count_nonzero(reference) == 0


def test_characterization_runs_requested_hardware_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration and A-through-D conditions should run in the requested order."""
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
            normalized_shots=np.array([value]),
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

    result = module.characterize_cr_pulse_coherence(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        n_values=[0, 1, 2],
        zx90_no_echo=_schedule(8.0),
        zx90_echo=_schedule(20.0),
        n_bootstrap=8,
        enable_tqdm=False,
        plot=False,
    )

    expected_per_n: list[object] = [
        ("gef", ("A_reference", "A", "B_reference", "B")),
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
    assert result.data["n_values"] == (0, 1, 2)
    assert result.data["cr_pulse_counts"] == (0, 4, 8)
    assert result.data["state_order"] == ("g", "e", "f")
    assert result.data["transition_rates"]["unit"] == "1/ns"
    assert result.data["decay_times"]["unit"] == "ns"
    assert result.data["sequence_durations"]["A"][0] == pytest.approx(4.0)
    assert result.data["sequence_durations"]["B"][0] == pytest.approx(6.0)
    assert result.data["sequence_durations"]["C"][0] == pytest.approx(2.0)
    assert set(result.figures or {}) == {
        "A_control",
        "A_target",
        "B_control",
        "B_target",
        "C_control",
        "D_target",
    }
    a_control = result.get_figure("A_control")
    a_control_data: Any = a_control.data
    assert a_control_data[0].name == "reference Pg"
    assert a_control_data[-1].name == "actual fit Pf"
    a_control_layout: Any = a_control.layout
    assert tuple(a_control_layout.yaxis.range) == (0.0, 1.0)
    assert a_control_layout.xaxis.range[0] == 0.0
    assert tuple(a_control_layout.xaxis2.ticktext) == ("0", "4", "8")
    a_target = result.get_figure("A_target")
    a_target_layout: Any = a_target.layout
    assert tuple(a_target_layout.yaxis.range) == (-1.05, 1.05)
    assert tuple(a_target_layout.yaxis2.range) == (0.0, 1.0)
    assert tuple(a_target_layout.xaxis3.ticktext) == ("0", "4", "8")


def test_pauli_measurement_uses_only_requested_basis_analyzer() -> None:
    """An X measurement should append -Y90 while a Z measurement appends nothing."""
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
    assert measured_z.expectation == pytest.approx(2.0)
    assert calls[0]["sequence"].duration == pytest.approx(10.0)  # type: ignore[union-attr]
    assert calls[1]["sequence"].duration == pytest.approx(8.0)  # type: ignore[union-attr]
    for call in calls:
        assert call["mode"] == "single"
        assert call["state_classification"] is False


def test_unavailable_bootstrap_does_not_report_polarization_error() -> None:
    """An unavailable GEF bootstrap should yield no target-polarization error."""
    bootstrap: Any = SimpleNamespace(
        unavailable_reason="success_rate_below_threshold",
        samples=np.array(
            [
                [0.8, 0.18, 0.02],
                [0.7, 0.27, 0.03],
            ]
        ),
    )

    assert np.isnan(module._polarization_standard_error(bootstrap))


def test_error_bars_do_not_present_missing_uncertainty_as_zero() -> None:
    """Unavailable uncertainties should remain missing and hide the error bars."""
    error_config = module._error_array(np.array([np.nan, np.nan]))

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
