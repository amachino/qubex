"""Shared fixtures for CR dissipation tests."""

# ruff: noqa: SLF001

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import qubex.contrib.experiment.cr_dissipation as health


@dataclass
class FakeSchedule:
    """Small schedule stand-in used only by characterize() orchestration tests."""

    duration: float
    name: str = "schedule"
    cr_duration: float | None = None
    echo: bool = False

    def repeated(self, count: int) -> FakeSchedule:
        """Return a duration-scaled schedule stand-in."""
        return FakeSchedule(
            duration=self.duration * count,
            name=f"{self.name}*{count}",
            cr_duration=(
                None if self.cr_duration is None else self.cr_duration * count
            ),
            echo=self.echo,
        )

    def get_frequencies(self) -> dict[str, float]:
        """Return deterministic frequency metadata."""
        return {"fake": 5.0}


class FakePulse:
    """Provide the single-qubit pulses used by reference construction."""

    def x90(self, target: str) -> tuple[str, str]:
        """Return a positive-X90 stand-in."""
        return ("x90", target)

    def x90m(self, target: str) -> tuple[str, str]:
        """Return a negative-X90 stand-in."""
        return ("x90m", target)


class FakeContext:
    """Resolve already-canonical fake qubit labels."""

    def __init__(self) -> None:
        """Provide config-backed measurement defaults."""
        self.experiment_system = SimpleNamespace(
            measurement_defaults={
                "execution": {"shot_interval_ns": 500_000.0},
            }
        )

    def resolve_qubit_label(self, label: str) -> str:
        """Return the supplied fake qubit label."""
        return label


class FakeExperiment:
    """Expose the context and pulse interfaces needed by characterize tests."""

    def __init__(self) -> None:
        self.ctx = FakeContext()
        self.pulse = FakePulse()


@pytest.fixture
def fake_exp() -> FakeExperiment:
    """Return a hardware-free experiment stand-in."""
    return FakeExperiment()


def make_target_population(
    times: np.ndarray,
    *,
    x_rate: float,
    leakage_rate: float,
) -> np.ndarray:
    """Build a normalized target GEF trajectory with X decay and leakage."""
    x = np.exp(-x_rate * times)
    pf = 1.0 - np.exp(-leakage_rate * times)
    computational = 1.0 - pf
    pg = computational * (1.0 + x) / 2.0
    pe = computational * (1.0 - x) / 2.0
    return np.column_stack([pg, pe, pf])


@pytest.fixture
def synthetic_measurements() -> tuple[
    health.CrDissipationMeasurements, dict[str, float]
]:
    """Return deterministic measurements with known effective rates."""
    n_values = (0, 1, 2, 3, 5, 8, 13)
    n = np.asarray(n_values, dtype=np.float64)

    control_times = 400.0 * n
    echo_total_times = 800.0 * n
    echo_cr_times = 400.0 * n

    expected = {
        "gamma_control_e_to_g": 5.0e-5,
        "gamma_control_g_to_e": 3.0e-5,
        "gamma_control_e_to_f": 2.0e-5,
        "gamma_target_t1rho": 8.0e-5,
        "gamma_target_leakage": 2.0e-5,
        "gamma_control_transverse": 1.2e-4,
        "gamma_target_t2rho": 1.0e-4,
    }

    control_rates = {
        "gamma_e_to_g": expected["gamma_control_e_to_g"],
        "gamma_g_to_e": expected["gamma_control_g_to_e"],
        "gamma_f_to_e": 0.0,
        "gamma_e_to_f": expected["gamma_control_e_to_f"],
    }
    ground_initial = np.array([1.0, 0.0, 0.0])
    excited_initial = np.array([0.0, 1.0, 0.0])
    control_ground_cr_population_actual = health._population_trajectory(
        control_times, ground_initial, control_rates
    )
    control_excited_cr_population_actual = health._population_trajectory(
        control_times, excited_initial, control_rates
    )
    control_ground_cr_population_reference = np.tile(ground_initial, (n.size, 1))
    control_excited_cr_population_reference = np.tile(excited_initial, (n.size, 1))

    target_ground_actual = make_target_population(
        control_times,
        x_rate=expected["gamma_target_t1rho"],
        leakage_rate=expected["gamma_target_leakage"],
    )
    target_excited_actual = make_target_population(
        control_times,
        x_rate=expected["gamma_target_t1rho"],
        leakage_rate=expected["gamma_target_leakage"],
    )
    target_ground_reference = make_target_population(
        control_times, x_rate=1.0e-5, leakage_rate=0.0
    )
    target_excited_reference = make_target_population(
        control_times, x_rate=1.0e-5, leakage_rate=0.0
    )

    population_se = np.full((n.size, 3), 2.0e-4)
    target_x_se = np.full(n.size, 2.0e-4)
    pauli_se = np.full(n.size, 2.0e-4)

    duty = 0.5
    control_ref_active_rate = 2.0e-5
    control_idle_rate = 1.0 / 24_000.0
    control_ref_rate = duty * control_ref_active_rate + (1.0 - duty) * control_idle_rate
    control_actual_total_rate = (
        duty * expected["gamma_control_transverse"] + (1.0 - duty) * control_idle_rate
    )
    target_ref_active_rate = 1.0e-5
    target_idle_rate = 1.0 / 28_000.0
    target_ref_rate = duty * target_ref_active_rate + (1.0 - duty) * target_idle_rate
    target_actual_total_rate = (
        duty * expected["gamma_target_t2rho"] + (1.0 - duty) * target_idle_rate
    )

    measurements = health.CrDissipationMeasurements(
        control_qubit="Q0",
        target_qubit="Q1",
        n_values=n_values,
        total_times={
            "control_ground_cr_population": control_times,
            "control_excited_cr_population": control_times,
            "control_cr_transverse_echo": echo_total_times,
            "target_cr_rotating_frame_echo": echo_total_times,
        },
        cr_active_times={
            "control_ground_cr_population": control_times,
            "control_excited_cr_population": control_times,
            "control_cr_transverse_echo": echo_cr_times,
            "target_cr_rotating_frame_echo": echo_cr_times,
        },
        populations={
            "control_ground_cr_population": {
                "actual": {
                    "Q0": control_ground_cr_population_actual,
                    "Q1": target_ground_actual,
                },
                "reference": {
                    "Q0": control_ground_cr_population_reference,
                    "Q1": target_ground_reference,
                },
            },
            "control_excited_cr_population": {
                "actual": {
                    "Q0": control_excited_cr_population_actual,
                    "Q1": target_excited_actual,
                },
                "reference": {
                    "Q0": control_excited_cr_population_reference,
                    "Q1": target_excited_reference,
                },
            },
        },
        population_standard_errors={
            protocol: {
                kind: {"Q0": population_se.copy(), "Q1": population_se.copy()}
                for kind in ("actual", "reference")
            }
            for protocol in (
                "control_ground_cr_population",
                "control_excited_cr_population",
            )
        },
        target_x_standard_errors={
            protocol: {kind: target_x_se.copy() for kind in ("actual", "reference")}
            for protocol in (
                "control_ground_cr_population",
                "control_excited_cr_population",
            )
        },
        pauli_expectations={
            "control_cr_transverse_echo": {
                "X": {
                    "actual": np.exp(-control_actual_total_rate * echo_total_times),
                    "reference": np.exp(-control_ref_rate * echo_total_times),
                }
            },
            "target_cr_rotating_frame_echo": {
                "Y": {
                    "actual": np.exp(-target_actual_total_rate * echo_total_times),
                    "reference": np.exp(-target_ref_rate * echo_total_times),
                }
            },
        },
        pauli_standard_errors={
            "control_cr_transverse_echo": {
                "X": {
                    "actual": pauli_se.copy(),
                    "reference": pauli_se.copy(),
                }
            },
            "target_cr_rotating_frame_echo": {
                "Y": {
                    "actual": pauli_se.copy(),
                    "reference": pauli_se.copy(),
                }
            },
        },
        diagnostic_components_measured=False,
    )
    return measurements, expected


@pytest.fixture
def install_characterize_stubs(monkeypatch: pytest.MonkeyPatch):
    """Install hardware-free stubs while leaving characterize() itself unpatched."""

    def install(*, control_f: float = 0.01, target_f: float = 0.002) -> SimpleNamespace:
        state = SimpleNamespace(
            gef_calls=[],
            pauli_request_counts=[],
            analyze_calls=[],
            reference_pulses=[],
        )

        def reference_unit(
            labels: object, duration: float, **kwargs: object
        ) -> FakeSchedule:
            state.reference_pulses.append(kwargs.get("pulse"))
            return FakeSchedule(duration=duration, name="reference")

        monkeypatch.setattr(
            health,
            "_reference_unit",
            reference_unit,
        )
        monkeypatch.setattr(
            health,
            "_control_cr_transverse_echo_block",
            lambda exp, control, target, zx90: FakeSchedule(
                duration=4.0 * zx90.duration, name="control_echo_block"
            ),
        )
        monkeypatch.setattr(
            health,
            "_target_cr_rotating_frame_echo_block",
            lambda exp, control, target, zx90: FakeSchedule(
                duration=4.0 * zx90.duration, name="target_echo_block"
            ),
        )
        monkeypatch.setattr(
            health,
            "_control_state_base_sequence",
            lambda exp, control, target, control_state, evolution: evolution,
        )
        monkeypatch.setattr(
            health,
            "_control_state_gef_sequence",
            lambda exp, base, target: base,
        )
        monkeypatch.setattr(
            health,
            "_echo_protocol_sequence",
            lambda exp, control, target, protocol, evolution: evolution,
        )
        monkeypatch.setattr(
            health,
            "_append_pauli_analyzer",
            lambda exp, sequence, target, basis: sequence,
        )
        monkeypatch.setattr(
            health,
            "_load_idle_noise",
            lambda *args, **kwargs: (
                health.IdleNoiseParameters(30_000.0, 24_000.0, 18_000.0),
                health.IdleNoiseParameters(35_000.0, 28_000.0, 20_000.0),
            ),
        )

        calibration = {"Q0": object(), "Q1": object()}
        monkeypatch.setattr(
            health,
            "calibrate_gef_population",
            lambda exp, targets, n_shots, shot_interval: calibration,
        )

        def measure_gef_populations(
            exp: Any,
            *,
            targets: list[str],
            sequences: dict[str, FakeSchedule],
            calibration: Any,
            n_shots: int,
            shot_interval: float,
            covariance_rcond: float,
            n_bootstrap: int,
        ) -> SimpleNamespace:
            if sequences and all(name.startswith("pauli_") for name in sequences):
                state.pauli_request_counts.append(len(sequences))
            state.gef_calls.append(
                {
                    "targets": tuple(targets),
                    "sequences": dict(sequences),
                    "n_shots": n_shots,
                    "shot_interval": shot_interval,
                    "covariance_rcond": covariance_rcond,
                    "n_bootstrap": n_bootstrap,
                }
            )
            populations: dict[str, dict[str, np.ndarray]] = {}
            fits: dict[str, dict[str, object]] = {}
            raw_iq: dict[str, dict[str, dict[str, np.ndarray]]] = {}
            moments: dict[str, object] = {}
            for condition in sequences:
                populations[condition] = {
                    "Q0": np.array([0.9, 0.1 - control_f, control_f]),
                    "Q1": np.array([0.55, 0.45 - target_f, target_f]),
                }
                fits[condition] = {"Q0": object(), "Q1": object()}
                raw_iq[condition] = {
                    configuration: {
                        "Q0": np.array([0.0 + 0.0j]),
                        "Q1": np.array([0.0 + 0.0j]),
                    }
                    for configuration in ("s1", "s4", "s5")
                }
                moments[condition] = {"fake": True}
            return SimpleNamespace(
                data={
                    "populations": populations,
                    "fits": fits,
                    "raw_iq": raw_iq,
                    "moment_summaries": moments,
                }
            )

        monkeypatch.setattr(health, "measure_gef_populations", measure_gef_populations)
        monkeypatch.setattr(
            health,
            "_analytic_population_error",
            lambda fit: np.array([1.0e-3, 1.0e-3, 1.0e-3]),
        )
        monkeypatch.setattr(
            health,
            "_analytic_computational_polarization_error",
            lambda fit: 1.0e-3,
        )

        analysis = SimpleNamespace(name="analysis")

        def analyze(measurements: Any, **kwargs: Any) -> SimpleNamespace:
            state.analyze_calls.append((measurements, kwargs))
            return analysis

        monkeypatch.setattr(health, "analyze_cr_dissipation", analyze)
        monkeypatch.setattr(
            health,
            "plot_cr_dissipation",
            lambda measurements, analysis: {"summary": object()},
        )
        monkeypatch.setattr(health, "_print_dissipation_summary", lambda analysis: None)
        monkeypatch.setattr(
            health,
            "Result",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )
        return state

    return install
