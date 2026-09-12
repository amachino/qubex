"""Shared fixtures for CR-pulse health-check tests."""

# ruff: noqa: SLF001

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import qubex.contrib.experiment.cr_pulse_health_check as health


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
    pg = computational * (1.0 - x) / 2.0
    pe = computational * (1.0 + x) / 2.0
    return np.column_stack([pg, pe, pf])


@pytest.fixture
def synthetic_measurements() -> tuple[
    health.CrPulseHealthMeasurements, dict[str, float]
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
        "gamma_target_x_decay": 8.0e-5,
        "gamma_target_leakage": 2.0e-5,
        "gamma_control_xy_decay": 1.2e-4,
        "gamma_target_z_decay": 1.0e-4,
    }

    control_rates = {
        "gamma_e_to_g": expected["gamma_control_e_to_g"],
        "gamma_g_to_e": expected["gamma_control_g_to_e"],
        "gamma_f_to_e": 0.0,
        "gamma_e_to_f": expected["gamma_control_e_to_f"],
    }
    ground_initial = np.array([1.0, 0.0, 0.0])
    excited_initial = np.array([0.0, 1.0, 0.0])
    control_ground_actual = health._population_trajectory(
        control_times, ground_initial, control_rates
    )
    control_excited_actual = health._population_trajectory(
        control_times, excited_initial, control_rates
    )
    control_ground_reference = np.tile(ground_initial, (n.size, 1))
    control_excited_reference = np.tile(excited_initial, (n.size, 1))

    target_ground_actual = make_target_population(
        control_times,
        x_rate=expected["gamma_target_x_decay"],
        leakage_rate=expected["gamma_target_leakage"],
    )
    target_excited_actual = make_target_population(
        control_times,
        x_rate=expected["gamma_target_x_decay"],
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
    control_ref_rate = 2.0e-5
    control_actual_total_rate = (
        duty * expected["gamma_control_xy_decay"] + (1.0 - duty) * control_ref_rate
    )
    target_ref_rate = 1.0e-5
    target_actual_total_rate = (
        duty * expected["gamma_target_z_decay"] + (1.0 - duty) * target_ref_rate
    )

    measurements = health.CrPulseHealthMeasurements(
        control_qubit="Q0",
        target_qubit="Q1",
        n_values=n_values,
        total_times={
            "control_ground": control_times,
            "control_excited": control_times,
            "control_t2_echo": echo_total_times,
            "target_t2rho_echo": echo_total_times,
        },
        cr_active_times={
            "control_ground": control_times,
            "control_excited": control_times,
            "control_t2_echo": echo_cr_times,
            "target_t2rho_echo": echo_cr_times,
        },
        populations={
            "control_ground": {
                "actual": {
                    "Q0": control_ground_actual,
                    "Q1": target_ground_actual,
                },
                "reference": {
                    "Q0": control_ground_reference,
                    "Q1": target_ground_reference,
                },
            },
            "control_excited": {
                "actual": {
                    "Q0": control_excited_actual,
                    "Q1": target_excited_actual,
                },
                "reference": {
                    "Q0": control_excited_reference,
                    "Q1": target_excited_reference,
                },
            },
        },
        population_standard_errors={
            protocol: {
                kind: {"Q0": population_se.copy(), "Q1": population_se.copy()}
                for kind in ("actual", "reference")
            }
            for protocol in ("control_ground", "control_excited")
        },
        target_x_standard_errors={
            protocol: {kind: target_x_se.copy() for kind in ("actual", "reference")}
            for protocol in ("control_ground", "control_excited")
        },
        pauli_expectations={
            "control_t2_echo": {
                "X": {
                    "actual": np.exp(-control_actual_total_rate * echo_total_times),
                    "reference": np.exp(-control_ref_rate * echo_total_times),
                }
            },
            "target_t2rho_echo": {
                "Z": {
                    "actual": np.exp(-target_actual_total_rate * echo_total_times),
                    "reference": np.exp(-target_ref_rate * echo_total_times),
                }
            },
        },
        pauli_standard_errors={
            "control_t2_echo": {
                "X": {
                    "actual": pauli_se.copy(),
                    "reference": pauli_se.copy(),
                }
            },
            "target_t2rho_echo": {
                "Z": {
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

    def install(*, target_f: float = 0.002) -> SimpleNamespace:
        state = SimpleNamespace(
            gef_calls=[],
            pauli_request_counts=[],
            analyze_calls=[],
        )

        monkeypatch.setattr(
            health,
            "_reference_unit",
            lambda labels, duration, **kwargs: FakeSchedule(
                duration=duration, name="reference"
            ),
        )
        monkeypatch.setattr(
            health,
            "_control_t2_echo_block",
            lambda exp, control, target, zx90: FakeSchedule(
                duration=4.0 * zx90.duration, name="control_echo_block"
            ),
        )
        monkeypatch.setattr(
            health,
            "_target_t2rho_echo_block",
            lambda exp, control, target, zx90: FakeSchedule(
                duration=2.0 * zx90.duration, name="target_echo_block"
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
            raw_iq: dict[str, dict[str, np.ndarray]] = {}
            moments: dict[str, object] = {}
            for condition in sequences:
                populations[condition] = {
                    "Q0": np.array([0.9, 0.09, 0.01]),
                    "Q1": np.array([0.55, 0.45 - target_f, target_f]),
                }
                fits[condition] = {"Q0": object(), "Q1": object()}
                raw_iq[condition] = {
                    "Q0": np.array([0.0 + 0.0j]),
                    "Q1": np.array([0.0 + 0.0j]),
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
            "_analytic_target_x_error",
            lambda fit: 1.0e-3,
        )

        def measure_pauli_batch(
            exp: Any,
            requests: list[tuple[FakeSchedule, str, str]],
            *,
            n_shots: int,
            shot_interval: float,
        ) -> list[SimpleNamespace]:
            state.pauli_request_counts.append(len(requests))
            return [
                SimpleNamespace(
                    expectation=0.75,
                    standard_error=1.0e-3,
                    raw_iq=np.array([0.0 + 0.0j]),
                )
                for _ in requests
            ]

        monkeypatch.setattr(health, "_measure_pauli_batch", measure_pauli_batch)

        analysis = SimpleNamespace(name="analysis")

        def analyze(measurements: Any, **kwargs: Any) -> SimpleNamespace:
            state.analyze_calls.append((measurements, kwargs))
            return analysis

        monkeypatch.setattr(health, "analyze_cr_pulse_health", analyze)
        monkeypatch.setattr(
            health,
            "plot_cr_pulse_health",
            lambda measurements, analysis: {"summary": object()},
        )
        monkeypatch.setattr(health, "_print_health_summary", lambda analysis: None)
        monkeypatch.setattr(
            health,
            "Result",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )
        return state

    return install
