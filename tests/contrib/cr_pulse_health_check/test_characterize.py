"""Tests for CR-pulse health-check measurement orchestration."""

# ruff: noqa: SLF001

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

import qubex.contrib.experiment.cr_pulse_health_check as health
from qubex.pulse import PulseSchedule


@dataclass
class FakeSchedule:
    """Provide the schedule metadata needed by orchestration tests."""

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


def make_zx90_overrides() -> tuple[PulseSchedule, PulseSchedule]:
    """Return full un-echoed and echoed ZX90 schedule stand-ins."""
    no_echo = FakeSchedule(
        duration=100.0,
        name="zx90_no_echo",
        cr_duration=100.0,
        echo=False,
    )
    echoed = FakeSchedule(
        duration=160.0,
        name="zx90_echo",
        cr_duration=50.0,
        echo=True,
    )
    return cast(PulseSchedule, no_echo), cast(PulseSchedule, echoed)


def test_characterize_collects_expected_times_and_measurements(
    fake_exp,
    install_characterize_stubs,
) -> None:
    """The workflow should collect matched timings and all primary measurements."""
    state = install_characterize_stubs()
    no_echo, echoed = make_zx90_overrides()

    result = health.characterize_cr_pulse_health(
        fake_exp,
        "Q0",
        "Q1",
        n_values=(0, 1, 2),
        zx90_no_echo=no_echo,
        zx90_echo=echoed,
        n_shots=256,
        calibration_n_shots=512,
        shot_interval=12_000.0,
        estimate_fidelity=False,
        enable_tqdm=False,
        plot=False,
    )

    measurements = result.data["measurements"]

    np.testing.assert_allclose(
        measurements.total_times["control_ground"],
        [0.0, 400.0, 800.0],
    )
    np.testing.assert_allclose(
        measurements.cr_active_times["control_ground"],
        [0.0, 400.0, 800.0],
    )
    np.testing.assert_allclose(
        measurements.total_times["control_t2_echo"],
        [0.0, 640.0, 1280.0],
    )
    np.testing.assert_allclose(
        measurements.total_times["target_t2rho_echo"],
        [0.0, 640.0, 1280.0],
    )
    np.testing.assert_allclose(
        measurements.cr_active_times["control_t2_echo"],
        [0.0, 400.0, 800.0],
    )
    np.testing.assert_allclose(
        measurements.cr_active_times["target_t2rho_echo"],
        [0.0, 400.0, 800.0],
    )

    assert len(state.gef_calls) == 3
    assert state.pauli_request_counts == [4, 4, 4]
    assert all(call["n_bootstrap"] == 0 for call in state.gef_calls)
    assert all(call["n_shots"] == 256 for call in state.gef_calls)

    assert len(state.analyze_calls) == 1
    analyzed_measurements, analyze_kwargs = state.analyze_calls[0]
    assert analyzed_measurements is measurements
    assert analyze_kwargs["estimate_fidelity"] is False

    assert result.data["measurement_options"]["population_uncertainty_method"] == (
        "analytic_GLS_full_covariance_and_component_SE_no_bootstrap"
    )
    assert measurements.population_covariances is not None
    assert measurements.population_covariances["control_ground"]["actual"][
        "Q0"
    ].shape == (3, 3, 3)
    assert result.figure is None
    assert result.figures == {}
    assert result.data["pulse_timing"]["zx90_no_echo"].cr_active_duration == 100.0
    assert result.data["pulse_timing"]["zx90_echo"].cr_active_duration == 100.0


def test_characterize_diagnostic_mode_requests_orthogonal_pauli_components(
    fake_exp,
    install_characterize_stubs,
) -> None:
    """Diagnostic mode should acquire every documented orthogonal component."""
    state = install_characterize_stubs()
    no_echo, echoed = make_zx90_overrides()

    result = health.characterize_cr_pulse_health(
        fake_exp,
        "Q0",
        "Q1",
        n_values=(0, 1, 2),
        measure_diagnostic_components=True,
        zx90_no_echo=no_echo,
        zx90_echo=echoed,
        estimate_fidelity=False,
        enable_tqdm=False,
        plot=False,
    )

    measurements = result.data["measurements"]
    assert measurements.diagnostic_components_measured
    assert state.pauli_request_counts == [28, 28, 28]

    control_key = health._diagnostic_key("control_ground", "control")
    target_key = health._diagnostic_key("control_ground", "target")
    assert set(measurements.pauli_expectations[control_key]) == {"X", "Y"}
    assert set(measurements.pauli_expectations[target_key]) == {"Y", "Z"}
    assert set(measurements.pauli_expectations["control_t2_echo"]) == {"X", "Y", "Z"}
    assert set(measurements.pauli_expectations["target_t2rho_echo"]) == {"X", "Y", "Z"}


def test_characterize_warns_when_target_f_population_exceeds_threshold(
    fake_exp,
    install_characterize_stubs,
) -> None:
    """The workflow should warn when actual target F population exceeds its limit."""
    install_characterize_stubs(target_f=0.02)
    no_echo, echoed = make_zx90_overrides()

    with pytest.warns(RuntimeWarning, match="Maximum measured target F population"):
        health.characterize_cr_pulse_health(
            fake_exp,
            "Q0",
            "Q1",
            n_values=(0, 1, 2),
            zx90_no_echo=no_echo,
            zx90_echo=echoed,
            target_leakage_warning_threshold=0.01,
            estimate_fidelity=False,
            enable_tqdm=False,
            plot=False,
        )


def test_characterize_rejects_identical_control_and_target(fake_exp) -> None:
    """The workflow should reject identical canonical control and target labels."""
    with pytest.raises(ValueError, match="must differ"):
        health.characterize_cr_pulse_health(
            fake_exp,
            "Q0",
            "Q0",
            n_values=(0, 1, 2),
            estimate_fidelity=False,
            enable_tqdm=False,
            plot=False,
        )


def test_characterize_rejects_analysis_threshold_before_calibration(
    fake_exp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid analysis thresholds should fail before hardware calibration starts."""
    no_echo, echoed = make_zx90_overrides()

    def unexpected_calibration(*args: object, **kwargs: object) -> None:
        raise AssertionError("calibration should not run")

    monkeypatch.setattr(health, "calibrate_gef_population", unexpected_calibration)

    with pytest.raises(ValueError, match="pauli_minimum_change"):
        health.characterize_cr_pulse_health(
            fake_exp,
            "Q0",
            "Q1",
            n_values=(0, 1, 2),
            zx90_no_echo=no_echo,
            zx90_echo=echoed,
            pauli_minimum_change=-0.1,
            estimate_fidelity=False,
            enable_tqdm=False,
            plot=False,
        )
