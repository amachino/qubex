"""Tests for sampling-period usage in experiment services."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import PulseSchedule, Rect

import qubex.visualization as viz
from qubex.experiment.models.experiment_result import ExperimentResult
from qubex.experiment.services.characterization_service import CharacterizationService
from qubex.experiment.services.measurement_service import MeasurementService


def test_pulse_tomography_plot_uses_measurement_sampling_period(monkeypatch) -> None:
    """Given measurement dt, when plotting pulse tomography, then time axis uses that dt."""
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = SimpleNamespace(
        measurement=SimpleNamespace(sampling_period=0.4)
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        validate_rabi_params=lambda *_args: None
    )

    captured: dict[str, np.ndarray] = {}

    def _fake_state_evolution_tomography(
        self: MeasurementService,
        *,
        sequences: list[dict[str, object]],
        **_: object,
    ) -> dict[str, list[np.ndarray]]:
        return {"Q00": [np.array([0.0, 0.0, 1.0]) for _ in sequences]}

    monkeypatch.setattr(
        service,
        "state_evolution_tomography",
        MethodType(_fake_state_evolution_tomography, service),
    )
    monkeypatch.setattr(PulseSchedule, "plot", lambda self, title=None: None)
    monkeypatch.setattr(
        viz,
        "plot_bloch_vectors",
        lambda *, times, **kwargs: captured.__setitem__("times", np.asarray(times)),
    )

    with PulseSchedule(["Q00"]) as sequence:
        sequence.add("Q00", Rect(duration=4, amplitude=0.1))

    n_samples = 2
    service.pulse_tomography(sequence=sequence, n_samples=n_samples, plot=True)

    pulse_length = next(iter(sequence.get_sequences().values())).length
    if pulse_length < n_samples:
        indices = np.arange(pulse_length + 1)
    else:
        indices = np.linspace(0, pulse_length, n_samples).astype(int)
    expected_times = indices * 0.4

    assert np.allclose(captured["times"], expected_times)


def test_t2_experiment_discretization_uses_measurement_sampling_period() -> None:
    """Given measurement dt, when running T2 experiment, then discretization uses backend-derived dt."""
    captured: dict[str, float] = {}

    def _discretize_time_range(
        *,
        time_range: np.ndarray,
        sampling_period: float,
    ) -> np.ndarray:
        captured["sampling_period"] = sampling_period
        return np.array([], dtype=int)

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        measurement=SimpleNamespace(sampling_period=0.4),
        util=SimpleNamespace(
            discretize_time_range=_discretize_time_range,
            create_qubit_subgroups=lambda targets: [],
        ),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        validate_rabi_params=lambda _targets: None
    )

    result = service.t2_experiment(
        targets=["Q00"],
        time_range=np.array([100.0, 200.0]),
        n_cpmg=3,
        plot=False,
        save_image=False,
    )

    assert isinstance(result, ExperimentResult)
    assert captured["sampling_period"] == 2 * 0.4 * 3


def test_t1_experiment_discretization_uses_measurement_sampling_period() -> None:
    """Given measurement dt, when running T1 experiment, then discretization uses backend-derived dt."""
    captured: dict[str, float] = {}

    def _discretize_time_range(
        time_range: np.ndarray,
        *,
        sampling_period: float,
    ) -> np.ndarray:
        captured["sampling_period"] = sampling_period
        return np.array([], dtype=int)

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        measurement=SimpleNamespace(sampling_period=0.4),
        util=SimpleNamespace(
            discretize_time_range=_discretize_time_range,
            create_qubit_subgroups=lambda targets: [],
        ),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        validate_rabi_params=lambda _targets: None
    )

    result = service.t1_experiment(
        targets=["Q00"],
        time_range=np.array([100.0, 200.0]),
        plot=False,
        save_image=False,
    )

    assert isinstance(result, ExperimentResult)
    assert captured["sampling_period"] == 0.4


def test_ramsey_experiment_discretization_uses_measurement_sampling_period() -> None:
    """Given measurement dt, when running Ramsey experiment, then discretization uses backend-derived dt."""
    captured: dict[str, float] = {}

    def _discretize_time_range(
        time_range: np.ndarray,
        *,
        sampling_period: float,
    ) -> np.ndarray:
        captured["sampling_period"] = sampling_period
        return np.array([], dtype=int)

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        measurement=SimpleNamespace(sampling_period=0.4),
        util=SimpleNamespace(
            discretize_time_range=_discretize_time_range,
            create_qubit_subgroups=lambda targets: [],
        ),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        validate_rabi_params=lambda _targets: None
    )

    result = service.ramsey_experiment(
        targets=["Q00"],
        time_range=np.array([100.0, 200.0]),
        plot=False,
        save_image=False,
    )

    assert isinstance(result, ExperimentResult)
    assert captured["sampling_period"] == 0.4


def test_rabi_experiment_builds_control_pulse_with_measurement_sampling_period() -> (
    None
):
    """Given measurement dt, when building a Rabi sweep, then control pulses use that sampling period."""
    target = "Q00"
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = SimpleNamespace(
        targets={target: SimpleNamespace(frequency=5.0)},
        measurement=SimpleNamespace(sampling_period=0.4),
    )

    def _obtain_reference_points(
        self: MeasurementService,
        targets: list[str],
        **_: object,
    ) -> dict[str, dict[str, complex]]:
        return {"iq": dict.fromkeys(targets, 0j)}

    def _sweep_parameter(
        self: MeasurementService,
        *,
        sequence: Any,
        **_: object,
    ) -> Any:
        schedule = sequence(8.0)
        pulse_array = schedule.get_sequence(target, copy=False)
        waveforms = pulse_array.get_flattened_waveforms(apply_frame_shifts=True)
        pulse = waveforms[0]
        assert pulse.sampling_period == pytest.approx(0.4)
        raise RuntimeError("stop after sampling-period check")

    service.obtain_reference_points = MethodType(_obtain_reference_points, service)
    service.sweep_parameter = MethodType(_sweep_parameter, service)

    with pytest.raises(RuntimeError, match="sampling-period check"):
        service.rabi_experiment(
            amplitudes={target: 0.1},
            time_range=np.array([8.0], dtype=float),
            n_shots=1,
            shot_interval=1.0,
            plot=False,
        )
