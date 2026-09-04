"""Tests for async batched randomized benchmarking execution."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from plotly.graph_objects import Figure
from qxpulse import Blank, PulseArray, PulseSchedule

from qubex.experiment.services import benchmarking_service as benchmarking_module
from qubex.experiment.services.benchmarking_service import BenchmarkingService
from qubex.measurement import (
    CaptureData,
    MeasurementResult,
    SweepMeasurementResult,
    SweepValue,
)
from qubex.measurement.models.measurement_config import MeasurementConfig


def test_benchmarking_service_api_remains_synchronous() -> None:
    """Benchmarking entry points should retain their synchronous API contract."""
    method_names = (
        "rb_experiment_1q",
        "rb_experiment_2q",
        "irb_experiment",
        "randomized_benchmarking",
        "interleaved_randomized_benchmarking",
        "benchmark_1q",
        "benchmark_2q",
    )

    assert all(
        not inspect.iscoroutinefunction(getattr(BenchmarkingService, method_name))
        for method_name in method_names
    )


def test_rb_trial_sweep_is_async() -> None:
    """The internal trial sweep should remain available for async orchestration."""
    assert inspect.iscoroutinefunction(
        BenchmarkingService._run_rb_trial_sweep  # noqa: SLF001
    )


def _patch_fit(monkeypatch: Any) -> None:
    """Replace RB fitting with a deterministic lightweight result."""
    monkeypatch.setattr(
        benchmarking_module.fitting,
        "fit_rb",
        lambda **_kwargs: {"fig": Figure()},
    )


def _make_sweep_result(
    *,
    sweep_values: Sequence[int],
    captures_by_target: dict[str, list[np.ndarray]],
    n_shots: int,
    shot_averaging: bool,
    time_integration: bool = True,
) -> SweepMeasurementResult:
    """Build a canonical sweep result with one final capture per target."""
    config = MeasurementConfig(
        n_shots=n_shots,
        shot_interval=1024.0,
        shot_averaging=shot_averaging,
        time_integration=time_integration,
        state_classification=False,
    )
    results = [
        MeasurementResult(
            data={
                target: [
                    CaptureData.from_primary_data(
                        target=target,
                        data=captures[point_index],
                        config=config,
                        sampling_period=0.8,
                    )
                ]
                for target, captures in captures_by_target.items()
            },
            measurement_config=config,
        )
        for point_index in range(len(sweep_values))
    ]
    return SweepMeasurementResult(
        sweep_values=[cast(SweepValue, value) for value in sweep_values],
        config=config,
        results=results,
    )


def test_rb_1q_awaits_canonical_trial_sweeps(monkeypatch: Any) -> None:
    """1Q RB should await one canonical sweep per Clifford length."""
    _patch_fit(monkeypatch)
    seeds = np.array([11, 22], dtype=int)
    sequence_calls: list[tuple[int, int]] = []
    sweep_calls: list[dict[str, object]] = []
    reset_calls: list[set[str]] = []
    normalized_signal = {11: 0.8, 22: 0.4}

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(is_cr=False)
        ),
        resolve_qubit_label=lambda label: label,
        reset_awg_and_capunits=lambda *, qubits: reset_calls.append(set(qubits)),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        rabi_params={"Q00": SimpleNamespace(normalize=lambda value: np.real(value))}
    )
    service.__dict__["_clifford_generator"] = None

    async def _run_sweep_measurement(
        schedule: Any,
        *,
        sweep_values: Any,
        **kwargs: object,
    ) -> SweepMeasurementResult:
        values = [int(value) for value in sweep_values]
        schedules = [schedule(value) for value in values]
        sweep_calls.append({"values": values, "schedules": schedules, **kwargs})
        return _make_sweep_result(
            sweep_values=values,
            captures_by_target={
                "Q00": [
                    np.asarray(
                        [normalized_signal[value] / 2] * 2,
                        dtype=np.complex128,
                    )
                    for value in values
                ]
            },
            n_shots=128,
            shot_averaging=True,
            time_integration=False,
        )

    def _legacy_measure(**_kwargs: object) -> None:
        raise AssertionError("legacy measure must not be called by RB")

    service.__dict__["_measurement_service"] = SimpleNamespace(
        measure=_legacy_measure,
        run_sweep_measurement=_run_sweep_measurement,
    )

    def _rb_sequence_1q(
        self: BenchmarkingService,
        target: str,
        *,
        n: int,
        seed: int,
        **_kwargs: object,
    ) -> PulseArray:
        assert target == "Q00"
        sequence_calls.append((n, seed))
        return PulseArray([])

    service.__dict__["rb_sequence_1q"] = MethodType(_rb_sequence_1q, service)

    result = service.rb_experiment_1q(
        targets="Q00",
        n_cliffords_range=[1, 2],
        n_trials=2,
        seeds=seeds,
        shots=128,
        interval=2048.0,
        time_integration=False,
        plot=False,
        save_image=False,
        reset_awg_and_capunits=True,
    )

    assert len(sweep_calls) == 2
    assert [call["values"] for call in sweep_calls] == [[11, 22], [11, 22]]
    assert all(
        isinstance(schedule, PulseSchedule)
        for call in sweep_calls
        for schedule in cast(list[object], call["schedules"])
    )
    assert all(call["n_shots"] == 128 for call in sweep_calls)
    assert all(call["shot_interval"] == 2048.0 for call in sweep_calls)
    assert all(call["shot_averaging"] is True for call in sweep_calls)
    assert all(call["time_integration"] is False for call in sweep_calls)
    assert all(call["state_classification"] is False for call in sweep_calls)
    assert all(call["final_measurement"] is True for call in sweep_calls)
    assert all(call["readout_amplification"] is False for call in sweep_calls)
    assert all(call["plot"] is False for call in sweep_calls)
    assert all(call["enable_tqdm"] is False for call in sweep_calls)
    assert sequence_calls == [(1, 11), (1, 22), (2, 11), (2, 22)]
    assert reset_calls == [{"Q00"}, {"Q00"}]
    np.testing.assert_allclose(
        result["Q00"]["trials"],
        np.array([[0.9, 0.7], [0.9, 0.7]]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("time_integration", [True, False])
def test_rb_2q_interprets_canonical_capture_data(
    monkeypatch: Any,
    time_integration: bool,
) -> None:
    """2Q RB should classify canonical captures and mitigate joint probabilities."""
    _patch_fit(monkeypatch)
    seeds = np.array([31, 32], dtype=int)
    sequence_calls: list[tuple[int, int]] = []
    sweep_calls: list[dict[str, object]] = []
    reset_calls: list[set[str]] = []
    inverse_confusion_calls: list[tuple[str, ...]] = []

    class _BinaryClassifier:
        n_states = 2

        def predict(self, data: np.ndarray) -> np.ndarray:
            return np.asarray(np.real(data), dtype=int)

    def _get_inverse_confusion_matrix(targets: list[str]) -> np.ndarray:
        inverse_confusion_calls.append(tuple(targets))
        return np.eye(4, dtype=float)

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        classifiers={"Q00": _BinaryClassifier(), "Q01": _BinaryClassifier()},
        measurement=SimpleNamespace(
            get_inverse_confusion_matrix=_get_inverse_confusion_matrix
        ),
        state_centers=object(),
        calib_note=SimpleNamespace(cr_params={"CR00-01": object()}),
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(is_cr=True)
        ),
        cr_pair=lambda _label: ("Q00", "Q01"),
        reset_awg_and_capunits=lambda *, qubits: reset_calls.append(set(qubits)),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator"] = None

    async def _run_sweep_measurement(
        schedule: Any,
        *,
        sweep_values: Any,
        **kwargs: object,
    ) -> SweepMeasurementResult:
        def capture(states: list[int]) -> np.ndarray:
            integrated = np.asarray(states, dtype=np.complex128)
            if time_integration:
                return integrated
            return np.column_stack((integrated, np.zeros_like(integrated)))

        values = [int(value) for value in sweep_values]
        schedules = [schedule(value) for value in values]
        sweep_calls.append({"values": values, "schedules": schedules, **kwargs})
        return _make_sweep_result(
            sweep_values=values,
            captures_by_target={
                "Q00": [
                    capture([0, 0, 1, 1]),
                    capture([0, 0, 0, 1]),
                ],
                "Q01": [
                    capture([0, 1, 0, 1]),
                    capture([0, 0, 1, 1]),
                ],
            },
            n_shots=4,
            shot_averaging=False,
            time_integration=time_integration,
        )

    def _legacy_measure(**_kwargs: object) -> None:
        raise AssertionError("legacy measure must not be called by RB")

    service.__dict__["_measurement_service"] = SimpleNamespace(
        measure=_legacy_measure,
        run_sweep_measurement=_run_sweep_measurement,
    )

    def _rb_sequence_2q(
        self: BenchmarkingService,
        target: str,
        *,
        n: int,
        seed: int,
        **_kwargs: object,
    ) -> PulseSchedule:
        assert target == "CR00-01"
        sequence_calls.append((n, seed))
        with PulseSchedule(["Q00", "CR00-01", "Q01"]) as schedule:
            schedule.add("Q00", Blank(4.0))
            schedule.add("CR00-01", Blank(4.0))
            schedule.add("Q01", Blank(4.0))
        return schedule

    service.__dict__["rb_sequence_2q"] = MethodType(_rb_sequence_2q, service)

    result = service.rb_experiment_2q(
        targets="CR00-01",
        n_cliffords_range=[3],
        n_trials=2,
        seeds=seeds,
        shots=4,
        interval=4096.0,
        time_integration=time_integration,
        plot=False,
        save_image=False,
        reset_awg_and_capunits=True,
    )

    assert len(sweep_calls) == 1
    assert sweep_calls[0]["values"] == [31, 32]
    assert all(
        isinstance(schedule, PulseSchedule)
        for schedule in cast(list[object], sweep_calls[0]["schedules"])
    )
    assert sweep_calls[0]["shot_averaging"] is False
    assert sweep_calls[0]["state_classification"] is False
    assert sequence_calls == [(3, 31), (3, 32)]
    assert reset_calls == [{"Q00", "Q01"}]
    assert inverse_confusion_calls == [("Q00", "Q01")]
    np.testing.assert_allclose(
        result["CR00-01"]["trials"],
        np.array([[0.25, 0.5]]),
        rtol=0.0,
        atol=1e-12,
    )


def test_rb_1q_auto_range_stops_after_completed_trial_sweep(
    monkeypatch: Any,
) -> None:
    """Auto-range 1Q RB should stop between awaited Clifford-length sweeps."""
    _patch_fit(monkeypatch)
    seeds = np.array([41, 42], dtype=int)
    active_n_cliffords = 0
    measured_n_cliffords: list[int] = []

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(is_cr=False)
        ),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        rabi_params={"Q00": SimpleNamespace(normalize=lambda value: np.real(value))}
    )
    service.__dict__["_clifford_generator"] = None

    def _rb_sequence_1q(
        self: BenchmarkingService,
        target: str,
        *,
        n: int,
        seed: int,
        **_kwargs: object,
    ) -> PulseArray:
        nonlocal active_n_cliffords
        assert target == "Q00"
        assert seed in seeds
        active_n_cliffords = n
        return PulseArray([])

    service.__dict__["rb_sequence_1q"] = MethodType(_rb_sequence_1q, service)

    async def _run_sweep_measurement(
        schedule: Any,
        *,
        sweep_values: Any,
        **_kwargs: object,
    ) -> SweepMeasurementResult:
        values = [int(value) for value in sweep_values]
        for value in values:
            assert isinstance(schedule(value), PulseSchedule)
        measured_n_cliffords.append(active_n_cliffords)
        signal = 1.0 if active_n_cliffords == 0 else -1.0
        return _make_sweep_result(
            sweep_values=values,
            captures_by_target={
                "Q00": [np.asarray(signal, dtype=np.complex128) for _ in values]
            },
            n_shots=128,
            shot_averaging=True,
        )

    service.__dict__["_measurement_service"] = SimpleNamespace(
        run_sweep_measurement=_run_sweep_measurement,
    )

    result = service.rb_experiment_1q(
        targets="Q00",
        n_trials=2,
        seeds=seeds,
        max_n_cliffords=8,
        plot=False,
        save_image=False,
        reset_awg_and_capunits=False,
    )

    assert measured_n_cliffords == [0, 1]
    np.testing.assert_array_equal(
        result["Q00"]["n_cliffords"],
        np.array([0, 1], dtype=int),
    )
