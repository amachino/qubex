"""Regression tests for readout optimization through the shared sweep executor."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule, Rect, get_sampling_period

from qubex.experiment.services.characterization_service import CharacterizationService
from qubex.experiment.services.measurement_service import MeasurementService
from qubex.measurement import CaptureData, MeasurementResult, MeasurementSchedule
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.services.measurement_execution_service import (
    MeasurementExecutionService,
)


class _Runner:
    """Supply deterministic IQ captures at the schedule execution boundary."""

    def __init__(self) -> None:
        self.signals: list[complex] = []
        self.offset = 0
        self.single_calls: list[MeasurementSchedule] = []
        self.batch_calls: list[list[MeasurementSchedule]] = []
        self.configs: list[MeasurementConfig] = []
        self.error: Exception | None = None

    def result(
        self, schedule: MeasurementSchedule, config: MeasurementConfig
    ) -> MeasurementResult:
        """Return one IQ value per visible capture in execution order."""
        if self.error is not None:
            raise self.error
        self.configs.append(config)
        captures = []
        for _ in schedule.capture_schedule.captures:
            signal = self.signals[self.offset]
            self.offset += 1
            data = (
                np.asarray(signal)
                if config.shot_averaging
                else np.full(config.n_shots, signal, dtype=np.complex128)
            )
            captures.append(
                CaptureData.from_primary_data(
                    target="Q00",
                    data=data,
                    config=config,
                    sampling_period=get_sampling_period(),
                )
            )
        return MeasurementResult(data={"Q00": captures}, measurement_config=config)

    async def execute_async(
        self, *, schedule: MeasurementSchedule, config: MeasurementConfig, **_: Any
    ) -> MeasurementResult:
        """Record individual or packed executions."""
        self.single_calls.append(schedule)
        return self.result(schedule, config)

    async def execute_batch_async(
        self, *, schedules: list[MeasurementSchedule], config: MeasurementConfig
    ) -> list[MeasurementResult]:
        """Record a batch and preserve schedule order."""
        self.batch_calls.append(schedules)
        return [self.result(schedule, config) for schedule in schedules]


@pytest.fixture
def readout(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Connect characterization and measurement services to the real sweep executor."""
    runner = _Runner()
    builds: list[dict[str, Any]] = []
    sweeps: list[list[int]] = []
    classifier_calls: list[dict[str, Any]] = []
    reset_calls: list[list[str]] = []
    ctx = SimpleNamespace(
        targets={"RQ00": SimpleNamespace(frequency=5.105)},
        params=SimpleNamespace(readout_amplitude={"Q00": 0.25}),
        reference_phases={"Q00": 0.3},
        resolve_qubit_label=lambda _target: "Q00",
        resolve_read_label=lambda _target: "RQ00",
        reset_awg_and_capunits=lambda *, qubits: reset_calls.append(qubits),
    )

    @contextmanager
    def modified_frequencies(frequencies: dict[str, float] | None) -> Iterator[None]:
        original = ctx.targets["RQ00"].frequency
        if frequencies is not None:
            ctx.targets["RQ00"].frequency = frequencies["RQ00"]
        try:
            yield
        finally:
            ctx.targets["RQ00"].frequency = original

    ctx.modified_frequencies = modified_frequencies
    options = SimpleNamespace(packing=True, limit=None)
    backend = SimpleNamespace(execute_batch_async=runner.execute_batch_async)
    execution = MeasurementExecutionService(
        context=cast(
            Any,
            SimpleNamespace(
                experiment_system=SimpleNamespace(
                    target_registry=SimpleNamespace(
                        measurement_output_label=lambda _label: "Q00"
                    )
                )
            ),
        ),
        session_service=cast(Any, SimpleNamespace(backend_controller=backend)),
        classifiers={},
    )
    monkeypatch.setattr(
        MeasurementExecutionService,
        "measurement_schedule_runner",
        property(lambda _self: runner),
    )

    def build_schedule(
        *, pulse_schedule: PulseSchedule, **kwargs: Any
    ) -> MeasurementSchedule:
        builds.append({"pulse_schedule": pulse_schedule, **kwargs})
        assert kwargs["final_measurement"] is True
        readout_amplitudes = kwargs["readout_amplitudes"]
        amplitude = 0.25 if readout_amplitudes is None else readout_amplitudes["Q00"]
        with PulseSchedule(["Q00", "RQ00"]) as schedule:
            schedule.call(pulse_schedule)
            schedule.barrier()
            schedule.add("RQ00", Rect(duration=8, amplitude=amplitude))
        schedule.set_frequencies(kwargs["frequencies"] or {"RQ00": 5.105})
        return MeasurementSchedule(
            pulse_schedule=schedule,
            capture_schedule=CaptureSchedule(
                captures=[
                    Capture(
                        channels=["RQ00"],
                        start_time=pulse_schedule.duration,
                        duration=8,
                    )
                ]
            ),
        )

    def create_config(**kwargs: Any) -> MeasurementConfig:
        return MeasurementConfig(
            **kwargs,
            schedule_packing_enabled=options.packing,
            max_repeated_timeline_duration_ns=options.limit,
        )

    async def run_sweep(schedule: Any, *, sweep_values: Any, **kwargs: Any) -> Any:
        sweeps.append(list(sweep_values))
        return await execution.run_sweep_measurement(
            schedule, sweep_values=sweep_values, **kwargs
        )

    def build_classifier(**kwargs: Any) -> Any:
        classifier_calls.append({"frequency": ctx.targets["RQ00"].frequency, **kwargs})
        return SimpleNamespace(figures={})

    def legacy_measure(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Readout parameter data must be acquired through sweep execution.")

    ctx.measurement = SimpleNamespace(
        build_measurement_schedule=build_schedule,
        create_measurement_config=create_config,
        run_sweep_measurement=run_sweep,
    )
    pulse = SimpleNamespace(
        get_pulse_for_state=lambda target, state: (
            Blank(0) if state == "0" else Rect(duration=4, amplitude=0.5)
        ),
        readout_duration=8,
        readout_pre_margin=0,
        readout_post_margin=0,
    )
    measurement = cast(Any, object.__new__(MeasurementService))
    measurement.__dict__.update(_ctx=ctx, _pulse_service=pulse)
    measurement.build_classifier = build_classifier
    measurement.measure = legacy_measure
    measurement.measure_state = legacy_measure
    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__.update(
        _experiment_context=ctx, _measurement_service=measurement, _pulse_service=pulse
    )
    return SimpleNamespace(
        service=service,
        runner=runner,
        builds=builds,
        sweeps=sweeps,
        ctx=ctx,
        options=options,
        execution=execution,
        classifier_calls=classifier_calls,
        reset_calls=reset_calls,
        measurement=measurement,
    )


@pytest.mark.parametrize("execution_mode", ["packed", "chunked", "batch", "sequential"])
@pytest.mark.parametrize("shots", [1, 4])
def test_amplitude_distance_preserves_signals_across_execution_modes(
    readout: Any, execution_mode: str, shots: int
) -> None:
    """Amplitude distance should preserve IQ order while using available batch and packing capabilities."""
    readout.runner.signals = [1, 2, 1, 4, 1, 3]
    if execution_mode == "chunked":
        readout.options.limit = 31 * shots
    elif execution_mode == "batch":
        readout.options.packing = False
    elif execution_mode == "sequential":
        readout.execution.session_service.backend_controller = object()

    result = readout.service.find_optimal_readout_amplitude(
        "Q00",
        amplitude_range=[0.1, 0.2, 0.3],
        objective="distance",
        shots=shots,
        interval=10,
        plot=False,
        save_image=False,
    )

    assert result["optimal_amplitude"] == pytest.approx(0.2)
    np.testing.assert_allclose(result["signals_0"], [1, 1, 1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["signals_1"], [2, 4, 3], rtol=0, atol=1e-12)
    assert readout.sweeps == [list(range(6))]
    assert [build["readout_amplitudes"]["Q00"] for build in readout.builds] == [
        0.1,
        0.1,
        0.2,
        0.2,
        0.3,
        0.3,
    ]
    assert [build["pulse_schedule"].duration for build in readout.builds] == [0, 4] * 3
    assert all(
        config.n_shots == shots and config.shot_interval == 10
        for config in readout.runner.configs
    )
    assert all(
        config.shot_averaging
        and config.time_integration
        and not config.state_classification
        for config in readout.runner.configs
    )
    assert len(readout.runner.batch_calls) == (1 if execution_mode == "batch" else 0)
    assert (
        len(readout.runner.single_calls)
        == {"packed": 1, "chunked": 3, "batch": 0, "sequential": 6}[execution_mode]
    )


@pytest.mark.parametrize("parameter", ["frequency", "amplitude"])
def test_fidelity_sweep_preserves_plateau_and_classifier_update(
    readout: Any, monkeypatch: pytest.MonkeyPatch, parameter: str
) -> None:
    """Fidelity sweeps should preserve single-shot IQ and select the first point within the peak ratio."""
    readout.runner.signals = [1, 2, 3, 4, 5, 6]

    def fit(data: dict[int, np.ndarray], *, phase: float) -> Any:
        assert phase == 0.3
        assert all(iq.shape == (4,) for iq in data.values())
        predictions = {
            id(data[0]): np.array([0, 1, 1, 1]) if data[0][0] == 1 else np.zeros(4),
            id(data[1]): np.array([1, 1, 1, 0]) if data[0][0] == 1 else np.ones(4),
        }
        return SimpleNamespace(predict=lambda iq: predictions[id(iq)])

    monkeypatch.setattr(
        "qubex.measurement.classifiers.state_classifier_gmm.StateClassifierGMM.fit", fit
    )
    kwargs = (
        {"df": 0.1, "frequency_width": 0.21, "readout_amplitude": 0.17}
        if parameter == "frequency"
        else {"amplitude_range": [0.1, 0.2, 0.3]}
    )
    result = getattr(readout.service, f"find_optimal_readout_{parameter}")(
        "Q00",
        **kwargs,
        objective="fidelity",
        fidelity_ratio=0.99,
        shots=4,
        interval=10,
        plot=False,
        save_image=False,
    )

    expected_optimal = 5.1 if parameter == "frequency" else 0.2
    assert result[f"optimal_{parameter}"] == pytest.approx(expected_optimal)
    np.testing.assert_allclose(
        result["readout_fidelity"], [0.5, 1, 1], rtol=0, atol=1e-12
    )
    assert readout.sweeps == [list(range(6))]
    assert readout.classifier_calls == [
        {
            "frequency": pytest.approx(5.1 if parameter == "frequency" else 5.105),
            "targets": "Q00",
            "readout_amplitudes": {"Q00": 0.17 if parameter == "frequency" else 0.2},
            "n_shots": 4,
            "shot_interval": 10,
            "plot": False,
        }
    ]
    assert readout.ctx.targets["RQ00"].frequency == 5.105
    assert all(
        not config.shot_averaging
        and config.time_integration
        and not config.state_classification
        for config in readout.runner.configs
    )
    if parameter == "frequency":
        np.testing.assert_allclose(
            [build["frequencies"]["RQ00"] for build in readout.builds],
            [5, 5, 5.1, 5.1, 5.2, 5.2],
            rtol=0,
            atol=1e-12,
        )
        assert all(
            build["readout_amplitudes"] == {"Q00": 0.17} for build in readout.builds
        )
        assert len(readout.runner.batch_calls) == 1
        assert not readout.runner.single_calls
        assert result["signals_0"].shape == result["signals_1"].shape == (3, 4)


def test_sync_optimization_runs_inside_an_event_loop(readout: Any) -> None:
    """Synchronous optimization should return its result when an event loop is already running."""
    readout.runner.signals = [1, 2, 1, 4]

    async def run() -> Any:
        return readout.service.find_optimal_readout_amplitude(
            "Q00",
            amplitude_range=[0.1, 0.2],
            objective="distance",
            shots=1,
            interval=10,
            plot=False,
            save_image=False,
        )

    result = asyncio.run(run())
    assert result["optimal_amplitude"] == pytest.approx(0.2)
    assert len(readout.runner.single_calls) == 1


@pytest.mark.parametrize("failure_stage", ["sweep", "classifier"])
def test_frequency_failure_restores_frequency(
    readout: Any, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """A failed sweep or classifier rebuild should leave the configured readout frequency intact."""
    if failure_stage == "sweep":
        readout.runner.error = RuntimeError("measurement failed")
    else:
        readout.runner.signals = [0, 1]
        monkeypatch.setattr(
            "qubex.measurement.classifiers.state_classifier_gmm.StateClassifierGMM.fit",
            lambda data, **_: SimpleNamespace(
                predict=lambda iq: np.full(len(iq), 0 if iq is data[0] else 1)
            ),
        )

        def fail(**_kwargs: Any) -> None:
            assert readout.ctx.targets["RQ00"].frequency == pytest.approx(5.1)
            raise RuntimeError("measurement failed")

        monkeypatch.setattr(readout.measurement, "build_classifier", fail)

    with pytest.raises(RuntimeError, match="measurement failed"):
        readout.service.find_optimal_readout_frequency(
            "Q00",
            df=0.1,
            frequency_width=0.01,
            objective="fidelity",
            shots=4,
            interval=10,
            plot=False,
            save_image=False,
        )
    assert readout.ctx.targets["RQ00"].frequency == 5.105
