"""Tests for qxschema sweep measurement execution."""

from __future__ import annotations

import asyncio
from types import MethodType
from typing import cast

import numpy as np
import pytest

from qubex.core import Time
from qubex.measurement import SweepMeasurementExecutor
from qubex.measurement.measurement import Measurement
from qubex.measurement.models import CaptureData, MeasurementConfig, MeasurementResult
from qubex.schema import (
    DataAcquisitionConfig,
    FrequencyConfig,
    ParameterSweepConfig,
    ParameterSweepContent,
    ParametricSequenceConfig,
    ParametricSequencePulseCommand,
    SweepMeasurementConfig,
)


def _make_config(*, channel: str = "Q00") -> SweepMeasurementConfig:
    return SweepMeasurementConfig(
        channel_list=[channel],
        sequence=ParametricSequenceConfig(
            delta_time=Time(2.0, "ns"),
            variable_list=["amp"],
            command_list=[
                ParametricSequencePulseCommand(
                    name="Rect",
                    channel_list=[channel],
                    argument_list=[10.0, "amp"],
                )
            ],
        ),
        frequency=FrequencyConfig(
            channel_to_frequency={},
            channel_to_frequency_reference={channel: "f_ref"},
            channel_to_frequency_shift={},
            keep_oscillator_relative_phase=False,
        ),
        data_acquisition=DataAcquisitionConfig(
            shot_count=16,
            shot_repetition_margin=Time(100.0, "ns"),
            data_acquisition_duration=Time(4.0, "ns"),
            data_acquisition_delay=Time(2.0, "ns"),
            data_acquisition_timeout=Time(10.0, "ms"),
            flag_average_waveform=False,
            flag_average_shots=True,
            delta_time=Time(2.0, "ns"),
            channel_to_averaging_time={channel: Time(4.0, "ns")},
            channel_to_averaging_window={channel: [1.0 + 0.0j]},
        ),
        sweep_parameter=ParameterSweepConfig(
            sweep_content_list={
                "amp": ParameterSweepContent(
                    category="sequence_variable",
                    sweep_target=["amp"],
                    value_list=[0.1, 0.2],
                )
            },
            sweep_axis=[["amp"]],
        ),
    )


def _make_measurement_result(
    *,
    config: MeasurementConfig,
    channel: str,
    data: np.ndarray,
) -> MeasurementResult:
    return MeasurementResult(
        data={
            channel: [
                CaptureData.from_primary_data(
                    target=channel,
                    data=data,
                    config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
        device_config={"backend": "stub"},
    )


def test_run_returns_qxschema_sweep_measurement_result(monkeypatch) -> None:
    """Given valid config, when executor runs, then qxschema sweep result is returned with deterministic layout."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    config = _make_config()
    called: dict[str, object] = {}
    schedules: list[object] = []
    point_counter = {"index": 0}

    monkeypatch.setattr(
        Measurement,
        "targets",
        property(lambda self: {"Q00": object()}),
    )
    monkeypatch.setattr(
        Measurement,
        "backend_controller",
        property(lambda self: object()),
    )

    def _create_measurement_config(
        self: Measurement,
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
    ) -> MeasurementConfig:
        called["create_args"] = {
            "n_shots": n_shots,
            "shot_interval": shot_interval,
            "shot_averaging": shot_averaging,
            "time_integration": time_integration,
            "state_classification": state_classification,
        }
        return MeasurementConfig(
            n_shots=cast(int, n_shots),
            shot_interval=cast(float, shot_interval),
            shot_averaging=cast(bool, shot_averaging),
            time_integration=cast(bool, time_integration),
            state_classification=cast(bool, state_classification),
        )

    async def _run_measurement(
        self: Measurement,
        schedule: object,
        *,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        _ = self
        schedules.append(schedule)
        index = point_counter["index"]
        point_counter["index"] += 1
        return _make_measurement_result(
            config=config,
            channel="Q00",
            data=np.asarray([complex(index + 1, 0.0), complex(index + 2, 0.0)]),
        )

    measurement.create_measurement_config = MethodType(
        _create_measurement_config,
        measurement,
    )
    measurement.run_measurement = MethodType(_run_measurement, measurement)

    result = asyncio.run(SweepMeasurementExecutor(measurement=measurement).run(config))

    assert called["create_args"] == {
        "n_shots": 16,
        "shot_interval": 100.0,
        "shot_averaging": True,
        "time_integration": False,
        "state_classification": False,
    }
    assert result.sweep_key_list == ["amp"]
    assert result.data_key_list == ["Q00"]
    assert result.data_shape == [2, 1, 2]
    np.testing.assert_array_equal(
        result.data,
        np.asarray(
            [
                [[1.0 + 0.0j, 2.0 + 0.0j]],
                [[2.0 + 0.0j, 3.0 + 0.0j]],
            ]
        ),
    )
    assert result.metadata["sweep_axis"] == [["amp"]]
    assert result.metadata["measurement_config"]["n_shots"] == 16


def test_run_rejects_unknown_runtime_channel(monkeypatch) -> None:
    """Given unknown runtime channel, when executor runs, then validation fails before execution."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called = {"run_measurement": 0}

    monkeypatch.setattr(
        Measurement,
        "targets",
        property(lambda self: {"Q00": object()}),
    )
    monkeypatch.setattr(
        Measurement,
        "backend_controller",
        property(lambda self: object()),
    )

    async def _run_measurement(
        self: Measurement,
        schedule: object,
        *,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        _ = (self, schedule, config)
        called["run_measurement"] += 1
        raise AssertionError("should not be called")

    measurement.run_measurement = MethodType(_run_measurement, measurement)

    with pytest.raises(ValueError, match="Unknown channel"):
        asyncio.run(
            SweepMeasurementExecutor(measurement=measurement).run(
                _make_config(channel="Q01")
            )
        )

    assert called["run_measurement"] == 0


def test_run_rejects_mismatched_payload_shapes(monkeypatch) -> None:
    """Given varying payload shapes, when executor converts results, then conversion fails."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    point_counter = {"index": 0}

    monkeypatch.setattr(
        Measurement,
        "targets",
        property(lambda self: {"Q00": object()}),
    )
    monkeypatch.setattr(
        Measurement,
        "backend_controller",
        property(lambda self: object()),
    )

    def _create_measurement_config(
        self: Measurement,
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
    ) -> MeasurementConfig:
        _ = self
        return MeasurementConfig(
            n_shots=cast(int, n_shots),
            shot_interval=cast(float, shot_interval),
            shot_averaging=cast(bool, shot_averaging),
            time_integration=cast(bool, time_integration),
            state_classification=cast(bool, state_classification),
        )

    async def _run_measurement(
        self: Measurement,
        schedule: object,
        *,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        _ = (self, schedule)
        if point_counter["index"] == 0:
            data = np.asarray([1.0 + 0.0j, 2.0 + 0.0j])
        else:
            data = np.asarray([3.0 + 0.0j, 4.0 + 0.0j, 5.0 + 0.0j])
        point_counter["index"] += 1
        return _make_measurement_result(
            config=config,
            channel="Q00",
            data=data,
        )

    measurement.create_measurement_config = MethodType(
        _create_measurement_config,
        measurement,
    )
    measurement.run_measurement = MethodType(_run_measurement, measurement)

    with pytest.raises(ValueError, match="payload shape"):
        asyncio.run(
            SweepMeasurementExecutor(measurement=measurement).run(_make_config())
        )
