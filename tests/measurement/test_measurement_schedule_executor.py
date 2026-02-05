"""Tests for measurement schedule executor orchestration."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from qubex.backend import (
    BackendExecutionRequest,
    BackendExecutor,
    DeviceController,
    RawResult,
)
from qubex.measurement.measurement_backend_adapter import MeasurementBackendAdapter
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.measurement_result_factory import MeasurementResultFactory
from qubex.measurement.measurement_schedule_executor import (
    MeasurementScheduleExecutor,
)
from qubex.measurement.models import (
    DspConfig,
    MeasurementConfig,
    MeasurementSchedule,
    ReadoutConfig,
)
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MultipleMeasureResult,
)
from qubex.pulse import PulseSchedule


def _make_config() -> MeasurementConfig:
    return MeasurementConfig(
        mode="avg",
        shots=2,
        interval=100.0,
        readout=ReadoutConfig(
            readout_amplitudes={},
            readout_duration=384.0,
            readout_pre_margin=32.0,
            readout_post_margin=128.0,
            readout_ramptime=32.0,
            readout_drag_coeff=0.0,
            readout_ramp_type="RaisedCosine",
        ),
        dsp=DspConfig(
            enable_dsp_demodulation=True,
            enable_dsp_sum=False,
            enable_dsp_classification=False,
            line_param0=(1.0, 0.0, 0.0),
            line_param1=(0.0, 1.0, 0.0),
        ),
    )


def _make_multiple_result() -> MultipleMeasureResult:
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([1.0 + 0.0j]),
        classifier=None,
    )
    return MultipleMeasureResult(
        mode=MeasureMode.AVG,
        data={"Q00": [data]},
        config={"shots": 2},
    )


def test_execute_validates_builds_executes_and_creates_result() -> None:
    """Given executor inputs, when execute is called, then it validates, runs backend, and builds result."""
    called: dict[str, object] = {}
    request = BackendExecutionRequest(payload=object())
    backend_result = RawResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())

    class _Adapter:
        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            called["validated"] = schedule

        def build_execution_request(
            self, *, schedule: MeasurementSchedule, config: MeasurementConfig
        ) -> BackendExecutionRequest:
            called["request_schedule"] = schedule
            called["request_config"] = config
            return request

    class _Executor:
        def execute(self, *, request: BackendExecutionRequest) -> RawResult:
            called["execute_request"] = request
            return backend_result

    class _ResultFactory:
        def create(self, **kwargs: object):  # type: ignore[no-untyped-def]
            called["result_kwargs"] = kwargs
            return expected

    device_controller = type("_DC", (), {"box_config": {"shots": 2}})()
    executor = MeasurementScheduleExecutor(
        backend_executor=cast(BackendExecutor, _Executor()),
        measurement_backend_adapter=cast(MeasurementBackendAdapter, _Adapter()),
        measurement_result_factory=cast(MeasurementResultFactory, _ResultFactory()),
        device_controller=cast(DeviceController, device_controller),
    )

    schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = _make_config()

    result = executor.execute(schedule=schedule, config=config)

    assert called["validated"] is schedule
    assert called["request_schedule"] is schedule
    assert called["request_config"] is config
    assert called["execute_request"] is request
    result_kwargs = called["result_kwargs"]
    assert isinstance(result_kwargs, dict)
    typed_kwargs = cast(dict[str, Any], result_kwargs)
    assert typed_kwargs["backend_result"] is backend_result
    assert typed_kwargs["measurement_config"] is config
    assert typed_kwargs["device_config"] == {"shots": 2}
    assert result is expected
