"""Tests for the high-level measurement result model."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.measurement.models import MeasurementResult
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from qubex.measurement.models.measurement_result import PulseScheduleSnapshot


def _make_multiple_measure_result() -> MultipleMeasureResult:
    data0 = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([1.0 + 0.0j]),
        classifier=None,
    )
    data1 = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([2.0 + 0.0j]),
        classifier=None,
    )
    return MultipleMeasureResult(
        mode=MeasureMode.AVG,
        data={"Q00": [data0, data1]},
        config={"shots": 2},
    )


def test_to_multiple_measure_result_returns_wrapped_result() -> None:
    """Given legacy multiple result, when converting round-trip, then mode and config are preserved."""
    multiple = _make_multiple_measure_result()
    result = MeasurementResult.from_multiple(multiple)
    restored = result.to_multiple_measure_result(config=multiple.config)

    assert restored.mode == multiple.mode
    assert restored.config == multiple.config
    assert result.device_config == multiple.config
    assert np.array_equal(restored.data["Q00"][0].raw, multiple.data["Q00"][0].raw)
    assert result.mode == "avg"
    assert result.measure_mode == MeasureMode.AVG


def test_to_measure_result_selects_requested_index() -> None:
    """Given wrapped multiple result, when converting with an index, then selected capture is returned."""
    multiple = _make_multiple_measure_result()
    result = MeasurementResult.from_multiple(multiple)

    single: MeasureResult = result.to_measure_result(index=1, config=multiple.config)

    assert single.mode == MeasureMode.AVG
    assert np.array_equal(single.data["Q00"].raw, multiple.data["Q00"][1].raw)
    assert single.config == multiple.config


def test_to_measure_result_raises_for_invalid_index() -> None:
    """Given wrapped multiple result, when index is out of range, then IndexError is raised."""
    result = MeasurementResult.from_multiple(_make_multiple_measure_result())

    with pytest.raises(IndexError):
        result.to_measure_result(index=10)


def test_json_roundtrip_preserves_raw_arrays() -> None:
    """Given serialized measurement result, when deserializing, then raw arrays are preserved."""
    original = MeasurementResult(
        mode="avg",
        data={"Q00": [np.array([1.0 + 0.0j]), np.array([2.0 + 0.0j])]},
        device_config={"shots": 2},
        measurement_config={"mode": "avg", "shots": 2},
        pulse_schedule=PulseScheduleSnapshot(
            target_labels=["RQ00"],
            total_duration=256.0,
            total_length=128,
            waveforms={"RQ00": np.array([0.1 + 0.2j, 0.2 + 0.3j])},
        ),
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(channels=["RQ00"], start_time=0.0, duration=32.0),
                Capture(channels=["RQ00"], start_time=64.0, duration=32.0),
            ]
        ),
    )

    restored = MeasurementResult.from_json(original.to_json())

    assert restored.mode == original.mode
    assert np.array_equal(restored.data["Q00"][0], original.data["Q00"][0])
    assert restored.device_config == original.device_config
    assert restored.measurement_config == original.measurement_config
    assert restored.pulse_schedule is not None
    assert original.pulse_schedule is not None
    assert (
        restored.pulse_schedule.target_labels == original.pulse_schedule.target_labels
    )
    assert (
        restored.pulse_schedule.total_duration == original.pulse_schedule.total_duration
    )
    assert restored.pulse_schedule.total_length == original.pulse_schedule.total_length
    assert restored.pulse_schedule.waveforms is not None
    assert original.pulse_schedule.waveforms is not None
    assert np.array_equal(
        restored.pulse_schedule.waveforms["RQ00"],
        original.pulse_schedule.waveforms["RQ00"],
    )
    assert restored.capture_schedule is not None
    assert len(restored.capture_schedule.captures) == 2


def test_netcdf_roundtrip_preserves_raw_arrays(tmp_path) -> None:
    """Given NetCDF save/load, when round-tripping, then raw data and metadata are preserved."""
    original = MeasurementResult(
        mode="single",
        data={
            "Q00": [np.array([[1.0 + 2.0j], [3.0 + 4.0j]])],
            "Q01": [np.array([5.0 + 6.0j]), np.array([7.0 + 8.0j])],
        },
        device_config={"shots": 2},
        measurement_config={"mode": "single", "shots": 2},
        pulse_schedule=PulseScheduleSnapshot(
            target_labels=["RQ00", "RQ01"],
            total_duration=512.0,
            total_length=256,
            waveforms={
                "RQ00": np.array([0.1 + 0.2j, 0.2 + 0.3j]),
                "RQ01": np.array([0.3 + 0.4j, 0.4 + 0.5j]),
            },
        ),
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(channels=["RQ00", "RQ01"], start_time=0.0, duration=32.0),
            ]
        ),
    )
    path = tmp_path / "measurement_result.nc"

    saved = original.save_netcdf(path)
    restored = MeasurementResult.load_netcdf(saved)

    assert restored.mode == original.mode
    assert restored.device_config == original.device_config
    assert restored.measurement_config == original.measurement_config
    assert restored.pulse_schedule is not None
    assert original.pulse_schedule is not None
    assert restored.pulse_schedule.waveforms is not None
    assert original.pulse_schedule.waveforms is not None
    assert np.array_equal(
        restored.pulse_schedule.waveforms["RQ00"],
        original.pulse_schedule.waveforms["RQ00"],
    )
    assert np.array_equal(
        restored.pulse_schedule.waveforms["RQ01"],
        original.pulse_schedule.waveforms["RQ01"],
    )
    assert np.array_equal(restored.data["Q00"][0], original.data["Q00"][0])
    assert np.array_equal(restored.data["Q01"][0], original.data["Q01"][0])
    assert np.array_equal(restored.data["Q01"][1], original.data["Q01"][1])


def test_save_writes_netcdf_file(tmp_path) -> None:
    """Given save(), when called, then it writes a readable NetCDF file."""
    result = MeasurementResult(
        mode="avg",
        data={"Q00": [np.array([1.0 + 0.0j])]},
    )

    path = result.save(tmp_path, file_name="result.nc")
    restored = MeasurementResult.load_netcdf(path)

    assert path.name == "result.nc"
    assert np.array_equal(restored.data["Q00"][0], np.array([1.0 + 0.0j]))
