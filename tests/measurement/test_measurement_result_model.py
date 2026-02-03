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
    restored = result.to_multiple_measure_result()

    assert restored.mode == multiple.mode
    assert restored.config == multiple.config
    assert np.array_equal(restored.data["Q00"][0].raw, multiple.data["Q00"][0].raw)
    assert result.mode == "avg"
    assert result.measure_mode == MeasureMode.AVG


def test_to_measure_result_selects_requested_index() -> None:
    """Given wrapped multiple result, when converting with an index, then selected capture is returned."""
    multiple = _make_multiple_measure_result()
    result = MeasurementResult.from_multiple(multiple)

    single: MeasureResult = result.to_measure_result(index=1)

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
        config={"shots": 2},
        measurement_config={"mode": "avg", "shots": 2},
        pulse_metadata={"labels": ["RQ00"], "duration": 256.0, "length": 128},
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
    assert restored.measurement_config == original.measurement_config
    assert restored.pulse_metadata == original.pulse_metadata
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
        config={"shots": 2},
        measurement_config={"mode": "single", "shots": 2},
        pulse_metadata={"labels": ["RQ00", "RQ01"], "duration": 512.0, "length": 256},
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
    assert restored.config == original.config
    assert restored.measurement_config == original.measurement_config
    assert restored.pulse_metadata == original.pulse_metadata
    assert np.array_equal(restored.data["Q00"][0], original.data["Q00"][0])
    assert np.array_equal(restored.data["Q01"][0], original.data["Q01"][0])
    assert np.array_equal(restored.data["Q01"][1], original.data["Q01"][1])
