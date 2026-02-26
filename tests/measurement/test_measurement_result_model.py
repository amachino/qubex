"""Tests for the high-level measurement result model."""

from __future__ import annotations

import json

import numpy as np
import pytest
from netCDF4 import Dataset
from pydantic import ValidationError

from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.models import MeasurementConfig, MeasurementResult
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from qubex.typing import MeasurementMode


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


def _make_config(*, mode: MeasurementMode = "avg", shots: int = 2) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=shots,
        shot_interval_ns=100.0,
        shot_averaging=(mode == "avg"),
        time_integration=False,
        state_classification=False,
    )


def test_to_multiple_measure_result_returns_wrapped_result() -> None:
    """Given legacy multiple result, when converting round-trip, then mode and config are preserved."""
    multiple = _make_multiple_measure_result()
    config = _make_config()
    result = MeasurementResultConverter.from_multiple(
        multiple,
        measurement_config=config,
    )
    restored = MeasurementResultConverter.to_multiple_measure_result(
        result,
        config=multiple.config,
    )

    assert restored.mode == multiple.mode
    assert restored.config == multiple.config
    assert result.device_config == multiple.config
    assert result.measurement_config == config
    assert np.array_equal(restored.data["Q00"][0].raw, multiple.data["Q00"][0].raw)
    assert result.measurement_config.shot_averaging is True


def test_to_measure_result_selects_requested_index() -> None:
    """Given wrapped multiple result, when converting with an index, then selected capture is returned."""
    multiple = _make_multiple_measure_result()
    result = MeasurementResultConverter.from_multiple(
        multiple,
        measurement_config=_make_config(),
    )

    single: MeasureResult = MeasurementResultConverter.to_measure_result(
        result,
        index=1,
        config=multiple.config,
    )

    assert single.mode == MeasureMode.AVG
    assert np.array_equal(single.data["Q00"].raw, multiple.data["Q00"][1].raw)
    assert single.config == multiple.config


def test_to_measure_result_raises_for_invalid_index() -> None:
    """Given wrapped multiple result, when index is out of range, then IndexError is raised."""
    result = MeasurementResultConverter.from_multiple(
        _make_multiple_measure_result(),
        measurement_config=_make_config(),
    )

    with pytest.raises(IndexError):
        MeasurementResultConverter.to_measure_result(result, index=10)


def test_to_measure_result_propagates_sampling_period() -> None:
    """Given canonical result with sampling period, when converting, then legacy data keeps it."""
    result = MeasurementResult(
        data={"Q00": [np.array([1.0 + 0.0j]), np.array([2.0 + 0.0j])]},
        device_config={"shots": 2},
        measurement_config=_make_config(mode="avg", shots=2),
        sampling_period_ns=0.8,
    )

    single = MeasurementResultConverter.to_measure_result(result, index=1)

    assert single.data["Q00"].sampling_period_ns == 0.8
    assert np.array_equal(single.data["Q00"].times, np.array([0.0]))


def test_json_roundtrip_preserves_raw_arrays() -> None:
    """Given serialized measurement result, when deserializing, then raw arrays are preserved."""
    original = MeasurementResult(
        data={"Q00": [np.array([1.0 + 0.0j]), np.array([2.0 + 0.0j])]},
        device_config={"shots": 2},
        measurement_config=_make_config(mode="avg", shots=2),
    )
    serialized = original.to_dict()

    restored = MeasurementResult.from_json(original.to_json())

    assert serialized["__meta__"]["format"] == "qxdata"
    assert serialized["__meta__"]["version"] == 1
    assert np.array_equal(restored.data["Q00"][0], original.data["Q00"][0])
    assert restored.device_config == original.device_config
    assert restored.measurement_config == original.measurement_config


def test_netcdf_roundtrip_preserves_raw_arrays(tmp_path) -> None:
    """Given NetCDF save/load, when round-tripping, then raw data and metadata are preserved."""
    original = MeasurementResult(
        data={
            "Q00": [np.array([[1.0 + 2.0j], [3.0 + 4.0j]])],
            "Q01": [np.array([5.0 + 6.0j]), np.array([7.0 + 8.0j])],
        },
        device_config={"shots": 2},
        measurement_config=_make_config(mode="single", shots=2),
    )
    path = tmp_path / "measurement_result.nc"

    saved = original.save_netcdf(path)
    restored = MeasurementResult.load_netcdf(saved)

    assert restored.device_config == original.device_config
    assert restored.measurement_config == original.measurement_config
    assert np.array_equal(restored.data["Q00"][0], original.data["Q00"][0])
    assert np.array_equal(restored.data["Q01"][0], original.data["Q01"][0])
    assert np.array_equal(restored.data["Q01"][1], original.data["Q01"][1])


def test_save_writes_netcdf_file(tmp_path) -> None:
    """Given save(), when called, then it writes a readable NetCDF file."""
    result = MeasurementResult(
        data={"Q00": [np.array([1.0 + 0.0j])]},
        measurement_config=_make_config(mode="avg", shots=2),
    )

    path = result.save(tmp_path, file_name="result.nc")
    restored = MeasurementResult.load_netcdf(path)

    assert path.name == "result.nc"
    assert np.array_equal(restored.data["Q00"][0], np.array([1.0 + 0.0j]))


def test_measurement_result_requires_measurement_config() -> None:
    """Given missing measurement config, result construction raises validation error."""
    with pytest.raises(ValidationError):
        _ = MeasurementResult.model_validate(
            {"data": {"Q00": [np.array([1.0 + 0.0j])]}}
        )


def test_converter_falls_back_to_empty_config_when_missing() -> None:
    """Given canonical result without device config, converter returns legacy results with empty config."""
    result = MeasurementResult(
        data={"Q00": [np.array([1.0 + 0.0j]), np.array([2.0 + 0.0j])]},
        measurement_config=_make_config(mode="avg", shots=2),
    )

    multiple = MeasurementResultConverter.to_multiple_measure_result(result)
    single = MeasurementResultConverter.to_measure_result(result)

    assert multiple.config == {}
    assert single.config == {}


def test_netcdf_writes_codec_metadata_attributes(tmp_path) -> None:
    """Given NetCDF save, when opening the file, then codec metadata attributes are present."""
    result = MeasurementResult(
        data={"Q00": [np.array([1.0 + 2.0j])]},
        device_config={"backend": "quel"},
        measurement_config=_make_config(mode="single", shots=1),
    )
    path = result.save_netcdf(tmp_path / "metadata.nc")

    with Dataset(path, mode="r") as ds:
        format_name = ds.getncattr("format")
        format_version = ds.getncattr("format_version")
        model_class = ds.getncattr("model_class")
        payload_json = ds.getncattr("payload_json")

        assert format_name == "qxdata"
        assert int(format_version) == 1
        assert model_class.endswith(".MeasurementResult")

        payload = json.loads(payload_json)
        assert "Q00" in payload["data"]
