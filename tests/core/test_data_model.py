"""Tests for DataModel NetCDF serialization."""

from __future__ import annotations

import numpy as np
import tunits
from netCDF4 import Dataset

from qubex.core import DataModel, Model, ValueArray


class _DemoDataModel(DataModel):
    values: np.ndarray
    delay: ValueArray


class _DemoConfig(Model):
    shots: int
    thresholds: np.ndarray


class _DemoDataModelWithNestedConfig(DataModel):
    values: np.ndarray
    config: _DemoConfig


def test_netcdf_roundtrip_handles_int64_arrays(tmp_path) -> None:
    """Given int64 arrays, when round-tripped via NetCDF, then values are preserved."""
    original = _DemoDataModel(
        values=np.array([1, 2, 3], dtype=np.int64),
        delay=ValueArray([10, 20], tunits.ns),
    )

    path = original.save_netcdf(tmp_path / "demo.nc")
    restored = _DemoDataModel.load_netcdf(path)

    assert np.array_equal(restored.values, original.values)
    assert np.array_equal(restored.delay.value, original.delay.value)
    assert restored.delay.units == original.delay.units


def test_netcdf_writes_complex_variable_with_netcdf4_type(tmp_path) -> None:
    """Given complex arrays, when saved, then they are stored as NetCDF4 complex variables."""
    original = _DemoDataModel(
        values=np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128),
        delay=ValueArray([10, 20], tunits.ns),
    )

    path = original.save_netcdf(tmp_path / "complex.nc")
    restored = _DemoDataModel.load_netcdf(path)

    with Dataset(path, mode="r", auto_complex=True) as ds:
        assert "values" in ds.variables
        assert ds.variables["values"].dtype == np.dtype(np.complex128)
        assert "arr_real_0" not in ds.variables
        assert "arr_imag_0" not in ds.variables

    assert np.array_equal(restored.values, original.values)


def test_netcdf_roundtrip_handles_nested_model_fields(tmp_path) -> None:
    """Given nested model fields, when round-tripped via NetCDF, then model values are preserved."""
    original = _DemoDataModelWithNestedConfig(
        values=np.array([1.0 + 0.0j], dtype=np.complex128),
        config=_DemoConfig(
            shots=128, thresholds=np.array([0.1, 0.2], dtype=np.float64)
        ),
    )

    path = original.save_netcdf(tmp_path / "nested-model.nc")
    restored = _DemoDataModelWithNestedConfig.load_netcdf(path)

    assert np.array_equal(restored.values, original.values)
    assert restored.config.shots == original.config.shots
    assert np.array_equal(restored.config.thresholds, original.config.thresholds)
