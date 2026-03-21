"""NetCDF-specific serialization utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, TypeGuard

import numpy as np
import tunits
from netCDF4 import Dataset
from pydantic import BaseModel

from .constants import (
    DATA_TYPE_KEY,
    META_FORMAT_KEY,
    META_KEY,
    META_VERSION_KEY,
    TYPE_NUMPY_NDARRAY,
    TYPE_TUNITS_VALUE_ARRAY,
)
from .json_serializer import (
    deserialize_tunits,
    deserialize_value,
    serialize_tunits,
    serialize_value,
)

_VARIABLE_REF_KEY: Final[str] = "__variable__"
_ATTR_FORMAT: Final[str] = "format"
_ATTR_FORMAT_VERSION: Final[str] = "format_version"
_ATTR_MODEL_CLASS: Final[str] = "model_class"
_ATTR_PAYLOAD_JSON: Final[str] = "payload_json"


class _VariableRef(TypedDict):
    """Reference metadata for a variable stored in NetCDF."""

    name: str
    type: str
    shape: list[int]
    units: list[dict[str, Any]]


class _EncodedArray(TypedDict):
    """Array payload queued for NetCDF variable writing."""

    name: str
    type: str
    values: np.ndarray


def deserialize_tunits_value_array(
    *,
    units: list[dict[str, Any]],
    values: np.ndarray,
) -> Any:
    """
    Deserialize a tunits.ValueArray from NetCDF metadata.

    Parameters
    ----------
    units : list[dict[str, Any]]
        Serialized unit metadata produced by `serialize_tunits`.
    values : np.ndarray
        Numeric values stored in the NetCDF variable.

    Returns
    -------
    Any
        Restored `tunits.ValueArray` instance.

    Raises
    ------
    TypeError
        If the payload cannot be restored to a `tunits.ValueArray`.
    """
    payload: dict[str, Any] = {
        DATA_TYPE_KEY: TYPE_TUNITS_VALUE_ARRAY,
        "units": units,
        "shape": list(values.shape),
    }

    flat = np.asarray(values).reshape(-1)
    if np.iscomplexobj(flat):
        payload["complexes"] = {
            "values": [
                {"real": float(item.real), "imaginary": float(item.imag)}
                for item in flat
            ]
        }
    else:
        payload["reals"] = {"values": [float(item) for item in flat]}

    restored = deserialize_tunits(payload)
    if _is_tunits_value_array(restored):
        return restored
    raise TypeError("Failed to deserialize tunits.ValueArray.")


def save_netcdf_file(
    *,
    model: Any,
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Save model field data as a NetCDF file.

    Parameters
    ----------
    model : Any
        Model instance providing format metadata.
    data : dict[str, Any]
        Field data payload to serialize.
    path : str | Path
        Destination path for the NetCDF file.

    Returns
    -------
    Path
        Path to the written NetCDF file.
    """
    path_obj = Path(path)
    arrays: list[_EncodedArray] = []
    payload = _encode_value(data, arrays, path=())
    payload[META_KEY] = {
        META_FORMAT_KEY: model.format_name,
        META_VERSION_KEY: model.format_version,
    }

    with _open_netcdf(path_obj, mode="w") as ds:
        ds.setncattr(_ATTR_FORMAT, model.format_name)
        ds.setncattr(_ATTR_FORMAT_VERSION, model.format_version)
        ds.setncattr(
            _ATTR_MODEL_CLASS,
            f"{model.__class__.__module__}.{model.__class__.__name__}",
        )
        ds.setncattr(_ATTR_PAYLOAD_JSON, json.dumps(payload, ensure_ascii=False))
        for item in arrays:
            _write_array(ds, item)
    return path_obj


def load_netcdf_file(
    *,
    model_cls: type[Any],
    path: str | Path,
) -> dict[str, Any]:
    """
    Load model field data from a NetCDF file.

    Parameters
    ----------
    model_cls : type[Any]
        Model class providing format metadata.
    path : str | Path
        Source NetCDF path.

    Returns
    -------
    dict[str, Any]
        Restored field data.

    Raises
    ------
    TypeError
        If the decoded payload is not a mapping.
    ValueError
        If the NetCDF format metadata is incompatible.
    """
    path_obj = Path(path)
    with _open_netcdf(path_obj, mode="r") as ds:
        _validate_netcdf_format(
            ds,
            expected_format=model_cls.format_name,
            expected_version=model_cls.format_version,
        )
        payload_json = _as_str(ds.getncattr(_ATTR_PAYLOAD_JSON))
        payload = json.loads(payload_json)
        restored = _decode_value(payload, ds)
        data = deserialize_value(restored)
    if not isinstance(data, dict):
        raise TypeError("Decoded NetCDF payload is not a mapping.")
    return data


def _encode_value(
    value: Any,
    arrays: list[_EncodedArray],
    *,
    path: tuple[str, ...],
) -> Any:
    """
    Encode a nested value for NetCDF storage.

    Parameters
    ----------
    value : Any
        Value to encode.
    arrays : list[_EncodedArray]
        Accumulator for array payloads to store as NetCDF variables.
    path : tuple[str, ...]
        Path segments used to generate stable variable names.

    Returns
    -------
    Any
        Encoded payload with array references where applicable.
    """
    if isinstance(value, np.ndarray):
        name = _path_name(path, fallback=f"field_{len(arrays)}")
        arrays.append(
            {
                "name": name,
                "type": TYPE_NUMPY_NDARRAY,
                "values": np.asarray(value),
            }
        )
        ref: _VariableRef = {
            "name": name,
            "type": TYPE_NUMPY_NDARRAY,
            "shape": list(value.shape),
            "units": [],
        }
        return {_VARIABLE_REF_KEY: ref}
    if _is_tunits_value_array(value):
        values = np.asarray(value.value)
        name = _path_name(path, fallback=f"field_{len(arrays)}")
        tunits_payload = serialize_tunits(value)
        units = tunits_payload.get("units")
        if not isinstance(units, list):
            raise TypeError("Serialized tunits payload does not include `units`.")
        arrays.append(
            {
                "name": name,
                "type": TYPE_TUNITS_VALUE_ARRAY,
                "values": values,
            }
        )
        ref: _VariableRef = {
            "name": name,
            "type": TYPE_TUNITS_VALUE_ARRAY,
            "shape": list(values.shape),
            "units": units,
        }
        return {_VARIABLE_REF_KEY: ref}
    if isinstance(value, BaseModel):
        return _encode_value(
            value.model_dump(mode="python"),
            arrays,
            path=path,
        )
    if isinstance(value, dict):
        return {
            k: _encode_value(v, arrays, path=(*path, str(k))) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _encode_value(v, arrays, path=(*path, f"i{idx}"))
            for idx, v in enumerate(value)
        ]
    return serialize_value(value)


def _decode_value(value: Any, ds: Dataset) -> Any:
    """
    Decode a NetCDF payload into Python values.

    Parameters
    ----------
    value : Any
        Encoded payload to decode.
    ds : Dataset
        Open NetCDF dataset containing array variables.

    Returns
    -------
    Any
        Decoded value with arrays restored.
    """
    if isinstance(value, dict):
        ref = value.get(_VARIABLE_REF_KEY)
        if _is_variable_ref(ref):
            shape = tuple(ref["shape"])
            array_type = str(ref["type"])
            name = str(ref["name"])
            array = _read_array(ds, name, shape)
            if array_type == TYPE_TUNITS_VALUE_ARRAY:
                return deserialize_tunits_value_array(
                    units=ref["units"],
                    values=array,
                )
            return array
        return {k: _decode_value(v, ds) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v, ds) for v in value]
    return value


def _is_variable_ref(value: Any) -> TypeGuard[_VariableRef]:
    """
    Return whether a value is a variable-reference payload.

    Parameters
    ----------
    value : Any
        Candidate payload to inspect.

    Returns
    -------
    bool
        `True` if the payload matches the variable-reference shape.
    """
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("name"), str)
        and isinstance(value.get("type"), str)
        and isinstance(value.get("shape"), list)
        and isinstance(value.get("units"), list)
    )


def _write_array(ds: Dataset, item: _EncodedArray) -> None:
    """
    Write an encoded array as a NetCDF variable.

    Parameters
    ----------
    ds : Dataset
        Open NetCDF dataset to write into.
    item : _EncodedArray
        Encoded array payload to persist.
    """
    values = np.asarray(item["values"])
    base_name = str(item["name"])
    dim_names = _dimension_names(base_name, values.ndim)
    for axis, dim_name in enumerate(dim_names):
        ds.createDimension(dim_name, values.shape[axis])
    var_name = _variable_name(base_name)
    var, write_values = _create_variable(ds, var_name, values, dim_names)
    var[:] = write_values


def _create_variable(
    ds: Dataset,
    name: str,
    values: np.ndarray,
    dim_names: tuple[str, ...],
) -> tuple[Any, np.ndarray]:
    """
    Create a NetCDF variable and return normalized write values.

    Parameters
    ----------
    ds : Dataset
        Open NetCDF dataset.
    name : str
        Variable name to create.
    values : np.ndarray
        Array values to store.
    dim_names : tuple[str, ...]
        Dimension names for the variable.

    Returns
    -------
    tuple[Any, np.ndarray]
        Newly created variable and array values to write.
    """
    if np.iscomplexobj(values):
        typecode = "c8" if np.dtype(values.dtype) == np.dtype(np.complex64) else "c16"
        return ds.createVariable(name, typecode, dim_names), values

    dtype = np.dtype(values.dtype)
    if np.issubdtype(dtype, np.floating):
        return ds.createVariable(name, np.float64, dim_names), values
    if np.issubdtype(dtype, np.integer):
        return ds.createVariable(name, np.int64, dim_names), values
    if np.issubdtype(dtype, np.bool_):
        return ds.createVariable(name, np.int8, dim_names), values.astype(np.int8)
    return ds.createVariable(name, np.float64, dim_names), values.astype(np.float64)


def _read_array(ds: Dataset, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """
    Read a NetCDF variable and reshape it to the expected shape.

    Parameters
    ----------
    ds : Dataset
        Open NetCDF dataset.
    name : str
        Base variable name.
    shape : tuple[int, ...]
        Expected shape of the array.

    Returns
    -------
    np.ndarray
        Restored array values.
    """
    var_name = _variable_name(name)
    data = np.asarray(ds.variables[var_name][:]).copy()
    if data.shape == shape:
        return data
    return data.reshape(shape)


def _path_name(path: tuple[str, ...], *, fallback: str) -> str:
    """
    Build a deterministic variable name from a path tuple.

    Parameters
    ----------
    path : tuple[str, ...]
        Path segments used to build the name.
    fallback : str
        Name to use when the path is empty.

    Returns
    -------
    str
        Sanitized variable name.
    """
    if not path:
        return fallback
    return "_".join(_sanitize_name(part) for part in path)


def _sanitize_name(name: str) -> str:
    """
    Sanitize a string to be a valid NetCDF identifier.

    Parameters
    ----------
    name : str
        Raw name to sanitize.

    Returns
    -------
    str
        Name containing only alphanumerics and underscores.
    """
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not sanitized:
        return "field"
    return sanitized


def _variable_name(base_name: str) -> str:
    """
    Return the NetCDF variable name for a base name.

    Parameters
    ----------
    base_name : str
        Base name derived from the payload path.

    Returns
    -------
    str
        Variable name used in the dataset.
    """
    return base_name


def _dimension_names(base_name: str, ndim: int) -> tuple[str, ...]:
    """
    Build dimension names for a variable.

    Parameters
    ----------
    base_name : str
        Base variable name.
    ndim : int
        Number of dimensions in the array.

    Returns
    -------
    tuple[str, ...]
        Dimension names in axis order.
    """
    if ndim == 0:
        return ()
    return tuple(f"{base_name}_d{axis}" for axis in range(ndim))


def _validate_netcdf_format(
    ds: Dataset,
    *,
    expected_format: str,
    expected_version: int,
) -> None:
    """
    Validate NetCDF format metadata against expected values.

    Parameters
    ----------
    ds : Dataset
        Open NetCDF dataset.
    expected_format : str
        Required format name.
    expected_version : int
        Maximum supported format version.

    Raises
    ------
    ValueError
        If the dataset uses an unsupported format or version.
    """
    format_name = _as_str(ds.getncattr(_ATTR_FORMAT))
    if format_name and format_name != expected_format:
        raise ValueError("Unsupported NetCDF format.")

    version = ds.getncattr(_ATTR_FORMAT_VERSION)
    if int(version) > expected_version:
        raise ValueError("Unsupported NetCDF format version.")


def _open_netcdf(path: Path, mode: Literal["r", "w"]) -> Dataset:
    """
    Open a NetCDF4 dataset.

    Parameters
    ----------
    path : Path
        Path to the NetCDF file.
    mode : Literal["r", "w"]
        Open mode (`"r"` for read, `"w"` for write).

    Returns
    -------
    Dataset
        Open NetCDF dataset instance.

    """
    if mode == "w":
        return Dataset(path, mode=mode, format="NETCDF4", auto_complex=True)
    return Dataset(path, mode=mode, auto_complex=True)


def _as_str(value: Any) -> str:
    """
    Coerce NetCDF attribute values to text.

    Parameters
    ----------
    value : Any
        Attribute value to convert.

    Returns
    -------
    str
        Text representation of the attribute value.
    """
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, bytes):
            return scalar.decode()
        return str(scalar)
    return str(value)


def _is_tunits_value_array(value: Any) -> bool:
    """
    Return whether a value is a tunits ValueArray instance.

    Parameters
    ----------
    value : Any
        Candidate value to inspect.

    Returns
    -------
    bool
        `True` if the value is a tunits `ValueArray`.
    """
    return isinstance(value, tunits.ValueArray)
