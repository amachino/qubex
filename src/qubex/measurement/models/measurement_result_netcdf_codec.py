"""NetCDF codec for MeasurementResult."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.io import netcdf_file

if TYPE_CHECKING:
    from qubex.measurement.models.measurement_result import MeasurementResult


class MeasurementResultNetCDFCodec:
    """Codec that standardizes NetCDF serialization for MeasurementResult."""

    FORMAT_NAME = "measurement_result_netcdf"
    FORMAT_VERSION = 1

    ATTR_FORMAT = "qubex_format"
    ATTR_FORMAT_VERSION = "qubex_format_version"
    ATTR_RESULT_MODE = "result_mode"
    ATTR_DEVICE_CONFIG_JSON = "device_config_json"
    ATTR_MEASUREMENT_CONFIG_JSON = "measurement_config_json"
    ATTR_TARGETS_JSON = "targets_json"
    ATTR_INDEX_MAP_JSON = "index_map_json"

    def save(
        self,
        result: MeasurementResult,
        path: str | Path,
    ) -> Path:
        """
        Save measurement result to a NetCDF file.

        Parameters
        ----------
        result : MeasurementResult
            Result to serialize.
        path : str | Path
            Output `.nc` path.

        Returns
        -------
        Path
            Saved path.
        """
        path_obj = Path(path)
        target_names = list(result.data.keys())
        index_map: dict[str, list[dict[str, Any]]] = {}

        with netcdf_file(path_obj, mode="w") as ds:
            ds.qubex_format = self.FORMAT_NAME
            ds.qubex_format_version = self.FORMAT_VERSION
            ds.result_mode = result.mode
            ds.device_config_json = json.dumps(
                result.device_config,
                ensure_ascii=False,
            )
            ds.measurement_config_json = json.dumps(
                result.measurement_config,
                ensure_ascii=False,
            )
            ds.targets_json = json.dumps(target_names, ensure_ascii=False)

            for target_idx, target in enumerate(target_names):
                index_map[target] = []
                for capture_idx, raw in enumerate(result.data[target]):
                    values = np.asarray(raw)
                    dim = f"n_t{target_idx}_c{capture_idx}"
                    real_name = f"raw_real_t{target_idx}_c{capture_idx}"
                    imag_name = f"raw_imag_t{target_idx}_c{capture_idx}"
                    ds.createDimension(dim, values.size)
                    real_var = ds.createVariable(real_name, "f8", (dim,))
                    imag_var = ds.createVariable(imag_name, "f8", (dim,))
                    flat = values.reshape(-1)
                    real_var[:] = np.real(flat)
                    imag_var[:] = np.imag(flat)
                    index_map[target].append(
                        {
                            "capture_idx": capture_idx,
                            "shape": list(values.shape),
                            "real_var": real_name,
                            "imag_var": imag_name,
                        }
                    )

            ds.index_map_json = json.dumps(index_map, ensure_ascii=False)

        return path_obj

    def load(
        self,
        path: str | Path,
    ) -> MeasurementResult:
        """
        Load measurement result from a NetCDF file.

        Parameters
        ----------
        path : str | Path
            Input `.nc` path.

        Returns
        -------
        MeasurementResult
            Restored measurement result.
        """
        from qubex.measurement.models.measurement_result import MeasurementResult

        path_obj = Path(path)
        with netcdf_file(path_obj, mode="r") as ds:
            attrs = ds.__dict__
            self._validate_format(attrs)
            mode = self._as_str(attrs[self.ATTR_RESULT_MODE])
            device_config = self._parse_json_attr(attrs, self.ATTR_DEVICE_CONFIG_JSON)
            measurement_config = self._parse_json_attr(
                attrs,
                self.ATTR_MEASUREMENT_CONFIG_JSON,
            )
            index_map = self._parse_json_attr(attrs, self.ATTR_INDEX_MAP_JSON)

            data: dict[str, list[np.ndarray]] = {}
            for target, entries in index_map.items():
                captures: list[np.ndarray] = []
                for entry in entries:
                    real = np.copy(ds.variables[entry["real_var"]].data)
                    imag = np.copy(ds.variables[entry["imag_var"]].data)
                    flat = real + 1j * imag
                    shape = tuple(entry["shape"])
                    captures.append(flat.reshape(shape))
                data[target] = captures
        return MeasurementResult(
            mode=mode,  # type: ignore[arg-type]
            data=data,
            device_config=device_config,
            measurement_config=measurement_config,
        )

    def _validate_format(self, attrs: dict[str, Any]) -> None:
        """Validate format metadata if present."""
        format_name = attrs.get(self.ATTR_FORMAT)
        if format_name is None:
            return
        if self._as_str(format_name) != self.FORMAT_NAME:
            raise ValueError("Unsupported NetCDF format.")

        format_version = attrs.get(self.ATTR_FORMAT_VERSION)
        if format_version is None:
            return
        if int(format_version) > self.FORMAT_VERSION:
            raise ValueError("Unsupported NetCDF format version.")

    def _parse_json_attr(
        self,
        attrs: dict[str, Any],
        key: str,
    ) -> Any:
        """Parse JSON string attribute."""
        return json.loads(self._as_str(attrs[key]))

    @staticmethod
    def _as_str(value: Any) -> str:
        """Convert bytes-like or scalar value to str."""
        if isinstance(value, bytes):
            return value.decode()
        if hasattr(value, "item"):
            scalar = value.item()
            if isinstance(scalar, bytes):
                return scalar.decode()
            return str(scalar)
        return str(value)
