"""Measurement result model."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field
from scipy.io import netcdf_file

from qubex.core.model import Model
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)


class MeasurementResult(Model):
    """Canonical serializable result of a measurement run."""

    mode: Literal["single", "avg"]
    data: dict[str, list[np.ndarray]]
    device_config: dict[str, Any] = Field(default_factory=dict)
    measurement_config: dict[str, Any] = Field(default_factory=dict)

    @property
    def measure_mode(self) -> MeasureMode:
        """Return the mode as `MeasureMode` enum."""
        return MeasureMode(self.mode)

    @classmethod
    def from_multiple(
        cls,
        multiple: MultipleMeasureResult,
    ) -> MeasurementResult:
        """
        Create a `MeasurementResult` from a legacy `MultipleMeasureResult`.

        Parameters
        ----------
        multiple : MultipleMeasureResult
            Legacy multiple-capture result.

        Returns
        -------
        MeasurementResult
            Canonical serializable measurement result.
        """
        data = {
            target: [np.asarray(item.raw) for item in captures]
            for target, captures in multiple.data.items()
        }
        return cls(
            mode=multiple.mode.value,
            data=data,
            device_config=multiple.config,
        )

    def to_multiple_measure_result(
        self,
        *,
        config: dict[str, Any] | None = None,
    ) -> MultipleMeasureResult:
        """
        Convert to the legacy multi-capture result type.

        Returns
        -------
        MultipleMeasureResult
            Legacy multi-capture result.
        """
        resolved_config: dict[str, Any] = (
            self.device_config if config is None else config
        )
        legacy_data = {
            target: [
                MeasureData(
                    target=target,
                    mode=self.measure_mode,
                    raw=np.asarray(raw),
                    classifier=None,
                )
                for raw in captures
            ]
            for target, captures in self.data.items()
        }
        return MultipleMeasureResult(
            mode=self.measure_mode,
            data=legacy_data,
            config=resolved_config,
        )

    def to_measure_result(
        self,
        *,
        index: int = 0,
        config: dict[str, Any] | None = None,
    ) -> MeasureResult:
        """
        Convert one capture index to a `MeasureResult`.

        Parameters
        ----------
        index : int, optional
            Capture index in each target's result list.
        config : dict[str, Any] | None, optional
            Optional legacy configuration to attach.

        Returns
        -------
        MeasureResult
            Per-target result for the selected capture index.

        Raises
        ------
        IndexError
            If `index` is out of range for any target.
        """
        single_data: dict[str, MeasureData] = {}
        for target, captures in self.data.items():
            if not (-len(captures) <= index < len(captures)):
                raise IndexError(
                    f"Capture index {index} is out of range for target {target}."
                )
            single_data[target] = MeasureData(
                target=target,
                mode=self.measure_mode,
                raw=np.asarray(captures[index]),
                classifier=None,
            )

        return MeasureResult(
            mode=self.measure_mode,
            data=single_data,
            config=self.device_config if config is None else config,
        )

    def save_netcdf(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save measurement result raw data and metadata in NetCDF format.

        Parameters
        ----------
        path : str | Path
            Output `.nc` path.

        Returns
        -------
        Path
            Saved path.
        """
        path_obj = Path(path)
        target_names = list(self.data.keys())
        index_map: dict[str, list[dict[str, Any]]] = {}

        with netcdf_file(path_obj, mode="w") as ds:
            ds.result_mode = self.mode
            ds.device_config_json = json.dumps(self.device_config, ensure_ascii=False)
            ds.measurement_config_json = json.dumps(
                self.measurement_config,
                ensure_ascii=False,
            )
            ds.targets_json = json.dumps(target_names, ensure_ascii=False)

            for target_idx, target in enumerate(target_names):
                index_map[target] = []
                for capture_idx, raw in enumerate(self.data[target]):
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

    def save(
        self,
        data_dir: str | Path | None = None,
        *,
        file_name: str | None = None,
    ) -> Path:
        """
        Save measurement result to a timestamped NetCDF file.

        Parameters
        ----------
        data_dir : str | Path | None, optional
            Output directory. If omitted, uses `.rawdata`.
        file_name : str | None, optional
            Output file name. Defaults to a timestamped `.nc` file.

        Returns
        -------
        Path
            Saved file path.
        """
        output_dir = Path(".rawdata" if data_dir is None else data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{timestamp}.nc"
        path = output_dir / file_name
        return self.save_netcdf(path)

    @classmethod
    def load_netcdf(
        cls,
        path: str | Path,
    ) -> MeasurementResult:
        """
        Load a measurement result from a NetCDF file.

        Parameters
        ----------
        path : str | Path
            Input `.nc` path.

        Returns
        -------
        MeasurementResult
            Restored measurement result.
        """
        path_obj = Path(path)
        with netcdf_file(path_obj, mode="r") as ds:
            attrs = ds.__dict__
            mode_attr = attrs["result_mode"]
            if isinstance(mode_attr, bytes):
                mode = mode_attr.decode()
            else:
                mode = str(mode_attr)
            device_config = json.loads(attrs["device_config_json"])
            measurement_config = json.loads(attrs["measurement_config_json"])
            index_map = json.loads(attrs["index_map_json"])

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
        return cls(
            mode=mode,  # type: ignore[arg-type]
            data=data,
            device_config=device_config,
            measurement_config=measurement_config,
        )
