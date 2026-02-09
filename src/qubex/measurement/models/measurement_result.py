"""Measurement result model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from qubex.constants import DEFAULT_RAWDATA_DIR
from qubex.core import DataModel
from qubex.typing import MeasurementMode


class MeasurementResult(DataModel):
    """Canonical serializable result of a measurement run."""

    mode: MeasurementMode
    data: dict[str, list[np.ndarray]]
    device_config: dict[str, Any] = Field(default_factory=dict)
    measurement_config: dict[str, Any] = Field(default_factory=dict)

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
        output_dir = Path(DEFAULT_RAWDATA_DIR if data_dir is None else data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{timestamp}.nc"
        path = output_dir / file_name
        return self.save_netcdf(path)
