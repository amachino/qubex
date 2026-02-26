"""Measurement result model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import qubex.visualization as viz
from qubex.constants import DEFAULT_RAWDATA_DIR
from qubex.core import DataModel

from .measurement_config import MeasurementConfig


class MeasurementResult(DataModel):
    """Canonical serializable result of a measurement run."""

    data: dict[str, list[np.ndarray]]
    measurement_config: MeasurementConfig
    device_config: dict[str, Any] | None = None
    sampling_period_ns: float

    def plot(self) -> None:
        """Plot measurement data for each capture."""
        sampling_period = float(self.sampling_period_ns)

        for target, captures in self.data.items():
            for capture_index, raw in enumerate(captures):
                title = f"{target} : data[{capture_index}]"
                if self.measurement_config.shot_averaging:
                    waveform = np.asarray(raw)
                    viz.plot_waveform(
                        data=waveform,
                        sampling_period=sampling_period,
                        title=title,
                        xlabel="Capture time (ns)",
                        ylabel="Signal (arb. units)",
                        save_image=False,
                    )
                else:
                    shots = np.asarray(raw)
                    kerneled = shots if shots.ndim <= 1 else np.sum(shots, axis=1)
                    iq_data = {target: kerneled}
                    viz.scatter_iq_data(
                        data=iq_data,
                        title=title,
                        save_image=False,
                    )

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
