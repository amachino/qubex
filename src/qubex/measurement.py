from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from qubecalib.neopulse import (
    Arbit,
    Capture,
    Flushleft,
    Flushright,
    RaisedCosFlatTop,
    Sequence,
    padding,
)

from .config import Config
from .qube_backend import QubeBackend

DEFAULT_CONFIG_DIR = "./config"
DEFAULT_SHOTS = 3000
DEFAULT_INTERVAL = 150 * 1024  # ns
DEFAULT_CONTROL_WINDOW = 1024  # ns
DEFAULT_CAPTURE_WINDOW = 1024  # ns
DEFAULT_READOUT_DURATION = 768  # ns


@dataclass
class MeasurementResult:
    """Dataclass for measurement results."""

    data: dict[str, npt.NDArray[np.complex64]]


class Measurement:

    def __init__(
        self,
        chip_id: str,
        *,
        config_dir: str = DEFAULT_CONFIG_DIR,
    ):
        """
        Initialize the MeasurementService.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        config_dir : str, optional
            The configuration directory, by default "./config".

        Examples
        --------
        >>> from qubex import Measurement
        >>> meas = Measurement("64Q")
        """
        config = Config(config_dir)
        config.configure_system_settings(chip_id)
        config_path = config.get_system_settings_path(chip_id)
        self._backend: Final = QubeBackend(config_path)

    def connect(self) -> None:
        """Connect to the backend."""
        available_boxes = self._backend.available_boxes
        self._backend.linkup_boxes(available_boxes)
        self._backend.sync_clocks(available_boxes)

    def dump_box_config(self, box_id: str) -> dict:
        """
        Dump the configuration of the box.

        Parameters
        ----------
        box_id : str
            The box ID.

        Returns
        -------
        dict
            The configuration of the box.

        Examples
        --------
        >>> config = meas.dump_box_config("Q73A")
        """
        return self._backend.dump_box(box_id)

    def measure(
        self,
        waveforms: dict[str, npt.NDArray[np.complex128]],
        *,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ) -> MeasurementResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms : dict[str, npt.NDArray[np.complex128]]
            The control waveforms for each target.
            Waveforms are complex I/Q arrays with the sampling period of 2 ns.
        shots : int, optional
            The number of shots, by default DEFAULT_SHOTS.
        interval : int, optional
            The interval in ns, by default DEFAULT_INTERVAL.

        Returns
        -------
        MeasurementResult
            The measurement results.

        Examples
        --------
        >>> result = meas.measure({
        ...     "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...     "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ... })
        """
        readout_pulse = RaisedCosFlatTop(
            duration=readout_duration,
            amplitude=0.1,
            rise_time=64,
        )
        capture = Capture(duration=capture_window)

        with Sequence() as sequence:
            with Flushright():
                padding(control_window)
                for target, waveform in waveforms.items():
                    Arbit(waveform).target(target)
            with Flushleft():
                for target in waveforms.keys():
                    read_target = f"R{target}"
                    readout_pulse.target(read_target)
                    capture.target(read_target)

        raw_result = self._backend.execute_sequence(
            sequence=sequence,
            repeats=shots,
            interval=interval,
        )

        data: dict[str, npt.NDArray[np.complex64]] = {
            target[1:]: np.array(iqs[0], dtype=np.complex64)
            for target, iqs in raw_result.data.items()
        }

        result = MeasurementResult(
            data=data,
        )

        return result
