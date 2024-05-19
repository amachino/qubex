from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import numpy.typing as npt
from qubecalib.neopulse import (
    Arbit,
    Capture,
    Flushleft,
    Flushright,
    RaisedCosFlatTop,
    padding,
)
from qubecalib.neopulse import (
    Sequence as Sequence,
)

from .config import Config
from .qube_backend import QubeBackend, QubeBackendResult

DEFAULT_CONFIG_DIR = "./config"
DEFAULT_SHOTS = 3000
DEFAULT_INTERVAL = 150 * 1024  # ns
DEFAULT_CONTROL_WINDOW = 1024  # ns
DEFAULT_CAPTURE_WINDOW = 1024  # ns
DEFAULT_READOUT_DURATION = 768  # ns


class MeasLevel(Enum):
    RAW = "raw"
    KERNELED = "kerneled"
    CLASSIFIED = "classified"


@dataclass
class MeasResult:
    meas_level: MeasLevel
    meas_data: dict[str, npt.NDArray]


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

    @property
    def targets(self) -> dict[str, float]:
        """Return the list of target names."""
        target_settings = self._backend.target_settings
        return {
            target: settings["frequency"]
            for target, settings in target_settings.items()
        }

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
        meas_level: MeasLevel = MeasLevel.KERNELED,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ) -> MeasResult:
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
        control_window : int, optional
            The control window in ns, by default DEFAULT_CONTROL_WINDOW.
        capture_window : int, optional
            The capture window in ns, by default DEFAULT_CAPTURE_WINDOW.
        readout_duration : int, optional
            The readout duration in ns, by default DEFAULT_READOUT_DURATION.

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
        sequence = self._create_sequence(
            waveforms=waveforms,
            control_window=control_window,
            capture_window=capture_window,
            readout_duration=readout_duration,
        )
        backend_result = self._backend.execute_sequence(
            sequence=sequence,
            repeats=shots,
            interval=interval,
        )
        result = self._create_measure_result(
            meas_level=meas_level,
            backend_result=backend_result,
        )
        return result

    def measure_batch(
        self,
        waveforms_list: list[dict[str, npt.NDArray[np.complex128]]],
        *,
        meas_level: MeasLevel = MeasLevel.KERNELED,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ):
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms_list : list[dict[str, npt.NDArray[np.complex128]]]
            The control waveforms for each target.
            Waveforms are complex I/Q arrays with the sampling period of 2 ns.
        shots : int, optional
            The number of shots, by default DEFAULT_SHOTS.
        interval : int, optional
            The interval in ns, by default DEFAULT_INTERVAL.
        control_window : int, optional
            The control window in ns, by default DEFAULT_CONTROL_WINDOW.
        capture_window : int, optional
            The capture window in ns, by default DEFAULT_CAPTURE_WINDOW.
        readout_duration : int, optional
            The readout duration in ns, by default DEFAULT_READOUT_DURATION.

        Yields
        ------
        MeasurementResult
            The measurement results.
        """
        self._backend.clear_command_queue()
        for waveforms in waveforms_list:
            sequence = self._create_sequence(
                waveforms=waveforms,
                control_window=control_window,
                capture_window=capture_window,
                readout_duration=readout_duration,
            )
            self._backend.add_sequence(sequence)
        backend_results = self._backend.execute(
            repeats=shots,
            interval=interval,
        )
        for backend_result in backend_results:
            backend_result = self._create_measure_result(
                meas_level=meas_level,
                backend_result=backend_result,
            )
            yield backend_result

    def _create_sequence(
        self,
        *,
        waveforms: dict[str, npt.NDArray[np.complex128]],
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ) -> Sequence:
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
        return sequence

    def _create_measure_result(
        self,
        *,
        meas_level: MeasLevel,
        backend_result: QubeBackendResult,
    ) -> MeasResult:
        label_slice = slice(1, None)
        capture_index = 0

        if meas_level == MeasLevel.RAW:
            meas_data = {
                target[label_slice]: iqs[capture_index]
                for target, iqs in backend_result.data.items()
            }
        elif meas_level == MeasLevel.KERNELED:
            meas_data = {
                target[label_slice]: iqs[capture_index].mean()
                for target, iqs in backend_result.data.items()
            }
        elif meas_level == MeasLevel.CLASSIFIED:
            raise NotImplementedError(f"MeasLevel {meas_level} is not supported.")

        result = MeasResult(
            meas_level=meas_level,
            meas_data=meas_data,
        )
        return result
