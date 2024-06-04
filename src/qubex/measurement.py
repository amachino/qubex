from __future__ import annotations

import typing
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from qubecalib.neopulse import (
    Arbit,
    Capture,
    Flushleft,
    Flushright,
    RaisedCosFlatTop,
    Sequence,
    padding,
)

from . import visualization as viz
from .config import Config
from .qube_backend import QubeBackend, QubeBackendResult
from .typing import IQArray, TargetMap

DEFAULT_CONFIG_DIR = "./config"
DEFAULT_SHOTS = 1024
DEFAULT_INTERVAL = 150 * 1024  # ns
DEFAULT_CONTROL_WINDOW = 1024  # ns
DEFAULT_CAPTURE_WINDOW = 1024  # ns
DEFAULT_READOUT_DURATION = 512  # ns


class MeasureMode(Enum):
    SINGLE = "single"
    AVG = "avg"

    @property
    def integral_mode(self) -> str:
        return {
            MeasureMode.SINGLE: "single",
            MeasureMode.AVG: "integral",
        }[self]


@dataclass
class MeasureData:
    raw: NDArray
    kerneled: NDArray
    classified: NDArray


@dataclass
class MeasureResult:
    mode: MeasureMode
    data: dict[str, MeasureData]

    def plot(self):
        if self.mode == MeasureMode.SINGLE:
            data = {qubit: data.kerneled for qubit, data in self.data.items()}
            viz.scatter_iq_data(data=data)
        elif self.mode == MeasureMode.AVG:
            for qubit, data in self.data.items():
                viz.plot_waveform(
                    data=data.raw,
                    sampling_period=8,
                    title=f"Readout waveform of {qubit}",
                    xlabel="Capture time (ns)",
                    ylabel="Amplitude (arb. units)",
                )


class Measurement:
    def __init__(
        self,
        chip_id: str,
        *,
        config_dir: str = DEFAULT_CONFIG_DIR,
    ):
        """
        Initialize the Measurement.

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
        self._params: Final = config.get_params(chip_id)

    @property
    def targets(self) -> dict[str, float]:
        """Return the list of target names."""
        target_settings = self._backend.target_settings
        return {
            target: settings["frequency"]
            for target, settings in target_settings.items()
        }

    def linkup(self, box_list: list[str]):
        """
        Link up the boxes and synchronize the clocks.

        Parameters
        ----------
        box_list : list[str]
            The list of box IDs.

        Examples
        --------
        >>> meas.linkup(["Q73A", "U10B"])
        """
        self._backend.linkup_boxes(box_list)
        self._backend.sync_clocks(box_list)

    def measure_noise(
        self,
        targets: list[str],
        duration: int,
    ) -> MeasureResult:
        """
        Measure the readout noise.

        Parameters
        ----------
        targets : list[str]
            The list of target names.
        duration : int, optional
            The duration in ns.

        Returns
        -------
        MeasureResult
            The measurement results.

        Examples
        --------
        >>> result = meas.measure_noise()
        """
        capture = Capture(duration=duration)
        with Sequence() as sequence:
            with Flushleft():
                for target in targets:
                    capture.target(f"R{target}")
        backend_result = self._backend.execute_sequence(
            sequence=sequence,
            repeats=1,
            interval=DEFAULT_INTERVAL,
            integral_mode="single",
        )
        return self._create_measure_result(backend_result, MeasureMode.SINGLE)

    def measure(
        self,
        waveforms: TargetMap[IQArray],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ) -> MeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms : TargetMap[IQArray]
            The control waveforms for each target.
            Waveforms are complex I/Q arrays with the sampling period of 2 ns.
        mode : Literal["single", "avg"], optional
            The measurement mode, by default "single".
            - "single": Measure once.
            - "avg": Measure multiple times and average the results.
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
        MeasureResult
            The measurement results.

        Examples
        --------
        >>> result = meas.measure({
        ...     "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...     "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ... })
        """
        measure_mode = MeasureMode(mode)
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
            integral_mode=measure_mode.integral_mode,
        )
        return self._create_measure_result(backend_result, measure_mode)

    def measure_batch(
        self,
        waveforms_list: typing.Sequence[TargetMap[IQArray]],
        *,
        mode: Literal["single", "avg"] = "avg",
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
        waveforms_list : Sequence[TargetMap[IQArray]]
            The control waveforms for each target.
            Waveforms are complex I/Q arrays with the sampling period of 2 ns.
        mode : Literal["single", "avg"], optional
            The measurement mode, by default "single".
            - "single": Measure once.
            - "avg": Measure multiple times and average the results.
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
        MeasureResult
            The measurement results.
        """
        measure_mode = MeasureMode(mode)
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
            integral_mode=measure_mode.integral_mode,
        )
        for backend_result in backend_results:
            yield self._create_measure_result(backend_result, measure_mode)

    def _create_sequence(
        self,
        *,
        waveforms: TargetMap[IQArray],
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ) -> Sequence:
        readout_amplitude = self._params.readout_amplitude
        capture = Capture(duration=capture_window)
        with Sequence() as sequence:
            with Flushright():
                padding(control_window)
                for target, waveform in waveforms.items():
                    Arbit(np.array(waveform)).target(target)
            with Flushleft():
                for target in waveforms.keys():
                    read_target = f"R{target}"
                    RaisedCosFlatTop(
                        duration=readout_duration,
                        amplitude=readout_amplitude[target],
                        rise_time=32,
                    ).target(read_target)
                    capture.target(read_target)
        return sequence

    def _create_measure_result(
        self,
        backend_result: QubeBackendResult,
        measure_mode: MeasureMode,
    ) -> MeasureResult:
        label_slice = slice(1, None)  # Remove the prefix "R"
        capture_index = 0

        measure_data = {}
        for target, iqs in backend_result.data.items():
            qubit = target[label_slice]

            if measure_mode == MeasureMode.SINGLE:
                # iqs: ndarray[duration, shots]
                raw = iqs[capture_index].T.squeeze()
                kerneled = np.mean(iqs[capture_index], axis=0) * 2 ** (-32)
                classified_data = np.array([])
            elif measure_mode == MeasureMode.AVG:
                # iqs: ndarray[duration, 1]
                raw = iqs[capture_index].squeeze()
                kerneled = np.mean(iqs) * 2 ** (-32)
                classified_data = np.array([])
            else:
                raise ValueError(f"Invalid measure mode: {measure_mode}")

            measure_data[qubit] = MeasureData(
                raw=raw,
                kerneled=kerneled,
                classified=classified_data,
            )

        return MeasureResult(
            mode=measure_mode,
            data=measure_data,
        )

    @contextmanager
    def modified_frequencies(self, target_frequencies: dict[str, float]):
        """
        Temporarily modify the target frequencies.

        Parameters
        ----------
        target_frequencies : dict[str, float]
            The target frequencies to be modified.

        Examples
        --------
        >>> with meas.modified_frequencies({"Q00": 5.0}):
        ...     result = meas.measure({
        ...         "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...         "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ...     })
        """
        original_frequencies = self.targets
        self._backend.modify_target_frequencies(target_frequencies)
        try:
            yield
        finally:
            self._backend.modify_target_frequencies(original_frequencies)
