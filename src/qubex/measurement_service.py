from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from qubecalib.neopulse import (
    DEFAULT_SAMPLING_PERIOD,
    Arbit,
    Blank,
    Capture,
    Flushleft,
    Flushright,
    RaisedCosFlatTop,
    Sequence,
)

from .qube_calib_wrapper import QubeCalibWrapper

DEFAULT_SHOTS = 3000
DEFAULT_INTERVAL = 150 * 1024
DEFAULT_CONTROL_WINDOW = 1024


@dataclass
class MeasurementResult:
    """Dataclass for measurement results."""

    data: dict[str, npt.NDArray[np.complex64]]


class MeasurementService:

    def __init__(
        self,
        config_file: str,
    ):
        self.backend = QubeCalibWrapper(config_file)

    def measure(
        self,
        waveforms: dict[str, list | npt.NDArray],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
    ) -> MeasurementResult:
        """
        Measure the given waveforms.

        Parameters
        ----------
        waveforms : dict[str, list | npt.NDArray]
            The waveforms to measure.
        shots : int, optional
            The number of shots, by default DEFAULT_SHOTS.
        interval : int, optional
            The interval in ns, by default DEFAULT_INTERVAL.

        Returns
        -------
        MeasurementResult
            The measurement results.
        """
        readout_pulse = RaisedCosFlatTop(
            duration=1024,
            amplitude=0.1,
            rise_time=128,
        )
        capture = Capture(duration=3 * 1024)

        with Sequence() as sequence:
            with Flushright():
                Blank(control_window).target()
                for target, waveform in waveforms.items():
                    arbit = Arbit(duration=len(waveform) * DEFAULT_SAMPLING_PERIOD)
                    arbit.iq[:] = waveform
                    arbit.target(f"C{target}")
            with Flushleft():
                for target in waveforms.keys():
                    readout_pulse.target(f"R{target}")
                    capture.target(f"R{target}")

        raw_result = self.backend.execute_sequence(
            sequence=sequence,
            repeats=shots,
            interval=interval,
        )

        data: dict[str, npt.NDArray[np.complex64]] = {
            target[1:]: iqs[0].squeeze() for target, iqs in raw_result.data.items()
        }

        result = MeasurementResult(
            data=data,
        )

        return result
