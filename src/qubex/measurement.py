from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

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
from .qube_calib_wrapper import QubeCalibWrapper

DEFAULT_CONFIG_DIR = "./config"
DEFAULT_SHOTS = 3000
DEFAULT_INTERVAL = 150 * 1024  # ns
DEFAULT_CONTROL_WINDOW = 1024  # ns
DEFAULT_CAPTURE_WINDOW = 1024  # ns
DEFAULT_READOUT_DURATION = 768  # ns


class TargetType(Enum):
    """Enum for target types."""

    CTRL_GE = ""
    CTRL_EF = "_ef"
    CTRL_CR = "_CR"
    READ = "R"


def qubit_to_target(qubit: str, target_type: TargetType) -> str:
    """
    Convert qubit to target.

    Parameters
    ----------
    qubit : str
        The qubit name.
    target_type : TargetType
        The target type.

    Returns
    -------
    str
        The target name.

    Examples
    --------
    >>> qubit_to_target("Q00", TargetType.CTRL_GE)
    'Q00'
    >>> qubit_to_target("Q00", TargetType.CTRL_EF)
    'Q00_ef'
    >>> qubit_to_target("Q00", TargetType.CTRL_CR)
    'Q00_CR'
    >>> qubit_to_target("Q00", TargetType.READ)
    'RQ00'
    """
    result = ""
    if target_type == TargetType.READ:
        result = f"R{qubit}"
    else:
        result = f"{qubit}{target_type.value}"
    return result


def target_to_qubit(target: str) -> str:
    """
    Convert target to qubit.

    Parameters
    ----------
    target : str
        The target name.

    Returns
    -------
    str
        The qubit name.

    Examples
    --------
    >>> target_to_qubit("Q00")
    'Q00'
    >>> target_to_qubit("Q00_ef")
    'Q00'
    >>> target_to_qubit("Q00_CR")
    'Q00'
    >>> target_to_qubit("RQ00")
    'Q00'
    """
    if target.startswith("R"):
        return target[1:]
    return target.split("_")[0]


@dataclass
class MeasurementResult:
    """Dataclass for measurement results."""

    data: dict[str, complex]


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
        config_path = Config(config_dir).get_system_settings_path(chip_id)
        self.backend: Final = QubeCalibWrapper(config_path)

    @property
    def targets(self) -> list[str]:
        """Get the list of targets."""
        return list(self.backend.target_settings.keys())

    def measure(
        self,
        waveforms: dict[str, list | npt.NDArray],
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
        waveforms : dict[str, list | npt.NDArray]
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
                    read_target = qubit_to_target(target, TargetType.READ)
                    readout_pulse.target(read_target)
                    capture.target(read_target)

        raw_result = self.backend.execute_sequence(
            sequence=sequence,
            repeats=shots,
            interval=interval,
        )

        data: dict[str, complex] = {
            target_to_qubit(target): iqs[0].squeeze().mean()
            for target, iqs in raw_result.data.items()
        }

        result = MeasurementResult(
            data=data,
        )

        return result
