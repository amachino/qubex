from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qubecalib import QubeCalib
from qubecalib.neopulse import Sequence
from quel_ic_config import Quel1Box
from rich.console import Console

console = Console()


@dataclass
class QubeCalibResult:
    """Dataclass for measurement results."""

    status: dict
    data: dict
    config: dict


class QubeCalibWrapper:

    def __init__(self, config_path: str | Path):
        """
        Initialize the QubeCalibWrapper.

        Parameters
        ----------
        config_path : str | Path
            Path to the JSON configuration file of qube-calib.

        Examples
        --------
        >>> from qubex.qube_calib_wrapper import QubeCalibWrapper
        >>> qcw = QubeCalibWrapper("./system_settings.json")
        """
        try:
            self.qubecalib: Final = QubeCalib(str(config_path))
        except FileNotFoundError:
            console.print(
                f"Configuration file {config_path} not found.",
                style="bold red",
            )
            raise

    @property
    def system_config(self) -> dict[str, dict]:
        """Get the system configuration."""
        config = self.qubecalib.system_config_database.asdict()
        return config

    @property
    def box_settings(self) -> dict[str, dict]:
        """Get the box settings."""
        return self.system_config["box_settings"]

    @property
    def port_settings(self) -> dict[str, dict]:
        """Get the port settings."""
        return self.system_config["port_settings"]

    @property
    def target_settings(self) -> dict[str, dict]:
        """Get the target settings."""
        return self.system_config["target_settings"]

    @property
    def available_boxes(self) -> list[str]:
        """
        Get the list of available boxes.

        Returns
        -------
        list[str]
            List of available boxes.
        """
        return list(self.box_settings.keys())

    def _check_box_availabilty(self, box_name: str):
        if box_name not in self.available_boxes:
            raise ValueError(
                f"Box {box_name} not in available boxes: {self.available_boxes}"
            )

    def link_status(self, box_name: str) -> dict[int, bool]:
        """
        Get the link status of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.

        Returns
        -------
        dict[int, bool]
            Dictionary of link status.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.

        Examples
        --------
        >>> from qubex.qube_calib_wrapper import QubeCalibWrapper
        >>> qcw = QubeCalibWrapper("./system_settings.json")
        >>> qcw.link_status("Q73A")
        {0: True, 1: True}
        """
        self._check_box_availabilty(box_name)
        box = self.qubecalib.create_box(box_name, reconnect=False)
        return box.link_status()

    def linkup(
        self,
        box_name: str,
        noise_threshold: int = 500,
    ) -> Quel1Box:
        """
        Linkup a box and return the box object.

        Parameters
        ----------
        box_name : str
            Name of the box to linkup.

        Returns
        -------
        Quel1Box
            The linked up box object.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.

        Examples
        --------
        >>> from qubex.qube_calib_wrapper import QubeCalibWrapper
        >>> qcw = QubeCalibWrapper("./system_settings.json")
        >>> box = qcw.linkup("Q73A")
        """
        # check if the box is available
        self._check_box_availabilty(box_name)
        # connect to the box
        box = self.qubecalib.create_box(box_name, reconnect=False)
        # relinkup the box if any of the links are down
        if not all(box.link_status().values()):
            box.relinkup(use_204b=False, background_noise_threshold=noise_threshold)
        box.reconnect()
        # check if all links are up
        status = box.link_status()
        if not all(status.values()):
            console.print(
                f"Failed to linkup box {box_name}. Status: {status}",
                style="bold red",
            )
        # return the box
        return box

    def linkup_all(self) -> dict[str, Quel1Box]:
        """
        Linkup all available boxes.

        Returns
        -------
        dict[str, Quel1Box]
            Dictionary of linked up boxes.

        Examples
        --------
        >>> from qubex.qube_calib_wrapper import QubeCalibWrapper
        >>> qcw = QubeCalibWrapper("./system_settings.json")
        >>> boxes = qcw.linkup_all()
        """
        boxes = {}
        for box_name in self.available_boxes:
            try:
                boxes[box_name] = self.linkup(box_name)
                console.print(box_name, "Linked up", style="bold green")
            except Exception as e:
                console.print(box_name, "Error", e, style="bold red")
        return boxes

    def dump_box(self, box_name: str) -> dict:
        """
        Dump the box configuration.

        Parameters
        ----------
        box_name : str
            Name of the box.

        Returns
        -------
        dict
            Dictionary of box configuration.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.

        Examples
        --------
        >>> from qubex.qube_calib_wrapper import QubeCalibWrapper
        >>> qcw = QubeCalibWrapper("./system_settings.json")
        >>> qcw.dump_box("Q73A")
        """
        self._check_box_availabilty(box_name)
        box = self.linkup(box_name)
        box_config = box.dump_box()
        return box_config

    def add_sequence(self, sequence: Sequence):
        """
        Add a sequence to the queue.

        Parameters
        ----------
        sequence : Sequence
            The sequence to add to the queue.

        Examples
        --------
        >>> qcw = QubeCalibWrapper("./system_settings.json")
        >>> with Sequence() as sequence:
        ...     ...
        >>> qcw.add_sequence(sequence)
        """
        self.qubecalib.add_sequence(sequence)

    def show_command_queue(self):
        """Show the current command queue."""
        console.print(self.qubecalib.show_command_queue())

    def clear_command_queue(self):
        """Clear the command queue."""
        self.qubecalib.clear_command_queue()

    def execute(
        self,
        *,
        repeats: int,
        interval: int,
    ):
        """
        Execute the queue and yield measurement results.

        Parameters
        ----------
        repeats : int
            Number of repeats of each sequence.
        interval : int
            Interval between sequences.

        Yields
        ------
        QubeCalibResult
            Measurement result.

        Examples
        --------
        >>> qcw = QubeCalibWrapper("./system_settings.json")
        >>> with Sequence() as sequence:
        ...     ...
        >>> qcw.add_sequence(sequence)
        >>> for result in qcw.execute(repeats=100, interval=1024):
        ...     print(result)
        """
        for status, data, config in self.qubecalib.step_execute(
            repeats=repeats,
            interval=interval,
            integral_mode="integral",
            dsp_demodulation=True,
            software_demodulation=False,
        ):
            result = QubeCalibResult(
                status=status,
                data=data,
                config=config,
            )
            yield result

    def execute_sequence(
        self,
        sequence: Sequence,
        *,
        repeats: int,
        interval: int,
    ) -> QubeCalibResult:
        """
        Execute a single sequence and return the measurement result.

        Parameters
        ----------
        sequence : Sequence
            The sequence to execute.
        repeats : int
            Number of repeats of the sequence.
        interval : int
            Interval between sequences.

        Returns
        -------
        QubeCalibResult
            Measurement result.

        Examples
        --------
        >>> qcw = QubeCalibWrapper("./system_settings.json")
        >>> with Sequence() as sequence:
        ...     ...
        >>> result = qcw.execute_sequence(sequence, repeats=100, interval=1024)
        """
        self.clear_command_queue()
        self.add_sequence(sequence)
        return next(self.execute(repeats=repeats, interval=interval))
