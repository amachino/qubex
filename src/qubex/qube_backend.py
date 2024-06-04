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
class QubeBackendResult:
    """Dataclass for measurement results."""

    status: dict
    data: dict
    config: dict


class QubeBackend:
    def __init__(self, config_path: str | Path):
        """
        Initialize the QubeBackend.

        Parameters
        ----------
        config_path : str | Path
            Path to the JSON configuration file of qube-calib.

        Examples
        --------
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
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
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.link_status("Q73A")
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
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
        >>> box = backend.linkup("Q73A")
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

    def linkup_boxes(self, box_list: list[str]) -> dict[str, Quel1Box]:
        """
        Linkup all the boxes in the list.

        Returns
        -------
        dict[str, Quel1Box]
            Dictionary of linked up boxes.

        Examples
        --------
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.linkup_boxes(["Q73A", "U10B"])
        """
        boxes = {}
        for box_name in box_list:
            try:
                boxes[box_name] = self.linkup(box_name)
                print(f"{box_name:5}", ":", "Linked up")
            except Exception as e:
                print(f"{box_name:5}", ":", "Error", e)
        return boxes

    def read_clocks(self, box_list: list[str]) -> list[tuple[bool, int, int]]:
        """
        Read the clocks of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.

        Returns
        -------
        list[tuple[bool, int, int]]
            List of clocks.

        Examples
        --------
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.read_clocks(["Q73A", "U10B"])
        """
        result = list(self.qubecalib.read_clock(*box_list))
        return result

    def check_clocks(self, box_list: list[str]) -> bool:
        """
        Check the clock of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.

        Returns
        -------
        bool
            True if the clocks are synchronized, False otherwise.

        Examples
        --------
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.check_clocks(["Q73A", "U10B"])
        """

        result = self.qubecalib.read_clock(*box_list)
        timestamps: list[str] = []
        accuracy = -8
        for _, clock, sysref_latch in result:
            timestamps.append(str(clock)[:accuracy])
            timestamps.append(str(sysref_latch)[:accuracy])
        timestamps = list(set(timestamps))
        synchronized = len(timestamps) == 1
        return synchronized

    def resync_clocks(self, box_list: list[str]) -> bool:
        """
        Resync the clock of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.

        Examples
        --------
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.resync_clocks(["Q73A", "U10B"])
        """
        self.qubecalib.resync(*box_list)
        return self.check_clocks(box_list)

    def sync_clocks(self, box_list: list[str]) -> bool:
        """
        Sync the clocks of the boxes if not synchronized.

        Parameters
        ----------
        box_list : list[str]
            List of box names.

        Examples
        --------
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.sync_clocks(["Q73A", "U10B"])
        """
        synchronized = self.check_clocks(box_list)
        if not synchronized:
            synchronized = self.resync_clocks(box_list)
            if not synchronized:
                console.print("Failed to synchronize clocks.", style="bold red")
        console.print("All clocks are synchronized.", style="bold green")
        return synchronized

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
        >>> from qubex.qube_backend import QubeBackend
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.dump_box("Q73A")
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
        >>> backend = QubeBackend("./system_settings.json")
        >>> with Sequence() as sequence:
        ...     ...
        >>> backend.add_sequence(sequence)
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
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
    ):
        """
        Execute the queue and yield measurement results.

        Parameters
        ----------
        repeats : int
            Number of repeats of each sequence.
        interval : int
            Interval between sequences.
        integral_mode : {"integral", "single"}, optional
            Integral mode.
        dsp_demodulation : bool, optional
            Enable DSP demodulation.
        software_demodulation : bool, optional
            Enable software demodulation.

        Yields
        ------
        QubeBackendResult
            Measurement result.

        Examples
        --------
        >>> backend = QubeBackend("./system_settings.json")
        >>> with Sequence() as sequence:
        ...     ...
        >>> backend.add_sequence(sequence)
        >>> for result in backend.execute(repeats=100, interval=1024):
        ...     print(result)
        """
        for status, data, config in self.qubecalib.step_execute(
            repeats=repeats,
            interval=interval,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
        ):
            result = QubeBackendResult(
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
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
    ) -> QubeBackendResult:
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
        integral_mode : {"integral", "single"}, optional
            Integral mode.
        dsp_demodulation : bool, optional
            Enable DSP demodulation.
        software_demodulation : bool, optional
            Enable software demodulation.

        Returns
        -------
        QubeBackendResult
            Measurement result.

        Examples
        --------
        >>> backend = QubeBackend("./system_settings.json")
        >>> with Sequence() as sequence:
        ...     ...
        >>> result = backend.execute_sequence(sequence, repeats=100, interval=1024)
        """
        self.clear_command_queue()
        self.add_sequence(sequence)
        return next(
            self.execute(
                repeats=repeats,
                interval=interval,
                integral_mode=integral_mode,
                dsp_demodulation=dsp_demodulation,
                software_demodulation=software_demodulation,
            )
        )

    def modify_target_frequency(self, target: str, frequency: float):
        """
        Modify the target frequency.

        Parameters
        ----------
        target : str
            Name of the target.
        frequency : float
            Modified frequency in GHz.

        Examples
        --------
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.modify_target_frequency("Q00", 10.0)
        """
        self.qubecalib.modify_target_frequency(target, frequency)

    def modify_target_frequencies(self, frequencies: dict[str, float]):
        """
        Modify the target frequencies.

        Parameters
        ----------
        frequencies : dict[str, float]
            Dictionary of target frequencies.

        Examples
        --------
        >>> backend = QubeBackend("./system_settings.json")
        >>> backend.modify_target_frequencies({"Q00": 10.0, "Q01": 10.0})
        """
        for target, frequency in frequencies.items():
            self.modify_target_frequency(target, frequency)
