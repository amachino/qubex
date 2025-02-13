from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

try:
    from qubecalib import QubeCalib, Sequencer
    from qubecalib.neopulse import Sequence
    from quel_ic_config import Quel1Box
except ImportError:
    pass

SAMPLING_PERIOD: Final[float] = 2.0  # ns


@dataclass
class RawResult:
    status: dict
    data: dict
    config: dict


class DeviceController:
    def __init__(
        self,
        config_path: str | Path | None = None,
    ):
        if config_path is None:
            self.qubecalib = QubeCalib()
        else:
            try:
                self.qubecalib = QubeCalib(str(config_path))
            except FileNotFoundError:
                print(f"Configuration file {config_path} not found.")
                raise
        self._boxpool = None

    @property
    def system_config(self) -> dict[str, dict]:
        """Get the system configuration."""
        config = self.qubecalib.system_config_database.asdict()
        return config

    @property
    def system_config_json(self) -> str:
        """Get the system configuration as JSON."""
        config = self.qubecalib.system_config_database.asjson()
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

    @property
    def boxpool(self):
        """
        Get the boxpool.

        Returns
        -------
        BoxPool
            The boxpool.
        """
        if self._boxpool is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        return self._boxpool

    @property
    def hash(self) -> int:
        """
        Get the hash of the system configuration.

        Returns
        -------
        int
            Hash of the system configuration.
        """
        return hash(self.qubecalib.system_config_database.asjson())

    def _check_box_availabilty(self, box_name: str):
        if box_name not in self.available_boxes:
            raise ValueError(
                f"Box {box_name} not in available boxes: {self.available_boxes}"
            )

    def get_resource_map(self, targets: list[str]) -> dict[str, list[dict]]:
        db = self.qubecalib.system_config_database
        result = {}
        for target in targets:
            if target not in db._target_settings:
                raise ValueError(f"Target {target} not in available targets.")

            channels = db.get_channels_by_target(target)
            bpc_list = [db.get_channel(channel) for channel in channels]
            result[target] = [
                {
                    "box": db._box_settings[box_name],
                    "port": db._port_settings[port_name],
                    "channel_number": channel_number,
                    "target": db._target_settings[target],
                }
                for box_name, port_name, channel_number in bpc_list
            ]
        return result

    def clear_cache(self):
        if self._boxpool is not None:
            self._boxpool._box_config_cache.clear()

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
        """
        self._check_box_availabilty(box_name)
        box = self.qubecalib.create_box(box_name, reconnect=False)
        return box.link_status()

    def connect(self, box_names: list[str] | None = None):
        """
        Connect to the boxes.

        Parameters
        ----------
        box_names : list[str], optional
            List of box names to connect to. If None, connect to all available boxes.
        """
        if box_names is None:
            box_names = self.available_boxes
        self._boxpool = self.qubecalib.create_boxpool(*box_names)

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
            print(f"Failed to linkup box {box_name}. Status: {status}")
        # return the box
        return box

    def linkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: int = 500,
    ) -> dict[str, Quel1Box]:
        """
        Linkup all the boxes in the list.

        Returns
        -------
        dict[str, Quel1Box]
            Dictionary of linked up boxes.
        """
        boxes = {}
        for box_name in box_list:
            try:
                boxes[box_name] = self.linkup(box_name, noise_threshold=noise_threshold)
                print(f"{box_name:5}", ":", "Linked up")
            except Exception as e:
                print(f"{box_name:5}", ":", "Error", e)
        return boxes

    def relinkup(self, box_name: str, noise_threshold: int = 500):
        """
        Relinkup a box.

        Parameters
        ----------
        box_name : str
            Name of the box to relinkup.
        """
        box = self.qubecalib.create_box(box_name, reconnect=False)
        box.relinkup(use_204b=False, background_noise_threshold=noise_threshold)
        box.reconnect()

    def relinkup_boxes(self, box_list: list[str], noise_threshold: int = 500):
        """
        Relinkup all the boxes in the list.
        """
        for box_name in box_list:
            self.relinkup(box_name, noise_threshold=noise_threshold)

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
        """
        if len(box_list) < 2:
            return True
        synchronized = self.resync_clocks(box_list)
        if not synchronized:
            print("Failed to synchronize clocks.")
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
        """
        self._check_box_availabilty(box_name)
        try:
            box = self.qubecalib.create_box(box_name, reconnect=False)
            box.reconnect()
            box_config = box.dump_box()
        except Exception as e:
            print(f"Failed to dump box {box_name}. Error: {e}")
            box_config = {}
        return box_config

    def dump_port(self, box_name: str, port_number: int) -> dict:
        """
        Dump the port configuration.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port_number : int
            Port number.

        Returns
        -------
        dict
            Dictionary of port configuration.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        self._check_box_availabilty(box_name)
        try:
            box = self.qubecalib.create_box(box_name, reconnect=False)
            box.reconnect()
            port_config = box.dump_port(port_number)
        except Exception as e:
            print(f"Failed to dump port {port_number} of box {box_name}. Error: {e}")
            port_config = {}
        return port_config

    def add_sequence(
        self,
        sequence: Sequence,
        *,
        interval: float,
        time_offset: dict[str, int] = {},  # {box_name: time_offset}
        time_to_start: dict[str, int] = {},  # {box_name: time_to_start}
    ):
        """
        Add a sequence to the queue.

        Parameters
        ----------
        sequence : Sequence
            The sequence to add to the queue.
        """
        self.qubecalib.add_sequence(
            sequence,
            interval=interval,
            time_offset=time_offset,
            time_to_start=time_to_start,
        )

    def add_sequencer(self, sequencer: Sequencer):
        """
        Add a sequencer to the queue.

        Parameters
        ----------
        sequencer : Sequencer
            The sequencer to add to the queue.
        """
        self.qubecalib._executor.add_command(sequencer)

    def show_command_queue(self):
        """Show the current command queue."""
        print(self.qubecalib.show_command_queue())

    def clear_command_queue(self):
        """Clear the command queue."""
        self.qubecalib.clear_command_queue()

    def execute(
        self,
        *,
        repeats: int,
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
        integral_mode : {"integral", "single"}, optional
            Integral mode.
        dsp_demodulation : bool, optional
            Enable DSP demodulation.
        software_demodulation : bool, optional
            Enable software demodulation.

        Yields
        ------
        RawResult
            Measurement result.
        """
        for status, data, config in self.qubecalib.step_execute(
            repeats=repeats,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
        ):
            result = RawResult(
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
        time_offset: dict[str, int] = {},  # {box_name: time_offset}
        time_to_start: dict[str, int] = {},  # {box_name: time_to_start}
    ) -> RawResult:
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
        RawResult
            Measurement result.
        """
        self.clear_command_queue()
        self.add_sequence(
            sequence,
            interval=interval,
            time_offset=time_offset,
            time_to_start=time_to_start,
        )
        return next(
            self.execute(
                repeats=repeats,
                integral_mode=integral_mode,
                dsp_demodulation=dsp_demodulation,
                software_demodulation=software_demodulation,
            )
        )

    def execute_sequencer(
        self,
        sequencer: Sequencer,
        *,
        repeats: int,
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
    ) -> RawResult:
        """
        Execute a single sequence and return the measurement result.

        Parameters
        ----------
        sequencer : Sequencer
            The sequencer to execute.
        repeats : int
            Number of repeats of the sequence.
        integral_mode : {"integral", "single"}, optional
            Integral mode.
        dsp_demodulation : bool, optional
            Enable DSP demodulation.
        software_demodulation : bool, optional
            Enable software demodulation.

        Returns
        -------
        RawResult
            Measurement result.
        """
        if self._boxpool is None:
            self.clear_command_queue()
            self.add_sequencer(sequencer)
            return next(
                self.execute(
                    repeats=repeats,
                    integral_mode=integral_mode,
                    dsp_demodulation=dsp_demodulation,
                    software_demodulation=software_demodulation,
                )
            )
        else:
            sequencer.set_measurement_option(
                repeats=repeats,
                interval=sequencer.interval,  # type: ignore
                integral_mode=integral_mode,
                dsp_demodulation=dsp_demodulation,
                software_demodulation=software_demodulation,
            )
            status, data, config = sequencer.execute(self.boxpool)
            return RawResult(
                status=status,
                data=data,
                config=config,
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
        """
        self.qubecalib.modify_target_frequency(target, frequency)

    def modify_target_frequencies(self, frequencies: dict[str, float]):
        """
        Modify the target frequencies.

        Parameters
        ----------
        frequencies : dict[str, float]
            Dictionary of target frequencies.
        """
        for target, frequency in frequencies.items():
            self.modify_target_frequency(target, frequency)

    def define_target(
        self,
        target_name: str,
        channel_name: str,
        target_frequency: float | None = None,
    ):
        """
        Define a target.

        Parameters
        ----------
        target_name : str
            Name of the target.
        channel_name : str
            Name of the channel.
        target_frequency : float, optional
            Frequency of the target in GHz.
        """
        self.qubecalib.define_target(
            target_name=target_name,
            channel_name=channel_name,
            target_frequency=target_frequency,
        )
