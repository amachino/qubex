from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Final, Literal

from typing_extensions import deprecated

logger = logging.getLogger(__name__)

try:
    from qubecalib import QubeCalib, Sequencer
    from qubecalib.instrument.quel.quel1 import Quel1System
    from qubecalib.instrument.quel.quel1.driver import (
        Action,
        AwgId,
        AwgSetting,
        RunitId,
        RunitSetting,
        TriggerSetting,
    )
    from qubecalib.instrument.quel.quel1.tool import Skew
    from qubecalib.neopulse import (
        DEFAULT_SAMPLING_PERIOD,
        CapSampledSequence,
        GenSampledSequence,
        Sequence,
    )
    from qubecalib.qubecalib import (
        BoxPool,
        CaptureParamTools,
        Converter,
        WaveSequenceTools,
    )
    from quel_clock_master import QuBEMasterClient
    from quel_ic_config import Quel1Box, Quel1ConfigOption
except ImportError as e:
    logger.info(e)


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
        try:
            if config_path is None:
                self._qubecalib = QubeCalib()
            else:
                try:
                    self._qubecalib = QubeCalib(str(config_path))
                except FileNotFoundError:
                    print(f"Configuration file {config_path} not found.")
                    raise
        except Exception:
            self._qubecalib = None
        self._cap_resource_map: dict | None = None
        self._gen_resource_map: dict | None = None
        self._boxpool: BoxPool | None = None
        self._quel1system: Quel1System | None = None

    @property
    def qubecalib(self) -> QubeCalib:
        if self._qubecalib is None:
            raise ModuleNotFoundError(name="qubecalib")
        return self._qubecalib

    @property
    def box_config(self) -> dict[str, Any]:
        """Get the box configuration."""
        if self._boxpool is None:
            box_config = {}
        else:
            box_config = self._boxpool._box_config_cache
        return box_config

    @property
    def system_config(self) -> dict[str, Any]:
        """Get the system configuration."""
        config = self.qubecalib.system_config_database.asdict()
        return config

    @property
    def system_config_json(self) -> str:
        """Get the system configuration as JSON."""
        config = self.qubecalib.system_config_database.asjson()
        return config

    @property
    def box_settings(self) -> dict[str, Any]:
        """Get the box settings."""
        return self.system_config["box_settings"]

    @property
    def port_settings(self) -> dict[str, Any]:
        """Get the port settings."""
        return self.system_config["port_settings"]

    @property
    def target_settings(self) -> dict[str, Any]:
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
    def boxpool(self) -> BoxPool:
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
    def quel1system(self) -> Quel1System:
        """
        Get the Quel1 system.

        Returns
        -------
        Quel1System
            The Quel1 system.
        """
        if self._quel1system is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        return self._quel1system

    @property
    def cap_resource_map(self) -> dict[str, dict]:
        """
        Get the cap resource map.

        Returns
        -------
        dict[str, dict]
            The cap resource map.
        """
        if self._cap_resource_map is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        return self._cap_resource_map

    @property
    def gen_resource_map(self) -> dict[str, dict]:
        """
        Get the gen resource map.

        Returns
        -------
        dict[str, dict]
            The gen resource map.
        """
        if self._gen_resource_map is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        return self._gen_resource_map

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

    def get_cap_resource_map(self, targets: Collection[str]) -> dict[str, dict]:
        """
        Get the resource map for the given targets.

        Parameters
        ----------
        targets : Collection[str]
            List of target names.
        """
        return {
            target: self.cap_resource_map[target]
            for target in targets
            if target in self.cap_resource_map
        }

    def get_gen_resource_map(self, targets: Collection[str]) -> dict[str, dict]:
        """
        Get the resource map for the given targets.

        Parameters
        ----------
        targets : Collection[str]
            List of target names.
        """
        return {
            target: self.gen_resource_map[target]
            for target in targets
            if target in self.gen_resource_map
        }

    def create_resource_map(
        self,
        type: Literal["cap", "gen"],
    ) -> dict[str, dict]:
        if self._boxpool is None:
            raise ValueError("Boxes not connected. Call connect() method first.")
        db = self.qubecalib.system_config_database
        result = {}
        for target in db._target_settings:
            channels = db.get_channels_by_target(target)
            bpc_list = [db.get_channel(channel) for channel in channels]
            for box_name, port_name, channel_number in bpc_list:
                if box_name not in self._boxpool._boxes:
                    continue
                box = self.get_box(box_name, reconnect=False)
                port_setting = db._port_settings[port_name]
                if (
                    type == "cap"
                    and port_setting.port in box.get_input_ports()
                    or type == "gen"
                    and port_setting.port in box.get_output_ports()
                ):
                    result[target] = {
                        "box": db._box_settings[box_name],
                        "port": db._port_settings[port_name],
                        "channel_number": channel_number,
                        "target": db._target_settings[target],
                    }
        return result

    def clear_cache(self):
        if self._boxpool is not None:
            self._boxpool._box_config_cache.clear()

    @deprecated("Use qubecalib.sysdb.load_skew_yaml instead.")
    def load_skew_file(self, box_list: list[str], file_path: str | Path):
        if len(box_list) == 0:
            return
        clockmaster_setting = self.qubecalib.sysdb._clockmaster_setting
        if clockmaster_setting is None:
            raise ValueError("Clockmaster setting not found in system configuration.")
        system = Quel1System.create(
            clockmaster=QuBEMasterClient(str(clockmaster_setting.ipaddr)),
            boxes=[self.get_box(box_name, reconnect=False) for box_name in box_list],  # type: ignore
        )
        skew = Skew(system, qubecalib=self.qubecalib)
        skew.load(str(file_path))

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

    def connect(self, box_names: str | list[str] | None = None):
        """
        Connect to the boxes.

        Parameters
        ----------
        box_names : str | list[str], optional
            List of box names to connect to. If None, connect to all available boxes.
        """
        if box_names is None:
            box_names = self.available_boxes
        if isinstance(box_names, str):
            box_names = [box_names]
        self._boxpool = self.qubecalib.create_boxpool(*box_names)
        self._quel1system = self.qubecalib.sysdb.create_quel1system(*box_names)
        self._cap_resource_map = self.create_resource_map("cap")
        self._gen_resource_map = self.create_resource_map("gen")

    def get_box(self, box_name: str, reconnect: bool = True) -> Quel1Box:
        """
        Get the box object.

        Parameters
        ----------
        box_name : str
            Name of the box.

        Returns
        -------
        Quel1Box
            The box object.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        self._check_box_availabilty(box_name)
        if self._boxpool is None or box_name not in self._boxpool._boxes:
            box = self.qubecalib.create_box(box_name, reconnect=reconnect)
        else:
            box = self._boxpool._boxes[box_name][0]
        return box

    def initialize_awg_and_capunits(
        self,
        box_names: str | Collection[str],
    ):
        """
        Initialize all awg and capture units in the specified boxes.

        Parameters
        ----------
        box_names : str | list[str]
            List of box names to initialize.
        """
        if isinstance(box_names, str):
            box_names = [box_names]
        for box_name in box_names:
            self._check_box_availabilty(box_name)
            box = self.get_box(box_name, reconnect=False)
            box.initialize_all_awgs()
            box.initialize_all_capunits()

    def linkup(
        self,
        box_name: str,
        noise_threshold: int | None = None,
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

        if noise_threshold is None:
            # TODO: use appropriate noise threshold
            noise_threshold = 10000

        if not all(box.link_status().values()):
            if box.boxtype == "quel1se-riken8":
                config_options = [Quel1ConfigOption.SE8_MXFE1_AWG2222]
            else:
                config_options = None
            box.relinkup(
                use_204b=False,
                background_noise_threshold=noise_threshold,
                config_options=config_options,
            )
        box.reconnect(background_noise_threshold=noise_threshold)

        # check if all links are up
        status = box.link_status()
        if not all(status.values()):
            print(f"Failed to linkup box {box_name}. Status: {status}")
        # return the box
        return box

    def linkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: int | None = None,
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

    def relinkup(self, box_name: str, noise_threshold: int | None = None):
        """
        Relinkup a box.

        Parameters
        ----------
        box_name : str
            Name of the box to relinkup.
        """
        if noise_threshold is None:
            noise_threshold = 10000
        box = self.qubecalib.create_box(box_name, reconnect=False)
        if box.boxtype == "quel1se-riken8":
            config_options = [Quel1ConfigOption.SE8_MXFE1_AWG2222]
        else:
            config_options = None
        box.relinkup(
            use_204b=False,
            background_noise_threshold=noise_threshold,
            config_options=config_options,
        )
        # TODO: use appropriate noise threshold
        box.reconnect(background_noise_threshold=10000)

    def relinkup_boxes(self, box_list: list[str], noise_threshold: int | None = None):
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
        if len(box_list) < 2:
            # NOTE: clockmaster will crash if there is only one box
            return True
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
        try:
            box = self.get_box(box_name)
            box_config = box.dump_box()
        except Exception as e:
            print(f"Failed to dump box {box_name}. Error: {e}")
            box_config = {}
        return box_config

    def dump_port(self, box_name: str, port_number: int | tuple[int, int]) -> dict:
        """
        Dump the port configuration.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port_number : int | tuple[int, int]
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
        try:
            box = self.get_box(box_name)
            port_config = box.dump_port(port_number)
        except Exception as e:
            print(f"Failed to dump port {port_number} of box {box_name}. Error: {e}")
            port_config = {}
        return port_config

    def config_port(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        lo_freq: float | None = None,
        cnco_freq: float | None = None,
        vatt: int | None = None,
        sideband: str | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
    ):
        """
        Configure the port of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        lo_freq : float | None, optional
            Local oscillator frequency in GHz.
        cnco_freq : float | None, optional
            CNCO frequency in GHz.
        vatt : int | None, optional
            VATT value.
        sideband : str | None, optional
            Sideband value.
        fullscale_current : int | None, optional
            Fullscale current value.
        rfswitch : str | None, optional
            RF switch value.
        """
        box = self.get_box(box_name)
        if box.boxtype == "quel1se-riken8":
            vatt = None
            sideband = None
        if box.boxtype == "quel1se-riken8" and port not in box.get_input_ports():
            lo_freq = None
        box.config_port(
            port=port,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )

    def config_channel(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        channel: int,
        fnco_freq: float | None = None,
    ):
        """
        Configure the channel of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        channel : int
            Channel number.
        fnco_freq : float | None, optional
            FNCO frequency in GHz.
        """
        box = self.get_box(box_name)
        box.config_channel(
            port=port,
            channel=channel,
            fnco_freq=fnco_freq,
        )

    def config_runit(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        runit: int,
        fnco_freq: float | None = None,
    ):
        """
        Configure the runit of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        runit : int
            Runit number.
        fnco_freq : float | None, optional
            FNCO frequency in GHz.
        """
        box = self.get_box(box_name)
        box.config_runit(
            port=port,
            runit=runit,
            fnco_freq=fnco_freq,
        )

    @deprecated("Use add_sequencer instead.")
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

    def execute_sequencer(
        self,
        sequencer: Sequencer,
        *,
        repeats: int,
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
        enable_sum: bool = False,
        enable_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
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
        if line_param0 is None:
            line_param0 = (1, 0, 0)
        if line_param1 is None:
            line_param1 = (0, 1, 0)
        sequencer.set_measurement_option(
            repeats=repeats,
            interval=sequencer.interval,  # type: ignore
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
            enable_sum=enable_sum,
            enable_classification=enable_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
        status, data, config = sequencer.execute(self.boxpool)
        return RawResult(
            status=status,
            data=data,
            config=config,
        )

    def _execute_sequencer(
        self,
        sequencer: Sequencer,
        *,
        repeats: int | None = None,
        interval_samples: int | None = None,
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        capture_delay_words: int | None = None,
        wait_words: int = 0,
    ) -> RawResult:
        # TODO: support skew adjustment

        if repeats is None:
            repeats = 1024
        if interval_samples is None:
            if sequencer.interval is None:
                raise ValueError("Interval is not set.")
            else:
                if sequencer.interval % DEFAULT_SAMPLING_PERIOD != 0:
                    raise ValueError(
                        f"Interval {sequencer.interval} is not a multiple of {DEFAULT_SAMPLING_PERIOD}"
                    )
                interval_samples = int(sequencer.interval / DEFAULT_SAMPLING_PERIOD)

        if capture_delay_words is None:
            capture_delay_words = 7 * 16

        settings: list[RunitSetting | AwgSetting | TriggerSetting] = []

        # capture settings
        cap_sequences_map = defaultdict(dict[str, CapSampledSequence])
        for cap_label, cap_sequence in sequencer.cap_sampled_sequence.items():
            cap_resource = self.cap_resource_map[cap_label]
            cap_id = (
                cap_resource["box"].box_name,
                cap_resource["port"].port,
                cap_resource["channel_number"],
            )
            cap_sequences_map[cap_id][cap_label] = cap_sequence

        for cap_id, cap_sequences in cap_sequences_map.items():
            if len(cap_sequences) > 1:
                raise ValueError(
                    f"Duplicate capture ID found: {cap_id}\n{cap_sequences}"
                )
            cap_sequence = next(iter(cap_sequences.values()))
            cap_param = CaptureParamTools.create(
                sequence=cap_sequence,
                capture_delay_words=capture_delay_words,
                repeats=repeats,
                interval_samples=interval_samples,
            )
            if integral_mode == "integral":
                CaptureParamTools.enable_integration(
                    capprm=cap_param,
                )
            if dsp_demodulation:
                CaptureParamTools.enable_demodulation(
                    capprm=cap_param,
                    f_GHz=cap_sequence.modulation_frequency or 0,
                )
            settings.append(
                RunitSetting(
                    runit=RunitId(
                        box=cap_id[0],
                        port=cap_id[1],
                        runit=cap_id[2],
                    ),
                    cprm=cap_param,
                )
            )

        # awg settings
        gen_sequences_map = defaultdict(dict[str, GenSampledSequence])
        for gen_label, gen_sequence in sequencer.gen_sampled_sequence.items():
            gen_resource = self.gen_resource_map[gen_label]
            gen_id = (
                gen_resource["box"].box_name,
                gen_resource["port"].port,
                gen_resource["channel_number"],
            )
            gen_sequences_map[gen_id][gen_label] = gen_sequence

        for gen_id, gen_sequences in gen_sequences_map.items():
            muxed_sequence = Converter.multiplex(
                sequences=gen_sequences,
                modfreqs={
                    label: gen_sequence.modulation_frequency or 0
                    for label, gen_sequence in gen_sequences.items()
                },
            )
            wave_seq = WaveSequenceTools.create(
                sequence=muxed_sequence,
                wait_words=wait_words,
                repeats=repeats,
                interval_samples=interval_samples,
            )
            settings.append(
                AwgSetting(
                    awg=AwgId(
                        box=gen_id[0],
                        port=gen_id[1],
                        channel=gen_id[2],
                    ),
                    wseq=wave_seq,
                )
            )

        # trigger settings
        settings += sequencer.select_trigger(self.quel1system, settings)

        if len(settings) == 0:
            raise ValueError("no settings")

        # execute
        action = Action.build(system=self.quel1system, settings=settings)
        status, results = action.action()
        status, data, config = sequencer.parse_capture_results(
            status=status,
            results=results,
            action=action,
            crmap=self.get_cap_resource_map(sequencer.cap_sampled_sequence.keys()),
        )

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
