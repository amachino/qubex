from __future__ import annotations

import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from typing_extensions import Sequence

from .config_loader import ConfigLoader
from .control_system import CapPort, GenPort, PortType
from .device_controller import DeviceController
from .experiment_system import ExperimentSystem

console = Console()


@dataclass
class StateHash:
    experiment_system: int
    device_controller: int
    device_settings: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateHash):
            return NotImplemented
        return (
            self.experiment_system == other.experiment_system
            and self.device_controller == other.device_controller
            and self.device_settings == other.device_settings
        )


class SystemManager:
    """
    Singleton class to manage the system state.

    Attributes
    ----------
    _instance : SystemManager
        Shared instance of the SystemManager.
    _initialized : bool
        Whether the SystemManager is initialized.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def shared(cls):
        """
        Get the shared instance of the SystemManager.

        Returns
        -------

        SystemManager
            Shared instance of the SystemManager.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Initialize the SystemManager.

        Notes
        -----
        This class is a singleton. Use `SystemManager.shared()` to get the shared instance.
        """
        if self._initialized:
            return
        self._experiment_system = None
        self._device_controller = DeviceController()
        self._device_settings = {}
        self._cached_state = StateHash(0, 0, 0)
        self._rawdata_dir = None
        self._initialized = True

    @property
    def rawdata_dir(self) -> Path | None:
        """Get the directory for raw data."""
        return self._rawdata_dir

    @rawdata_dir.setter
    def rawdata_dir(self, value: Path | str | None):
        """
        Set the directory for raw data.

        Parameters
        ----------
        value : Path | str | None
            The directory path for raw data.
        """
        if value is None:
            self._rawdata_dir = None
        else:
            self._rawdata_dir = Path(value)
            self._rawdata_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_loader(self) -> ConfigLoader:
        """Get the configuration loader."""
        return self._config_loader

    @property
    def is_loaded(self) -> bool:
        """Check if the experiment system is loaded."""
        return self._experiment_system is not None

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        if self._experiment_system is None:
            raise ValueError("Experiment system is not loaded.")
        return self._experiment_system

    @property
    def device_controller(self) -> DeviceController:
        """Get the device controller."""
        return self._device_controller

    @property
    def device_settings(self) -> dict:
        """Get the device settings."""
        return self._device_settings or {}

    @property
    def state(self) -> StateHash:
        """Get the current state."""
        return StateHash(
            experiment_system=self.experiment_system.hash,
            device_controller=self.device_controller.hash,
            device_settings=hash(str(self.device_settings)),
        )

    @property
    def cached_state(self) -> StateHash:
        """Get the cached state."""
        return self._cached_state

    def set_experiment_system(self, experiment_system: ExperimentSystem):
        """
        Set the experiment system.

        Parameters
        ----------
        experiment_system : ExperimentSystem
            Experiment system to set.

        Notes
        -----
        This method also updates the device controller to reflect the new experiment system.
        """
        self._experiment_system = experiment_system
        # update device controller to reflect the new experiment system
        if self._mock_mode:
            print("Experiment system created in mock mode (device controller updates bypassed)")
            return
        self._update_device_controller(experiment_system)
        self.update_cache()

    def set_device_settings(self, device_settings: dict):
        """
        Set the device settings.

        Parameters
        ----------
        device_settings : dict
            Device settings to set.

        Notes
        -----
        This method also updates the experiment system to reflect the new device settings.
        """
        self._device_settings = device_settings
        # update experiment system to reflect the new device settings
        self._experiment_system = self._create_experiment_system(device_settings)
        self.update_cache()

    def update_cache(self):
        """Update the cached state."""
        self._cached_state = self.state

    def is_synced(
        self,
        *,
        box_ids: Sequence[str],
    ) -> bool:
        """
        Check if the state is synced.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to check.

        Returns
        -------
        bool
            Whether the state is synced.
        """
        if self.state != self.cached_state:
            # Provide explicit category and stacklevel so users can trace call site.
            warnings.warn(
                "The current state is different from the cached state. ",
                category=UserWarning,
                stacklevel=2,
            )
            return False
        device_settings = self._fetch_device_settings(box_ids=box_ids)
        if self.device_settings != device_settings:
            warnings.warn(
                "The current device settings are different from the fetched device settings. ",
                category=UserWarning,
                stacklevel=2,
            )
            return False
        return True

    def load(
        self,
        *,
        chip_id: str,
        config_dir: Path | str | None,
        params_dir: Path | str | None,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
        mock_mode: bool = False,
    ):
        """
        Load the experiment system and device controller.

        Parameters
        ----------
        chip_id : str
            Chip ID.
        config_dir : str, optional
            Configuration directory.
        params_dir : str, optional
            Parameters directory.
        targets_to_exclude : list[str], optional
            List of target labels to exclude, by default None.
        configuration_mode : Literal["ge-ef-cr", "ge-cr-cr"], optional
            Configuration mode, by default "ge-cr-cr".
        """
        self._config_loader = ConfigLoader(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            targets_to_exclude=targets_to_exclude,
            configuration_mode=configuration_mode,
        )
        self._mock_mode = mock_mode
        experiment_system = self.config_loader.get_experiment_system(chip_id)
        self.set_experiment_system(experiment_system)

    def load_skew_file(
        self,
        box_ids: list[str],  # deprecated
    ):
        skew_file_path = self.config_loader.config_path / "skew.yaml"
        if not Path(skew_file_path).exists():
            print(f"Skew file not found: {skew_file_path}")
        else:
            try:
                self.device_controller.qubecalib.sysdb.load_skew_yaml(
                    str(skew_file_path)
                )
            except Exception as e:
                print(f"Failed to load the skew file: {e}")

    def pull(
        self,
        box_ids: Sequence[str],
    ):
        """
        Pull the hardware state to the software state.

        This method updates the software state to reflect the hardware state.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to fetch the device settings for.
        """
        device_settings = self._fetch_device_settings(box_ids=box_ids)
        self.set_device_settings(device_settings)

    def push(
        self,
        box_ids: Sequence[str],
        confirm: bool = True,
    ):
        """
        Push the software state to the hardware state.

        This method updates the hardware state to reflect the software state.

        Parameters
        ----------
        box_ids : Sequence[str]
            Box IDs to configure.
        confirm : bool, optional
            Whether to confirm the operation, by default True.
        """
        boxes = [self.experiment_system.get_box(box_id) for box_id in box_ids]
        boxes_str = "\n".join([f"{box.id} ({box.name})" for box in boxes])

        if confirm:
            confirmed = Confirm.ask(
                f"""
You are going to configure the following boxes:

[bold bright_green]{boxes_str}[/bold bright_green]

This operation will overwrite the existing device settings. Do you want to continue?
"""
            )
            if not confirmed:
                print("Operation cancelled.")
                return

        for box in boxes:
            for port in box.ports:
                if isinstance(port, GenPort):
                    if port.type in (PortType.CTRL, PortType.READ_OUT, PortType.PUMP):
                        try:
                            self.device_controller.config_port(
                                box_name=box.id,
                                port=port.number,
                                lo_freq=port.lo_freq,
                                cnco_freq=port.cnco_freq,
                                vatt=port.vatt,
                                sideband=port.sideband,
                                fullscale_current=port.fullscale_current,
                                rfswitch=port.rfswitch,
                            )
                            for gen_channel in port.channels:
                                self.device_controller.config_channel(
                                    box_name=box.id,
                                    port=port.number,
                                    channel=gen_channel.number,
                                    fnco_freq=gen_channel.fnco_freq,
                                )
                        except Exception as e:
                            print(e, port.id)
                elif isinstance(port, CapPort):
                    if port.type in (PortType.READ_IN,):
                        try:
                            self.device_controller.config_port(
                                box_name=box.id,
                                port=port.number,
                                lo_freq=port.lo_freq,
                                cnco_freq=port.cnco_freq,
                                rfswitch=port.rfswitch,
                            )
                            for cap_channel in port.channels:
                                self.device_controller.config_runit(
                                    box_name=box.id,
                                    port=port.number,
                                    runit=cap_channel.number,
                                    fnco_freq=cap_channel.fnco_freq,
                                )
                        except Exception as e:
                            print(e, port.id)

        self._device_settings = self._fetch_device_settings(box_ids=box_ids)
        self.update_cache()

    def print_box_info(
        self,
        box_id: str,
        *,
        fetch: bool = False,
    ) -> None:
        """
        Print the information of a box.

        Parameters
        ----------
        box_id : str
            Box ID.
        fetch : bool, optional
            Whether to fetch the device settings, by default False.
        """
        if fetch:
            device_settings = self._fetch_device_settings([box_id])
            experiment_system = self._create_experiment_system(device_settings)
        else:
            experiment_system = self.experiment_system

        box_ids = [box.id for box in experiment_system.boxes]
        if box_id not in box_ids:
            print(f"Box {box_id} is not found.")
            return

        box = experiment_system.get_box(box_id)

        table1 = Table(
            show_header=True,
            header_style="bold",
            title=f"BOX INFO ({box.id})",
        )
        table2 = Table(
            show_header=True,
            header_style="bold",
        )
        table1.add_column("PORT", justify="center")
        table1.add_column("TYPE", justify="center")
        table1.add_column("SSB", justify="center")
        table1.add_column("LO", justify="right")
        table1.add_column("CNCO", justify="right")
        table1.add_column("VATT", justify="right")
        table1.add_column("FSC", justify="right")
        table2.add_column("PORT", justify="center")
        table2.add_column("TYPE", justify="center")
        table2.add_column("SSB", justify="center")
        table2.add_column("FNCO-0", justify="right")
        table2.add_column("FNCO-1", justify="right")
        table2.add_column("FNCO-2", justify="right")

        for port in box.ports:
            number = str(port.number)
            type = port.type.value
            if port.type == PortType.MNTR_OUT:
                continue
            if isinstance(port, CapPort):
                ssb = ""
                lo = f"{port.lo_freq:_}"
                cnco = f"{port.cnco_freq:_}"
                vatt = ""
                fsc = ""
            elif isinstance(port, GenPort):
                ssb = port.sideband if port.sideband is not None else ""
                lo = f"{port.lo_freq:_}" if port.lo_freq is not None else ""
                cnco = f"{port.cnco_freq:_}"
                vatt = str(port.vatt) if port.vatt is not None else ""
                fsc = str(port.fullscale_current)

            table1.add_row(
                number,
                type,
                ssb,
                lo,
                cnco,
                vatt,
                fsc,
            )
            if isinstance(port, GenPort):
                table2.add_row(
                    number,
                    type,
                    ssb,
                    *[f"{ch.fnco_freq:_}" for ch in port.channels],
                )
        console.print(table1)
        console.print(table2)

    def save_qubecalib_config(
        self,
        path_to_save: str = "./qubecalib.json",
    ):
        """
        Save the Qubecalib configuration to a file.

        Parameters
        ----------
        path_to_save : str, optional
            Path to save the Qubecalib configuration, by default "./qubecalib.json".
        """
        path = Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.device_controller.system_config_json)
        print(f"Qubecalib configuration saved to {path}.")

    def _fetch_device_settings(
        self,
        box_ids: Sequence[str],
    ):
        boxes = [self.experiment_system.get_box(box_id) for box_id in box_ids]
        result: dict = {}
        for box in boxes:
            # TODO: run this in a separate thread
            box_config = self.device_controller.dump_box(box.id)
            self.device_controller.boxpool._box_config_cache[box.id] = box_config
            result[box.id] = {"ports": {}}
            for port in box.ports:
                if port.type not in (PortType.NOT_AVAILABLE, PortType.MNTR_OUT):
                    try:
                        result[box.id]["ports"][port.number] = box_config["ports"][
                            port.number
                        ]
                    except Exception as e:
                        print(e)
        return result

    def _update_device_controller(
        self,
        experiment_system: ExperimentSystem,
    ):
        control_system = experiment_system.control_system
        control_params = experiment_system.control_params

        qc = self.device_controller.qubecalib

        qc.define_clockmaster(
            ipaddr=control_system.clock_master_address,
            reset=True,  # this option has no effect and will be removed in the future
        )

        for box in control_system.boxes:
            qc.define_box(
                box_name=box.id,
                ipaddr_wss=box.address,
                boxtype=box.type.value,
            )

            for port in box.ports:
                if port.type == PortType.NOT_AVAILABLE:
                    continue
                qc.define_port(
                    port_name=port.id,
                    box_name=box.id,
                    port_number=port.number,  # type: ignore
                )

                for channel in port.channels:
                    if port.type == PortType.READ_IN:
                        mux = experiment_system.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        ndelay_or_nwait = control_params.get_capture_delay(mux.index)
                    elif port.type == PortType.MNTR_IN:
                        ndelay_or_nwait = 7  # TODO: make this configurable
                    else:
                        ndelay_or_nwait = 0
                    qc.define_channel(
                        channel_name=channel.id,
                        port_name=port.id,
                        channel_number=channel.number,
                        ndelay_or_nwait=ndelay_or_nwait,
                    )

                if port.type in (
                    PortType.PUMP,
                    PortType.MNTR_OUT,
                    PortType.MNTR_IN,
                ):
                    rel = (port.channels[0].id, port.id)
                    if rel not in qc.sysdb._relation_channel_target:
                        qc.sysdb._relation_channel_target.append(rel)

        for target in experiment_system.all_targets:
            qc.define_target(
                target_name=target.label,
                channel_name=target.channel.id,
                target_frequency=target.frequency,
            )

        # reset the cache
        qc.clear_command_queue()
        self.device_controller.clear_cache()

    def _create_experiment_system(
        self,
        device_settings: dict,
    ) -> ExperimentSystem:
        experiment_system = deepcopy(self.experiment_system)
        control_system = experiment_system.control_system
        for box_id, box in device_settings.items():
            for port_number, port in box["ports"].items():
                direction = port["direction"]
                lo_freq = port.get("lo_freq")
                lo_freq = int(lo_freq) if lo_freq is not None else None
                cnco_freq = int(port["cnco_freq"])
                if direction == "out":
                    sideband = port.get("sideband")
                    fullscale_current = int(port["fullscale_current"])
                    fnco_freqs = [
                        int(channel["fnco_freq"])
                        for channel in port["channels"].values()
                    ]
                elif direction == "in":
                    sideband = None
                    fullscale_current = None
                    fnco_freqs = [
                        int(channel["fnco_freq"]) for channel in port["runits"].values()
                    ]
                control_system.set_port_params(
                    box_id=box_id,
                    port_number=port_number,
                    sideband=sideband,
                    lo_freq=lo_freq,
                    cnco_freq=cnco_freq,
                    fnco_freqs=fnco_freqs,
                    fullscale_current=fullscale_current,
                )
        return experiment_system

    @contextmanager
    def modified_frequencies(self, target_frequencies: dict[str, float]):
        """
        Temporarily modify the target frequencies.

        Parameters
        ----------
        target_frequencies : dict[str, float]
            The target frequencies to be modified.
        """
        original_frequencies = {
            target.label: target.frequency
            for target in self.experiment_system.targets
            if target.label in target_frequencies
        }
        self.experiment_system.modify_target_frequencies(target_frequencies)
        self.device_controller.modify_target_frequencies(target_frequencies)
        try:
            yield
        finally:
            self.experiment_system.modify_target_frequencies(original_frequencies)
            self.device_controller.modify_target_frequencies(original_frequencies)

    @contextmanager
    def modified_device_settings(
        self,
        label: str,
        *,
        lo_freq: int | None,
        cnco_freq: int,
        fnco_freq: int,
    ):
        """
        Temporarily modify the device settings.

        Parameters
        ----------
        device_settings : dict[str, Any]
            The device settings to be modified.

        Examples
        --------
        >>> with system_manager.modified_device_settings(
        ...     "Q00",
        ...     lo_freq=10_000_000_000,
        ...     cnco_freq=1_500,
        ...     fnco_freq=750,
        ... ):
        ...     ...
        """
        target = self.experiment_system.get_target(label)
        channel = target.channel
        port = channel.port
        box_cache = self.device_controller.boxpool._box_config_cache

        original_lo_freq = port.lo_freq
        original_cnco_freq = port.cnco_freq
        original_fnco_freq = channel.fnco_freq
        original_box_cache = deepcopy(box_cache)

        self.device_controller.config_port(
            box_name=port.box_id,
            port=port.number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
        )
        self.device_controller.config_channel(
            box_name=port.box_id,
            port=port.number,
            channel=channel.number,
            fnco_freq=fnco_freq,
        )
        port_cache = box_cache[port.box_id]["ports"][port.number]
        port_cache["lo_freq"] = lo_freq
        port_cache["cnco_freq"] = cnco_freq
        port_cache["channels"][channel.number]["fnco_freq"] = fnco_freq
        self.device_controller.initialize_awg_and_capunits(port.box_id)

        if target.is_read:
            cap_channel = self.experiment_system.get_cap_target(label).channel
            cap_port = cap_channel.port
            self.device_controller.config_port(
                box_name=cap_port.box_id,
                port=cap_port.number,
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
            )
            self.device_controller.config_runit(
                box_name=cap_port.box_id,
                port=cap_port.number,
                runit=cap_channel.number,
                fnco_freq=fnco_freq,
            )
            cap_port_cache = box_cache[cap_port.box_id]["ports"][cap_port.number]
            cap_port_cache["lo_freq"] = lo_freq
            cap_port_cache["cnco_freq"] = cnco_freq
            cap_port_cache["runits"][cap_channel.number]["fnco_freq"] = fnco_freq
            self.device_controller.initialize_awg_and_capunits(cap_port.box_id)

        self.experiment_system.update_port_params(
            label,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            fnco_freq=fnco_freq,
        )
        try:
            yield
        finally:
            # restore the original device settings
            self.experiment_system.update_port_params(
                label,
                lo_freq=original_lo_freq,
                cnco_freq=original_cnco_freq,
                fnco_freq=original_fnco_freq,
            )
            self.device_controller.config_port(
                box_name=port.box_id,
                port=port.number,
                lo_freq=original_lo_freq,
                cnco_freq=original_cnco_freq,
            )
            self.device_controller.config_channel(
                box_name=port.box_id,
                port=port.number,
                channel=channel.number,
                fnco_freq=original_fnco_freq,
            )
            if target.is_read:
                self.device_controller.config_port(
                    box_name=cap_port.box_id,
                    port=cap_port.number,
                    lo_freq=original_lo_freq,
                    cnco_freq=original_cnco_freq,
                )
                self.device_controller.config_runit(
                    box_name=cap_port.box_id,
                    port=cap_port.number,
                    runit=cap_channel.number,
                    fnco_freq=original_fnco_freq,
                )

            # restore the original box config
            self.device_controller.boxpool._box_config_cache = original_box_cache

    @contextmanager
    def save_rawdata(
        self,
        *,
        rawdata_dir: Path | str = ".rawdata",
        tag: str | None = None,
    ):
        """
        Context manager to save raw data to a specified directory.

        Parameters
        ----------
        rawdata_dir : Path | str | None, optional
            Directory to save raw data.
        tag : str | None, optional
            Tag to append to the raw data file name, by default None.
        """
        original_rawdata_dir = self.rawdata_dir
        rawdata_dir = Path(rawdata_dir)
        if tag is not None:
            rawdata_dir = rawdata_dir / tag
        self.rawdata_dir = rawdata_dir
        try:
            yield
        finally:
            self.rawdata_dir = original_rawdata_dir
