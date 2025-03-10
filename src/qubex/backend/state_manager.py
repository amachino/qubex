from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from typing_extensions import Sequence

from .config_loader import DEFAULT_CONFIG_DIR, DEFAULT_PARAMS_DIR, ConfigLoader
from .control_system import CapPort, GenPort, PortType
from .device_controller import DeviceController
from .experiment_system import ExperimentSystem

console = Console()


@dataclass
class State:
    system: int
    controller: int
    device: int


class StateManager:
    """
    Singleton class that manages the state of the experiment system and the device controller.

    Attributes
    ----------
    _instance : StateManager
        Shared instance of the StateManager.
    _initialized : bool
        Whether the StateManager is initialized.
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
        Get the shared instance of the StateManager.

        Returns
        -------

        StateManager
            Shared instance of the StateManager.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Initialize the StateManager.

        Notes
        -----
        This class is a singleton. Use `StateManager.shared()` to get the shared instance.
        """
        if self._initialized:
            return
        self._experiment_system = None
        self._device_controller = DeviceController()
        self._device_settings = {}
        self._cached_state = State(0, 0, 0)
        self._initialized = True

    @property
    def config_loader(self) -> ConfigLoader:
        """Get the configuration loader."""
        return self._config_loader

    @property
    def is_loaded(self) -> bool:
        """Check if the experiment system is loaded."""
        return self.experiment_system is not None

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        if self._experiment_system is None:
            raise ValueError("Experiment system is not loaded.")
        return self._experiment_system

    @experiment_system.setter
    def experiment_system(self, experiment_system: ExperimentSystem):
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
        self._update_device_controller(experiment_system)
        self.update_cache()

    @property
    def device_controller(self) -> DeviceController:
        """Get the device controller."""
        return self._device_controller

    @device_controller.setter
    def device_controller(self, device_controller: DeviceController):
        """
        Set the device controller.

        Parameters
        ----------
        device_controller : DeviceController
            Device controller to set.
        """
        self._device_controller = device_controller
        self.update_cache()

    @property
    def device_settings(self) -> dict:
        """Get the device settings."""
        return self._device_settings or {}

    @device_settings.setter
    def device_settings(self, device_settings: dict):
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

    @property
    def state(self) -> State:
        """Get the current state."""
        return State(
            system=self.experiment_system.hash,
            controller=self.device_controller.hash,
            device=hash(str(self.device_settings)),
        )

    @property
    def cached_state(self) -> State:
        """Get the cached state."""
        return self._cached_state

    def update_cache(self):
        """Update the cached state."""
        self._cached_state = self.state

    def is_synced(
        self,
        *,
        box_ids: Sequence[str] | None = None,
    ) -> bool:
        """
        Check if the state is synced.

        Parameters
        ----------
        box_ids : Sequence[str], optional

        Returns
        -------
        bool
            Whether the state is synced.
        """
        if self.state != self.cached_state:
            # print("Local state is changed.")
            return False
        device_settings = self._fetch_device_settings(box_ids=box_ids)
        if self.device_settings != device_settings:
            # print("Remote state is different.")
            return False
        return True

    def load(
        self,
        *,
        chip_id: str,
        config_dir: str = DEFAULT_CONFIG_DIR,
        params_dir: str = DEFAULT_PARAMS_DIR,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
    ):
        """
        Load the experiment system and the device controller.

        Parameters
        ----------
        chip_id : str
            Chip ID.
        config_dir : str, optional
            Configuration directory, by default DEFAULT_CONFIG_DIR.
        params_dir : str, optional
            Parameters directory, by default DEFAULT_PARAMS_DIR.
        """
        self._config_loader = ConfigLoader(
            config_dir=config_dir,
            params_dir=params_dir,
            targets_to_exclude=targets_to_exclude,
            configuration_mode=configuration_mode,
        )
        self.experiment_system = self.config_loader.get_experiment_system(chip_id)

    def pull(
        self,
        box_ids: Sequence[str] | None = None,
    ):
        """
        Pull the hardware state to the software state.

        This method updates the software state to reflect the hardware state.

        Parameters
        ----------
        box_ids : Sequence[str], optional
            Box IDs to fetch, by default None.
        """
        boxes = self.experiment_system.boxes
        if box_ids is not None:
            boxes = [box for box in boxes if box.id in box_ids]

        device_settings = self._fetch_device_settings(box_ids=box_ids)
        self.device_settings = device_settings

    def push(
        self,
        box_ids: Sequence[str] | None = None,
        confirm: bool = True,
    ):
        """
        Push the software state to the hardware state.

        This method updates the hardware state to reflect the software state.

        Parameters
        ----------
        box_ids : Sequence[str], optional
            Box IDs to configure, by default None.
        """
        boxes = self.experiment_system.boxes
        if box_ids is not None:
            boxes = [box for box in boxes if box.id in box_ids]

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

        qc = self.device_controller.qubecalib

        for box in boxes:
            quel1_box = qc.create_box(box.id, reconnect=False)
            quel1_box.reconnect()
            for port in box.ports:
                if isinstance(port, GenPort):
                    if port.type in (PortType.CTRL, PortType.READ_OUT, PortType.PUMP):
                        try:
                            quel1_box.config_port(
                                port=port.number,
                                lo_freq=port.lo_freq,
                                cnco_freq=port.cnco_freq,
                                vatt=port.vatt,
                                sideband=port.sideband,
                                fullscale_current=port.fullscale_current,
                                rfswitch=port.rfswitch,
                            )
                            for gen_channel in port.channels:
                                quel1_box.config_channel(
                                    port=port.number,
                                    channel=gen_channel.number,
                                    fnco_freq=gen_channel.fnco_freq,
                                )
                        except Exception as e:
                            print(e, port.id)
                elif isinstance(port, CapPort):
                    if port.type in (PortType.READ_IN,):
                        try:
                            quel1_box.config_port(
                                port=port.number,
                                lo_freq=port.lo_freq,
                                cnco_freq=port.cnco_freq,
                                rfswitch=port.rfswitch,
                            )
                            for cap_channel in port.channels:
                                quel1_box.config_runit(
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
            if isinstance(port, CapPort):
                ssb = ""
                lo = f"{port.lo_freq:_}"
                cnco = f"{port.cnco_freq:_}"
                vatt = ""
                fsc = ""
            elif isinstance(port, GenPort):
                ssb = port.sideband
                lo = f"{port.lo_freq:_}"
                cnco = f"{port.cnco_freq:_}"
                vatt = str(port.vatt)
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
        box_ids: Sequence[str] | None = None,
    ):
        boxes = self.experiment_system.boxes
        if box_ids is not None:
            boxes = [box for box in boxes if box.id in box_ids]
        result: dict = {}
        for box in boxes:
            result[box.id] = {"ports": {}}
            for port in box.ports:
                if port.type not in (PortType.NOT_AVAILABLE, PortType.MNTR_OUT):
                    try:
                        result[box.id]["ports"][port.number] = (
                            self.device_controller.dump_port(box.id, port.number)
                        )
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
                    port_number=port.number,
                )

                for channel in port.channels:
                    if port.type == PortType.READ_IN:
                        mux = experiment_system.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        ndelay_or_nwait = control_params.capture_delay[mux.index]
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
                    qc.sysdb._relation_channel_target.append(
                        (port.channels[0].id, port.id),
                    )

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
                lo_freq = int(port["lo_freq"])
                cnco_freq = int(port["cnco_freq"])
                if direction == "out":
                    sideband = port["sideband"]
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
        lo_freq: int,
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
        >>> with state_manager.modified_device_settings(
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

        original_lo_freq = port.lo_freq
        original_cnco_freq = port.cnco_freq
        original_fnco_freq = channel.fnco_freq

        qc = self.device_controller.qubecalib
        quel1_box = qc.create_box(port.box_id, reconnect=False)
        quel1_box.reconnect()

        quel1_box.config_port(
            port=port.number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
        )
        quel1_box.config_channel(
            port=port.number,
            channel=channel.number,
            fnco_freq=fnco_freq,
        )
        if target.is_read:
            cap_channel = self.experiment_system.get_cap_target(label).channel
            cap_port = cap_channel.port
            quel1_box.config_port(
                port=cap_port.number,
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
            )
            quel1_box.config_runit(
                port=cap_port.number,
                runit=cap_channel.number,
                fnco_freq=fnco_freq,
            )

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
            quel1_box.config_port(
                port=port.number,
                lo_freq=original_lo_freq,
                cnco_freq=original_cnco_freq,
            )
            quel1_box.config_channel(
                port=port.number,
                channel=channel.number,
                fnco_freq=original_fnco_freq,
            )
            if target.is_read:
                quel1_box.config_port(
                    port=cap_port.number,
                    lo_freq=original_lo_freq,
                    cnco_freq=original_cnco_freq,
                )
                quel1_box.config_runit(
                    port=cap_port.number,
                    runit=cap_channel.number,
                    fnco_freq=original_fnco_freq,
                )

            # clear the cache
            self.device_controller.clear_cache()
