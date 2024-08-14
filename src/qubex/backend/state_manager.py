from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from typing_extensions import Sequence, deprecated

from .config_loader import CONFIG_DIR, ConfigLoader
from .control_system import CapPort, GenPort, PortType
from .device_controller import DeviceController
from .experiment_system import ExperimentSystem

console = Console()


class StateManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @deprecated("Use StateManager.shared() instead.")
    def __init__(self):
        if self._initialized:
            return
        self._experiment_system = None
        self._device_controller = DeviceController()
        self._device_settings = {}
        self._initialized = True

    @property
    def experiment_system(self) -> ExperimentSystem:
        if self._experiment_system is None:
            raise ValueError("Experiment system is not loaded.")
        return self._experiment_system

    @experiment_system.setter
    def experiment_system(self, experiment_system: ExperimentSystem):
        self._experiment_system = experiment_system
        # update device controller to reflect the new experiment system
        self._device_controller = self._create_device_controller(experiment_system)

    @property
    def device_controller(self) -> DeviceController:
        return self._device_controller

    @device_controller.setter
    def device_controller(self, device_controller: DeviceController):
        self._device_controller = device_controller

    @property
    def device_settings(self) -> dict:
        return self._device_settings or {}

    @device_settings.setter
    def device_settings(self, device_settings: dict):
        self._device_settings = device_settings
        # update experiment system to reflect the new device settings
        self._experiment_system = self._create_experiment_system(device_settings)

    @property
    def system_state(self) -> int:
        return self.experiment_system.state.hash

    @property
    def controller_state(self) -> int:
        return self.device_controller.hash

    @property
    def device_state(self) -> int:
        return hash(str(self.device_settings))

    def load(
        self,
        *,
        chip_id: str,
        config_dir: str = CONFIG_DIR,
    ):
        config = ConfigLoader(config_dir)
        self.experiment_system = config.get_experiment_system(chip_id)

    def pull_state(self):
        device_settings = {
            box.id: self.device_controller.dump_box(box.id)
            for box in self.experiment_system.boxes
        }
        self.device_settings = device_settings

    def push_state(
        self,
        box_ids: Sequence[str] | None = None,
        *,
        exclude: Sequence[str] | None = None,
    ):
        boxes = self.experiment_system.boxes

        if box_ids is not None:
            boxes = [box for box in boxes if box.id in box_ids]

        if exclude is not None:
            boxes = [box for box in boxes if box.id not in exclude]

        boxes_str = "\n".join([f"{box.id} ({box.name})" for box in boxes])
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
                    if port.type in (PortType.CTRL, PortType.READ_OUT):
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
        # self.pull_state()

    def print_box_info(
        self,
        box_id: str,
        *,
        fetch: bool = False,
    ) -> None:
        if fetch:
            device_settings = {
                box.id: self.device_controller.dump_box(box.id)
                for box in self.experiment_system.boxes
            }
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
        path = Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.device_controller.system_config_json)
        print(f"Qubecalib configuration saved to {path}.")

    def _create_device_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> DeviceController:
        control_system = experiment_system.control_system
        control_params = experiment_system.control_params

        device_controller = DeviceController()
        qc = device_controller.qubecalib

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
                            raise ValueError(
                                f"No mux found for readout port: {port.id}"
                            )
                        ndelay_or_nwait = control_params.capture_delay[mux.index]
                    else:
                        ndelay_or_nwait = 0
                    qc.define_channel(
                        channel_name=channel.id,
                        port_name=port.id,
                        channel_number=channel.number,
                        ndelay_or_nwait=ndelay_or_nwait,
                    )

        target_gen_channel_map = experiment_system.target_gen_channel_map
        for target, gen_channel in target_gen_channel_map.items():
            qc.define_target(
                target_name=target.label,
                channel_name=gen_channel.id,
                target_frequency=target.frequency,
            )
        target_cap_channel_map = experiment_system.target_cap_channel_map
        for target, cap_channel in target_cap_channel_map.items():
            qc.define_target(
                target_name=target.label,
                channel_name=cap_channel.id,
                target_frequency=target.frequency,
            )
        return device_controller

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
