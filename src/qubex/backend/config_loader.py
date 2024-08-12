from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Final

import yaml
from rich.prompt import Confirm

try:
    from qubecalib import QubeCalib
except ImportError:
    pass

from .control_system import Box, CapPort, ControlSystem, GenPort, PortType
from .experiment_system import ControlParams, ExperimentSystem, WiringInfo
from .quantum_system import Chip, QuantumSystem

CONFIG_DIR: Final = "config"
BUILD_DIR: Final = "build"
CHIP_FILE: Final = "chip.yaml"
BOX_FILE: Final = "box.yaml"
WIRING_FILE: Final = "wiring.yaml"
PROPS_FILE: Final = "props.yaml"
PARAMS_FILE: Final = "params.yaml"
SYSTEM_SETTINGS_FILE: Final = "system_settings.json"
BOX_SETTINGS_FILE: Final = "box_settings.json"

DEFAULT_CONTROL_AMPLITUDE: Final = 0.03
DEFAULT_READOUT_AMPLITUDE: Final = 0.01
DEFAULT_CONTROL_VATT: Final = 3072
DEFAULT_READOUT_VATT: Final = 2048
DEFAULT_CONTROL_FSC: Final = 40527
DEFAULT_READOUT_FSC: Final = 40527
DEFAULT_CAPTURE_DELAY: Final = 7


class ConfigLoader:
    def __init__(
        self,
        config_dir: str = CONFIG_DIR,
        *,
        chip_file: str = CHIP_FILE,
        box_file: str = BOX_FILE,
        wiring_file: str = WIRING_FILE,
        props_file: str = PROPS_FILE,
        params_file: str = PARAMS_FILE,
        system_settings_file: str = SYSTEM_SETTINGS_FILE,
        box_settings_file: str = BOX_SETTINGS_FILE,
    ):
        """
        Initializes the Config object.

        Parameters
        ----------
        config_dir : str, optional
            The directory where the configuration files are stored, by default "./config".
        chip_file : str, optional
            The name of the chip configuration file, by default "chip.yaml".
        box_file : str, optional
            The name of the box configuration file, by default "box.yaml".
        wiring_file : str, optional
            The name of the wiring configuration file, by default "wiring.yaml".
        props_file : str, optional
            The name of the properties configuration file, by default "props.yaml".
        params_file : str, optional
            The name of the parameters configuration file, by default "params.yaml".
        system_settings_file : str, optional
            The name of the system settings file, by default "system_settings.json".
        box_settings_file : str, optional
            The name of the box settings file, by default "box_settings.json".

        Examples
        --------
        >>> config = ConfigLoader()
        """
        self._config_dir = config_dir
        self._system_settings_file = system_settings_file
        self._box_settings_file = box_settings_file
        self._chip_dict = self._load_config_file(chip_file)
        self._box_dict = self._load_config_file(box_file)
        self._wiring_dict = self._load_config_file(wiring_file)
        self._props_dict = self._load_config_file(props_file)
        self._params_dict = self._load_config_file(params_file)
        self._quantum_system_dict = self._load_quantum_system()
        self._control_system_dict = self._load_control_system()
        self._wiring_info_dict = self._load_wiring_info()
        self._control_params_dict = self._load_control_params()
        self._experiment_system_dict = self._load_experiment_system()

    @property
    def config_path(self) -> Path:
        """Returns the absolute path to the configuration directory."""
        return Path(self._config_dir).resolve()

    def get_experiment_system(self, chip_id: str) -> ExperimentSystem:
        """
        Returns the ExperimentSystem object for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        ExperimentSystem
            The ExperimentSystem object for the given chip ID.

        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.get_experiment_system("64Q")
        """
        return self._experiment_system_dict[chip_id]

    def get_system_settings_path(self, chip_id: str) -> Path:
        """
        Returns the path to the system settings file for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        Path
            The path to the system settings file.

        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.get_system_settings_path("64Q")
        PosixPath('config/build/64Q/system_settings.json')
        """
        return Path(self._config_dir) / BUILD_DIR / chip_id / self._system_settings_file

    def get_box_settings_path(self, chip_id: str) -> Path:
        """
        Returns the path to the box settings file for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        Path
            The path to the box settings file.

        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.get_box_settings_path("64Q")
        PosixPath('config/build/64Q/box_settings.json')
        """
        return Path(self._config_dir) / BUILD_DIR / chip_id / self._box_settings_file

    def _load_config_file(self, file_name) -> dict:
        path = Path(self._config_dir) / file_name
        try:
            with open(path, "r") as file:
                result = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Configuration file not found: {path}")
            raise
        return result

    def _load_quantum_system(self) -> dict[str, QuantumSystem]:
        quantum_system_dict = {}
        for chip_id, chip_info in self._chip_dict.items():
            chip = Chip.new(
                id=chip_id,
                name=chip_info["name"],
                n_qubits=chip_info["n_qubits"],
            )
            props = self._props_dict[chip_id]
            for qubit in chip.qubits:
                qubit.frequency = props["qubit_frequency"][qubit.label]
                qubit.anharmonicity = props["anharmonicity"][qubit.label]
            for resonator in chip.resonators:
                resonator.frequency = props["resonator_frequency"][resonator.qubit]
            quantum_system = QuantumSystem(chip=chip)
            quantum_system_dict[chip_id] = quantum_system
        return quantum_system_dict

    def _load_control_system(self) -> dict[str, ControlSystem]:
        control_system_dict = {}
        for chip_id in self._chip_dict:
            box_ids = []
            for wiring in self._wiring_dict[chip_id]:
                box_ids.append(wiring["read_out"].split("-")[0])
                box_ids.append(wiring["read_in"].split("-")[0])
                for ctrl in wiring["ctrl"]:
                    box_ids.append(ctrl.split("-")[0])
            boxes = [
                Box.new(
                    id=id,
                    name=box["name"],
                    type=box["type"],
                    address=box["address"],
                    adapter=box["adapter"],
                )
                for id, box in self._box_dict.items()
                if id in box_ids
            ]
            control_system = ControlSystem(boxes=boxes)
            control_system_dict[chip_id] = control_system
        return control_system_dict

    def _load_wiring_info(self) -> dict[str, WiringInfo]:
        wiring_info_dict = {}
        for chip_id in self._chip_dict:
            quantum_system = self._quantum_system_dict[chip_id]
            control_system = self._control_system_dict[chip_id]

            def get_port(specifier: str):
                box_id = specifier.split("-")[0]
                port_num = int(specifier.split("-")[1])
                port = control_system.get_port(box_id, port_num)
                return port

            ctrl = []
            read_out = []
            read_in = []
            for wiring in self._wiring_dict[chip_id]:
                mux_num = int(wiring["mux"])
                mux = quantum_system.get_mux(mux_num)
                qubits = quantum_system.get_qubits_in_mux(mux_num)
                for identifier, qubit in zip(wiring["ctrl"], qubits):
                    ctrl_port: GenPort = get_port(identifier)  # type: ignore
                    ctrl.append((qubit, ctrl_port))
                read_out_port: GenPort = get_port(wiring["read_out"])  # type: ignore
                read_out.append((mux, read_out_port))
                read_in_port: CapPort = get_port(wiring["read_in"])  # type: ignore
                read_in.append((mux, read_in_port))

            wiring_info = WiringInfo(
                ctrl=ctrl,
                read_out=read_out,
                read_in=read_in,
            )
            wiring_info_dict[chip_id] = wiring_info
        return wiring_info_dict

    def _load_control_params(self) -> dict[str, ControlParams]:
        control_params_dict = {}
        for chip_id, params in self._params_dict.items():
            control_params = ControlParams(
                control_amplitude=params.get(
                    "control_amplitude",
                    defaultdict(lambda: DEFAULT_CONTROL_AMPLITUDE),
                ),
                readout_amplitude=params.get(
                    "readout_amplitude",
                    defaultdict(lambda: DEFAULT_READOUT_AMPLITUDE),
                ),
                control_vatt=params.get(
                    "control_vatt",
                    defaultdict(lambda: DEFAULT_CONTROL_VATT),
                ),
                readout_vatt=params.get(
                    "readout_vatt",
                    defaultdict(lambda: DEFAULT_READOUT_VATT),
                ),
                control_fsc=params.get(
                    "control_fsc",
                    defaultdict(lambda: DEFAULT_CONTROL_FSC),
                ),
                readout_fsc=params.get(
                    "readout_fsc",
                    defaultdict(lambda: DEFAULT_READOUT_FSC),
                ),
                capture_delay=params.get(
                    "capture_delay",
                    defaultdict(lambda: DEFAULT_CAPTURE_DELAY),
                ),
            )
            control_params_dict[chip_id] = control_params
        return control_params_dict

    def _load_experiment_system(self) -> dict[str, ExperimentSystem]:
        experiment_system_dict = {}
        for chip_id in self._chip_dict:
            quantum_system = self._quantum_system_dict[chip_id]
            control_system = self._control_system_dict[chip_id]
            wiring_info = self._wiring_info_dict[chip_id]
            control_params = self._control_params_dict[chip_id]
            experiment_system = ExperimentSystem(
                quantum_system=quantum_system,
                control_system=control_system,
                wiring_info=wiring_info,
                control_params=control_params,
            )
            experiment_system_dict[chip_id] = experiment_system
        return experiment_system_dict

    def generate_system_settings(
        self,
        chip_id: str,
        path_to_save: str | None = None,
    ) -> Path:
        """
        Configures the system settings for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        path_to_save : str | None, optional
            The path to save the system settings file, by default None.

        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.configure_system_settings("64Q")
        """
        control_system = self.get_control_system(chip_id)
        qube_system = control_system.qube_system
        params = self.get_params(chip_id)

        qc = QubeCalib()

        # define clock master
        qc.define_clockmaster(
            ipaddr=qube_system.clock_master_address,
            reset=True,
        )

        # define boxes, ports, and channels
        for box in qube_system.boxes.values():
            # define box
            qc.define_box(
                box_name=box.id,
                ipaddr_wss=box.address,
                boxtype=box.type.value,
            )

            # define ports
            for port in box.ports:
                if port.type == PortType.NA:
                    continue
                qc.define_port(
                    port_name=port.id,
                    box_name=box.id,
                    port_number=port.number,
                )

                # define channels
                for channel in port.channels:
                    if port.type == PortType.READ_IN:
                        mux = control_system.get_mux_by_port(port)
                        ndelay_or_nwait = params.capture_delay[mux.number]
                    else:
                        ndelay_or_nwait = 0
                    qc.define_channel(
                        channel_name=channel.id,
                        port_name=port.id,
                        channel_number=channel.number,
                        ndelay_or_nwait=ndelay_or_nwait,
                    )

        # define control targets
        targets = control_system.targets
        for target, channel in control_system.ctrl_channel_map.items():
            qc.define_target(
                target_name=target,
                channel_name=channel.id,
                target_frequency=targets[target].frequency,
            )
        # define readout (out) targets
        for target, channel in control_system.read_out_channel_map.items():
            qc.define_target(
                target_name=target,
                channel_name=channel.id,
                target_frequency=targets[target].frequency,
            )
        # define readout (in) targets
        for target, channel in control_system.read_in_channel_map.items():
            qc.define_target(
                target_name=target,
                channel_name=channel.id,
                target_frequency=targets[target].frequency,
            )

        # save the system settings to a JSON file
        if path_to_save is None:
            system_settings_path = self.get_system_settings_path(chip_id)
        else:
            system_settings_path = Path(path_to_save)
        system_settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(system_settings_path, "w") as f:
            f.write(qc.system_config_database.asjson())

        return system_settings_path

    def configure_box_settings(
        self,
        chip_id: str,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        loopback: bool = False,
    ):
        """
        Configures the box settings for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        include : list[str] | None, optional
            The list of box IDs to include, by default None.
        exclude : list[str] | None, optional
            The list of box IDs to exclude, by default None.
        loopback : bool, optional
            Whether to enable loopback mode, by default False.

        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.configure_box_settings("64Q")

        >>> config.configure_box_settings(
        ...     chip_id="64Q",
        ...     include=["Q2A", "Q73A"],
        ...     loopback=True,
        ... )

        >>> config.configure_box_settings(
        ...     chip_id="64Q",
        ...     exclude=["Q2A", "Q73A"],
        ...     loopback=True,
        ... )
        """

        try:
            system_settings_path = self.get_system_settings_path(chip_id)
            qc = QubeCalib(str(system_settings_path))
        except FileNotFoundError:
            print(f"System settings file not found for chip ID: {chip_id}")
            return

        control_system = self.get_control_system(chip_id)
        control_system = self.configure_control_system(chip_id, loopback=loopback)
        qube_system = control_system.qube_system
        boxes = list(qube_system.boxes.values())

        if include is not None:
            boxes = [box for box in boxes if box.id in include]
        if exclude is not None:
            boxes = [box for box in boxes if box.id not in exclude]

        box_list_str = "\n".join([f"{box.id} ({box.name})" for box in boxes])
        confirmed = Confirm.ask(
            f"""
You are going to configure the following boxes:

[bold bright_green]{box_list_str}[/bold bright_green]

This operation will overwrite the existing device settings. Do you want to continue?
"""
        )
        if not confirmed:
            print("Operation cancelled.")
            return

        for box in boxes:
            quel1_box = qc.create_box(box.id, reconnect=False)
            quel1_box.reconnect()
            for port in box.ports:
                if port.type in [PortType.NA, PortType.CTRL]:
                    quel1_box.config_port(
                        port=port.number,
                        lo_freq=port.lo_freq,
                        cnco_freq=port.cnco_freq,
                        vatt=port.vatt,
                        sideband=port.sideband,
                        fullscale_current=port.fullscale_current,
                        rfswitch=port.rfswitch,
                    )
                    for channel in port.channels:
                        quel1_box.config_channel(
                            port=port.number,
                            channel=channel.number,
                            fnco_freq=channel.fnco_freq,
                        )
                elif port.type == PortType.READ_IN:
                    quel1_box.config_port(
                        port=port.number,
                        lo_freq=port.lo_freq,
                        cnco_locked_with=qube_system.get_readout_pair(port).number,
                        rfswitch=port.rfswitch,
                    )
                    for channel in port.channels:
                        quel1_box.config_runit(
                            port=port.number,
                            runit=channel.number,
                            fnco_freq=channel.fnco_freq,
                        )
        print("Box settings configured.")

    def save_box_settings(
        self,
        *,
        chip_id: str,
        path_to_save: str | None = None,
    ):
        """
        Saves the box settings for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.save_box_settings("64Q")
        """
        try:
            system_settings_path = self.get_system_settings_path(chip_id)
            qc = QubeCalib(str(system_settings_path))
        except FileNotFoundError:
            print(f"System settings file not found for chip ID: {chip_id}")
            return
        if path_to_save is None:
            box_settings_path = self.get_box_settings_path(chip_id)
        else:
            box_settings_path = Path(path_to_save)
        box_settings_path.parent.mkdir(parents=True, exist_ok=True)
        qc.store_all_box_configs(box_settings_path)

        print(f"Box settings saved to: {box_settings_path}")
