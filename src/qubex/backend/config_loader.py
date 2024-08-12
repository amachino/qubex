from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml
from rich.prompt import Confirm

try:
    from qubecalib import QubeCalib
except ImportError:
    pass

from .experiment_system import ExperimentSystem, Mux, Target
from .quantum_system import Chip, QuantumSystem, Qubit, Resonator
from .control_system import Box, BoxType, Port, PortType, ControlSystem

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


@dataclass(frozen=True)
class Params:
    control_amplitude: dict[str, float]
    readout_amplitude: dict[str, float]
    control_vatt: dict[str, int]
    readout_vatt: dict[int, int]
    control_fsc: dict[str, int]
    readout_fsc: dict[int, int]
    capture_delay: dict[int, int]


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

    @property
    def config_path(self) -> Path:
        """Returns the absolute path to the configuration directory."""
        return Path(self._config_dir).resolve()

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

    def get_params(self, chip_id: str) -> Params:
        """
        Returns the Params object for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        Params
            The Params object for the given chip ID.
        """
        try:
            params = self._params_dict[chip_id]
        except KeyError:
            print(f"Parameters not found for chip ID: {chip_id}")
            raise
        return Params(
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

    def get_chip(self, id: str) -> Chip:
        """
        Returns the Chip object for the given ID.

        Parameters
        ----------
        id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        Chip
            The Chip object for the given ID.
        """
        chip_info = self._chip_dict[id]
        return Chip(
            id=id,
            name=chip_info["name"],
            n_qubits=chip_info["n_qubits"],
        )

    def get_all_chips(self) -> list[Chip]:
        """
        Returns a list of all Chip objects.

        Returns
        -------
        list[Chip]
            A list of all Chip objects
        """
        return [self.get_chip(id) for id in self._chip_dict.keys()]

    def get_qubits(self, chip_id: str) -> list[Qubit]:
        """
        Returns a list of Qubit objects for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        list[Qubit]
            A list of Qubit objects for the given chip ID.
        """
        chip = self.get_chip(chip_id)
        props = self._props_dict[chip_id]
        return [
            Qubit(
                index=index,
                label=label,
                frequency=props["qubit_frequency"][label],
                anharmonicity=props["anharmonicity"][label],
                resonator=Target.get_readout_label(label),
            )
            for index, label in enumerate(chip.qubits)
        ]

    def get_resonators(self, chip_id: str) -> list[Resonator]:
        """
        Returns a list of Resonator objects for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        list[Resonator]
            A list of Resonator objects for the given chip ID.
        """
        qubits = self.get_qubits(chip_id)
        props = self._props_dict[chip_id]
        return [
            Resonator(
                index=qubit.index,
                label=Target.get_readout_label(qubit.label),
                frequency=props["resonator_frequency"][qubit.label],
                qubit=qubit.label,
            )
            for qubit in qubits
        ]

    def get_quantum_system(self, chip_id: str) -> QuantumSystem:
        """
        Returns the QuantumSystem object for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        QuantumSystem
            The QuantumSystem object for the given chip ID.
        """
        chip = self.get_chip(chip_id)
        qubits = self.get_qubits(chip_id)
        resonators = self.get_resonators(chip_id)
        return QuantumSystem(
            chip=chip,
            qubits=qubits,
            resonators=resonators,
        )

    def get_box(self, id: str) -> Box:
        """
        Returns the Box object for the given ID.

        Parameters
        ----------
        id : str
            The box ID (e.g., "Q2A").

        Returns
        -------
        Box
            The Box object for the given ID.
        """
        box_info = self._box_dict[id]
        return Box(
            id=id,
            name=box_info["name"],
            type=BoxType(box_info["type"]),
            address=box_info["address"],
            adapter=box_info["adapter"],
        )

    def get_all_boxes(self) -> list[Box]:
        """
        Returns a list of all Box objects.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        list[Box]
            A list of all Box objects
        """
        return [self.get_box(id) for id in self._box_dict.keys()]

    def get_boxes(
        self,
        chip_id: str,
        qubits: list[str] | None = None,
    ) -> list[Box]:
        """
        Returns a list of Box objects for the given chip ID and qubits.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        qubits : list[str] | None, optional
            The list of qubits, by default None.

        Returns
        -------
        list[Box]
            A list of Box objects for the given chip ID and qubits.
        """
        wiring_list = self._wiring_dict[chip_id]
        box_ids = set()
        for wiring in wiring_list:
            if qubits:
                # skip if no qubit in the mux is in the given list
                mux = wiring["mux"]
                quantum_system = self.get_quantum_system(chip_id)
                qubits_in_mux = quantum_system.get_qubits_in_mux(mux)
                if not any(qubit.label in qubits for qubit in qubits_in_mux):
                    continue
            box_ids.add(wiring["read_out"].split("-")[0])
            box_ids.add(wiring["read_in"].split("-")[0])
            for ctrl in wiring["ctrl"]:
                box_ids.add(ctrl.split("-")[0])
        return [box for box in self.get_all_boxes() if box.id in box_ids]

    def get_qube_system(
        self,
        chip_id: str | None = None,
    ) -> ControlSystem:
        """
        Returns the QubeSystem object for the given chip ID.

        Parameters
        ----------
        chip_id : str | None, optional
            The quantum chip ID (e.g., "64Q"), by default None.

        Returns
        -------
        QubeSystem
            The QubeSystem object for the given chip ID.
        """
        if chip_id is None:
            boxes = self.get_all_boxes()
        else:
            boxes = self.get_boxes(chip_id)
        return ControlSystem(boxes=boxes)

    def get_port_from_specifier(
        self,
        specifier: str,
    ) -> Port:
        """
        Returns the Port object for the given specifier.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        specifier : str
            The specifier for the port (e.g., "Q73A-11").

        Returns
        -------
        Port
            The Port object for the given specifier.
        """
        box_id, port_str = specifier.split("-")
        port_number = int(port_str)
        qube_system = self.get_qube_system()
        box = qube_system.get_box(box_id)
        port = box.ports[port_number]
        return port

    def get_muxes(self, chip_id: str) -> list[Mux]:
        """
        Returns a list of Mux objects for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        list[Mux]
            A list of Mux objects for the given chip ID.
        """
        wiring_list = self._wiring_dict[chip_id]
        quantum_system = self.get_quantum_system(chip_id)
        muxes: list[Mux] = []
        for wiring in wiring_list:
            mux_number = wiring["mux"]
            qubits = quantum_system.get_qubits_in_mux(mux_number)
            read_out = self.get_port_from_specifier(wiring["read_out"])
            read_in = self.get_port_from_specifier(wiring["read_in"])
            ctrls = [self.get_port_from_specifier(port) for port in wiring["ctrl"]]
            mux = Mux(
                number=mux_number,
                qubits=tuple(qubits),
                ctrl_ports=tuple(ctrls),
                read_in_port=read_in,
                read_out_port=read_out,
            )
            muxes.append(mux)
        return muxes

    def get_control_system(self, chip_id: str) -> ExperimentSystem:
        """
        Returns the ControlSystem object for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        ControlSystem
            The ControlSystem object for the given chip ID.
        """
        quantum_system = self.get_quantum_system(chip_id)
        qube_system = self.get_qube_system(chip_id)
        muxes = self.get_muxes(chip_id)
        return ExperimentSystem(
            quantum_system=quantum_system,
            control_system=qube_system,
            muxes=muxes,
        )

    def configure_control_system(
        self,
        chip_id: str,
        loopback: bool = False,
    ) -> ExperimentSystem:
        """
        Configures the ControlSystem object for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        ControlSystem
            The configured ControlSystem object for the given chip ID.
        """
        control_system = self.get_control_system(chip_id)
        qube_system = control_system.qube_system

        params = self.get_params(chip_id)
        control_vatt = params.control_vatt
        readout_vatt = params.readout_vatt
        control_fsc = params.control_fsc
        readout_fsc = params.readout_fsc
        capture_delay = params.capture_delay

        for box in qube_system.boxes.values():
            for port in box.ports:
                if port.type == PortType.READ_OUT:
                    mux = control_system.get_mux_by_port(port)
                    lo, cnco, fnco = self.find_read_lo_nco(
                        chip_id=chip_id,
                        qubits=mux.qubit_labels,
                    )
                    port.lo_freq = lo
                    port.cnco_freq = cnco
                    port.vatt = readout_vatt[mux.number]
                    port.sideband = "U"
                    port.fullscale_current = readout_fsc[mux.number]
                    port.loopback = loopback
                    port.channels[0].fnco_freq = fnco
                elif port.type == PortType.READ_IN:
                    mux = control_system.get_mux_by_port(port)
                    lo, cnco, fnco = self.find_read_lo_nco(
                        chip_id=chip_id,
                        qubits=mux.qubit_labels,
                    )
                    port.lo_freq = lo
                    port.loopback = loopback
                    for channel in port.channels:
                        channel.fnco_freq = fnco
                        channel.ndelay = capture_delay[mux.number]
                elif port.type == PortType.CTRL:
                    qubit = control_system.port_qubit_map[port.id].label
                    lo, cnco, fncos = self.find_ctrl_lo_nco(
                        chip_id=chip_id,
                        qubit=qubit,
                        n_channels=port.n_channels,
                    )
                    port.lo_freq = lo
                    port.cnco_freq = cnco
                    port.vatt = control_vatt[qubit]
                    port.sideband = "L"
                    port.fullscale_current = control_fsc[qubit]
                    port.loopback = loopback
                    for idx, channel in enumerate(port.channels):
                        channel.fnco_freq = fncos[idx]
        return control_system

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

    def find_read_lo_nco(
        self,
        chip_id: str,
        qubits: list[str],
        *,
        lo_min: int = 8_000_000_000,
        lo_max: int = 11_000_000_000,
        lo_step: int = 500_000_000,
        nco_step: int = 23_437_500,
        cnco: int = 1_500_000_000,
        fnco_min: int = -234_375_000,
        fnco_max: int = +234_375_000,
    ) -> tuple[int, int, int]:
        """
        Finds the (lo, cnco, fnco) values for the readout qubits.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        qubits : list[str]
            The readout qubits.
        lo_min : int, optional
            The minimum LO frequency, by default 8_000_000_000.
        lo_max : int, optional
            The maximum LO frequency, by default 11_000_000_000.
        lo_step : int, optional
            The LO frequency step, by default 500_000_000.
        nco_step : int, optional
            The NCO frequency step, by default 23_437_500.
        cnco : int, optional
            The CNCO frequency, by default 2_250_000_000.
        fnco_min : int, optional
            The minimum FNCO frequency, by default -750_000_000.
        fnco_max : int, optional
            The maximum FNCO frequency, by default +750_000_000.

        Returns
        -------
        tuple[int, int, int]
            The tuple (lo, cnco, fnco) for the readout qubits.
        """
        default_frequency = 10.0

        props = self._props_dict[chip_id]
        frequencies = props["resonator_frequency"]

        mux_frequencies = [
            frequencies.get(qubit, default_frequency) * 1e9 for qubit in qubits
        ]
        target_frequency = (max(mux_frequencies) + min(mux_frequencies)) / 2

        min_diff = float("inf")
        best_lo = None
        best_fnco = None

        for lo in range(lo_min, lo_max + 1, lo_step):
            for fnco in range(fnco_min, fnco_max + 1, nco_step):
                current_value = lo + cnco + fnco
                current_diff = abs(current_value - target_frequency)
                if current_diff < min_diff:
                    min_diff = current_diff
                    best_lo = lo
                    best_fnco = fnco
        if best_lo is None or best_fnco is None:
            raise ValueError("No valid (lo, fnco) pair found.")
        return best_lo, cnco, best_fnco

    def find_ctrl_lo_nco(
        self,
        chip_id: str,
        qubit: str,
        n_channels: int,
        *,
        lo_min: int = 8_000_000_000,
        lo_max: int = 11_000_000_000,
        lo_step: int = 500_000_000,
        nco_step: int = 23_437_500,
        cnco: int = 2_250_000_000,
        fnco_min: int = -750_000_000,
        fnco_max: int = +750_000_000,
        max_diff: int = 2_000_000_000,
    ) -> tuple[int, int, tuple[int, int, int]]:
        """
        Finds the (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) values for the control qubit.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        qubit : str
            The control qubit.
        n_channels : int
            The number of channels.
        lo_min : int, optional
            The minimum LO frequency, by default 8_000_000_000.
        lo_max : int, optional
            The maximum LO frequency, by default 11_000_000_000.
        lo_step : int, optional
            The LO frequency step, by default 500_000_000.
        nco_step : int, optional
            The NCO frequency step, by default 23_437_500.
        cnco : int, optional
            The CNCO frequency, by default 2_250_000_000.
        fnco_min : int, optional
            The minimum FNCO frequency, by default -750_000_000.
        fnco_max : int, optional
            The maximum FNCO frequency, by default +750_000_000.
        max_diff : int, optional
            The maximum difference between CR and ge frequencies, by default 1_500_000_000.

        Returns
        -------
        tuple[int, int, tuple[int, int, int]]
            The tuple (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) for the control qubit.
        """
        control_system = self.get_control_system(chip_id)

        target_ge = f"{qubit}-ge"
        target_ef = f"{qubit}-ef"
        target_cr = f"{qubit}-CR"

        f_ge = control_system.get_ge_target(qubit).frequency * 1e9
        f_ef = control_system.get_ef_target(qubit).frequency * 1e9
        f_cr = control_system.get_cr_target(qubit).frequency * 1e9

        if n_channels == 1:
            f_med = f_ge
        elif n_channels == 3:
            freq = {
                target_ge: f_ge,
                target_ef: f_ef,
                target_cr: f_cr,
            }
            target_max, f_max = max(freq.items(), key=lambda item: item[1])
            target_min, f_min = min(freq.items(), key=lambda item: item[1])

            if f_max - f_min > max_diff:
                print(
                    f"Warning: {target_max} ({f_max * 1e-9:.3f} GHz) is too far from {target_min} ({f_min * 1e-9:.3f} GHz). Ignored {target_cr}."
                )
                freq[target_cr] = f_ge
                f_max = f_ge
                f_min = f_ef

            f_med = (f_max + f_min) / 2
        else:
            raise ValueError("Invalid number of channels: ", n_channels)

        min_diff = float("inf")
        best_lo = None
        for lo in range(lo_min, lo_max + 1, lo_step):
            current_value = lo - cnco
            current_diff = abs(current_value - f_med)
            if current_diff < min_diff:
                min_diff = current_diff
                best_lo = lo
        if best_lo is None:
            raise ValueError("No valid lo value found for: ", freq)

        def find_fnco(target_frequency: float):
            min_diff = float("inf")
            best_fnco = None
            for fnco in range(fnco_min, fnco_max + 1, nco_step):
                current_value = abs(best_lo - cnco - fnco)
                current_diff = abs(current_value - target_frequency)
                if current_diff < min_diff:
                    min_diff = current_diff
                    best_fnco = fnco
            if best_fnco is None:
                raise ValueError("No valid fnco value found for: ", freq)
            return best_fnco

        fnco_ge = find_fnco(f_ge)

        if n_channels == 1:
            return best_lo, cnco, (fnco_ge, 0, 0)

        fnco_ef = find_fnco(f_ef)
        fnco_cr = find_fnco(f_cr)

        return best_lo, cnco, (fnco_ge, fnco_ef, fnco_cr)
