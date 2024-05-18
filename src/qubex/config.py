from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml
from qubecalib import QubeCalib
from rich.console import Console
from rich.prompt import Confirm

from .hardware import Box, BoxType, CtrlPort, Port, PortType, ReadInPort, ReadOutPort

console = Console()


@dataclass
class Chip:
    id: str
    name: str
    n_qubits: int


@dataclass
class Qubit:
    label: str
    frequency: float
    anharmonicity: float


@dataclass
class Resonator:
    label: str
    frequency: float


@dataclass
class Props:
    resonator_frequency: dict[str, float]
    qubit_frequency: dict[str, float]
    anharmonicity: dict[str, float]


@dataclass
class Params:
    control_amplitude: dict[str, float]
    readout_amplitude: dict[str, float]
    control_vatt: dict[str, int]
    readout_vatt: dict[int, int]


class TargetType(Enum):
    CTRL_GE = "CTRL_GE"
    CTRL_EF = "CTRL_EF"
    CTRL_CR = "CTRL_CR"
    READ = "READ"


@dataclass
class Target:
    label: str
    frequency: float
    type: TargetType
    qubit: str


CONFIG_DIR = "config"
BUILD_DIR = "build"
CHIP_FILE = "chip.yaml"
BOX_FILE = "box.yaml"
WIRING_FILE = "wiring.yaml"
PROPS_FILE = "props.yaml"
PARAMS_FILE = "params.yaml"
SYSTEM_SETTINGS_FILE = "system_settings.json"
BOX_SETTINGS_FILE = "box_settings.json"


class Config:
    """
    Config class provides methods to configure the QubeX system.
    """

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
        >>> config = Config()
        """
        self._config_dir = config_dir
        self._system_settings_file = system_settings_file
        self._box_settings_file = box_settings_file
        self._chip_dict = self._load_config_file(chip_file)
        self._box_dict = self._load_config_file(box_file)
        self._wiring_dict = self._load_config_file(wiring_file)
        self._props_dict = self._load_config_file(props_file)
        self._params_dict = self._load_config_file(params_file)

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
        >>> config = Config()
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
        >>> config = Config()
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
            console.print(
                f"Configuration file not found: {path}",
                style="red bold",
            )
        return result

    def get_props(self, chip_id: str) -> Props:
        """
        Returns the properties of the quantum chip.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        Props
            The properties of the quantum chip.
        """
        try:
            props = self._props_dict[chip_id]
        except KeyError:
            console.print(
                f"Properties not found for chip ID: {chip_id}",
                style="red bold",
            )
            raise
        return Props(
            resonator_frequency=props["resonator_frequency"],
            qubit_frequency=props["qubit_frequency"],
            anharmonicity=props["anharmonicity"],
        )

    def get_params(self, chip_id: str) -> Params:
        """
        Returns the parameters of the quantum chip.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        Params
            The parameters of the quantum chip.
        """
        try:
            params = self._params_dict[chip_id]
        except KeyError:
            console.print(
                f"Parameters not found for chip ID: {chip_id}",
                style="red bold",
            )
            raise
        return Params(
            control_amplitude=params["control_amplitude"],
            readout_amplitude=params["readout_amplitude"],
            control_vatt=params["control_vatt"],
            readout_vatt=params["readout_vatt"],
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
            A list of all Chip objects.
        """
        return [self.get_chip(id) for id in self._chip_dict.keys()]

    def get_qubit(self, chip_id: str, label: str) -> Qubit:
        """
        Returns the Qubit object for the given chip ID and label.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        label : str
            The qubit label (e.g., "Q00").

        Returns
        -------
        Qubit
            The Qubit object for the given chip ID and label.
        """
        props = self.get_props(chip_id)
        return Qubit(
            label=label,
            frequency=props.qubit_frequency[label],
            anharmonicity=props.anharmonicity[label],
        )

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
        qubit_labels = [f"Q{i:02d}" for i in range(chip.n_qubits)]
        props = self.get_props(chip_id)
        return [
            Qubit(
                label=label,
                frequency=props.qubit_frequency[label],
                anharmonicity=props.anharmonicity[label],
            )
            for label in qubit_labels
        ]

    def get_resonator(self, chip_id: str, label: str) -> Resonator:
        """
        Returns the Resonator object for the given chip ID and label.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        label : str
            The qubit label (e.g., "Q00").

        Returns
        -------
        Resonator
            The Resonator object for the given chip ID and label.
        """
        props = self.get_props(chip_id)
        return Resonator(
            label=f"R{label}",
            frequency=props.resonator_frequency[label],
        )

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
        props = self.get_props(chip_id)
        qubits = self.get_qubits(chip_id)
        return [
            Resonator(
                label=f"R{qubit.label}",
                frequency=props.resonator_frequency[qubit.label],
            )
            for qubit in qubits
        ]

    def get_read_target(self, chip_id: str, label: str) -> Target:
        """
        Returns the readout Target object for the given chip ID and qubit label.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        label : str
            The qubit label (e.g., "Q00").

        Returns
        -------
        Target
            The readout Target object for the given chip ID and qubit label.
        """
        resonator = self.get_resonator(chip_id, label)
        return Target(
            label=resonator.label,
            frequency=resonator.frequency,
            type=TargetType.READ,
            qubit=label,
        )

    def get_ctrl_ge_target(self, chip_id: str, label: str) -> Target:
        """
        Returns the control (ge) Target object for the given chip ID and qubit label.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        label : str
            The qubit label (e.g., "Q00").

        Returns
        -------
        Target
            The control (ge) Target object for the given chip ID and qubit label.
        """
        qubit = self.get_qubit(chip_id, label)
        return Target(
            label=label,
            frequency=qubit.frequency,
            type=TargetType.CTRL_GE,
            qubit=label,
        )

    def get_ctrl_ef_target(self, chip_id: str, label: str) -> Target:
        """
        Returns the control (ef) Target object for the given chip ID and qubit label.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        label : str
            The qubit label (e.g., "Q00").

        Returns
        -------
        Target
            The control (ef) Target object for the given chip ID and qubit label.
        """
        qubit = self.get_qubit(chip_id, label)
        return Target(
            label=f"{label}-ef",
            frequency=qubit.frequency + qubit.anharmonicity,
            type=TargetType.CTRL_EF,
            qubit=label,
        )

    def get_ctrl_cr_targets(self, chip_id: str, label: str) -> list[Target]:
        """
        Returns a list of control (CR) Target objects for the given chip ID and qubit label.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        label : str
            The qubit label (e.g., "Q00").

        Returns
        -------
        list[Target]
            A list of control (CR) Target objects for the given chip ID and qubit label.
        """
        target_qubits = self.get_cr_target_qubits(chip_id, label)
        return [
            Target(
                label=f"{label}-CR",
                # label=f"{label}-{target_qubit.label}",
                frequency=target_qubit.frequency,
                type=TargetType.CTRL_CR,
                qubit=label,
            )
            for target_qubit in target_qubits
        ]

    def get_cr_target_qubits(self, chip_id: str, label: str) -> list[Qubit]:
        """
        Returns a list of target qubits for the cross-resonance control qubit.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        label : str
            The cross-resonance control qubit label (e.g., "Q00").

        Returns
        -------
        list[Qubit]
            A list of target qubits for the cross-resonance control qubit.
        """
        qubits = self.get_qubits(chip_id)
        control = self.get_qubit(chip_id, label)
        control_index = qubits.index(control)
        mux_number = control_index // 4
        control_index_in_mux = control_index % 4
        edges = {
            0: (1, 2),
            1: (3, 0),
            2: (0, 3),
            3: (2, 1),
        }
        target_indices = [4 * mux_number + edge for edge in edges[control_index_in_mux]]
        target_qubits = [qubits[i] for i in target_indices]
        return target_qubits

    def get_box(self, id: str) -> Box:
        """
        Returns the Box object for the given ID.

        Parameters
        ----------
        id : str
            The ID of the box (e.g., "Q73A").

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

        Returns
        -------
        list[Box]
            A list of all Box objects.
        """
        return [self.get_box(id) for id in self._box_dict.keys()]

    def get_boxes(self, chip_id: str) -> list[Box]:
        """
        Returns a list of Box objects for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        list[Box]
            A list of Box objects for the given chip ID.
        """
        ports = self.get_port_details(chip_id)
        box_set = set([port.box.id for port in ports])
        return [self.get_box(box_id) for box_id in box_set]

    def get_port(self, box: Box, port_number: int) -> Port:
        """
        Returns the Port object for the given box and port number.

        Parameters
        ----------
        box : Box
            The Box object.
        port_number : int
            The port number.

        Returns
        -------
        Port
            The Port object for the given box and port number.
        """
        return Port(number=port_number, box=box)

    def get_ports(self, box: Box) -> list[Port]:
        """
        Returns a list of Port objects for the given box.

        Parameters
        ----------
        box : Box
            The Box object.

        Returns
        -------
        list[Port]
            A list of Port objects for the given box.
        """
        return [self.get_port(box, port_number) for port_number in range(14)]

    def get_port_from_specifier(
        self,
        specifier: str,
    ) -> Port:
        """
        Returns the Port object for the given specifier.

        Parameters
        ----------
        specifier : str
            The specifier for the port (e.g., "Q73A-11").

        Returns
        -------
        Port
            The Port object for the given specifier.
        """
        box_id, port_str = specifier.split("-")
        port_number = int(port_str)
        box = self.get_box(box_id)
        port = self.get_port(box, port_number)
        return port

    def get_qubit_names(self, mux: int) -> list[str]:
        """
        Returns the qubit names for the given multiplexer.

        Parameters
        ----------
        mux : int
            The multiplexer number.

        Returns
        -------
        list[str]
            A list of qubit names for the given multiplexer.
        """
        return [f"Q{4 * mux + i:02d}" for i in range(4)]

    def get_port_details(self, chip_id: str) -> list[Port]:
        """
        Returns a list of Port objects for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Returns
        -------
        list[Port]
            A list of Port objects for the given chip ID.
        """
        try:
            wiring_list = self._wiring_dict[chip_id]
        except KeyError:
            console.print(
                f"Wiring configuration not found for chip ID: {chip_id}",
                style="red bold",
            )
            raise
        ports: list[Port] = []
        for wiring in wiring_list:
            mux = wiring["mux"]
            qubits = self.get_qubit_names(mux)
            read_out = self.get_port_from_specifier(wiring["read_out"])
            read_in = self.get_port_from_specifier(wiring["read_in"])
            ctrls = [self.get_port_from_specifier(port) for port in wiring["ctrl"]]
            read_out_port = ReadOutPort(
                number=read_out.number,
                box=read_out.box,
                mux=mux,
            )
            read_in_port = ReadInPort(
                number=read_in.number,
                box=read_in.box,
                read_out=read_out_port,
            )
            ctrl_ports = [
                CtrlPort(
                    number=ctrl.number,
                    box=ctrl.box,
                    ctrl_qubit=qubit,
                )
                for ctrl, qubit in zip(ctrls, qubits)
            ]
            ports.append(read_out_port)
            ports.append(read_in_port)
            ports.extend(ctrl_ports)
        return ports

    def configure_system_settings(self, chip_id: str):
        """
        Configures the system settings for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Examples
        --------
        >>> config = Config()
        >>> config.configure_system_settings("64Q")
        """
        qc = QubeCalib()

        boxes = self.get_boxes(chip_id)

        # define boxes, ports, and channels
        for box in boxes:
            # define box
            qc.define_box(
                box_name=box.id,
                ipaddr_wss=box.address,
                boxtype=box.type.value,
            )

            # define ports
            ports = self.get_ports(box)
            for port in ports:
                if port.type == PortType.NOT_AVAILABLE:
                    continue
                qc.define_port(
                    port_name=port.name,
                    box_name=box.id,
                    port_number=port.number,
                )

        # define channels
        ports = self.get_port_details(chip_id)
        for port in ports:
            if isinstance(port, ReadOutPort):
                qc.define_channel(
                    channel_name=f"{port.name}0",
                    port_name=port.name,
                    channel_number=0,
                )
            elif isinstance(port, ReadInPort):
                for runit_index in range(4):
                    qc.define_channel(
                        channel_name=f"{port.name}{runit_index}",
                        port_name=port.name,
                        channel_number=runit_index,
                    )
            elif isinstance(port, CtrlPort):
                for channel_index in range(port.n_channel):
                    qc.define_channel(
                        channel_name=f"{port.name}.CH{channel_index}",
                        port_name=port.name,
                        channel_number=channel_index,
                    )

        # define targets
        for port in ports:
            if isinstance(port, ReadOutPort):
                for qubit in port.read_qubits:
                    target = self.get_read_target(chip_id, qubit)
                    qc.define_target(
                        target_name=target.label,
                        channel_name=f"{port.name}0",
                        target_frequency=target.frequency,
                    )
            elif isinstance(port, ReadInPort):
                for idx, qubit in enumerate(port.read_out.read_qubits):
                    target = self.get_read_target(chip_id, qubit)
                    qc.define_target(
                        target_name=target.label,
                        channel_name=f"{port.name}{idx}",
                        target_frequency=target.frequency,
                    )
            elif isinstance(port, CtrlPort):
                qubit = port.ctrl_qubit
                target_ge = self.get_ctrl_ge_target(chip_id, qubit)
                qc.define_target(
                    target_name=target_ge.label,
                    channel_name=f"{port.name}.CH0",
                    target_frequency=target_ge.frequency,
                )
                if port.n_channel == 3:
                    target_ef = self.get_ctrl_ef_target(chip_id, qubit)
                    target_cr = self.get_ctrl_cr_targets(chip_id, qubit)[0]
                    qc.define_target(
                        target_name=target_ef.label,
                        channel_name=f"{port.name}.CH1",
                        target_frequency=target_ef.frequency,
                    )
                    qc.define_target(
                        target_name=target_cr.label,
                        channel_name=f"{port.name}.CH2",
                        target_frequency=target_cr.frequency,
                    )

        # save the system settings to a JSON file
        system_settings_path = self.get_system_settings_path(chip_id)
        system_settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(system_settings_path, "w") as f:
            f.write(qc.system_config_database.asjson())

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
        >>> config = Config()
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
            console.print(
                f"System settings file not found for chip ID: {chip_id}",
                style="red bold",
            )
            return
        boxes = self.get_boxes(chip_id)
        if include is not None:
            boxes = [box for box in boxes if box.id in include]
        if exclude is not None:
            boxes = [box for box in boxes if box.id not in exclude]

        box_list_str = "\n".join([f"{box.id} ({box.name})" for box in boxes])
        confirmed = Confirm.ask(
            f"""
We are going to configure the following boxes:

[bold green]{box_list_str}

[bold italic yellow]This operation will overwrite the existing box settings. Do you want to continue?
"""
        )
        if not confirmed:
            console.print("Operation cancelled.", style="red bold")
            return

        ports = self.get_port_details(chip_id)
        props = self.get_props(chip_id)
        params = self.get_params(chip_id)
        read_frequencies = props.resonator_frequency
        ctrl_frequencies = props.qubit_frequency
        readout_vatt = params.readout_vatt
        control_vatt = params.control_vatt

        for box in boxes:
            quel1_box = qc.create_box(box.id)
            for port in ports:
                if port.box.id != box.id:
                    continue

                if isinstance(port, ReadOutPort):
                    lo, nco = ConfigUtils.find_read_lo_nco(
                        frequencies=read_frequencies,
                        qubits=port.read_qubits,
                    )
                    quel1_box.config_port(
                        port=port.number,
                        lo_freq=lo,
                        cnco_freq=nco,
                        sideband="U",
                        vatt=readout_vatt[port.mux],
                    )
                    quel1_box.config_channel(
                        port=port.number,
                        channel=0,
                        fnco_freq=0,
                    )
                    quel1_box.config_rfswitch(
                        port=port.number,
                        rfswitch="block" if loopback else "pass",
                    )
                elif isinstance(port, ReadInPort):
                    lo, nco = ConfigUtils.find_read_lo_nco(
                        frequencies=read_frequencies,
                        qubits=port.read_out.read_qubits,
                    )
                    quel1_box.config_port(
                        port=port.number,
                        lo_freq=lo,
                        cnco_locked_with=port.read_out.number,
                    )
                    for idx in range(4):
                        quel1_box.config_runit(
                            port=port.number,
                            runit=idx,
                            fnco_freq=0,
                        )
                    quel1_box.config_rfswitch(
                        port=port.number,
                        rfswitch="loop" if loopback else "open",
                    )
                elif isinstance(port, CtrlPort):
                    lo, nco = ConfigUtils.find_ctrl_lo_nco(
                        frequencies=ctrl_frequencies,
                        qubit=port.ctrl_qubit,
                    )
                    quel1_box.config_port(
                        port=port.number,
                        lo_freq=lo,
                        cnco_freq=nco,
                        sideband="L",
                        vatt=control_vatt[port.ctrl_qubit],
                    )
                    if port.n_channel == 1:
                        quel1_box.config_channel(
                            port=port.number,
                            channel=0,
                            fnco_freq=0,
                        )
                    elif port.n_channel == 3:
                        quel1_box.config_channel(
                            port=port.number,
                            channel=0,
                            fnco_freq=0,
                        )
                        quel1_box.config_channel(
                            port=port.number,
                            channel=1,
                            fnco_freq=0,
                        )
                        quel1_box.config_channel(
                            port=port.number,
                            channel=2,
                            fnco_freq=0,
                        )
                    quel1_box.config_rfswitch(
                        port=port.number,
                        rfswitch="block" if loopback else "pass",
                    )

        # save the box settings
        box_settings_path = self.get_box_settings_path(chip_id)
        box_settings_path.mkdir(parents=True, exist_ok=True)
        qc.store_all_box_configs(box_settings_path)


class ConfigUtils:
    """
    ConfigUtils class provides utility methods for configuring the QubeX system.
    """

    @staticmethod
    def find_lo_nco_pair(
        target_frequency: float,
        ssb: str,
        *,
        lo_min: int = 8_000_000_000,
        lo_max: int = 10_500_000_000,
        lo_step: int = 500_000_000,
        nco_min: int = 1_500_000_000,
        nco_max: int = 1_992_187_500,
        nco_step: int = 23_437_500,
    ) -> tuple[int, int]:
        """
        Finds the pair (lo, nco) such that the value of lo ± nco is closest to the target_frequency.
        The operation depends on the value of 'ssb'. If 'ssb' is 'LSB', it uses lo - nco. If 'ssb' is 'USB', it uses lo + nco.

        Parameters
        ----------
        target_frequency : float
            The target frequency in GHz.
        ssb : str
            The sideband (either 'LSB' or 'USB').
        lo_min : int, optional
            The minimum value of lo, by default 8_000_000_000.
        lo_max : int, optional
            The maximum value of lo, by default 10_500_000_000.
        lo_step : int, optional
            The step value of lo, by default 500_000_000.
        nco_min : int, optional
            The minimum value of nco, by default 1_500_000_000.
        nco_max : int, optional
            The maximum value of nco, by default 1_992_187_500.
        nco_step : int, optional
            The step value of nco, by default 23_437_500.

        Returns
        -------
        tuple[int, int]
            The pair (lo, nco) that results in the closest value to target_frequency.
        """

        target_value = target_frequency * 1e9

        # Initialize the minimum difference to infinity to ensure any real difference is smaller.
        min_diff = float("inf")
        best_lo = None
        best_nco = None

        # Iterate over possible values of lo from lo_min to lo_max, in steps of lo_step.
        for lo in range(lo_min, lo_max + 1, lo_step):
            # Iterate over possible values of nco from nco_min to nco_max, in steps of nco_step.
            for nco in range(nco_min, nco_max + 1, nco_step):
                # Calculate the current value based on ssb.
                if ssb == "LSB":
                    current_value = lo - nco
                elif ssb == "USB":
                    current_value = lo + nco
                else:
                    raise ValueError("ssb must be 'LSB' or 'USB'")

                # Calculate the absolute difference from the target_frequency.
                current_diff = abs(current_value - target_value)

                # If this is the smallest difference we've found, update our best estimates.
                if current_diff < min_diff:
                    min_diff = current_diff
                    best_lo = lo
                    best_nco = nco

        if best_lo is None or best_nco is None:
            raise ValueError("No valid (lo, nco) pair found.")

        # Return the pair (lo, nco) that results in the closest value to target_frequency.
        return best_lo, best_nco

    @staticmethod
    def find_read_lo_nco(
        frequencies: dict[str, float],
        qubits: list[str],
    ) -> tuple[int, int]:
        """
        Finds the lo and nco values for the read frequencies.

        Parameters
        ----------
        frequencies : dict[str, float]
            The readout frequencies.
        qubits : list[str]
            The readout qubits.

        Returns
        -------
        tuple[int, int]
            The pair (lo, nco) for the read frequencies.
        """
        default_value = 0.0
        frequencies = defaultdict(lambda: default_value, frequencies)
        values = [frequencies[qubit] for qubit in qubits]
        median_value = (max(values) + min(values)) / 2
        return ConfigUtils.find_lo_nco_pair(median_value, ssb="USB")

    @staticmethod
    def find_ctrl_lo_nco(
        frequencies: dict[str, float],
        qubit: str,
    ) -> tuple[int, int]:
        """
        Finds the lo and nco values for the control frequencies.

        Parameters
        ----------
        frequencies : dict[str, float]
            The control frequencies.
        qubit : str
            The control qubit.

        Returns
        -------
        tuple[int, int]
            The pair (lo, nco) for the control frequencies.
        """
        default_value = 0.0
        frequencies = defaultdict(lambda: default_value, frequencies)
        return ConfigUtils.find_lo_nco_pair(frequencies[qubit], ssb="LSB")
