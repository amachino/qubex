"""
Config module provides classes and functions to configure the QubeX system.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml
from qubecalib import QubeCalib


class BoxType(Enum):
    QUEL1_A = "quel1-a"
    QUEL1_B = "quel1-b"
    QUBE_RIKEN_A = "qube-riken-a"
    QUBE_RIKEN_B = "qube-riken-b"
    QUBE_OU_A = "qube-ou-a"
    QUBE_OU_B = "qube-ou-b"


class PortType(Enum):
    NOT_AVAILABLE = "N/A"
    READ0_IN = "READ0.IN"
    READ0_OUT = "READ0.OUT"
    READ1_IN = "READ1.IN"
    READ1_OUT = "READ1.OUT"
    CTRL0 = "CTRL0"
    CTRL1 = "CTRL1"
    CTRL2 = "CTRL2"
    CTRL3 = "CTRL3"
    CTRL4 = "CTRL4"
    CTRL5 = "CTRL5"
    CTRL6 = "CTRL6"
    CTRL7 = "CTRL7"
    PUMP0 = "PUMP0"
    PUMP1 = "PUMP1"
    MONITOR0_IN = "MONITOR0.IN"
    MONITOR0_OUT = "MONITOR0.OUT"
    MONITOR1_IN = "MONITOR1.IN"
    MONITOR1_OUT = "MONITOR1.OUT"


PORT_MAPPING = {
    BoxType.QUEL1_A: {
        0: PortType.READ0_IN,
        1: PortType.READ0_OUT,
        2: PortType.CTRL0,
        3: PortType.PUMP0,
        4: PortType.CTRL1,
        5: PortType.MONITOR0_IN,
        6: PortType.MONITOR0_OUT,
        7: PortType.READ1_IN,
        8: PortType.READ1_OUT,
        9: PortType.CTRL2,
        10: PortType.PUMP1,
        11: PortType.CTRL3,
        12: PortType.MONITOR1_IN,
        13: PortType.MONITOR1_OUT,
    },
    BoxType.QUEL1_B: {
        0: PortType.NOT_AVAILABLE,
        1: PortType.CTRL0,
        2: PortType.CTRL1,
        3: PortType.CTRL2,
        4: PortType.CTRL3,
        5: PortType.MONITOR0_IN,
        6: PortType.MONITOR0_OUT,
        7: PortType.NOT_AVAILABLE,
        8: PortType.CTRL4,
        9: PortType.CTRL5,
        10: PortType.CTRL6,
        11: PortType.CTRL7,
        12: PortType.MONITOR1_IN,
        13: PortType.MONITOR1_OUT,
    },
    BoxType.QUBE_RIKEN_A: {
        0: PortType.READ0_OUT,
        1: PortType.READ0_IN,
        2: PortType.PUMP0,
        3: PortType.MONITOR0_OUT,
        4: PortType.MONITOR0_IN,
        5: PortType.CTRL0,
        6: PortType.CTRL1,
        7: PortType.CTRL2,
        8: PortType.CTRL3,
        9: PortType.MONITOR1_IN,
        10: PortType.MONITOR1_OUT,
        11: PortType.PUMP1,
        12: PortType.READ1_IN,
        13: PortType.READ1_OUT,
    },
    BoxType.QUBE_RIKEN_B: {
        0: PortType.CTRL0,
        1: PortType.NOT_AVAILABLE,
        2: PortType.CTRL1,
        3: PortType.MONITOR0_OUT,
        4: PortType.MONITOR0_IN,
        5: PortType.CTRL2,
        6: PortType.CTRL3,
        7: PortType.CTRL4,
        8: PortType.CTRL5,
        9: PortType.MONITOR1_IN,
        10: PortType.MONITOR1_OUT,
        11: PortType.CTRL6,
        12: PortType.NOT_AVAILABLE,
        13: PortType.CTRL7,
    },
    BoxType.QUBE_OU_A: {
        0: PortType.READ0_OUT,
        1: PortType.READ0_IN,
        2: PortType.PUMP0,
        3: PortType.NOT_AVAILABLE,
        4: PortType.NOT_AVAILABLE,
        5: PortType.CTRL0,
        6: PortType.CTRL1,
        7: PortType.CTRL2,
        8: PortType.CTRL3,
        9: PortType.NOT_AVAILABLE,
        10: PortType.NOT_AVAILABLE,
        11: PortType.PUMP1,
        12: PortType.READ1_IN,
        13: PortType.READ1_OUT,
    },
    BoxType.QUBE_OU_B: {
        0: PortType.CTRL0,
        1: PortType.NOT_AVAILABLE,
        2: PortType.CTRL1,
        3: PortType.NOT_AVAILABLE,
        4: PortType.NOT_AVAILABLE,
        5: PortType.CTRL2,
        6: PortType.CTRL3,
        7: PortType.CTRL4,
        8: PortType.CTRL5,
        9: PortType.NOT_AVAILABLE,
        10: PortType.NOT_AVAILABLE,
        11: PortType.CTRL6,
        12: PortType.NOT_AVAILABLE,
        13: PortType.CTRL7,
    },
}

NUMBER_OF_CHANNELS = {
    BoxType.QUEL1_A: {
        2: 3,
        4: 3,
        9: 3,
        11: 3,
    },
    BoxType.QUEL1_B: {
        1: 1,
        2: 3,
        3: 1,
        4: 3,
        8: 1,
        9: 3,
        10: 1,
        11: 3,
    },
    BoxType.QUBE_RIKEN_A: {
        5: 3,
        6: 3,
        7: 3,
        8: 3,
    },
    BoxType.QUBE_RIKEN_B: {
        0: 1,
        2: 1,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        11: 1,
        13: 1,
    },
    BoxType.QUBE_OU_A: {
        5: 3,
        6: 3,
        7: 3,
        8: 3,
    },
    BoxType.QUBE_OU_B: {
        0: 1,
        2: 1,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        11: 1,
        13: 1,
    },
}

READOUT_PAIRS = {
    BoxType.QUEL1_A: {
        0: 1,
        7: 8,
    },
    BoxType.QUEL1_B: {},
    BoxType.QUBE_RIKEN_A: {
        1: 0,
        12: 13,
    },
    BoxType.QUBE_RIKEN_B: {},
    BoxType.QUBE_OU_A: {
        1: 0,
        12: 13,
    },
    BoxType.QUBE_OU_B: {},
}


@dataclass
class Box:
    id: str
    name: str
    type: BoxType
    address: str
    adapter: str


@dataclass
class Port:
    number: int
    box: Box

    @property
    def type(self) -> PortType:
        return PORT_MAPPING[self.box.type][self.number]

    @property
    def name(self) -> str:
        return f"{self.box.id}.{self.type.value}"


@dataclass
class ReadInPort(Port):
    number: int
    box: Box
    read_out: ReadOutPort


@dataclass
class ReadOutPort(Port):
    number: int
    box: Box
    mux: int

    @property
    def read_qubits(self) -> list[str]:
        return [f"Q{4 * self.mux + i:02d}" for i in range(4)]


@dataclass
class CtrlPort(Port):
    number: int
    box: Box
    ctrl_qubit: str

    @property
    def n_channel(self) -> int:
        return NUMBER_OF_CHANNELS[self.box.type][self.number]


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


class Config:
    """
    Config class provides methods to configure the QubeX system.
    """

    N_RUNITS = 4
    N_PER_MUX = 4

    def __init__(
        self,
        package_id: str,
        *,
        config_dir: str = "./config",
        box_file: str = "box.yaml",
        wiring_file: str = "wiring.yaml",
        props_file: str = "props.yaml",
        params_file: str = "params.yaml",
        system_settings_file: str = "system_settings.json",
        box_settings_file: str = "box_settings.json",
    ):
        """
        Initializes the Config object.

        Parameters
        ----------
        package_id : str
            The package ID of a quantum chip (e.g., "64Q").
        config_dir : str, optional
            The directory where the configuration files are stored, by default "./config".
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
        """
        self._package_id = package_id
        self._config_dir = config_dir
        self._system_settings_file = system_settings_file
        self._box_settings_file = box_settings_file
        self._box_dict = self._load_config_file(box_file)
        self._wiring_dict = self._load_config_file(wiring_file)
        self._props_dict = self._load_config_file(props_file)
        self._params_dict = self._load_config_file(params_file)

    @property
    def system_settings_path(self) -> Path:
        return Path(self._config_dir) / self._system_settings_file

    @property
    def box_settings_path(self) -> Path:
        return Path(self._config_dir) / self._box_settings_file

    def _load_config_file(self, file_name) -> dict:
        path = Path(self._config_dir) / file_name
        with open(path, "r") as file:
            result = yaml.safe_load(file)
        return result

    @property
    def props(self) -> Props:
        """
        Returns the properties of the quantum chip.

        Returns
        -------
        Props
            The properties of the quantum chip.
        """
        props = self._props_dict[self._package_id]
        return Props(
            resonator_frequency=props["resonator_frequency"],
            qubit_frequency=props["qubit_frequency"],
            anharmonicity=props["anharmonicity"],
        )

    @property
    def params(self) -> Params:
        """
        Returns the parameters of the quantum chip.

        Returns
        -------
        Params
            The parameters of the quantum chip.
        """
        params = self._params_dict[self._package_id]
        return Params(
            control_amplitude=params["control_amplitude"],
            readout_amplitude=params["readout_amplitude"],
            control_vatt=params["control_vatt"],
            readout_vatt=params["readout_vatt"],
        )

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

    def get_boxes(self) -> list[Box]:
        """
        Returns an available list of Box objects.

        Returns
        -------
        list[Box]
            A list of Box objects.
        """
        return [self.get_box(id) for id in self._box_dict.keys()]

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
        return [f"Q{4 * mux + i:02d}" for i in range(self.N_PER_MUX)]

    def get_port_configuration(self) -> list[Port]:
        """
        Returns a list of Port objects based on the wiring configuration.

        Returns
        -------
        list[Port]
            A list of Port objects based on the wiring configuration.
        """
        wiring_list = self._wiring_dict[self._package_id]
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

    def configure_system_settings(self):
        """
        Configures and saves the system settings to a JSON file.
        """
        qc = QubeCalib()

        # define boxes, ports, and channels
        boxes = self.get_boxes()
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
        ports = self.get_port_configuration()
        for port in ports:
            if isinstance(port, ReadOutPort):
                qc.define_channel(
                    channel_name=f"{port.name}0",
                    port_name=port.name,
                    channel_number=0,
                )
            elif isinstance(port, ReadInPort):
                read_qubits = port.read_out.read_qubits
                for runit_index in range(self.N_RUNITS):
                    qc.define_channel(
                        channel_name=f"{port.name}{runit_index}",
                        port_name=port.name,
                        channel_number=runit_index,
                    )
                    qc.define_target(
                        target_name=f"R{read_qubits[runit_index]}",
                        channel_name=f"{port.name}{runit_index}",
                        target_frequency=0.0,
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
                qubits = port.read_qubits
                for qubit in qubits:
                    target_frequency = self.props.resonator_frequency[qubit]
                    qc.define_target(
                        target_name=f"R{qubit}",
                        channel_name=f"{port.name}0",
                        target_frequency=target_frequency,
                    )
            elif isinstance(port, ReadInPort):
                qubits = port.read_out.read_qubits
                for idx, qubit in enumerate(qubits):
                    target_frequency = self.props.resonator_frequency[qubit]
                    qc.define_target(
                        target_name=f"R{qubit}",
                        channel_name=f"{port.name}{idx}",
                        target_frequency=target_frequency,
                    )
            elif isinstance(port, CtrlPort):
                qubit = port.ctrl_qubit
                target_name = qubit
                target_frequency = self.props.qubit_frequency[qubit]
                qc.define_target(
                    target_name=target_name,
                    channel_name=f"{port.name}.CH{idx}",
                    target_frequency=target_frequency,
                )

        # save the system settings to a JSON file
        with open(self.system_settings_path, "w") as f:
            f.write(qc.system_config_database.asjson())
        print(f"System setting saved to {self.system_settings_path}")

    def configure_box_settings(
        self,
        *,
        box_list: list[str] | None = None,
        loopback: bool = False,
    ):
        """
        Configures the box with the given ID.

        Parameters
        ----------
        box_list : list[str] | None, optional
            The list of box IDs to configure, by default None.
        loopback : bool, optional
            Whether to enable loopback mode, by default False.
        """
        qc = QubeCalib(self._system_settings_file)
        boxes = (
            self.get_boxes()
            if box_list is None
            else [self.get_box(id) for id in box_list]
        )
        ports = self.get_port_configuration()
        read_frequencies = self.props.resonator_frequency
        ctrl_frequencies = self.props.qubit_frequency
        readout_vatt = self.params.readout_vatt
        control_vatt = self.params.control_vatt

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
                    for idx in range(self.N_RUNITS):
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
        qc.store_all_box_configs(self.box_settings_path)


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
