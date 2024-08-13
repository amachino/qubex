from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from .control_system import Box, CapPort, ControlSystem, GenPort
from .experiment_system import ControlParams, ExperimentSystem, WiringInfo
from .quantum_system import Chip, QuantumSystem

CONFIG_DIR: Final = "config"
CHIP_FILE: Final = "chip.yaml"
BOX_FILE: Final = "box.yaml"
WIRING_FILE: Final = "wiring.yaml"
PROPS_FILE: Final = "props.yaml"
PARAMS_FILE: Final = "params.yaml"


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
    ):
        """
        Initializes the ConfigLoader object.

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

        Examples
        --------
        >>> config = ConfigLoader()
        """
        self._config_dir = config_dir
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
                control_amplitude=params.get("control_amplitude", {}),
                readout_amplitude=params.get("readout_amplitude", {}),
                control_vatt=params.get("control_vatt", {}),
                readout_vatt=params.get("readout_vatt", {}),
                control_fsc=params.get("control_fsc", {}),
                readout_fsc=params.get("readout_fsc", {}),
                capture_delay=params.get("capture_delay", {}),
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
