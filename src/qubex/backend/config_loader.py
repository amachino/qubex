from __future__ import annotations

from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Final, Literal

import yaml

from .control_system import Box, CapPort, ControlSystem, GenPort
from .experiment_system import ControlParams, ExperimentSystem, WiringInfo
from .quantum_system import Chip, QuantumSystem

logger = getLogger(__name__)

DEFAULT_CONFIG_DIR: Final = "/home/shared/qubex-config"

CHIP_FILE: Final = "chip.yaml"
BOX_FILE: Final = "box.yaml"
WIRING_FILE: Final = "wiring.yaml"
PROPS_FILE: Final = "props.yaml"
PARAMS_FILE: Final = "params.yaml"


class ConfigLoader:
    def __init__(
        self,
        *,
        chip_id: str,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        chip_file: str = CHIP_FILE,
        box_file: str = BOX_FILE,
        wiring_file: str = WIRING_FILE,
        props_file: str = PROPS_FILE,
        params_file: str = PARAMS_FILE,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
    ):
        """
        Initializes the ConfigLoader object.

        Parameters
        ----------
        chip_id : str, optional
            The quantum chip ID (e.g., "64Q"). If provided, the configuration will be loaded for this specific chip.
        config_dir : Path | str, optional
            The directory where the configuration files are stored.
        params_dir : Path | str, optional
            The directory where the parameter files are stored.
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
        targets_to_exclude : list[str], optional
            The list of target labels to exclude, by default None.

        Examples
        --------
        >>> config = ConfigLoader()
        """
        if config_dir is None:
            config_dir = Path(DEFAULT_CONFIG_DIR) / chip_id / "config"
        if params_dir is None:
            params_dir = Path(DEFAULT_CONFIG_DIR) / chip_id / "params"
        if configuration_mode is None:
            configuration_mode = "ge-cr-cr"
        self._chip_id = chip_id
        self._config_dir = config_dir
        self._params_dir = params_dir
        self._chip_dict = self._load_config_file(chip_file)
        self._box_dict = self._load_config_file(box_file)
        self._wiring_dict = self._load_config_file(wiring_file)
        self._props_dict = self._load_params_file(props_file)
        self._params_dict = self._load_params_file(params_file)
        self._quantum_system_dict = self._load_quantum_system()
        self._control_system_dict = self._load_control_system()
        self._wiring_info_dict = self._load_wiring_info()
        self._control_params_dict = self._load_control_params()
        self._experiment_system_dict = self._load_experiment_system(
            targets_to_exclude=targets_to_exclude,
            configuration_mode=configuration_mode,
        )

    @property
    def config_path(self) -> Path:
        """Returns the absolute path to the configuration directory."""
        return Path(self._config_dir).resolve()

    @property
    def params_path(self) -> Path:
        """Returns the absolute path to the parameters directory."""
        return Path(self._params_dir).resolve()

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
        except FileNotFoundError as e:
            print(f"Configuration file not found: {path}\n\n{e}")
            raise e
        except yaml.YAMLError as e:
            print(f"Error loading configuration file: {path}\n\n{e}")
            raise e
        return result

    def _load_params_file(self, file_name) -> dict:
        path = Path(self._params_dir) / file_name
        try:
            with open(path, "r") as file:
                result = yaml.safe_load(file)
        except FileNotFoundError as e:
            print(f"Parameter file not found: {path}\n\n{e}")
            raise e
        except yaml.YAMLError as e:
            print(f"Error loading parameter file: {path}\n\n{e}")
            raise e
        return result

    def _load_quantum_system(self) -> dict[str, QuantumSystem]:
        quantum_system_dict = {}
        for chip_id, chip_info in self._chip_dict.items():
            if self._chip_id is not None and chip_id != self._chip_id:
                continue
            chip = Chip.new(
                id=chip_id,
                name=chip_info["name"],
                n_qubits=chip_info["n_qubits"],
            )
            props = self._props_dict.get(chip_id)
            if props is None:
                logger.warning(f"Chip `{chip_id}` is missing in `{PROPS_FILE}`. ")
                continue
            qubit_frequency_dict = props.get("qubit_frequency", {})
            qubit_anharmonicity_dict = props.get("anharmonicity", {})
            resonator_frequency_dict = props.get("resonator_frequency", {})
            for qubit in chip.qubits:
                qubit.frequency = qubit_frequency_dict.get(
                    qubit.label, float("nan")
                ) or float("nan")
                anharmonicity = qubit_anharmonicity_dict.get(qubit.label)
                if anharmonicity is None:
                    factor = -1 / 19  # E_J / E_C = 50
                    qubit.anharmonicity = qubit.frequency * factor
                else:
                    qubit.anharmonicity = anharmonicity
            for resonator in chip.resonators:
                resonator.frequency = resonator_frequency_dict.get(
                    resonator.qubit, float("nan")
                ) or float("nan")
            quantum_system = QuantumSystem(chip=chip)
            quantum_system_dict[chip_id] = quantum_system
        return quantum_system_dict

    def _load_control_system(self) -> dict[str, ControlSystem]:
        control_system_dict = {}
        for chip_id in self._chip_dict:
            if self._chip_id is not None and chip_id != self._chip_id:
                continue
            box_ports = defaultdict(list)
            wirings = self._wiring_dict.get(chip_id)
            if wirings is None:
                logger.warning(f"Chip `{chip_id}` is missing in `{WIRING_FILE}`. ")
                continue
            for wiring in wirings:
                box, port = wiring["read_out"].split("-")
                box_ports[box].append(int(port))
                box, port = wiring["read_in"].split("-")
                box_ports[box].append(int(port))
                for ctrl in wiring["ctrl"]:
                    box, port = ctrl.split("-")
                    box_ports[box].append(int(port))
            boxes = [
                Box.new(
                    id=id,
                    name=box["name"],
                    type=box["type"],
                    address=box["address"],
                    adapter=box["adapter"],
                    port_numbers=box_ports[id],
                )
                for id, box in self._box_dict.items()
                if id in box_ports
            ]
            control_system_dict[chip_id] = ControlSystem(
                boxes=boxes,
                clock_master_address=self._chip_dict[chip_id].get("clock_master"),
            )
        return control_system_dict

    def _load_wiring_info(self) -> dict[str, WiringInfo]:
        wiring_info_dict = {}
        for chip_id in self._chip_dict:
            if self._chip_id is not None and chip_id != self._chip_id:
                continue
            try:
                wirings = self._wiring_dict[chip_id]
                quantum_system = self._quantum_system_dict[chip_id]
                control_system = self._control_system_dict[chip_id]
            except KeyError:
                continue

            def get_port(specifier: str | None):
                if specifier is None:
                    return None
                box_id = specifier.split("-")[0]
                port_num = int(specifier.split("-")[1])
                port = control_system.get_port(box_id, port_num)
                return port

            ctrl = []
            read_out = []
            read_in = []
            pump = []
            for wiring in wirings:
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
                pump_port: GenPort = get_port(wiring.get("pump"))  # type: ignore
                if pump_port is not None:
                    pump.append((mux, pump_port))

            wiring_info = WiringInfo(
                ctrl=ctrl,
                read_out=read_out,
                read_in=read_in,
                pump=pump,
            )
            wiring_info_dict[chip_id] = wiring_info
        return wiring_info_dict

    def _load_control_params(self) -> dict[str, ControlParams]:
        control_params_dict = {}
        for chip_id in self._chip_dict:
            if self._chip_id is not None and chip_id != self._chip_id:
                continue
            params = self._params_dict.get(chip_id)
            if params is None:
                logger.warning(f"Chip `{chip_id}` is missing in `{PARAMS_FILE}`. ")
                continue
            control_params = ControlParams(
                control_amplitude=params.get("control_amplitude", {}),
                readout_amplitude=params.get("readout_amplitude", {}),
                control_vatt=params.get("control_vatt", {}),
                readout_vatt=params.get("readout_vatt", {}),
                pump_vatt=params.get("pump_vatt", {}),
                control_fsc=params.get("control_fsc", {}),
                readout_fsc=params.get("readout_fsc", {}),
                pump_fsc=params.get("pump_fsc", {}),
                capture_delay=params.get("capture_delay", {}),
                capture_delay_word=params.get("capture_delay_word", {}),
                jpa_params=params.get("jpa_params", {}),
            )
            control_params_dict[chip_id] = control_params
        return control_params_dict

    def _load_experiment_system(
        self,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
    ) -> dict[str, ExperimentSystem]:
        experiment_system_dict = {}
        for chip_id in self._chip_dict:
            if self._chip_id is not None and chip_id != self._chip_id:
                continue
            quantum_system = self._quantum_system_dict[chip_id]
            control_system = self._control_system_dict[chip_id]
            wiring_info = self._wiring_info_dict[chip_id]
            control_params = self._control_params_dict[chip_id]
            experiment_system = ExperimentSystem(
                quantum_system=quantum_system,
                control_system=control_system,
                wiring_info=wiring_info,
                control_params=control_params,
                targets_to_exclude=targets_to_exclude,
                configuration_mode=configuration_mode,
            )
            experiment_system_dict[chip_id] = experiment_system
        return experiment_system_dict
