from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Final

import yaml
from pydantic.dataclasses import dataclass

from .model import Model
from .quantum_system import Chip, Mux, QuantumSystem, Qubit
from .control_system import Box, CapPort, GenPort, Port, ControlSystem


class TargetType(Enum):
    CTRL_GE = "CTRL_GE"
    CTRL_EF = "CTRL_EF"
    CTRL_CR = "CTRL_CR"
    READ = "READ"
    UNKNOWN = "UNKNOWN"


@dataclass
class Target(Model):
    label: str
    qubit: str
    type: TargetType
    frequency: float

    @classmethod
    def from_label(
        cls,
        label: str,
        frequency: float = 0.0,
    ) -> Target:
        if match := re.match(r"^R(Q\d+)$", label):
            qubit = match.group(1)
            type = TargetType.READ
        elif match := re.match(r"^(Q\d+)$", label):
            qubit = match.group(1)
            type = TargetType.CTRL_GE
        elif match := re.match(r"^(Q\d+)-ef$", label):
            qubit = match.group(1)
            type = TargetType.CTRL_EF
        elif match := re.match(r"^(Q\d+)-CR$", label):
            qubit = match.group(1)
            type = TargetType.CTRL_CR
        elif match := re.match(r"^(Q\d+)(-|_)[a-zA-Z0-9]+$", label):
            qubit = match.group(1)
            type = TargetType.UNKNOWN
        else:
            raise ValueError(f"Invalid target label `{label}`.")

        return cls(
            label=label,
            qubit=qubit,
            type=type,
            frequency=frequency,
        )

    @classmethod
    def get_target_type(cls, label: str) -> TargetType:
        target = cls.from_label(label)
        return target.type

    @classmethod
    def get_qubit_label(cls, label: str) -> str:
        target = cls.from_label(label)
        return target.qubit

    @classmethod
    def get_ge_label(cls, label: str) -> str:
        qubit = cls.get_qubit_label(label)
        return f"{qubit}"

    @classmethod
    def get_ef_label(cls, label: str) -> str:
        qubit = cls.get_qubit_label(label)
        return f"{qubit}-ef"

    @classmethod
    def get_cr_label(cls, label: str) -> str:
        qubit = cls.get_qubit_label(label)
        return f"{qubit}-CR"

    @classmethod
    def get_readout_label(cls, label: str) -> str:
        qubit = cls.get_qubit_label(label)
        return f"R{qubit}"

    @classmethod
    def is_ge_control(cls, label: str) -> bool:
        type = cls.get_target_type(label)
        return type == TargetType.CTRL_GE

    @classmethod
    def is_ef_control(cls, label: str) -> bool:
        type = cls.get_target_type(label)
        return type == TargetType.CTRL_EF

    @classmethod
    def is_cr_control(cls, label: str) -> bool:
        type = cls.get_target_type(label)
        return type == TargetType.CTRL_CR

    @classmethod
    def is_control(cls, label: str) -> bool:
        type = cls.get_target_type(label)
        return type != TargetType.READ

    @classmethod
    def is_readout(cls, label: str) -> bool:
        type = cls.get_target_type(label)
        return type == TargetType.READ


@dataclass
class WiringInfo(Model):
    ctrl: list[tuple[Qubit, GenPort]]
    read_out: list[tuple[Mux, GenPort]]
    read_in: list[tuple[Mux, CapPort]]


@dataclass
class PortSet:
    ctrl_port: Port
    read_in_port: Port
    read_out_port: Port


class ExperimentSystem:
    def __init__(
        self,
        quantum_system: QuantumSystem,
        control_system: ControlSystem,
        wiring_info: WiringInfo,
    ):
        self._quantum_system: Final = quantum_system
        self._control_system: Final = control_system
        self._wiring_info: Final = wiring_info
        # self._targets: Final = self._create_targets()
        # self._ctrl_channel_map: Final = self._create_ctrl_channel_map()
        # self._read_in_channel_map: Final = self._create_read_in_channel_map()
        # self._read_out_channel_map: Final = self._create_read_out_channel_map()
        # self._port_qubit_map: Final = self._create_port_qubit_map()
        # self._qubit_port_map: Final = self._create_qubit_port_map()
        # self._resonator_port_map: Final = self._create_resonator_port_map()
        # self._qubit_port_set_map: Final = self._create_qubit_port_set_map()

    @classmethod
    def load_from_config_files(cls, chip_id: str):
        with open(Path("config/chip.yaml"), "r") as file:
            chip_dict = yaml.safe_load(file)
        with open(Path("config/box.yaml"), "r") as file:
            box_dict = yaml.safe_load(file)
        with open(Path("config/wiring.yaml"), "r") as file:
            wiring_dict = yaml.safe_load(file)
        chip = Chip.new(
            id=chip_id,
            name=chip_dict[chip_id]["name"],
            n_qubits=chip_dict[chip_id]["n_qubits"],
        )
        quantum_system = QuantumSystem(chip=chip)
        boxes = [
            Box.new(
                id=id,
                name=box["name"],
                type=box["type"],
                address=box["address"],
                adapter=box["adapter"],
            )
            for id, box in box_dict.items()
        ]
        control_system = ControlSystem(boxes=boxes)

        def get_port(specifier: str):
            box_id = specifier.split("-")[0]
            port_num = int(specifier.split("-")[1])
            port = control_system.get_port(box_id, port_num)
            return port

        ctrl: list[tuple[Qubit, GenPort]] = []
        read_out: list[tuple[Mux, GenPort]] = []
        read_in: list[tuple[Mux, CapPort]] = []

        for wiring in wiring_dict[chip_id]:
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

        return cls(
            quantum_system=quantum_system,
            control_system=control_system,
            wiring_info=wiring_info,
        )

    def _create_targets(self) -> dict[str, Target]:
        targets = {}

        # control targets
        qubits = self.quantum_system.qubits
        for qubit in qubits.values():
            # ge
            ge_label = Target.get_ge_label(qubit.label)
            targets[ge_label] = Target(
                label=qubit.label,
                frequency=qubit.frequency,
                type=TargetType.CTRL_GE,
                qubit=qubit.label,
            )
            # ef
            ef_label = Target.get_ef_label(qubit.label)
            targets[ef_label] = Target(
                label=ef_label,
                frequency=qubit.frequency + qubit.anharmonicity,
                type=TargetType.CTRL_EF,
                qubit=qubit.label,
            )
            # cr
            spectators = self.quantum_system.chip.graph.get_spectators(qubit.label)
            cr_initial_frequency = sum(
                [qubits[spectator].frequency for spectator in spectators]
            ) / len(spectators)
            cr_label = Target.get_cr_label(qubit.label)
            targets[cr_label] = Target(
                label=cr_label,
                frequency=cr_initial_frequency,
                type=TargetType.CTRL_CR,
                qubit=qubit.label,
            )

        # readout targets
        resonators = self.quantum_system.resonators
        for resonator in resonators.values():
            targets[resonator.label] = Target(
                label=resonator.label,
                frequency=resonator.frequency,
                type=TargetType.READ,
                qubit=resonator.qubit,
            )

        return targets

    def _create_ctrl_channel_map(self) -> dict[str, Channel]:
        map = {}
        for mux in self.muxes:
            for qubit, port in zip(mux.qubits, mux.ctrl_ports):
                label = qubit.label
                n_channels = len(port.channels)
                if n_channels == 1:
                    ge = Target.get_ge_label(label)
                    map[ge] = port.channels[0]
                elif n_channels == 3:
                    ge = Target.get_ge_label(label)
                    ef = Target.get_ef_label(label)
                    cr = Target.get_cr_label(label)
                    map[ge] = port.channels[0]
                    map[ef] = port.channels[1]
                    map[cr] = port.channels[2]
                else:
                    raise ValueError("Invalid number of channels.")
        return map

    def _create_read_out_channel_map(self) -> dict[str, Channel]:
        map = {}
        for mux in self.muxes:
            port = mux.read_out_port
            for qubit in mux.qubits:
                label = Target.get_readout_label(qubit.label)
                map[label] = port.channels[0]
        return map

    def _create_read_in_channel_map(self) -> dict[str, Channel]:
        map = {}
        for mux in self.muxes:
            port = mux.read_in_port
            for index, qubit in enumerate(mux.qubits):
                label = Target.get_readout_label(qubit.label)
                map[label] = port.channels[index]
        return map

    def _create_port_qubit_map(self) -> dict[str, Qubit]:
        map = {}
        for mux in self.muxes:
            for qubit, port in zip(mux.qubits, mux.ctrl_ports):
                map[port.id] = qubit
        return map

    def _create_qubit_port_map(self) -> dict[str, Port]:
        map = {}
        for mux in self.muxes:
            for qubit, port in zip(mux.qubits, mux.ctrl_ports):
                map[qubit.label] = port
        return map

    def _create_resonator_port_map(self) -> dict[str, Port]:
        map = {}
        for mux in self.muxes:
            for qubit in mux.qubits:
                label = Target.get_readout_label(qubit.label)
                map[label] = mux.read_out_port
        return map

    def _create_qubit_port_set_map(self) -> dict[str, PortSet]:
        map = {}
        for mux in self.muxes:
            for qubit, ctrl_port in zip(mux.qubits, mux.ctrl_ports):
                map[qubit.label] = PortSet(
                    ctrl_port=ctrl_port,
                    read_in_port=mux.read_in_port,
                    read_out_port=mux.read_out_port,
                )
        return map

    @property
    def quantum_system(self) -> QuantumSystem:
        return self._quantum_system

    @property
    def qube_system(self) -> ControlSystem:
        return self._control_system

    @property
    def muxes(self) -> list[Wiring]:
        return self._muxes

    @property
    def targets(self) -> dict[str, Target]:
        return self._targets

    @property
    def ctrl_channel_map(self) -> dict[str, Channel]:
        return self._ctrl_channel_map

    @property
    def read_in_channel_map(self) -> dict[str, Channel]:
        return self._read_in_channel_map

    @property
    def read_out_channel_map(self) -> dict[str, Channel]:
        return self._read_out_channel_map

    @property
    def port_qubit_map(self) -> dict[str, Qubit]:
        return self._port_qubit_map

    @property
    def qubit_port_map(self) -> dict[str, Port]:
        return self._qubit_port_map

    @property
    def resonator_port_map(self) -> dict[str, Port]:
        return self._resonator_port_map

    @property
    def qubit_port_set_map(self) -> dict[str, PortSet]:
        return self._qubit_port_set_map

    @property
    def ge_targets(self) -> dict[str, Target]:
        ge_labels = [
            Target.get_ge_label(qubit.label)
            for qubit in self.quantum_system.qubits.values()
        ]
        return {label: self.get_ge_target(label) for label in ge_labels}

    @property
    def ef_targets(self) -> dict[str, Target]:
        ef_labels = [
            Target.get_ef_label(qubit.label)
            for qubit in self.quantum_system.qubits.values()
        ]
        return {label: self.get_ef_target(label) for label in ef_labels}

    @property
    def cr_targets(self) -> dict[str, Target]:
        cr_labels = [
            Target.get_cr_label(qubit.label)
            for qubit in self.quantum_system.qubits.values()
        ]
        return {label: self.get_cr_target(label) for label in cr_labels}

    @property
    def control_targets(self) -> dict[str, Target]:
        return self.ge_targets | self.ef_targets | self.cr_targets

    @property
    def readout_targets(self) -> dict[str, Target]:
        return {label: self.targets[label] for label in self.quantum_system.resonators}

    def get_target(self, label: str) -> Target:
        try:
            return self._targets[label]
        except KeyError:
            raise KeyError(f"Target `{label}` not found.")

    def get_ge_target(self, label: str) -> Target:
        label = Target.get_ge_label(label)
        return self.get_target(label)

    def get_ef_target(self, label: str) -> Target:
        label = Target.get_ef_label(label)
        return self.get_target(label)

    def get_cr_target(self, label: str) -> Target:
        label = Target.get_cr_label(label)
        return self.get_target(label)

    def get_readout_target(self, label: str) -> Target:
        label = Target.get_readout_label(label)
        return self.get_target(label)

    def get_mux_by_port(self, port: Port) -> Wiring:
        for mux in self.muxes:
            if port in mux.ctrl_ports or port in (mux.read_in_port, mux.read_out_port):
                return mux
        raise ValueError(f"Port `{port}` not found in any MUX.")
