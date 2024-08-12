from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Final

import yaml
from pydantic.dataclasses import dataclass

from .control_system import Box, CapPort, ControlSystem, GenPort
from .model import Model
from .quantum_system import Chip, Mux, QuantumSystem, Qubit, Resonator


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
    def ge_target(
        cls,
        label: str,
        frequency: float,
    ) -> Target:
        return cls(
            label=Target.ge_label(label),
            qubit=Target.qubit_label(label),
            type=TargetType.CTRL_GE,
            frequency=frequency,
        )

    @classmethod
    def ef_target(
        cls,
        label: str,
        frequency: float,
    ) -> Target:
        return cls(
            label=Target.ef_label(label),
            qubit=Target.qubit_label(label),
            type=TargetType.CTRL_EF,
            frequency=frequency,
        )

    @classmethod
    def cr_target(
        cls,
        label: str,
        frequency: float,
    ) -> Target:
        return cls(
            label=Target.cr_label(label),
            qubit=Target.qubit_label(label),
            type=TargetType.CTRL_CR,
            frequency=frequency,
        )

    @classmethod
    def readout_target(
        cls,
        label: str,
        frequency: float,
    ) -> Target:
        return cls(
            label=Target.readout_label(label),
            qubit=Target.qubit_label(label),
            type=TargetType.READ,
            frequency=frequency,
        )

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
    def target_type(cls, label: str) -> TargetType:
        target = cls.from_label(label)
        return target.type

    @classmethod
    def qubit_label(cls, label: str) -> str:
        target = cls.from_label(label)
        return target.qubit

    @classmethod
    def ge_label(cls, label: str) -> str:
        qubit = cls.qubit_label(label)
        return f"{qubit}"

    @classmethod
    def ef_label(cls, label: str) -> str:
        qubit = cls.qubit_label(label)
        return f"{qubit}-ef"

    @classmethod
    def cr_label(cls, label: str) -> str:
        qubit = cls.qubit_label(label)
        return f"{qubit}-CR"

    @classmethod
    def readout_label(cls, label: str) -> str:
        qubit = cls.qubit_label(label)
        return f"R{qubit}"

    @classmethod
    def is_ge_control(cls, label: str) -> bool:
        type = cls.target_type(label)
        return type == TargetType.CTRL_GE

    @classmethod
    def is_ef_control(cls, label: str) -> bool:
        type = cls.target_type(label)
        return type == TargetType.CTRL_EF

    @classmethod
    def is_cr_control(cls, label: str) -> bool:
        type = cls.target_type(label)
        return type == TargetType.CTRL_CR

    @classmethod
    def is_control(cls, label: str) -> bool:
        type = cls.target_type(label)
        return type != TargetType.READ

    @classmethod
    def is_readout(cls, label: str) -> bool:
        type = cls.target_type(label)
        return type == TargetType.READ


@dataclass
class WiringInfo(Model):
    ctrl: list[tuple[Qubit, GenPort]]
    read_out: list[tuple[Mux, GenPort]]
    read_in: list[tuple[Mux, CapPort]]


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
        self._ge_target_dict: Final = self._create_ge_target_dict()
        self._ef_target_dict: Final = self._create_ef_target_dict()
        self._cr_target_dict: Final = self._create_cr_target_dict()
        self._readout_target_dict: Final = self._create_readout_target_dict()
        self._target_dict: Final = (
            self._ge_target_dict
            | self._ef_target_dict
            | self._cr_target_dict
            | self._readout_target_dict
        )

    @classmethod
    def load_from_config_files(cls, chip_id: str):
        with open(Path("config/chip.yaml"), "r") as file:
            chip_dict = yaml.safe_load(file)
        with open(Path("config/props.yaml"), "r") as file:
            props_dict = yaml.safe_load(file)
        with open(Path("config/box.yaml"), "r") as file:
            box_dict = yaml.safe_load(file)
        with open(Path("config/wiring.yaml"), "r") as file:
            wiring_dict = yaml.safe_load(file)
        chip = Chip.new(
            id=chip_id,
            name=chip_dict[chip_id]["name"],
            n_qubits=chip_dict[chip_id]["n_qubits"],
        )
        props = props_dict[chip_id]
        for qubit in chip.qubits:
            qubit.frequency = props["qubit_frequency"][qubit.label]
            qubit.anharmonicity = props["anharmonicity"][qubit.label]
        for resonator in chip.resonators:
            resonator.frequency = props["resonator_frequency"][resonator.qubit]
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

    def _create_ge_target_dict(self) -> dict[str, Target]:
        targets = [
            Target.ge_target(
                label=qubit.label,
                frequency=qubit.ge_frequency,
            )
            for qubit in self.qubits
        ]
        return {target.label: target for target in targets}

    def _create_ef_target_dict(self) -> dict[str, Target]:
        targets = [
            Target.ef_target(
                label=qubit.label,
                frequency=qubit.ef_frequency,
            )
            for qubit in self.qubits
        ]
        return {target.label: target for target in targets}

    def _create_cr_target_dict(self) -> dict[str, Target]:
        targets = [
            Target.cr_target(
                label=qubit.label,
                frequency=sum(
                    [
                        spectator.ge_frequency
                        for spectator in self.get_spectator_qubits(qubit.label)
                    ]
                )
                / len(self.get_spectator_qubits(qubit.label)),
            )
            for qubit in self.qubits
        ]
        return {target.label: target for target in targets}

    def _create_readout_target_dict(self) -> dict[str, Target]:
        targets = [
            Target.readout_target(
                label=resonator.label,
                frequency=resonator.frequency,
            )
            for resonator in self.resonators
        ]
        return {target.label: target for target in targets}

    @property
    def quantum_system(self) -> QuantumSystem:
        return self._quantum_system

    @property
    def qube_system(self) -> ControlSystem:
        return self._control_system

    @property
    def wiring_info(self) -> WiringInfo:
        return self._wiring_info

    @property
    def qubits(self) -> list[Qubit]:
        return self.quantum_system.qubits

    @property
    def resonators(self) -> list[Resonator]:
        return self.quantum_system.resonators

    @property
    def ge_targets(self) -> list[Target]:
        return list(self._ge_target_dict.values())

    @property
    def ef_targets(self) -> list[Target]:
        return list(self._ef_target_dict.values())

    @property
    def cr_targets(self) -> list[Target]:
        return list(self._cr_target_dict.values())

    @property
    def control_targets(self) -> list[Target]:
        return self.ge_targets + self.ef_targets + self.cr_targets

    @property
    def readout_targets(self) -> list[Target]:
        return list(self._readout_target_dict.values())

    @property
    def targets(self) -> list[Target]:
        return (
            self.ge_targets + self.ef_targets + self.cr_targets + self.readout_targets
        )

    def get_spectator_qubits(self, qubit: int | str) -> list[Qubit]:
        return self.quantum_system.get_spectator_qubits(qubit)

    def get_target(self, label: str) -> Target:
        try:
            return self._target_dict[label]
        except KeyError:
            raise KeyError(f"Target `{label}` not found.")

    def get_ge_target(self, label: str) -> Target:
        label = Target.ge_label(label)
        return self.get_target(label)

    def get_ef_target(self, label: str) -> Target:
        label = Target.ef_label(label)
        return self.get_target(label)

    def get_cr_target(self, label: str) -> Target:
        label = Target.cr_label(label)
        return self.get_target(label)

    def get_readout_target(self, label: str) -> Target:
        label = Target.readout_label(label)
        return self.get_target(label)
