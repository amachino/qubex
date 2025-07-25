from __future__ import annotations

import re
from enum import Enum
from typing import Collection, Literal, Union

from pydantic.dataclasses import dataclass

from .control_system import CapChannel, GenChannel
from .model import Model
from .quantum_system import Mux, Qubit, Resonator


class TargetType(Enum):
    CTRL_GE = "CTRL_GE"
    CTRL_EF = "CTRL_EF"
    CTRL_CR = "CTRL_CR"
    READ = "READ"
    PUMP = "PUMP"
    UNKNOWN = "UNKNOWN"


PhysicalObject = Union[Qubit, Resonator, Mux]


@dataclass
class CapTarget(Model):
    label: str
    frequency: float
    object: PhysicalObject
    channel: CapChannel
    type: TargetType

    @classmethod
    def new_read_target(
        cls,
        *,
        resonator: Resonator,
        channel: CapChannel,
    ) -> CapTarget:
        return cls(
            label=Target.read_label(resonator.label),
            object=resonator,
            frequency=resonator.frequency,
            channel=channel,
            type=TargetType.READ,
        )


@dataclass
class Target(Model):
    label: str
    frequency: float
    object: PhysicalObject
    channel: GenChannel
    type: TargetType

    def __repr__(self) -> str:
        return f"Target(label={self.label}, frequency={self.frequency}, channel={self.channel.id}, object={self.object.label})"

    @property
    def qubit(self) -> str:
        if isinstance(self.object, Qubit):
            return self.object.label
        elif isinstance(self.object, Resonator):
            return self.object.qubit
        elif isinstance(self.object, Mux):
            return ""
        else:
            raise ValueError("Invalid quantum object.")

    @property
    def sideband(self) -> Literal["U", "L"] | None:
        return self.channel.port.sideband

    @property
    def coarse_frequency(self) -> float:
        if isinstance(self.channel, GenChannel):
            return self.channel.coarse_frequency * 1e-9
        else:
            raise ValueError("Invalid channel.")

    @property
    def fine_frequency(self) -> float:
        if isinstance(self.channel, GenChannel):
            return self.channel.fine_frequency * 1e-9
        else:
            raise ValueError("Invalid channel.")

    @property
    def is_available(self) -> bool:
        return abs(self.frequency - self.fine_frequency) < 250 * 1e-3  # 250 MHz

    @property
    def is_ge(self) -> bool:
        return self.type == TargetType.CTRL_GE

    @property
    def is_ef(self) -> bool:
        return self.type == TargetType.CTRL_EF

    @property
    def is_cr(self) -> bool:
        return self.type == TargetType.CTRL_CR

    @property
    def is_read(self) -> bool:
        return self.type == TargetType.READ

    @property
    def is_pump(self) -> bool:
        return self.type == TargetType.PUMP

    def is_related_to_qubits(self, qubits: Collection[str]) -> bool:
        if isinstance(self.object, Qubit):
            return self.object.label in qubits
        elif isinstance(self.object, Resonator):
            return self.object.qubit in qubits
        elif isinstance(self.object, Mux):
            return any(
                qubit in qubits
                for qubit in [resonator.qubit for resonator in self.object.resonators]
            )
        else:
            raise ValueError("Invalid quantum object.")

    @classmethod
    def new_target(
        cls,
        *,
        label: str,
        frequency: float,
        object: PhysicalObject,
        channel: GenChannel,
        type: TargetType = TargetType.UNKNOWN,
    ) -> Target:
        return cls(
            label=label,
            frequency=frequency,
            object=object,
            channel=channel,
            type=type,
        )

    @classmethod
    def new_ge_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        return cls(
            label=Target.ge_label(qubit.label),
            object=qubit,
            frequency=qubit.ge_frequency,
            channel=channel,
            type=TargetType.CTRL_GE,
        )

    @classmethod
    def new_ef_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        return cls(
            label=Target.ef_label(qubit.label),
            object=qubit,
            frequency=qubit.ef_frequency,
            channel=channel,
            type=TargetType.CTRL_EF,
        )

    @classmethod
    def new_cr_target(
        cls,
        *,
        control_qubit: Qubit,
        target_qubit: Qubit | None = None,
        channel: GenChannel,
    ) -> Target:
        if target_qubit is not None:
            return cls(
                label=Target.cr_label(control_qubit.label, target_qubit.label),
                frequency=target_qubit.ge_frequency,
                object=control_qubit,
                channel=channel,
                type=TargetType.CTRL_CR,
            )
        else:
            return cls(
                label=Target.cr_label(control_qubit.label),
                frequency=round(channel.fine_frequency * 1e-9, 6),
                object=control_qubit,
                channel=channel,
                type=TargetType.CTRL_CR,
            )

    @classmethod
    def new_read_target(
        cls,
        *,
        resonator: Resonator,
        channel: GenChannel,
    ) -> Target:
        return cls(
            label=Target.read_label(resonator.label),
            object=resonator,
            frequency=resonator.frequency,
            channel=channel,
            type=TargetType.READ,
        )

    @classmethod
    def new_pump_target(
        cls,
        *,
        mux: Mux,
        frequency: float,
        channel: GenChannel,
    ) -> Target:
        return cls(
            label=mux.label,
            object=mux,
            frequency=frequency,
            channel=channel,
            type=TargetType.PUMP,
        )

    @classmethod
    def qubit_label(
        cls,
        label: str,
    ) -> str:
        if match := re.match(r"^R(Q\d+)$", label):
            qubit_label = match.group(1)
        elif match := re.match(r"^(Q\d+)$", label):
            qubit_label = match.group(1)
        elif match := re.match(r"^(Q\d+)-ef$", label):
            qubit_label = match.group(1)
        elif match := re.match(r"^(Q\d+)-CR$", label):
            qubit_label = match.group(1)
        elif match := re.match(r"^(Q\d+)-(Q\d+)$", label):
            qubit_label = match.group(1)
        elif match := re.match(r"^(Q\d+)(-|_)[a-zA-Z0-9]+$", label):
            qubit_label = match.group(1)
        else:
            raise ValueError(f"Invalid target label `{label}`.")
        return qubit_label

    @classmethod
    def ge_label(cls, label: str) -> str:
        qubit = cls.qubit_label(label)
        return f"{qubit}"

    @classmethod
    def ef_label(cls, label: str) -> str:
        qubit = cls.qubit_label(label)
        return f"{qubit}-ef"

    @classmethod
    def cr_label(cls, control_label: str, target_label: str | None = None) -> str:
        control_qubit = cls.qubit_label(control_label)
        if target_label is None:
            target_label = "CR"
        else:
            target_label = cls.qubit_label(target_label)
        return f"{control_qubit}-{target_label}"

    @classmethod
    def read_label(cls, label: str) -> str:
        qubit = cls.qubit_label(label)
        return f"R{qubit}"

    @staticmethod
    def cr_qubit_pair(
        label: str,
    ) -> tuple[str, str]:
        if match := re.match(r"^(Q\d+)-(Q\d+)$", label):
            control_qubit = match.group(1)
            target_qubit = match.group(2)
        else:
            raise ValueError(f"Invalid target label `{label}`.")
        return control_qubit, target_qubit
