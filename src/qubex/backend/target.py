from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TargetType(Enum):
    CTRL_GE = "CTRL_GE"
    CTRL_EF = "CTRL_EF"
    CTRL_CR = "CTRL_CR"
    READ = "READ"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Target:
    label: str
    qubit: str
    type: TargetType
    frequency: float

    def __repr__(self) -> str:
        return f"Target(label={self.label}, qubit={self.qubit}, type={self.type.value}, frequency={self.frequency})"

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
