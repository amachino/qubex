"""Target and target label helpers for control and readout channels."""

from __future__ import annotations

import re
from collections.abc import Collection
from enum import Enum
from typing import Literal

from pydantic.dataclasses import dataclass

from .control_system import CapChannel, GenChannel
from .model import Model
from .quantum_system import Mux, Qubit, Resonator

# TODO: Make target label formats configurable


class TargetType(Enum):
    """Enumerate supported target types."""

    CTRL_GE = "CTRL_GE"
    CTRL_EF = "CTRL_EF"
    CTRL_CR = "CTRL_CR"
    READ = "READ"
    PUMP = "PUMP"
    UNKNOWN = "UNKNOWN"


PhysicalObject = Qubit | Resonator | Mux


@dataclass
class CapTarget(Model):
    """Capture target metadata for readout channels."""

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
        """Create a readout capture target for a resonator."""
        return cls(
            label=Target.read_label(resonator.label),
            object=resonator,
            frequency=resonator.frequency,
            channel=channel,
            type=TargetType.READ,
        )


@dataclass
class Target(Model):
    """Generator target metadata for control and readout channels."""

    label: str
    frequency: float
    object: PhysicalObject
    channel: GenChannel
    type: TargetType

    def __repr__(self) -> str:
        """Return the debug representation of the target."""
        return f"Target(label={self.label}, frequency={self.frequency}, channel={self.channel.id}, object={self.object.label})"

    @property
    def qubit(self) -> str:
        """Return the associated qubit label."""
        if isinstance(self.object, Qubit):
            return self.object.label
        elif isinstance(self.object, Resonator):
            return self.object.qubit
        elif isinstance(self.object, Mux):
            return ""
        else:
            raise TypeError("Invalid quantum object.")

    @property
    def sideband(self) -> Literal["U", "L"] | None:
        """Return the sideband setting of the channel."""
        return self.channel.port.sideband

    @property
    def coarse_frequency(self) -> float:
        """Return the channel coarse frequency in GHz."""
        if isinstance(self.channel, GenChannel):
            return self.channel.coarse_frequency * 1e-9
        else:
            raise TypeError("Invalid channel.")

    @property
    def fine_frequency(self) -> float:
        """Return the channel fine frequency in GHz."""
        if isinstance(self.channel, GenChannel):
            return self.channel.fine_frequency * 1e-9
        else:
            raise TypeError("Invalid channel.")

    @property
    def awg_frequency(self) -> float:
        """Return the AWG frequency in GHz for the target."""
        if isinstance(self.channel, GenChannel):
            if self.sideband == "L":
                return self.fine_frequency - self.frequency
            else:
                return self.frequency - self.fine_frequency
        else:
            raise TypeError("Invalid channel.")

    @property
    def is_available(self) -> bool:
        """Report whether the target frequency is within the tuning range."""
        return abs(self.frequency - self.fine_frequency) < 250 * 1e-3  # 250 MHz

    @property
    def is_ge(self) -> bool:
        """Report whether this is a GE control target."""
        return self.type == TargetType.CTRL_GE

    @property
    def is_ef(self) -> bool:
        """Report whether this is an EF control target."""
        return self.type == TargetType.CTRL_EF

    @property
    def is_cr(self) -> bool:
        """Report whether this is a CR control target."""
        return self.type == TargetType.CTRL_CR

    @property
    def is_read(self) -> bool:
        """Report whether this is a readout target."""
        return self.type == TargetType.READ

    @property
    def is_pump(self) -> bool:
        """Report whether this is a pump target."""
        return self.type == TargetType.PUMP

    def is_related_to_qubits(self, qubits: Collection[str]) -> bool:
        """Return whether the target relates to the provided qubits."""
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
            raise TypeError("Invalid quantum object.")

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
        """Create a generic target from metadata."""
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
        """Create a GE control target for a qubit."""
        return cls(
            label=Target.ge_label(qubit.label),
            object=qubit,
            frequency=qubit.frequency,
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
        """Create an EF control target for a qubit."""
        return cls(
            label=Target.ef_label(qubit.label),
            object=qubit,
            frequency=qubit.control_frequency_ef,
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
        """Create a CR control target for a qubit pair."""
        if target_qubit is not None:
            return cls(
                label=Target.cr_label(control_qubit.label, target_qubit.label),
                frequency=target_qubit.frequency,
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
        """Create a readout target for a resonator."""
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
        """Create a pump target for a mux."""
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
        """Extract a qubit label from a target label."""
        if (
            (match := re.match(r"^R(Q\d+)$", label))
            or (match := re.match(r"^(Q\d+)$", label))
            or (match := re.match(r"^(Q\d+)-ef$", label))
            or (match := re.match(r"^(Q\d+)-CR$", label))
            or (match := re.match(r"^(Q\d+)-(Q\d+)$", label))
            or (match := re.match(r"^(Q\d+)(-|_)[a-zA-Z0-9]+$", label))
        ):
            qubit_label = match.group(1)
        else:
            raise ValueError(f"Invalid target label `{label}`.")
        return qubit_label

    @classmethod
    def ge_label(cls, label: str) -> str:
        """Return the GE target label for a qubit label."""
        qubit = cls.qubit_label(label)
        return f"{qubit}"

    @classmethod
    def ef_label(cls, label: str) -> str:
        """Return the EF target label for a qubit label."""
        qubit = cls.qubit_label(label)
        return f"{qubit}-ef"

    @classmethod
    def cr_label(cls, control_label: str, target_label: str | None = None) -> str:
        """Return the CR target label for a control/target pair."""
        control_qubit = cls.qubit_label(control_label)
        if target_label is None:
            target_label = "CR"
        else:
            target_label = cls.qubit_label(target_label)
        return f"{control_qubit}-{target_label}"

    @classmethod
    def read_label(cls, label: str) -> str:
        """Return the readout target label for a qubit label."""
        qubit = cls.qubit_label(label)
        return f"R{qubit}"

    @staticmethod
    def cr_qubit_pair(
        label: str,
    ) -> tuple[str, str]:
        """Parse a CR target label into a qubit pair."""
        if match := re.match(r"^(Q\d+)-(Q\d+)$", label):
            control_qubit = match.group(1)
            target_qubit = match.group(2)
        else:
            raise ValueError(f"Invalid target label `{label}`.")
        return control_qubit, target_qubit
