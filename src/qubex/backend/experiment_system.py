from __future__ import annotations

import re
from enum import Enum
from typing import Final

from pydantic.dataclasses import dataclass

from .control_system import Box, CapPort, ControlSystem, GenPort, PortType
from .model import Model
from .quantum_system import Mux, QuantumSystem, Qubit, Resonator


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


@dataclass
class ControlParams(Model):
    control_amplitude: dict[str, float]
    readout_amplitude: dict[str, float]
    control_vatt: dict[str, int]
    readout_vatt: dict[int, int]
    control_fsc: dict[str, int]
    readout_fsc: dict[int, int]
    capture_delay: dict[int, int]


class ExperimentSystem:
    def __init__(
        self,
        quantum_system: QuantumSystem,
        control_system: ControlSystem,
        wiring_info: WiringInfo,
        control_params: ControlParams,
    ):
        self._quantum_system: Final = quantum_system
        self._control_system: Final = control_system
        self._wiring_info: Final = wiring_info
        self._control_params: Final = control_params
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
    def control_system(self) -> ControlSystem:
        return self._control_system

    @property
    def wiring_info(self) -> WiringInfo:
        return self._wiring_info

    @property
    def control_params(self) -> ControlParams:
        return self._control_params

    @property
    def qubits(self) -> list[Qubit]:
        return self.quantum_system.qubits

    @property
    def resonators(self) -> list[Resonator]:
        return self.quantum_system.resonators

    @property
    def boxes(self) -> list[Box]:
        return self.control_system.boxes

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

    def get_mux_by_readout_port(self, port: GenPort | CapPort) -> Mux:
        for mux, cap_port in self.wiring_info.read_in:
            if cap_port == port:
                return mux
        for mux, gen_port in self.wiring_info.read_out:
            if gen_port == port:
                return mux
        raise KeyError(f"Port `{port.id}` not found in wiring info.")

    def get_qubit_by_control_port(self, port: GenPort) -> Qubit:
        for qubit, gen_port in self.wiring_info.ctrl:
            if gen_port == port:
                return qubit
        raise KeyError(f"Port `{port.id}` not found in wiring info.")

    def configure_control_system(self, loopback: bool = False):
        control_vatt = self.control_params.control_vatt
        readout_vatt = self.control_params.readout_vatt
        control_fsc = self.control_params.control_fsc
        readout_fsc = self.control_params.readout_fsc
        capture_delay = self.control_params.capture_delay

        for box in self.boxes:
            for port in box.ports:
                if isinstance(port, GenPort):
                    if port.type == PortType.READ_OUT:
                        mux = self.get_mux_by_readout_port(port)
                        lo, cnco, fnco = self.find_readout_lo_nco(mux=mux)
                        port.lo_freq = lo
                        port.cnco_freq = cnco
                        port.vatt = readout_vatt[mux.index]
                        port.sideband = "U"
                        port.fullscale_current = readout_fsc[mux.index]
                        port.rfswitch = "block" if loopback else "pass"
                        port.channels[0].fnco_freq = fnco
                    elif port.type == PortType.CTRL:
                        qubit = self.get_qubit_by_control_port(port)
                        lo, cnco, fncos = self.find_control_lo_nco(
                            qubit=qubit,
                            n_channels=port.n_channels,
                        )
                        port.lo_freq = lo
                        port.cnco_freq = cnco
                        port.vatt = control_vatt[qubit.label]
                        port.sideband = "L"
                        port.fullscale_current = control_fsc[qubit.label]
                        port.rfswitch = "block" if loopback else "pass"
                        for idx, gen_channel in enumerate(port.channels):
                            gen_channel.fnco_freq = fncos[idx]
                elif isinstance(port, CapPort):
                    if port.type == PortType.READ_IN:
                        mux = self.get_mux_by_readout_port(port)
                        lo, cnco, fnco = self.find_readout_lo_nco(mux=mux)
                        port.lo_freq = lo
                        port.rfswitch = "loop" if loopback else "open"
                        for cap_channel in port.channels:
                            cap_channel.fnco_freq = fnco
                            cap_channel.ndelay = capture_delay[mux.index]

    def find_readout_lo_nco(
        self,
        mux: Mux,
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
        Finds the (lo, cnco, fnco) values for the readout mux.

        Parameters
        ----------
        mux : Mux
            The readout mux.
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
            The tuple (lo, cnco, fnco) for the readout mux.
        """
        frequencies = [resonator.frequency * 1e9 for resonator in mux.resonators]
        target_frequency = (max(frequencies) + min(frequencies)) / 2

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

    def find_control_lo_nco(
        self,
        qubit: Qubit,
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
        qubit : Qubit
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
        ge_target = self.get_ge_target(qubit.label)
        ef_target = self.get_ef_target(qubit.label)
        cr_target = self.get_cr_target(qubit.label)

        f_ge = ge_target.frequency * 1e9
        f_ef = ef_target.frequency * 1e9
        f_cr = cr_target.frequency * 1e9

        if n_channels == 1:
            f_med = f_ge
        elif n_channels == 3:
            targets = [ge_target, ef_target, cr_target]
            target_max = max(targets, key=lambda target: target.frequency)
            target_min = min(targets, key=lambda target: target.frequency)
            f_max = target_max.frequency * 1e9
            f_min = target_min.frequency * 1e9
            if f_max - f_min > max_diff:
                print(
                    f"Warning: {target_max.label} ({target_max.frequency:.3f} GHz) is too far from {target_min.label} ({target_min.frequency:.3f} GHz). Ignored {cr_target.label}."
                )
                f_med = (f_ge + f_ef) / 2
            else:
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
            raise ValueError("No valid lo value found for: ", f_med)

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
                raise ValueError("No valid fnco value found for: ", target_frequency)
            return best_fnco

        fnco_ge = find_fnco(f_ge)

        if n_channels == 1:
            return best_lo, cnco, (fnco_ge, 0, 0)

        fnco_ef = find_fnco(f_ef)
        fnco_cr = find_fnco(f_cr)

        return best_lo, cnco, (fnco_ge, fnco_ef, fnco_cr)
