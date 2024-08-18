from __future__ import annotations

from typing import Final, Sequence

from pydantic.dataclasses import dataclass

from .control_system import (
    Box,
    CapPort,
    ControlSystem,
    GenPort,
    PortType,
)
from .model import Model
from .quantum_system import Chip, Mux, QuantumSystem, Qubit, Resonator
from .target import Target

DEFAULT_CONTROL_AMPLITUDE: Final = 0.03
DEFAULT_READOUT_AMPLITUDE: Final = 0.01
DEFAULT_CONTROL_VATT: Final = 3072
DEFAULT_READOUT_VATT: Final = 2048
DEFAULT_CONTROL_FSC: Final = 40527
DEFAULT_READOUT_FSC: Final = 40527
DEFAULT_CAPTURE_DELAY: Final = 7


@dataclass
class WiringInfo(Model):
    ctrl: list[tuple[Qubit, GenPort]]
    read_out: list[tuple[Mux, GenPort]]
    read_in: list[tuple[Mux, CapPort]]


@dataclass
class QubitPortSet(Model):
    ctrl_port: GenPort
    read_out_port: GenPort
    read_in_port: CapPort


@dataclass
class ControlParams(Model):
    control_amplitude: dict[str, float]
    readout_amplitude: dict[str, float]
    control_vatt: dict[str, int]
    readout_vatt: dict[int, int]
    control_fsc: dict[str, int]
    readout_fsc: dict[int, int]
    capture_delay: dict[int, int]

    def get_control_amplitude(self, qubit: str) -> float:
        return self.control_amplitude.get(qubit, DEFAULT_CONTROL_AMPLITUDE)

    def get_readout_amplitude(self, qubit: str) -> float:
        return self.readout_amplitude.get(qubit, DEFAULT_READOUT_AMPLITUDE)

    def get_control_vatt(self, qubit: str) -> int:
        return self.control_vatt.get(qubit, DEFAULT_CONTROL_VATT)

    def get_readout_vatt(self, mux: int) -> int:
        return self.readout_vatt.get(mux, DEFAULT_READOUT_VATT)

    def get_control_fsc(self, qubit: str) -> int:
        return self.control_fsc.get(qubit, DEFAULT_CONTROL_FSC)

    def get_readout_fsc(self, mux: int) -> int:
        return self.readout_fsc.get(mux, DEFAULT_READOUT_FSC)

    def get_capture_delay(self, mux: int) -> int:
        return self.capture_delay.get(mux, DEFAULT_CAPTURE_DELAY)


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
        self._qubit_port_set_map: Final = self._create_qubit_port_set_map()
        self._initialize_ports()
        self._initialize_targets()

    @property
    def hash(self) -> int:
        return hash(
            (
                self.quantum_system.hash,
                self.control_system.hash,
                self.wiring_info.hash,
                self.control_params.hash,
            )
        )

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
    def chip(self) -> Chip:
        return self.quantum_system.chip

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
        return list(self._read_target_dict.values())

    @property
    def targets(self) -> list[Target]:
        return (
            self.ge_targets + self.ef_targets + self.cr_targets + self.readout_targets
        )

    def get_mux(self, label: int | str) -> Mux:
        return self.quantum_system.get_mux(label)

    def get_qubit(self, label: int | str) -> Qubit:
        return self.quantum_system.get_qubit(label)

    def get_resonator(self, label: int | str) -> Resonator:
        return self.quantum_system.get_resonator(label)

    def get_spectator_qubits(self, qubit: int | str) -> list[Qubit]:
        return self.quantum_system.get_spectator_qubits(qubit)

    def get_box(self, box_id: str) -> Box:
        return self.control_system.get_box(box_id)

    def get_boxes_for_qubits(self, qubits: Sequence[str]) -> list[Box]:
        box_ids = set()
        for qubit in qubits:
            ports = self.get_qubit_port_set(qubit)
            if ports is None:
                continue
            box_ids.add(ports.ctrl_port.box_id)
            box_ids.add(ports.read_out_port.box_id)
            box_ids.add(ports.read_in_port.box_id)
        return [self.get_box(box_id) for box_id in box_ids]

    def get_target(self, label: str) -> Target:
        try:
            return self._target_dict[label]
        except KeyError:
            raise KeyError(f"Target `{label}` not found.") from None

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
        label = Target.read_label(label)
        return self.get_target(label)

    def get_qubit_port_set(self, qubit: int | str) -> QubitPortSet | None:
        if isinstance(qubit, int):
            qubit = self.qubits[qubit].label
        return self._qubit_port_set_map.get(qubit)

    def get_control_port(self, qubit: int | str) -> GenPort:
        ports = self.get_qubit_port_set(qubit)
        if ports is None:
            raise ValueError(f"Qubit `{qubit}` not found.")
        return ports.ctrl_port

    def get_base_frequency(self, label: str) -> float:
        target = self.get_target(label)
        return target.coarse_frequency

    def get_diff_frequency(self, label: str) -> float:
        target = self.get_target(label)
        return round(target.frequency - self.get_base_frequency(label), 10)

    def get_mux_by_readout_port(self, port: GenPort | CapPort) -> Mux | None:
        if isinstance(port, CapPort):
            for mux, cap_port in self.wiring_info.read_in:
                if cap_port == port:
                    return mux
        elif isinstance(port, GenPort):
            for mux, gen_port in self.wiring_info.read_out:
                if gen_port == port:
                    return mux
        return None

    def get_qubit_by_control_port(self, port: GenPort) -> Qubit | None:
        for qubit, gen_port in self.wiring_info.ctrl:
            if gen_port == port:
                return qubit
        return None

    def get_readout_pair(self, port: CapPort) -> GenPort:
        cap_mux = self.get_mux_by_readout_port(port)
        if cap_mux is None:
            raise ValueError(f"No mux found for port: {port}")
        for gen_mux, gen_port in self.wiring_info.read_out:
            if gen_mux.index == cap_mux.index:
                return gen_port
        raise ValueError(f"No readout pair found for port: {port}")

    def _create_qubit_port_set_map(self) -> dict[str, QubitPortSet]:
        ctrl_port_map: dict[str, GenPort] = {}
        read_out_port_map: dict[str, GenPort] = {}
        read_in_port_map: dict[str, CapPort] = {}
        for qubit, gen_port in self.wiring_info.ctrl:
            ctrl_port_map[qubit.label] = gen_port
        for mux, gen_port in self.wiring_info.read_out:
            for resonator in mux.resonators:
                read_out_port_map[resonator.qubit] = gen_port
        for mux, cap_port in self.wiring_info.read_in:
            for resonator in mux.resonators:
                read_in_port_map[resonator.qubit] = cap_port
        return {
            qubit: QubitPortSet(
                ctrl_port=ctrl_port_map[qubit],
                read_out_port=read_out_port_map[qubit],
                read_in_port=read_in_port_map[qubit],
            )
            for qubit in ctrl_port_map
        }

    def _initialize_ports(self):
        params = self.control_params
        for box in self.boxes:
            for port in box.ports:
                if isinstance(port, GenPort):
                    self._initialize_gen_port(port, params)
                elif isinstance(port, CapPort):
                    self._initialize_cap_port(port, params)

    def _initialize_gen_port(
        self,
        port: GenPort,
        params: ControlParams,
    ) -> None:
        port.rfswitch = "pass"
        if port.type == PortType.READ_OUT:
            mux = self.get_mux_by_readout_port(port)
            if mux is None:
                return
            lo, cnco, fnco = self._find_readout_lo_nco(mux)
            port.lo_freq = lo
            port.cnco_freq = cnco
            port.sideband = "U"
            port.vatt = params.get_readout_vatt(mux.index)
            port.fullscale_current = params.get_readout_fsc(mux.index)
            port.channels[0].fnco_freq = fnco
        elif port.type == PortType.CTRL:
            qubit = self.get_qubit_by_control_port(port)
            if qubit is None:
                return
            lo, cnco, fncos = self._find_control_lo_nco(
                qubit=qubit,
                n_channels=port.n_channels,
            )
            port.lo_freq = lo
            port.cnco_freq = cnco
            port.sideband = "L"
            port.vatt = params.get_control_vatt(qubit.label)
            port.fullscale_current = params.get_control_fsc(qubit.label)
            for idx, gen_channel in enumerate(port.channels):
                gen_channel.fnco_freq = fncos[idx]

    def _initialize_cap_port(
        self,
        port: CapPort,
        params: ControlParams,
    ) -> None:
        port.rfswitch = "open"
        if port.type == PortType.READ_IN:
            mux = self.get_mux_by_readout_port(port)
            if mux is None:
                return
            lo, cnco, fnco = self._find_readout_lo_nco(mux)
            port.lo_freq = lo
            port.cnco_freq = cnco
            for cap_channel in port.channels:
                cap_channel.fnco_freq = fnco
                cap_channel.ndelay = params.get_capture_delay(mux.index)

    def _find_readout_lo_nco(
        self,
        mux: Mux,
        *,
        cnco: int = 1_500_000_000,
        lo_step: int = 500_000_000,
        nco_step: int = 23_437_500,
    ) -> tuple[int, int, int]:
        """
        Finds the (lo, cnco, fnco) values for the readout mux.

        Parameters
        ----------
        mux : Mux
            The readout mux.
        cnco : int, optional
            The CNCO frequency, by default 1_500_000_000.
        lo_step : int, optional
            The LO frequency step, by default 500_000_000.
        nco_step : int, optional
            The NCO frequency step, by default 23_437_500.

        Returns
        -------
        tuple[int, int, int]
            The tuple (lo, cnco, fnco) for the readout mux.
        """
        freqs = [resonator.frequency * 1e9 for resonator in mux.resonators]
        f_target = (max(freqs) + min(freqs)) / 2
        lo = round((f_target - cnco) / lo_step) * lo_step
        fnco = round((f_target - lo - cnco) / nco_step) * nco_step
        return lo, cnco, fnco

    def _find_control_lo_nco(
        self,
        qubit: Qubit,
        n_channels: int,
        *,
        cnco: int = 2_250_000_000,
        lo_step: int = 500_000_000,
        nco_step: int = 23_437_500,
        max_diff: int = 750_000_000,
    ) -> tuple[int, int, tuple[int, int, int]]:
        """
        Finds the (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) values for the control qubit.

        Parameters
        ----------
        qubit : Qubit
            The control qubit.
        n_channels : int
            The number of channels.
        cnco : int, optional
            The CNCO frequency, by default 2_250_000_000.
        lo_step : int, optional
            The LO frequency step, by default 500_000_000.
        nco_step : int, optional
            The NCO frequency step, by default 23_437_500.
        max_diff : int, optional
            The maximum difference between frequencies, by default 750_000_000.

        Returns
        -------
        tuple[int, int, tuple[int, int, int]]
            The tuple (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) for the control qubit.
        """
        if n_channels == 1:
            f_target = qubit.ge_frequency * 1e9
            lo = round((f_target + cnco) / lo_step) * lo_step
            fnco = round((lo - cnco - f_target) / nco_step) * nco_step
            return lo, cnco, (fnco, 0, 0)
        elif n_channels != 3:
            raise ValueError("Invalid number of channels.")

        f_ge = qubit.ge_frequency * 1e9
        f_ef = qubit.ef_frequency * 1e9
        f_spectators = [
            spectator.ge_frequency * 1e9
            for spectator in self.get_spectator_qubits(qubit.label)
            if spectator.ge_frequency > 0
        ]
        f_cr = (max(f_spectators) + min(f_spectators)) / 2
        freqs = [f_ge, f_ef] + f_spectators
        f_target = (max(freqs) + min(freqs)) / 2

        lo = round((f_target + cnco) / lo_step) * lo_step
        fnco_ge = round((lo - cnco - f_ge) / nco_step) * nco_step
        fnco_ef = round((lo - cnco - f_ef) / nco_step) * nco_step
        fnco_cr = round((lo - cnco - f_cr) / nco_step) * nco_step
        return lo, cnco, (fnco_ge, fnco_ef, fnco_cr)

    def _initialize_targets(self) -> None:
        ge_target_dict: dict[str, Target] = {}
        ef_target_dict: dict[str, Target] = {}
        cr_target_dict: dict[str, Target] = {}
        read_target_dict: dict[str, Target] = {}

        for box in self.boxes:
            for port in box.ports:
                # gen ports
                if isinstance(port, GenPort):
                    # ctrl ports
                    if port.type == PortType.CTRL:
                        qubit = self.get_qubit_by_control_port(port)
                        if qubit is None:
                            continue

                        if port.n_channels == 1:
                            # ge only
                            ge_target = Target.new_ge_target(
                                qubit=qubit,
                                channel=port.channels[0],
                            )
                            ge_target_dict[ge_target.label] = ge_target
                        elif port.n_channels == 3:
                            # ge
                            ge_target = Target.new_ge_target(
                                qubit=qubit,
                                channel=port.channels[0],
                            )
                            ge_target_dict[ge_target.label] = ge_target
                            # ef
                            ef_target = Target.new_ef_target(
                                qubit=qubit,
                                channel=port.channels[1],
                            )
                            ef_target_dict[ef_target.label] = ef_target
                            # cr
                            cr_target = Target.new_cr_target(
                                control_qubit=qubit,
                                channel=port.channels[2],
                            )
                            cr_target_dict[cr_target.label] = cr_target
                            for spectator in self.get_spectator_qubits(qubit.label):
                                cr_target = Target.new_cr_target(
                                    control_qubit=qubit,
                                    target_qubit=spectator,
                                    channel=port.channels[2],
                                )
                                cr_target_dict[cr_target.label] = cr_target

                    # readout ports
                    elif port.type == PortType.READ_OUT:
                        mux = self.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        for resonator in mux.resonators:
                            readout_target = Target.new_read_target(
                                resonator=resonator,
                                channel=port.channels[0],
                            )
                            read_target_dict[readout_target.label] = readout_target

        self._ge_target_dict = dict(sorted(ge_target_dict.items()))
        self._ef_target_dict = dict(sorted(ef_target_dict.items()))
        self._cr_target_dict = dict(sorted(cr_target_dict.items()))
        self._read_target_dict = dict(sorted(read_target_dict.items()))
        self._target_dict = (
            self._ge_target_dict
            | self._ef_target_dict
            | self._cr_target_dict
            | self._read_target_dict
        )
