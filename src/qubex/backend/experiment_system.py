from __future__ import annotations

from typing import Final, Literal, Sequence

import numpy as np
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
from .target import CapTarget, Target

DEFAULT_CONTROL_AMPLITUDE: Final = 0.03
DEFAULT_READOUT_AMPLITUDE: Final = 0.01
DEFAULT_CONTROL_VATT: Final = 3072
DEFAULT_READOUT_VATT: Final = 2048
DEFAULT_CONTROL_FSC: Final = 40527
DEFAULT_READOUT_FSC: Final = 40527
DEFAULT_CAPTURE_DELAY: Final = 7


LO_STEP = 500_000_000
NCO_STEP = 23_437_500
CNCO_CENTER_CTRL = 2_250_000_000
CNCO_CETNER_READ = 1_500_000_000
FNCO_MAX = 750_000_000
AWG_MAX = 250_000_000


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
        targets_to_exclude: list[str] | None = None,
    ):
        self._quantum_system: Final = quantum_system
        self._control_system: Final = control_system
        self._wiring_info: Final = wiring_info
        self._control_params: Final = control_params
        self._targets_to_exclude: Final = targets_to_exclude or []
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
        return [target for target in self._gen_target_dict.values() if target.is_ge]

    @property
    def ef_targets(self) -> list[Target]:
        return [target for target in self._gen_target_dict.values() if target.is_ef]

    @property
    def cr_targets(self) -> list[Target]:
        return [target for target in self._gen_target_dict.values() if target.is_cr]

    @property
    def ctrl_targets(self) -> list[Target]:
        return self.ge_targets + self.ef_targets + self.cr_targets

    @property
    def read_out_targets(self) -> list[Target]:
        return [target for target in self._gen_target_dict.values() if target.is_read]

    @property
    def targets(self) -> list[Target]:
        return [target for target in self._gen_target_dict.values()]

    @property
    def read_in_targets(self) -> list[CapTarget]:
        return list(self._cap_target_dict.values())

    @property
    def all_targets(self) -> list[Target | CapTarget]:
        return self.targets + self.read_in_targets

    def get_mux(self, label: int | str) -> Mux:
        return self.quantum_system.get_mux(label)

    def get_qubit(self, label: int | str) -> Qubit:
        return self.quantum_system.get_qubit(label)

    def get_resonator(self, label: int | str) -> Resonator:
        return self.quantum_system.get_resonator(label)

    def get_spectator_qubits(
        self,
        qubit: int | str,
        *,
        in_same_mux: bool = False,
    ) -> list[Qubit]:
        return self.quantum_system.get_spectator_qubits(qubit, in_same_mux=in_same_mux)

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
            return self._gen_target_dict[label]
        except KeyError:
            raise KeyError(f"Target `{label}` not found.") from None

    def get_cap_target(self, label: str) -> CapTarget:
        try:
            return self._cap_target_dict[label]
        except KeyError:
            raise KeyError(f"CapTarget `{label}` not found.") from None

    def get_ge_target(self, label: str) -> Target:
        label = Target.ge_label(label)
        return self.get_target(label)

    def get_ef_target(self, label: str) -> Target:
        label = Target.ef_label(label)
        return self.get_target(label)

    def get_cr_target(self, label: str) -> Target:
        label = Target.cr_label(label)
        return self.get_target(label)

    def get_read_out_target(self, label: str) -> Target:
        label = Target.read_label(label)
        return self.get_target(label)

    def get_read_in_target(self, label: str) -> CapTarget:
        return self.get_cap_target(label)

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

    def get_mux_by_qubit(self, label: str) -> Mux:
        ports = self.get_qubit_port_set(label)
        if ports is None:
            raise ValueError(f"Qubit `{label}` not found.")
        mux = self.get_mux_by_readout_port(ports.read_out_port)
        if mux is None:
            raise ValueError(f"No mux found for qubit `{label}`.")
        return mux

    def get_readout_pair(self, port: CapPort) -> GenPort:
        cap_mux = self.get_mux_by_readout_port(port)
        if cap_mux is None:
            raise ValueError(f"No mux found for port: {port}")
        for gen_mux, gen_port in self.wiring_info.read_out:
            if gen_mux.index == cap_mux.index:
                return gen_port
        raise ValueError(f"No readout pair found for port: {port}")

    def modify_target_frequencies(
        self,
        frequencies: dict[str, float],
    ):
        for label, frequency in frequencies.items():
            target = self.get_target(label)
            target.frequency = frequency

    def update_port_params(
        self,
        label: str,
        *,
        lo_freq: int,
        cnco_freq: int,
        fnco_freq: int,
    ):
        target = self.get_target(label)
        gen_channel = target.channel
        original_values = (
            gen_channel.port.lo_freq,
            gen_channel.port.cnco_freq,
            gen_channel.fnco_freq,
        )
        try:
            gen_channel.port.lo_freq = lo_freq
            gen_channel.port.cnco_freq = cnco_freq
            gen_channel.fnco_freq = fnco_freq
            if target.is_read:
                cap_channel = self.get_read_in_target(label).channel
                cap_channel.port.lo_freq = lo_freq
                cap_channel.port.cnco_freq = cnco_freq
                cap_channel.fnco_freq = fnco_freq
        except Exception as e:
            # rollback
            (
                gen_channel.port.lo_freq,
                gen_channel.port.cnco_freq,
                gen_channel.fnco_freq,
            ) = original_values
            if target.is_read:
                (
                    cap_channel.port.lo_freq,
                    cap_channel.port.cnco_freq,
                    cap_channel.fnco_freq,
                ) = original_values
            raise ValueError(f"Error setting readout port params: {e}") from None

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
            if mux is None or not mux.is_valid:
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
            if qubit is None or not qubit.is_valid:
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
            if mux is None or not mux.is_valid:
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
        ssb: Literal["U", "L"] = "U",
        cnco_center: int = CNCO_CETNER_READ,
    ) -> tuple[int, int, int]:
        """
        Finds the (lo, cnco, fnco) values for the readout mux.

        Parameters
        ----------
        mux : Mux
            The readout mux.
        ssb : Literal["U", "L"], optional
            The sideband, by default "U".
        cnco_center : int, optional
            The center frequency of the CNCO, by default CNCO_CETNER_READ.

        Returns
        -------
        tuple[int, int, int]
            The tuple (lo, cnco, fnco) for the readout mux.
        """
        freqs = [resonator.frequency * 1e9 for resonator in mux.resonators]
        f_target = (max(freqs) + min(freqs)) / 2
        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            f=f_target,
            ssb=ssb,
            cnco_center=cnco_center,
        )
        fnco, _ = MixingUtil.calc_fnco(
            f=f_target,
            ssb=ssb,
            lo=lo,
            cnco=cnco,
        )
        return lo, cnco, fnco

    def _find_control_lo_nco(
        self,
        qubit: Qubit,
        n_channels: int,
        *,
        ssb: Literal["U", "L"] = "L",
        cnco_center: int = CNCO_CENTER_CTRL,
    ) -> tuple[int, int, tuple[int, int, int]]:
        """
        Finds the (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) values for the control qubit.

        Parameters
        ----------
        qubit : Qubit
            The control qubit.
        n_channels : int
            The number of channels.
        ssb : Literal["U", "L"], optional
            The sideband, by default "L".
        cnco_center : int, optional
            The center frequency of the CNCO, by default CNCO_CENTER_CTRL.

        Returns
        -------
        tuple[int, int, tuple[int, int, int]]
            The tuple (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) for the control qubit.
        """
        if n_channels == 1:
            f_target = qubit.ge_frequency * 1e9
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f=f_target,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            fnco, _ = MixingUtil.calc_fnco(
                f=f_target,
                ssb=ssb,
                lo=lo,
                cnco=cnco,
            )
            return lo, cnco_center, (fnco, 0, 0)
        elif n_channels != 3:
            raise ValueError("Invalid number of channels.")

        f_ge = qubit.ge_frequency * 1e9
        f_ef = qubit.ef_frequency * 1e9
        f_CRs = [
            spectator.ge_frequency * 1e9
            for spectator in self.get_spectator_qubits(qubit.label)
            if spectator.ge_frequency > 0
            and spectator.label not in self._targets_to_exclude
            and f"{qubit.label}-{spectator.label}" not in self._targets_to_exclude
        ]

        if not f_CRs:
            f_CRs = [f_ge]

        f_CR_max = max(f_CRs)
        if f_CR_max > f_ge:
            # if any CR is larger than GE, then let EF be the smallest
            lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                f=f_ef + FNCO_MAX,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            f_CRs_valid = [f for f in f_CRs if f < f_coarse + FNCO_MAX + AWG_MAX]
        else:
            # if all CRs are smaller than GE, then let GE be the largest
            lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                f=f_ge - FNCO_MAX,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            f_CRs_valid = [f for f in f_CRs if f > f_coarse - FNCO_MAX - AWG_MAX]
        f_CR = self._find_center_freq_for_cr(
            f_coarse=f_coarse,
            f_CRs=f_CRs_valid,
        )
        fnco_ge, _ = MixingUtil.calc_fnco(f=f_ge, ssb=ssb, lo=lo, cnco=cnco)
        fnco_ef, _ = MixingUtil.calc_fnco(f=f_ef, ssb=ssb, lo=lo, cnco=cnco)
        fnco_CR, _ = MixingUtil.calc_fnco(f=f_CR, ssb=ssb, lo=lo, cnco=cnco)
        return (lo, cnco, (fnco_ge, fnco_ef, fnco_CR))

    def _find_center_freq_for_cr(
        self,
        f_coarse: int,
        f_CRs: list[float],
    ) -> int:
        if not f_CRs:
            return f_coarse
        # possible range
        min_center_freq = f_coarse - FNCO_MAX
        max_center_freq = f_coarse + FNCO_MAX

        # search range
        search_range = np.arange(
            max(min(f_CRs), min_center_freq),
            min(max(f_CRs), max_center_freq) + 1,
            NCO_STEP,
        )
        # count the number of CR frequencies within the range of each search point
        center_freqs_by_count = [
            (
                np.sum([1 for f_CR in f_CRs if f - AWG_MAX <= f_CR <= f + AWG_MAX]),
                np.median(
                    [f_CR for f_CR in f_CRs if f - AWG_MAX <= f_CR <= f + AWG_MAX]
                    or [0]
                ),
            )
            for f in search_range
        ]
        if not center_freqs_by_count:
            return f_coarse
        # sort by count and then by frequency
        center_freqs_by_count.sort(key=lambda x: (x[0], x[1]), reverse=True)
        # choose the one with the highest count and frequency
        center_freq = center_freqs_by_count[0][1].astype(int)
        # round to the nearest NCO step
        center_freq = round(center_freq / NCO_STEP) * NCO_STEP
        # clip to the possible range
        center_freq = max(min_center_freq, min(center_freq, max_center_freq))
        return center_freq

    def _initialize_targets(self) -> None:
        self._gen_target_dict: dict[str, Target] = {}
        self._cap_target_dict: dict[str, CapTarget] = {}

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
                            self._gen_target_dict[ge_target.label] = ge_target
                        elif port.n_channels == 3:
                            # ge
                            ge_target = Target.new_ge_target(
                                qubit=qubit,
                                channel=port.channels[0],
                            )
                            self._gen_target_dict[ge_target.label] = ge_target
                            # ef
                            ef_target = Target.new_ef_target(
                                qubit=qubit,
                                channel=port.channels[1],
                            )
                            self._gen_target_dict[ef_target.label] = ef_target
                            # cr
                            cr_target = Target.new_cr_target(
                                control_qubit=qubit,
                                channel=port.channels[2],
                            )
                            self._gen_target_dict[cr_target.label] = cr_target
                            for spectator in self.get_spectator_qubits(qubit.label):
                                cr_target = Target.new_cr_target(
                                    control_qubit=qubit,
                                    target_qubit=spectator,
                                    channel=port.channels[2],
                                )
                                self._gen_target_dict[cr_target.label] = cr_target

                    # readout ports
                    elif port.type == PortType.READ_OUT:
                        mux = self.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        for resonator in mux.resonators:
                            read_out_target = Target.new_read_target(
                                resonator=resonator,
                                channel=port.channels[0],
                            )
                            self._gen_target_dict[read_out_target.label] = (
                                read_out_target
                            )

                # cap ports
                elif isinstance(port, CapPort):
                    if port.type == PortType.READ_IN:
                        mux = self.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        for idx, resonator in enumerate(mux.resonators):
                            read_in_target = CapTarget.new_read_target(
                                resonator=resonator,
                                channel=port.channels[idx],
                            )
                            self._cap_target_dict[read_in_target.label] = read_in_target

        self._gen_target_dict = dict(sorted(self._gen_target_dict.items()))
        self._cap_target_dict = dict(sorted(self._cap_target_dict.items()))


class MixingUtil:
    @staticmethod
    def calc_lo_cnco(
        f: float,
        ssb: Literal["U", "L"],
        cnco_center: int,
        lo_step: int = LO_STEP,
        nco_step: int = NCO_STEP,
    ) -> tuple[int, int, int]:
        if ssb == "U":
            lo = round((f - cnco_center) / lo_step) * lo_step
            cnco = round((f - lo) / nco_step) * nco_step
        elif ssb == "L":
            lo = round((f + cnco_center) / lo_step) * lo_step
            cnco = round((lo - f) / nco_step) * nco_step
        else:
            raise ValueError("Invalid SSB")
        f_mix = lo + cnco if ssb == "U" else lo - cnco
        return lo, cnco, f_mix

    @staticmethod
    def calc_fnco(
        f: float,
        ssb: Literal["U", "L"],
        lo: int,
        cnco: int,
        nco_step: int = NCO_STEP,
    ) -> tuple[int, int]:
        if ssb == "U":
            fnco = round((f - (lo + cnco)) / nco_step) * nco_step
        elif ssb == "L":
            fnco = round(((lo - cnco) - f) / nco_step) * nco_step
        else:
            raise ValueError("Invalid SSB")
        f_mix = lo + cnco + fnco if ssb == "U" else lo - cnco - fnco
        return fnco, f_mix
