from __future__ import annotations

import logging
import re
from typing import Collection, Final, Literal, Optional

import numpy as np
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from .control_system import (
    Box,
    BoxType,
    CapPort,
    ControlSystem,
    GenPort,
    PortType,
)
from .model import Model
from .quantum_system import Chip, Mux, QuantumSystem, Qubit, Resonator
from .target import CapTarget, Target

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_AMPLITUDE: Final = 0.03
DEFAULT_READOUT_AMPLITUDE: Final = 0.01
DEFAULT_CONTROL_VATT: Final = 3072
DEFAULT_READOUT_VATT: Final = 2048
DEFAULT_PUMP_VATT: Final = 3072
DEFAULT_CONTROL_FSC: Final = 40527
DEFAULT_READOUT_FSC: Final = 40527
DEFAULT_PUMP_FSC: Final = 40527
DEFAULT_CAPTURE_DELAY: Final = 7
DEFAULT_CAPTURE_DELAY_WORD: Final = 0
DEFAULT_PUMP_FREQUENCY: Final = 10.0
DEFAULT_PUMP_AMPLITUDE: Final = 0.0
DEFAULT_DC_VOLTAGE: Final = 0.0


LO_STEP = 500_000_000
NCO_STEP = 23_437_500
CNCO_CENTER_CTRL = 2_250_000_000
CNCO_CETNER_READ = 1_500_000_000
CNCO_CETNER_READ_R8 = 2_250_000_000
FNCO_MAX = 750_000_000
AWG_MAX = 250_000_000


@dataclass
class WiringInfo(Model):
    ctrl: list[tuple[Qubit, GenPort]]
    read_out: list[tuple[Mux, GenPort]]
    read_in: list[tuple[Mux, CapPort]]
    pump: list[tuple[Mux, GenPort]]


@dataclass
class QubitPortSet(Model):
    ctrl_port: GenPort
    read_out_port: GenPort
    read_in_port: CapPort


class JPAParam(TypedDict):
    dc_voltage: Optional[float]
    pump_frequency: Optional[float]
    pump_amplitude: Optional[float]


@dataclass
class ControlParams(Model):
    control_amplitude: dict[str, float]
    readout_amplitude: dict[str, float]
    control_vatt: dict[str, Optional[int]]
    readout_vatt: dict[int, int]
    pump_vatt: dict[int, int]
    control_fsc: dict[str, int]
    readout_fsc: dict[int, int]
    pump_fsc: dict[int, int]
    capture_delay: dict[int, int]
    capture_delay_word: dict[int, int]
    jpa_params: dict[int, Optional[JPAParam]]

    def get_control_amplitude(self, qubit: str) -> float:
        return self.control_amplitude.get(qubit, DEFAULT_CONTROL_AMPLITUDE)

    def get_ef_control_amplitude(self, qubit: str) -> float:
        return self.get_control_amplitude(qubit) / np.sqrt(2)

    def get_readout_amplitude(self, qubit: str) -> float:
        return self.readout_amplitude.get(qubit, DEFAULT_READOUT_AMPLITUDE)

    def get_control_vatt(self, qubit: str) -> int | None:
        return self.control_vatt.get(qubit, DEFAULT_CONTROL_VATT)

    def get_readout_vatt(self, mux: int) -> int:
        return self.readout_vatt.get(mux, DEFAULT_READOUT_VATT)

    def get_pump_vatt(self, mux: int) -> int:
        return self.pump_vatt.get(mux, DEFAULT_PUMP_VATT)

    def get_control_fsc(self, qubit: str) -> int:
        return self.control_fsc.get(qubit, DEFAULT_CONTROL_FSC)

    def get_readout_fsc(self, mux: int) -> int:
        return self.readout_fsc.get(mux, DEFAULT_READOUT_FSC)

    def get_pump_fsc(self, mux: int) -> int:
        return self.pump_fsc.get(mux, DEFAULT_PUMP_FSC)

    def get_capture_delay(self, mux: int) -> int:
        return self.capture_delay.get(mux, DEFAULT_CAPTURE_DELAY)

    def get_capture_delay_word(self, mux: int) -> int:
        return self.capture_delay_word.get(mux, DEFAULT_CAPTURE_DELAY_WORD)

    def get_pump_frequency(self, mux: int) -> float:
        jpa_param = self.jpa_params.get(mux)
        if jpa_param is None:
            return DEFAULT_PUMP_FREQUENCY
        else:
            return jpa_param.get("pump_frequency") or DEFAULT_PUMP_FREQUENCY

    def get_pump_amplitude(self, mux: int) -> float:
        jpa_param = self.jpa_params.get(mux)
        if jpa_param is None:
            return DEFAULT_PUMP_AMPLITUDE
        else:
            return jpa_param.get("pump_amplitude") or DEFAULT_PUMP_AMPLITUDE

    def get_dc_voltage(self, mux: int) -> float:
        jpa_param = self.jpa_params.get(mux)
        if jpa_param is None:
            return DEFAULT_DC_VOLTAGE
        else:
            return jpa_param.get("dc_voltage") or DEFAULT_DC_VOLTAGE


class ExperimentSystem:
    def __init__(
        self,
        quantum_system: QuantumSystem,
        control_system: ControlSystem,
        wiring_info: WiringInfo,
        control_params: ControlParams,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
    ):
        self._quantum_system: Final = quantum_system
        self._control_system: Final = control_system
        self._wiring_info: Final = wiring_info
        self._control_params: Final = control_params
        self._targets_to_exclude: Final = targets_to_exclude or []
        self._qubit_port_set_map: Final = self._create_qubit_port_set_map()
        self.configure(mode=configuration_mode)

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

    def add_target(self, target: Target | CapTarget):
        if isinstance(target, Target):
            self._gen_target_dict[target.label] = target
        elif isinstance(target, CapTarget):
            self._cap_target_dict[target.label] = target

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

    def get_boxes_for_qubits(self, qubits: Collection[str]) -> list[Box]:
        box_ids = set()
        for qubit in qubits:
            ports = self.get_qubit_port_set(qubit)
            if ports is None:
                continue
            box_ids.add(ports.ctrl_port.box_id)
            box_ids.add(ports.read_out_port.box_id)
            box_ids.add(ports.read_in_port.box_id)
        return [self.get_box(box_id) for box_id in box_ids]

    def get_control_box_for_qubit(self, qubit: int | str) -> Box:
        if isinstance(qubit, int):
            qubit = self.qubits[qubit].label
        ports = self.get_qubit_port_set(qubit)
        if ports is None:
            raise ValueError(f"QubitPortSet for `{qubit}` not found.")
        return self.get_box(ports.ctrl_port.box_id)

    def get_readout_box_for_qubit(self, qubit: int | str) -> Box:
        if isinstance(qubit, int):
            qubit = self.qubits[qubit].label
        ports = self.get_qubit_port_set(qubit)
        if ports is None:
            raise ValueError(f"QubitPortSet for `{qubit}` not found.")
        return self.get_box(ports.read_out_port.box_id)

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

    def get_nco_frequency(self, label: str) -> float:
        target = self.get_target(label)
        return target.fine_frequency

    def get_awg_frequency(self, label: str) -> float:
        target = self.get_target(label)
        if target.channel.port.sideband == "U":
            f_awg = target.frequency - self.get_nco_frequency(label)
        elif target.channel.port.sideband == "L":
            f_awg = self.get_nco_frequency(label) - target.frequency
        elif target.channel.port.sideband is None:
            f_awg = self.get_nco_frequency(label)
        else:
            raise ValueError(
                f"Invalid sideband `{target.channel.port.sideband}` for target `{label}`.",
            )
        return round(f_awg, 10)

    def get_diff_frequency(self, label: str) -> float:
        target = self.get_target(label)
        f_diff = target.frequency - self.get_nco_frequency(label)
        return round(f_diff, 10)

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

    def get_mux_by_pump_port(self, port: GenPort) -> Mux | None:
        for mux, gen_port in self.wiring_info.pump:
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
        lo_freq: int | None,
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
                cap_channel.port.lo_freq = lo_freq  # type: ignore
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
                    cap_channel.port.lo_freq,  # type: ignore
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

    def configure(
        self,
        mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
    ):
        params = self.control_params

        self._gen_target_dict: dict[str, Target] = {}
        self._cap_target_dict: dict[str, CapTarget] = {}

        for box in self.boxes:
            for port in box.ports:
                if isinstance(port, GenPort):
                    port.rfswitch = "pass"
                    if port.type == PortType.READ_OUT:
                        self._configure_readout_port(
                            box=box,
                            port=port,
                            params=params,
                        )
                    elif port.type == PortType.CTRL:
                        self._configure_control_port(
                            box=box,
                            port=port,
                            params=params,
                            mode=mode,
                        )
                    elif port.type == PortType.PUMP:
                        self._configure_pump_port(
                            port=port,
                            params=params,
                        )
                elif isinstance(port, CapPort):
                    port.rfswitch = "open"
                    if port.type == PortType.READ_IN:
                        self._configure_capture_port(
                            box=box,
                            port=port,
                            params=params,
                        )

        self._gen_target_dict = dict(sorted(self._gen_target_dict.items()))
        self._cap_target_dict = dict(sorted(self._cap_target_dict.items()))

    def _configure_control_port(
        self,
        box: Box,
        port: GenPort,
        params: ControlParams,
        mode: Literal["ge-ef-cr", "ge-cr-cr"],
    ) -> None:
        qubit = self.get_qubit_by_control_port(port)
        if qubit is None or not qubit.is_valid:
            return

        if box.type == BoxType.QUEL1SE_R8:
            ssb = None
            min_frequency = 0.0
            vatt = None
        else:
            ssb = "L"
            min_frequency = 6.5e9
            vatt = params.get_control_vatt(qubit.label)

        config = self._create_control_configuration(
            mode=mode,
            qubit=qubit,
            n_channels=port.n_channels,
            ssb=ssb,
            min_frequency=min_frequency,
        )
        port.lo_freq = config["lo"]
        port.cnco_freq = config["cnco"]
        port.sideband = ssb
        port.vatt = vatt
        port.fullscale_current = params.get_control_fsc(qubit.label)
        for idx, gen_channel in enumerate(port.channels):
            gen_channel.fnco_freq = config["channels"][idx]["fnco"]

        if port.n_channels == 1:
            # ge only
            ge_target = Target.new_ge_target(
                qubit=qubit,
                channel=port.channels[0],
            )
            self._gen_target_dict[ge_target.label] = ge_target
        elif port.n_channels == 2:
            # ge
            ge_target = Target.new_ge_target(
                qubit=qubit,
                channel=port.channels[0],
            )
            self._gen_target_dict[ge_target.label] = ge_target
            # cr
            cr_target = Target.new_cr_target(
                control_qubit=qubit,
                channel=port.channels[1],
            )
            self._gen_target_dict[cr_target.label] = cr_target
            for spectator in self.get_spectator_qubits(qubit.label):
                cr_target = Target.new_cr_target(
                    control_qubit=qubit,
                    target_qubit=spectator,
                    channel=port.channels[1],
                )
                self._gen_target_dict[cr_target.label] = cr_target
        elif port.n_channels == 3:
            if mode == "ge-ef-cr":
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
            elif mode == "ge-cr-cr":
                for i, ch in config["channels"].items():
                    for label in ch["targets"]:
                        if match := re.match(r"^(Q\d+)$", label):
                            qubit_label = match.group(1)
                            qubit = self.get_qubit(qubit_label)
                            target = Target.new_ge_target(
                                qubit=qubit,
                                channel=port.channels[i],
                            )
                        elif match := re.match(r"^(Q\d+)-(Q\d+)$", label):
                            control_label = match.group(1)
                            target_label = match.group(2)
                            control_qubit = self.get_qubit(control_label)
                            target_qubit = self.get_qubit(target_label)
                            target = Target.new_cr_target(
                                control_qubit=control_qubit,
                                target_qubit=target_qubit,
                                channel=port.channels[i],
                            )
                        else:
                            raise ValueError(f"Invalid target label `{label}`.")
                        self._gen_target_dict[target.label] = target

    def _configure_pump_port(
        self,
        port: GenPort,
        params: ControlParams,
    ) -> None:
        mux = self.get_mux_by_pump_port(port)
        if mux is None:
            return

        frequency = params.get_pump_frequency(mux.index)
        ssb = "U"
        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            f=frequency * 1e9,
            ssb=ssb,
            cnco_center=CNCO_CETNER_READ,
        )
        fnco, _ = MixingUtil.calc_fnco(
            f=frequency * 1e9,
            ssb=ssb,
            lo=lo,
            cnco=cnco,
        )
        port.lo_freq = lo
        port.cnco_freq = cnco
        port.sideband = ssb
        port.vatt = params.get_pump_vatt(mux.index)
        port.fullscale_current = params.get_pump_fsc(mux.index)
        port.channels[0].fnco_freq = fnco

        pump_target = Target.new_pump_target(
            mux=mux,
            frequency=frequency,
            channel=port.channels[0],
        )
        self._gen_target_dict[pump_target.label] = pump_target

    def _configure_readout_port(
        self,
        box: Box,
        port: GenPort,
        params: ControlParams,
    ) -> None:
        mux = self.get_mux_by_readout_port(port)
        if mux is None:
            logger.warning(
                f"Readout port `{port.id}` not connected to a mux. Skipping configuration.",
            )
            return
        if mux.is_not_available:
            return

        if box.type == BoxType.QUEL1SE_R8:
            ssb = "L"
            cnco_center = CNCO_CETNER_READ_R8
        else:
            ssb = "U"
            cnco_center = CNCO_CETNER_READ

        config = self._create_readout_configuration(
            mux,
            ssb=ssb,
            cnco_center=cnco_center,
        )
        port.lo_freq = config["lo"]
        port.cnco_freq = config["cnco"]
        port.sideband = ssb
        port.vatt = params.get_readout_vatt(mux.index)
        port.fullscale_current = params.get_readout_fsc(mux.index)
        port.channels[0].fnco_freq = config["fnco"]

        for resonator in mux.resonators:
            if not resonator.is_valid:
                logger.debug(
                    f"Resonator `{resonator.label}` not valid. Skipping configuration.",
                )
                continue
            read_out_target = Target.new_read_target(
                resonator=resonator,
                channel=port.channels[0],
            )
            self._gen_target_dict[read_out_target.label] = read_out_target

    def _configure_capture_port(
        self,
        box: Box,
        port: CapPort,
        params: ControlParams,
    ) -> None:
        mux = self.get_mux_by_readout_port(port)
        if mux is None:
            logger.warning(
                f"Capture port `{port.id}` not connected to a mux. Skipping configuration.",
            )
            return

        if mux.is_not_available:
            return

        if box.type == BoxType.QUEL1SE_R8:
            ssb = "L"
            cnco_center = CNCO_CETNER_READ_R8
        else:
            ssb = "U"
            cnco_center = CNCO_CETNER_READ

        config = self._create_readout_configuration(
            mux,
            ssb=ssb,
            cnco_center=cnco_center,
        )
        port.lo_freq = config["lo"]
        port.cnco_freq = config["cnco"]
        for cap_channel in port.channels:
            cap_channel.fnco_freq = config["fnco"]
            cap_channel.ndelay = params.get_capture_delay(mux.index)

        for idx, resonator in enumerate(mux.resonators):
            if not resonator.is_valid:
                logger.debug(
                    f"Resonator `{resonator.label}` not valid. Skipping configuration.",
                )
                continue
            read_in_target = CapTarget.new_read_target(
                resonator=resonator,
                channel=port.channels[idx],
            )
            self._cap_target_dict[read_in_target.label] = read_in_target

    def _create_readout_configuration(
        self,
        mux: Mux,
        ssb: Literal["U", "L"] = "U",
        cnco_center: int = CNCO_CETNER_READ,
    ) -> dict:
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
        dict[str, int]
            The dictionary containing the lo, cnco, and fnco values.
        """
        resonators = [
            resonator
            for resonator in mux.resonators
            if resonator.is_valid and resonator.label not in self._targets_to_exclude
        ]
        freqs = [resonator.frequency * 1e9 for resonator in resonators]
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
        return {
            "lo": lo,
            "cnco": cnco,
            "fnco": fnco,
        }

    def _create_control_configuration(
        self,
        qubit: Qubit,
        n_channels: int,
        *,
        mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
        ssb: Literal["U", "L"] | None = "L",
        cnco_center: int = CNCO_CENTER_CTRL,
        min_frequency: float = 6.5e9,
    ) -> dict:
        """
        Finds the (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) values for the control qubit.

        Parameters
        ----------
        mode : Literal["ge-ef-cr", "ge-cr-cr"]
            The mode to configure the control qubit.
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
        dict[str, int]
            The dictionary containing the lo, cnco, and fnco values.
        """
        if n_channels == 1:
            f_target = qubit.frequency * 1e9
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
            return {
                "lo": lo,
                "cnco": cnco,
                "channels": {
                    0: {
                        "fnco": fnco,
                        "targets": [qubit.label],
                    },
                },
            }
        elif n_channels == 2:
            f_ge = qubit.frequency * 1e9
            f_ef = qubit.control_frequency_ef * 1e9
            spectators = self.get_spectator_qubits(qubit.label)
            f_CRs = [
                spectator.frequency * 1e9
                for spectator in spectators
                if spectator.frequency > 0
                and spectator.label not in self._targets_to_exclude
                and f"{qubit.label}-{spectator.label}" not in self._targets_to_exclude
            ]
            if not f_CRs:
                f_CRs = [f_ge]

            f_CR_max = max(f_CRs)
            if f_CR_max > f_ge:
                # if any CR is larger than GE, then let EF be the smallest
                if f_ef < min_frequency:
                    f_ef = f_ge
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
            fnco_ge, _ = MixingUtil.calc_fnco(
                f=(f_ge + f_ef) * 0.5,
                ssb=ssb,
                lo=lo,
                cnco=cnco,
            )
            fnco_CR, _ = MixingUtil.calc_fnco(
                f=f_CR,
                ssb=ssb,
                lo=lo,
                cnco=cnco,
            )
            return {
                "lo": lo,
                "cnco": cnco,
                "channels": {
                    0: {
                        "fnco": fnco_ge,
                        "targets": [qubit.label],
                    },
                    1: {
                        "fnco": fnco_CR,
                        "targets": [
                            f"{qubit.label}-{spectator.label}"
                            for spectator in self.get_spectator_qubits(qubit.label)
                        ],
                    },
                },
            }

        if mode == "ge-ef-cr":
            f_ge = qubit.frequency * 1e9
            f_ef = qubit.control_frequency_ef * 1e9
            spectators = self.get_spectator_qubits(qubit.label)
            f_CRs = [
                spectator.frequency * 1e9
                for spectator in spectators
                if spectator.frequency > 0
                and spectator.label not in self._targets_to_exclude
                and f"{qubit.label}-{spectator.label}" not in self._targets_to_exclude
            ]
            if not f_CRs:
                f_CRs = [f_ge]

            f_CR_max = max(f_CRs)
            if f_CR_max > f_ge:
                # if any CR is larger than GE, then let EF be the smallest
                if f_ef < min_frequency:
                    f_ef = f_ge
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
            return {
                "lo": lo,
                "cnco": cnco,
                "channels": {
                    0: {
                        "fnco": fnco_ge,
                        "targets": [qubit.label],
                    },
                    1: {
                        "fnco": fnco_ef,
                        "targets": [f"{qubit.label}-ef"],
                    },
                    2: {
                        "fnco": fnco_CR,
                        "targets": [
                            f"{qubit.label}-{spectator.label}"
                            for spectator in self.get_spectator_qubits(qubit.label)
                        ],
                    },
                },
            }
        elif mode == "ge-cr-cr":
            f_ge = qubit.frequency * 1e9
            f_ef = qubit.control_frequency_ef * 1e9

            spectators = self.get_spectator_qubits(qubit.label)
            cr_targets = [
                {
                    "label": f"{qubit.label}-{spectator.label}",
                    "frequency": spectator.frequency * 1e9,
                }
                for spectator in spectators
                if spectator.frequency > 0
                and spectator.label not in self._targets_to_exclude
                and f"{qubit.label}-{spectator.label}" not in self._targets_to_exclude
            ]

            if not cr_targets:
                cr_targets = [
                    {
                        "label": f"{qubit.label}",
                        "frequency": f_ge,
                    },
                    {
                        "label": f"{qubit.label}-ef",
                        "frequency": f_ef,
                    },
                ]

            group1, group2 = self._split_cr_target_group(cr_targets)
            f_CR_1 = np.mean([target["frequency"] for target in group1]).astype(float)
            f_CR_2 = np.mean([target["frequency"] for target in group2]).astype(float)
            f_min = min(f_ge, f_CR_1, f_CR_2)
            f_max = max(f_ge, f_CR_1, f_CR_2)
            lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                f=(f_min + f_max) / 2,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            fnco_ge, _ = MixingUtil.calc_fnco(f=f_ge, ssb=ssb, lo=lo, cnco=cnco)
            fnco_CR_1, _ = MixingUtil.calc_fnco(f=f_CR_1, ssb=ssb, lo=lo, cnco=cnco)
            fnco_CR_2, _ = MixingUtil.calc_fnco(f=f_CR_2, ssb=ssb, lo=lo, cnco=cnco)
            return {
                "lo": lo,
                "cnco": cnco,
                "channels": {
                    0: {
                        "fnco": fnco_ge,
                        "targets": [qubit.label],
                    },
                    1: {
                        "fnco": fnco_CR_1,
                        "targets": [target["label"] for target in group1],
                    },
                    2: {
                        "fnco": fnco_CR_2,
                        "targets": [target["label"] for target in group2],
                    },
                },
            }
        else:
            raise ValueError("Invalid mode.")

    def _split_cr_target_group(
        self,
        group: list[dict[str, float]],
    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        group = sorted(group, key=lambda x: x["frequency"])
        if len(group) == 0:
            raise ValueError("No CR target found.")
        elif len(group) == 1:
            return [group[0]], [group[0]]
        elif len(group) == 2:
            return [group[0]], [group[1]]
        elif len(group) == 3:
            split_options = [
                ([group[0], group[1]], [group[2]]),
                ([group[0]], [group[1], group[2]]),
            ]
        elif len(group) == 4:
            split_options = [
                ([group[0], group[1]], [group[2], group[3]]),
                ([group[0], group[1], group[2]], [group[3]]),
                ([group[0]], [group[1], group[2], group[3]]),
            ]
        else:
            raise ValueError("Too many CR targets.")

        best_split = None
        best_max_bandwidth = float("inf")

        for group1, group2 in split_options:
            f_min1 = min(target["frequency"] for target in group1)
            f_max1 = max(target["frequency"] for target in group1)
            f_min2 = min(target["frequency"] for target in group2)
            f_max2 = max(target["frequency"] for target in group2)
            bandwidth1 = f_max1 - f_min1 if len(group1) > 1 else 0
            bandwidth2 = f_max2 - f_min2 if len(group2) > 1 else 0
            max_band = max(bandwidth1, bandwidth2)

            if max_band < best_max_bandwidth:
                best_max_bandwidth = max_band
                best_split = (group1, group2)

        if best_split is None:
            raise ValueError("No split found.")

        return best_split

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
        center_freqs_by_count = []
        for f in search_range:
            valid_f_CRs = [f_CR for f_CR in f_CRs if f - AWG_MAX <= f_CR <= f + AWG_MAX]
            if not valid_f_CRs:
                continue
            center = (min(valid_f_CRs) + max(valid_f_CRs)) / 2
            center_freqs_by_count.append((len(valid_f_CRs), center))
        if not center_freqs_by_count:
            return f_coarse
        # sort by count and then by frequency
        center_freqs_by_count.sort(key=lambda x: (x[0], x[1]), reverse=True)
        # choose the one with the highest count and frequency
        center_freq = int(center_freqs_by_count[0][1])
        # round to the nearest NCO step
        center_freq = round(center_freq / NCO_STEP) * NCO_STEP
        # clip to the possible range
        center_freq = max(min_center_freq, min(center_freq, max_center_freq))
        return center_freq


class MixingUtil:
    @staticmethod
    def calc_lo_cnco(
        f: float,
        cnco_center: int,
        ssb: Literal["U", "L"] | None,
        lo_step: int = LO_STEP,
        nco_step: int = NCO_STEP,
    ) -> tuple[int | None, int, int]:
        if ssb is None:
            lo = None
            cnco = round(f / nco_step) * nco_step
            f_mix = cnco
        else:
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
        ssb: Literal["U", "L"] | None,
        lo: int | None,
        cnco: int,
        nco_step: int = NCO_STEP,
    ) -> tuple[int, int]:
        if ssb is None and lo is None:
            fnco = round((f - cnco) / nco_step) * nco_step
            f_mix = cnco + fnco
        elif lo is None:
            raise ValueError("LO frequency is required when SSB is not None.")
        elif ssb is None:
            raise ValueError("SSB is required when LO frequency is not None.")
        else:
            if ssb == "U":
                fnco = round((f - (lo + cnco)) / nco_step) * nco_step
            elif ssb == "L":
                fnco = round(((lo - cnco) - f) / nco_step) * nco_step
            else:
                raise ValueError("Invalid SSB")
            f_mix = lo + cnco + fnco if ssb == "U" else lo - cnco - fnco
        return fnco, f_mix
