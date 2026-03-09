"""Experiment system configuration and target mapping utilities."""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Sequence
from typing import Final, Literal

import numpy as np
from typing_extensions import TypedDict, deprecated

from qubex.backend.quel1.quel1_backend_constants import (
    AWG_MAX_HZ,
    CNCO_CENTER_CTRL_HZ,
    CNCO_CENTER_READ_HZ,
    DEFAULT_CAPTURE_DELAY,
    DEFAULT_CAPTURE_DELAY_WORD,
    DEFAULT_CNCO_FREQUENCY_HZ,
    DEFAULT_CONTROL_AMPLITUDE,
    DEFAULT_CONTROL_FSC,
    DEFAULT_CONTROL_VATT,
    DEFAULT_DC_VOLTAGE,
    DEFAULT_FNCO_FREQUENCY_HZ,
    DEFAULT_LO_FREQUENCY_HZ,
    DEFAULT_PUMP_AMPLITUDE,
    DEFAULT_PUMP_FREQUENCY_GHZ,
    DEFAULT_PUMP_FSC,
    DEFAULT_PUMP_VATT,
    DEFAULT_READOUT_AMPLITUDE,
    DEFAULT_READOUT_FSC,
    DEFAULT_READOUT_VATT,
    FNCO_MAX_HZ,
    LO_STEP_HZ,
    NCO_STEP_HZ,
)
from qubex.core import MutableModel
from qubex.typing import ConfigurationMode

from .control_system import (
    Box,
    BoxType,
    CapPort,
    ControlSystem,
    GenChannel,
    GenPort,
    PortType,
)
from .quantum_system import Chip, Mux, QuantumSystem, Qubit, Resonator
from .target import CapTarget, Target, TargetType
from .target_registry import TargetRegistry

logger = logging.getLogger(__name__)


class WiringInfo(MutableModel):
    """Wiring relationships between qubits, muxes, and ports."""

    ctrl: list[tuple[Qubit, GenPort]]
    read_out: list[tuple[Mux, GenPort]]
    read_in: list[tuple[Mux, CapPort]]
    pump: list[tuple[Mux, GenPort]]


class QubitPortSet(MutableModel):
    """Port assignments for a single qubit."""

    ctrl_port: GenPort
    read_out_port: GenPort
    read_in_port: CapPort


class JPAParam(TypedDict):
    """JPA parameter dictionary for a mux."""

    dc_voltage: float | None
    pump_frequency: float | None
    pump_amplitude: float | None


class ReadoutMixingConfig(TypedDict):
    """Readout mixing configuration for one mux."""

    lo: int | None
    cnco: int
    fnco: int


class ControlChannelConfig(TypedDict):
    """Per-channel control mixing configuration."""

    fnco: int
    targets: list[str]


class ControlMixingConfig(TypedDict):
    """Control mixing configuration for one qubit."""

    lo: int | None
    cnco: int
    channels: dict[int, ControlChannelConfig]


class CrTargetConfig(TypedDict):
    """CR target label and frequency pair."""

    label: str
    frequency: float


class ControlParams(MutableModel):
    """Control parameters and access helpers for the system."""

    frequency_margin: dict[str, float]
    control_amplitude: dict[str, float]
    readout_amplitude: dict[str, float]
    control_vatt: dict[str, int | None]
    readout_vatt: dict[int, int]
    pump_vatt: dict[int, int]
    control_fsc: dict[str, int]
    readout_fsc: dict[int, int]
    pump_fsc: dict[int, int]
    capture_delay: dict[int, int]
    capture_delay_word: dict[int, int]
    jpa_params: dict[int, JPAParam | None]

    _DEFAULT_FREQUENCY_MARGIN_BY_TYPE: Final[dict[str, float]] = {
        TargetType.READ.value: 0.1,
        TargetType.CTRL_GE.value: 0.1,
        TargetType.CTRL_EF.value: 0.1,
        TargetType.CTRL_CR.value: 0.1,
        TargetType.PUMP.value: 0.1,
    }

    def get_frequency_margin(self, target_type: TargetType | str) -> float:
        """Return the QuEL-3 deploy frequency margin for a target type."""
        key = target_type.value if isinstance(target_type, TargetType) else target_type
        return self.frequency_margin.get(
            key,
            self._DEFAULT_FREQUENCY_MARGIN_BY_TYPE.get(key, 0.1),
        )

    def get_control_amplitude(self, qubit: str) -> float:
        """Return the control amplitude for a qubit."""
        return self.control_amplitude.get(qubit, DEFAULT_CONTROL_AMPLITUDE)

    def get_ef_control_amplitude(self, qubit: str) -> float:
        """Return the ef control amplitude for a qubit."""
        return self.get_control_amplitude(qubit) / np.sqrt(2)

    def get_readout_amplitude(self, qubit: str) -> float:
        """Return the readout amplitude for a qubit."""
        return self.readout_amplitude.get(qubit, DEFAULT_READOUT_AMPLITUDE)

    def get_control_vatt(self, qubit: str) -> int | None:
        """Return the control VATT for a qubit."""
        return self.control_vatt.get(qubit, DEFAULT_CONTROL_VATT)

    def get_readout_vatt(self, mux: int) -> int:
        """Return the readout VATT for a mux."""
        return self.readout_vatt.get(mux, DEFAULT_READOUT_VATT)

    def get_pump_vatt(self, mux: int) -> int:
        """Return the pump VATT for a mux."""
        return self.pump_vatt.get(mux, DEFAULT_PUMP_VATT)

    def get_control_fsc(self, qubit: str) -> int:
        """Return the control FSC for a qubit."""
        return self.control_fsc.get(qubit, DEFAULT_CONTROL_FSC)

    def get_readout_fsc(self, mux: int) -> int:
        """Return the readout FSC for a mux."""
        return self.readout_fsc.get(mux, DEFAULT_READOUT_FSC)

    def get_pump_fsc(self, mux: int) -> int:
        """Return the pump FSC for a mux."""
        return self.pump_fsc.get(mux, DEFAULT_PUMP_FSC)

    def get_capture_delay(self, mux: int) -> int:
        """Return the capture delay for a mux."""
        return self.capture_delay.get(mux, DEFAULT_CAPTURE_DELAY)

    def get_capture_delay_word(self, mux: int) -> int:
        """Return the capture delay word for a mux."""
        return self.capture_delay_word.get(mux, DEFAULT_CAPTURE_DELAY_WORD)

    def get_pump_frequency(self, mux: int) -> float:
        """Return the pump frequency for a mux."""
        jpa_param = self.jpa_params.get(mux)
        if jpa_param is None:
            return DEFAULT_PUMP_FREQUENCY_GHZ
        else:
            return jpa_param.get("pump_frequency") or DEFAULT_PUMP_FREQUENCY_GHZ

    def get_pump_amplitude(self, mux: int) -> float:
        """Return the pump amplitude for a mux."""
        jpa_param = self.jpa_params.get(mux)
        if jpa_param is None:
            return DEFAULT_PUMP_AMPLITUDE
        else:
            return jpa_param.get("pump_amplitude") or DEFAULT_PUMP_AMPLITUDE

    def get_dc_voltage(self, mux: int) -> float:
        """Return the DC voltage for a mux."""
        jpa_param = self.jpa_params.get(mux)
        if jpa_param is None:
            return DEFAULT_DC_VOLTAGE
        else:
            return jpa_param.get("dc_voltage") or DEFAULT_DC_VOLTAGE


class ExperimentSystem:
    """Experiment system containing wiring and target mappings."""

    def __init__(
        self,
        quantum_system: QuantumSystem,
        control_system: ControlSystem,
        wiring_info: WiringInfo,
        control_params: ControlParams,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: ConfigurationMode = "ge-cr-cr",
    ):
        self._quantum_system: Final = quantum_system
        self._control_system: Final = control_system
        self._wiring_info: Final = wiring_info
        self._control_params: Final = control_params
        self._targets_to_exclude: Final = targets_to_exclude or []
        self._configuration_mode: ConfigurationMode = configuration_mode
        self._qubit_port_set_map: Final = self._create_qubit_port_set_map()
        boxes_to_configure = [box for box in self.boxes if box.type != BoxType.QUEL3]
        if boxes_to_configure:
            self._configure_ports(
                mode=self._configuration_mode,
                boxes=boxes_to_configure,
            )
        self._target_registry = self._build_target_registry()

    @property
    def hash(self) -> int:
        """Return a hash representing the system configuration."""
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
        """Return the underlying quantum system."""
        return self._quantum_system

    @property
    def control_system(self) -> ControlSystem:
        """Return the underlying control system."""
        return self._control_system

    @property
    def wiring_info(self) -> WiringInfo:
        """Return wiring information for the system."""
        return self._wiring_info

    @property
    def control_params(self) -> ControlParams:
        """Return the control parameters."""
        return self._control_params

    @property
    def chip(self) -> Chip:
        """Return the chip model."""
        return self.quantum_system.chip

    @property
    def qubits(self) -> list[Qubit]:
        """Return the list of qubits."""
        return self.quantum_system.qubits

    @property
    def resonators(self) -> list[Resonator]:
        """Return the list of resonators."""
        return self.quantum_system.resonators

    @property
    def boxes(self) -> list[Box]:
        """Return the list of control boxes."""
        return self.control_system.boxes

    @property
    def ge_targets(self) -> list[Target]:
        """Return the ge targets."""
        return [target for target in self.gen_targets.values() if target.is_ge]

    @property
    def ef_targets(self) -> list[Target]:
        """Return the ef targets."""
        return [target for target in self.gen_targets.values() if target.is_ef]

    @property
    def cr_targets(self) -> list[Target]:
        """Return the cr targets."""
        return [target for target in self.gen_targets.values() if target.is_cr]

    @property
    def ctrl_targets(self) -> list[Target]:
        """Return the control targets."""
        return self.ge_targets + self.ef_targets + self.cr_targets

    @property
    def read_out_targets(self) -> list[Target]:
        """Return the readout targets."""
        return [target for target in self.gen_targets.values() if target.is_read]

    @property
    def gen_targets(self) -> dict[str, Target]:
        """Return generator targets keyed by label."""
        return self.target_registry.gen_targets

    @property
    def cap_targets(self) -> dict[str, CapTarget]:
        """Return capture targets keyed by label."""
        return self.target_registry.cap_targets

    @property
    def targets(self) -> list[Target]:
        """Return all generator targets."""
        return list(self.gen_targets.values())

    @property
    def read_in_targets(self) -> list[CapTarget]:
        """Return all capture targets."""
        return list(self.cap_targets.values())

    @property
    def all_targets(self) -> list[Target | CapTarget]:
        """Return all generator and capture targets."""
        return self.targets + self.read_in_targets

    @property
    def target_registry(self) -> TargetRegistry:
        """Return registry used for target-label resolution."""
        return self._target_registry

    def add_target(self, target: Target | CapTarget) -> None:
        """Add a target to the system mapping."""
        self._target_registry.register(target)

    def get_mux(self, label: int | str) -> Mux:
        """Return a mux by label or index."""
        return self.quantum_system.get_mux(label)

    def get_qubit(self, label: int | str) -> Qubit:
        """Return a qubit by label or index."""
        return self.quantum_system.get_qubit(label)

    def get_resonator(self, label: int | str) -> Resonator:
        """Return a resonator by label or index."""
        return self.quantum_system.get_resonator(label)

    def get_spectator_qubits(
        self,
        qubit: int | str,
        *,
        in_same_mux: bool = False,
    ) -> list[Qubit]:
        """Return spectator qubits for the specified qubit."""
        return self.quantum_system.get_spectator_qubits(qubit, in_same_mux=in_same_mux)

    def get_box(self, box_id: str) -> Box:
        """Return a control box by ID."""
        return self.control_system.get_box(box_id)

    def get_boxes_for_qubits(self, qubits: Collection[str]) -> list[Box]:
        """Return control boxes associated with the given qubits."""
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
        """Return the control box for a qubit."""
        if isinstance(qubit, int):
            qubit = self.qubits[qubit].label
        ports = self.get_qubit_port_set(qubit)
        if ports is None:
            raise ValueError(f"QubitPortSet for `{qubit}` not found.")
        return self.get_box(ports.ctrl_port.box_id)

    def get_readout_box_for_qubit(self, qubit: int | str) -> Box:
        """Return the readout box for a qubit."""
        if isinstance(qubit, int):
            qubit = self.qubits[qubit].label
        ports = self.get_qubit_port_set(qubit)
        if ports is None:
            raise ValueError(f"QubitPortSet for `{qubit}` not found.")
        return self.get_box(ports.read_out_port.box_id)

    def get_target(self, label: str) -> Target:
        """Return a generator target by label."""
        try:
            return self.target_registry.get_gen_target(label)
        except KeyError:
            raise KeyError(f"Target `{label}` not found.") from None

    def resolve_qubit_label(self, label: str) -> str:
        """Resolve qubit label via target registry."""
        return self.target_registry.resolve_qubit_label(label, allow_legacy=True)

    def resolve_ge_label(self, label: str) -> str:
        """Resolve GE label via target registry."""
        return self.target_registry.resolve_ge_label(label, allow_legacy=True)

    def resolve_ef_label(self, label: str) -> str:
        """Resolve EF label via target registry."""
        return self.target_registry.resolve_ef_label(label, allow_legacy=True)

    def resolve_read_label(self, label: str) -> str:
        """Resolve readout label via target registry."""
        return self.target_registry.resolve_read_label(label, allow_legacy=True)

    def resolve_cr_label(self, label: str) -> str:
        """Resolve CR label via target registry."""
        return self.target_registry.resolve_cr_label(label)

    def resolve_cr_pair(self, label: str) -> tuple[str, str]:
        """Resolve CR pair via target registry."""
        return self.target_registry.resolve_cr_pair(label, allow_legacy=True)

    def ordered_qubit_labels(self, labels: Sequence[str]) -> list[str]:
        """Return qubit labels in first appearance order."""
        return list(dict.fromkeys(self.resolve_qubit_label(label) for label in labels))

    def get_cap_target(self, label: str) -> CapTarget:
        """Return a capture target by label."""
        try:
            return self.target_registry.get_cap_target(label)
        except KeyError:
            raise KeyError(f"CapTarget `{label}` not found.") from None

    def get_ge_target(self, label: str) -> Target:
        """Return a ge target by label."""
        return self.get_target(self.resolve_ge_label(label))

    def get_ef_target(self, label: str) -> Target:
        """Return an ef target by label."""
        return self.get_target(self.resolve_ef_label(label))

    def get_cr_target(self, label: str) -> Target:
        """Return a cr target by label."""
        return self.get_target(self.resolve_cr_label(label))

    def get_read_out_target(self, label: str) -> Target:
        """Return a readout target by label."""
        return self.get_target(self.resolve_read_label(label))

    def get_read_in_target(self, label: str) -> CapTarget:
        """Return a read-in target by label."""
        return self.get_cap_target(self.resolve_read_label(label))

    def get_qubit_port_set(self, qubit: int | str) -> QubitPortSet | None:
        """Return the port set for a qubit if available."""
        if isinstance(qubit, int):
            qubit = self.qubits[qubit].label
        return self._qubit_port_set_map.get(qubit)

    def get_control_port(self, qubit: int | str) -> GenPort:
        """Return the control port for a qubit."""
        ports = self.get_qubit_port_set(qubit)
        if ports is None:
            raise ValueError(f"Qubit `{qubit}` not found.")
        return ports.ctrl_port

    def get_nco_frequency(self, label: str) -> float:
        """Return the NCO frequency for a target."""
        target = self.get_target(label)
        return target.fine_frequency

    def get_awg_frequency(self, label: str) -> float:
        """Return the AWG frequency for a target."""
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
        """Return the difference frequency for a target."""
        target = self.get_target(label)
        f_diff = target.frequency - self.get_nco_frequency(label)
        return round(f_diff, 10)

    def get_mux_by_readout_port(self, port: GenPort | CapPort) -> Mux | None:
        """Return the mux associated with a readout port."""
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
        """Return the mux associated with a pump port."""
        for mux, gen_port in self.wiring_info.pump:
            if gen_port == port:
                return mux
        return None

    def get_qubit_by_control_port(self, port: GenPort) -> Qubit | None:
        """Return the qubit associated with a control port."""
        for qubit, gen_port in self.wiring_info.ctrl:
            if gen_port == port:
                return qubit
        return None

    def get_mux_by_qubit(self, label: str) -> Mux:
        """Return the mux associated with a qubit."""
        ports = self.get_qubit_port_set(label)
        if ports is None:
            raise ValueError(f"Qubit `{label}` not found.")
        mux = self.get_mux_by_readout_port(ports.read_out_port)
        if mux is None:
            raise ValueError(f"No mux found for qubit `{label}`.")
        return mux

    def get_readout_pair(self, port: CapPort) -> GenPort:
        """Return the generator port paired with a capture port."""
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
    ) -> None:
        """Update target frequencies in place."""
        for label, frequency in frequencies.items():
            target = self.get_target(label)
            target.frequency = frequency

    def update_port_params(
        self,
        label: str,
        *,
        lo_freq: int | None,
        cnco_freq: int | None,
        fnco_freq: int | None,
    ) -> None:
        """Update LO/CNCO/FNCO parameters for a target port."""
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

    @deprecated(
        "Use `SystemManager.load(..., configuration_mode=...)` to rebuild configuration."
    )
    def configure(
        self,
        mode: ConfigurationMode = "ge-cr-cr",
    ) -> None:
        """Configure port/channel parameters and rebuild target mappings."""
        self._configuration_mode = mode
        self._configure_ports(mode=mode)
        self._target_registry = self._build_target_registry(mode=mode)

    def _configure_ports(
        self,
        mode: ConfigurationMode | None = None,
        boxes: Sequence[Box] | None = None,
    ) -> None:
        """Apply per-port hardware model parameters for the specified mode."""
        if mode is None:
            mode = self._configuration_mode
        else:
            self._configuration_mode = mode
        params = self.control_params
        for box in self.boxes if boxes is None else boxes:
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
                    elif port.type == PortType.MNTR_IN:
                        self._configure_monitor_port(
                            box=box,
                            port=port,
                        )

    def _build_target_registry(
        self,
        mode: ConfigurationMode | None = None,
    ) -> TargetRegistry:
        """Build target registry from the current control/quantum model state."""
        if mode is None:
            mode = self._configuration_mode
        params = self.control_params
        gen_targets: dict[str, Target] = {}
        cap_targets: dict[str, CapTarget] = {}

        for box in self.boxes:
            for port in box.ports:
                if isinstance(port, GenPort):
                    if port.type == PortType.READ_OUT:
                        self._build_readout_targets(
                            port=port,
                            gen_targets=gen_targets,
                        )
                    elif port.type == PortType.CTRL:
                        self._build_control_targets(
                            box=box,
                            port=port,
                            mode=mode,
                            gen_targets=gen_targets,
                        )
                    elif port.type == PortType.PUMP:
                        self._build_pump_targets(
                            port=port,
                            params=params,
                            gen_targets=gen_targets,
                        )
                elif isinstance(port, CapPort):
                    if port.type == PortType.READ_IN:
                        self._build_capture_targets(
                            port=port,
                            cap_targets=cap_targets,
                        )

        return TargetRegistry(
            gen_targets=dict(sorted(gen_targets.items())),
            cap_targets=dict(sorted(cap_targets.items())),
        )

    def _configure_control_port(
        self,
        box: Box,
        port: GenPort,
        params: ControlParams,
        mode: ConfigurationMode,
    ) -> None:
        qubit = self.get_qubit_by_control_port(port)
        if qubit is None or not qubit.is_valid:
            return

        traits = box.traits
        ssb = traits.ctrl_ssb
        min_frequency = traits.ctrl_min_frequency_hz
        vatt = params.get_control_vatt(qubit.label) if traits.ctrl_uses_vatt else None

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
            cnco_center=CNCO_CENTER_READ_HZ,
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

        traits = box.traits
        ssb = traits.readout_ssb
        cnco_center = traits.readout_cnco_center

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

        traits = box.traits
        ssb = traits.readout_ssb
        cnco_center = traits.readout_cnco_center

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

    def _configure_monitor_port(
        self,
        box: Box,
        port: CapPort,
    ) -> None:
        """Initialize monitor input with default frequencies for hardware sync."""
        port.lo_freq = DEFAULT_LO_FREQUENCY_HZ
        port.cnco_freq = DEFAULT_CNCO_FREQUENCY_HZ
        for channel in port.channels:
            channel.fnco_freq = DEFAULT_FNCO_FREQUENCY_HZ
            channel.ndelay = DEFAULT_CAPTURE_DELAY

    def _build_control_targets(
        self,
        box: Box,
        port: GenPort,
        mode: ConfigurationMode,
        gen_targets: dict[str, Target],
    ) -> None:
        """Build generator targets for one control port."""
        qubit = self.get_qubit_by_control_port(port)
        if qubit is None or not qubit.is_valid:
            return

        if port.n_channels == 1:
            ge_target = Target.new_ge_target(
                qubit=qubit,
                channel=port.channels[0],
            )
            gen_targets[ge_target.label] = ge_target
            return

        if port.n_channels == 2:
            ge_target = Target.new_ge_target(
                qubit=qubit,
                channel=port.channels[0],
            )
            gen_targets[ge_target.label] = ge_target
            traits = box.traits
            config = self._create_control_configuration(
                mode=mode,
                qubit=qubit,
                n_channels=port.n_channels,
                ssb=traits.ctrl_ssb,
                min_frequency=traits.ctrl_min_frequency_hz,
            )
            cr_target = self._new_default_cr_target(
                control_qubit=qubit,
                channel=port.channels[1],
                lo=config["lo"],
                cnco=config["cnco"],
                fnco=config["channels"][1]["fnco"],
                ssb=traits.ctrl_ssb,
            )
            gen_targets[cr_target.label] = cr_target
            for spectator in self.get_spectator_qubits(qubit.label):
                cr_target = Target.new_cr_target(
                    control_qubit=qubit,
                    target_qubit=spectator,
                    channel=port.channels[1],
                )
                gen_targets[cr_target.label] = cr_target
            return

        if port.n_channels != 3:
            return

        if mode == "ge-ef-cr":
            ge_target = Target.new_ge_target(
                qubit=qubit,
                channel=port.channels[0],
            )
            ef_target = Target.new_ef_target(
                qubit=qubit,
                channel=port.channels[1],
            )
            traits = box.traits
            config = self._create_control_configuration(
                mode=mode,
                qubit=qubit,
                n_channels=port.n_channels,
                ssb=traits.ctrl_ssb,
                min_frequency=traits.ctrl_min_frequency_hz,
            )
            cr_target = self._new_default_cr_target(
                control_qubit=qubit,
                channel=port.channels[2],
                lo=config["lo"],
                cnco=config["cnco"],
                fnco=config["channels"][2]["fnco"],
                ssb=traits.ctrl_ssb,
            )
            gen_targets[ge_target.label] = ge_target
            gen_targets[ef_target.label] = ef_target
            gen_targets[cr_target.label] = cr_target
            for spectator in self.get_spectator_qubits(qubit.label):
                cr_target = Target.new_cr_target(
                    control_qubit=qubit,
                    target_qubit=spectator,
                    channel=port.channels[2],
                )
                gen_targets[cr_target.label] = cr_target
            return

        if mode != "ge-cr-cr":
            return

        traits = box.traits
        config = self._create_control_configuration(
            mode=mode,
            qubit=qubit,
            n_channels=port.n_channels,
            ssb=traits.ctrl_ssb,
            min_frequency=traits.ctrl_min_frequency_hz,
        )
        for channel_idx, channel_config in config["channels"].items():
            for target_label in channel_config["targets"]:
                if match := re.match(r"^(Q\d+)$", target_label):
                    ge_qubit = self.get_qubit(match.group(1))
                    target = Target.new_ge_target(
                        qubit=ge_qubit,
                        channel=port.channels[channel_idx],
                    )
                elif match := re.match(r"^(Q\d+)-(Q\d+)$", target_label):
                    control_qubit = self.get_qubit(match.group(1))
                    target_qubit = self.get_qubit(match.group(2))
                    target = Target.new_cr_target(
                        control_qubit=control_qubit,
                        target_qubit=target_qubit,
                        channel=port.channels[channel_idx],
                    )
                else:
                    raise ValueError(f"Invalid target label `{target_label}`.")
                gen_targets[target.label] = target

    def _new_default_cr_target(
        self,
        *,
        control_qubit: Qubit,
        channel: GenChannel,
        lo: int | None,
        cnco: int,
        fnco: int,
        ssb: Literal["U", "L"] | None,
    ) -> Target:
        """Create default CR target without requiring configured channel frequencies."""
        frequency_hz = self._resolve_mixed_frequency_hz(
            lo=lo,
            cnco=cnco,
            fnco=fnco,
            ssb=ssb,
        )
        return Target.new_target(
            label=Target.cr_label(control_qubit.label),
            frequency=round(frequency_hz * 1e-9, 6),
            object=control_qubit,
            channel=channel,
            type=TargetType.CTRL_CR,
        )

    @staticmethod
    def _resolve_mixed_frequency_hz(
        *,
        lo: int | None,
        cnco: int,
        fnco: int,
        ssb: Literal["U", "L"] | None,
    ) -> float:
        """Resolve output frequency from LO/CNCO/FNCO and sideband."""
        nco = cnco + fnco
        if lo is None and ssb is None:
            return float(nco)
        if lo is None:
            raise ValueError("LO frequency is required when sideband is set.")
        if ssb is None:
            raise ValueError("Sideband is required when LO frequency is set.")
        if ssb == "U":
            return float(lo + nco)
        if ssb == "L":
            return float(lo - nco)
        raise ValueError(f"Invalid sideband: {ssb}")

    def _build_pump_targets(
        self,
        port: GenPort,
        params: ControlParams,
        gen_targets: dict[str, Target],
    ) -> None:
        """Build generator targets for one pump port."""
        mux = self.get_mux_by_pump_port(port)
        if mux is None:
            return
        frequency = params.get_pump_frequency(mux.index)
        pump_target = Target.new_pump_target(
            mux=mux,
            frequency=frequency,
            channel=port.channels[0],
        )
        gen_targets[pump_target.label] = pump_target

    def _build_readout_targets(
        self,
        port: GenPort,
        gen_targets: dict[str, Target],
    ) -> None:
        """Build generator readout targets for one readout port."""
        mux = self.get_mux_by_readout_port(port)
        if mux is None or mux.is_not_available:
            return
        for resonator in mux.resonators:
            if not resonator.is_valid:
                continue
            read_out_target = Target.new_read_target(
                resonator=resonator,
                channel=port.channels[0],
            )
            gen_targets[read_out_target.label] = read_out_target

    def _build_capture_targets(
        self,
        port: CapPort,
        cap_targets: dict[str, CapTarget],
    ) -> None:
        """Build capture targets for one read-in port."""
        mux = self.get_mux_by_readout_port(port)
        if mux is None or mux.is_not_available:
            return
        for idx, resonator in enumerate(mux.resonators):
            if not resonator.is_valid:
                continue
            read_in_target = CapTarget.new_read_target(
                resonator=resonator,
                channel=port.channels[idx],
            )
            cap_targets[read_in_target.label] = read_in_target

    def _create_readout_configuration(
        self,
        mux: Mux,
        ssb: Literal["U", "L"] | None = "U",
        cnco_center: int = CNCO_CENTER_READ_HZ,
    ) -> ReadoutMixingConfig:
        """
        Find the (lo, cnco, fnco) values for the readout mux.

        Parameters
        ----------
        mux : Mux
            Readout mux.

        ssb : Literal["U", "L"] | None, optional
            Sideband, by default "U".

        cnco_center : int, optional
            Center frequency of the CNCO, by default CNCO_CETNER_READ.


        Returns
        -------
        ReadoutMixingConfig
            Dictionary containing the lo, cnco, and fnco values.

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
        mode: ConfigurationMode = "ge-cr-cr",
        ssb: Literal["U", "L"] | None = "L",
        cnco_center: int = CNCO_CENTER_CTRL_HZ,
        min_frequency: float = 6.5e9,
    ) -> ControlMixingConfig:
        """
        Find the (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) values for the control qubit.

        Parameters
        ----------
        mode : ConfigurationMode
            Mode to configure the control qubit.

        qubit : Qubit
            Control qubit.

        n_channels : int
            Number of channels.

        ssb : Literal["U", "L"] | None, optional
            Sideband, by default "L".

        cnco_center : int, optional
            Center frequency of the CNCO, by default CNCO_CENTER_CTRL_HZ.


        Returns
        -------
        ControlMixingConfig
            Dictionary containing the lo, cnco, and fnco values.

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
                    f=f_ef + FNCO_MAX_HZ,
                    ssb=ssb,
                    cnco_center=cnco_center,
                )
                f_CRs_valid = [
                    f for f in f_CRs if f < f_coarse + FNCO_MAX_HZ + AWG_MAX_HZ
                ]
            else:
                # if all CRs are smaller than GE, then let GE be the largest
                lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                    f=f_ge - FNCO_MAX_HZ,
                    ssb=ssb,
                    cnco_center=cnco_center,
                )
                f_CRs_valid = [
                    f for f in f_CRs if f > f_coarse - FNCO_MAX_HZ - AWG_MAX_HZ
                ]
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
                    f=f_ef + FNCO_MAX_HZ,
                    ssb=ssb,
                    cnco_center=cnco_center,
                )
                f_CRs_valid = [
                    f for f in f_CRs if f < f_coarse + FNCO_MAX_HZ + AWG_MAX_HZ
                ]
            else:
                # if all CRs are smaller than GE, then let GE be the largest
                lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                    f=f_ge - FNCO_MAX_HZ,
                    ssb=ssb,
                    cnco_center=cnco_center,
                )
                f_CRs_valid = [
                    f for f in f_CRs if f > f_coarse - FNCO_MAX_HZ - AWG_MAX_HZ
                ]
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
            cr_targets: list[CrTargetConfig] = [
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
        group: list[CrTargetConfig],
    ) -> tuple[list[CrTargetConfig], list[CrTargetConfig]]:
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
        min_center_freq = f_coarse - FNCO_MAX_HZ
        max_center_freq = f_coarse + FNCO_MAX_HZ

        # search range
        search_range = np.arange(
            max(min(f_CRs), min_center_freq),
            min(max(f_CRs), max_center_freq) + 1,
            NCO_STEP_HZ,
        )
        # count the number of CR frequencies within the range of each search point
        center_freqs_by_count = []
        for f in search_range:
            valid_f_CRs = [
                f_CR for f_CR in f_CRs if f - AWG_MAX_HZ <= f_CR <= f + AWG_MAX_HZ
            ]
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
        center_freq = round(center_freq / NCO_STEP_HZ) * NCO_STEP_HZ
        # clip to the possible range
        center_freq = max(min_center_freq, min(center_freq, max_center_freq))
        return center_freq


class MixingUtil:
    """Utility helpers for LO/NCO mixing calculations."""

    @staticmethod
    def calc_lo_cnco(
        f: float,
        cnco_center: int,
        ssb: Literal["U", "L"] | None,
        lo_step: int = LO_STEP_HZ,
        nco_step: int = NCO_STEP_HZ,
    ) -> tuple[int | None, int, int]:
        """Calculate LO/CNCO settings for a target frequency."""
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
        nco_step: int = NCO_STEP_HZ,
    ) -> tuple[int, int]:
        """Calculate FNCO settings for a target frequency."""
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
