"""Control system models for boxes, ports, and channels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal

from pydantic import Field
from typing_extensions import deprecated

from qubex.core import MutableModel

from .quel1.quel1_backend_constants import (
    CNCO_CENTER_READ,
    CNCO_CENTER_READ_R8,
    DEFAULT_CLOCK_MASTER_ADDRESS,
    DEFAULT_CNCO_FREQ,
    DEFAULT_FNCO_FREQ,
    DEFAULT_FULLSCALE_CURRENT,
    DEFAULT_LO_FREQ,
    DEFAULT_NDELAY,
    DEFAULT_NWAIT,
    DEFAULT_VATT,
)


class BoxType(Enum):
    """Supported box types."""

    QUEL1_A = "quel1-a"
    QUEL1_B = "quel1-b"
    QUBE_RIKEN_A = "qube-riken-a"
    QUBE_RIKEN_B = "qube-riken-b"
    QUBE_OU_A = "qube-ou-a"
    QUBE_OU_B = "qube-ou-b"
    QUEL1SE_A = "quel1se-fujitsu11-a"
    QUEL1SE_B = "quel1se-fujitsu11-b"
    QUEL1SE_R8 = "quel1se-riken8"


@dataclass(frozen=True)
class BoxTraits:
    """Hardware traits used to configure per-box behavior without type branches."""

    ctrl_uses_lo: bool
    ctrl_ssb: Literal["L", "U"] | None
    ctrl_min_frequency_hz: float
    ctrl_uses_vatt: bool
    readout_ssb: Literal["L", "U"]
    readout_cnco_center: int
    default_readout_frequency_range: tuple[float, float, float]
    default_control_frequency_range: tuple[float, float, float]


_DEFAULT_BOX_TRAITS: Final = BoxTraits(
    ctrl_uses_lo=True,
    ctrl_ssb="L",
    ctrl_min_frequency_hz=6.5e9,
    ctrl_uses_vatt=True,
    readout_ssb="U",
    readout_cnco_center=CNCO_CENTER_READ,
    default_readout_frequency_range=(9.75, 10.75, 0.002),
    default_control_frequency_range=(6.5, 9.5, 0.005),
)

_R8_BOX_TRAITS: Final = BoxTraits(
    ctrl_uses_lo=False,
    ctrl_ssb=None,
    ctrl_min_frequency_hz=0.0,
    ctrl_uses_vatt=False,
    readout_ssb="L",
    readout_cnco_center=CNCO_CENTER_READ_R8,
    default_readout_frequency_range=(5.75, 6.75, 0.002),
    default_control_frequency_range=(3.0, 5.0, 0.005),
)

_BOX_TRAITS_BY_TYPE: Final[dict[BoxType, BoxTraits]] = {
    BoxType.QUEL1SE_R8: _R8_BOX_TRAITS,
}


class PortType(Enum):
    """Supported port types."""

    NOT_AVAILABLE = "NA"
    READ_IN = "READ_IN"
    READ_OUT = "READ_OUT"
    CTRL = "CTRL"
    PUMP = "PUMP"
    MNTR_IN = "MNTR_IN"
    MNTR_OUT = "MNTR_OUT"
    FOGI = "FOGI"


PORT_DIRECTION: Final = {
    PortType.READ_IN: "in",
    PortType.READ_OUT: "out",
    PortType.CTRL: "out",
    PortType.PUMP: "out",
    PortType.MNTR_IN: "in",
    PortType.MNTR_OUT: "out",
    PortType.FOGI: "out",
}


PORT_MAPPING: Final = {
    BoxType.QUEL1SE_R8: {
        0: PortType.READ_IN,
        1: PortType.READ_OUT,
        (1, 1): PortType.FOGI,
        2: PortType.PUMP,
        3: PortType.CTRL,
        4: PortType.MNTR_IN,
        5: PortType.MNTR_OUT,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.CTRL,
        10: PortType.MNTR_IN,
        11: PortType.MNTR_OUT,
    },
    BoxType.QUEL1SE_A: {
        0: PortType.READ_IN,
        1: PortType.READ_OUT,
        2: PortType.CTRL,
        3: PortType.PUMP,
        4: PortType.CTRL,
        5: PortType.MNTR_IN,
        6: PortType.MNTR_OUT,
        7: PortType.READ_IN,
        8: PortType.READ_OUT,
        9: PortType.CTRL,
        10: PortType.PUMP,
        11: PortType.CTRL,
        12: PortType.MNTR_IN,
        13: PortType.MNTR_OUT,
    },
    BoxType.QUEL1SE_B: {
        0: PortType.NOT_AVAILABLE,
        1: PortType.CTRL,
        2: PortType.CTRL,
        3: PortType.CTRL,
        4: PortType.CTRL,
        5: PortType.MNTR_IN,
        6: PortType.MNTR_OUT,
        7: PortType.NOT_AVAILABLE,
        8: PortType.CTRL,
        9: PortType.CTRL,
        10: PortType.CTRL,
        11: PortType.CTRL,
        12: PortType.MNTR_IN,
        13: PortType.MNTR_OUT,
    },
    BoxType.QUEL1_A: {
        0: PortType.READ_IN,
        1: PortType.READ_OUT,
        2: PortType.CTRL,
        3: PortType.PUMP,
        4: PortType.CTRL,
        5: PortType.MNTR_IN,
        6: PortType.MNTR_OUT,
        7: PortType.READ_IN,
        8: PortType.READ_OUT,
        9: PortType.CTRL,
        10: PortType.PUMP,
        11: PortType.CTRL,
        12: PortType.MNTR_IN,
        13: PortType.MNTR_OUT,
    },
    BoxType.QUEL1_B: {
        0: PortType.NOT_AVAILABLE,
        1: PortType.CTRL,
        2: PortType.CTRL,
        3: PortType.CTRL,
        4: PortType.CTRL,
        5: PortType.MNTR_IN,
        6: PortType.NOT_AVAILABLE,
        7: PortType.NOT_AVAILABLE,
        8: PortType.CTRL,
        9: PortType.CTRL,
        10: PortType.CTRL,
        11: PortType.CTRL,
        12: PortType.MNTR_IN,
        13: PortType.NOT_AVAILABLE,
    },
    BoxType.QUBE_RIKEN_A: {
        0: PortType.READ_OUT,
        1: PortType.READ_IN,
        2: PortType.PUMP,
        3: PortType.MNTR_OUT,
        4: PortType.MNTR_IN,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.MNTR_IN,
        10: PortType.MNTR_OUT,
        11: PortType.PUMP,
        12: PortType.READ_IN,
        13: PortType.READ_OUT,
    },
    BoxType.QUBE_RIKEN_B: {
        0: PortType.CTRL,
        1: PortType.NOT_AVAILABLE,
        2: PortType.CTRL,
        3: PortType.MNTR_OUT,
        4: PortType.MNTR_IN,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.MNTR_IN,
        10: PortType.MNTR_OUT,
        11: PortType.CTRL,
        12: PortType.NOT_AVAILABLE,
        13: PortType.CTRL,
    },
    BoxType.QUBE_OU_A: {
        0: PortType.READ_OUT,
        1: PortType.READ_IN,
        2: PortType.PUMP,
        3: PortType.NOT_AVAILABLE,
        4: PortType.NOT_AVAILABLE,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.NOT_AVAILABLE,
        10: PortType.NOT_AVAILABLE,
        11: PortType.PUMP,
        12: PortType.READ_IN,
        13: PortType.READ_OUT,
    },
    BoxType.QUBE_OU_B: {
        0: PortType.CTRL,
        1: PortType.NOT_AVAILABLE,
        2: PortType.CTRL,
        3: PortType.NOT_AVAILABLE,
        4: PortType.NOT_AVAILABLE,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.NOT_AVAILABLE,
        10: PortType.NOT_AVAILABLE,
        11: PortType.CTRL,
        12: PortType.NOT_AVAILABLE,
        13: PortType.CTRL,
    },
}

NUMBER_OF_CHANNELS: Final = {
    BoxType.QUEL1SE_R8: {
        0: 4,
        1: 1,
        (1, 1): 1,
        2: 3,
        3: 3,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 1,
        11: 1,
    },
    BoxType.QUEL1SE_A: {
        0: 4,
        1: 1,
        2: 3,
        3: 1,
        4: 3,
        5: 1,
        6: 1,
        7: 4,
        8: 1,
        9: 3,
        10: 1,
        11: 3,
        12: 1,
        13: 1,
    },
    BoxType.QUEL1SE_B: {
        0: 0,
        1: 1,
        2: 3,
        3: 1,
        4: 3,
        5: 1,
        6: 1,
        7: 0,
        8: 1,
        9: 3,
        10: 1,
        11: 3,
        12: 1,
        13: 1,
    },
    BoxType.QUEL1_A: {
        0: 4,
        1: 1,
        2: 3,
        3: 1,
        4: 3,
        5: 1,
        6: 1,
        7: 4,
        8: 1,
        9: 3,
        10: 1,
        11: 3,
        12: 1,
        13: 1,
    },
    BoxType.QUEL1_B: {
        0: 0,
        1: 1,
        2: 1,
        3: 3,
        4: 3,
        5: 1,
        6: 0,
        7: 0,
        8: 1,
        9: 1,
        10: 3,
        11: 3,
        12: 1,
        13: 0,
    },
    BoxType.QUBE_RIKEN_A: {
        0: 1,
        1: 4,
        2: 1,
        3: 1,
        4: 1,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 1,
        10: 1,
        11: 1,
        12: 4,
        13: 1,
    },
    BoxType.QUBE_RIKEN_B: {
        0: 1,
        1: 0,
        2: 1,
        3: 1,
        4: 1,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 1,
        10: 1,
        11: 1,
        12: 0,
        13: 1,
    },
    BoxType.QUBE_OU_A: {
        0: 1,
        1: 4,
        2: 1,
        3: 0,
        4: 0,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 0,
        10: 0,
        11: 1,
        12: 4,
        13: 1,
    },
    BoxType.QUBE_OU_B: {
        0: 1,
        1: 0,
        2: 1,
        3: 0,
        4: 0,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 0,
        10: 0,
        11: 1,
        12: 0,
        13: 1,
    },
}

_QUEL1SE_R8_AWG_OPTIONS: Final[set[str]] = {
    "se8_mxfe1_awg1331",
    "se8_mxfe1_awg2222",
    "se8_mxfe1_awg3113",
}
_QUEL1SE_R8_DEFAULT_AWG_OPTION: Final = "se8_mxfe1_awg2222"
_QUEL1SE_R8_PORTS_BY_AWG_OPTION: Final[dict[str, dict[int, int]]] = {
    "se8_mxfe1_awg1331": {6: 1, 7: 3, 8: 3, 9: 1},
    "se8_mxfe1_awg2222": {6: 2, 7: 2, 8: 2, 9: 2},
    "se8_mxfe1_awg3113": {6: 3, 7: 1, 8: 1, 9: 3},
}


def resolve_quel1se_r8_awg_option(options: Sequence[str] | None = None) -> str:
    """Resolve a single AWG option for QuEL-1 SE R8 from optional labels."""
    option_labels = tuple(options or ())
    awg_options = [label for label in option_labels if label in _QUEL1SE_R8_AWG_OPTIONS]
    if len(awg_options) > 1:
        raise ValueError("Multiple AWG options are not allowed for quel1se-riken8.")
    if len(awg_options) == 1:
        return awg_options[0]
    return _QUEL1SE_R8_DEFAULT_AWG_OPTION


def get_number_of_channels(
    box_type: BoxType,
    port_number: int | tuple[int, int],
    options: Sequence[str] | None = None,
) -> int:
    """Return the number of channels for a box port with optional profile overrides."""
    if box_type == BoxType.QUEL1SE_R8:
        awg_option = resolve_quel1se_r8_awg_option(options)
        if isinstance(port_number, int):
            override = _QUEL1SE_R8_PORTS_BY_AWG_OPTION[awg_option].get(port_number)
            if override is not None:
                return override
    return NUMBER_OF_CHANNELS[box_type].get(port_number, 0)


def create_ports(
    box_id: str,
    box_type: BoxType,
    port_numbers: Sequence[int] | None = None,
    options: Sequence[str] | None = None,
) -> tuple[GenPort | CapPort, ...]:
    """Create ports for a box based on mapping rules."""
    ports: list[GenPort | CapPort] = []
    port_index = {
        PortType.NOT_AVAILABLE: 0,
        PortType.READ_IN: 0,
        PortType.READ_OUT: 0,
        PortType.CTRL: 0,
        PortType.PUMP: 0,
        PortType.MNTR_IN: 0,
        PortType.MNTR_OUT: 0,
        PortType.FOGI: 0,
    }
    for port_num, port_type in PORT_MAPPING[box_type].items():
        if port_type in (PortType.READ_IN, PortType.READ_OUT, PortType.CTRL):
            # skip if the port is not used in the experiment
            if port_numbers is not None and port_num not in port_numbers:
                continue
        index = port_index[port_type]
        if port_type == PortType.NOT_AVAILABLE:
            port_id = f"{box_id}.NA{index}"
        elif port_type == PortType.READ_IN:
            port_id = f"{box_id}.READ{index}.IN"
        elif port_type == PortType.READ_OUT:
            port_id = f"{box_id}.READ{index}.OUT"
        elif port_type == PortType.CTRL:
            port_id = f"{box_id}.CTRL{index}"
        elif port_type == PortType.PUMP:
            port_id = f"{box_id}.PUMP{index}"
        elif port_type == PortType.MNTR_IN:
            port_id = f"{box_id}.MNTR{index}.IN"
        elif port_type == PortType.MNTR_OUT:
            port_id = f"{box_id}.MNTR{index}.OUT"
        elif port_type == PortType.FOGI:
            port_id = f"{box_id}.FOGI{index}"
        else:
            raise ValueError(f"Invalid port type: {port_type}")
        n_channels = get_number_of_channels(box_type, port_num, options=options)
        port: GenPort | CapPort | Port
        if port_type == PortType.NOT_AVAILABLE:
            continue
        elif port_type in (PortType.READ_IN, PortType.MNTR_IN):
            port = CapPort(
                id=port_id,
                box_id=box_id,
                number=port_num,
                type=port_type,
                channels=(),
            )
            port.channels = tuple(
                CapChannel(
                    id=f"{port_id}{channel_num}",
                    _port=port,
                    number=channel_num,
                )
                for channel_num in range(n_channels)
            )
        elif port_type == PortType.READ_OUT:
            port = GenPort(
                id=port_id,
                box_id=box_id,
                number=port_num,
                type=port_type,
                sideband="U",
            )
            port.channels = tuple(
                GenChannel(
                    id=f"{port_id}{channel_num}",
                    _port=port,
                    number=channel_num,
                )
                for channel_num in range(n_channels)
            )
        elif port_type == PortType.MNTR_OUT:
            port = GenPort(
                id=port_id,
                box_id=box_id,
                number=port_num,
                type=port_type,
                sideband="L",
            )
            port.channels = tuple(
                GenChannel(
                    id=f"{port_id}{channel_num}",
                    _port=port,
                    number=channel_num,
                )
                for channel_num in range(n_channels)
            )
        elif port_type == PortType.CTRL or port_type == PortType.PUMP:
            port = GenPort(
                id=port_id,
                box_id=box_id,
                number=port_num,
                type=port_type,
                sideband="L",
            )
            port.channels = tuple(
                GenChannel(
                    id=f"{port_id}.CH{channel_num}",
                    _port=port,
                    number=channel_num,
                )
                for channel_num in range(n_channels)
            )
        elif port_type == PortType.FOGI:
            port = GenPort(
                id=port_id,
                box_id=box_id,
                number=port_num,
                type=port_type,
                sideband=None,
                vatt=None,
            )
            port.channels = tuple(
                GenChannel(
                    id=f"{port_id}.CH{channel_num}",
                    _port=port,
                    number=channel_num,
                )
                for channel_num in range(n_channels)
            )
        else:
            raise ValueError(f"Invalid port type: {port_type}")
        ports.append(port)
        port_index[port_type] += 1
    return tuple(ports)


class BoxPool(MutableModel):
    """Collection of boxes and clock master configuration."""

    clock_master_address: str
    boxes: tuple[Box, ...]

    @property
    def hash(self) -> int:
        """Return a hash of the serialized box pool."""
        return hash(self.to_json(indent=0))


class Box(MutableModel):
    """
    Representation of a control box and its ports.

    Notes
    -----
    This model graph includes back-references from channels to parent ports and
    is intended for runtime use. Generic round-trip restoration via
    `from_dict()` / `from_json()` is not supported.
    """

    id: str
    name: str
    type: BoxType
    address: str
    adapter: str
    options: tuple[str, ...] = ()
    ports: tuple[GenPort | CapPort, ...]

    @classmethod
    def new(
        cls,
        *,
        id: str,
        name: str,
        type: BoxType | str,
        address: str,
        adapter: str,
        port_numbers: Sequence[int] | None = None,
        options: Sequence[str] | None = None,
    ) -> Box:
        """Create a box with ports from settings."""
        type = BoxType(type) if isinstance(type, str) else type
        options_tuple = tuple(options or ())
        return cls(
            id=id,
            name=name,
            type=type,
            address=address,
            adapter=adapter,
            options=options_tuple,
            ports=create_ports(id, type, port_numbers, options=options_tuple),
        )

    @property
    def input_ports(self) -> list[CapPort]:
        """Return input ports for capture."""
        return [port for port in self.ports if isinstance(port, CapPort)]

    @property
    def traits(self) -> BoxTraits:
        """Return behavioral traits for this box type."""
        return _BOX_TRAITS_BY_TYPE.get(self.type, _DEFAULT_BOX_TRAITS)

    @property
    def output_ports(self) -> list[GenPort]:
        """Return output ports for generation."""
        return [port for port in self.ports if isinstance(port, GenPort)]

    @property
    def control_ports(self) -> list[Port]:
        """Return ports used for control."""
        return [port for port in self.ports if port.is_control_port]

    @property
    def readout_ports(self) -> list[Port]:
        """Return ports used for readout."""
        return [port for port in self.ports if port.is_readout_port]

    @property
    def monitor_ports(self) -> list[Port]:
        """Return ports used for monitoring."""
        return [port for port in self.ports if port.is_monitor_port]

    @property
    def pump_ports(self) -> list[Port]:
        """Return ports used for pumping."""
        return [port for port in self.ports if port.is_pump_port]

    def get_port(self, port_number: int) -> GenPort | CapPort:
        """Return a port by number."""
        try:
            return next(port for port in self.ports if port.number == port_number)
        except StopIteration:
            raise IndexError(
                f"Port number `{port_number}` not found in box `{self.id}`."
            ) from None


class Port(MutableModel):
    """Base port definition shared by Gen and Cap ports."""

    id: str
    box_id: str
    number: int | tuple[int, int]
    type: PortType
    channels: tuple[GenChannel, ...] | tuple[CapChannel, ...]

    @property
    def direction(self) -> str:
        """Return port direction string."""
        return PORT_DIRECTION[self.type]

    @property
    def n_channels(self) -> int:
        """Return number of channels on the port."""
        return len(self.channels)

    @property
    def is_input_port(self) -> bool:
        """Return whether the port is an input."""
        return self.direction == "in"

    @property
    def is_output_port(self) -> bool:
        """Return whether the port is an output."""
        return self.direction == "out"

    @property
    def is_control_port(self) -> bool:
        """Return whether the port is a control port."""
        return self.type == PortType.CTRL

    @property
    def is_readout_port(self) -> bool:
        """Return whether the port is a readout port."""
        return self.type in (PortType.READ_IN, PortType.READ_OUT)

    @property
    def is_monitor_port(self) -> bool:
        """Return whether the port is a monitor port."""
        return self.type in (PortType.MNTR_IN, PortType.MNTR_OUT)

    @property
    def is_pump_port(self) -> bool:
        """Return whether the port is a pump port."""
        return self.type == PortType.PUMP

    @property
    def is_fogi_port(self) -> bool:
        """Return whether the port is a FOGI port."""
        return self.type == PortType.FOGI


class GenPort(Port):
    """Generator port with frequency and output settings."""

    channels: tuple[GenChannel, ...] = ()
    sideband: Literal["U", "L"] | None = None
    lo_freq: int | None = DEFAULT_LO_FREQ
    cnco_freq: int = DEFAULT_CNCO_FREQ
    vatt: int | None = DEFAULT_VATT
    fullscale_current: int = DEFAULT_FULLSCALE_CURRENT
    rfswitch: Literal["pass", "block"] = "pass"

    @property
    def base_frequencies(self) -> tuple[int, ...]:
        """Return coarse frequencies for each channel."""
        return tuple(channel.coarse_frequency for channel in self.channels)


class CapPort(Port):
    """Capture port with frequency settings."""

    channels: tuple[CapChannel, ...] = ()
    lo_freq: int = DEFAULT_LO_FREQ
    cnco_freq: int = DEFAULT_CNCO_FREQ
    rfswitch: Literal["open", "loop"] = "open"


class Channel(MutableModel):
    """Base channel with identifier and number."""

    id: str
    number: int


class GenChannel(Channel):
    """Generator channel with frequency parameters."""

    port_ref: GenPort = Field(alias="_port", exclude=True, repr=False)
    fnco_freq: int = DEFAULT_FNCO_FREQ
    nwait: int = DEFAULT_NWAIT

    @property
    def port(self) -> GenPort:
        """Return the parent generator port."""
        return self.port_ref

    @property
    @deprecated("Use `port` instead.")
    def _port(self) -> GenPort:
        """Backward-compatible alias for the parent generator port."""
        return self.port_ref

    @_port.setter
    @deprecated("Use `port` instead.")
    def _port(self, port: GenPort) -> None:
        """Set the parent generator port via legacy attribute name."""
        self.port_ref = port

    @property
    def lo_freq(self) -> int:
        """Return the LO frequency for the channel."""
        if self.port.lo_freq is None:
            raise ValueError("LO frequency is not set.")
        return self.port.lo_freq

    @property
    def cnco_freq(self) -> int:
        """Return the CNCO frequency for the channel."""
        return self.port.cnco_freq

    @property
    def nco_freq(self) -> int:
        """Return the NCO frequency for the channel."""
        return self.port.cnco_freq + self.fnco_freq

    @property
    def coarse_frequency(self) -> int:
        """Return the coarse frequency for the channel."""
        sideband = self.port.sideband
        lo = self.port.lo_freq
        cnco = self.port.cnco_freq

        if lo is None and sideband is None:
            return cnco
        elif lo is None:
            raise ValueError("LO frequency is not set.")
        elif sideband is None:
            raise ValueError("Sideband is not set.")
        else:
            if sideband == "U":
                return lo + cnco
            elif sideband == "L":
                return lo - cnco
            else:
                raise ValueError(f"Invalid sideband: {sideband}")

    @property
    def fine_frequency(self) -> int:
        """Return the fine frequency for the channel."""
        sideband = self.port.sideband
        lo = self.port.lo_freq
        cnco = self.port.cnco_freq
        fnco = self.fnco_freq
        nco = cnco + fnco

        if lo is None and sideband is None:
            return nco
        elif lo is None:
            raise ValueError("LO frequency is not set.")
        elif sideband is None:
            raise ValueError("Sideband is not set.")
        else:
            if sideband == "U":
                return lo + nco
            elif sideband == "L":
                return lo - nco
            else:
                raise ValueError(f"Invalid sideband: {sideband}")


class CapChannel(Channel):
    """Capture channel with frequency parameters."""

    port_ref: CapPort = Field(alias="_port", exclude=True, repr=False)
    fnco_freq: int = DEFAULT_FNCO_FREQ
    ndelay: int = DEFAULT_NDELAY

    @property
    def port(self) -> CapPort:
        """Return the parent capture port."""
        return self.port_ref

    @property
    @deprecated("Use `port` instead.")
    def _port(self) -> CapPort:
        """Backward-compatible alias for the parent capture port."""
        return self.port_ref

    @_port.setter
    @deprecated("Use `port` instead.")
    def _port(self, port: CapPort) -> None:
        """Set the parent capture port via legacy attribute name."""
        self.port_ref = port


class ControlSystem:
    """Collection of boxes and access helpers for ports."""

    class NotGivenType:
        """Sentinel class for unset values."""

        pass

    NotGiven = NotGivenType()

    def __init__(
        self,
        boxes: Sequence[Box],
        clock_master_address: str | None = None,
    ):
        if clock_master_address is None:
            clock_master_address = DEFAULT_CLOCK_MASTER_ADDRESS
        self._box_pool: Final = BoxPool(
            boxes=tuple(boxes),
            clock_master_address=clock_master_address,
        )
        self._box_dict: Final = {box.id: box for box in self._box_pool.boxes}

    @property
    def box_pool(self) -> BoxPool:
        """Return the box pool."""
        return self._box_pool

    @property
    def hash(self) -> int:
        """Return a hash for the control system."""
        return self.box_pool.hash

    @property
    def clock_master_address(self) -> str:
        """Return the clock master address."""
        return self.box_pool.clock_master_address

    @property
    def boxes(self) -> list[Box]:
        """Return the list of boxes."""
        return list(self._box_pool.boxes)

    def get_box(self, box_id: str) -> Box:
        """Return a box by ID."""
        try:
            return self._box_dict[box_id]
        except KeyError:
            raise KeyError(f"Box `{box_id}` not found.") from None

    def get_port(self, box_id: str, port_number: int) -> GenPort | CapPort:
        """Return a port by box ID and port number."""
        box = self.get_box(box_id)
        try:
            return next(port for port in box.ports if port.number == port_number)
        except StopIteration:
            raise IndexError(
                f"Port number `{port_number}` not found in box `{box_id}`."
            ) from None

    def get_gen_port(self, box_id: str, port_number: int) -> GenPort:
        """Return a generator port by box ID and port number."""
        port = self.get_port(box_id, port_number)
        if not isinstance(port, GenPort):
            raise TypeError(f"Port `{port.id}` is not a GenPort (type: {type(port)}).")
        return port

    def get_cap_port(self, box_id: str, port_number: int) -> CapPort:
        """Return a capture port by box ID and port number."""
        port = self.get_port(box_id, port_number)
        if not isinstance(port, CapPort):
            raise TypeError(f"Port `{port.id}` is not a CapPort (type: {type(port)}).")
        return port

    def get_port_by_id(self, port_id: str) -> GenPort | CapPort:
        """Return a port by its ID."""
        for box in self.boxes:
            for port in box.ports:
                if port.id == port_id:
                    return port
        raise KeyError(f"Port `{port_id}` not found.")

    def set_port_params(
        self,
        box_id: str,
        port_number: int,
        *,
        rfswitch: Literal["pass", "block", "open", "loop"] | None = None,
        sideband: Literal["U", "L"] | None | NotGivenType = NotGiven,
        lo_freq: int | None | NotGivenType = NotGiven,
        cnco_freq: int | None = None,
        fnco_freqs: Sequence[int] | None = None,
        vatt: int | None | NotGivenType = NotGiven,
        fullscale_current: int | None = None,
        nwait: int | None = None,
        ndelay: int | None = None,
    ) -> None:
        """Set port and channel parameters for a box port."""
        port = self.get_port(box_id, port_number)

        if isinstance(port, GenPort):
            if rfswitch is not None:
                port.rfswitch = rfswitch  # type: ignore
            if not isinstance(sideband, self.NotGivenType):
                port.sideband = sideband
            if not isinstance(lo_freq, self.NotGivenType):
                port.lo_freq = lo_freq
            if cnco_freq is not None:
                port.cnco_freq = cnco_freq
            if fnco_freqs is not None:
                if len(fnco_freqs) != len(port.channels):
                    raise ValueError(
                        f"Expected {len(port.channels)} fnco_freqs, "
                        f"but got {len(fnco_freqs)}."
                    )
                for gen_channel, fnco_freq in zip(
                    port.channels, fnco_freqs, strict=True
                ):
                    gen_channel.fnco_freq = fnco_freq
            if not isinstance(vatt, self.NotGivenType):
                port.vatt = vatt
            if fullscale_current is not None:
                port.fullscale_current = fullscale_current

            if nwait is not None:
                for gen_channel in port.channels:
                    gen_channel.nwait = nwait
        elif isinstance(port, CapPort):
            if rfswitch is not None:
                port.rfswitch = rfswitch  # type: ignore
            if lo_freq is not None:
                port.lo_freq = lo_freq  # type: ignore
            if cnco_freq is not None:
                port.cnco_freq = cnco_freq
            if fnco_freqs is not None:
                if len(fnco_freqs) != len(port.channels):
                    raise ValueError(
                        f"Expected {len(port.channels)} fnco_freqs, "
                        f"but got {len(fnco_freqs)}."
                    )
                for cap_channel, fnco_freq in zip(
                    port.channels, fnco_freqs, strict=True
                ):
                    cap_channel.fnco_freq = fnco_freq
            if ndelay is not None:
                for cap_channel in port.channels:
                    cap_channel.ndelay = ndelay
        else:
            raise TypeError(f"Invalid port type: {type(port)}")
