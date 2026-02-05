from __future__ import annotations

from enum import Enum
from typing import Final, Literal, Optional, Sequence, Union

from pydantic import Field
from pydantic.dataclasses import dataclass

from .model import Model

DEFAULT_CLOCK_MASTER_ADDRESS: Final = "10.3.0.255"
DEFAULT_LO_FREQ: Final = 9_000_000_000
DEFAULT_CNCO_FREQ: Final = 1_500_000_000
DEFAULT_FNCO_FREQ: Final = 0
DEFAULT_VATT: Final = 3072  # 0xC00
DEFAULT_FULLSCALE_CURRENT: Final = 40527
DEFAULT_NDELAY: Final = 7
DEFAULT_NWAIT: Final = 0


class BoxType(Enum):
    QUEL1_A = "quel1-a"
    QUEL1_B = "quel1-b"
    QUBE_RIKEN_A = "qube-riken-a"
    QUBE_RIKEN_B = "qube-riken-b"
    QUBE_OU_A = "qube-ou-a"
    QUBE_OU_B = "qube-ou-b"
    QUEL1SE_A = "quel1se-fujitsu11-a"
    QUEL1SE_B = "quel1se-fujitsu11-b"
    QUEL1SE_R8 = "quel1se-riken8"


class PortType(Enum):
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


def create_ports(
    box_id: str,
    box_type: BoxType,
    port_numbers: Sequence[int] | None = None,
) -> tuple[Union[GenPort, CapPort], ...]:
    ports: list[Union[GenPort, CapPort]] = []
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
        n_channels = NUMBER_OF_CHANNELS[box_type].get(port_num, 0)
        port: Union[GenPort, CapPort, Port]
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
        elif port_type == PortType.CTRL:
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
        elif port_type == PortType.PUMP:
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


@dataclass
class BoxPool(Model):
    clock_master_address: str
    boxes: tuple[Box, ...]


@dataclass
class Box(Model):
    id: str
    name: str
    type: BoxType
    address: str
    adapter: str
    ports: tuple[Union[GenPort, CapPort], ...]

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
    ) -> Box:
        type = BoxType(type) if isinstance(type, str) else type
        return cls(
            id=id,
            name=name,
            type=type,
            address=address,
            adapter=adapter,
            ports=create_ports(id, type, port_numbers),
        )

    @property
    def input_ports(self) -> list[CapPort]:
        return [port for port in self.ports if isinstance(port, CapPort)]

    @property
    def output_ports(self) -> list[GenPort]:
        return [port for port in self.ports if isinstance(port, GenPort)]

    @property
    def control_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_control_port]

    @property
    def readout_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_readout_port]

    @property
    def monitor_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_monitor_port]

    @property
    def pump_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_pump_port]

    def get_port(self, port_number: int) -> GenPort | CapPort:
        try:
            return next(port for port in self.ports if port.number == port_number)
        except StopIteration:
            raise IndexError(
                f"Port number `{port_number}` not found in box `{self.id}`."
            ) from None


@dataclass
class Port(Model):
    id: str
    box_id: str
    number: Union[int, tuple[int, int]]
    type: PortType
    channels: Union[tuple[GenChannel, ...], tuple[CapChannel, ...]]

    @property
    def direction(self) -> str:
        return PORT_DIRECTION[self.type]

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def is_input_port(self) -> bool:
        return self.direction == "in"

    @property
    def is_output_port(self) -> bool:
        return self.direction == "out"

    @property
    def is_control_port(self) -> bool:
        return self.type == PortType.CTRL

    @property
    def is_readout_port(self) -> bool:
        return self.type in (PortType.READ_IN, PortType.READ_OUT)

    @property
    def is_monitor_port(self) -> bool:
        return self.type in (PortType.MNTR_IN, PortType.MNTR_OUT)

    @property
    def is_pump_port(self) -> bool:
        return self.type == PortType.PUMP

    @property
    def is_fogi_port(self) -> bool:
        return self.type == PortType.FOGI


@dataclass
class GenPort(Port):
    channels: tuple[GenChannel, ...] = ()
    sideband: Optional[Literal["U", "L"]] = None
    lo_freq: Optional[int] = DEFAULT_LO_FREQ
    cnco_freq: int = DEFAULT_CNCO_FREQ
    vatt: Optional[int] = DEFAULT_VATT
    fullscale_current: int = DEFAULT_FULLSCALE_CURRENT
    rfswitch: Literal["pass", "block"] = "pass"

    @property
    def base_frequencies(self) -> tuple[int, ...]:
        return tuple(channel.coarse_frequency for channel in self.channels)


@dataclass
class CapPort(Port):
    channels: tuple[CapChannel, ...] = ()
    lo_freq: int = DEFAULT_LO_FREQ
    cnco_freq: int = DEFAULT_CNCO_FREQ
    rfswitch: Literal["open", "loop"] = "open"


@dataclass
class Channel(Model):
    id: str
    number: int


@dataclass
class GenChannel(Channel):
    _port: GenPort = Field(exclude=True)
    fnco_freq: int = DEFAULT_FNCO_FREQ
    nwait: int = DEFAULT_NWAIT

    @property
    def port(self) -> GenPort:
        return self._port

    @property
    def lo_freq(self) -> int:
        if self.port.lo_freq is None:
            raise ValueError("LO frequency is not set.")
        return self.port.lo_freq

    @property
    def cnco_freq(self) -> int:
        return self.port.cnco_freq

    @property
    def nco_freq(self) -> int:
        return self.port.cnco_freq + self.fnco_freq

    @property
    def coarse_frequency(self) -> int:
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


@dataclass
class CapChannel(Channel):
    _port: CapPort = Field(exclude=True)
    fnco_freq: int = DEFAULT_FNCO_FREQ
    ndelay: int = DEFAULT_NDELAY

    @property
    def port(self) -> CapPort:
        return self._port


class ControlSystem:
    class NotGivenType:
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
        return self._box_pool

    @property
    def hash(self) -> int:
        return self.box_pool.hash

    @property
    def clock_master_address(self) -> str:
        return self.box_pool.clock_master_address

    @property
    def boxes(self) -> list[Box]:
        return list(self._box_pool.boxes)

    def get_box(self, box_id: str) -> Box:
        try:
            return self._box_dict[box_id]
        except KeyError:
            raise KeyError(f"Box `{box_id}` not found.") from None

    def get_port(self, box_id: str, port_number: int) -> GenPort | CapPort:
        box = self.get_box(box_id)
        try:
            return next(port for port in box.ports if port.number == port_number)
        except StopIteration:
            raise IndexError(
                f"Port number `{port_number}` not found in box `{box_id}`."
            ) from None

    def get_port_by_id(self, port_id: str) -> GenPort | CapPort:
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
                for gen_channel, fnco_freq in zip(port.channels, fnco_freqs):
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
                for cap_channel, fnco_freq in zip(port.channels, fnco_freqs):
                    cap_channel.fnco_freq = fnco_freq
            if ndelay is not None:
                for cap_channel in port.channels:
                    cap_channel.ndelay = ndelay
        else:
            raise ValueError(f"Invalid port type: {type(port)}")
