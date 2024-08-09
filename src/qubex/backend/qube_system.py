from __future__ import annotations

from dataclasses import astuple, dataclass, field
from enum import Enum, auto
from typing import Final, Literal

CLOCK_MASTER_ADDRESS: Final = "10.3.0.255"


class BoxType(Enum):
    QUEL1_A = "quel1-a"
    QUEL1_B = "quel1-b"
    QUBE_RIKEN_A = "qube-riken-a"
    QUBE_RIKEN_B = "qube-riken-b"
    QUBE_OU_A = "qube-ou-a"
    QUBE_OU_B = "qube-ou-b"


class PortType(Enum):
    NOT_AVAILABLE = 0
    READ_IN = auto()
    READ_OUT = auto()
    CTRL = auto()
    PUMP = auto()
    MONITOR_IN = auto()
    MONITOR_OUT = auto()


PORT_DIRECTION: Final = {
    PortType.READ_IN: "in",
    PortType.READ_OUT: "out",
    PortType.CTRL: "out",
    PortType.PUMP: "out",
    PortType.MONITOR_IN: "in",
    PortType.MONITOR_OUT: "out",
}


PORT_MAPPING: Final = {
    BoxType.QUEL1_A: {
        0: PortType.READ_IN,
        1: PortType.READ_OUT,
        2: PortType.CTRL,
        3: PortType.PUMP,
        4: PortType.CTRL,
        5: PortType.MONITOR_IN,
        6: PortType.MONITOR_OUT,
        7: PortType.READ_IN,
        8: PortType.READ_OUT,
        9: PortType.CTRL,
        10: PortType.PUMP,
        11: PortType.CTRL,
        12: PortType.MONITOR_IN,
        13: PortType.MONITOR_OUT,
    },
    BoxType.QUEL1_B: {
        0: PortType.NOT_AVAILABLE,
        1: PortType.CTRL,
        2: PortType.CTRL,
        3: PortType.CTRL,
        4: PortType.CTRL,
        5: PortType.MONITOR_IN,
        6: PortType.MONITOR_OUT,
        7: PortType.NOT_AVAILABLE,
        8: PortType.CTRL,
        9: PortType.CTRL,
        10: PortType.CTRL,
        11: PortType.CTRL,
        12: PortType.MONITOR_IN,
        13: PortType.MONITOR_OUT,
    },
    BoxType.QUBE_RIKEN_A: {
        0: PortType.READ_OUT,
        1: PortType.READ_IN,
        2: PortType.PUMP,
        3: PortType.MONITOR_OUT,
        4: PortType.MONITOR_IN,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.MONITOR_IN,
        10: PortType.MONITOR_OUT,
        11: PortType.PUMP,
        12: PortType.READ_IN,
        13: PortType.READ_OUT,
    },
    BoxType.QUBE_RIKEN_B: {
        0: PortType.CTRL,
        1: PortType.NOT_AVAILABLE,
        2: PortType.CTRL,
        3: PortType.MONITOR_OUT,
        4: PortType.MONITOR_IN,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.MONITOR_IN,
        10: PortType.MONITOR_OUT,
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
    BoxType.QUEL1_A: {
        2: 3,
        4: 3,
        9: 3,
        11: 3,
    },
    BoxType.QUEL1_B: {
        1: 1,
        2: 3,
        3: 1,
        4: 3,
        8: 1,
        9: 3,
        10: 1,
        11: 3,
    },
    BoxType.QUBE_RIKEN_A: {
        5: 3,
        6: 3,
        7: 3,
        8: 3,
    },
    BoxType.QUBE_RIKEN_B: {
        0: 1,
        2: 1,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        11: 1,
        13: 1,
    },
    BoxType.QUBE_OU_A: {
        5: 3,
        6: 3,
        7: 3,
        8: 3,
    },
    BoxType.QUBE_OU_B: {
        0: 1,
        2: 1,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        11: 1,
        13: 1,
    },
}

READOUT_PAIRS: Final = {
    BoxType.QUEL1_A: {
        0: 1,
        7: 8,
    },
    BoxType.QUEL1_B: {},
    BoxType.QUBE_RIKEN_A: {
        1: 0,
        12: 13,
    },
    BoxType.QUBE_RIKEN_B: {},
    BoxType.QUBE_OU_A: {
        1: 0,
        12: 13,
    },
    BoxType.QUBE_OU_B: {},
}


@dataclass(frozen=True)
class Box:
    id: str
    name: str
    type: BoxType
    address: str
    adapter: str
    ports: tuple[Port, ...] = field(init=False)

    def __post_init__(self):
        self._initialize_ports()

    def _initialize_ports(self):
        ports = []
        port_index = {
            PortType.NOT_AVAILABLE: 0,
            PortType.READ_IN: 0,
            PortType.READ_OUT: 0,
            PortType.CTRL: 0,
            PortType.PUMP: 0,
            PortType.MONITOR_IN: 0,
            PortType.MONITOR_OUT: 0,
        }
        for port_num, port_type in PORT_MAPPING[self.type].items():
            index = port_index[port_type]
            if port_type == PortType.NOT_AVAILABLE:
                port_id = f"{self.id}.NA{index}"
            elif port_type == PortType.READ_IN:
                port_id = f"{self.id}.READ{index}.IN"
            elif port_type == PortType.READ_OUT:
                port_id = f"{self.id}.READ{index}.OUT"
            elif port_type == PortType.CTRL:
                port_id = f"{self.id}.CTRL{index}"
            elif port_type == PortType.PUMP:
                port_id = f"{self.id}.PUMP{index}"
            elif port_type == PortType.MONITOR_IN:
                port_id = f"{self.id}.MONITOR{index}.IN"
            elif port_type == PortType.MONITOR_OUT:
                port_id = f"{self.id}.MONITOR{index}.OUT"
            else:
                raise ValueError(f"Invalid port type: {port_type}")

            # Initialize channels
            n_channels = NUMBER_OF_CHANNELS[self.type].get(port_num, 0)
            channels = []
            if port_type == PortType.READ_OUT:
                channel = Channel(
                    id=f"{port_id}0",
                    port_id=port_id,
                    number=0,
                )
                channels.append(channel)
            elif port_type == PortType.READ_IN:
                for runit_index in range(4):
                    channel = Channel(
                        id=f"{port_id}{runit_index}",
                        port_id=port_id,
                        number=runit_index,
                    )
                    channels.append(channel)
            else:
                for channel_num in range(n_channels):
                    channel_id = f"{port_id}.CH{channel_num}"
                    channel = Channel(
                        id=channel_id,
                        port_id=port_id,
                        number=channel_num,
                    )
                    channels.append(channel)

            # Create port
            port = Port(
                id=port_id,
                box_id=self.id,
                number=port_num,
                type=port_type,
                channels=tuple(channels),
            )
            ports.append(port)
            port_index[port_type] += 1
        object.__setattr__(self, "ports", tuple(ports))

    @property
    def hash(self) -> int:
        return hash(astuple(self))

    @property
    def readout_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_readout_port]

    @property
    def control_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_control_port]

    @property
    def pump_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_pump_port]

    @property
    def monitor_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_monitor_port]


@dataclass
class Port:
    id: str
    box_id: str
    number: int
    type: PortType
    channels: tuple[Channel, ...]
    loopback: bool | None = None
    sideband: Literal["U", "L"] | None = None
    lo_freq: float | None = None
    cnco_freq: float | None = None
    vatt: float | None = None
    fullscale_current: float | None = None

    @property
    def direction(self) -> str:
        return PORT_DIRECTION[self.type]

    @property
    def rfswitch(self) -> str:
        if self.direction == "out":
            return "block" if self.loopback else "pass"
        elif self.direction == "in":
            return "loop" if self.loopback else "open"
        else:
            raise ValueError(f"Invalid port direction: {self.direction}")

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def is_readout_port(self) -> bool:
        return self.type in [
            PortType.READ_IN,
            PortType.READ_OUT,
        ]

    @property
    def is_control_port(self) -> bool:
        return self.type == PortType.CTRL

    @property
    def is_pump_port(self) -> bool:
        return self.type == PortType.PUMP

    @property
    def is_monitor_port(self) -> bool:
        return self.type in [
            PortType.MONITOR_IN,
            PortType.MONITOR_OUT,
        ]


@dataclass
class Channel:
    id: str
    port_id: str
    number: int
    fnco: float | None = None
    ndelay: int | None = None
    nwait: int | None = None


class QubeSystem:
    def __init__(
        self,
        *,
        boxes: list[Box],
        clock_master_address: str = CLOCK_MASTER_ADDRESS,
    ):
        self._clock_master_address: Final = clock_master_address
        self._boxes: Final = {box.id: box for box in boxes}

    @property
    def clock_master_address(self) -> str:
        return self._clock_master_address

    @property
    def boxes(self) -> dict[str, Box]:
        return self._boxes

    def get_box(self, box_id: str) -> Box:
        try:
            return self._boxes[box_id]
        except KeyError:
            raise KeyError(f"Box `{box_id}` not found.")
