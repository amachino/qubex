from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BoxType(Enum):
    QUEL1_A = "quel1-a"
    QUEL1_B = "quel1-b"
    QUBE_RIKEN_A = "qube-riken-a"
    QUBE_RIKEN_B = "qube-riken-b"
    QUBE_OU_A = "qube-ou-a"
    QUBE_OU_B = "qube-ou-b"


class PortType(Enum):
    NOT_AVAILABLE = "N/A"
    READ0_IN = "READ0.IN"
    READ0_OUT = "READ0.OUT"
    READ1_IN = "READ1.IN"
    READ1_OUT = "READ1.OUT"
    CTRL0 = "CTRL0"
    CTRL1 = "CTRL1"
    CTRL2 = "CTRL2"
    CTRL3 = "CTRL3"
    CTRL4 = "CTRL4"
    CTRL5 = "CTRL5"
    CTRL6 = "CTRL6"
    CTRL7 = "CTRL7"
    PUMP0 = "PUMP0"
    PUMP1 = "PUMP1"
    MONITOR0_IN = "MONITOR0.IN"
    MONITOR0_OUT = "MONITOR0.OUT"
    MONITOR1_IN = "MONITOR1.IN"
    MONITOR1_OUT = "MONITOR1.OUT"


PORT_MAPPING = {
    BoxType.QUEL1_A: {
        0: PortType.READ0_IN,
        1: PortType.READ0_OUT,
        2: PortType.CTRL0,
        3: PortType.PUMP0,
        4: PortType.CTRL1,
        5: PortType.MONITOR0_IN,
        6: PortType.MONITOR0_OUT,
        7: PortType.READ1_IN,
        8: PortType.READ1_OUT,
        9: PortType.CTRL2,
        10: PortType.PUMP1,
        11: PortType.CTRL3,
        12: PortType.MONITOR1_IN,
        13: PortType.MONITOR1_OUT,
    },
    BoxType.QUEL1_B: {
        0: PortType.NOT_AVAILABLE,
        1: PortType.CTRL0,
        2: PortType.CTRL1,
        3: PortType.CTRL2,
        4: PortType.CTRL3,
        5: PortType.MONITOR0_IN,
        6: PortType.MONITOR0_OUT,
        7: PortType.NOT_AVAILABLE,
        8: PortType.CTRL4,
        9: PortType.CTRL5,
        10: PortType.CTRL6,
        11: PortType.CTRL7,
        12: PortType.MONITOR1_IN,
        13: PortType.MONITOR1_OUT,
    },
    BoxType.QUBE_RIKEN_A: {
        0: PortType.READ0_OUT,
        1: PortType.READ0_IN,
        2: PortType.PUMP0,
        3: PortType.MONITOR0_OUT,
        4: PortType.MONITOR0_IN,
        5: PortType.CTRL0,
        6: PortType.CTRL1,
        7: PortType.CTRL2,
        8: PortType.CTRL3,
        9: PortType.MONITOR1_IN,
        10: PortType.MONITOR1_OUT,
        11: PortType.PUMP1,
        12: PortType.READ1_IN,
        13: PortType.READ1_OUT,
    },
    BoxType.QUBE_RIKEN_B: {
        0: PortType.CTRL0,
        1: PortType.NOT_AVAILABLE,
        2: PortType.CTRL1,
        3: PortType.MONITOR0_OUT,
        4: PortType.MONITOR0_IN,
        5: PortType.CTRL2,
        6: PortType.CTRL3,
        7: PortType.CTRL4,
        8: PortType.CTRL5,
        9: PortType.MONITOR1_IN,
        10: PortType.MONITOR1_OUT,
        11: PortType.CTRL6,
        12: PortType.NOT_AVAILABLE,
        13: PortType.CTRL7,
    },
    BoxType.QUBE_OU_A: {
        0: PortType.READ0_OUT,
        1: PortType.READ0_IN,
        2: PortType.PUMP0,
        3: PortType.NOT_AVAILABLE,
        4: PortType.NOT_AVAILABLE,
        5: PortType.CTRL0,
        6: PortType.CTRL1,
        7: PortType.CTRL2,
        8: PortType.CTRL3,
        9: PortType.NOT_AVAILABLE,
        10: PortType.NOT_AVAILABLE,
        11: PortType.PUMP1,
        12: PortType.READ1_IN,
        13: PortType.READ1_OUT,
    },
    BoxType.QUBE_OU_B: {
        0: PortType.CTRL0,
        1: PortType.NOT_AVAILABLE,
        2: PortType.CTRL1,
        3: PortType.NOT_AVAILABLE,
        4: PortType.NOT_AVAILABLE,
        5: PortType.CTRL2,
        6: PortType.CTRL3,
        7: PortType.CTRL4,
        8: PortType.CTRL5,
        9: PortType.NOT_AVAILABLE,
        10: PortType.NOT_AVAILABLE,
        11: PortType.CTRL6,
        12: PortType.NOT_AVAILABLE,
        13: PortType.CTRL7,
    },
}

NUMBER_OF_CHANNELS = {
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

READOUT_PAIRS = {
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


@dataclass
class Box:
    id: str
    name: str
    type: BoxType
    address: str
    adapter: str


@dataclass
class Port:
    number: int
    box: Box

    @property
    def type(self) -> PortType:
        return PORT_MAPPING[self.box.type][self.number]

    @property
    def name(self) -> str:
        return f"{self.box.id}.{self.type.value}"


@dataclass
class ReadInPort(Port):
    number: int
    box: Box
    read_out: ReadOutPort


@dataclass
class ReadOutPort(Port):
    number: int
    box: Box
    mux: int

    @property
    def read_qubits(self) -> list[str]:
        return [f"Q{4 * self.mux + i:02d}" for i in range(4)]


@dataclass
class CtrlPort(Port):
    number: int
    box: Box
    ctrl_qubit: str

    @property
    def n_channel(self) -> int:
        return NUMBER_OF_CHANNELS[self.box.type][self.number]
