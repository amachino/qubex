from dataclasses import dataclass

from .lattice_chip_graph import LatticeChipGraph


@dataclass
class Chip:
    id: str
    name: str
    n_qubits: int

    @property
    def graph(self) -> LatticeChipGraph:
        if self.n_qubits == 16:
            return LatticeChipGraph(2, 2)
        elif self.n_qubits == 64:
            return LatticeChipGraph(4, 4)
        else:
            raise ValueError("Unsupported number of qubits.")


@dataclass
class Qubit:
    label: str
    frequency: float
    anharmonicity: float


@dataclass
class Resonator:
    label: str
    frequency: float
    qubit: str


@dataclass
class QuantumSystem:
    chip: Chip
    qubits: list[Qubit]
    resonators: list[Resonator]
