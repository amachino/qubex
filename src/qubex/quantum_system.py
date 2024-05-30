from dataclasses import dataclass


@dataclass
class Chip:
    id: str
    name: str
    n_qubits: int


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
