from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .lattice_graph import LatticeGraph


@dataclass(frozen=True)
class Chip:
    id: str
    name: str
    n_qubits: int

    @property
    def graph(self) -> LatticeGraph:
        if self.n_qubits == 16:
            return LatticeGraph(2, 2)
        elif self.n_qubits == 64:
            return LatticeGraph(4, 4)
        else:
            raise ValueError("Unsupported number of qubits.")

    @property
    def n_muxes(self) -> int:
        return self.graph.n_muxes

    @property
    def n_qubits_per_mux(self) -> int:
        return self.graph.n_qubits_per_mux

    @property
    def qubits(self) -> list[str]:
        return self.graph.qubits

    @property
    def qubit_edges(self) -> list[tuple[str, str]]:
        return self.graph.qubit_edges


@dataclass(frozen=True)
class Qubit:
    index: int
    label: str
    frequency: float
    anharmonicity: float
    resonator: str


@dataclass(frozen=True)
class Resonator:
    index: int
    label: str
    frequency: float
    qubit: str


class QuantumSystem:
    def __init__(
        self,
        chip: Chip,
        qubits: list[Qubit],
        resonators: list[Resonator],
    ):
        self._chip: Final = chip
        self._qubits: Final = {qubit.label: qubit for qubit in qubits}
        self._resonators: Final = {
            resonator.label: resonator for resonator in resonators
        }

    @property
    def chip(self) -> Chip:
        return self._chip

    @property
    def n_qubits(self) -> int:
        return self.chip.n_qubits

    @property
    def n_muxes(self) -> int:
        return self.chip.n_muxes

    @property
    def qubits(self) -> dict[str, Qubit]:
        return self._qubits

    @property
    def resonators(self) -> dict[str, Resonator]:
        return self._resonators

    def get_qubit(self, label: str) -> Qubit:
        try:
            return self._qubits[label]
        except KeyError:
            raise KeyError(f"Qubit `{label}` not found.")

    def get_resonator(self, label: str) -> Resonator:
        try:
            return self._resonators[label]
        except KeyError:
            raise KeyError(f"Resonator `{label}` not found.")

    def get_qubits_in_mux(self, mux: int) -> list[Qubit]:
        labels = self.chip.graph.get_qubits_in_mux(mux)
        return [self.get_qubit(label) for label in labels]

    def get_spectators(self, qubit: str) -> list[Qubit]:
        labels = self.chip.graph.get_spectators(qubit)
        return [self.get_qubit(label) for label in labels]
