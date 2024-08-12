from __future__ import annotations

from typing import Final

from pydantic.dataclasses import dataclass

from .lattice_graph import LatticeGraph
from .model import Model

DEFAULT_QUBIT_FREQUENCY: Final = 7.5
DEFAULT_QUBIT_ANHARMONICITY: Final = -0.3
DEFAULT_RESONATOR_FREQUENCY: Final = 10.5


@dataclass
class Chip(Model):
    id: str
    name: str
    qubits: tuple[Qubit, ...]
    resonators: tuple[Resonator, ...]
    muxes: tuple[Mux, ...]

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    @property
    def n_resonators(self) -> int:
        return len(self.resonators)

    @property
    def n_muxes(self) -> int:
        return len(self.muxes)

    @classmethod
    def new(
        cls,
        id: str,
        name: str,
        n_qubits: int,
    ) -> Chip:
        graph = LatticeGraph(n_qubits)

        qubits = tuple(
            Qubit(
                index=index,
                label=label,
                chip_id=id,
            )
            for index, label in enumerate(graph.qubits)
        )
        resonators = tuple(
            Resonator(
                index=index,
                label=label,
                chip_id=id,
            )
            for index, label in enumerate(graph.resonators)
        )
        muxes = tuple(
            Mux(
                index=index,
                label=label,
                chip_id=id,
                resonators=tuple(
                    resonators[index] for index in graph.get_indices_in_mux(label)
                ),
            )
            for index, label in enumerate(graph.muxes)
        )
        return cls(
            id=id,
            name=name,
            qubits=qubits,
            resonators=resonators,
            muxes=muxes,
        )


@dataclass
class Qubit(Model):
    index: int
    label: str
    chip_id: str
    frequency: float = DEFAULT_QUBIT_FREQUENCY
    anharmonicity: float = DEFAULT_QUBIT_ANHARMONICITY


@dataclass
class Resonator(Model):
    index: int
    label: str
    chip_id: str
    frequency: float = DEFAULT_RESONATOR_FREQUENCY


@dataclass
class Mux(Model):
    index: int
    label: str
    chip_id: str
    resonators: tuple[Resonator, ...]


class QuantumSystem:
    def __init__(
        self,
        chip: Chip,
    ):
        self._graph: Final = LatticeGraph(chip.n_qubits)
        self._chip: Final = chip
        self._qubit_dict: Final = {q.label: q for q in chip.qubits}
        self._resonator_dict: Final = {r.label: r for r in chip.resonators}
        self._mux_dict: Final = {m.label: m for m in chip.muxes}

    @property
    def chip(self) -> Chip:
        return self._chip

    @property
    def hash(self) -> int:
        return self.chip.hash

    @property
    def n_qubits(self) -> int:
        return self.chip.n_qubits

    @property
    def qubits(self) -> list[Qubit]:
        return list(self.chip.qubits)

    @property
    def resonators(self) -> list[Resonator]:
        return list(self.chip.resonators)

    @property
    def n_muxes(self) -> int:
        return self.chip.n_muxes

    @property
    def muxes(self) -> list[Mux]:
        return list(self.chip.muxes)

    def get_qubit(
        self,
        label: str,
    ) -> Qubit:
        try:
            return self._qubit_dict[label]
        except KeyError:
            raise KeyError(f"Qubit `{label}` not found.")

    def get_resonator(
        self,
        label: str,
    ) -> Resonator:
        try:
            return self._resonator_dict[label]
        except KeyError:
            raise KeyError(f"Resonator `{label}` not found.")

    def get_qubits_in_mux(
        self,
        mux: int | str,
    ) -> list[Qubit]:
        labels = self._graph.get_qubits_in_mux(mux)
        return [self.get_qubit(label) for label in labels]

    def get_spectator_qubits(
        self,
        qubit: int | str,
        *,
        in_same_mux: bool,
    ) -> list[Qubit]:
        labels = self._graph.get_spectator_qubits(qubit, in_same_mux=in_same_mux)
        return [self.get_qubit(label) for label in labels]
