from __future__ import annotations

import math
from typing import Final

from pydantic.dataclasses import dataclass

from .lattice_graph import LatticeGraph
from .model import Model


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
                resonator=graph.resonators[index],
                frequency=float("nan"),
                anharmonicity=float("nan"),
            )
            for index, label in enumerate(graph.qubits)
        )
        resonators = tuple(
            Resonator(
                index=index,
                label=label,
                chip_id=id,
                qubit=graph.qubits[index],
                frequency=float("nan"),
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
    resonator: str
    frequency: float
    anharmonicity: float

    @property
    def ge_frequency(self) -> float:
        return self.frequency

    @property
    def ef_frequency(self) -> float:
        return round(self.frequency + self.anharmonicity, 6)

    @property
    def alpha(self) -> float:
        return 2 * math.pi * self.anharmonicity

    @property
    def is_valid(self) -> bool:
        return not math.isnan(self.frequency) and not math.isnan(self.anharmonicity)


@dataclass
class Resonator(Model):
    index: int
    label: str
    chip_id: str
    qubit: str
    frequency: float

    @property
    def is_valid(self) -> bool:
        return not math.isnan(self.frequency)


@dataclass
class Mux(Model):
    index: int
    label: str
    chip_id: str
    resonators: tuple[Resonator, ...]

    @property
    def is_valid(self) -> bool:
        return all(resonator.is_valid for resonator in self.resonators)

    @property
    def is_not_available(self) -> bool:
        return all(not resonator.is_valid for resonator in self.resonators)


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
    def chip_graph(self) -> LatticeGraph:
        return self._graph

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
        label: int | str,
    ) -> Qubit:
        try:
            if isinstance(label, int):
                return self.qubits[label]
            else:
                return self._qubit_dict[label]
        except KeyError:
            raise KeyError(f"Qubit `{label}` not found.") from None

    def get_resonator(
        self,
        label: int | str,
    ) -> Resonator:
        try:
            if isinstance(label, int):
                return self.resonators[label]
            else:
                return self._resonator_dict[label]
        except KeyError:
            raise KeyError(f"Resonator `{label}` not found.") from None

    def get_mux(
        self,
        label: int | str,
    ) -> Mux:
        try:
            if isinstance(label, int):
                return self.muxes[label]
            else:
                return self._mux_dict[label]
        except KeyError:
            raise KeyError(f"Mux `{label}` not found.") from None

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
        in_same_mux: bool = False,
    ) -> list[Qubit]:
        labels = self._graph.get_spectator_qubits(qubit, in_same_mux=in_same_mux)
        return [self.get_qubit(label) for label in labels]

    def set_qubit_params(
        self,
        qubit: int | str,
        *,
        frequency: float | None = None,
        anharmonicity: float | None = None,
    ) -> None:
        obj = self.get_qubit(qubit)
        if frequency is not None:
            obj.frequency = frequency
        if anharmonicity is not None:
            obj.anharmonicity = anharmonicity

    def set_resonator_params(
        self,
        resonator: int | str,
        *,
        frequency: float | None = None,
    ) -> None:
        obj = self.get_resonator(resonator)
        if frequency is not None:
            obj.frequency = frequency
