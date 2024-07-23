from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal

import networkx as nx
import numpy as np
import qutip as qt
from typing_extensions import TypeAlias

StateAlias: TypeAlias = Literal["0", "1", "+", "-", "+i", "-i", "*"]


@dataclass
class Transmon:
    label: str
    dimension: int
    frequency: float
    anharmonicity: float
    decay_rate: float = 0.0
    dephasing_rate: float = 0.0


@dataclass
class Coupling:
    pair: tuple[str, str]
    strength: float

    @property
    def label(self) -> str:
        return f"{self.pair[0]}-{self.pair[1]}"


class System:
    def __init__(
        self,
        transmons: list[Transmon],
        couplings: list[Coupling] | None = None,
    ):
        self.transmons: Final = transmons
        self.couplings: Final = couplings or []
        self.graph: Final = nx.Graph()
        self.dimensions: Final = [transmon.dimension for transmon in self.transmons]
        self.hamiltonian = qt.tensor([qt.qzero(dim) for dim in self.dimensions])
        self._init_system()

    @property
    def basis_indices(self) -> list[tuple[int, ...]]:
        return list(np.ndindex(*[dim for dim in self.dimensions]))

    @property
    def basis_labels(self) -> list[str]:
        return ["".join(str(i) for i in basis) for basis in self.basis_indices]

    @property
    def identity(self) -> qt.Qobj:
        return qt.tensor([qt.qeye(dim) for dim in self.dimensions])

    def state(
        self,
        alias: StateAlias | dict[str, StateAlias],
        default: StateAlias = "0",
    ) -> qt.Qobj:
        if isinstance(alias, str):
            return qt.tensor([self._state(dim, alias) for dim in self.dimensions])
        elif isinstance(alias, dict):
            for label in alias:
                if label not in self.graph.nodes:
                    raise ValueError(f"Transmon {label} does not exist.")

            states = []
            for transmon in self.transmons:
                if transmon.label in alias:
                    states.append(
                        self._state(transmon.dimension, alias[transmon.label])
                    )
                else:
                    states.append(self._state(transmon.dimension, default))
            return qt.tensor(states)

    def _state(
        self,
        dim: int,
        alias: StateAlias,
    ) -> qt.Qobj:
        if alias == "0":
            state = qt.basis(dim, 0)
        elif alias == "1":
            state = qt.basis(dim, 1)
        elif alias == "+":
            state = (qt.basis(dim, 0) + qt.basis(dim, 1)).unit()
        elif alias == "-":
            state = (qt.basis(dim, 0) - qt.basis(dim, 1)).unit()
        elif alias == "+i":
            state = (qt.basis(dim, 0) + 1j * qt.basis(dim, 1)).unit()
        elif alias == "-i":
            state = (qt.basis(dim, 0) - 1j * qt.basis(dim, 1)).unit()
        elif alias == "*":
            # random state in qubit {|0>, |1>} subspace
            state = qt.Qobj(np.append(qt.rand_ket(2).full(), [0 + 0j] * (dim - 2)))
        else:
            raise ValueError(f"Invalid state alias: {alias}")
        return state

    def index(self, label: str) -> int:
        if label not in self.graph.nodes:
            raise ValueError(f"Transmon {label} does not exist.")
        return list(self.graph.nodes).index(label)

    def transmon(self, label: str) -> Transmon:
        if label not in self.graph.nodes:
            raise ValueError(f"Transmon {label} does not exist.")
        return Transmon(**self.graph.nodes[label])

    def coupling(self, pair: tuple[str, str]) -> Coupling:
        if pair not in self.graph.edges:
            raise ValueError(f"Coupling {pair} does not exist.")
        return Coupling(**self.graph.edges[pair])

    def transmon_hamiltonian(self, transmon: Transmon) -> qt.Qobj:
        omega = 2 * np.pi * transmon.frequency
        alpha = 2 * np.pi * transmon.anharmonicity
        b = self.lowering_operator(transmon.label)
        bd = b.dag()
        return omega * bd * b + 0.5 * alpha * (bd * bd * b * b)

    def coupling_hamiltonian(self, coupling: Coupling) -> qt.Qobj:
        g = 2 * np.pi * coupling.strength
        b_0 = self.lowering_operator(coupling.pair[0])
        bd_0 = b_0.dag()
        b_1 = self.lowering_operator(coupling.pair[1])
        bd_1 = b_1.dag()
        return g * (bd_0 * b_1 + bd_1 * b_0)

    def lowering_operator(self, label: str) -> qt.Qobj:
        if label not in self.graph.nodes:
            raise ValueError(f"Node {label} does not exist.")
        return qt.tensor(
            [
                (
                    qt.destroy(transmon.dimension)
                    if transmon.label == label
                    else qt.qeye(transmon.dimension)
                )
                for transmon in self.transmons
            ]
        )

    def draw(self, **kwargs):
        nx.draw(
            self.graph,
            with_labels=True,
            **kwargs,
        )

    def _init_system(self):
        # validate the transmons and couplings
        transmon_labels = [transmon.label for transmon in self.transmons]
        if len(transmon_labels) != len(set(transmon_labels)):
            raise ValueError("Transmons must have unique labels.")
        for coupling in self.couplings:
            if any(label not in transmon_labels for label in coupling.pair):
                raise ValueError("Couplings must be between existing transmons.")

        # create the graph and hamiltonian
        for transmon in self.transmons:
            self.graph.add_node(transmon.label, **asdict(transmon))
            self.hamiltonian += self.transmon_hamiltonian(transmon)
        for coupling in self.couplings:
            self.graph.add_edge(*coupling.pair, **asdict(coupling))
            self.hamiltonian += self.coupling_hamiltonian(coupling)
