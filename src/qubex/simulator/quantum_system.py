from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache, cached_property
from typing import Final, Literal, Mapping

import networkx as nx
import numpy as np
import qutip as qt
from typing_extensions import TypeAlias

StateAlias: TypeAlias = Literal["0", "1", "2", "+", "-", "+i", "-i", "*", "**"]


@dataclass(frozen=True)
class Object:
    label: str
    dimension: int
    frequency: float
    anharmonicity: float
    decay_rate: float
    dephasing_rate: float


class Qubit(Object):
    def __init__(
        self,
        *,
        label: str,
        frequency: float,
        decay_rate: float = 0.0,
        dephasing_rate: float = 0.0,
    ):
        super().__init__(
            label=label,
            dimension=2,
            frequency=frequency,
            anharmonicity=0.0,
            decay_rate=decay_rate,
            dephasing_rate=dephasing_rate,
        )


class Resonator(Object):
    def __init__(
        self,
        *,
        label: str,
        dimension: int,
        frequency: float,
        decay_rate: float = 0.0,
        dephasing_rate: float = 0.0,
    ):
        super().__init__(
            label=label,
            dimension=dimension,
            frequency=frequency,
            anharmonicity=0.0,
            decay_rate=decay_rate,
            dephasing_rate=dephasing_rate,
        )


class Transmon(Object):
    def __init__(
        self,
        *,
        label: str,
        dimension: int,
        frequency: float,
        anharmonicity: float,
        decay_rate: float = 0.0,
        dephasing_rate: float = 0.0,
    ):
        super().__init__(
            label=label,
            dimension=dimension,
            frequency=frequency,
            anharmonicity=anharmonicity,
            decay_rate=decay_rate,
            dephasing_rate=dephasing_rate,
        )


@dataclass(frozen=True)
class Coupling:
    pair: tuple[str, str]
    strength: float

    @property
    def label(self) -> str:
        return f"{self.pair[0]}-{self.pair[1]}"


class QuantumSystem:
    def __init__(
        self,
        *,
        objects: list[Object],
        couplings: list[Coupling] | None = None,
    ):
        couplings = couplings or []
        object_labels = [object.label for object in objects]
        if len(object_labels) != len(set(object_labels)):
            raise ValueError("Objects must have unique labels.")
        for coupling in couplings:
            if any(label not in object_labels for label in coupling.pair):
                raise ValueError("Couplings must be between existing objects.")

        self.objects: Final = objects
        self.couplings: Final = couplings

    @cached_property
    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        for object in self.objects:
            graph.add_node(object.label, **asdict(object))
        for coupling in self.couplings:
            graph.add_edge(*coupling.pair, **asdict(coupling))
        return graph

    @cached_property
    def object_dimensions(self) -> list[int]:
        return [object.dimension for object in self.objects]

    @cached_property
    def diagonal_hamiltonian(self) -> qt.Qobj:
        H = self.zero
        for object in self.objects:
            H += self.get_diagonal_hamiltonian(object.label)
        return H

    @cached_property
    def coupling_hamiltonian(self) -> qt.Qobj:
        H = self.zero
        for coupling in self.couplings:
            H += self.get_coupling_hamiltonian(coupling.pair)
        return H

    @cached_property
    def hamiltonian(self) -> qt.Qobj:
        return self.diagonal_hamiltonian + self.coupling_hamiltonian

    @cached_property
    def basis_indices(self) -> list[tuple[int, ...]]:
        return list(np.ndindex(*[dim for dim in self.object_dimensions]))

    @cached_property
    def basis_labels(self) -> list[str]:
        return ["".join(str(i) for i in basis) for basis in self.basis_indices]

    @cached_property
    def zero(self) -> qt.Qobj:
        return qt.tensor([qt.qzero(dim) for dim in self.object_dimensions])

    @cached_property
    def identity(self) -> qt.Qobj:
        return qt.tensor([qt.qeye(dim) for dim in self.object_dimensions])

    def state(
        self,
        states: StateAlias | Mapping[str, StateAlias | qt.Qobj],
        default: StateAlias = "0",
    ) -> qt.Qobj:
        if isinstance(states, str):
            return qt.tensor(
                [self.single_state(dim, states) for dim in self.object_dimensions]
            )
        elif isinstance(states, dict):
            for label in states:
                if label not in self.graph.nodes:
                    raise ValueError(f"Object {label} does not exist.")

            object_states = []
            for object in self.objects:
                if object.label in states:
                    state = states[object.label]
                    if isinstance(state, qt.Qobj):
                        if state.shape != (object.dimension, 1):
                            raise ValueError(
                                f"State for object {object.label} must have shape ({object.dimension}, 1)."
                            )
                        object_states.append(state)
                    elif isinstance(state, str):
                        object_states.append(self.single_state(object.dimension, state))
                else:
                    object_states.append(self.single_state(object.dimension, default))
            return qt.tensor(object_states)
        else:
            raise ValueError("Invalid state input.")

    @cache
    @staticmethod
    def single_state(
        dim: int,
        alias: StateAlias,
    ) -> qt.Qobj:
        if alias == "0":
            state = qt.basis(dim, 0)
        elif alias == "1":
            state = qt.basis(dim, 1)
        elif alias == "2":
            state = qt.basis(dim, 2)
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
        elif alias == "**":
            state = qt.rand_ket(dim)
        else:
            raise ValueError(f"Invalid state alias: {alias}")
        return state

    @cache
    def get_index(self, label: str) -> int:
        if label not in self.graph.nodes:
            raise ValueError(f"Object {label} does not exist.")
        return list(self.graph.nodes).index(label)

    @cache
    def get_object(self, label: str) -> Object:
        if label not in self.graph.nodes:
            raise ValueError(f"Object {label} does not exist.")
        return Object(**self.graph.nodes[label])

    @cache
    def get_coupling(self, pair: tuple[str, str]) -> Coupling:
        if pair not in self.graph.edges:
            raise ValueError(f"Coupling {pair} does not exist.")
        return Coupling(**self.graph.edges[pair])

    @cache
    def get_diagonal_hamiltonian(self, label: str) -> qt.Qobj:
        object = self.get_object(label)
        omega = 2 * np.pi * object.frequency
        alpha = 2 * np.pi * object.anharmonicity
        b = self.lowering_operator(object.label)
        bd = b.dag()
        return omega * bd * b + 0.5 * alpha * (bd * bd * b * b)

    @cache
    def get_coupling_hamiltonian(self, pair: tuple[str, str]) -> qt.Qobj:
        coupling = self.get_coupling(pair)
        g = 2 * np.pi * coupling.strength
        b_0 = self.lowering_operator(coupling.pair[0])
        bd_0 = b_0.dag()
        b_1 = self.lowering_operator(coupling.pair[1])
        bd_1 = b_1.dag()
        return g * (bd_0 * b_1 + bd_1 * b_0)

    @cache
    def lowering_operator(self, label: str) -> qt.Qobj:
        if label not in self.graph.nodes:
            raise ValueError(f"Node {label} does not exist.")
        return qt.tensor(
            [
                (
                    qt.destroy(object.dimension)
                    if object.label == label
                    else qt.qeye(object.dimension)
                )
                for object in self.objects
            ]
        )

    def draw(self, **kwargs):
        nx.draw(
            self.graph,
            with_labels=True,
            **kwargs,
        )
