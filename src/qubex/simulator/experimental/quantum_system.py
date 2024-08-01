from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache, cached_property
from typing import Final, Mapping, Sequence, Union

import networkx as nx
import numpy as np
import qutip as qt


@dataclass(frozen=True)
class Object:
    label: str
    dimension: int
    frequency: float
    anharmonicity: float
    relaxation_rate: float
    dephasing_rate: float


class Qubit(Object):
    def __init__(
        self,
        *,
        label: str,
        frequency: float,
        relaxation_rate: float = 0.0,
        dephasing_rate: float = 0.0,
    ):
        super().__init__(
            label=label,
            dimension=2,
            frequency=frequency,
            anharmonicity=0.0,
            relaxation_rate=relaxation_rate,
            dephasing_rate=dephasing_rate,
        )


class Resonator(Object):

    def __init__(
        self,
        *,
        label: str,
        dimension: int,
        frequency: float,
        relaxation_rate: float = 0.0,
        dephasing_rate: float = 0.0,
    ):
        super().__init__(
            label=label,
            dimension=dimension,
            frequency=frequency,
            anharmonicity=0.0,
            relaxation_rate=relaxation_rate,
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
        relaxation_rate: float = 0.0,
        dephasing_rate: float = 0.0,
    ):
        super().__init__(
            label=label,
            dimension=dimension,
            frequency=frequency,
            anharmonicity=anharmonicity,
            relaxation_rate=relaxation_rate,
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
        objects: Sequence[Object],
        couplings: Sequence[Coupling] | None = None,
    ):
        couplings = couplings or []

        # validate objects and couplings
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
            graph.add_node(
                object.label,
                type=type(object).__name__,
                props=asdict(object),
            )
        for coupling in self.couplings:
            graph.add_edge(
                *coupling.pair,
                type=type(coupling).__name__,
                props=asdict(coupling),
            )
        return graph

    @cached_property
    def object_labels(self) -> list[str]:
        return [object.label for object in self.objects]

    @cached_property
    def object_dimensions(self) -> list[int]:
        return [object.dimension for object in self.objects]

    @cached_property
    def coupling_labels(self) -> list[str]:
        return [coupling.label for coupling in self.couplings]

    @cached_property
    def object_hamiltonian(self) -> qt.Qobj:
        H = self.zero_matrix
        for label in self.object_labels:
            H += self.get_object_hamiltonian(label)
        return H

    @cached_property
    def coupling_hamiltonian(self) -> qt.Qobj:
        H = self.zero_matrix
        for label in self.coupling_labels:
            H += self.get_coupling_hamiltonian(label)
        return H

    @cached_property
    def hamiltonian(self) -> qt.Qobj:
        return self.object_hamiltonian + self.coupling_hamiltonian

    @cached_property
    def basis_indices(self) -> list[tuple[int, ...]]:
        return list(np.ndindex(*[dim for dim in self.object_dimensions]))

    @cached_property
    def basis_labels(self) -> list[str]:
        return ["".join(str(i) for i in basis) for basis in self.basis_indices]

    @cached_property
    def zero_matrix(self) -> qt.Qobj:
        return qt.tensor([qt.qzero(dim) for dim in self.object_dimensions])

    @cached_property
    def identity_matrix(self) -> qt.Qobj:
        return qt.tensor([qt.qeye(dim) for dim in self.object_dimensions])

    @cached_property
    def number_matrix(self) -> qt.Qobj:
        return qt.tensor([qt.num(dim) for dim in self.object_dimensions])

    @cache
    def get_index(self, label: str) -> int:
        if label not in self.graph.nodes:
            raise ValueError(f"Object {label} does not exist.")
        return list(self.graph.nodes).index(label)

    @cache
    def get_object(self, label: str) -> Object:
        if label not in self.graph.nodes:
            raise ValueError(f"Object {label} does not exist.")
        ObjectClass = globals()[self.graph.nodes[label]["type"]]
        return ObjectClass(**self.graph.nodes[label]["props"])

    @cache
    def get_coupling(self, label: str | tuple[str, str]) -> Coupling:
        pair = label if isinstance(label, tuple) else tuple(label.split("-"))
        if pair not in self.graph.edges:
            raise ValueError(f"Coupling {pair} does not exist.")
        CouplingClass = globals()[self.graph.edges[pair]["type"]]
        return CouplingClass(**self.graph.edges[pair]["props"])

    @cache
    def get_lowering_operator(self, label: str) -> qt.Qobj:
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

    @cache
    def get_raising_operator(self, label: str) -> qt.Qobj:
        return self.get_lowering_operator(label).dag()

    @cache
    def get_number_operator(self, label: str) -> qt.Qobj:
        return self.get_raising_operator(label) * self.get_lowering_operator(label)

    @cache
    def get_object_hamiltonian(self, label: str) -> qt.Qobj:
        object = self.get_object(label)
        omega = 2 * np.pi * object.frequency
        alpha = 2 * np.pi * object.anharmonicity
        a = self.get_lowering_operator(object.label)
        ad = a.dag()
        return omega * ad * a + 0.5 * alpha * (ad * ad * a * a)

    @cache
    def get_coupling_hamiltonian(self, label: str | tuple[str, str]) -> qt.Qobj:
        pair = label if isinstance(label, tuple) else tuple(label.split("-"))
        coupling = self.get_coupling(pair)
        g = 2 * np.pi * coupling.strength
        a_0 = self.get_lowering_operator(coupling.pair[0])
        ad_0 = a_0.dag()
        a_1 = self.get_lowering_operator(coupling.pair[1])
        ad_1 = a_1.dag()
        return g * (ad_0 * a_1 + ad_1 * a_0)

    def draw(self, **kwargs):
        nx.draw(
            self.graph,
            with_labels=True,
            **kwargs,
        )

    def state(
        self,
        states: Union[
            Mapping[str, int | str | qt.Qobj],
            Sequence[int | str | qt.Qobj],
            None,
        ] = None,
        default: int | str = 0,
    ) -> qt.Qobj:
        if states is None:
            return qt.tensor(
                [self.create_state(dim, default) for dim in self.object_dimensions]
            )

        if isinstance(states, Sequence):
            if len(states) != len(self.objects):
                raise ValueError(
                    f"Number of states ({len(states)}) must match number of objects ({len(self.objects)})."
                )
            states = {
                object.label: state for object, state in zip(self.objects, states)
            }

        if isinstance(states, Mapping):
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
                    else:
                        object_states.append(self.create_state(object.dimension, state))
                else:
                    object_states.append(self.create_state(object.dimension, default))
            return qt.tensor(object_states)
        else:
            raise ValueError("Invalid state input.")

    @staticmethod
    def create_state(dim: int, alias: int | str) -> qt.Qobj:
        if isinstance(alias, int):
            state = qt.basis(dim, alias)
        elif alias in ("0", "g"):
            state = qt.basis(dim, 0)
        elif alias in ("1", "e"):
            state = qt.basis(dim, 1)
        elif alias in ("2", "f"):
            state = qt.basis(dim, 2)
        elif alias == "+":
            state = (qt.basis(dim, 0) + qt.basis(dim, 1)).unit()
        elif alias == "-":
            state = (qt.basis(dim, 0) - qt.basis(dim, 1)).unit()
        elif alias in ("+i", "i"):
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
