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
        **kwargs,
    ):
        super().__init__(
            label=label,
            dimension=2,
            frequency=frequency,
            anharmonicity=np.inf,
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
        **kwargs,
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
        anharmonicity: float | None = None,
        relaxation_rate: float = 0.0,
        dephasing_rate: float = 0.0,
        **kwargs,
    ):
        if anharmonicity is None:
            anharmonicity = -0.05 * frequency  # typical transmon anharmonicity
        super().__init__(
            label=label,
            dimension=dimension,
            frequency=frequency,
            anharmonicity=anharmonicity,
            relaxation_rate=relaxation_rate,
            dephasing_rate=dephasing_rate,
        )


@dataclass(init=False)
class Coupling:
    pair: tuple[str, str]
    strength: float

    def __init__(
        self,
        *,
        pair: tuple[str, str] | tuple[Object, Object],
        strength: float,
    ):
        if len(pair) != 2:
            raise ValueError("Coupling pair must have exactly two elements.")
        self.pair: Final[tuple[str, str]] = (
            pair[0].label if isinstance(pair[0], Object) else pair[0],
            pair[1].label if isinstance(pair[1], Object) else pair[1],
        )
        self.strength: Final[float] = strength

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
        object_labels = [obj.label for obj in objects]
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
        for obj in self.objects:
            graph.add_node(
                obj.label,
                type=type(obj).__name__,
                props=asdict(obj),
            )
        for coupling in self.couplings:
            graph.add_edge(
                *coupling.pair,
                type=type(coupling).__name__,
                props=asdict(coupling),
            )
        return graph

    @cached_property
    def node_set(self) -> set[str]:
        return set(self.graph.nodes)

    @cached_property
    def edge_set(self) -> set[tuple[str, str]]:
        return set(self.graph.edges)

    @cached_property
    def node_list(self) -> list[str]:
        return list(self.graph.nodes)

    @cached_property
    def edge_list(self) -> list[tuple[str, str]]:
        return list(self.graph.edges)

    @cached_property
    def object_labels(self) -> list[str]:
        return [obj.label for obj in self.objects]

    @cached_property
    def object_dimensions(self) -> list[int]:
        return [obj.dimension for obj in self.objects]

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
        return qt.tensor(*[qt.qzero(dim) for dim in self.object_dimensions])

    @cached_property
    def identity_matrix(self) -> qt.Qobj:
        return qt.tensor(*[qt.qeye(dim) for dim in self.object_dimensions])

    @cached_property
    def number_matrix(self) -> qt.Qobj:
        return qt.tensor(*[qt.num(dim) for dim in self.object_dimensions])

    @cached_property
    def ground_state(self) -> qt.Qobj:
        return self.state({obj.label: "0" for obj in self.objects})

    @cache
    def get_index(self, label: str) -> int:
        if label not in self.node_set:
            raise ValueError(f"Object {label} does not exist.")
        return self.node_list.index(label)

    @cache
    def get_object(self, label: str) -> Object:
        if label not in self.node_set:
            raise ValueError(f"Object {label} does not exist.")
        node = self.graph.nodes[label]
        ObjectClass = globals()[node["type"]]
        return ObjectClass(**node["props"])

    @cache
    def get_coupling(self, label: str | tuple[str, str]) -> Coupling:
        pair = self.to_tuple_pair(label)
        if pair not in self.edge_set:
            if (pair[1], pair[0]) in self.edge_set:
                pair = (pair[1], pair[0])
            else:
                raise ValueError(f"Coupling {pair} does not exist.")
        edge = self.graph.get_edge_data(*pair)
        CouplingClass = globals()[edge["type"]]
        return CouplingClass(**edge["props"])

    @cache
    def get_lowering_operator(self, label: str) -> qt.Qobj:
        if label not in self.node_set:
            raise ValueError(f"Node {label} does not exist.")
        return qt.tensor(
            *[
                (
                    qt.destroy(obj.dimension)
                    if obj.label == label
                    else qt.qeye(obj.dimension)
                )
                for obj in self.objects
            ]
        )

    @cache
    def get_raising_operator(self, label: str) -> qt.Qobj:
        return self.get_lowering_operator(label).dag()

    @cache
    def get_number_operator(self, label: str) -> qt.Qobj:
        return self.get_raising_operator(label) @ self.get_lowering_operator(label)

    @cache
    def get_object_hamiltonian(self, label: str) -> qt.Qobj:
        obj = self.get_object(label)
        omega = 2 * np.pi * obj.frequency
        alpha = 2 * np.pi * obj.anharmonicity
        a = self.get_lowering_operator(obj.label)
        ad = a.dag()
        return omega * (ad @ a) + 0.5 * alpha * (ad @ ad @ a @ a)

    @cache
    def get_rotating_object_hamiltonian(self, label: str) -> qt.Qobj:
        obj = self.get_object(label)
        alpha = 2 * np.pi * obj.anharmonicity
        a = self.get_lowering_operator(obj.label)
        ad = a.dag()
        return 0.5 * alpha * (ad @ ad @ a @ a)

    @cache
    def get_coupling_term(self, label: str | tuple[str, str]) -> qt.Qobj:
        coupling = self.get_coupling(label)
        g = 2 * np.pi * coupling.strength
        ad_0 = self.get_lowering_operator(coupling.pair[0]).dag()
        a_1 = self.get_lowering_operator(coupling.pair[1])
        return g * (ad_0 @ a_1)

    @cache
    def get_coupling_hamiltonian(self, label: str | tuple[str, str]) -> qt.Qobj:
        term = self.get_coupling_term(label)
        return term + term.dag()

    @cache
    def get_coupling_detuning(self, label: str | tuple[str, str]) -> float:
        pair = self.to_tuple_pair(label)
        omega_0 = 2 * np.pi * self.get_object(pair[0]).frequency
        omega_1 = 2 * np.pi * self.get_object(pair[1]).frequency
        return omega_1 - omega_0

    def get_rotating_coupling_hamiltonian(self, label: str, time: float) -> qt.Qobj:
        term = self.get_coupling_term(label)
        detuning = self.get_coupling_detuning(label)
        term = term * np.exp(-1j * detuning * time)
        H = term + term.dag()
        return H

    def get_rotating_hamiltonian(self, time: float) -> qt.Qobj:
        H = self.zero_matrix
        for obj in self.objects:
            H += self.get_rotating_object_hamiltonian(obj.label)
        for coupling in self.couplings:
            H += self.get_rotating_coupling_hamiltonian(coupling.label, time)
        return H

    def get_projection_operator(
        self,
        levels: Sequence[int] = (0, 1),
    ) -> qt.Qobj:
        return qt.tensor(
            *[
                qt.Qobj(
                    sum(
                        qt.projection(obj.dimension, level, level)
                        for level in levels
                        if level < obj.dimension
                    )
                )
                for obj in self.objects
            ]
        )

    def truncate_superoperator(
        self,
        superoperator: qt.Qobj,
    ) -> qt.Qobj:
        if not isinstance(superoperator, qt.Qobj):
            raise ValueError("Input must be a Qobj.")

        choi = qt.to_choi(superoperator)
        n_objects = len(self.objects)
        dims = np.array(choi.dims).flatten()
        n_dims = len(dims)

        choi_truncated = qt.Qobj(
            choi.full()
            .reshape(*dims)[tuple(slice(0, 2) for _ in range(n_dims))]
            .reshape((4**n_objects, 4**n_objects)),
            dims=[[[2] * n_objects] * 2] * 2,
            superrep=choi.superrep,
        )
        return qt.to_super(choi_truncated)

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
                *[self.create_state(dim, default) for dim in self.object_dimensions]
            )

        if isinstance(states, Sequence):
            if len(states) != len(self.objects):
                raise ValueError(
                    f"Number of states ({len(states)}) must match number of objects ({len(self.objects)})."
                )
            states = {obj.label: state for obj, state in zip(self.objects, states)}

        if isinstance(states, Mapping):
            for label in states:
                if label not in self.node_set:
                    raise ValueError(f"Object {label} does not exist.")

            object_states = []
            for obj in self.objects:
                if obj.label in states:
                    state = states[obj.label]
                    if isinstance(state, qt.Qobj):
                        if state.shape != (obj.dimension, 1):
                            raise ValueError(
                                f"State for object {obj.label} must have shape ({obj.dimension}, 1)."
                            )
                        object_states.append(state)
                    else:
                        object_states.append(self.create_state(obj.dimension, state))
                else:
                    object_states.append(self.create_state(obj.dimension, default))
            return qt.tensor(*object_states)
        else:
            raise ValueError("Invalid state input.")

    def substate(self, label: str, alias: int | str) -> qt.Qobj:
        obj = self.get_object(label)
        return self.create_state(obj.dimension, alias)

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

    @staticmethod
    def to_tuple_pair(label: str | tuple[str, str]) -> tuple[str, str]:
        if isinstance(label, tuple):
            return label
        else:
            pair = tuple(label.split("-"))
            if len(pair) != 2:
                raise ValueError(f"Invalid coupling label: {label}")
            return pair

    def get_coupled_objects(self, label: str) -> list[Object]:
        if label not in self.node_set:
            raise ValueError(f"Object {label} does not exist.")
        neighbors = list(self.graph.neighbors(label))
        return [self.get_object(neighbor) for neighbor in neighbors]

    def get_effective_frequency(self, label: str) -> float:
        obj = self.get_object(label)
        shift = self.get_frequency_shift(label)
        return obj.frequency + shift

    def get_frequency_shift(self, label: str) -> float:
        shift = 0.0
        for neighbor in self.graph.neighbors(label):
            shift += self.get_lamb_shift((label, neighbor))
            shift += self.get_static_zz((label, neighbor))
        return shift

    def get_lamb_shift(self, label: str | tuple[str, str]) -> float:
        pair = self.to_tuple_pair(label)
        coupling = self.get_coupling(pair)
        obj_0 = self.get_object(pair[0])
        obj_1 = self.get_object(pair[1])

        g = coupling.strength
        delta = obj_0.frequency - obj_1.frequency
        return (g**2) / delta

    def get_static_zz(self, label: str | tuple[str, str]) -> float:
        pair = self.to_tuple_pair(label)
        obj_0 = self.get_object(pair[0])
        obj_1 = self.get_object(pair[1])

        if obj_0.dimension < 3 or obj_1.dimension < 3:
            return 0.0

        g = self.get_coupling(pair).strength
        delta = obj_0.frequency - obj_1.frequency
        alpha_0 = obj_0.anharmonicity
        alpha_1 = obj_1.anharmonicity
        xi = g**2 * (alpha_0 + alpha_1) / ((delta + alpha_0) * (delta - alpha_1))
        return xi

    def get_rotation_matrix(
        self,
        angles: dict[str, float],
    ) -> qt.Qobj:
        U = qt.tensor(
            *[
                qt.qeye(self.get_object(label).dimension)
                if label not in angles
                else (
                    1j * angles[label] * qt.num(self.get_object(label).dimension)
                ).expm()
                for label in self.object_labels
            ]
        )
        return U
