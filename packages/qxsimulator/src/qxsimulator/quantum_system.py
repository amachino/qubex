"""Quantum system objects and Hamiltonian utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Final

import networkx as nx
import numpy as np
import qutip as qt


@dataclass(frozen=True)
class Object:
    """Base object definition for a quantum system."""

    label: str
    dimension: int
    frequency: float
    anharmonicity: float
    relaxation_rate: float
    dephasing_rate: float


class Qubit(Object):
    """Two-level qubit object."""

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
    """Resonator object with configurable dimension."""

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
    """Transmon object with anharmonicity."""

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
    """Coupling between two objects in the system."""

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
        """Return coupling label in `a-b` form."""
        return f"{self.pair[0]}-{self.pair[1]}"


class QuantumSystem:
    """Container for system objects and couplings."""

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
        """Return a graph representation of the system."""
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
        """Return the set of node labels."""
        return set(self.graph.nodes)

    @cached_property
    def edge_set(self) -> set[tuple[str, str]]:
        """Return the set of edge tuples."""
        return set(self.graph.edges)

    @cached_property
    def node_list(self) -> list[str]:
        """Return the list of node labels."""
        return list(self.graph.nodes)

    @cached_property
    def edge_list(self) -> list[tuple[str, str]]:
        """Return the list of edge tuples."""
        return list(self.graph.edges)

    @cached_property
    def object_labels(self) -> list[str]:
        """Return labels for all objects."""
        return [obj.label for obj in self.objects]

    @cached_property
    def object_dimensions(self) -> list[int]:
        """Return Hilbert-space dimensions for each object."""
        return [obj.dimension for obj in self.objects]

    @cached_property
    def coupling_labels(self) -> list[str]:
        """Return labels for all couplings."""
        return [coupling.label for coupling in self.couplings]

    @cached_property
    def object_hamiltonian(self) -> qt.Qobj:
        """Return the sum of object Hamiltonians."""
        H = self.zero_matrix
        for label in self.object_labels:
            H += self.get_object_hamiltonian(label)
        return H

    @cached_property
    def coupling_hamiltonian(self) -> qt.Qobj:
        """Return the sum of coupling Hamiltonians."""
        H = self.zero_matrix
        for label in self.coupling_labels:
            H += self.get_coupling_hamiltonian(label)
        return H

    @cached_property
    def hamiltonian(self) -> qt.Qobj:
        """Return the total Hamiltonian."""
        return self.object_hamiltonian + self.coupling_hamiltonian

    @cached_property
    def basis_indices(self) -> list[tuple[int, ...]]:
        """Return basis indices for the full system."""
        return list(np.ndindex(*list(self.object_dimensions)))

    @cached_property
    def basis_labels(self) -> list[str]:
        """Return string labels for basis indices."""
        return ["".join(str(i) for i in basis) for basis in self.basis_indices]

    @cached_property
    def zero_matrix(self) -> qt.Qobj:
        """Return the system-sized zero operator."""
        return qt.tensor(*[qt.qzero(dim) for dim in self.object_dimensions])

    @cached_property
    def identity_matrix(self) -> qt.Qobj:
        """Return the system-sized identity operator."""
        return qt.tensor(*[qt.qeye(dim) for dim in self.object_dimensions])

    @cached_property
    def number_matrix(self) -> qt.Qobj:
        """Return the system-sized number operator."""
        return qt.tensor(*[qt.num(dim) for dim in self.object_dimensions])

    @cached_property
    def ground_state(self) -> qt.Qobj:
        """Return the ground state of the system."""
        return self.state({obj.label: "0" for obj in self.objects})

    def get_index(self, label: str) -> int:
        """Return the index of the object with the given label."""
        if label not in self.node_set:
            raise ValueError(f"Object {label} does not exist.")
        return self.node_list.index(label)

    def get_object(self, label: str) -> Object:
        """Return the object instance for the given label."""
        if label not in self.node_set:
            raise ValueError(f"Object {label} does not exist.")
        node = self.graph.nodes[label]
        ObjectClass = globals()[node["type"]]
        return ObjectClass(**node["props"])

    def get_coupling(self, label: str | tuple[str, str]) -> Coupling:
        """Return the coupling instance for a label or pair."""
        pair = self.to_tuple_pair(label)
        if pair not in self.edge_set:
            if (pair[1], pair[0]) in self.edge_set:
                pair = (pair[1], pair[0])
            else:
                raise ValueError(f"Coupling {pair} does not exist.")
        edge = self.graph.get_edge_data(*pair)
        CouplingClass = globals()[edge["type"]]
        return CouplingClass(**edge["props"])

    def get_lowering_operator(self, label: str) -> qt.Qobj:
        """Return the lowering operator for the target object."""
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

    def get_raising_operator(self, label: str) -> qt.Qobj:
        """Return the raising operator for the target object."""
        return self.get_lowering_operator(label).dag()

    def get_number_operator(self, label: str) -> qt.Qobj:
        """Return the number operator for the target object."""
        return self.get_raising_operator(label) @ self.get_lowering_operator(label)

    def get_object_hamiltonian(self, label: str) -> qt.Qobj:
        """Return the bare Hamiltonian for an object."""
        obj = self.get_object(label)
        omega = 2 * np.pi * obj.frequency
        alpha = 2 * np.pi * obj.anharmonicity
        a = self.get_lowering_operator(obj.label)
        ad = a.dag()
        return omega * (ad @ a) + 0.5 * alpha * (ad @ ad @ a @ a)

    def get_rotating_object_hamiltonian(self, label: str) -> qt.Qobj:
        """Return the rotating-frame Hamiltonian for an object."""
        obj = self.get_object(label)
        alpha = 2 * np.pi * obj.anharmonicity
        a = self.get_lowering_operator(obj.label)
        ad = a.dag()
        return 0.5 * alpha * (ad @ ad @ a @ a)

    def get_coupling_term(self, label: str | tuple[str, str]) -> qt.Qobj:
        """Return the coupling term for a pair of objects."""
        coupling = self.get_coupling(label)
        g = 2 * np.pi * coupling.strength
        ad_0 = self.get_lowering_operator(coupling.pair[0]).dag()
        a_1 = self.get_lowering_operator(coupling.pair[1])
        return g * (ad_0 @ a_1)

    def get_coupling_hamiltonian(self, label: str | tuple[str, str]) -> qt.Qobj:
        """Return the coupling Hamiltonian for a pair of objects."""
        term = self.get_coupling_term(label)
        return term + term.dag()

    def get_coupling_detuning(self, label: str | tuple[str, str]) -> float:
        """Return the detuning for a coupling pair."""
        pair = self.to_tuple_pair(label)
        omega_0 = 2 * np.pi * self.get_object(pair[0]).frequency
        omega_1 = 2 * np.pi * self.get_object(pair[1]).frequency
        return omega_1 - omega_0

    def get_rotating_coupling_hamiltonian(self, label: str, time: float) -> qt.Qobj:
        """Return the rotating-frame coupling Hamiltonian."""
        term = self.get_coupling_term(label)
        detuning = self.get_coupling_detuning(label)
        term = term * np.exp(-1j * detuning * time)
        H = term + term.dag()
        return H

    def get_rotating_hamiltonian(self, time: float) -> qt.Qobj:
        """Return the total rotating-frame Hamiltonian at time."""
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
        """Return the projection operator onto specified levels."""
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
        """Truncate a superoperator to the qubit subspace."""
        if not isinstance(superoperator, qt.Qobj):
            raise TypeError("Input must be a Qobj.")

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

    def draw(self, **kwargs) -> None:
        """Draw the system graph."""
        nx.draw(
            self.graph,
            with_labels=True,
            **kwargs,
        )

    def state(
        self,
        states: Mapping[str, int | str | qt.Qobj]
        | Sequence[int | str | qt.Qobj]
        | None = None,
        default: int | str = 0,
    ) -> qt.Qobj:
        """Return a composite state from labels or indices."""
        if states is None:
            return qt.tensor(
                *[self.create_state(dim, default) for dim in self.object_dimensions]
            )

        if isinstance(states, Sequence):
            if len(states) != len(self.objects):
                raise ValueError(
                    f"Number of states ({len(states)}) must match number of objects ({len(self.objects)})."
                )
            states = {
                obj.label: state
                for obj, state in zip(self.objects, states, strict=True)
            }

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
            raise TypeError("Invalid state input.")

    def substate(self, label: str, alias: int | str) -> qt.Qobj:
        """Return a single-object state for the given alias."""
        obj = self.get_object(label)
        return self.create_state(obj.dimension, alias)

    @staticmethod
    def create_state(dim: int, alias: int | str) -> qt.Qobj:
        """Create a basis or superposition state from an alias."""
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
        """Normalize a coupling label to a tuple pair."""
        if isinstance(label, tuple):
            return label
        else:
            pair = tuple(label.split("-"))
            if len(pair) != 2:
                raise ValueError(f"Invalid coupling label: {label}")
            return pair

    def get_coupled_objects(self, label: str) -> list[Object]:
        """Return objects coupled to the given label."""
        if label not in self.node_set:
            raise ValueError(f"Object {label} does not exist.")
        neighbors = list(self.graph.neighbors(label))
        return [self.get_object(neighbor) for neighbor in neighbors]

    def get_effective_frequency(self, label: str) -> float:
        """Return the effective frequency including shifts."""
        obj = self.get_object(label)
        shift = self.get_frequency_shift(label)
        return obj.frequency + shift

    def get_frequency_shift(self, label: str) -> float:
        """Return total frequency shift from couplings."""
        shift = 0.0
        for neighbor in self.graph.neighbors(label):
            shift += self.get_lamb_shift((label, neighbor))
            shift += self.get_static_zz((label, neighbor))
        return shift

    def get_lamb_shift(self, label: str | tuple[str, str]) -> float:
        """Return the Lamb shift for a coupling pair."""
        pair = self.to_tuple_pair(label)
        coupling = self.get_coupling(pair)
        obj_0 = self.get_object(pair[0])
        obj_1 = self.get_object(pair[1])

        g = coupling.strength
        delta = obj_0.frequency - obj_1.frequency
        return (g**2) / delta

    def get_static_zz(self, label: str | tuple[str, str]) -> float:
        """Return the static ZZ shift for a coupling pair."""
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
        """Return a local rotation matrix for the given angles."""
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
