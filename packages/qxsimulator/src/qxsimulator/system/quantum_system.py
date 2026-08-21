"""Quantum system objects and Hamiltonian utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from functools import cached_property
from typing import Final, Literal, Protocol, TypeAlias, cast, runtime_checkable

import networkx as nx
import numpy as np
import qutip as qt
from scipy.optimize import linear_sum_assignment
from typing_extensions import deprecated

import qxsimulator.gates as gates

from .compiled_object import CompiledObject
from .models import Coupling, Object, Qubit, Resonator, Transmon

EvaluationSpace: TypeAlias = (
    Literal["computational", "full"] | Mapping[str, Sequence[int]]
)
GateTarget: TypeAlias = str | tuple[str, ...]
GateLike: TypeAlias = str | qt.Qobj


@runtime_checkable
class UnitarySpecification(Protocol):
    """Define the mapping interface accepted by `QuantumSystem.unitary`."""

    def items(self) -> Iterable[tuple[GateTarget, GateLike]]:
        """
        Iterate over gate targets and their gate specifications.

        Returns
        -------
        Iterable[tuple[GateTarget, GateLike]]
            Target-gate pairs. Target tuple order determines gate orientation.
        """
        ...


__all__ = [
    "Coupling",
    "EvaluationSpace",
    "GateLike",
    "GateTarget",
    "Object",
    "QuantumSystem",
    "Qubit",
    "Resonator",
    "Transmon",
    "UnitarySpecification",
]


class QuantumSystem:
    """
    Assemble local object models and pairwise couplings into one system.

    Parameters
    ----------
    objects : Sequence[Object]
        Local object specifications in tensor-factor order. Labels must be
        unique.
    couplings : Sequence[Coupling] | None, optional
        Pairwise couplings between objects in `objects`. Omit for an uncoupled
        system.

    Raises
    ------
    ValueError
        If object labels are duplicated, a coupling references an unknown
        object, multiple couplings describe the same unordered pair, or a
        local object cannot be compiled.

    Notes
    -----
    Input sequences are copied to tuples, and every local model is compiled
    during initialization. Hamiltonians are represented as `H / hbar` in
    rad/ns; object and coupling specifications store cyclic frequencies in
    GHz.
    """

    def __init__(
        self,
        *,
        objects: Sequence[Object],
        couplings: Sequence[Coupling] | None = None,
    ):
        objects = tuple(objects)
        couplings = tuple(couplings or ())

        # validate objects and couplings
        objects_by_label = {obj.label: obj for obj in objects}
        if len(objects_by_label) != len(objects):
            raise ValueError("Objects must have unique labels.")
        for coupling in couplings:
            if any(label not in objects_by_label for label in coupling.pair):
                raise ValueError("Couplings must be between existing objects.")
        couplings_by_pair = {
            self._normalize_pair(coupling.pair): coupling for coupling in couplings
        }
        if len(couplings_by_pair) != len(couplings):
            raise ValueError("Couplings must have unique object pairs.")

        self.objects: Final = objects
        self.couplings: Final = couplings
        self._objects_by_label: Final = objects_by_label
        self._object_indices_by_label: Final = {
            obj.label: index for index, obj in enumerate(objects)
        }
        self._couplings_by_pair: Final = couplings_by_pair
        self._compiled_objects_by_label: Final = {
            obj.label: obj.compile() for obj in objects
        }
        self._dressed_computational_energies_by_pair: Final[
            dict[tuple[str, str], np.ndarray]
        ] = {}

    @property
    def graph(self) -> nx.Graph:
        """
        Build a graph representation of the system topology.

        Returns
        -------
        nx.Graph
            Fresh graph whose label-string nodes identify objects and whose
            edges identify couplings. Each node and edge includes its model
            class name as `type` and its dataclass fields as `props`.
        """
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

    @property
    @deprecated("node_set is deprecated; use set(object_labels) instead.")
    def node_set(self) -> set[str]:
        """Return all object labels as a set."""
        return set(self.graph.nodes)

    @property
    @deprecated("edge_set is deprecated; use set(graph.edges) instead.")
    def edge_set(self) -> set[tuple[str, str]]:
        """Return all graph edges as unordered label pairs."""
        return set(self.graph.edges)

    @property
    @deprecated("node_list is deprecated; use object_labels instead.")
    def node_list(self) -> list[str]:
        """Return object labels in tensor-factor order."""
        return list(self.graph.nodes)

    @property
    @deprecated("edge_list is deprecated; use list(graph.edges) instead.")
    def edge_list(self) -> list[tuple[str, str]]:
        """Return graph edges as unordered label pairs."""
        return list(self.graph.edges)

    @property
    def object_labels(self) -> list[str]:
        """Return object labels in tensor-factor order."""
        return [obj.label for obj in self.objects]

    @property
    def object_dimensions(self) -> list[int]:
        """Return local Hilbert-space dimensions in tensor-factor order."""
        return [obj.dimension for obj in self.objects]

    @property
    def coupling_labels(self) -> list[str]:
        """Return coupling labels in the order supplied at initialization."""
        return [coupling.label for coupling in self.couplings]

    @cached_property
    def object_hamiltonian(self) -> qt.Qobj:
        """Return the sum of embedded bare-object Hamiltonians in rad/ns."""
        H = self.zero_matrix
        for label in self.object_labels:
            H += self.get_object_hamiltonian(label)
        return H

    @cached_property
    def coupling_hamiltonian(self) -> qt.Qobj:
        """Return the sum of exchange-coupling Hamiltonians in rad/ns."""
        H = self.zero_matrix
        for label in self.coupling_labels:
            H += self.get_coupling_hamiltonian(label)
        return H

    @cached_property
    def hamiltonian(self) -> qt.Qobj:
        """Return the time-independent lab-frame Hamiltonian in rad/ns."""
        return self.object_hamiltonian + self.coupling_hamiltonian

    @cached_property
    def basis_indices(self) -> list[tuple[int, ...]]:
        """Return all product-basis indices in tensor-flattening order."""
        return list(np.ndindex(*list(self.object_dimensions)))

    @cached_property
    def basis_labels(self) -> list[str]:
        """
        Return compact string labels for all product-basis indices.

        Notes
        -----
        Each label concatenates the decimal local indices without separators;
        use `basis_indices` when a local dimension can exceed ten.
        """
        return ["".join(str(i) for i in basis) for basis in self.basis_indices]

    @cached_property
    def zero_matrix(self) -> qt.Qobj:
        """Return the zero operator on the full tensor-product Hilbert space."""
        return qt.tensor(*[qt.qzero(dim) for dim in self.object_dimensions])

    @cached_property
    def identity_matrix(self) -> qt.Qobj:
        """Return the identity operator on the full tensor-product Hilbert space."""
        return qt.tensor(*[qt.qeye(dim) for dim in self.object_dimensions])

    @property
    @deprecated(
        "number_matrix is deprecated; use get_number_operator(label) "
        "for local number operators."
    )
    def number_matrix(self) -> qt.Qobj:
        """Return the tensor product of all local level-index operators."""
        return qt.tensor(*[qt.num(dim) for dim in self.object_dimensions])

    @property
    def ground_state(self) -> qt.Qobj:
        """Return the product state with every object in local level zero."""
        return self.state({obj.label: "0" for obj in self.objects})

    def get_index(self, label: str) -> int:
        """
        Return an object's tensor-factor index.

        Parameters
        ----------
        label : str
            Object label.

        Returns
        -------
        int
            Zero-based position in `objects`.

        Raises
        ------
        ValueError
            If `label` does not identify an object in the system.
        """
        try:
            return self._object_indices_by_label[label]
        except KeyError:
            raise ValueError(f"Object {label} does not exist.") from None

    def get_object(self, label: str) -> Object:
        """
        Return an object specification by label.

        Parameters
        ----------
        label : str
            Object label.

        Returns
        -------
        Object
            Original specification supplied at initialization.

        Raises
        ------
        ValueError
            If `label` does not identify an object in the system.
        """
        try:
            return self._objects_by_label[label]
        except KeyError:
            raise ValueError(f"Object {label} does not exist.") from None

    def _get_compiled_object(self, label: str) -> CompiledObject:
        """Return the cached local model compiled for an object label."""
        try:
            return self._compiled_objects_by_label[label]
        except KeyError:
            raise ValueError(f"Object {label} does not exist.") from None

    def get_coupling(self, label: str | tuple[str, str]) -> Coupling:
        """
        Return a coupling specification by label or endpoint pair.

        Parameters
        ----------
        label : str | tuple[str, str]
            Hyphen-separated coupling label or two object labels. Pair
            orientation is ignored for lookup.

        Returns
        -------
        Coupling
            Original specification supplied at initialization.

        Raises
        ------
        ValueError
            If the label is malformed or the unordered pair is not coupled.
        """
        pair = self.to_tuple_pair(label)
        try:
            return self._couplings_by_pair[self._normalize_pair(pair)]
        except KeyError:
            raise ValueError(f"Coupling {pair} does not exist.") from None

    def get_lowering_operator(self, label: str) -> qt.Qobj:
        """
        Return an object's lowering operator embedded in the full system.

        For a Duffing object this is the truncated annihilation operator. For
        a cosine transmon it contains adjacent transitions of the normalized
        projected charge operator.
        """
        return self._embed_local_operator(
            label,
            self._get_compiled_object(label).lowering_operator,
        )

    def get_raising_operator(self, label: str) -> qt.Qobj:
        """Return the adjoint lowering operator embedded in the full system."""
        return self.get_lowering_operator(label).dag()

    def get_number_operator(self, label: str) -> qt.Qobj:
        """
        Return an object's level-index operator embedded in the full system.

        Notes
        -----
        This is `diag(0, 1, ...)` in the retained local basis. For a cosine
        transmon it is not the Cooper-pair charge operator.
        """
        obj = self.get_object(label)
        return self._embed_local_operator(label, qt.num(obj.dimension))

    def get_interaction_operator(self, label: str) -> qt.Qobj:
        """
        Return an object's full local interaction operator embedded in the system.

        For a Duffing object the local operator is `a + a.dag()`. For a cosine
        transmon it is the normalized projected relative-charge operator,
        including diagonal and nonadjacent matrix elements.
        """
        return self._embed_local_operator(
            label,
            self._get_compiled_object(label).interaction_operator,
        )

    def get_collapse_operators(self, label: str) -> tuple[qt.Qobj, ...]:
        """
        Return an object's phenomenological collapse operators.

        Parameters
        ----------
        label : str
            Object label.

        Returns
        -------
        tuple[qt.Qobj, ...]
            Operators embedded in the full system, ordered as relaxation then
            pure dephasing when their corresponding rates are positive.
        """
        return tuple(
            self._embed_local_operator(label, operator)
            for operator in self._get_compiled_object(label).collapse_operators
        )

    def get_object_hamiltonian(self, label: str) -> qt.Qobj:
        """
        Return an embedded bare-object Hamiltonian in rad/ns.

        The returned operator represents `H / hbar` before coupling terms are
        added.
        """
        return self._embed_local_operator(
            label,
            self._get_compiled_object(label).hamiltonian,
        )

    def get_rotating_object_hamiltonian(self, label: str) -> qt.Qobj:
        """
        Return an object's Hamiltonian in its local frequency frame.

        The transformation subtracts `2 * pi * frequency * N` from the bare
        local Hamiltonian, where `N` is the retained-basis level-index
        operator. The result represents `H / hbar` in rad/ns.
        """
        obj = self.get_object(label)
        reference_hamiltonian = (
            2 * np.pi * obj.frequency * self.get_number_operator(label)
        )
        return self.get_object_hamiltonian(label) - reference_hamiltonian

    def get_coupling_term(self, label: str | tuple[str, str]) -> qt.Qobj:
        """
        Return the oriented exchange term for a coupled object pair.

        Parameters
        ----------
        label : str | tuple[str, str]
            Hyphen-separated coupling label or endpoint pair used to look up
            the coupling. Lookup ignores pair orientation.

        Returns
        -------
        qt.Qobj
            The term `2 * pi * g * L_0.dag() * L_1` in rad/ns, embedded in the
            full system. Here indices 0 and 1 follow the stored `Coupling.pair`
            orientation, not necessarily the lookup argument. The adjoint is
            not included.
        """
        coupling = self.get_coupling(label)
        g = 2 * np.pi * coupling.strength
        ad_0 = self.get_lowering_operator(coupling.pair[0]).dag()
        a_1 = self.get_lowering_operator(coupling.pair[1])
        return g * (ad_0 @ a_1)

    def get_coupling_hamiltonian(self, label: str | tuple[str, str]) -> qt.Qobj:
        """
        Return the Hermitian exchange-coupling Hamiltonian for a pair.

        Notes
        -----
        The coupling uses adjacent lowering and raising operators and therefore
        applies a coupling rotating-wave approximation independently of each
        object's local Hamiltonian model.
        """
        term = self.get_coupling_term(label)
        return term + term.dag()

    def get_coupling_detuning(self, label: str | tuple[str, str]) -> float:
        """
        Return the oriented angular-frequency detuning for a coupling pair.

        Returns
        -------
        float
            `2 * pi * (frequency_1 - frequency_0)` in rad/ns, using the given
            pair orientation.
        """
        pair = self.to_tuple_pair(label)
        omega_0 = 2 * np.pi * self.get_object(pair[0]).frequency
        omega_1 = 2 * np.pi * self.get_object(pair[1]).frequency
        return omega_1 - omega_0

    def get_rotating_coupling_hamiltonian(self, label: str, time: float) -> qt.Qobj:
        """
        Return a pair's exchange Hamiltonian in local frequency frames.

        Parameters
        ----------
        label : str
            Hyphen-separated coupling label. Its endpoint order fixes the
            detuning sign.
        time : float
            Evolution time in ns.

        Returns
        -------
        qt.Qobj
            Hermitian coupling Hamiltonian `H / hbar` in rad/ns, with phase
            `exp(-1j * detuning * time)` multiplying the exchange term.

        Notes
        -----
        The non-Hermitian exchange term follows the stored `Coupling.pair`
        orientation, while the detuning follows `label`. They coincide when
        the derived `Coupling.label` is used.
        """
        term = self.get_coupling_term(label)
        detuning = self.get_coupling_detuning(label)
        term = term * np.exp(-1j * detuning * time)
        H = term + term.dag()
        return H

    def get_rotating_hamiltonian(self, time: float) -> qt.Qobj:
        """
        Return the total Hamiltonian in local object-frequency frames.

        Parameters
        ----------
        time : float
            Evolution time in ns.

        Returns
        -------
        qt.Qobj
            Time-dependent `H / hbar` in rad/ns, including every local object
            term and exchange coupling.
        """
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
        """
        Return the product projector onto common local level indices.

        Parameters
        ----------
        levels : Sequence[int], optional
            Nonnegative local indices to retain for every object. Indices
            greater than or equal to a particular object's dimension are
            omitted for that tensor factor. The default is the computational
            levels `(0, 1)`.

        Returns
        -------
        qt.Qobj
            Projector on the full system Hilbert space.
        """
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

    def unitary(
        self,
        operations: UnitarySpecification,
        *,
        levels: Mapping[str, Sequence[int]] | None = None,
    ) -> qt.Qobj:
        """
        Build a physical-system unitary from parallel labeled gates.

        Parameters
        ----------
        operations : Mapping[str | tuple[str, ...], str | qt.Qobj]
            Gates keyed by object label or ordered object-label tuple. A
            hyphen-separated key such as `Q04-Q01` is equivalent to
            `("Q04", "Q01")` unless it exactly matches one object label.
            String values are resolved through `qxsimulator.gates`.
        levels : Mapping[str, Sequence[int]] | None, optional
            Physical levels used for each gate tensor factor. By default, a
            gate of local dimension `d` uses levels `0` through `d - 1`.

        Returns
        -------
        qt.Qobj
            Unitary embedded in the full physical Hilbert space. Objects not
            targeted by a gate and states outside each selected gate subspace
            are left unchanged.

        Raises
        ------
        ValueError
            If a target, gate, tensor-factor dimension, or level selection is
            invalid, or if operations target the same object more than once.

        Notes
        -----
        The mapping describes parallel operations on disjoint objects. The
        target order is significant for oriented gates such as `CNOT` and
        `ZX90`; the first gate tensor factor maps to the first target label.
        Sequential gates should be composed by multiplying the returned
        unitaries.
        """
        levels = {} if levels is None else levels
        self._validate_level_labels(levels)
        result = self.identity_matrix
        used_labels: set[str] = set()

        for target, gate in operations.items():
            target_labels = self._resolve_gate_target(target)
            overlap = used_labels.intersection(target_labels)
            if overlap:
                repeated = sorted(overlap)[0]
                raise ValueError(
                    f"Object {repeated} is targeted by more than one gate."
                )
            used_labels.update(target_labels)

            operator = gates.get(gate) if isinstance(gate, str) else gate
            gate_dimensions = self._validate_gate(operator, target_labels)
            selected_levels = {
                label: self._validate_levels(
                    label,
                    levels.get(label, tuple(range(gate_dimension))),
                    expected_count=gate_dimension,
                )
                for label, gate_dimension in zip(
                    target_labels,
                    gate_dimensions,
                    strict=True,
                )
            }
            result = (
                self._embed_gate(
                    target_labels,
                    operator,
                    gate_dimensions,
                    selected_levels,
                )
                @ result
            )

        return result

    def get_subspace_dimensions(
        self,
        levels: EvaluationSpace = "computational",
    ) -> list[int]:
        """
        Return local dimensions for an evaluation subspace.

        Parameters
        ----------
        levels : {"computational", "full"} | Mapping[str, Sequence[int]], optional
            Level selector. `"computational"` retains up to levels 0 and 1 for
            each object; `"full"` retains every local level. A mapping
            overrides selected objects while unspecified objects use their
            computational levels.

        Returns
        -------
        list[int]
            Selected level counts in tensor-factor order.

        Raises
        ------
        TypeError
            If a selected level is not an integer.
        ValueError
            If the selector, object label, or level selection is invalid.
        """
        return [
            len(selected_levels)
            for selected_levels in self._resolve_evaluation_levels(levels).values()
        ]

    def project_operator(
        self,
        operator: qt.Qobj,
        levels: EvaluationSpace = "computational",
    ) -> qt.Qobj:
        """
        Restrict a full-system operator to selected local levels.

        Parameters
        ----------
        operator : qt.Qobj
            Operator whose tensor dimensions match `object_dimensions`.
        levels : {"computational", "full"} | Mapping[str, Sequence[int]], optional
            Evaluation-space selector. Mapping values set both the retained
            levels and their output basis order.

        Returns
        -------
        qt.Qobj
            Operator on the selected tensor-product subspace.

        Raises
        ------
        TypeError
            If `operator` is not a `qt.Qobj` or a selected level is not an
            integer.
        ValueError
            If the input is not an operator, its dimensions do not match the
            system, or the level selector is invalid.
        """
        if not isinstance(operator, qt.Qobj):
            raise TypeError("Input must be a Qobj.")
        if not operator.isoper:
            raise ValueError("Input must be an operator.")
        if operator.dims != [self.object_dimensions, self.object_dimensions]:
            raise ValueError("Operator dimensions do not match the system.")

        selected_levels = self._resolve_evaluation_levels(levels)
        subspace_dimensions = [len(value) for value in selected_levels.values()]
        indices = tuple(
            np.ix_(*(list(selected_levels.values()) + list(selected_levels.values())))
        )
        projected = operator.full().reshape(
            *self.object_dimensions,
            *self.object_dimensions,
        )[indices]
        dimension = int(np.prod(subspace_dimensions))
        return qt.Qobj(
            projected.reshape((dimension, dimension)),
            dims=[subspace_dimensions, subspace_dimensions],
        )

    def project_superoperator(
        self,
        superoperator: qt.Qobj,
        levels: EvaluationSpace = "computational",
    ) -> qt.Qobj:
        """
        Restrict a full-system superoperator to selected local levels.

        Parameters
        ----------
        superoperator : qt.Qobj
            Superoperator whose input and output tensor dimensions match the
            system.
        levels : {"computational", "full"} | Mapping[str, Sequence[int]], optional
            Evaluation-space selector. Mapping values set both the retained
            levels and their output basis order.

        Returns
        -------
        qt.Qobj
            Superoperator on the selected tensor-product subspace.

        Raises
        ------
        TypeError
            If `superoperator` is not a `qt.Qobj` or a selected level is not an
            integer.
        ValueError
            If the input is not a superoperator, its dimensions do not match
            the system, or the level selector is invalid.

        Notes
        -----
        Projection is performed through the Choi representation. If population
        can leave the selected subspace, the result need not be trace
        preserving.
        """
        if not isinstance(superoperator, qt.Qobj):
            raise TypeError("Input must be a Qobj.")
        if not superoperator.issuper:
            raise ValueError("Input must be a superoperator.")

        selected_levels = self._resolve_evaluation_levels(levels)
        subspace_dimensions = [len(value) for value in selected_levels.values()]
        choi = qt.to_choi(superoperator)
        choi_dimensions = np.asarray(choi.dims).flatten()
        if list(choi_dimensions) != self.object_dimensions * 4:
            raise ValueError("Superoperator dimensions do not match the system.")
        indices = tuple(np.ix_(*(list(selected_levels.values()) * 4)))
        dimension = int(np.prod(subspace_dimensions))
        projected_choi = qt.Qobj(
            choi.full()
            .reshape(*choi_dimensions)[indices]
            .reshape((dimension**2, dimension**2)),
            dims=[
                [subspace_dimensions.copy(), subspace_dimensions.copy()],
                [subspace_dimensions.copy(), subspace_dimensions.copy()],
            ],
            superrep=choi.superrep,
        )
        return qt.to_super(projected_choi)

    def truncate_superoperator(
        self,
        superoperator: qt.Qobj,
    ) -> qt.Qobj:
        """Restrict a full-system superoperator to computational levels."""
        return self.project_superoperator(superoperator)

    def truncate_operator(
        self,
        operator: qt.Qobj,
    ) -> qt.Qobj:
        """Restrict a full-system operator to computational levels."""
        return self.project_operator(operator)

    def draw(self, **kwargs) -> None:
        """
        Draw the system topology with NetworkX.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to `networkx.draw`.

        Notes
        -----
        This function draws on the active Matplotlib figure or axes.
        """
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
        """
        Build a product ket from local state specifications.

        Parameters
        ----------
        states : Mapping[str, int | str | qt.Qobj] | Sequence[int | str | qt.Qobj] | None, optional
            Local states. A mapping is keyed by object label; a sequence must
            contain one entry per object in tensor-factor order. Omit to use
            `default` for every object. A `qt.Qobj` entry must be a local ket
            with shape `(dimension, 1)`.
        default : int | str, optional
            State alias used for objects omitted from a mapping. The default is
            local level zero.

        Returns
        -------
        qt.Qobj
            Tensor-product ket in system order.

        Raises
        ------
        TypeError
            If `states` is neither a mapping nor a sequence.
        ValueError
            If a sequence length, object label, local-ket shape, or state alias
            is invalid.
        """
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
                if label not in self._objects_by_label:
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
        """
        Build a local basis or named superposition state for one object.

        Parameters
        ----------
        label : str
            Object label.
        alias : int | str
            Local level index or alias accepted by `create_state`.

        Returns
        -------
        qt.Qobj
            Local ket with the object's Hilbert-space dimension.
        """
        obj = self.get_object(label)
        return self.create_state(obj.dimension, alias)

    @staticmethod
    def create_state(dim: int, alias: int | str) -> qt.Qobj:
        """
        Create a local basis, superposition, or random ket.

        Parameters
        ----------
        dim : int
            Local Hilbert-space dimension.
        alias : int | str
            Integer basis index; `"0"`/`"g"`, `"1"`/`"e"`, or `"2"`/`"f"`;
            qubit superposition `"+"`, `"-"`, `"+i"`/`"i"`, or `"-i"`;
            random qubit-subspace state `"*"`; or random full-space state
            `"**"`.

        Returns
        -------
        qt.Qobj
            Normalized ket of shape `(dim, 1)`.

        Raises
        ------
        ValueError
            If `alias` is unsupported or selects a basis level outside the
            local dimension.

        Notes
        -----
        Random aliases use QuTiP's active random-number generator state.
        """
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
    def _normalize_pair(pair: tuple[str, str]) -> tuple[str, str]:
        """Sort a two-label pair for orientation-independent lookup."""
        label_0, label_1 = pair
        return (label_0, label_1) if label_0 <= label_1 else (label_1, label_0)

    @staticmethod
    def to_tuple_pair(label: str | tuple[str, str]) -> tuple[str, str]:
        """
        Convert a coupling label to an ordered endpoint pair.

        Parameters
        ----------
        label : str | tuple[str, str]
            Hyphen-separated string or an existing pair tuple.

        Returns
        -------
        tuple[str, str]
            Pair with its input orientation preserved.

        Raises
        ------
        ValueError
            If a string does not contain exactly two hyphen-separated labels.
        """
        if isinstance(label, tuple):
            return label
        else:
            pair = tuple(label.split("-"))
            if len(pair) != 2:
                raise ValueError(f"Invalid coupling label: {label}")
            return pair

    def get_coupled_objects(self, label: str) -> list[Object]:
        """
        Return the neighboring objects of an object label.

        Parameters
        ----------
        label : str
            Object label whose graph neighbors are requested.

        Returns
        -------
        list[Object]
            Neighbors in NetworkX adjacency order.

        Raises
        ------
        ValueError
            If `label` does not identify an object in the system.
        """
        if label not in self._objects_by_label:
            raise ValueError(f"Object {label} does not exist.")
        neighbors = list(self.graph.neighbors(label))
        return [self.get_object(neighbor) for neighbor in neighbors]

    def _embed_local_operator(self, label: str, operator: qt.Qobj) -> qt.Qobj:
        """Tensor a local operator with identities on every other object."""
        self.get_object(label)
        return qt.tensor(
            *[
                operator if obj.label == label else qt.qeye(obj.dimension)
                for obj in self.objects
            ]
        )

    def _resolve_gate_target(self, target: GateTarget) -> tuple[str, ...]:
        """Resolve and validate the ordered object labels for one gate."""
        if isinstance(target, str):
            target_labels = (
                (target,)
                if target in self._objects_by_label
                else tuple(target.split("-"))
            )
        elif isinstance(target, tuple):
            target_labels = target
        else:
            raise TypeError("Gate targets must be object labels or label tuples.")
        if not target_labels:
            raise ValueError("A gate must target at least one object.")
        for label in target_labels:
            self.get_object(label)
        if len(set(target_labels)) != len(target_labels):
            raise ValueError("A gate target cannot contain duplicate object labels.")
        return target_labels

    @staticmethod
    def _validate_gate(
        gate: qt.Qobj,
        target_labels: tuple[str, ...],
    ) -> tuple[int, ...]:
        """Validate a unitary gate and return its tensor-factor dimensions."""
        if not isinstance(gate, qt.Qobj):
            raise TypeError("Gates must be names or Qobj instances.")
        if not gate.isoper:
            raise ValueError("A gate must be an operator.")
        if not gate.isunitary:
            raise ValueError("A gate must be unitary.")
        if gate.dims[0] != gate.dims[1]:
            raise ValueError("A gate must have matching input and output dimensions.")
        input_dimensions = cast(list[int], gate.dims[0])
        gate_dimensions = tuple(input_dimensions)
        if len(gate_dimensions) != len(target_labels):
            raise ValueError(
                f"Gate has {len(gate_dimensions)} tensor factors but "
                f"{len(target_labels)} target objects were given."
            )
        return gate_dimensions

    def _embed_gate(
        self,
        target_labels: tuple[str, ...],
        gate: qt.Qobj,
        gate_dimensions: tuple[int, ...],
        levels: Mapping[str, tuple[int, ...]],
    ) -> qt.Qobj:
        """
        Embed a gate on selected levels while preserving the complement.

        The supplied target order maps gate tensor factors to physical object
        factors. All states outside the selected target subspace retain their
        identity-matrix entries.
        """
        system_dimensions = tuple(self.object_dimensions)
        target_indices = tuple(self.get_index(label) for label in target_labels)
        spectator_indices = tuple(
            index for index in range(len(self.objects)) if index not in target_indices
        )
        spectator_dimensions = tuple(
            system_dimensions[index] for index in spectator_indices
        )
        spectator_basis = list(np.ndindex(*spectator_dimensions))
        gate_basis = list(np.ndindex(*gate_dimensions))
        gate_matrix = gate.full()
        system_dimension = int(np.prod(system_dimensions))
        embedded = np.eye(system_dimension, dtype=np.complex128)

        for spectator_state in spectator_basis:
            base_state = [0] * len(self.objects)
            for index, level in zip(
                spectator_indices,
                spectator_state,
                strict=True,
            ):
                base_state[index] = level
            for output_state in gate_basis:
                output = base_state.copy()
                for index, label, local_level in zip(
                    target_indices,
                    target_labels,
                    output_state,
                    strict=True,
                ):
                    output[index] = levels[label][local_level]
                output_flat = np.ravel_multi_index(output, system_dimensions)
                gate_output_flat = np.ravel_multi_index(
                    output_state,
                    gate_dimensions,
                )
                for input_state in gate_basis:
                    input_ = base_state.copy()
                    for index, label, local_level in zip(
                        target_indices,
                        target_labels,
                        input_state,
                        strict=True,
                    ):
                        input_[index] = levels[label][local_level]
                    input_flat = np.ravel_multi_index(input_, system_dimensions)
                    gate_input_flat = np.ravel_multi_index(
                        input_state,
                        gate_dimensions,
                    )
                    embedded[output_flat, input_flat] = gate_matrix[
                        gate_output_flat,
                        gate_input_flat,
                    ]

        return qt.Qobj(
            embedded,
            dims=[self.object_dimensions, self.object_dimensions],
        )

    def _resolve_evaluation_levels(
        self,
        levels: EvaluationSpace,
    ) -> dict[str, tuple[int, ...]]:
        """Resolve an evaluation-space selector into ordered levels per object."""
        if levels == "computational":
            overrides: Mapping[str, Sequence[int]] = {}
            use_full_space = False
        elif levels == "full":
            overrides = {}
            use_full_space = True
        elif isinstance(levels, Mapping):
            overrides = levels
            use_full_space = False
        else:
            raise ValueError(
                "Levels must be 'computational', 'full', or a mapping by object label."
            )
        self._validate_level_labels(overrides)

        resolved = {}
        for obj in self.objects:
            if obj.label in overrides:
                selected = overrides[obj.label]
            elif use_full_space:
                selected = tuple(range(obj.dimension))
            else:
                selected = tuple(range(min(2, obj.dimension)))
            resolved[obj.label] = self._validate_levels(obj.label, selected)
        return resolved

    def _validate_level_labels(
        self,
        levels: Mapping[str, Sequence[int]],
    ) -> None:
        """Require every key in a level mapping to identify a system object."""
        for label in levels:
            self.get_object(label)

    def _validate_levels(
        self,
        label: str,
        levels: Sequence[int],
        *,
        expected_count: int | None = None,
    ) -> tuple[int, ...]:
        """Validate and normalize an ordered local-level selection."""
        selected = tuple(levels)
        if expected_count is not None and len(selected) != expected_count:
            raise ValueError(
                f"Gate on {label} requires exactly {expected_count} levels."
            )
        if not selected:
            raise ValueError(f"At least one level must be selected for {label}.")
        if len(set(selected)) != len(selected):
            raise ValueError(f"Selected levels for {label} must be unique.")
        if any(not isinstance(level, (int, np.integer)) for level in selected):
            raise TypeError(f"Selected levels for {label} must be integers.")
        dimension = self.get_object(label).dimension
        if any(level < 0 or level >= dimension for level in selected):
            raise ValueError(
                f"Selected levels contain an index outside the dimension of {label}."
            )
        return tuple(int(level) for level in selected)

    def get_effective_frequency(
        self,
        label: str,
        *,
        method: Literal["perturbative", "numerical"] = "perturbative",
    ) -> float:
        """
        Return an object's bare frequency plus its center-frequency shift.

        Parameters
        ----------
        label : str
            Object label.
        method : {"perturbative", "numerical"}, optional
            Frequency-shift calculation. The default is `"perturbative"`.

        Returns
        -------
        float
            Effective cyclic frequency in GHz.

        Raises
        ------
        ValueError
            If the object label or method is invalid, or numerical shift
            evaluation is unavailable for a coupled pair.
        """
        obj = self.get_object(label)
        shift = self.get_frequency_shift(label, method=method)
        return obj.frequency + shift

    def get_frequency_shift(
        self,
        label: str,
        *,
        method: Literal["perturbative", "numerical"] = "perturbative",
    ) -> float:
        """
        Return the total center-frequency shift from neighboring objects.

        Parameters
        ----------
        label : str
            Object label.
        method : {"perturbative", "numerical"}, optional
            Shift calculation applied consistently to the Lamb and static-ZZ
            terms. The default is `"perturbative"`.

        Returns
        -------
        float
            Cyclic-frequency shift in GHz, calculated for each neighbor as
            `Lamb shift + static ZZ / 2` and then summed.

        Raises
        ------
        ValueError
            If the method is invalid or numerical shift evaluation is
            unavailable for a coupled pair.
        nx.NetworkXError
            If `label` does not identify an object in the system.
        """
        self._validate_shift_method(method)
        shift = 0.0
        for neighbor in self.graph.neighbors(label):
            shift += self.get_lamb_shift((label, neighbor), method=method)
            shift += 0.5 * self.get_static_zz((label, neighbor), method=method)
        return shift

    def get_lamb_shift(
        self,
        label: str | tuple[str, str],
        *,
        method: Literal["perturbative", "numerical"] = "perturbative",
    ) -> float:
        """
        Return the ground-neighbor transition shift for one object in a pair.

        Parameters
        ----------
        label : str | tuple[str, str]
            Oriented pair. The first object is the transition whose shift is
            returned.
        method : {"perturbative", "numerical"}, optional
            Use `g**2 / (frequency_0 - frequency_1)` or diagonalize the isolated
            compiled pair Hamiltonian. The default is `"perturbative"`.

        Returns
        -------
        float
            Lamb shift of the first object's 0-1 cyclic frequency in GHz, with
            the second object associated with its local ground state.

        Raises
        ------
        ValueError
            If the pair, method, or numerical-model dimensions are invalid.

        Notes
        -----
        Numerical diagonalization uses the compiled local Hamiltonians and the
        adjacent-transition exchange coupling. Dressed states are assigned
        jointly by maximum total overlap with the four bare computational
        states.
        """
        self._validate_shift_method(method)
        pair = self.to_tuple_pair(label)
        if method == "numerical":
            normalized_pair = self._normalize_pair(pair)
            energies = self._get_dressed_computational_energies(normalized_pair)
            if pair[0] == normalized_pair[0]:
                dressed_transition = energies[1] - energies[0]
            else:
                dressed_transition = energies[2] - energies[0]
            compiled = self._get_compiled_object(pair[0])
            bare_energies = np.asarray(compiled.hamiltonian.diag(), dtype=np.float64)
            bare_transition = (bare_energies[1] - bare_energies[0]) / (2 * np.pi)
            return float(dressed_transition - bare_transition)

        coupling = self.get_coupling(pair)
        obj_0 = self.get_object(pair[0])
        obj_1 = self.get_object(pair[1])

        g = coupling.strength
        delta = obj_0.frequency - obj_1.frequency
        return (g**2) / delta

    def get_static_zz(
        self,
        label: str | tuple[str, str],
        *,
        method: Literal["perturbative", "numerical"] = "perturbative",
    ) -> float:
        """
        Return the full static-ZZ conditional frequency splitting in GHz.

        Parameters
        ----------
        label : str | tuple[str, str]
            Coupled object pair. Pair orientation does not change the static-ZZ
            result.
        method : {"perturbative", "numerical"}, optional
            Use the dispersive expression or diagonalize the isolated compiled
            pair Hamiltonian. The default is `"perturbative"`.

        Returns
        -------
        float
            `E_11 - E_10 - E_01 + E_00` after division by Planck's constant,
            expressed as a cyclic frequency in GHz. The perturbative result is
            zero if either object retains fewer than three levels.

        Raises
        ------
        ValueError
            If the pair, method, or numerical-model dimensions are invalid.

        Notes
        -----
        The perturbative method uses the dispersive Schrieffer-Wolff result.
        Numerical diagonalization uses the compiled local Hamiltonians and the
        adjacent-transition exchange coupling. Dressed states are assigned
        jointly by maximizing total overlap with the four bare computational
        states.
        """
        self._validate_shift_method(method)
        pair = self.to_tuple_pair(label)
        if method == "numerical":
            energies = self._get_dressed_computational_energies(
                self._normalize_pair(pair)
            )
            energy_00, energy_10, energy_01, energy_11 = energies
            return float(energy_11 - energy_10 - energy_01 + energy_00)

        obj_0 = self.get_object(pair[0])
        obj_1 = self.get_object(pair[1])
        if obj_0.dimension < 3 or obj_1.dimension < 3:
            return 0.0

        g = self.get_coupling(pair).strength
        delta = obj_0.frequency - obj_1.frequency
        alpha_0 = obj_0.anharmonicity
        alpha_1 = obj_1.anharmonicity
        return 2 * g**2 * (alpha_0 + alpha_1) / ((delta + alpha_0) * (delta - alpha_1))

    @staticmethod
    def _validate_shift_method(method: str) -> None:
        """Require a supported perturbative or numerical shift method."""
        if method not in ("perturbative", "numerical"):
            raise ValueError(f"Unsupported frequency-shift method: {method}")

    def _get_dressed_computational_energies(
        self,
        pair: tuple[str, str],
    ) -> np.ndarray:
        """
        Return cached dressed pair energies in bare-state order.

        The output order is `|00>`, `|10>`, `|01>`, and `|11>`, and each energy
        divided by Planck's constant is expressed in GHz.
        """
        if pair not in self._dressed_computational_energies_by_pair:
            self._dressed_computational_energies_by_pair[pair] = (
                self._calculate_dressed_computational_energies(pair)
            )
        return self._dressed_computational_energies_by_pair[pair]

    def _calculate_dressed_computational_energies(
        self,
        pair: tuple[str, str],
    ) -> np.ndarray:
        """
        Calculate dressed pair energies assigned to bare computational states.

        The isolated pair uses compiled local Hamiltonians and the Hermitian
        adjacent-transition exchange coupling. A linear assignment maximizes
        total bare-to-dressed overlap so the four returned eigenstates are
        distinct. Energies divided by Planck's constant are returned in GHz in
        `|00>`, `|10>`, `|01>`, `|11>` order.
        """
        coupling = self.get_coupling(pair)
        compiled_0 = self._get_compiled_object(pair[0])
        compiled_1 = self._get_compiled_object(pair[1])
        dimension_0 = compiled_0.source.dimension
        dimension_1 = compiled_1.source.dimension
        if dimension_0 < 2 or dimension_1 < 2:
            raise ValueError(
                "Numerical shifts require object dimensions of at least 2."
            )

        pair_hamiltonian = qt.tensor(
            compiled_0.hamiltonian,
            qt.qeye(dimension_1),
        ) + qt.tensor(
            qt.qeye(dimension_0),
            compiled_1.hamiltonian,
        )
        coupling_term = qt.tensor(
            compiled_0.lowering_operator.dag(),
            compiled_1.lowering_operator,
        )
        pair_hamiltonian += (
            2 * np.pi * coupling.strength * (coupling_term + coupling_term.dag())
        )

        dressed_energies, dressed_states = pair_hamiltonian.eigenstates()
        dressed_vectors = np.column_stack(
            [state.full().ravel() for state in dressed_states]
        )
        # QuTiP flattens |n_0, n_1> to n_0 * dimension_1 + n_1.
        bare_indices = np.array(
            [
                0,
                dimension_1,
                1,
                dimension_1 + 1,
            ]
        )
        overlaps = np.abs(dressed_vectors[bare_indices, :]) ** 2
        # Assign four distinct dressed states to the computational basis by
        # maximizing their total bare-state overlap.
        bare_rows, dressed_columns = linear_sum_assignment(-overlaps)
        computational_energies = np.empty(4, dtype=np.float64)
        computational_energies[bare_rows] = dressed_energies[dressed_columns] / (
            2 * np.pi
        )
        return computational_energies

    def get_rotation_matrix(
        self,
        angles: dict[str, float],
    ) -> qt.Qobj:
        """
        Return a tensor product of local level-phase rotations.

        Parameters
        ----------
        angles : dict[str, float]
            Rotation angle in radians for each object label. Objects absent
            from the mapping receive the identity; unknown mapping keys are
            ignored.

        Returns
        -------
        qt.Qobj
            Product of `exp(1j * angle * N)` in tensor-factor order, where `N`
            is each object's retained-basis level-index operator.
        """
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
