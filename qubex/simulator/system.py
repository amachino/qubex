from dataclasses import asdict, dataclass
from typing import Final, Literal

import networkx as nx  # type: ignore
import qutip as qt  # type: ignore


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
        couplings: list[Coupling],
    ):
        self._validate(transmons, couplings)
        self.transmons: Final = transmons
        self.couplings: Final = couplings
        self.graph: Final = nx.Graph()
        self.hamiltonian = qt.Qobj()
        self._init_system()

    def ground_state(self) -> qt.Qobj:
        return self.pauli_state("0")

    def excited_state(self) -> qt.Qobj:
        return self.pauli_state("1")

    def superposition_state(self) -> qt.Qobj:
        return self.pauli_state("-i")

    def random_state(self) -> qt.Qobj:
        return qt.tensor(
            [(qt.rand_ket_haar(transmon.dimension)) for transmon in self.transmons]
        )

    def pauli_state(
        self,
        label: Literal["0", "1", "+", "-", "+i", "-i"],
    ) -> qt.Qobj:
        states = []
        for transmon in self.transmons:
            dim = transmon.dimension
            if label == "0":
                state = qt.basis(dim, 0)
            elif label == "1":
                state = qt.basis(dim, 1)
            elif label == "+":
                state = qt.basis(dim, 0) + qt.basis(dim, 1)
            elif label == "-":
                state = qt.basis(dim, 0) - qt.basis(dim, 1)
            elif label == "+i":
                state = qt.basis(dim, 0) + 1j * qt.basis(dim, 1)
            elif label == "-i":
                state = qt.basis(dim, 0) - 1j * qt.basis(dim, 1)
            else:
                raise ValueError(f"Invalid Pauli state {label}.")
            states.append(state.unit())
        return qt.tensor(states)

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
        omega = transmon.frequency
        alpha = transmon.anharmonicity
        b = self.lowering_operator(transmon.label)
        bd = b.dag()
        return omega * bd * b + 0.5 * alpha * (bd * bd * b * b)

    def coupling_hamiltonian(self, coupling: Coupling) -> qt.Qobj:
        g = coupling.strength
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
                qt.destroy(transmon.dimension)
                if transmon.label == label
                else qt.qeye(transmon.dimension)
                for transmon in self.transmons
            ]
        )

    def draw(self, **kwargs):
        nx.draw(self.graph, **kwargs)

    def _validate(self, transmons: list[Transmon], couplings: list[Coupling]):
        transmon_labels = [transmon.label for transmon in transmons]
        if len(transmon_labels) != len(set(transmon_labels)):
            raise ValueError("Transmons must have unique labels.")
        for coupling in couplings:
            if any(label not in transmon_labels for label in coupling.pair):
                raise ValueError("Couplings must be between existing transmons.")

    def _init_system(self):
        for transmon in self.transmons:
            self.graph.add_node(transmon.label, **asdict(transmon))
            self.hamiltonian += self.transmon_hamiltonian(transmon)
        for coupling in self.couplings:
            self.graph.add_edge(*coupling.pair, **asdict(coupling))
            self.hamiltonian += self.coupling_hamiltonian(coupling)
