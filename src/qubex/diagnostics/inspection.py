from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from ..backend.lattice_graph import LatticeGraph


@dataclass(frozen=True)
class InspectionParams:
    max_frequency: float = 9.5
    min_frequency: float = 6.5
    max_detuning: float = 1.5
    min_t1: float = 3e3
    min_t2: float = 3e3
    adiabatic_limit: float = 0.2
    cr_control_limit: float = 0.75
    cnot_time: float = 500
    default_t1: float = 10e3
    default_t2: float = 10e3
    default_coupling: float = 8e-3
    default_nnn_coupling: float = 8e-3 * (8e-3 / 0.8)


class Inspection(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        self.name = name
        self.description = description
        self.graph = graph
        self.params = InspectionParams()
        if params is not None:
            self.params = replace(self.params, **params)
        self._invalid_nodes = defaultdict(list[str])
        self._invalid_edges = defaultdict(list[str])

    @abstractmethod
    def execute(self):
        raise NotImplementedError

    @property
    def invalid_nodes(self) -> dict[str, list[str]]:
        return dict(sorted(self._invalid_nodes.items()))

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        return dict(sorted(self._invalid_edges.items()))

    @cached_property
    def nearest_neighbors(self) -> dict[int, list[int]]:
        nn = {
            i: sorted(list(self.graph.qubit_graph.neighbors(i)))
            for i in self.graph.qubit_nodes.keys()
        }
        return dict(sorted(nn.items()))

    @cached_property
    def next_nearest_neighbors(self) -> dict[int, list[int]]:
        nn = self.nearest_neighbors
        nnm = {}
        for i, neighbors in nn.items():
            one_hop = set(neighbors)
            two_hop = set()
            for j in neighbors:
                two_hop.update(nn[j])
            nnm[i] = sorted(list(two_hop - one_hop - {i}))
        return dict(sorted(nnm.items()))

    def get_label(
        self,
        target: int | tuple[int, int],
    ) -> str:
        if isinstance(target, int):
            return self.graph.qubit_nodes[target]["label"]
        elif isinstance(target, tuple):
            edge = self.graph.qubit_edges.get(target)
            if edge is not None:
                return edge["label"]
            else:
                label_0 = self.graph.qubit_nodes[target[0]]["label"]
                label_1 = self.graph.qubit_nodes[target[1]]["label"]
                return f"{label_0}-{label_1}"
        else:
            raise ValueError("Invalid target type.")

    def get_property(
        self,
        target: int | tuple[int, int],
        property_type: str,
    ) -> float | None:
        if isinstance(target, int):
            value = self.graph.qubit_nodes[target]["properties"].get(property_type)
        elif isinstance(target, tuple):
            value = self.graph.qubit_edges[target]["properties"].get(property_type)
        else:
            raise ValueError("Invalid target type.")

        # treat nan as None
        if value is None or np.isnan(value):
            return None
        else:
            return value

    def get_ge_frequency(
        self,
        target: int,
    ) -> float:
        return self.get_property(target, "frequency") or np.nan

    def get_anharmonicity(
        self,
        target: int,
    ) -> float:
        anharmonicity = self.get_property(target, "anharmonicity")
        if anharmonicity is None:
            f = self.get_ge_frequency(target)
            return (-1 / 19) * f  # E_J/E_C = 50
        else:
            return anharmonicity

    def get_ef_frequency(
        self,
        target: int,
    ) -> float:
        return self.get_ge_frequency(target) + self.get_anharmonicity(target)

    def get_t1(
        self,
        target: int,
    ) -> float:
        return self.get_property(target, "t1") or self.params.default_t1

    def get_t2(
        self,
        target: int,
    ) -> float:
        return self.get_property(target, "t2_echo") or self.params.default_t2

    def get_ge_ge_detuning(
        self,
        target: tuple[int, int],
    ) -> float:
        return self.get_ge_frequency(target[0]) - self.get_ge_frequency(target[1])

    def get_ef_ge_detuning(
        self,
        target: tuple[int, int],
    ) -> float:
        return self.get_ef_frequency(target[0]) - self.get_ge_frequency(target[1])

    def get_stark_shift(
        self,
        target: tuple[int, int],
    ) -> float:
        c, t = target
        a_c = self.get_anharmonicity(c)
        D_ct = self.get_ge_ge_detuning((c, t))
        O_d_ct = self.get_cr_drive_frequency((c, t))
        return O_d_ct**2 * a_c / (2 * D_ct * (D_ct + a_c))

    def get_nn_coupling(
        self,
        target: tuple[int, int],
    ) -> float:
        return self.get_property(target, "coupling") or self.params.default_coupling

    def get_nnn_coupling(
        self,
        target: tuple[int, int],
    ) -> float:
        return (
            self.get_property(target, "nnn_coupling")
            or self.params.default_nnn_coupling
        )

    def get_cr_rabi_frequency(
        self,
        target: tuple[int, int],
    ) -> float:
        cnot = self.params.cnot_time
        return 1 / (4 * cnot)

    def get_cr_drive_frequency(
        self,
        target: tuple[int, int],
    ) -> float:
        c, t = target
        D_ct = self.get_ge_ge_detuning((c, t))
        a_c = self.get_anharmonicity(c)
        g_ct = self.get_nn_coupling((c, t))
        O_r_ct = self.get_cr_rabi_frequency((c, t))
        return -O_r_ct * D_ct * (D_ct + a_c) / (g_ct * a_c)

    def add_invalid_nodes(
        self,
        nodes: list[str],
        message: str,
    ):
        nodes = nodes or []
        for node in nodes:
            self._invalid_nodes[node].append(message)

    def add_invalid_edges(
        self,
        edges: list[str],
        message: str,
    ):
        edges = edges or []
        for edge in edges:
            self._invalid_edges[edge].append(message)

    def print(self):
        print(f"[{self.name}]")
        print(f"{self.description}")
        print()
        print(f"{len(self._invalid_nodes)} invalid nodes: ")
        if self._invalid_nodes:
            for label, messages in self.invalid_nodes.items():
                print(f"  {label}:")
                for message in messages:
                    print(f"    - {message}")
        print(f"{len(self._invalid_edges)} invalid edges: ")
        if self._invalid_edges:
            for label, messages in self.invalid_edges.items():
                print(f"  {label}:")
                for message in messages:
                    print(f"    - {message}")

    def draw(self):
        node_hovertexts = {
            label: f"{'<br>'.join(messages)}"
            for label, messages in self.invalid_nodes.items()
        }
        node_values = {label: 1 for label in self.invalid_nodes.keys()}
        edge_values = {label: 1 for label in self.invalid_edges.keys()}
        self.graph.plot_graph_data(
            title=f"{self.name}: {self.description}",
            node_overlay=True,
            node_overlay_values=node_values,
            node_overlay_color="red",
            node_overlay_linecolor="black",
            node_overlay_textcolor="white",
            node_overlay_hovertexts=node_hovertexts,
            edge_color="ghostwhite",
            edge_overlay=True,
            edge_overlay_values=edge_values,
            edge_overlay_color="red",
            edge_overlay_hovertexts=self.invalid_edges,
        )
