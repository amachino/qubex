from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from ..backend.lattice_graph import LatticeGraph


@dataclass(frozen=True)
class InspectionParams:
    max_frequency: float = 9.5
    min_frequency: float = 6.5
    max_detuning: float = 1.5
    min_t1: float = 3e3
    min_t2: float = 3e3
    adiabatic_limit: float = np.sqrt(0.05 / 2.474)
    cr_control_limit: float = 0.75
    cnot_time: float = 500
    default_t1: float = 10e3
    default_t2: float = 10e3
    default_coupling: float = 8e-3


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
        value = self.graph.get_property(target, property_type)

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
        def get_composite_coupling(i, j, k):
            f_i = self.get_ge_frequency(i)
            f_j = self.get_ge_frequency(j)
            f_k = self.get_ge_frequency(k)
            g_ij = self.get_nn_coupling((i, j))
            g_jk = self.get_nn_coupling((j, k))
            return 0.5 * (g_ij * g_jk) * (1 / (f_i - f_j) + 1 / (f_j - f_k))

        i, k = target
        if (i, k) not in self.graph.next_nearest_pairs:
            raise ValueError("Distance between qubits is not 2.")
        common_neighbors = self.graph.common_neighbors[i, k]
        if len(common_neighbors) == 1:
            j = common_neighbors[0]
            g = get_composite_coupling(i, j, k)
        elif len(common_neighbors) == 2:
            j1, j2 = common_neighbors
            g1 = get_composite_coupling(i, j1, k)
            g2 = get_composite_coupling(i, j2, k)
            g = g1 + g2
        else:
            raise ValueError("Invalid number of common neighbors.")
        return g

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

    def create_node_hovertext(
        self,
        label: str,
        invalid_types: list[str] | None = None,
    ) -> str:
        if invalid_types is None:
            invalid_types = []

        node = self.graph.get_qubit_node_by_label(label)
        id = node["id"]
        p = node["properties"]
        f_ge = p.get("frequency")
        alpha = p.get("anharmonicity")
        t1 = p.get("t1")
        t2 = p.get("t2_echo")
        f_ef = f_ge + alpha if f_ge is not None and alpha is not None else None

        if f_ge is not None:
            f_ge = f"{f_ge * 1e3:.0f} MHz"
        else:
            f_ge = f"({self.get_ge_frequency(id) * 1e3:.0f}) MHz"
        if f_ef is not None:
            f_ef = f"{f_ef * 1e3:.0f} MHz"
        else:
            f_ef = f"({self.get_ef_frequency(id) * 1e3:.0f}) MHz"
        if alpha is not None:
            alpha = f"{alpha * 1e3:.0f} MHz"
        else:
            alpha = f"({self.get_anharmonicity(id) * 1e3:.0f}) MHz"
        if t1 is not None:
            t1 = f"{t1 * 1e-3:.1f} µs"
        else:
            t1 = f"({self.get_t1(id) * 1e-3:.1f}) µs"
        if t2 is not None:
            t2 = f"{t2 * 1e-3:.1f} µs"
        else:
            t2 = f"({self.get_t2(id) * 1e-3:.1f}) µs"

        if len(invalid_types) == 0:
            hovertext = f"{label}<br>"
        else:
            hovertext = f"{label}: {{{','.join(invalid_types)}}}<br>"

        hovertext += "<br>".join(
            [
                f"f_ge = {f_ge}",
                f"f_ef = {f_ef}",
                f"α    = {alpha}",
                f"T1   = {t1}",
                f"T2   = {t2}",
            ]
        )
        return hovertext

    def create_edge_hovertext(
        self,
        label: str,
        invalid_types: list[str] | None = None,
    ) -> str:
        if invalid_types is None:
            invalid_types = []

        edge = self.graph.get_qubit_edge_by_label(label)
        i, j = edge["id"]
        f_ge_i = self.graph.qubit_nodes[i]["properties"].get("frequency")
        f_ge_j = self.graph.qubit_nodes[j]["properties"].get("frequency")
        a_i = self.graph.qubit_nodes[i]["properties"].get("anharmonicity")

        Delta_ge_ge = None
        if f_ge_i is not None and f_ge_j is not None:
            Delta_ge_ge = f_ge_i - f_ge_j
            Delta_ge_ge = f"{Delta_ge_ge * 1e3:.0f} MHz"
        else:
            Delta_ge_ge = f"({self.get_ge_ge_detuning((i, j)) * 1e3:.0f}) MHz"

        Delta_ef_ge = None
        if f_ge_i is not None and a_i is not None and f_ge_j is not None:
            Delta_ef_ge = f_ge_i + a_i - f_ge_j
            Delta_ef_ge = f"{Delta_ef_ge * 1e3:.0f} MHz"
        else:
            Delta_ef_ge = f"({self.get_ef_ge_detuning((i, j)) * 1e3:.0f}) MHz"

        g = edge["properties"].get("coupling")
        if g is not None:
            g = f"{g * 1e3:.1f} MHz"
        else:
            g = f"({self.get_nn_coupling((i, j)) * 1e3:.1f}) MHz"

        if len(invalid_types) == 0:
            hovertext = f"{label}<br>"
        else:
            hovertext = f"{label}: {{{','.join(invalid_types)}}}<br>"

        hovertext += "<br>".join(
            [
                f"Δ_ge = {Delta_ge_ge}",
                f"Δ_ef = {Delta_ef_ge}",
                f"g    = {g}",
            ]
        )
        return hovertext

    def print(self):
        print(f"[{self.name}]")
        print(f"{self.description}")
        print()
        print(f"{len(self.invalid_nodes)} invalid nodes: ")
        if self.invalid_nodes:
            for label, messages in self.invalid_nodes.items():
                print(f"  {label}:")
                for message in messages:
                    print(f"    - {message}")
        print(f"{len(self.invalid_edges)} invalid edges: ")
        if self.invalid_edges:
            for label, messages in self.invalid_edges.items():
                print(f"  {label}:")
                for message in messages:
                    print(f"    - {message}")

    def draw(
        self,
        save_image: bool = False,
        images_dir: str = "./images",
    ):
        node_values = {label: 1 for label in self.invalid_nodes.keys()}
        edge_values = {label: 1 for label in self.invalid_edges.keys()}
        node_hovertexts = {
            data["label"]: self.create_node_hovertext(data["label"])
            for data in self.graph.qubit_nodes.values()
        }
        edge_hovertexts = {
            data["label"]: self.create_edge_hovertext(data["label"])
            for data in self.graph.qubit_edges.values()
        }
        self.graph.plot_graph_data(
            title=f"{self.name}: {self.description}",
            node_hovertexts=node_hovertexts,
            node_overlay=True,
            node_overlay_values=node_values,
            node_overlay_color="#ef553b",
            node_overlay_linecolor="black",
            node_overlay_textcolor="white",
            node_overlay_hovertexts=node_hovertexts,
            edge_color="ghostwhite",
            edge_hovertexts=edge_hovertexts,
            edge_overlay=True,
            edge_overlay_values=edge_values,
            edge_overlay_color="#ef553b",
            edge_overlay_hovertexts=edge_hovertexts,
            save_image=save_image,
            image_name=f"{self.name.replace(' ', '_').lower()}",
            images_dir=images_dir,
        )
