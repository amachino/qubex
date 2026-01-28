"""Base inspection classes and parameter definitions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import NoReturn

import numpy as np

from qubex.backend.lattice_graph import LatticeGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InspectionParams:
    """Parameter defaults for inspection checks."""

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
    """Base class for inspection checks on lattice graphs."""

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
    def execute(self) -> NoReturn:
        """Execute the inspection and populate invalid nodes/edges."""
        raise NotImplementedError

    @property
    def invalid_nodes(self) -> dict[str, list[str]]:
        """Return invalid nodes keyed by label."""
        return dict(sorted(self._invalid_nodes.items()))

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        """Return invalid edges keyed by label."""
        return dict(sorted(self._invalid_edges.items()))

    def get_label(
        self,
        target: int | tuple[int, int],
    ) -> str:
        """Return a label for the given node or edge target."""
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
            raise TypeError("Invalid target type.")

    def get_property(
        self,
        target: int | tuple[int, int],
        property_type: str,
    ) -> float | None:
        """Return a numeric property for the target if available."""
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
        """Return the GE frequency for a node in GHz."""
        return self.get_property(target, "frequency") or np.nan

    def get_anharmonicity(
        self,
        target: int,
    ) -> float:
        """Return the anharmonicity for a node in GHz."""
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
        """Return the EF frequency for a node in GHz."""
        return self.get_ge_frequency(target) + self.get_anharmonicity(target)

    def get_t1(
        self,
        target: int,
    ) -> float:
        """Return the T1 value for a node in ns."""
        return self.get_property(target, "t1") or self.params.default_t1

    def get_t2(
        self,
        target: int,
    ) -> float:
        """Return the T2 echo value for a node in ns."""
        return self.get_property(target, "t2_echo") or self.params.default_t2

    def get_ge_ge_detuning(
        self,
        target: tuple[int, int],
    ) -> float:
        """Return GE-GE detuning for a pair in GHz."""
        return self.get_ge_frequency(target[0]) - self.get_ge_frequency(target[1])

    def get_ef_ge_detuning(
        self,
        target: tuple[int, int],
    ) -> float:
        """Return EF-GE detuning for a pair in GHz."""
        return self.get_ef_frequency(target[0]) - self.get_ge_frequency(target[1])

    def get_stark_shift(
        self,
        target: tuple[int, int],
    ) -> float:
        """Return the Stark shift for a control-target pair."""
        c, t = target
        a_c = self.get_anharmonicity(c)
        D_ct = self.get_ge_ge_detuning((c, t))
        O_d_ct = self.get_cr_drive_frequency((c, t))
        return O_d_ct**2 * a_c / (2 * D_ct * (D_ct + a_c))

    def get_nn_coupling(
        self,
        target: tuple[int, int],
    ) -> float:
        """Return nearest-neighbor coupling strength in GHz."""
        return self.get_property(target, "coupling") or self.params.default_coupling

    def get_nnn_coupling(
        self,
        target: tuple[int, int],
    ) -> float:
        """Return next-nearest-neighbor coupling strength in GHz."""

        def get_composite_coupling(i: int, j: int, k: int) -> float:
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
        """Return the CR Rabi frequency estimate in GHz."""
        cnot = self.params.cnot_time
        return 1 / (4 * cnot)

    def get_cr_drive_frequency(
        self,
        target: tuple[int, int],
    ) -> float:
        """Return the CR drive frequency estimate in GHz."""
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
    ) -> None:
        """Record a validation message for invalid nodes."""
        nodes = nodes or []
        for node in nodes:
            self._invalid_nodes[node].append(message)

    def add_invalid_edges(
        self,
        edges: list[str],
        message: str,
    ) -> None:
        """Record a validation message for invalid edges."""
        edges = edges or []
        for edge in edges:
            self._invalid_edges[edge].append(message)

    def create_node_hovertext(
        self,
        label: str,
        invalid_types: list[str] | None = None,
    ) -> str:
        """Create hover text for a node label."""
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
        """Create hover text for an edge label."""
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

    def log_report(self) -> None:
        """Log a formatted inspection report."""
        logger.info(f"[{self.name}]")
        logger.info(f"{self.description}")
        logger.info("")
        logger.info(f"{len(self.invalid_nodes)} invalid nodes: ")
        if self.invalid_nodes:
            for label, messages in self.invalid_nodes.items():
                logger.info(f"  {label}:")
                for message in messages:
                    logger.info(f"    - {message}")
        logger.info(f"{len(self.invalid_edges)} invalid edges: ")
        if self.invalid_edges:
            for label, messages in self.invalid_edges.items():
                logger.info(f"  {label}:")
                for message in messages:
                    logger.info(f"    - {message}")

    def draw(
        self,
        save_image: bool = False,
        images_dir: str = "./images",
    ) -> None:
        """Render the inspection results on the lattice graph."""
        node_values = dict.fromkeys(self.invalid_nodes, 1)
        edge_values = dict.fromkeys(self.invalid_edges, 1)
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
