from __future__ import annotations

from collections import defaultdict
from typing import Literal

from ..backend.config_loader import ConfigLoader
from ..backend.lattice_graph import LatticeGraph
from . import inspection_library
from .inspection import Inspection

InspectionType = Literal[
    "Type0A",
    "Type0B",
    "Type1A",
    "Type1B",
    "Type1C",
    "Type2A",
    "Type2B",
    "Type3A",
    "Type3B",
    "Type7",
    "Type8",
    "Type9",
]


class ChipInspector:
    def __init__(
        self,
        chip_id: str,
        props_dir: str | None = None,
    ):
        self._init_graph(chip_id, props_dir)

    def _init_graph(
        self,
        chip_id: str,
        props_dir: str | None = None,
    ):
        if props_dir is None:
            config_loader = ConfigLoader()
        else:
            config_loader = ConfigLoader(params_dir=props_dir)

        experiment_system = config_loader.get_experiment_system(chip_id)
        self.graph = experiment_system.quantum_system._graph
        props = config_loader._props_dict[chip_id]

        for node in self.graph.qubit_nodes.values():
            label = node["label"]
            node["properties"] = {
                "frequency": props["qubit_frequency"].get(label),
                "anharmonicity": props["anharmonicity"].get(label),
                "t1": props["t1"].get(label),
                "t2_echo": props["t2_echo"].get(label),
            }

        for edge in self.graph.qubit_edges.values():
            label = edge["label"]
            edge["properties"] = {
                "coupling": props["qubit_qubit_coupling_strength"].get(label),
            }

    def execute(
        self,
        types: InspectionType | list[InspectionType] | None = None,
        params: dict | None = None,
    ) -> InspectionSummary:
        if types is None:
            types = list(InspectionType.__args__)  # type: ignore
        elif isinstance(types, str):
            types = [types]

        inspections = {}
        for type in types:
            inspection: Inspection = getattr(inspection_library, type)(
                graph=self.graph,
                params=params,
            )
            inspection.execute()
            inspections[type] = inspection

        return InspectionSummary(
            graph=self.graph,
            inspections=inspections,
        )


class InspectionSummary:
    def __init__(
        self,
        graph: LatticeGraph,
        inspections: dict[InspectionType, Inspection],
    ):
        self.graph = graph
        self.inspections = inspections

    @property
    def invalid_nodes(self) -> dict[str, list[str]]:
        invalid_nodes = defaultdict(list)
        for result in self.inspections.values():
            for label, messages in result.invalid_nodes.items():
                invalid_nodes[label].extend(
                    [f"[{result.name}] {message}" for message in messages]
                )
        return dict(sorted(invalid_nodes.items()))

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        invalid_edges = defaultdict(list)
        for result in self.inspections.values():
            for label, messages in result.invalid_edges.items():
                invalid_edges[label].extend(
                    [f"[{result.name}] {message}" for message in messages]
                )
        return dict(sorted(invalid_edges.items()))

    def print(self):
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

    def draw(self):
        all_nodes = {node["label"] for node in self.graph.qubit_nodes.values()}
        invalid_nodes = set(self.invalid_nodes)
        valid_nodes = all_nodes - invalid_nodes

        all_edges = {edge["label"] for edge in self.graph.qubit_edges.values()}
        invalid_edges = set(self.invalid_edges)
        for edge in self.graph.qubit_edges.values():
            label = edge["label"]
            node0, node1 = edge["id"]
            qubit0 = self.graph.qubit_nodes[node0]["label"]
            qubit1 = self.graph.qubit_nodes[node1]["label"]
            if qubit0 in invalid_nodes or qubit1 in invalid_nodes:
                invalid_edges.add(label)
        valid_edges = all_edges - invalid_edges

        invalid_edge_values = {label: 1 for label in invalid_edges}
        valid_edge_values = {label: 1 for label in valid_edges}

        valid_node_hovertexts = {}
        for data in self.graph.qubit_nodes.values():
            label = data["label"]
            p = data["properties"]
            f_ge = p.get("frequency")
            alpha = p.get("anharmonicity")
            t1 = p.get("t1")
            t2 = p.get("t2_echo")
            f_ef = f_ge + alpha if f_ge is not None and alpha is not None else None

            if f_ge is not None:
                f_ge = f"{f_ge * 1e3:.0f} MHz"
            if f_ef is not None:
                f_ef = f"{f_ef * 1e3:.0f} MHz"
            if alpha is not None:
                alpha = f"{alpha * 1e3:.0f} MHz"
            if t1 is not None:
                t1 = f"{t1 * 1e-3:.0f} µs"
            if t2 is not None:
                t2 = f"{t2 * 1e-3:.0f} µs"

            valid_node_hovertexts[label] = "<br>".join(
                [
                    f"f_ge: {f_ge}",
                    f"f_ef: {f_ef}",
                    f"α: {alpha}",
                    f"T1: {t1}",
                    f"T2: {t2}",
                ]
            )

        invalid_node_hovertexts = {
            label: f"{'<br>'.join(messages)}"
            for label, messages in self.invalid_nodes.items()
        }

        invalid_edge_hovertexts = {
            label: f"{'<br>'.join(messages)}"
            for label, messages in self.invalid_edges.items()
        }

        self.graph.plot_graph_data(
            title="Valid nodes and edges",
            node_labels=valid_nodes,
            node_color="blue",
            node_linecolor="black",
            node_textcolor="white",
            node_hovertexts=valid_node_hovertexts,
            edge_values=valid_edge_values,
            edge_color="blue",
        )

        self.graph.plot_graph_data(
            title="Invalid nodes and edges",
            node_labels=invalid_nodes,
            node_color="red",
            node_linecolor="black",
            node_textcolor="white",
            node_hovertexts=invalid_node_hovertexts,
            edge_values=invalid_edge_values,
            edge_color="red",
            edge_hovertexts=invalid_edge_hovertexts,
        )

        for inspection in self.inspections.values():
            inspection.draw()
