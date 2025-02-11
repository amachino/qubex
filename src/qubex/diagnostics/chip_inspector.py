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

    def create_node_hovertext(
        self,
        label: str,
        invalid_types: list[str] | None = None,
    ) -> str:
        if invalid_types is None:
            invalid_types = []
        node = self.graph.get_qubit_node_by_label(label)

        p = node["properties"]
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

        Delta_ef_ge = None
        if f_ge_i is not None and a_i is not None and f_ge_j is not None:
            Delta_ef_ge = f_ge_i + a_i - f_ge_j
            Delta_ef_ge = f"{Delta_ef_ge * 1e3:.0f} MHz"

        g = edge["properties"].get("coupling")
        if g is not None:
            g = f"{g * 1e3:.0f} MHz"

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

    def draw(
        self,
        save_image: bool = False,
        images_dir: str = "./images",
    ):
        all_nodes = {node["label"] for node in self.graph.qubit_nodes.values()}
        invalid_nodes = set(self.invalid_nodes)
        valid_nodes = all_nodes - invalid_nodes

        invalid_node_values = {label: 1 for label in invalid_nodes}
        valid_node_values = {label: 1 for label in valid_nodes}

        all_edges = {edge["label"] for edge in self.graph.qubit_edges.values()}
        invalid_edges = set(self.invalid_edges)
        for edge in self.graph.qubit_edges.values():
            label = edge["label"]
            node0, node1 = edge["id"]
            qubit0 = self.graph.qubit_nodes[node0]["label"]
            qubit1 = self.graph.qubit_nodes[node1]["label"]
            if qubit0 in invalid_nodes or qubit1 in invalid_nodes:
                all_edges.remove(label)
        valid_edges = all_edges - invalid_edges

        invalid_edge_values = {label: 1 for label in invalid_edges}
        valid_edge_values = {label: 1 for label in valid_edges}

        invalid_types: dict[str, list[str]] = defaultdict(list)
        for type, inspection in self.inspections.items():
            for label in inspection.invalid_nodes:
                invalid_types[label].append(type.replace("Type", ""))
            for label in inspection.invalid_edges:
                invalid_types[label].append(type.replace("Type", ""))

        node_hovertexts = {}
        for data in self.graph.qubit_nodes.values():
            label = data["label"]
            hovertext = self.create_node_hovertext(label, invalid_types.get(label))
            node_hovertexts[label] = hovertext
        edge_hovertexts = {}
        for data in self.graph.qubit_edges.values():
            label = data["label"]
            hovertext = self.create_edge_hovertext(label, invalid_types.get(label))
            edge_hovertexts[label] = hovertext

        self.graph.plot_graph_data(
            title="Valid nodes and edges",
            node_hovertexts=node_hovertexts,
            node_overlay=True,
            node_overlay_values=valid_node_values,
            node_overlay_color="#636EFA",
            node_overlay_linecolor="black",
            node_overlay_textcolor="white",
            node_overlay_hovertexts=node_hovertexts,
            edge_color="ghostwhite",
            edge_hovertexts=edge_hovertexts,
            edge_overlay=True,
            edge_overlay_values=valid_edge_values,
            edge_overlay_color="#636EFA",
            edge_overlay_hovertexts=edge_hovertexts,
        )

        self.graph.plot_graph_data(
            title="Invalid nodes and edges",
            node_hovertexts=node_hovertexts,
            node_overlay=True,
            node_overlay_values=invalid_node_values,
            node_overlay_color="#ef553b",
            node_overlay_linecolor="black",
            node_overlay_textcolor="white",
            node_overlay_hovertexts=node_hovertexts,
            edge_color="ghostwhite",
            edge_hovertexts=edge_hovertexts,
            edge_overlay=True,
            edge_overlay_values=invalid_edge_values,
            edge_overlay_color="#ef553b",
            edge_overlay_hovertexts=edge_hovertexts,
            save_image=save_image,
            image_name="invalid_nodes_and_edges",
            images_dir=images_dir,
        )

        for inspection in self.inspections.values():
            inspection.draw(
                save_image=save_image,
                images_dir=images_dir,
            )
