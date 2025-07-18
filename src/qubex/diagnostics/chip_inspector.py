from __future__ import annotations

from collections import defaultdict
from pathlib import Path
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
        config_dir: Path | str | None = None,
        props_dir: Path | str | None = None,
    ):
        self._init_graph(
            chip_id=chip_id,
            config_dir=config_dir,
            props_dir=props_dir,
        )

    def _init_graph(
        self,
        chip_id: str,
        config_dir: Path | str | None = None,
        props_dir: Path | str | None = None,
    ):
        config_loader = ConfigLoader(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=props_dir,
        )
        experiment_system = config_loader.get_experiment_system(chip_id)
        self.graph = experiment_system.quantum_system._graph
        props = config_loader._props_dict[chip_id]

        frequency_dict = props.get("qubit_frequency", {})
        anharmonicity_dict = props.get("anharmonicity", {})
        t1_dict = props.get("t1", {})
        t2_echo_dict = props.get("t2_echo", {})
        coupling_dict = props.get("qubit_qubit_coupling_strength", {})

        for node in self.graph.qubit_nodes.values():
            label = node["label"]
            node["properties"] = {
                "frequency": frequency_dict.get(label),
                "anharmonicity": anharmonicity_dict.get(label),
                "t1": t1_dict.get(label),
                "t2_echo": t2_echo_dict.get(label),
            }

        for edge in self.graph.qubit_edges.values():
            label = edge["label"]
            edge["properties"] = {
                "coupling": coupling_dict.get(label),
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
            try:
                inspection.execute()
            except Exception as e:
                print(f"Error in {inspection.name}: {e}")
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

    def draw(
        self,
        draw_individual_results: bool = True,
        save_image: bool = False,
        images_dir: str = "./images",
    ):
        if len(self.inspections) == 0:
            raise ValueError("No inspections have been executed.")

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

        inspection = list(self.inspections.values())[0]
        node_hovertexts = {}
        for data in self.graph.qubit_nodes.values():
            label = data["label"]
            hovertext = inspection.create_node_hovertext(
                label, invalid_types.get(label)
            )
            node_hovertexts[label] = hovertext
        edge_hovertexts = {}
        for data in self.graph.qubit_edges.values():
            label = data["label"]
            hovertext = inspection.create_edge_hovertext(
                label, invalid_types.get(label)
            )
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
            save_image=save_image,
            image_name="inspection_summary_valid",
            images_dir=images_dir,
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
            image_name="inspection_summary_invalid",
            images_dir=images_dir,
        )

        if draw_individual_results:
            for inspection in self.inspections.values():
                inspection.draw(
                    save_image=save_image,
                    images_dir=images_dir,
                )
