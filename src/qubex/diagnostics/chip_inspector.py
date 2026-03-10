"""Chip inspection routines and reporting utilities."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

from qubex.system import ConfigLoader
from qubex.system.lattice_graph import LatticeGraph

from . import inspection_library
from .inspection import Inspection

logger = logging.getLogger(__name__)

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
    """Run inspection suites against chip configuration data."""

    def __init__(
        self,
        chip_id: str | None = None,
        *,
        system_id: str | None = None,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        props_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize one inspector from system or chip configuration.

        `system_id` is the canonical selector. `chip_id` remains available as a
        compatibility input for single-system chip configurations.
        """
        self._init_graph(
            chip_id=chip_id,
            system_id=system_id,
            config_dir=config_dir,
            params_dir=params_dir,
            props_dir=props_dir,
        )

    def _init_graph(
        self,
        chip_id: str | None = None,
        *,
        system_id: str | None = None,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        props_dir: Path | str | None = None,
    ) -> None:
        """Load config data and build the inspection graph."""
        if chip_id is None and system_id is None:
            raise ValueError("Either `system_id` or `chip_id` must be provided.")
        resolved_params_dir = params_dir if params_dir is not None else props_dir
        config_loader = ConfigLoader(
            chip_id=chip_id,
            system_id=system_id,
            config_dir=config_dir,
            params_dir=resolved_params_dir,
        )
        experiment_system = config_loader.get_experiment_system()
        self.graph = experiment_system.quantum_system.chip_graph

        frequency_dict = config_loader.load_param_data("qubit_frequency")
        anharmonicity_dict = config_loader.load_param_data("qubit_anharmonicity")
        t1_dict = config_loader.load_param_data("t1")
        t2_echo_dict = config_loader.load_param_data("t2_echo")
        coupling_dict = config_loader.load_param_data("qubit_qubit_coupling_strength")

        def _get_val(d: dict, k: str) -> float:
            v = d.get(k)
            return v if v is not None else float("nan")

        for node in self.graph.qubit_nodes.values():
            label = node["label"]
            node["properties"] = {
                "frequency": _get_val(frequency_dict, label),
                "anharmonicity": _get_val(anharmonicity_dict, label),
                "t1": _get_val(t1_dict, label),
                "t2_echo": _get_val(t2_echo_dict, label),
            }

        for edge in self.graph.qubit_edges.values():
            label = edge["label"]
            edge["properties"] = {
                "coupling": _get_val(coupling_dict, label),
            }

    def execute(
        self,
        types: InspectionType | list[InspectionType] | None = None,
        params: dict | None = None,
    ) -> InspectionSummary:
        """Execute selected inspections and return a summary."""
        if types is None:
            types = list(InspectionType.__args__)
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
            except Exception:
                logger.exception("Error in %s", inspection.name)
            inspections[type] = inspection

        return InspectionSummary(
            graph=self.graph,
            inspections=inspections,
        )


class InspectionSummary:
    """Aggregate inspection results and provide reporting helpers."""

    def __init__(
        self,
        graph: LatticeGraph,
        inspections: dict[InspectionType, Inspection],
    ):
        self.graph = graph
        self.inspections = inspections

    @property
    def invalid_nodes(self) -> dict[str, list[str]]:
        """Return invalid nodes aggregated across inspections."""
        invalid_nodes = defaultdict(list)
        for result in self.inspections.values():
            for label, messages in result.invalid_nodes.items():
                invalid_nodes[label].extend(
                    [f"[{result.name}] {message}" for message in messages]
                )
        return dict(sorted(invalid_nodes.items()))

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        """Return invalid edges aggregated across inspections."""
        invalid_edges = defaultdict(list)
        for result in self.inspections.values():
            for label, messages in result.invalid_edges.items():
                invalid_edges[label].extend(
                    [f"[{result.name}] {message}" for message in messages]
                )
        return dict(sorted(invalid_edges.items()))

    def log_report(self) -> None:
        """Log a summary of invalid nodes and edges."""
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
        draw_individual_results: bool = True,
        save_image: bool = False,
        images_dir: str = "./images",
    ) -> None:
        """Visualize inspection results on the chip graph."""
        if len(self.inspections) == 0:
            raise ValueError("No inspections have been executed.")

        all_nodes = {node["label"] for node in self.graph.qubit_nodes.values()}
        invalid_nodes = set(self.invalid_nodes)
        valid_nodes = all_nodes - invalid_nodes

        invalid_node_values = dict.fromkeys(invalid_nodes, 1)
        valid_node_values = dict.fromkeys(valid_nodes, 1)

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

        invalid_edge_values = dict.fromkeys(invalid_edges, 1)
        valid_edge_values = dict.fromkeys(valid_edges, 1)

        invalid_types: dict[str, list[str]] = defaultdict(list)
        for type, inspection in self.inspections.items():
            for label in inspection.invalid_nodes:
                invalid_types[label].append(type.replace("Type", ""))
            for label in inspection.invalid_edges:
                invalid_types[label].append(type.replace("Type", ""))

        inspection = next(iter(self.inspections.values()))
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
