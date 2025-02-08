from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..backend.config_loader import ConfigLoader

CONDITIONS = {
    "max_frequency": 9.5,
    "min_frequency": 6.5,
    "max_detuning": 1.3,
    "min_t1": 3e3,
    "min_t2": 3e3,
    "b1": 0.2,
    "b2": 0.75,
}

DEFAULTS = {
    "t1": 3e3,
    "t2_echo": 3e3,
    "coupling": 0.008,
}


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


@dataclass
class InspectionData:
    label: str
    is_valid: bool
    messages: list[str]


class InspectionResult:
    def __init__(
        self,
        type: InspectionType,
        description: str,
        node_data: dict[str, InspectionData],
        edge_data: dict[str, InspectionData],
    ):
        self.type = type
        self.description = description
        self.node_data = dict(sorted(node_data.items()))
        self.edge_data = dict(sorted(edge_data.items()))

    @property
    def invalid_nodes(self) -> dict[str, list[str]]:
        return {
            label: data.messages
            for label, data in self.node_data.items()
            if not data.is_valid
        }

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        return {k: v.messages for k, v in self.edge_data.items() if not v.is_valid}

    def print(self):
        print(f"{self.type}:")
        print(f"  {self.description}")
        if self.invalid_nodes:
            print("Invalid nodes:")
            for label, messages in self.invalid_nodes.items():
                print(f"  {label}:")
                for message in messages:
                    print(f"    - {message}")
        if self.invalid_edges:
            print("Invalid edges:")
            for label, messages in self.invalid_edges.items():
                print(f"  {label}:")
                for message in messages:
                    print(f"    - {message}")


class InspectionSummary:
    def __init__(
        self,
        results: dict[InspectionType, InspectionResult],
    ):
        self.results = results

    @property
    def invalid_nodes(self) -> dict[str, list[str]]:
        invalid_nodes = defaultdict(list)
        for result in self.results.values():
            for label, messages in result.invalid_nodes.items():
                invalid_nodes[label].extend(messages)
        return dict(invalid_nodes)

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        invalid_edges = defaultdict(list)
        for result in self.results.values():
            for label, messages in result.invalid_edges.items():
                invalid_edges[label].extend(messages)
        return dict(invalid_edges)

    def print(self):
        for result in self.results.values():
            result.print()
            print()


class ChipInspector:
    def __init__(
        self,
        chip_id: str,
        params_dir: str | None = None,
        conditions: dict | None = None,
    ):
        if params_dir is None:
            config_loader = ConfigLoader()
        else:
            config_loader = ConfigLoader(params_dir=params_dir)

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

        self.valid_conditions = CONDITIONS
        if conditions is not None:
            self.valid_conditions.update(conditions)

    def get_value(
        self,
        target: int | tuple[int, int],
        property: str,
        default: float = np.nan,
    ) -> float:
        if isinstance(target, int):
            value = self.graph.qubit_nodes[target]["properties"][property]
            if value is None or np.isnan(value):
                return default
            else:
                return value
        elif isinstance(target, tuple):
            value = self.graph.qubit_edges[target]["properties"][property]
            if value is None or np.isnan(value):
                return default
            else:
                return value
        else:
            raise ValueError("Invalid target type.")

    def execute(
        self,
        *collision: InspectionType,
    ) -> InspectionSummary:
        results = {}
        for type in collision:
            results[type] = getattr(self, f"check_{type.lower()}")()
        return InspectionSummary(results)

    def check_type0a(
        self,
        min_frequency: float | None = None,
        max_frequency: float | None = None,
        min_t1: float | None = None,
        min_t2: float | None = None,
    ) -> InspectionResult:
        """
        conditions:
            (1) qubit unmeasured
            (2) qubit frequency out of range
            (3) qubit t1, t2 is too short
        """
        node_result = {}
        for i, data in self.graph.qubit_nodes.items():
            is_valid = True
            label = data["label"]
            messages = []
            f = self.get_value(i, "frequency", np.nan)
            f_min = min_frequency or self.valid_conditions["min_frequency"]
            f_max = max_frequency or self.valid_conditions["max_frequency"]
            t1 = self.get_value(i, "t1", DEFAULTS["t1"])
            t1_min = min_t1 or self.valid_conditions["min_t1"]
            t2 = self.get_value(i, "t2_echo", DEFAULTS["t2_echo"])
            t2_min = min_t2 or self.valid_conditions["min_t2"]
            if np.isnan(f):
                is_valid = False
                messages.append(f"Frequency of {label} is not defined.")
            if f < f_min:
                is_valid = False
                messages.append(
                    f"Frequency of {label} ({f:.3f} GHz) is lower than {f_min:.3f} GHz."
                )
            if f > f_max:
                is_valid = False
                messages.append(
                    f"Frequency of {label} ({f:.3f} GHz) is higher than {f_max:.3f} GHz."
                )
            if t1 < t1_min:
                is_valid = False
                messages.append(
                    f"T1 of {label} ({t1 * 1e-3:.1f} μs) is lower than {t1_min * 1e-3:.1f} μs."
                )
            if t2 < t2_min:
                is_valid = False
                messages.append(
                    f"T2 of {label} ({t2 * 1e-3:.1f} μs) is lower than {t2_min * 1e-3:.1f} μs."
                )
            node_result[label] = InspectionData(
                label=label,
                is_valid=is_valid,
                messages=messages,
            )
        return InspectionResult(
            type="Type0A",
            description="Qubit unmeasured, frequency out of range, T1, T2 too short.",
            node_data=node_result,
            edge_data={},
        )

    def check_type0b(
        self,
        max_detuning: float | None = None,
    ):
        """
        conditions:
            (1) ge(i)-ge(j) is too far to implement CR gate
        """
        edge_results = {}
        for (i, j), data in self.graph.qubit_edges.items():
            is_valid = True
            label = data["label"]
            messages = []
            f_i = self.get_value(i, "frequency")
            f_j = self.get_value(j, "frequency")
            f_diff = abs(f_i - f_j)
            f_diff_max = max_detuning or self.valid_conditions["max_detuning"]
            if f_diff > f_diff_max:
                is_valid = False
                messages.append(
                    f"Detuning of {label} ({f_diff:.3f} GHz) is higher than {f_diff_max:.3f} GHz."
                )
            edge_results[label] = InspectionData(
                label=label,
                is_valid=is_valid,
                messages=messages,
            )
        return InspectionResult(
            type="Type0B",
            description="Detuning too far to implement CR gate.",
            node_data={},
            edge_data=edge_results,
        )

    def check_type1a(self): ...

    def check_type1b(self): ...

    def check_type1c(self): ...

    def check_type2a(self): ...

    def check_type2b(self): ...

    def check_type3a(self): ...

    def check_type3b(self): ...

    def check_type7(self): ...

    def check_type8(self): ...

    def check_type9(self): ...
