from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

import numpy as np

from ..backend.config_loader import ConfigLoader


@dataclass(frozen=True)
class InspectionParams:
    max_frequency: float = 9.5
    min_frequency: float = 6.5
    max_detuning: float = 1.3
    min_t1: float = 3e3
    min_t2: float = 3e3
    safe_factor: float = 0.2
    cnot_time: float = 500
    default_t1: float = 3e3
    default_t2_echo: float = 3e3
    default_coupling: float = 0.008
    default_nnn_coupling: float = 0.008 * (0.008 / 0.8)

    @property
    def omega_cr(self) -> float:
        return 1 / (self.cnot_time * 4)


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
    messages: list[str]
    invalid_nodes: list[str]
    invalid_edges: list[str]


class InspectionResult:
    def __init__(
        self,
        inspection_type: InspectionType,
        short_description: str,
        description: str,
        inspection_data: dict[str, InspectionData],
    ):
        self.type = inspection_type
        self.short_description = short_description
        self.description = description
        self.data = dict(sorted(inspection_data.items()))

    @property
    def invalid_nodes(self) -> dict[str, list[str]]:
        nodes = defaultdict(list)
        for data in self.data.values():
            for node in data.invalid_nodes:
                nodes[node].extend(data.messages)
        return dict(nodes)

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        edges = defaultdict(list)
        for data in self.data.values():
            for edge in data.invalid_edges:
                edges[edge].extend(data.messages)
        return dict(edges)

    def print(self):
        print(f"[{self.type}] {self.short_description}")
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


class InspectionSummary:
    def __init__(
        self,
        inspection_result: dict[InspectionType, InspectionResult],
    ):
        self.inspection_result = inspection_result

    @property
    def invalid_nodes(self) -> dict[str, list[str]]:
        invalid_nodes = defaultdict(list)
        for result in self.inspection_result.values():
            for label, messages in result.invalid_nodes.items():
                invalid_nodes[label].extend(messages)
        return dict(sorted(invalid_nodes.items()))

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        invalid_edges = defaultdict(list)
        for result in self.inspection_result.values():
            for label, messages in result.invalid_edges.items():
                invalid_edges[label].extend(messages)
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


class ChipInspector:
    def __init__(
        self,
        chip_id: str,
        props_dir: str | None = None,
        params: dict | None = None,
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

        self.params = InspectionParams()
        if params is not None:
            self.params = replace(self.params, **params)

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
            return self.graph.qubit_edges[target]["label"]
        else:
            raise ValueError("Invalid target type.")

    def get_property(
        self,
        target: int | tuple[int, int],
        property_type: str,
        default: float = np.nan,
    ) -> float:
        if isinstance(target, int):
            value = self.graph.qubit_nodes[target]["properties"][property_type]
            if value is None or np.isnan(value):
                return default
            else:
                return value
        elif isinstance(target, tuple):
            value = self.graph.qubit_edges[target]["properties"][property_type]
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
        data = {}
        for i in self.graph.qubit_nodes:
            is_invalid = False
            messages = []

            label = self.get_label(i)
            omega = self.get_property(i, "frequency", np.nan)
            t1 = self.get_property(i, "t1", self.params.default_t1)
            t2 = self.get_property(i, "t2_echo", self.params.default_t2_echo)

            omega_min = min_frequency or self.params.min_frequency
            omega_max = max_frequency or self.params.max_frequency
            t1_min = min_t1 or self.params.min_t1
            t2_min = min_t2 or self.params.min_t2

            if np.isnan(omega):
                is_invalid = True
                messages.append(f"Frequency of {label} is not defined.")
            if omega < omega_min:
                is_invalid = True
                messages.append(
                    f"Frequency of {label} ({omega:.3f} GHz) is lower than {omega_min:.3f} GHz."
                )
            if omega > omega_max:
                is_invalid = True
                messages.append(
                    f"Frequency of {label} ({omega:.3f} GHz) is higher than {omega_max:.3f} GHz."
                )
            if t1 < t1_min:
                is_invalid = True
                messages.append(
                    f"T1 of {label} ({t1 * 1e-3:.1f} μs) is lower than {t1_min * 1e-3:.1f} μs."
                )
            if t2 < t2_min:
                is_invalid = True
                messages.append(
                    f"T2 of {label} ({t2 * 1e-3:.1f} μs) is lower than {t2_min * 1e-3:.1f} μs."
                )

            if is_invalid:
                data[label] = InspectionData(
                    label=label,
                    messages=messages,
                    invalid_nodes=[label],
                    invalid_edges=[],
                )

        return InspectionResult(
            inspection_type="Type0A",
            short_description="bad qubit",
            description="Qubit unmeasured, frequency out of range, T1, T2 too short.",
            inspection_data=data,
        )

    def check_type0b(
        self,
        max_detuning: float | None = None,
    ) -> InspectionResult:
        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            Delta = abs(omega_i - omega_j)

            Delta_max = max_detuning or self.params.max_detuning

            if Delta > Delta_max:
                is_invalid = True
                messages.append(
                    f"Detuning of {label} ({Delta * 1e3:.0f} MHz) is higher than {Delta_max * 1e3:.0f} MHz."
                )

            if is_invalid:
                data[label] = InspectionData(
                    label=label,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label],
                )

        return InspectionResult(
            inspection_type="Type0B",
            short_description="too far detuning",
            description="ge(i)-ge(j) detuning too large.",
            inspection_data=data,
        )

    def check_type1a(
        self,
        safe_factor: float | None = None,
    ) -> InspectionResult:
        data = {}

        safe_factor = safe_factor or self.params.safe_factor

        for i, j in self.graph.qubit_edges:
            if i > j:
                continue
            is_invalid = False
            messages = []

            label = self.get_label((i, j))
            label_i = self.get_label(i)
            label_j = self.get_label(j)
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            Delta = abs(omega_i - omega_j)
            g = self.get_property((i, j), "coupling")

            if abs(2 * g / Delta) > safe_factor:
                is_invalid = True
                messages.append(
                    f"|2g/Δ| of {label} ({abs(2 * g / Delta):.3f}) is higher than {safe_factor} (g={g * 1e3:.0f} MHz, Δ={Delta * 1e3:.0f} MHz)."
                )

            if is_invalid:
                data[label] = InspectionData(
                    label=label,
                    messages=messages,
                    invalid_nodes=[label_i, label_j],
                    invalid_edges=[],
                )

        for i, nnn in self.next_nearest_neighbors.items():
            label_i = self.get_label(i)
            for j in nnn:
                if i > j:
                    continue
                is_invalid = False
                messages = []
                label_j = self.get_label(j)
                label = f"{label_i}-{label_j}"
                omega_i = self.get_property(i, "frequency")
                omega_j = self.get_property(j, "frequency")
                Delta = abs(omega_i - omega_j)
                g = self.params.default_nnn_coupling
                if abs(2 * g / Delta) > safe_factor:
                    is_invalid = True
                    messages.append(
                        f"|2g/Δ| of {label} ({abs(2 * g / Delta):.3f}) is higher than {safe_factor} (g={g * 1e6:.0f} kHz, Δ={Delta * 1e6:.0f} kHz)."
                    )
                if is_invalid:
                    data[label] = InspectionData(
                        label=label,
                        messages=messages,
                        invalid_nodes=[label_i, label_j],
                        invalid_edges=[],
                    )

        return InspectionResult(
            inspection_type="Type1A",
            short_description="ge and ge too close",
            description="ge(i)-ge(j) detuning too small.",
            inspection_data=data,
        )

    def check_type1b(
        self,
        safe_factor: float | None = None,
    ) -> InspectionResult:
        data = {}

        safe_factor = safe_factor or self.params.safe_factor

        for i, nnn in self.next_nearest_neighbors.items():
            label_i = self.get_label(i)
            for j in nnn:
                is_invalid = False
                messages = []
                label_j = self.get_label(j)
                label = f"{label_i}-{label_j}"
                omega_i = self.get_property(i, "frequency")
                omega_j = self.get_property(j, "frequency")
                Delta = abs(omega_i - omega_j)
                Omega_CR = self.params.omega_cr
                if abs(2 * Omega_CR / Delta) > safe_factor:
                    is_invalid = True
                    messages.append(
                        f"|Ω_CR/Δ| of {label} ({abs(Omega_CR / Delta):.3f}) is higher than {safe_factor} (Ω_CR={Omega_CR * 1e6:.0f} kHz, Δ={Delta * 1e6:.0f} kHz)."
                    )
                if is_invalid:
                    data[label] = InspectionData(
                        label=label,
                        messages=messages,
                        invalid_nodes=[],
                        invalid_edges=[label],
                    )

        return InspectionResult(
            inspection_type="Type1B",
            short_description="CR not selective",
            description="CR from node k drives both k->i and k->j at the same time.",
            inspection_data=data,
        )

    def check_type1c(self): ...

    def check_type2a(self): ...

    def check_type2b(self): ...

    def check_type3a(self): ...

    def check_type3b(self): ...

    def check_type7(self): ...

    def check_type8(self): ...

    def check_type9(self): ...
