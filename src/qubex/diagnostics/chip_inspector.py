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
    adiabatic_limit: float = 0.1
    cr_control_limit: float = 0.75
    cnot_time: float = 500
    default_t1: float = 3e3
    default_t2_echo: float = 3e3
    default_coupling: float = 0.008
    default_nnn_coupling: float = 0.008 * (0.008 / 0.8)


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
                invalid_nodes[label].extend(
                    [f"[{result.type}] {message}" for message in messages]
                )
        return dict(sorted(invalid_nodes.items()))

    @property
    def invalid_edges(self) -> dict[str, list[str]]:
        invalid_edges = defaultdict(list)
        for result in self.inspection_result.values():
            for label, messages in result.invalid_edges.items():
                invalid_edges[label].extend(
                    [f"[{result.type}] {message}" for message in messages]
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
    ) -> float:
        if isinstance(target, int):
            value = self.graph.qubit_nodes[target]["properties"][property_type]
            if value is None or np.isnan(value):
                if property_type == "anharmonicity":
                    f = self.get_property(target, "frequency")
                    return (-1 / 19) * f
                elif property_type == "t1":
                    return self.params.default_t1
                elif property_type == "t2_echo":
                    return self.params.default_t2_echo
                else:
                    return np.nan
            else:
                return value
        elif isinstance(target, tuple):
            value = self.graph.qubit_edges[target]["properties"][property_type]
            if value is None or np.isnan(value):
                if property_type == "coupling":
                    return self.params.default_coupling
                elif property_type == "nnn_coupling":
                    return self.params.default_nnn_coupling
                else:
                    return np.nan
            else:
                return value
        else:
            raise ValueError("Invalid target type.")

    def execute(
        self,
        *collision: InspectionType,
    ) -> InspectionSummary:
        if not collision:
            collision = (
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
            )

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
        if min_frequency is None:
            min_frequency = self.params.min_frequency
        if max_frequency is None:
            max_frequency = self.params.max_frequency
        if min_t1 is None:
            min_t1 = self.params.min_t1
        if min_t2 is None:
            min_t2 = self.params.min_t2

        data = {}
        for i in self.graph.qubit_nodes:
            is_invalid = False
            messages = []

            label = self.get_label(i)
            f = self.get_property(i, "frequency")
            t1 = self.get_property(i, "t1")
            t2 = self.get_property(i, "t2_echo")

            if np.isnan(f):
                is_invalid = True
                messages.append(f"Frequency of {label} is not defined.")
            if f < min_frequency:
                is_invalid = True
                messages.append(
                    f"Frequency of {label} ({f:.3f} GHz) is lower than {min_frequency:.3f} GHz."
                )
            if f > max_frequency:
                is_invalid = True
                messages.append(
                    f"Frequency of {label} ({f:.3f} GHz) is higher than {max_frequency:.3f} GHz."
                )
            if t1 < min_t1:
                is_invalid = True
                messages.append(
                    f"T1 of {label} ({t1 * 1e-3:.1f} μs) is lower than {min_t1 * 1e-3:.1f} μs."
                )
            if t2 < min_t2:
                is_invalid = True
                messages.append(
                    f"T2 of {label} ({t2 * 1e-3:.1f} μs) is lower than {min_t2 * 1e-3:.1f} μs."
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
        if max_detuning is None:
            max_detuning = self.params.max_detuning

        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            Delta = omega_i - omega_j

            if Delta > max_detuning:
                is_invalid = True
                messages.append(
                    f"Detuning of {label} ({Delta * 1e3:.0f} MHz) is higher than {max_detuning * 1e3:.0f} MHz."
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
        adiabatic_limit: float | None = None,
    ) -> InspectionResult:
        if adiabatic_limit is None:
            adiabatic_limit = self.params.adiabatic_limit

        data = {}
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
            g = self.get_property((i, j), "coupling")
            Delta = omega_i - omega_j

            if abs(2 * g / Delta) > adiabatic_limit:
                is_invalid = True
                messages.append(
                    f"|2g/Δ| of {label} ({abs(2 * g / Delta):.3f}) is higher than {adiabatic_limit} (g={g * 1e3:.0f} MHz, Δ={Delta * 1e3:.0f} MHz)."
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
                Delta = omega_i - omega_j
                g = self.params.default_nnn_coupling
                if abs(2 * g / Delta) > adiabatic_limit:
                    is_invalid = True
                    messages.append(
                        f"|2g/Δ| of {label} ({abs(2 * g / Delta):.3f}) is higher than {adiabatic_limit} (g={g * 1e6:.0f} kHz, Δ={Delta * 1e6:.0f} kHz)."
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
        adiabatic_limit: float | None = None,
    ) -> InspectionResult:
        if adiabatic_limit is None:
            adiabatic_limit = self.params.adiabatic_limit

        data = {}
        for k in self.graph.qubit_nodes:
            label_k = self.get_label(k)
            for i in self.nearest_neighbors[k]:
                label_i = self.get_label(i)
                omega_i = self.get_property(i, "frequency")
                for j in self.nearest_neighbors[k]:
                    if i > j:
                        continue
                    if i == j:
                        continue

                    is_invalid = False
                    messages = []

                    label_j = self.get_label(j)
                    label_ij = f"{label_i}-{label_j}"
                    label_ki = f"{label_k}-{label_i}"
                    label_kj = f"{label_k}-{label_j}"
                    omega_j = self.get_property(j, "frequency")
                    Delta_ij = abs(omega_i - omega_j)
                    Omega_CR = 1 / (self.params.cnot_time * 4)

                    if abs(Omega_CR / Delta_ij) > adiabatic_limit:
                        is_invalid = True
                        messages.append(
                            f"|Ω_CR/Δ| of {label_ij} ({abs(Omega_CR / Delta_ij):.3f}) is higher than {adiabatic_limit} (Ω_CR={Omega_CR * 1e6:.0f} kHz, Δ={Delta_ij * 1e6:.0f} kHz)."
                        )

                    if is_invalid:
                        data[label_i] = InspectionData(
                            label=label_i,
                            messages=messages,
                            invalid_nodes=[],
                            invalid_edges=[label_ki, label_kj],
                        )

        return InspectionResult(
            inspection_type="Type1B",
            short_description="CR not selective",
            description="CR from node k drives both k->i and k->j at the same time.",
            inspection_data=data,
        )

    def check_type1c(
        self,
        cr_control_limit: float | None = None,
    ) -> InspectionResult:
        if cr_control_limit is None:
            cr_control_limit = self.params.cr_control_limit

        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            alpha_i = self.get_property(i, "anharmonicity")
            g = self.get_property((i, j), "coupling")
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = abs(Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR)

            if abs(Omega_d / Delta) > cr_control_limit:
                is_invalid = True
                messages.append(
                    f"|Ω_d/Δ| of {label} ({abs(Omega_d / Delta):.3f}) is higher than {cr_control_limit} (Ω_d={Omega_d * 1e3:.0f} MHz, Δ={Delta * 1e3:.0f} MHz)."
                )

            if is_invalid:
                data[label] = InspectionData(
                    label=label,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label],
                )

        return InspectionResult(
            inspection_type="Type1C",
            short_description="CR cause g-e",
            description="CR(i->j) excites g-e(i) transition.",
            inspection_data=data,
        )

    def check_type2a(
        self,
        adiabatic_limit: float | None = None,
    ) -> InspectionResult:
        if adiabatic_limit is None:
            adiabatic_limit = self.params.adiabatic_limit

        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            alpha_i = self.get_property(i, "anharmonicity")
            g = self.get_property((i, j), "coupling")
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = abs(Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR)
            Omega_eff = abs(
                2 ** (-1.5) * Omega_d**2 * (1 / (Delta + alpha_i) - 1 / Delta)
            )

            val = abs(Omega_eff / (2 * Delta + alpha_i))
            if val > adiabatic_limit:
                is_invalid = True
                messages.append(
                    f"|Ω_eff/(2Δ+α)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ+α={2 * Delta + alpha_i * 1e3:.0f} MHz)."
                )

            if is_invalid:
                data[label] = InspectionData(
                    label=label,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label],
                )

        return InspectionResult(
            inspection_type="Type2A",
            short_description="CR cause gf",
            description="CR(i->j) excites g-f(i) transition.",
            inspection_data=data,
        )

    def check_type2b(
        self,
        adiabatic_limit: float | None = None,
    ) -> InspectionResult:
        if adiabatic_limit is None:
            adiabatic_limit = self.params.adiabatic_limit

        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            alpha_i = self.get_property(i, "anharmonicity")
            g = self.get_property((i, j), "coupling")
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = abs(Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR)
            Omega_eff = abs(
                2 ** (0.5) * g * Omega_d * (1 / (Delta + alpha_i) - 1 / Delta)
            )

            val = abs(Omega_eff / (2 * Delta + alpha_i))
            if val > adiabatic_limit:
                is_invalid = True
                messages.append(
                    f"|Ω_eff/(2Δ+α)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ+α={2 * Delta + alpha_i * 1e3:.0f} MHz)."
                )

            if is_invalid:
                data[label] = InspectionData(
                    label=label,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label],
                )

        return InspectionResult(
            inspection_type="Type2B",
            short_description="CR cause fogi",
            description="CR(i->j) drive excites fogi (fg<->ge) transition.",
            inspection_data=data,
        )

    def check_type3a(
        self,
        adiabatic_limit: float | None = None,
    ) -> InspectionResult:
        if adiabatic_limit is None:
            adiabatic_limit = self.params.adiabatic_limit

        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            alpha_i = self.get_property(i, "anharmonicity")
            g = self.get_property((i, j), "coupling")
            Delta = omega_i - omega_j

            val = abs(2**1.5 * g / (Delta + alpha_i))

            if val > adiabatic_limit:
                is_invalid = True
                messages.append(
                    f"|2√2g/(Δ+α)| of {label} ({val:.3f}) is higher than {adiabatic_limit} (g={g * 1e3:.0f} MHz, Δ+α={Delta + alpha_i * 1e3:.0f} MHz)."
                )

            if is_invalid:
                data[label] = InspectionData(
                    label=label,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label],
                )

        for i, nnn in self.next_nearest_neighbors.items():
            label_i = self.get_label(i)
            for j in nnn:
                is_invalid = False
                messages = []

                label_j = self.get_label(j)
                label = f"{label_i}-{label_j}"
                omega_i = self.get_property(i, "frequency")
                omega_j = self.get_property(j, "frequency")
                alpha_i = self.get_property(i, "anharmonicity")
                Delta = omega_i - omega_j
                g = self.params.default_nnn_coupling

                val = abs(2**1.5 * g / (Delta + alpha_i))

                if val > adiabatic_limit:
                    is_invalid = True
                    messages.append(
                        f"|2√2g/(Δ+α)| of {label} ({val:.3f}) is higher than {adiabatic_limit} (g={g * 1e6:.0f} kHz, Δ+α={Delta + alpha_i * 1e6:.0f} kHz)."
                    )
                if is_invalid:
                    data[label] = InspectionData(
                        label=label,
                        messages=messages,
                        invalid_nodes=[],
                        invalid_edges=[label],
                    )

        return InspectionResult(
            inspection_type="Type3A",
            short_description="ef and ge too close",
            description="ef(i) and ge(j) too close.",
            inspection_data=data,
        )

    def check_type3b(
        self,
        cr_control_limit: float | None = None,
    ) -> InspectionResult:
        if cr_control_limit is None:
            cr_control_limit = self.params.cr_control_limit

        data = {}
        for i, j in self.graph.qubit_edges:
            if i > j:
                continue

            is_invalid = False
            messages = []

            label = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            alpha_i = self.get_property(i, "anharmonicity")
            g = self.get_property((i, j), "coupling")
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = abs(Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR)

            val = abs(2**0.5 * Omega_d / (Delta + alpha_i))

            if val > cr_control_limit:
                is_invalid = True
                messages.append(
                    f"|√2Ω_d/(Δ+α)| of {label} ({val:.3f}) is higher than {cr_control_limit} (Ω_d={Omega_d * 1e3:.0f} MHz, Δ+α={Delta + alpha_i * 1e3:.0f} MHz)."
                )
            if is_invalid:
                data[label] = InspectionData(
                    label=label,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label],
                )

        return InspectionResult(
            inspection_type="Type3B",
            short_description="CR cause e-f",
            description="CR (i->j) excites the e-f(i) transition.",
            inspection_data=data,
        )

    def check_type7(
        self,
        adiabatic_limit: float | None = None,
    ) -> InspectionResult:
        if adiabatic_limit is None:
            adiabatic_limit = self.params.adiabatic_limit

        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label_ij = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            alpha_i = self.get_property(i, "anharmonicity")
            g_ij = self.get_property((i, j), "coupling")
            Delta_ij = abs(omega_i - omega_j)
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = abs(
                Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR
            )

            for k in self.nearest_neighbors[i]:
                if k == j:
                    continue

                label_ik = self.get_label((i, k))
                omega_k = self.get_property(k, "frequency")
                Delta_ik = abs(omega_i - omega_k)
                g_ik = self.get_property((i, k), "coupling")
                Omega_ik = abs(
                    2 ** (-0.5)
                    * g_ik
                    * Omega_d_ij
                    * (
                        1 / (Delta_ij + alpha_i)
                        - 1 / Delta_ij
                        + 1 / (Delta_ik + alpha_i)
                        - 1 / Delta_ik
                    )
                )

                val = abs(Omega_ik / (2 * Delta_ik + alpha_i))

                if val > adiabatic_limit:
                    is_invalid = True
                    messages.append(
                        f"|Ω_ik/(2Δ+α)| of {label_ik} ({val:.3g}) is higher than {adiabatic_limit} (Ω_ik={Omega_ik * 1e3:.0f} MHz, 2Δ+α={2 * Delta_ik + alpha_i * 1e3:.0f} MHz)."
                    )

            if is_invalid:
                data[label_ij] = InspectionData(
                    label=label_ij,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label_ij],
                )

        return InspectionResult(
            inspection_type="Type7",
            short_description="CR cause fogi, between spectator",
            description="CR(i->j) drive excites fogi (i->k) transition.",
            inspection_data=data,
        )

    def check_type8(
        self,
    ) -> InspectionResult:
        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label_i = self.get_label(i)
            label_ij = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            alpha_i = self.get_property(i, "anharmonicity")
            g_ij = self.get_property((i, j), "coupling")
            Delta_ij = abs(omega_i - omega_j)
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = abs(
                Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR
            )
            Delta_stark = (
                Omega_d_ij**2 * alpha_i / (2 * Delta_ij * (Delta_ij + alpha_i))
            )

            for k in self.nearest_neighbors[i]:
                if k == j:
                    continue

                label_ik = self.get_label((i, k))
                omega_k = self.get_property(k, "frequency")
                Delta_ik = abs(omega_i - omega_k)

                val = Delta_ik * (Delta_ik + Delta_stark)

                if val < 0:
                    is_invalid = True
                    messages.append(
                        f"Delta_ik * (Delta_ik + Delta_stark) of {label_ik} ({val:.3g}) is negative (Delta_ik={Delta_ik:.3g}, Delta_stark={Delta_stark:.3g})."
                    )

            for k in self.next_nearest_neighbors[i]:
                label_k = self.get_label(k)
                label_ik = f"{label_i}-{label_k}"
                omega_k = self.get_property(k, "frequency")
                Delta_ik = abs(omega_i - omega_k)

                val = Delta_ik * (Delta_ik + Delta_stark)

                if val < 0:
                    is_invalid = True
                    messages.append(
                        f"Delta_ik * (Delta_ik + Delta_stark) of {label_ik} ({val:.3g}) is negative (Delta_ik={Delta_ik:.3g}, Delta_stark={Delta_stark:.3g})."
                    )

            if is_invalid:
                data[label_ij] = InspectionData(
                    label=label_ij,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label_ij],
                )

        return InspectionResult(
            inspection_type="Type8",
            short_description="ge-ge too close with Stark shift",
            description="ge(i) and ge(k) become close during CR(i->j).",
            inspection_data=data,
        )

    def check_type9(
        self,
    ) -> InspectionResult:
        data = {}
        for i, j in self.graph.qubit_edges:
            is_invalid = False
            messages = []

            label_i = self.get_label(i)
            label_ij = self.get_label((i, j))
            omega_i = self.get_property(i, "frequency")
            omega_j = self.get_property(j, "frequency")
            alpha_i = self.get_property(i, "anharmonicity")
            g_ij = self.get_property((i, j), "coupling")
            Delta_ij = abs(omega_i - omega_j)
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = abs(
                Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR
            )
            Delta_stark = (
                Omega_d_ij**2 * alpha_i / (2 * Delta_ij * (Delta_ij + alpha_i))
            )

            for k in self.nearest_neighbors[i]:
                if k == j:
                    continue

                label_ik = self.get_label((i, k))
                omega_k = self.get_property(k, "frequency")
                Delta_ik = abs(omega_i - omega_k)

                val = (Delta_ik + alpha_i) * (Delta_ik + alpha_i + Delta_stark)

                if val < 0:
                    is_invalid = True
                    messages.append(
                        f"(Delta_ik + α)(Delta_ik + α + Δstark) of {label_ik} ({val:.3g}) is negative (Delta_ik={Delta_ik:.3g}, α={alpha_i:.3g}, Δstark={Delta_stark:.3g})."
                    )

            for k in self.next_nearest_neighbors[i]:
                label_k = self.get_label(k)
                label_ik = f"{label_i}-{label_k}"
                omega_k = self.get_property(k, "frequency")
                Delta_ik = abs(omega_i - omega_k)

                val = (Delta_ik + alpha_i) * (Delta_ik + alpha_i + Delta_stark)

                if val < 0:
                    is_invalid = True
                    messages.append(
                        f"(Delta_ik + α)(Delta_ik + α + Δstark) of {label_ik} ({val:.3g}) is negative (Delta_ik={Delta_ik:.3g}, α={alpha_i:.3g}, Δstark={Delta_stark:.3g})."
                    )

            if is_invalid:
                data[label_ij] = InspectionData(
                    label=label_ij,
                    messages=messages,
                    invalid_nodes=[],
                    invalid_edges=[label_ij],
                )

        return InspectionResult(
            inspection_type="Type9",
            short_description="ge-ef too close with Stark shift",
            description="ge(i) and ef(k) become close during CR(i->j).",
            inspection_data=data,
        )
