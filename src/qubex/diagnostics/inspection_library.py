from __future__ import annotations

import numpy as np

from ..backend.lattice_graph import LatticeGraph
from .inspection import Inspection


class Type0A(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type0A",
            description="Qubit unmeasured, frequency out of range, T1, T2 too short.",
            graph=graph,
            params=params,
        )

    def execute(self):
        min_frequency = self.params.min_frequency
        max_frequency = self.params.max_frequency
        min_t1 = self.params.min_t1
        min_t2 = self.params.min_t2

        for i in self.graph.qubit_nodes:
            label = self.get_label(i)
            f = self.get_frequency(i)
            t1 = self.get_t1(i)
            t2 = self.get_t2(i)

            if np.isnan(f):
                self.add_invalid_nodes(
                    [label],
                    f"Frequency of {label} is not defined.",
                )
            if f < min_frequency:
                self.add_invalid_nodes(
                    [label],
                    f"Frequency of {label} ({f:.3f} GHz) is lower than {min_frequency:.3f} GHz.",
                )
            if f > max_frequency:
                self.add_invalid_nodes(
                    [label],
                    f"Frequency of {label} ({f:.3f} GHz) is higher than {max_frequency:.3f} GHz.",
                )
            if t1 < min_t1:
                self.add_invalid_nodes(
                    [label],
                    f"T1 of {label} ({t1 * 1e-3:.1f} μs) is lower than {min_t1 * 1e-3:.1f} μs.",
                )
            if t2 < min_t2:
                self.add_invalid_nodes(
                    [label],
                    f"T2 of {label} ({t2 * 1e-3:.1f} μs) is lower than {min_t2 * 1e-3:.1f} μs.",
                )


class Type0B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type0B",
            description="ge(i)-ge(j) detuning too large.",
            graph=graph,
            params=params,
        )

    def execute(self):
        max_detuning = self.params.max_detuning

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            Delta = omega_i - omega_j

            if Delta > max_detuning:
                self.add_invalid_edges(
                    [label],
                    f"Detuning of {label} ({Delta * 1e3:.0f} MHz) is higher than {max_detuning * 1e3:.0f} MHz.",
                )


class Type1A(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type1A",
            description="ge(i)-ge(j) detuning too small.",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            if i > j:
                continue

            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j

            if abs(2 * g / Delta) > adiabatic_limit:
                self.add_invalid_edges(
                    [label],
                    f"|2g/Δ| of {label} ({abs(2 * g / Delta):.3f}) is higher than {adiabatic_limit} (g={g * 1e3:.0f} MHz, Δ={Delta * 1e3:.0f} MHz).",
                )

        for i, nnn in self.next_nearest_neighbors.items():
            for j in nnn:
                if i > j:
                    continue

                label = self.get_label((i, j))
                omega_i = self.get_frequency(i)
                omega_j = self.get_frequency(j)
                Delta = omega_i - omega_j
                g = self.params.default_nnn_coupling

                if abs(2 * g / Delta) > adiabatic_limit:
                    self.add_invalid_edges(
                        [label],
                        f"|2g/Δ| of {label} ({abs(2 * g / Delta):.3f}) is higher than {adiabatic_limit} (g={g * 1e6:.0f} kHz, Δ={Delta * 1e6:.0f} kHz).",
                    )


class Type1B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type1B",
            description="CR from node k drives both k->i and k->j at the same time.",
            graph=graph,
            params=params,
        )

    def execute(self):
        cr_control_limit = self.params.cr_control_limit

        for k in self.graph.qubit_nodes:
            for i in self.nearest_neighbors[k]:
                omega_i = self.get_frequency(i)
                for j in self.nearest_neighbors[k]:
                    if i > j:
                        continue
                    if i == j:
                        continue

                    label_ij = self.get_label((i, j))
                    label_ki = self.get_label((k, i))
                    label_kj = self.get_label((k, j))
                    omega_j = self.get_frequency(j)
                    Delta_ij = omega_i - omega_j
                    Omega_CR = 1 / (self.params.cnot_time * 4)

                    if abs(Omega_CR / Delta_ij) > cr_control_limit:
                        self.add_invalid_edges(
                            [label_ki, label_kj],
                            f"|Ω_CR/Δ| of {label_ij} ({abs(Omega_CR / Delta_ij):.3f}) is higher than {cr_control_limit} (Ω_CR={Omega_CR * 1e6:.0f} kHz, Δ={Delta_ij * 1e6:.0f} kHz).",
                        )


class Type1C(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type1C",
            description="CR(i->j) excites g-e(i) transition.",
            graph=graph,
            params=params,
        )

    def execute(self):
        cr_control_limit = self.params.cr_control_limit

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR

            if abs(Omega_d / Delta) > cr_control_limit:
                self.add_invalid_edges(
                    [label],
                    f"|Ω_d/Δ| of {label} ({abs(Omega_d / Delta):.3f}) is higher than {cr_control_limit} (Ω_d={Omega_d * 1e3:.0f} MHz, Δ={Delta * 1e3:.0f} MHz).",
                )


class Type2A(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type2A",
            description="CR(i->j) excites g-f(i) transition.",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR
            Omega_eff = 2 ** (-1.5) * Omega_d**2 * (1 / (Delta + alpha_i) - 1 / Delta)

            val = abs(Omega_eff / (2 * Delta + alpha_i))
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label],
                    f"|Ω_eff/(2Δ+α)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ+α={2 * Delta + alpha_i * 1e3:.0f} MHz).",
                )


class Type2B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type2B",
            description="CR(i->j) excites f-g(i) transition.",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR
            Omega_eff = 2 ** (-1.5) * Omega_d**2 * (1 / (Delta + alpha_i) - 1 / Delta)

            val = abs(Omega_eff / (2 * Delta))
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label],
                    f"|Ω_eff/(2Δ)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ={2 * Delta * 1e3:.0f} MHz).",
                )


class Type3A(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type3A",
            description="CR(i->j) excites g-f(i) transition.",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR
            Omega_eff = 2 ** (-1.5) * Omega_d**2 * (1 / (Delta + alpha_i) - 1 / Delta)

            val = abs(Omega_eff / (2 * Delta + alpha_i))
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label],
                    f"|Ω_eff/(2Δ+α)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ+α={2 * Delta + alpha_i * 1e3:.0f} MHz).",
                )


class Type3B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type3B",
            description="CR(i->j) excites f-g(i) transition.",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR
            Omega_eff = 2 ** (-1.5) * Omega_d**2 * (1 / (Delta + alpha_i) - 1 / Delta)

            val = abs(Omega_eff / (2 * Delta))
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label],
                    f"|Ω_eff/(2Δ)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ={2 * Delta * 1e3:.0f} MHz).",
                )


class Type7(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type7",
            description="CR(i->j) excites g-f(i) transition.",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR
            Omega_eff = 2 ** (-1.5) * Omega_d**2 * (1 / (Delta + alpha_i) - 1 / Delta)

            val = abs(Omega_eff / (2 * Delta + alpha_i))
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label],
                    f"|Ω_eff/(2Δ+α)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ+α={2 * Delta + alpha_i * 1e3:.0f} MHz).",
                )


class Type8(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type8",
            description="CR(i->j) excites f-g(i) transition.",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR
            Omega_eff = 2 ** (-1.5) * Omega_d**2 * (1 / (Delta + alpha_i) - 1 / Delta)

            val = abs(Omega_eff / (2 * Delta))
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label],
                    f"|Ω_eff/(2Δ)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ={2 * Delta * 1e3:.0f} MHz).",
                )


class Type9(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type9",
            description="CR(i->j) excites f-g(i) transition.",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g = self.get_nn_coupling((i, j))
            Delta = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d = Delta * (Delta + alpha_i) / (g * alpha_i) * Omega_CR
            Omega_eff = 2 ** (-1.5) * Omega_d**2 * (1 / (Delta + alpha_i) - 1 / Delta)

            val = abs(Omega_eff / (2 * Delta))
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label],
                    f"|Ω_eff/(2Δ)| of {label} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff * 1e3:.0f} MHz, 2Δ={2 * Delta * 1e3:.0f} MHz).",
                )
