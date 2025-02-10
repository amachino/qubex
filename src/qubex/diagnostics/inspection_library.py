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
            description="bad qubit",
            graph=graph,
            params=params,
        )

    def execute(self):
        min_frequency = self.params.min_frequency
        max_frequency = self.params.max_frequency
        min_t1 = self.params.min_t1
        min_t2 = self.params.min_t2

        for i in self.graph.qubit_nodes:
            label_i = self.get_label(i)
            f_i = self.get_frequency(i)
            t1_i = self.get_t1(i)
            t2_i = self.get_t2(i)

            if np.isnan(f_i):
                self.add_invalid_nodes(
                    [label_i],
                    f"Frequency of {label_i} is not defined.",
                )
            if f_i < min_frequency:
                self.add_invalid_nodes(
                    [label_i],
                    f"Frequency of {label_i} ({f_i:.3f} GHz) is lower than {min_frequency:.3f} GHz.",
                )
            if f_i > max_frequency:
                self.add_invalid_nodes(
                    [label_i],
                    f"Frequency of {label_i} ({f_i:.3f} GHz) is higher than {max_frequency:.3f} GHz.",
                )
            if t1_i < min_t1:
                self.add_invalid_nodes(
                    [label_i],
                    f"T1 of {label_i} ({t1_i * 1e-3:.1f} μs) is lower than {min_t1 * 1e-3:.1f} μs.",
                )
            if t2_i < min_t2:
                self.add_invalid_nodes(
                    [label_i],
                    f"T2 of {label_i} ({t2_i * 1e-3:.1f} μs) is lower than {min_t2 * 1e-3:.1f} μs.",
                )


class Type0B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type0B",
            description="too far detuning",
            graph=graph,
            params=params,
        )

    def execute(self):
        max_detuning = self.params.max_detuning

        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            Delta_ij = omega_i - omega_j

            if abs(Delta_ij) > max_detuning:
                self.add_invalid_edges(
                    [label_ij],
                    f"Detuning of {label_ij} ({Delta_ij * 1e3:.0f} MHz) is higher than {max_detuning * 1e3:.0f} MHz.",
                )


class Type1A(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type1A",
            description="ge and ge too close",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            if i > j:
                continue

            label_ij = self.get_label((i, j))
            label_i = self.get_label(i)
            label_j = self.get_label(j)
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j

            val = abs(2 * g_ij / Delta_ij)
            if val > adiabatic_limit:
                self.add_invalid_nodes(
                    [label_i, label_j],
                    f"|2g/Δ| of {label_ij} ({val:.3f}) is higher than {adiabatic_limit} (g={g_ij * 1e3:.0f} MHz, Δ={Delta_ij * 1e3:.0f} MHz).",
                )

        for i, nnn in self.next_nearest_neighbors.items():
            for k in nnn:
                if i > k:
                    continue

                label_ik = self.get_label((i, k))
                label_i = self.get_label(i)
                label_k = self.get_label(k)
                omega_i = self.get_frequency(i)
                omega_k = self.get_frequency(k)
                Delta_ik = omega_i - omega_k
                g_ik = self.params.default_nnn_coupling

                val = abs(2 * g_ik / Delta_ik)
                if val > adiabatic_limit:
                    self.add_invalid_nodes(
                        [label_i, label_k],
                        f"|2g/Δ| of {label_ik} ({val:.3f}) is higher than {adiabatic_limit} (g={g_ik * 1e6:.0f} kHz, Δ={Delta_ik * 1e6:.0f} kHz).",
                    )


class Type1B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type1B",
            description="CR not selective",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

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

                    val = abs(Omega_CR / Delta_ij)
                    if val > adiabatic_limit:
                        self.add_invalid_edges(
                            [label_ki, label_kj],
                            f"|Ω_CR/Δ| of {label_ij} ({val:.3f}) is higher than {adiabatic_limit} (Ω_CR={Omega_CR * 1e6:.0f} kHz, Δ={Delta_ij * 1e6:.0f} kHz).",
                        )


class Type1C(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type1C",
            description="CR cause g-e",
            graph=graph,
            params=params,
        )

    def execute(self):
        cr_control_limit = self.params.cr_control_limit

        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR

            val = abs(Omega_d_ij / Delta_ij)
            if val > cr_control_limit:
                self.add_invalid_edges(
                    [label_ij],
                    f"|Ω_d/Δ| of {label_ij} ({val:.3f}) is higher than {cr_control_limit} (Ω_d={Omega_d_ij * 1e3:.0f} MHz, Δ={Delta_ij * 1e3:.0f} MHz).",
                )


class Type2A(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type2A",
            description="CR cause g-f",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR
            Omega_eff_ij = (
                2 ** (-1.5) * Omega_d_ij**2 * (1 / (Delta_ij + alpha_i) - 1 / Delta_ij)
            )
            Delta_eff_ij = 2 * Delta_ij + alpha_i

            val = abs(Omega_eff_ij / Delta_eff_ij)
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label_ij],
                    f"|Ω_eff/(2Δ+α)| of {label_ij} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff_ij * 1e3:.0f} MHz, 2Δ+α={Delta_eff_ij * 1e3:.0f} MHz).",
                )


class Type2B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type2B",
            description="CR cause fogi",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR
            Omega_eff_ij = (
                2**0.5 * Omega_d_ij * g_ij / (1 / (Delta_ij + alpha_i) - 1 / Delta_ij)
            )
            Delta_eff_ij = 2 * Delta_ij + alpha_i

            val = abs(Omega_eff_ij / Delta_eff_ij)
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label_ij],
                    f"|Ω_eff/(2Δ+α)| of {label_ij} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff_ij * 1e3:.0f} MHz, 2Δ+α={Delta_eff_ij * 1e3:.0f} MHz).",
                )


class Type3A(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type3A",
            description="ef and ge too close",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j
            Omega_eff_ij = 2**1.5 * g_ij
            Delta_eff_ij = Delta_ij + alpha_i

            val = abs(Omega_eff_ij / Delta_eff_ij)
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label_ij],
                    f"|2√2g/(Δ+α)| of {label_ij} ({val:.3g}) is higher than {adiabatic_limit} (2√2g={Omega_eff_ij * 1e3:.0f} MHz, Δ+α={Delta_eff_ij * 1e3:.0f} MHz).",
                )

        for i, nnn in self.next_nearest_neighbors.items():
            for k in nnn:
                label_ik = self.get_label((i, k))
                omega_i = self.get_frequency(i)
                omega_k = self.get_frequency(k)
                alpha_i = self.get_anharmonicity(i)
                Delta_ik = omega_i - omega_k
                g_ik = self.params.default_nnn_coupling
                Omega_eff_ik = 2**1.5 * g_ik
                Delta_eff_ik = Delta_ik + alpha_i

                val = abs(Omega_eff_ik / Delta_eff_ik)
                if val > adiabatic_limit:
                    self.add_invalid_edges(
                        [label_ik],
                        f"|2√2g/(Δ+α)| of {label_ik} ({val:.3g}) is higher than {adiabatic_limit} (2√2g={Omega_eff_ik * 1e3:.0f} MHz, Δ+α={Delta_eff_ik * 1e3:.0f} MHz).",
                    )


class Type3B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type3B",
            description="CR cause e-f",
            graph=graph,
            params=params,
        )

    def execute(self):
        cr_control_limit = self.params.cr_control_limit

        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR
            Omega_eff_ij = 2**0.5 * Omega_d_ij
            Delta_eff_ij = Delta_ij + alpha_i

            val = abs(Omega_eff_ij / Delta_eff_ij)
            if val > cr_control_limit:
                self.add_invalid_edges(
                    [label_ij],
                    f"|√2Ω_d/(Δ+α)| of {label_ij} ({val:.3g}) is higher than {cr_control_limit} (Ω_d={Omega_d_ij * 1e3:.0f} MHz, Δ+α={Delta_eff_ij * 1e3:.0f} MHz).",
                )


class Type7(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type7",
            description="CR cause fogi with spectators",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR

            for k in self.nearest_neighbors[i]:
                if k == j:
                    continue

                label_ik = self.get_label((i, k))
                omega_k = self.get_frequency(k)
                g_ik = self.get_nn_coupling((i, k))
                Delta_ik = omega_i - omega_k
                Omega_eff_ik = (
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
                Delta_eff_ik = 2 * Delta_ik + alpha_i

                val = abs(Omega_eff_ik / Delta_eff_ik)
                if val > adiabatic_limit:
                    self.add_invalid_edges(
                        [label_ik],
                        f"|Ω_eff/(2Δ+α)| of {label_ij} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={Omega_eff_ik * 1e3:.0f} MHz, 2Δ+α={Delta_eff_ik * 1e3:.0f} MHz).",
                    )


class Type8(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type8",
            description="ge-ge too close with Stark shift",
            graph=graph,
            params=params,
        )

    def execute(self):
        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR
            Delta_stark = (
                Omega_d_ij**2 * alpha_i / (2 * Delta_ij * (Delta_ij + alpha_i))
            )

            for k in self.nearest_neighbors[i]:
                if k == j:
                    continue

                omega_k = self.get_frequency(k)
                Delta_ik = omega_i - omega_k

                val = Delta_ik * (Delta_ik + Delta_stark)
                if val < 0:
                    self.add_invalid_edges(
                        [label_ij],
                        f"Δ(Δ+Δ_stark) of {label_ij} ({val:.3g}) is negative (Δ={Delta_ik * 1e3:.0f} MHz, Δ_stark={Delta_stark * 1e3:.0f} MHz).",
                    )

            for k in self.next_nearest_neighbors[i]:
                omega_k = self.get_frequency(k)
                Delta_ik = omega_i - omega_k

                val = Delta_ik * (Delta_ik + Delta_stark)

                if val < 0:
                    self.add_invalid_edges(
                        [label_ij],
                        f"Δ(Δ+Δ_stark) of {label_ij} ({val:.3g}) is negative (Δ={Delta_ik * 1e3:.0f} MHz, Δ_stark={Delta_stark * 1e3:.0f} MHz).",
                    )


class Type9(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type9",
            description="ge-ef too close with Stark shift",
            graph=graph,
            params=params,
        )

    def execute(self):
        for i, j in self.graph.qubit_edges:
            label_ij = self.get_label((i, j))
            omega_i = self.get_frequency(i)
            omega_j = self.get_frequency(j)
            alpha_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            Delta_ij = omega_i - omega_j
            Omega_CR = 1 / (self.params.cnot_time * 4)
            Omega_d_ij = Delta_ij * (Delta_ij + alpha_i) / (g_ij * alpha_i) * Omega_CR
            Delta_stark = (
                Omega_d_ij**2 * alpha_i / (2 * Delta_ij * (Delta_ij + alpha_i))
            )

            for k in self.nearest_neighbors[i]:
                if k == j:
                    continue

                omega_k = self.get_frequency(k)
                alpha_k = self.get_anharmonicity(k)
                Delta_ik = omega_i - omega_k

                val = (Delta_ik - alpha_k) * (Delta_ik - alpha_k + Delta_stark)
                if val < 0:
                    self.add_invalid_edges(
                        [label_ij],
                        f"(Δ-α)(Δ-α+Δ_stark) of {label_ij} ({val:.3g}) is negative (Δ={Delta_ik * 1e3:.0f} MHz, α={alpha_k * 1e3:.0f} MHz, Δ_stark={Delta_stark * 1e3:.0f} MHz).",
                    )

            for k in self.next_nearest_neighbors[i]:
                omega_k = self.get_frequency(k)
                alpha_k = self.get_anharmonicity(k)
                Delta_ik = omega_i - omega_k

                val = (Delta_ik - alpha_k) * (Delta_ik - alpha_k + Delta_stark)
                if val < 0:
                    self.add_invalid_edges(
                        [label_ij],
                        f"(Δ-α)(Δ-α+Δ_stark) of {label_ij} ({val:.3g}) is negative (Δ={Delta_ik * 1e3:.0f} MHz, α={alpha_k * 1e3:.0f} MHz, Δ_stark={Delta_stark * 1e3:.0f} MHz).",
                    )
