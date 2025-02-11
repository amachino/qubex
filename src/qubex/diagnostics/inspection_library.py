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
            f_ge_i = self.get_ge_frequency(i)
            t1_i = self.get_t1(i)
            t2_i = self.get_t2(i)

            if np.isnan(f_ge_i):
                self.add_invalid_nodes(
                    [label_i],
                    f"Frequency of {label_i} is not defined.",
                )
            if f_ge_i < min_frequency:
                self.add_invalid_nodes(
                    [label_i],
                    f"Frequency of {label_i} ({f_ge_i:.3f} GHz) is lower than {min_frequency:.3f} GHz.",
                )
            if f_ge_i > max_frequency:
                self.add_invalid_nodes(
                    [label_i],
                    f"Frequency of {label_i} ({f_ge_i:.3f} GHz) is higher than {max_frequency:.3f} GHz.",
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
            f_ge_i = self.get_ge_frequency(i)
            f_ge_j = self.get_ge_frequency(j)
            D_ij = f_ge_i - f_ge_j

            if abs(D_ij) > max_detuning:
                self.add_invalid_edges(
                    [label_ij],
                    f"Detuning of {label_ij} ({D_ij * 1e3:.0f} MHz) is higher than {max_detuning * 1e3:.0f} MHz.",
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
            g_ij = self.get_nn_coupling((i, j))
            D_ij = self.get_ge_ge_detuning((i, j))

            val = abs(2 * g_ij / D_ij)
            if val > adiabatic_limit:
                self.add_invalid_nodes(
                    [label_i, label_j],
                    f"|2g/Δ| of {label_ij} ({val:.3f}) is higher than {adiabatic_limit} (g={g_ij * 1e3:.0f} MHz, Δ={D_ij * 1e3:.0f} MHz).",
                )

        for i, nnn in self.next_nearest_neighbors.items():
            for k in nnn:
                if i > k:
                    continue

                label_ik = self.get_label((i, k))
                label_i = self.get_label(i)
                label_k = self.get_label(k)
                D_ik = self.get_ge_ge_detuning((i, k))
                g_ik = self.params.default_nnn_coupling

                val = abs(2 * g_ik / D_ik)
                if val > adiabatic_limit:
                    self.add_invalid_nodes(
                        [label_i, label_k],
                        f"|2g/Δ| of {label_ik} ({val:.3f}) is higher than {adiabatic_limit} (g={g_ik * 1e6:.0f} kHz, Δ={D_ik * 1e6:.0f} kHz).",
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

        for c in self.graph.qubit_nodes:
            for t in self.nearest_neighbors[c]:
                for s in self.nearest_neighbors[c]:
                    if t >= s:
                        continue

                    label_ts = self.get_label((t, s))
                    label_ct = self.get_label((c, t))
                    label_cs = self.get_label((c, s))
                    D_ts = self.get_ge_ge_detuning((t, s))
                    O_CR = self.get_cr_rabi_frequency((c, t))

                    val = abs(O_CR / D_ts)
                    if val > adiabatic_limit:
                        self.add_invalid_edges(
                            [label_ct, label_cs],
                            f"|Ω_CR/Δ| of {label_ts} ({val:.3f}) is higher than {adiabatic_limit} (Ω_CR={O_CR * 1e6:.0f} kHz, Δ={D_ts * 1e6:.0f} kHz).",
                        )


class Type1C(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type1C",
            description="CR causes g-e",
            graph=graph,
            params=params,
        )

    def execute(self):
        cr_control_limit = self.params.cr_control_limit

        for c, t in self.graph.qubit_edges:
            label_ct = self.get_label((c, t))
            D_ct = self.get_ge_ge_detuning((c, t))
            O_d_ct = self.get_cr_drive_frequency((c, t))

            val = abs(O_d_ct / D_ct)
            if val > cr_control_limit:
                self.add_invalid_edges(
                    [label_ct],
                    f"|Ω_d/Δ| of {label_ct} ({val:.3f}) is higher than {cr_control_limit} (Ω_d={O_d_ct * 1e3:.0f} MHz, Δ={D_ct * 1e3:.0f} MHz).",
                )


class Type2A(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type2A",
            description="CR causes g-f",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for c, t in self.graph.qubit_edges:
            label_ct = self.get_label((c, t))
            a_c = self.get_anharmonicity(c)
            D_ct = self.get_ge_ge_detuning((c, t))
            O_d_ct = self.get_cr_drive_frequency((c, t))
            O_eff_ct = 2 ** (-1.5) * O_d_ct**2 * (1 / (D_ct + a_c) - 1 / D_ct)
            D_eff_ct = 2 * D_ct + a_c

            val = abs(O_eff_ct / D_eff_ct)
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label_ct],
                    f"|Ω_eff/(2Δ+α)| of {label_ct} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={O_eff_ct * 1e3:.0f} MHz, 2Δ+α={D_eff_ct * 1e3:.0f} MHz).",
                )


class Type2B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type2B",
            description="CR causes fg-ge",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for c, t in self.graph.qubit_edges:
            label_ct = self.get_label((c, t))
            a_c = self.get_anharmonicity(c)
            g_ct = self.get_nn_coupling((c, t))
            D_ct = self.get_ge_ge_detuning((c, t))
            O_d_ct = self.get_cr_drive_frequency((c, t))
            O_eff_ct = 2**0.5 * O_d_ct * g_ct / (1 / (D_ct + a_c) - 1 / D_ct)
            D_eff_ct = 2 * D_ct + a_c

            val = abs(O_eff_ct / D_eff_ct)
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label_ct],
                    f"|Ω_eff/(2Δ+α)| of {label_ct} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={O_eff_ct * 1e3:.0f} MHz, 2Δ+α={D_eff_ct * 1e3:.0f} MHz).",
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
            a_i = self.get_anharmonicity(i)
            g_ij = self.get_nn_coupling((i, j))
            D_ij = self.get_ge_ge_detuning((i, j))
            O_eff_ij = 2**1.5 * g_ij
            D_eff_ij = D_ij + a_i

            val = abs(O_eff_ij / D_eff_ij)
            if val > adiabatic_limit:
                self.add_invalid_edges(
                    [label_ij],
                    f"|2√2g/(Δ+α)| of {label_ij} ({val:.3g}) is higher than {adiabatic_limit} (2√2g={O_eff_ij * 1e3:.0f} MHz, Δ+α={D_eff_ij * 1e3:.0f} MHz).",
                )

        for i, nnn in self.next_nearest_neighbors.items():
            for k in nnn:
                label_ik = self.get_label((i, k))
                a_i = self.get_anharmonicity(i)
                D_ik = self.get_ge_ge_detuning((i, k))
                g_ik = self.params.default_nnn_coupling
                O_eff_ik = 2**1.5 * g_ik
                D_eff_ik = D_ik + a_i

                val = abs(O_eff_ik / D_eff_ik)
                if val > adiabatic_limit:
                    self.add_invalid_edges(
                        [label_ik],
                        f"|2√2g/(Δ+α)| of {label_ik} ({val:.3g}) is higher than {adiabatic_limit} (2√2g={O_eff_ik * 1e3:.0f} MHz, Δ+α={D_eff_ik * 1e3:.0f} MHz).",
                    )


class Type3B(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type3B",
            description="CR causes e-f",
            graph=graph,
            params=params,
        )

    def execute(self):
        cr_control_limit = self.params.cr_control_limit

        for c, t in self.graph.qubit_edges:
            label_ct = self.get_label((c, t))
            a_c = self.get_anharmonicity(c)
            D_ct = self.get_ge_ge_detuning((c, t))
            O_d_ct = self.get_cr_drive_frequency((c, t))
            O_eff_ct = 2**0.5 * O_d_ct
            D_eff_ct = D_ct + a_c

            val = abs(O_eff_ct / D_eff_ct)
            if val > cr_control_limit:
                self.add_invalid_edges(
                    [label_ct],
                    f"|√2Ω_d/(Δ+α)| of {label_ct} ({val:.3g}) is higher than {cr_control_limit} (Ω_d={O_d_ct * 1e3:.0f} MHz, Δ+α={D_eff_ct * 1e3:.0f} MHz).",
                )


class Type7(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type7",
            description="CR causes fg-ge with spectators",
            graph=graph,
            params=params,
        )

    def execute(self):
        adiabatic_limit = self.params.adiabatic_limit

        for c, t in self.graph.qubit_edges:
            label_ct = self.get_label((c, t))
            a_c = self.get_anharmonicity(c)
            D_ct = self.get_ge_ge_detuning((c, t))
            O_d_ct = self.get_cr_drive_frequency((c, t))

            for s in self.nearest_neighbors[c]:
                if s == t:
                    continue

                label_ik = self.get_label((c, s))
                g_cs = self.get_nn_coupling((c, s))
                D_cs = self.get_ge_ge_detuning((c, s))
                O_eff_cs = (
                    2 ** (-0.5)
                    * g_cs
                    * O_d_ct
                    * (1 / (D_ct + a_c) - 1 / D_ct + 1 / (D_cs + a_c) - 1 / D_cs)
                )
                D_eff_cs = 2 * D_cs + a_c

                val = abs(O_eff_cs / D_eff_cs)
                if val > adiabatic_limit:
                    self.add_invalid_edges(
                        [label_ik],
                        f"|Ω_eff/(2Δ+α)| of {label_ct} ({val:.3g}) is higher than {adiabatic_limit} (Ω_eff={O_eff_cs * 1e3:.0f} MHz, 2Δ+α={D_eff_cs * 1e3:.0f} MHz).",
                    )


class Type8(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type8",
            description="ge and ge too close with Stark shift",
            graph=graph,
            params=params,
        )

    def execute(self):
        for c, t in self.graph.qubit_edges:
            label_ct = self.get_label((c, t))
            D_stark_c = self.get_stark_shift((c, t))

            for s in self.nearest_neighbors[c]:
                if s == t:
                    continue

                D_cs = self.get_ge_ge_detuning((c, s))

                val = D_cs * (D_cs + D_stark_c)
                if val < 0:
                    self.add_invalid_edges(
                        [label_ct],
                        f"Δ(Δ+Δ_stark) of {label_ct} ({val:.3g}) is negative (Δ={D_cs * 1e3:.0f} MHz, Δ_stark={D_stark_c * 1e3:.0f} MHz).",
                    )

            for s in self.next_nearest_neighbors[c]:
                D_cs = self.get_ge_ge_detuning((c, s))

                val = D_cs * (D_cs + D_stark_c)
                if val < 0:
                    self.add_invalid_edges(
                        [label_ct],
                        f"Δ(Δ+Δ_stark) of {label_ct} ({val:.3g}) is negative (Δ={D_cs * 1e3:.0f} MHz, Δ_stark={D_stark_c * 1e3:.0f} MHz).",
                    )


class Type9(Inspection):
    def __init__(
        self,
        graph: LatticeGraph,
        params: dict | None = None,
    ):
        super().__init__(
            name="Type9",
            description="ge and ef too close with Stark shift",
            graph=graph,
            params=params,
        )

    def execute(self):
        for c, t in self.graph.qubit_edges:
            label_ct = self.get_label((c, t))
            D_stark_c = self.get_stark_shift((c, t))

            for s in self.nearest_neighbors[c]:
                if s == t:
                    continue

                a_s = self.get_anharmonicity(s)
                D_cs = self.get_ge_ge_detuning((c, s))

                val = (D_cs - a_s) * (D_cs - a_s + D_stark_c)
                if val < 0:
                    self.add_invalid_edges(
                        [label_ct],
                        f"(Δ-α)(Δ-α+Δ_stark) of {label_ct} ({val:.3g}) is negative (Δ={D_cs * 1e3:.0f} MHz, α={a_s * 1e3:.0f} MHz, Δ_stark={D_stark_c * 1e3:.0f} MHz).",
                    )

            for s in self.next_nearest_neighbors[c]:
                a_s = self.get_anharmonicity(s)
                D_cs = self.get_ge_ge_detuning((c, s))

                val = (D_cs - a_s) * (D_cs - a_s + D_stark_c)
                if val < 0:
                    self.add_invalid_edges(
                        [label_ct],
                        f"(Δ-α)(Δ-α+Δ_stark) of {label_ct} ({val:.3g}) is negative (Δ={D_cs * 1e3:.0f} MHz, α={a_s * 1e3:.0f} MHz, Δ_stark={D_stark_c * 1e3:.0f} MHz).",
                    )
