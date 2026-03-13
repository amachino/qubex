"""Contributed helpers for multipartite entanglement workflows."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Collection
from datetime import datetime
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis.state_tomography import (
    create_density_matrix,
    mle_fit_density_matrix,
    plot_ghz_state_tomography,
)
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.library.graph import (
    find_longest_1d_chain,
    get_max_undirected_weight,
    strong_edge_coloring,
    tree_center,
)
from qubex.experiment.models import Result
from qubex.pulse import (
    CPMG,
    Blank,
    PulseSchedule,
    VirtualZ,
)
from qubex.visualization import COLORS

from ._deprecated_options import resolve_shot_options


def create_entangle_sequence(
    exp: Experiment,
    entangle_steps: Collection[tuple[str | int, str | int]],
    *,
    initialization_pulse: str | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
) -> PulseSchedule:
    """
    Create an entangling sequence from edge steps.

    Parameters
    ----------
    entangle_steps
        Directed edges defining entanglement order.
    optimize_sequence
        Whether to optimize ordering for depth.
    decouple_all_zz
        Whether to apply global dynamical decoupling.
    """
    if optimize_sequence is None:
        optimize_sequence = False
    if as_late_as_possible is None:
        as_late_as_possible = False
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = False
    if decouple_entangled_zz is None:
        decouple_entangled_zz = False
    if decouple_all_zz is None:
        decouple_all_zz = False
    if initialization_pulse is None:
        initialization_pulse = "Y90"
    sampling_period = exp.ctx.measurement.sampling_period

    steps: list[tuple[str, str]] = []
    qubits: list[str] = []
    G = nx.DiGraph()

    for parent, child in entangle_steps:
        if isinstance(parent, int):
            parent = exp.ctx.quantum_system.get_qubit(parent).label
        if isinstance(child, int):
            child = exp.ctx.quantum_system.get_qubit(child).label
        steps.append((parent, child))

        if parent not in qubits:
            qubits.append(parent)
        if child not in qubits:
            qubits.append(child)

        weight = int(exp.pulse.cnot(parent, child, only_low_to_high=True).duration)
        G.add_edge(parent, child, weight=weight)

    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    leaf_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    leaf_edges = [step for step in steps if step[1] in leaf_nodes]

    substeps: list[list[tuple[str, str]]] = []

    if optimize_sequence:
        path_lengths = {}
        for leaf in leaf_nodes:
            for root in roots:
                if not nx.has_path(G, root, leaf):
                    continue
                path = tuple(nx.shortest_path(G, source=root, target=leaf))
                length = sum(G[u][v]["weight"] for u, v in pairwise(path))
                path_lengths[path] = length

        sorted_paths = sorted(path_lengths.items(), key=lambda x: x[1], reverse=True)

        all_edges = []
        for path, _ in sorted_paths:
            substeps.append([])
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge not in all_edges:
                    substeps[-1].append(edge)
                all_edges.append(edge)
    else:
        substeps = [steps]

    # print(substeps)

    with PulseSchedule() as ps:
        for root in roots:
            if initialization_pulse == "Y90":
                ps.add(root, exp.pulse.y90(root))
            elif initialization_pulse == "X90":
                ps.add(root, exp.pulse.x90(root))
            elif initialization_pulse == "H":
                ps.add(root, exp.pulse.hadamard(root))
            else:
                raise ValueError(f"Invalid initialize pulse: {initialization_pulse}")
        ps.barrier()
        for steps in substeps:
            for parent, child in steps:
                cnot = exp.pulse.cnot(parent, child, only_low_to_high=True)
                ps.call(cnot)
                if decouple_cr_crosstalk:
                    if exp.ctx.qubits[parent].index % 4 in [0, 3]:
                        control_qubit = parent
                        target_qubit = child
                    else:
                        control_qubit = child
                        target_qubit = parent
                    cr_ranges = cnot.get_pulse_ranges()[
                        f"{control_qubit}-{target_qubit}"
                    ]
                    spectators = exp.ctx.get_spectators(control_qubit)
                    for spectator in spectators:
                        spec = spectator.label
                        pi = exp.pulse.x180(spec)
                        if spec in ps.labels and spec != target_qubit:
                            cr_start = ps.get_offset(target_qubit) - cnot.duration
                            spec_end = ps.get_offset(spec)
                            space_before_cr = cr_start - spec_end
                            if space_before_cr >= 0:
                                ps.add(spec, Blank(space_before_cr))
                                blank1 = cr_ranges[0].stop * 2
                                ps.add(
                                    spec,
                                    Blank(blank1),
                                )
                                ps.add(spec, pi)
                                blank2 = (
                                    cr_ranges[1].stop - cr_ranges[0].stop
                                ) * 2 - pi.duration
                                ps.add(
                                    spec,
                                    Blank(blank2),
                                )
                                ps.add(spec, pi)

        if as_late_as_possible:
            # Put final CNOT gates as late as possible
            max_duration = ps._max_offset()  # noqa
            for leaf_edge in leaf_edges:
                if exp.ctx.qubits[leaf_edge[0]].index % 4 in [0, 3]:
                    control_label = leaf_edge[0]
                    target_label = leaf_edge[1]
                else:
                    control_label = leaf_edge[1]
                    target_label = leaf_edge[0]
                cr_label = f"{control_label}-{target_label}"
                if ps._offsets[control_label] == ps._offsets[target_label]:  # noqa
                    offset = ps._offsets[control_label]  # noqa
                    if offset < max_duration:
                        blank = max_duration - offset
                        for label in [control_label, target_label, cr_label]:
                            ps._channels[label].sequence._elements.insert(  # noqa
                                -1, Blank(duration=blank)
                            )
                            ps._offsets[label] += blank  # noqa

        if decouple_entangled_zz:
            # Apply CPMG to blanks after entanglement gates
            for qubit in qubits:
                if exp.ctx.qubits[qubit].index % 4 in [0, 3]:
                    dd_duration = ps._max_offset() - ps._offsets[qubit]  # noqa
                    pi = exp.pulse.x180(qubit)
                    if cpmg_duration_unit is None:
                        n_pi = 2
                        duration_unit = pi.duration * 10
                    else:
                        n_pi = 2 * int(dd_duration // cpmg_duration_unit)
                        duration_unit = cpmg_duration_unit
                    if dd_duration > duration_unit:
                        tau = (dd_duration - pi.duration * n_pi) // (2 * n_pi)
                        tau = (tau // sampling_period) * sampling_period
                        ps.add(qubit, CPMG(tau=tau, pi=pi, n=n_pi, alternating=False))

    if decouple_all_zz:
        # Apply CPMG to all blanks in the sequence
        with PulseSchedule() as ps_dd:
            for target, pulse_array in ps.get_sequences().items():
                for element in pulse_array.elements:
                    if isinstance(element, Blank) and target in exp.ctx.qubits:
                        if exp.ctx.qubits[target].index % 4 in [0, 3]:
                            dd_duration = element.duration
                            pi = exp.pulse.x180(target)
                            if cpmg_duration_unit is None:
                                n_pi = 2
                                duration_unit = pi.duration * 10
                            else:
                                n_pi = 2 * int(dd_duration // cpmg_duration_unit)
                                duration_unit = cpmg_duration_unit
                            if dd_duration > duration_unit:
                                tau = (dd_duration - pi.duration * n_pi) // (2 * n_pi)
                                tau = (tau // sampling_period) * sampling_period
                                ps_dd.add(target, CPMG(tau=tau, pi=pi, n=n_pi))
                                continue
                    ps_dd.add(target, element)

        seq = ps_dd
    else:
        seq = ps

    return seq


def create_ghz_sequence(
    exp: Experiment,
    entangle_steps: Collection[tuple[str | int, str | int]],
    *,
    initialization_pulse: str | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
) -> PulseSchedule:
    """
    Create a GHZ state preparation sequence based on the entanglement steps.

    Returns a PulseSchedule object.
    """
    if optimize_sequence is None:
        optimize_sequence = True
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = True
    if decouple_entangled_zz is None:
        decouple_entangled_zz = True
    if decouple_all_zz is None:
        decouple_all_zz = False
    steps: list[tuple[str, str]] = []

    for parent, child in entangle_steps:
        if isinstance(parent, int):
            parent = exp.ctx.quantum_system.get_qubit(parent).label
        if isinstance(child, int):
            child = exp.ctx.quantum_system.get_qubit(child).label
        steps.append((parent, child))

    qubits: list[str] = [steps[0][0]]
    for parent, child in steps:
        if parent not in qubits:
            raise ValueError(
                f"All qubits for GHZ state must branch from the first qubit: {qubits[0]}"
            )
        if child not in qubits:
            qubits.append(child)

    return create_entangle_sequence(
        exp,
        entangle_steps=steps,
        initialization_pulse=initialization_pulse,
        optimize_sequence=optimize_sequence,
        as_late_as_possible=as_late_as_possible,
        decouple_cr_crosstalk=decouple_cr_crosstalk,
        decouple_entangled_zz=decouple_entangled_zz,
        decouple_all_zz=decouple_all_zz,
        cpmg_duration_unit=cpmg_duration_unit,
    )


def measure_ghz_state(
    exp: Experiment,
    entangle_steps: Collection[tuple[str | int, str | int]],
    *,
    measurement_bases: Collection[str] | None = None,
    initialization_pulse: str | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure the n-qubit GHZ state in the specified bases.

    Returns dict with 'raw', 'mitigated', 'result', 'figure'.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="measure_ghz_state",
    )
    if optimize_sequence is None:
        optimize_sequence = True
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = True
    if decouple_entangled_zz is None:
        decouple_entangled_zz = True
    if decouple_all_zz is None:
        decouple_all_zz = False
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if exp.ctx.state_centers is None:
        exp.build_classifier(plot=False)

    steps: list[tuple[str, str]] = []
    qubits: list[str] = []

    for parent, child in entangle_steps:
        if isinstance(parent, int):
            parent = exp.ctx.quantum_system.get_qubit(parent).label
        if isinstance(child, int):
            child = exp.ctx.quantum_system.get_qubit(child).label
        steps.append((parent, child))

        if parent not in qubits:
            qubits.append(parent)
        if child not in qubits:
            qubits.append(child)

    n_qubits = len(qubits)

    if measurement_bases is None:
        measurement_bases = ["Z"] * n_qubits
    else:
        measurement_bases = list(measurement_bases)

    seq = create_ghz_sequence(
        exp,
        entangle_steps=steps,
        initialization_pulse=initialization_pulse,
        optimize_sequence=optimize_sequence,
        as_late_as_possible=as_late_as_possible,
        decouple_cr_crosstalk=decouple_cr_crosstalk,
        decouple_entangled_zz=decouple_entangled_zz,
        decouple_all_zz=decouple_all_zz,
        cpmg_duration_unit=cpmg_duration_unit,
    )

    with PulseSchedule() as ps:
        ps.call(seq)
        for qb, basis in zip(qubits, measurement_bases, strict=True):
            if basis == "X":
                ps.add(qb, exp.pulse.y90m(qb))
            elif basis == "Y":
                ps.add(qb, exp.pulse.x90(qb))

    result = exp.measure(
        ps,
        mode="single",
        n_shots=n_shots,
        shot_interval=shot_interval,
    )

    basis_labels = result.get_basis_labels(qubits)
    prob_dict_raw = result.get_probabilities(qubits)
    prob_dict_raw = {label: prob_dict_raw.get(label, 0) for label in basis_labels}
    prob_dict_mitigated = result.get_mitigated_probabilities(qubits)

    labels = [f"|{i}⟩" for i in prob_dict_raw]
    prob_arr_raw = np.array(list(prob_dict_raw.values()))
    prob_arr_mitigated = np.array(list(prob_dict_mitigated.values()))

    fig = viz.make_figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=prob_arr_raw,
            name="Raw",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=prob_arr_mitigated,
            name="Mitigated",
        )
    )
    fig.update_layout(
        title=f"GHZ state measurement: {'-'.join(qubits)}",
        xaxis_title=f"State ({''.join(measurement_bases)} basis)",
        yaxis_title="Probability",
        barmode="group",
        yaxis_range=[0, 1],
    )
    if plot:
        ps.plot(title=f"GHZ state measurement: {''.join(measurement_bases)} basis")
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )
        for label, p, mp in zip(
            labels,
            prob_arr_raw,
            prob_arr_mitigated,
            strict=True,
        ):
            print(f"{label} : {p:.2%} -> {mp:.2%}")
    if save_image:
        viz.save_figure(
            fig,
            f"ghz_state_measurement_{'-'.join(qubits)}",
        )

    return Result(
        data={
            "raw": prob_arr_raw,
            "mitigated": prob_arr_mitigated,
            "result": result,
            "figure": fig,
        },
        figure=fig,
    )


def ghz_state_tomography(
    exp: Experiment,
    entangle_steps: Collection[tuple[str | int, str | int]],
    *,
    readout_mitigation: bool | None = None,
    initialization_pulse: str | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    show_sequence: bool | None = None,
    save_image: bool | None = None,
    mle_fit: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Perform full state tomography on a n-qubit GHZ state.

    This involves:
    1. Measuring the GHZ state in all 3^n Pauli bases.
    2. Calculating the expectation values for all 4^n Pauli strings.
    3. Reconstructing the 2^n x 2^n density matrix using linear inversion or MLE.
    4. Calculating the fidelity with the ideal GHZ state.
    5. Plotting the resulting density matrix.

    Parameters
    ----------
    entangle_steps : list[tuple[str, str]]
        List of tuples representing the entanglement steps, e.g., [("Q00", "Q01"), ("Q01", "Q02")].
    readout_mitigation : bool
        Whether to apply readout error mitigation.
    n_shots : int
        Number of shots for each measurement.
    shot_interval : float
        Time interval between measurements.
    plot : bool
        Whether to plot the resulting density matrix.
    save_image : bool
        Whether to save the plot as an image.
    mle_fit : bool
        Whether to use Maximum Likelihood Estimation (MLE) for density matrix reconstruction.

    Returns
    -------
    dict
        A dictionary containing:
        - "probabilities": Measured probabilities in all bases.
        - "expected_values": Calculated expectation values for all Pauli strings.
        - "density_matrix": Reconstructed density matrix.
        - "fidelity": Fidelity with the ideal GHZ state.
        - "figure": Plotly figure of the density matrix.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="ghz_state_tomography",
    )
    if readout_mitigation is None:
        readout_mitigation = True
    if optimize_sequence is None:
        optimize_sequence = True
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = True
    if decouple_entangled_zz is None:
        decouple_entangled_zz = True
    if decouple_all_zz is None:
        decouple_all_zz = False
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if show_sequence is None:
        show_sequence = True
    if save_image is None:
        save_image = True
    if mle_fit is None:
        mle_fit = True

    qubits: list[str] = []
    steps: list[tuple[str, str]] = []

    for parent, child in entangle_steps:
        if isinstance(parent, int):
            parent = exp.ctx.quantum_system.get_qubit(parent).label
        if isinstance(child, int):
            child = exp.ctx.quantum_system.get_qubit(child).label
        steps.append((parent, child))

        if parent not in qubits:
            qubits.append(parent)
        if child not in qubits:
            qubits.append(child)

    n_qubits = len(qubits)
    dim = 2**n_qubits

    if show_sequence:
        seq = create_ghz_sequence(
            exp,
            entangle_steps=steps,
            initialization_pulse=initialization_pulse,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
        )
        seq.plot(title=f"GHZ state preparation sequence : {'-'.join(qubits)}")

    probs_raw = {}
    probs_mit = {}
    for measurement_bases in tqdm(
        product(["X", "Y", "Z"], repeat=n_qubits),
        total=3**n_qubits,
        desc="Measuring GHZ state in all bases",
    ):
        basis_label = "".join(measurement_bases)
        result = measure_ghz_state(
            exp,
            entangle_steps=steps,
            measurement_bases=measurement_bases,
            initialization_pulse=initialization_pulse,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            save_image=False,
        )
        probs_raw[basis_label] = result["raw"]
        if readout_mitigation:
            probs_mit[basis_label] = result["mitigated"]

    ghz_state = np.zeros((dim, 1), dtype=np.complex128)
    ghz_state[0, 0] = 1 / np.sqrt(2)
    ghz_state[-1, 0] = 1 / np.sqrt(2)

    rho_raw = create_density_matrix(probs_raw)
    fidelity_raw = float(np.real(ghz_state.T.conj() @ rho_raw @ ghz_state))

    if readout_mitigation:
        rho_mit = create_density_matrix(probs_mit, mle_fit=False)
        fidelity_mit = float(np.real(ghz_state.T.conj() @ rho_mit @ ghz_state))

        if mle_fit:
            rho_mle = create_density_matrix(probs_mit, mle_fit=True)
            fidelity_mle = float(np.real(ghz_state.T.conj() @ rho_mle @ ghz_state))

    width, height = 800, 455

    fig_raw = plot_ghz_state_tomography(
        rho=rho_raw,
        qubits=qubits,
        fidelity=fidelity_raw,
        plot=False,
        save_image=save_image,
        width=width,
        height=height,
        file_name=f"ghz_state_tomography_raw_{'-'.join(qubits)}",
    )["figure"]

    if readout_mitigation:
        fig_mit = plot_ghz_state_tomography(
            rho=rho_mit,
            qubits=qubits,
            fidelity=fidelity_mit,
            plot=False,
            save_image=save_image,
            width=width,
            height=height,
            file_name=f"ghz_state_tomography_mit_{'-'.join(qubits)}",
        )["figure"]

        if mle_fit:
            fig_mle = plot_ghz_state_tomography(
                rho=rho_mle,
                qubits=qubits,
                fidelity=fidelity_mle,
                plot=False,
                save_image=save_image,
                width=width,
                height=height,
                file_name=f"ghz_state_tomography_mle_{'-'.join(qubits)}",
            )["figure"]

    if plot:
        if not readout_mitigation:
            fig_raw.show()
        elif mle_fit:
            fig_mle.show()
        else:
            fig_mit.show()

    result = {
        "raw": {
            "probabilities": probs_raw,
            "density_matrix": rho_raw,
            "fidelity": fidelity_raw,
            "figure": fig_raw,
        },
    }
    if readout_mitigation:
        result["mitigated"] = {
            "probabilities": probs_mit,
            "density_matrix": rho_mit,
            "fidelity": fidelity_mit,
            "figure": fig_mit,
        }
        if mle_fit:
            result["mle"] = {
                "probabilities": probs_mit,
                "density_matrix": rho_mle,
                "fidelity": fidelity_mle,
                "figure": fig_mle,
            }

    figures = {"raw": fig_raw}
    primary_figure = fig_raw
    if readout_mitigation:
        figures["mitigated"] = fig_mit
        primary_figure = fig_mit
        if mle_fit:
            figures["mle"] = fig_mle
            primary_figure = fig_mle

    return Result(
        data=result,
        figure=primary_figure,
        figures=figures,
    )


def create_mqc_sequence(
    exp: Experiment,
    entangle_steps: Collection[tuple[str | int, str | int]],
    *,
    phi: float | None = None,
    echo: bool | None = None,
    initialization_pulse: str | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
) -> PulseSchedule:
    """
    Create an MQC sequence with a variable phase.

    Parameters
    ----------
    entangle_steps
        Entanglement edges defining the preparation.
    phi
        Phase applied as a virtual Z rotation.
    echo
        Whether to insert echo pi pulses.
    """
    if phi is None:
        phi = 0.0
    if echo is None:
        echo = True
    if optimize_sequence is None:
        optimize_sequence = True
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = False
    if decouple_entangled_zz is None:
        decouple_entangled_zz = True
    if decouple_all_zz is None:
        decouple_all_zz = False
    qubits: list[str] = []
    steps: list[tuple[str, str]] = []

    for parent, child in entangle_steps:
        if isinstance(parent, int):
            parent = exp.ctx.quantum_system.get_qubit(parent).label
        if isinstance(child, int):
            child = exp.ctx.quantum_system.get_qubit(child).label
        steps.append((parent, child))

        if parent not in qubits:
            qubits.append(parent)
        if child not in qubits:
            qubits.append(child)

    ghz_seq = create_entangle_sequence(
        exp,
        entangle_steps=steps,
        initialization_pulse=initialization_pulse,
        optimize_sequence=optimize_sequence,
        as_late_as_possible=as_late_as_possible,
        decouple_cr_crosstalk=decouple_cr_crosstalk,
        decouple_entangled_zz=decouple_entangled_zz,
        decouple_all_zz=decouple_all_zz,
        cpmg_duration_unit=cpmg_duration_unit,
    )

    with PulseSchedule() as seq:
        seq.call(ghz_seq)
        if echo:
            for qubit in qubits:
                seq.add(qubit, exp.pulse.x180(qubit))
        seq.barrier()
        for target in ghz_seq.get_targets():
            seq.add(target, VirtualZ(phi))
        seq.call(ghz_seq.inverted())
    return seq


def mqc_experiment(
    exp: Experiment,
    entangle_steps: Collection[tuple[str | int, str | int]],
    *,
    phi_range: np.ndarray | None = None,
    n_points_per_qubit: int | None = None,
    show_sequence: bool | None = None,
    echo: bool | None = None,
    initialization_pulse: str | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run an MQC experiment over a phase sweep.

    Parameters
    ----------
    entangle_steps
        Entanglement edges defining the preparation.
    phi_range
        Phase values to sweep.
    show_sequence
        Whether to display the pulse sequence.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="mqc_experiment",
    )
    if show_sequence is None:
        show_sequence = True
    if echo is None:
        echo = True
    if optimize_sequence is None:
        optimize_sequence = True
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = False
    if decouple_entangled_zz is None:
        decouple_entangled_zz = True
    if decouple_all_zz is None:
        decouple_all_zz = False
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    qubits: list[str] = []
    source_qubits: list[str] = []
    steps: list[tuple[str, str]] = []

    for parent, child in entangle_steps:
        if isinstance(parent, int):
            parent = exp.ctx.quantum_system.get_qubit(parent).label
        if isinstance(child, int):
            child = exp.ctx.quantum_system.get_qubit(child).label
        steps.append((parent, child))

        if parent not in qubits:
            source_qubits.append(parent)
            qubits.append(parent)
        if child not in qubits:
            qubits.append(child)

    n_qubits = len(qubits)

    if phi_range is None:
        if n_points_per_qubit is None:
            n_points_per_qubit = 6
        phi_range = np.linspace(0, 2 * np.pi, n_points_per_qubit * n_qubits + 1)

    if show_sequence:
        seq = create_mqc_sequence(
            exp,
            entangle_steps=steps,
            phi=0.0,
            echo=echo,
            initialization_pulse=initialization_pulse,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
        )
        seq.plot(title=f"{n_qubits}-qubits entanglement sequence")

    result = exp.sweep_parameter(
        lambda phi: create_mqc_sequence(
            exp,
            entangle_steps=entangle_steps,
            phi=phi,
            echo=echo,
            initialization_pulse=initialization_pulse,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
        ),
        plot=False,
        enable_tqdm=True,
        sweep_range=phi_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
    )

    for qubit, data in result.data.items():
        title = f"Measured signal : {qubit}"
        if qubit in source_qubits:
            title += " (source qubit)"
        fig = data.plot(
            normalize=True,
            title=title,
            xlabel="Z rotation : φ (rad)",
            return_figure=True,
        )
        if fig is None:
            raise RuntimeError("Expected figure from plot() when return_figure=True.")
        viz.save_figure(
            fig,
            name=f"mqc_n{n_qubits}_{qubit}",
            format="png",
        )
        viz.save_figure(
            fig,
            name=f"mqc_n{n_qubits}_{qubit}",
            format="svg",
        )

    coherences = {}
    for source_qubit in source_qubits:
        fourier_result = fourier_analysis(
            exp,
            result.data[source_qubit].data,
            qubit=source_qubit,
            title=f"Fourier analysis of {n_qubits}-qubits entanglement : {source_qubit}",
        )
        coherences[source_qubit] = fourier_result["C"]

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        os.makedirs("./data", exist_ok=True)

        np.savez(
            f"./data/{timestamp}_mqc_n{n_qubits}_raw.npz",
            result.data[source_qubit].data,
        )
        np.savez(
            f"./data/{timestamp}_mqc_n{n_qubits}_normalized.npz",
            result.data[source_qubit].normalized,
        )

    return Result(
        data={
            "phi_range": phi_range,
            "result": result,
            "coherences": coherences,
        }
    )


@staticmethod
def fourier_analysis(
    data: ArrayLike,
    *,
    qubit: str | None = None,
    title: str | None = None,
) -> Result:
    """
    Perform Fourier analysis on normalized data.

    Parameters
    ----------
    data
        Input signal to analyze.
    qubit
        Optional qubit label used for file naming.
    """
    if title is None:
        title = "Fourier analysis"
    data = np.asarray(data)

    S = (data + 1) / 2
    N = len(S)
    F = np.fft.fft(S)

    q = np.arange(N // 2)[1:]
    I = np.abs(F[1 : N // 2]) / N
    C = 2 * np.sqrt(I)

    fig = viz.make_figure()

    fig.add_trace(
        go.Bar(
            x=q,
            y=C,
            name="Amplitude",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Fourier modes",
        yaxis_title="Amplitude",
    )

    fig.show(
        config={
            "toImageButtonOptions": {
                "format": "png",
                "scale": 3,
            },
        }
    )

    file_name = f"fourier_analysis_{qubit}" if qubit else "fourier_analysis"

    viz.save_figure(
        fig,
        name=file_name,
        format="png",
    )

    viz.save_figure(
        fig,
        name=file_name,
        format="svg",
    )

    return Result(
        data={
            "figure": fig,
            "I": I,
            "C": C,
        },
        figure=fig,
    )


def parity_oscillation(
    exp: Experiment,
    entangle_steps: Collection[tuple[str | int, str | int]],
    *,
    phi_range: np.ndarray | None = None,
    n_points_per_qubit: int | None = None,
    show_sequence: bool | None = None,
    show_only_qubit_channels: bool | None = None,
    initialization_pulse: str | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
    readout_mitigation: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure parity oscillations of an entangled state.

    Parameters
    ----------
    entangle_steps
        Entanglement edges defining the preparation.
    phi_range
        Phase values to sweep.
    readout_mitigation
        Whether to apply readout mitigation.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="parity_oscillation",
    )
    if show_sequence is None:
        show_sequence = True
    if show_only_qubit_channels is None:
        show_only_qubit_channels = False
    if optimize_sequence is None:
        optimize_sequence = True
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = False
    if decouple_entangled_zz is None:
        decouple_entangled_zz = True
    if decouple_all_zz is None:
        decouple_all_zz = False
    if readout_mitigation is None:
        readout_mitigation = True
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if initialization_pulse is None:
        initialization_pulse = "Y90"

    qubits: list[str] = []
    source_qubits: list[str] = []
    steps: list[tuple[str, str]] = []

    for parent, child in entangle_steps:
        if isinstance(parent, int):
            parent = exp.ctx.quantum_system.get_qubit(parent).label
        if isinstance(child, int):
            child = exp.ctx.quantum_system.get_qubit(child).label
        steps.append((parent, child))

        if parent not in qubits:
            source_qubits.append(parent)
            qubits.append(parent)
        if child not in qubits:
            qubits.append(child)

    n_qubits = len(qubits)

    print(f"qubits: {qubits}")

    if phi_range is None:
        if n_points_per_qubit is None:
            n_points_per_qubit = 6
        phi_range = np.linspace(0, 2 * np.pi, n_points_per_qubit * n_qubits + 1)

    ghz_seq = create_entangle_sequence(
        exp,
        entangle_steps=steps,
        initialization_pulse=initialization_pulse,
        optimize_sequence=optimize_sequence,
        as_late_as_possible=as_late_as_possible,
        decouple_cr_crosstalk=decouple_cr_crosstalk,
        decouple_entangled_zz=decouple_entangled_zz,
        decouple_all_zz=decouple_all_zz,
        cpmg_duration_unit=cpmg_duration_unit,
    )

    def sequence(phi: float) -> PulseSchedule:
        with PulseSchedule() as seq:
            seq.call(ghz_seq)
            rz = VirtualZ(phi)
            for label in seq.labels:
                if label in exp.ctx.qubit_labels:
                    seq.add(label, rz)
                    if initialization_pulse == "Y90":
                        seq.add(label, exp.pulse.y90m(label))
                    elif initialization_pulse == "X90":
                        seq.add(label, exp.pulse.x90m(label))
                    elif initialization_pulse == "H":
                        seq.add(label, exp.pulse.hadamard(label))
                    else:
                        raise ValueError(
                            f"Invalid initialize pulse: {initialization_pulse}"
                        )
        return seq

    if show_sequence:
        seq_plot = sequence(0.0)
        if show_only_qubit_channels:
            for label in seq_plot.labels:
                if label not in exp.ctx.qubits:
                    del seq_plot._channels[label]  # noqa
        seq_plot.plot(
            title=f"{n_qubits}-qubits entanglement sequence",
            show_physical_pulse=False,
        )

    parities_raw = []
    parities_mit = []
    result = []
    for phi in tqdm(phi_range):
        res = exp.measure(
            sequence(phi),
            mode="single",
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        result.append(res)
        probs_raw = res.probabilities
        parity_raw = 0
        for label, prob in probs_raw.items():
            parity_raw += prob * (1 if label.count("1") % 2 == 0 else -1)
        parities_raw.append(parity_raw)

        if readout_mitigation:
            probs_mit = res.mitigated_probabilities
            parity_mit = 0
            for label, prob in probs_mit.items():
                parity_mit += prob * (1 if label.count("1") % 2 == 0 else -1)
            parities_mit.append(parity_mit)

    fig = viz.make_figure()
    fig.update_layout(
        title=f"Parity oscillation : {n_qubits}-qubit GHZ state",
        xaxis_title="Z rotation : φ (rad)",
        yaxis_title="Parity",
        yaxis_range=(-1.1, 1.1),
    )

    fig.add_scatter(
        x=phi_range,
        y=parities_raw,
        mode="lines+markers",
        name="Raw",
        marker=dict(size=5),
    )
    if readout_mitigation:
        fig.add_scatter(
            x=phi_range,
            y=parities_mit,
            mode="lines+markers",
            name="Mitigated",
            marker=dict(size=5),
        )
    fig.show(
        config={
            "toImageButtonOptions": {
                "format": "png",
                "scale": 3,
            },
        }
    )

    viz.save_figure(
        fig,
        name=f"parity_oscillation_n{n_qubits}",
        format="png",
    )
    viz.save_figure(
        fig,
        name=f"parity_oscillation_n{n_qubits}",
        format="svg",
    )

    fourier_analysis(
        exp,
        parities_raw if not readout_mitigation else parities_mit,
        title=f"Fourier analysis of {n_qubits}-qubit parity oscillation",
    )

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    os.makedirs("./data", exist_ok=True)

    np.savez(
        f"./data/{timestamp}_parity_oscillation_n{n_qubits}.npz",
        parities_raw,
    )

    return Result(
        data={
            "phi_range": phi_range,
            "result": result,
            "parities_raw": parities_raw,
            "parities_mit": parities_mit,
        }
    )


def create_1d_cluster_sequence(
    exp: Experiment,
    targets: Collection[str | int],
    *,
    bases: dict[int, str] | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
    with_readout_pulses: bool | None = None,
) -> PulseSchedule:
    """
    Create a 1D cluster state preparation sequence.

    Parameters
    ----------
    targets
        Target qubits to include in the chain.
    bases
        Measurement bases per position.
    with_readout_pulses
        Whether to append readout pulses.
    """
    if optimize_sequence is None:
        optimize_sequence = False
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = False
    if decouple_entangled_zz is None:
        decouple_entangled_zz = False
    if decouple_all_zz is None:
        decouple_all_zz = False
    if with_readout_pulses is None:
        with_readout_pulses = True
    targets = [
        exp.ctx.quantum_system.get_qubit(target).label
        if isinstance(target, int)
        else target
        for target in targets
    ]
    if bases is None:
        bases = {}

    qubits = [
        {
            "index": i,
            "label": label,
            "type": "L" if exp.ctx.qubits[label].index % 4 in [0, 3] else "H",
            "basis": bases.get(i, "Z"),
        }
        for i, label in enumerate(targets)
    ]
    n_qubits = len(qubits)

    l1_edges = []
    l1_max_duration = 0
    for i in range(n_qubits - 1):
        if qubits[i]["type"] == "L":
            edge = (qubits[i]["label"], qubits[i + 1]["label"])
            l1_edges.append(edge)
            cnot = exp.pulse.cnot(*edge, only_low_to_high=True)
            l1_max_duration = max(l1_max_duration, cnot.duration)

    l2_edges = []
    l2_max_duration = 0
    for i in range(n_qubits - 1):
        if qubits[i]["type"] == "H":
            edge = (qubits[i + 1]["label"], qubits[i]["label"])
            l2_edges.append(edge)
            cnot = exp.pulse.cnot(*edge, only_low_to_high=True)
            l2_max_duration = max(l2_max_duration, cnot.duration)

    with PulseSchedule(targets) as ps:
        for edge in l1_edges:
            with PulseSchedule() as l1:
                h = exp.pulse.hadamard(edge[0])
                l1.add(edge[0], h)
                l1.call(exp.pulse.cnot(*edge, only_low_to_high=True))
            l1.pad(
                total_duration=l1_max_duration + h.duration,
                pad_side="left",
            )
            ps.call(l1)

        for edge in l2_edges:
            with PulseSchedule() as l2:
                l2.call(exp.pulse.cnot(*edge, only_low_to_high=True))
                h = exp.pulse.hadamard(edge[1])
                l2.add(edge[1], h)
            ps.call(l2)

        # debug: no entanglement, just Hadamard gates
        # for target in targets:
        #     ps.add(target, exp.pulse.hadamard(target))

        for qubit in qubits:
            basis = qubit["basis"]
            if basis == "X":
                ps.add(qubit["label"], exp.pulse.y90m(qubit["label"]))
            elif basis == "Y":
                ps.add(qubit["label"], exp.pulse.x90(qubit["label"]))
            elif basis == "Z":
                pass
            else:
                raise ValueError(f"Unknown basis: {basis}")

        if with_readout_pulses:
            for qubit in qubits:
                resonator = exp.ctx.resonators[qubit["label"]].label
                ps.add(resonator, Blank(ps.get_offset(qubit["label"])))
                ps.add(resonator, exp.pulse.readout(resonator))
    return ps

    # cluster_seq = create_entangle_sequence(exp,
    #     entangle_steps=entangle_steps,
    #     initialization_pulse=initialization_pulse,
    #     optimize_sequence=optimize_sequence,
    #     as_late_as_possible=as_late_as_possible,
    #     decouple_cr_crosstalk=decouple_cr_crosstalk,
    #     decouple_entangled_zz=decouple_entangled_zz,
    #     decouple_all_zz=decouple_all_zz,
    #     cpmg_duration_unit=cpmg_duration_unit,
    # )

    # with PulseSchedule(targets) as ps:
    #     ps.call(cluster_seq)
    #     for qubit in qubits:
    #         if qubit["type"] == "H":
    #             ps.add(qubit["label"], exp.pulse.hadamard(qubit["label"]))
    #     for qubit in qubits:
    #         basis = qubit["basis"]
    #         if basis == "X":
    #             ps.add(qubit["label"], exp.pulse.y90m(qubit["label"]))
    #         elif basis == "Y":
    #             ps.add(qubit["label"], exp.pulse.x90(qubit["label"]))
    #         elif basis == "Z":
    #             pass
    #         else:
    #             raise ValueError(f"Unknown basis: {basis}")
    # return ps


def _measure_1d_cluster_state(
    exp: Experiment,
    targets: Collection[str | int],
    *,
    offset: int | None = None,
    mle_fit: bool | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    method: str | None = None,
    reset_awg_and_capunits: bool | None = None,
):
    """
    Measure 1D cluster state edges for a given offset.

    Parameters
    ----------
    targets
        Target qubits in the chain.
    offset
        Offset for selecting measured edges.
    method
        Measurement method to use.
    """
    if offset is None:
        offset = 0
    if mle_fit is None:
        mle_fit = True
    if optimize_sequence is None:
        optimize_sequence = False
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = False
    if decouple_entangled_zz is None:
        decouple_entangled_zz = False
    if decouple_all_zz is None:
        decouple_all_zz = False
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if method is None:
        method = "execute"
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    targets = [
        exp.ctx.quantum_system.get_qubit(target).label
        if isinstance(target, int)
        else target
        for target in targets
    ]
    n_qubits = len(targets)

    if offset > 2:
        raise ValueError("Offset must be 0, 1, or 2 for 1D cluster state measurement.")

    edges: dict[tuple[str, str], list[str]] = {}
    n_edges = n_qubits // 3 + 1
    for i in range(n_edges):
        if offset + i * 3 + 1 >= len(targets):
            break
        edge = (targets[offset + i * 3], targets[offset + i * 3 + 1])
        edge_spectators: list[str] = []
        for node in edge:
            node_spectators = exp.ctx.get_spectators(node)
            for spectator in node_spectators:
                node_index = targets.index(node)
                if spectator.label in targets:
                    spectator_index = targets.index(spectator.label)
                    is_adjacent = abs(node_index - spectator_index) == 1
                    if is_adjacent and spectator.label not in edge:
                        edge_spectators.append(spectator.label)
        edges[edge] = edge_spectators
        if plot:
            print(f"Edge: {edge}, Spectators: {edge_spectators}")

    seq = create_1d_cluster_sequence(
        exp,
        targets,
        optimize_sequence=optimize_sequence,
        as_late_as_possible=as_late_as_possible,
        decouple_cr_crosstalk=decouple_cr_crosstalk,
        decouple_entangled_zz=decouple_entangled_zz,
        decouple_all_zz=decouple_all_zz,
        cpmg_duration_unit=cpmg_duration_unit,
        with_readout_pulses=method == "execute",
    )
    if plot:
        seq.plot()

    if reset_awg_and_capunits:
        qubits = {exp.ctx.resolve_qubit_label(target) for target in seq.labels}
        exp.ctx.reset_awg_and_capunits(qubits=qubits)

    edge_sbits_result: dict[tuple[str, str], dict[str, dict]] = {
        edge: {} for edge in edges
    }
    edge_sbits_probabilities: dict[
        tuple[str, str], dict[str, dict[str, dict[str, float]]]
    ] = {
        # edge: {
        #     sbits: {
        #         pauli: {
        #             ebits: probability,
        #         }
        #     },
        # }
        edge: {}
        for edge in edges
    }

    for pauli0, pauli1 in tqdm(
        product(["X", "Y", "Z"], repeat=2),
    ):
        pauli_basis = f"{pauli0}{pauli1}"

        bases = {}
        for node0, node1 in edges:
            idx0 = targets.index(node0)
            idx1 = targets.index(node1)
            bases[idx0] = pauli0
            bases[idx1] = pauli1

        if method == "execute":
            result = exp.execute(
                create_1d_cluster_sequence(
                    exp,
                    targets,
                    bases=bases,
                    optimize_sequence=optimize_sequence,
                    as_late_as_possible=as_late_as_possible,
                    decouple_cr_crosstalk=decouple_cr_crosstalk,
                    decouple_entangled_zz=decouple_entangled_zz,
                    decouple_all_zz=decouple_all_zz,
                    cpmg_duration_unit=cpmg_duration_unit,
                    with_readout_pulses=True,
                ),
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                reset_awg_and_capunits=False,
            )
        else:
            result = exp.measure(
                create_1d_cluster_sequence(
                    exp,
                    targets,
                    bases=bases,
                    optimize_sequence=optimize_sequence,
                    as_late_as_possible=as_late_as_possible,
                    decouple_cr_crosstalk=decouple_cr_crosstalk,
                    decouple_entangled_zz=decouple_entangled_zz,
                    decouple_all_zz=decouple_all_zz,
                    cpmg_duration_unit=cpmg_duration_unit,
                    with_readout_pulses=False,
                ),
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                reset_awg_and_capunits=False,
            )

        for edge, spectators in edges.items():
            target_labels = list(edge) + spectators
            mitigated_counts = result.get_mitigated_counts(target_labels)
            n_spectators = len(spectators)
            spectators_bits = [
                "".join(bits) for bits in product("01", repeat=n_spectators)
            ]
            for sbits in spectators_bits:
                if sbits not in edge_sbits_probabilities[edge]:
                    edge_sbits_probabilities[edge][sbits] = {}

            sbits_ebits_counts: dict[str, dict[str, int]] = {
                sbits: {} for sbits in spectators_bits
            }

            for bits, count in mitigated_counts.items():
                ebits = bits[:2]
                sbits = bits[2:]
                sbits_ebits_counts[sbits][ebits] = count

            for sbits, ebits_counts in sbits_ebits_counts.items():
                total_count = sum(ebits_counts.values())
                edge_sbits_probabilities[edge][sbits][pauli_basis] = {
                    ebits: count / total_count if total_count > 0 else 0.0
                    for ebits, count in ebits_counts.items()
                }

    paulis = {
        "I": np.array([[1, 0], [0, 1]]),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }

    for edge, sbits_probabilities in edge_sbits_probabilities.items():
        for sbits, probabilities in sbits_probabilities.items():
            if sbits not in edge_sbits_result[edge]:
                edge_sbits_result[edge][sbits] = {}

            expected_values = {}
            rho = np.zeros((4, 4), dtype=np.complex128)
            for basis0, pauli0 in paulis.items():
                for basis1, pauli1 in paulis.items():
                    pauli_basis = f"{basis0}{basis1}"
                    # calculate the expectation values
                    if pauli_basis == "II":
                        # II is always 1
                        # 00: +1, 01: +1, 10: +1, 11: +1
                        p = probabilities["ZZ"]
                        e = p["00"] + p["01"] + p["10"] + p["11"]
                    elif pauli_basis in ["IX", "IY", "IZ"]:
                        # ignore the first qubit
                        # 00: +1, 01: -1, 10: +1, 11: -1
                        p = probabilities[f"Z{basis1}"]
                        e = p["00"] - p["01"] + p["10"] - p["11"]
                    elif pauli_basis in ["XI", "YI", "ZI"]:
                        # ignore the second qubit
                        # 00: +1, 01: +1, 10: -1, 11: -1
                        p = probabilities[f"{basis0}Z"]
                        e = p["00"] + p["01"] - p["10"] - p["11"]
                    else:
                        # two-qubit basis
                        # 00: +1, 01: -1, 10: -1, 11: +1
                        p = probabilities[pauli_basis]
                        e = p["00"] - p["01"] - p["10"] + p["11"]
                    pauli_matrix = np.kron(pauli0, pauli1)
                    rho += e * pauli_matrix
                    expected_values[pauli_basis] = e

            if mle_fit:
                rho = mle_fit_density_matrix(expected_values)
            else:
                rho = rho / 4

            rho_pt = partial_transpose(exp, rho)
            eigvals = np.linalg.eigvalsh(rho_pt)
            negativity = np.sum(np.abs(eigvals[eigvals < 0]))

            if plot:
                print(f"{edge[0]}-{edge[1]} ({sbits}) : Negativity = {negativity}")

            fig = viz.make_figure()
            fig.set_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Abs", "Phase"),
                horizontal_spacing=0.26,
            )
            fig.add_trace(
                go.Heatmap(
                    z=np.abs(rho),
                    zmin=0,
                    zmax=1,
                    colorscale="Hot_r",
                    colorbar=dict(
                        title="Abs",
                        x=0.37,
                        y=0.5,
                        thickness=15,
                        tickvals=[0, 0.5, 1],
                        ticktext=["0", "0.5", "1"],
                    ),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=np.angle(rho),
                    zmin=-np.pi,
                    zmax=np.pi,
                    colorscale="Edge",
                    colorbar=dict(
                        title="Phase (rad)",
                        x=1.0,
                        y=0.5,
                        thickness=15,
                        tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                        ticktext=["-π", "-π/2", "0", "π/2", "π"],
                    ),
                ),
                row=1,
                col=2,
            )

            tickvals = np.arange(4)
            ticktext = [f"{i:0{2}b}" for i in tickvals]
            tick_style = dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=0,
            )

            fig.update_xaxes(tick_style, row=1, col=1)
            fig.update_yaxes(
                dict(**tick_style, autorange="reversed", scaleanchor="x1"),
                row=1,
                col=1,
            )
            fig.update_xaxes(tick_style, row=1, col=2)
            fig.update_yaxes(
                dict(**tick_style, autorange="reversed", scaleanchor="x2"),
                row=1,
                col=2,
            )
            fig.update_layout(
                title=dict(
                    text=f"Negativity of graph state: 𝒩 = {negativity:.3f}",
                    subtitle=dict(
                        text=f"edge: {edge}, spectators: ({', '.join(edges[edge])}) = '{sbits}'",
                    ),
                ),
                margin=dict(t=110),
                width=600,
                height=342,
            )

            if plot:
                fig.show(
                    config={
                        "toImageButtonOptions": {
                            "format": "png",
                            "scale": 3,
                        },
                    }
                )

            edge_sbits_result[edge][sbits]["expected_values"] = expected_values
            edge_sbits_result[edge][sbits]["density_matrix"] = rho
            edge_sbits_result[edge][sbits]["partial_transpose"] = rho_pt
            edge_sbits_result[edge][sbits]["negativity"] = negativity
            edge_sbits_result[edge][sbits]["eigenvalues"] = eigvals
            edge_sbits_result[edge][sbits]["figure"] = fig

    result = {"best": {edge: {} for edge in edges}}

    for edge, sbits_results in edge_sbits_result.items():
        best_result = max(
            sbits_results.values(),
            key=lambda x: x.get("negativity", 0),
        )
        result["best"][edge] = best_result

    result["all"] = edge_sbits_result

    return result


def measure_1d_cluster_state(
    exp: Experiment,
    qubits: Collection[str | int],
    *,
    mle_fit: bool | None = None,
    optimize_sequence: bool | None = None,
    as_late_as_possible: bool | None = None,
    decouple_cr_crosstalk: bool | None = None,
    decouple_entangled_zz: bool | None = None,
    decouple_all_zz: bool | None = None,
    cpmg_duration_unit: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    method: str | None = None,
    reset_awg_and_capunits: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure negativities for a 1D cluster state.

    Parameters
    ----------
    qubits
        Target qubits in the chain.
    mle_fit
        Whether to use MLE for density matrix reconstruction.
    method
        Measurement method to use.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="measure_1d_cluster_state",
    )
    if mle_fit is None:
        mle_fit = True
    if optimize_sequence is None:
        optimize_sequence = False
    if as_late_as_possible is None:
        as_late_as_possible = True
    if decouple_cr_crosstalk is None:
        decouple_cr_crosstalk = False
    if decouple_entangled_zz is None:
        decouple_entangled_zz = False
    if decouple_all_zz is None:
        decouple_all_zz = False
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if method is None:
        method = "execute"
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True

    if plot:
        seq = create_1d_cluster_sequence(
            exp,
            qubits,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
            with_readout_pulses=False,
        )
        seq.plot(
            title=f"1D cluster state preparation sequence for {len(qubits)} qubits",
        )

    negativities = {}
    figures = {}
    for offset in range(3):
        print(f"[{offset + 1}/3] Measuring edges with offset {offset}")
        result = _measure_1d_cluster_state(
            exp,
            qubits,
            offset=offset,
            mle_fit=mle_fit,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            method=method,
            reset_awg_and_capunits=reset_awg_and_capunits,
        )
        for edge, data in result["best"].items():
            negativities[edge] = data["negativity"]
            figures[edge] = data["figure"]

    negativities = dict(
        sorted(negativities.items(), key=lambda item: item[1], reverse=False)
    )

    negativities_max = max(negativities.values())
    negativities_min = min(negativities.values())
    negativities_avg = np.mean(list(negativities.values()))
    negativities_std = np.std(list(negativities.values()))
    negativities_med = np.median(list(negativities.values()))
    if plot:
        for fig in figures.values():
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )
        print(f"Negativities of {len(negativities)} edges:")
        print(f"  max: {negativities_max:.3f}")
        print(f"  min: {negativities_min:.3f}")
        print(f"  med: {negativities_med:.3f}")
        print(f"  avg: {negativities_avg:.3f}")
        print(f"  std: {negativities_std:.3f}")
        print("Negativities per edge:")
        for edge, negativity in negativities.items():
            print(f"  {edge[0]}-{edge[1]}: {negativity:.3f}")

        x = [f"{edge[0]}-{edge[1]}" for edge in negativities]
        y = list(negativities.values())
        fig = viz.make_figure()
        fig.update_layout(
            title=f"Negativities of {len(qubits)}-qubit 1D cluster state",
            xaxis=dict(
                title="Edges",
                tickangle=45,
                tickmode="array",
                tickvals=list(range(len(x))),
                ticktext=x,
            ),
            yaxis=dict(
                title="Negativity",
                range=[0, 0.55],
                tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                ticktext=["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
            ),
            width=800,
            height=400,
            margin=dict(l=70, r=70, t=90, b=100),
        )
        fig.add_bar(x=x, y=y)
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

    return Result(
        data={
            "negativities_max": negativities_max,
            "negativities_min": negativities_min,
            "negativities_med": negativities_med,
            "negativities_avg": negativities_avg,
            "negativities_std": negativities_std,
            "negativities": negativities,
            "figures": figures,
        },
        figures=figures,
    )


@staticmethod
def partial_transpose(rho: NDArray, subsystem: int | None = None) -> NDArray:
    """
    Perform partial transpose on a 2-qubit density matrix.

    Parameters
    ----------
    rho : NDArray
        2-qubit density matrix, reshaped as a 4x4 array.

    subsystem : int
        Subsystem to transpose (0 for first qubit, 1 for second qubit).


    Returns
    -------
    NDArray
        Partially transposed density matrix, reshaped as a 4x4 array.

    """
    if subsystem is None:
        subsystem = 1
    rho_tensor = rho.reshape(2, 2, 2, 2)  # (iA, iB, jA, jB)

    if subsystem == 0:
        # (iA, iB, jA, jB) → (jA, iB, iA, jB)
        rho_pt = np.transpose(rho_tensor, (2, 1, 0, 3))
    elif subsystem == 1:
        # (iA, iB, jA, jB) → (iA, jB, jA, iB)
        rho_pt = np.transpose(rho_tensor, (0, 3, 2, 1))
    else:
        raise ValueError("subsystem must be 0 or 1")

    return rho_pt.reshape(4, 4)


def create_connected_graphs(
    exp: Experiment,
    fidelities: dict[str, float] | None = None,
    *,
    t1: dict[str, float] | None = None,
    t2_echo: dict[str, float] | None = None,
    threshold: float | None = None,
    plot: bool | None = None,
    show_labels: bool | None = None,
    show_data: bool | None = None,
) -> list[nx.DiGraph]:
    """
    Build connected subgraphs from edge fidelities.

    Parameters
    ----------
    fidelities
        Edge fidelities keyed by CR label.
    threshold
        Minimum fidelity threshold.
    plot
        Whether to visualize the graphs.
    """
    if threshold is None:
        threshold = 0.0
    if plot is None:
        plot = False
    if show_labels is None:
        show_labels = False
    if show_data is None:
        show_data = True
    if fidelities is None:
        fidelities = exp.ctx.load_property("bell_state_fidelity")
    if fidelities is None:
        fidelities = {}

    if t1 is None:
        t1 = {}
    if t2_echo is None:
        t2_echo = {}

    G = nx.DiGraph()
    cr_labels = exp.ctx.cr_labels
    for cr_label, fidelity in fidelities.items():
        if cr_label in cr_labels:
            if fidelity > threshold:
                control, target = cr_label.split("-")

                if not G.has_node(control):
                    G.add_node(
                        control,
                        t1=t1.get(control),
                        t2_echo=t2_echo.get(control),
                    )
                if not G.has_node(target):
                    G.add_node(
                        target,
                        t1=t1.get(target),
                        t2_echo=t2_echo.get(target),
                    )

                G.add_edge(
                    control,
                    target,
                    fidelity=fidelity,
                    cost=-np.log10(fidelity),
                )

    graphs = []
    for component in nx.weakly_connected_components(G):
        graph = G.subgraph(component)
        graphs.append(graph)
    graphs.sort(key=lambda x: x.number_of_nodes(), reverse=True)

    if plot:
        max_n = max(graph.number_of_nodes() for graph in graphs)
        visualize_graph(
            exp,
            G,
            title=f"Connected graphs : N (max) = {max_n}",
            show_labels=show_labels,
            show_data=show_data,
        )
    return graphs


def create_maximum_graph(
    exp: Experiment,
    fidelities: dict[str, float] | None = None,
    *,
    threshold: float | None = None,
    plot: bool | None = None,
    show_labels: bool | None = None,
    show_data: bool | None = None,
) -> nx.Graph:
    """
    Return the largest connected graph by node count.

    Parameters
    ----------
    fidelities
        Edge fidelities keyed by CR label.
    threshold
        Minimum fidelity threshold.
    """
    if threshold is None:
        threshold = 0.0
    if plot is None:
        plot = False
    if show_labels is None:
        show_labels = False
    if show_data is None:
        show_data = True
    if fidelities is None:
        fidelities = exp.ctx.load_property("bell_state_fidelity")

    graphs = create_connected_graphs(
        exp,
        fidelities=fidelities,
        threshold=threshold,
        plot=False,
    )
    if not graphs:
        raise ValueError("No connected graphs found")

    G = graphs[0]
    if plot:
        visualize_graph(
            exp,
            G,
            title=f"Maximum graph : N = {G.number_of_nodes()}",
            show_labels=show_labels,
            show_data=show_data,
        )

    return G


def create_maximum_1d_chain(
    exp: Experiment,
    fidelities: dict[str, float] | None = None,
    *,
    threshold: float | None = None,
    plot: bool | None = None,
    show_labels: bool | None = None,
    show_data: bool | None = None,
) -> nx.Graph:
    """
    Create the maximum 1D chain in a 2D lattice graph.

    Parameters
    ----------
    fidelities : dict[str, float]
        A dictionary mapping edge labels to their fidelities.

    Returns
    -------
    nx.Graph
        A graph representing the maximum 1D chain.
    """
    if threshold is None:
        threshold = 0.0
    if plot is None:
        plot = False
    if show_labels is None:
        show_labels = False
    if show_data is None:
        show_data = True
    if fidelities is None:
        fidelities = exp.ctx.load_property("bell_state_fidelity")

    graphs = create_connected_graphs(
        exp,
        fidelities,
        threshold=threshold,
        plot=False,
    )

    if not graphs:
        raise ValueError("No connected graphs found")

    G = graphs[0]
    path_nodes, path_edges, _ = find_longest_1d_chain(G)

    chain = nx.Graph()
    chain.add_nodes_from(path_nodes)
    for edge in path_edges:
        fidelity = get_max_undirected_weight(G, edge=edge, property="fidelity")
        chain.add_edge(*edge, fidelity=fidelity)

    if plot:
        visualize_graph(
            exp,
            chain,
            title=f"Maximum 1D chain : N = {len(path_nodes)}",
            show_labels=show_labels,
            show_data=show_data,
        )

    return chain


def create_maximum_spanning_tree(
    exp: Experiment,
    fidelities: dict[str, float] | None = None,
    *,
    threshold: float | None = None,
    t1: dict[str, float] | None = None,
    t2_echo: dict[str, float] | None = None,
    plot: bool | None = None,
    show_labels: bool | None = None,
    show_data: bool | None = None,
) -> nx.Graph:
    """
    Create a maximum spanning tree from fidelities.

    Parameters
    ----------
    fidelities
        Edge fidelities keyed by CR label.
    threshold
        Minimum fidelity threshold.
    """
    if threshold is None:
        threshold = 0.0
    if plot is None:
        plot = False
    if show_labels is None:
        show_labels = False
    if show_data is None:
        show_data = False
    if fidelities is None:
        fidelities = exp.ctx.load_property("bell_state_fidelity")

    graphs = create_connected_graphs(
        exp,
        fidelities,
        threshold=threshold,
        t1=t1,
        t2_echo=t2_echo,
        plot=False,
    )

    if not graphs:
        raise ValueError("No connected graphs found")

    G = graphs[0]
    UG = G.to_undirected()
    mst = nx.minimum_spanning_tree(UG, weight="cost")

    if plot:
        visualize_graph(
            exp,
            mst,
            title=f"Maximum spanning tree : N = {len(mst.nodes())}",
            show_labels=show_labels,
            show_data=show_data,
        )
    return mst


def create_maximum_directed_tree(
    exp: Experiment,
    fidelities: dict[str, float] | None = None,
    *,
    root: str | None = None,
    max_depth: int | None = None,
    max_node: int | None = None,
    threshold: float | None = None,
    t1: dict[str, float] | None = None,
    t2_echo: dict[str, float] | None = None,
    plot: bool | None = None,
    show_labels: bool | None = None,
    show_data: bool | None = None,
) -> nx.DiGraph:
    """
    Create a directed tree rooted at a specified node.

    Parameters
    ----------
    root
        Root node label.
    max_depth
        Maximum depth of the directed tree.
    max_node
        Maximum number of nodes to include.
    """
    if threshold is None:
        threshold = 0.0
    if plot is None:
        plot = False
    if show_labels is None:
        show_labels = False
    if show_data is None:
        show_data = True
    if fidelities is None:
        fidelities = exp.ctx.load_property("bell_state_fidelity")

    mst = create_maximum_spanning_tree(
        exp,
        fidelities,
        threshold=threshold,
        t1=t1,
        t2_echo=t2_echo,
        plot=False,
    )
    if root is None:
        root_qubit = str(tree_center(mst)[0])
    else:
        root_qubit = root

    # BFS to determine parents and depth from the root
    parents: dict[str, str | None] = {root_qubit: None}
    depths: dict[str, int] = {root_qubit: 0}
    q = deque([root_qubit])

    n_node = 1
    while q:
        if max_node is not None and n_node >= max_node:
            break
        u = q.popleft()
        if max_depth is not None and depths[u] >= max_depth:
            continue
        for v in mst.neighbors(u):
            if v in parents:
                continue
            parents[v] = u
            depths[v] = depths[u] + 1
            q.append(v)
            n_node += 1
            if max_node is not None and n_node >= max_node:
                break

    DG = nx.DiGraph()
    for child, parent in parents.items():
        # Carry over existing node attributes (if any) and store depth
        node_attrs = dict(mst.nodes[child]) if child in mst.nodes else {}
        node_attrs["depth"] = depths.get(child, 0)
        DG.add_node(child, **node_attrs)
        if parent is None:
            continue
        DG.add_edge(parent, child, **mst[parent][child])

    max_depth = max(depths.values(), default=0)

    if plot:
        visualize_graph(
            exp,
            DG,
            title=f"Maximum directed tree : N = {len(DG.nodes())}, root = {root_qubit}, depth = {max_depth}",
            show_labels=show_labels,
            show_data=show_data,
        )

    return DG


def create_cz_rounds(
    exp: Experiment,
    graph: nx.Graph,
    *,
    plot: bool | None = None,
) -> list[list[tuple[str, str]]]:
    """
    Group CZ edges into parallelizable rounds.

    Parameters
    ----------
    graph
        Graph defining CZ edges.
    plot
        Whether to visualize the rounds.
    """
    if plot is None:
        plot = False
    edges = list(graph.edges())
    edges_remaining = edges.copy()
    rounds: list[list[tuple[str, str]]] = []
    while edges_remaining:
        used: set[str] = set()
        batch: list[tuple[str, str]] = []
        for u, v in edges_remaining:
            if u not in used and v not in used:
                batch.append((u, v))
                used.add(u)
                used.add(v)
        # Remove scheduled edges and append this round
        edges_remaining = [e for e in edges_remaining if e not in batch]
        rounds.append(batch)

    chip_graph = exp.ctx.quantum_system.chip_graph
    if plot:
        for round_idx, round in enumerate(rounds):
            graph = nx.Graph()
            for u, v in round:
                graph.add_node(u)
                graph.add_node(v)
                graph.add_edge(u, v)

            node_values = dict.fromkeys(graph.nodes(), 1.0)
            edge_values = {f"{u}-{v}": 1.0 for u, v in graph.edges()}
            edge_overlay_values = {f"{u}-{v}": 1.0 for u, v in graph.edges()}

            chip_graph.plot_graph_data(
                directed=False,
                title=f"CZ round : {round_idx}",
                edge_values=edge_values,
                edge_color="#eef",
                edge_overlay=True,
                edge_overlay_values=edge_overlay_values,
                edge_overlay_color="turquoise",
                node_color="white",
                node_linecolor="ghostwhite",
                node_textcolor="ghostwhite",
                node_overlay=True,
                node_overlay_values=node_values,
                node_overlay_color="ghostwhite",
                node_overlay_linecolor="black",
                node_overlay_textcolor="black",
            )
    return rounds


def create_graph_sequence(
    exp: Experiment,
    graph: nx.Graph,
    *,
    bases: dict[str, str] | None = None,
    with_readout_pulses: bool | None = None,
) -> PulseSchedule:
    """
    Create a graph-state preparation sequence.

    Parameters
    ----------
    graph
        Graph defining CZ edges.
    bases
        Measurement bases for each node.
    with_readout_pulses
        Whether to append readout pulses.
    """
    if with_readout_pulses is None:
        with_readout_pulses = True
    nodes = list(graph.nodes())
    rounds = create_cz_rounds(exp, graph, plot=False)

    with PulseSchedule(nodes) as ps:
        # Prepare qubits in |+> with Hadamards (can run in parallel)
        for node in nodes:
            ps.add(node, exp.pulse.hadamard(node))

        # Apply CZ gates round by round so edges within a round run in parallel
        for batch in rounds:
            for u, v in batch:
                ps.call(exp.pulse.cz(u, v, only_low_to_high=True))

        # debug: no entanglement, just Hadamard gates
        # for node in nodes:
        #     ps.add(node, exp.pulse.hadamard(node))

        # Basis rotations prior to readout
        for node in nodes:
            basis = bases[node] if bases and node in bases else "Z"
            if basis == "X":
                ps.add(node, exp.pulse.y90m(node))
            elif basis == "Y":
                ps.add(node, exp.pulse.x90(node))
            elif basis == "Z":
                pass
            else:
                raise ValueError(f"Unknown basis: {basis}")

        if with_readout_pulses:
            for node in nodes:
                resonator = exp.ctx.resonators[node].label
                ps.add(resonator, Blank(ps.get_offset(node)))
                ps.add(resonator, exp.pulse.readout(resonator))
    return ps


def create_measurement_rounds(
    exp: Experiment,
    G: nx.Graph,
    plot: bool | None = None,
) -> dict[int, list[tuple[str, str]]]:
    """
    Color graph edges into measurement rounds.

    Parameters
    ----------
    G
        Graph defining the edges to color.
    plot
        Whether to visualize the rounds.
    """
    if plot is None:
        plot = False
    chip_graph = exp.ctx.quantum_system.chip_graph
    colored_edges = strong_edge_coloring(G)
    if plot:
        for color, edges in colored_edges.items():
            graph = nx.Graph()
            for u, v in edges:
                graph.add_node(u)
                graph.add_node(v)
                graph.add_edge(u, v, color=color)

            node_values = dict.fromkeys(G.nodes(), 1.0)
            edge_values = {f"{u}-{v}": 1.0 for u, v in G.edges()}
            edge_overlay_values = {f"{u}-{v}": 1.0 for u, v in graph.edges()}

            chip_graph.plot_graph_data(
                directed=False,
                title=f"Measurement round : {color}",
                edge_values=edge_values,
                edge_color="#eef",
                edge_overlay=True,
                edge_overlay_values=edge_overlay_values,
                edge_overlay_color="turquoise",
                node_color="white",
                node_linecolor="ghostwhite",
                node_textcolor="ghostwhite",
                node_overlay=True,
                node_overlay_values=node_values,
                node_overlay_color="ghostwhite",
                node_overlay_linecolor="black",
                node_overlay_textcolor="black",
            )
    return colored_edges


def _measure_graph_state(
    exp: Experiment,
    graph: nx.Graph,
    *,
    target_edges: list[tuple[str, str]],
    mle_fit: bool | None = None,
    use_all_spectator_pattern: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    method: str | None = None,
    reset_awg_and_capunits: bool | None = None,
    n_bootstrap: int | None = None,
    bootstrap_mle: bool | None = None,
):
    """
    Measure graph-state properties for specified edges.

    Parameters
    ----------
    graph
        Graph defining the state to measure.
    target_edges
        Edges to evaluate for negativity.
    mle_fit
        Whether to use MLE reconstruction.
    """
    if mle_fit is None:
        mle_fit = True
    if use_all_spectator_pattern is None:
        use_all_spectator_pattern = True
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if method is None:
        method = "execute"
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if bootstrap_mle is None:
        bootstrap_mle = False
    graph = graph.to_undirected()

    seq = create_graph_sequence(
        exp,
        graph=graph,
        with_readout_pulses=method == "execute",
    )
    if plot:
        seq.plot()

    if reset_awg_and_capunits:
        qubits = {exp.ctx.resolve_qubit_label(target) for target in seq.labels}
        exp.ctx.reset_awg_and_capunits(qubits=qubits)

    edge_and_spectators: dict[tuple[str, str], list[str]] = {}
    for edge in target_edges:
        edge_spectators: list[str] = []
        for node in edge:
            node_spectators = graph.neighbors(node)
            edge_spectators.extend(
                spectator for spectator in node_spectators if spectator not in edge
            )
        edge_and_spectators[edge] = edge_spectators
        if plot:
            print(f"Edge: {edge}, Spectators: {edge_spectators}")

    edge_sbits_result: dict[tuple[str, str], dict[str, dict]] = {
        edge: {} for edge in target_edges
    }

    edge_sbits_pauli_counts: dict[
        tuple[str, str], dict[str, dict[str, dict[str, int]]]
    ] = {
        # edge: {
        #     sbits: {
        #         pauli: {
        #             ebits: count,
        #         }
        #     },
        # }
        edge: {}
        for edge in target_edges
    }

    edge_sbits_pauli_probabilities: dict[
        tuple[str, str], dict[str, dict[str, dict[str, float]]]
    ] = {
        # edge: {
        #     sbits: {
        #         pauli: {
        #             ebits: probability,
        #         }
        #     },
        # }
        edge: {}
        for edge in target_edges
    }

    for pauli0, pauli1 in tqdm(
        product(["X", "Y", "Z"], repeat=2),
    ):
        pauli_basis = f"{pauli0}{pauli1}"

        bases = {}
        for node0, node1 in target_edges:
            bases[node0] = pauli0
            bases[node1] = pauli1

        if method == "execute":
            result = exp.execute(
                create_graph_sequence(
                    exp,
                    graph=graph,
                    bases=bases,
                    with_readout_pulses=True,
                ),
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                reset_awg_and_capunits=False,
            )
        else:
            result = exp.measure(
                create_graph_sequence(
                    exp,
                    graph=graph,
                    bases=bases,
                    with_readout_pulses=False,
                ),
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                reset_awg_and_capunits=False,
            )

        for edge, spectators in edge_and_spectators.items():
            target_labels = list(edge) + spectators
            mitigated_counts = result.get_mitigated_counts(target_labels)
            n_spectators = len(spectators)

            if use_all_spectator_pattern:
                spectators_bits = [
                    "".join(bits) for bits in product("01", repeat=n_spectators)
                ]
            else:
                spectators_bits = ["0" * n_spectators]

            for sbits in spectators_bits:
                if sbits not in edge_sbits_pauli_counts[edge]:
                    edge_sbits_pauli_counts[edge][sbits] = {}
                if sbits not in edge_sbits_pauli_probabilities[edge]:
                    edge_sbits_pauli_probabilities[edge][sbits] = {}

            sbits_counts: dict[str, dict[str, int]] = {
                sbits: {} for sbits in spectators_bits
            }

            for bits, count in mitigated_counts.items():
                ebits = bits[:2]
                sbits = bits[2:]
                if use_all_spectator_pattern:
                    sbits_counts[sbits][ebits] = count
                else:
                    if sbits == "0" * n_spectators:
                        sbits_counts[sbits][ebits] = count

            for sbits, counts in sbits_counts.items():
                edge_sbits_pauli_counts[edge][sbits][pauli_basis] = counts

                total_count = sum(counts.values())
                edge_sbits_pauli_probabilities[edge][sbits][pauli_basis] = {
                    ebits: count / total_count if total_count > 0 else 0.0
                    for ebits, count in counts.items()
                }

    paulis = {
        "I": np.array([[1, 0], [0, 1]]),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }

    def _compute_expected_values_and_rho_from_probs(
        probabilities: dict[str, dict[str, float]],
        use_mle: bool,
    ) -> tuple[dict[str, float], np.ndarray]:
        expected_values: dict[str, float] = {}
        rho_local = np.zeros((4, 4), dtype=np.complex128)
        for basis0, pauli0 in paulis.items():
            for basis1, pauli1 in paulis.items():
                pauli_basis = f"{basis0}{basis1}"
                if pauli_basis == "II":
                    p = probabilities["ZZ"]
                    e = p["00"] + p["01"] + p["10"] + p["11"]
                elif pauli_basis in ["IX", "IY", "IZ"]:
                    p = probabilities[f"Z{basis1}"]
                    e = p["00"] - p["01"] + p["10"] - p["11"]
                elif pauli_basis in ["XI", "YI", "ZI"]:
                    p = probabilities[f"{basis0}Z"]
                    e = p["00"] + p["01"] - p["10"] - p["11"]
                else:
                    p = probabilities[pauli_basis]
                    e = p["00"] - p["01"] - p["10"] + p["11"]
                pauli_matrix = np.kron(pauli0, pauli1)
                rho_local += e * pauli_matrix
                expected_values[pauli_basis] = e
        if use_mle:
            rho_local = mle_fit_density_matrix(expected_values)
        else:
            rho_local = rho_local / 4
        return expected_values, rho_local

    def _compute_negativity_from_probs(
        probabilities: dict[str, dict[str, float]],
        use_mle: bool,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        _, rho_local = _compute_expected_values_and_rho_from_probs(
            probabilities=probabilities,
            use_mle=use_mle,
        )
        rho_pt_local = partial_transpose(exp, rho_local)
        eigvals_local = np.linalg.eigvalsh(rho_pt_local)
        negativity_local = float(np.sum(np.abs(eigvals_local[eigvals_local < 0])))
        return negativity_local, rho_local, eigvals_local

    def _bootstrap_negativity(
        counts: dict[str, dict[str, int]],
        B: int,
        use_mle: bool,
    ) -> tuple[float, float, tuple[float, float]]:
        # Dirichlet bootstrap on the 4-outcome distributions for each Pauli basis
        rng = np.random.default_rng()
        samples: list[float] = []
        bases_keys = list(counts.keys())
        for _ in range(B):
            probs_b: dict[str, dict[str, float]] = {}
            for pb in bases_keys:
                c = counts[pb]
                # Order the 2-qubit outcome as 00,01,10,11
                vec = np.array(
                    [
                        max(c.get("00", 0.0), 0.0),
                        max(c.get("01", 0.0), 0.0),
                        max(c.get("10", 0.0), 0.0),
                        max(c.get("11", 0.0), 0.0),
                    ],
                    dtype=float,
                )
                total = vec.sum()
                if total <= 0:
                    vec = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
                else:
                    vec = vec / total
                # Dirichlet concentration ~ shots_eff * p
                shots_eff = int(total)
                alpha = vec * max(shots_eff, 1) + 1e-3
                vec_b = rng.dirichlet(alpha)
                probs_b[pb] = {
                    "00": float(vec_b[0]),
                    "01": float(vec_b[1]),
                    "10": float(vec_b[2]),
                    "11": float(vec_b[3]),
                }
            neg_b, _, _ = _compute_negativity_from_probs(
                probabilities=probs_b,
                use_mle=use_mle,
            )
            samples.append(neg_b)
        samples_arr = np.asarray(samples)
        mean_b = float(np.mean(samples_arr))
        std_b = float(np.std(samples_arr, ddof=1))
        lo, hi = np.percentile(samples_arr, [16.0, 84.0])
        return mean_b, std_b, (float(lo), float(hi))

    for edge, sbits_pauli_probabilities in edge_sbits_pauli_probabilities.items():
        for sbits, pauli_probabilities in sbits_pauli_probabilities.items():
            if sbits not in edge_sbits_result[edge]:
                edge_sbits_result[edge][sbits] = {}

            # Compute expected values and density matrix
            expected_values, rho = _compute_expected_values_and_rho_from_probs(
                probabilities=pauli_probabilities,
                use_mle=mle_fit,
            )

            rho_pt = partial_transpose(exp, rho)
            eigvals = np.linalg.eigvalsh(rho_pt)
            negativity = np.sum(np.abs(eigvals[eigvals < 0]))

            # Optional bootstrap error estimation
            if n_bootstrap is not None and n_bootstrap > 0:
                pauli_counts = edge_sbits_pauli_counts[edge][sbits]
                _, neg_std, (neg_lo, neg_hi) = _bootstrap_negativity(
                    counts=pauli_counts,
                    B=n_bootstrap,
                    use_mle=bootstrap_mle if mle_fit else False,
                )
            else:
                neg_std, neg_lo, neg_hi = None, None, None

            if plot:
                print(f"{edge[0]}-{edge[1]} ({sbits}) : Negativity = {negativity}")

            fig = viz.make_figure()
            fig.set_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Abs", "Phase"),
                horizontal_spacing=0.26,
            )
            fig.add_trace(
                go.Heatmap(
                    z=np.abs(rho),
                    zmin=0,
                    zmax=1,
                    colorscale="Hot_r",
                    colorbar=dict(
                        title="Abs",
                        x=0.37,
                        y=0.5,
                        thickness=15,
                        tickvals=[0, 0.5, 1],
                        ticktext=["0", "0.5", "1"],
                    ),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=np.angle(rho),
                    zmin=-np.pi,
                    zmax=np.pi,
                    colorscale="Edge",
                    colorbar=dict(
                        title="Phase (rad)",
                        x=1.0,
                        y=0.5,
                        thickness=15,
                        tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                        ticktext=["-π", "-π/2", "0", "π/2", "π"],
                    ),
                ),
                row=1,
                col=2,
            )

            tickvals = np.arange(4)
            ticktext = [f"{i:0{2}b}" for i in tickvals]
            tick_style = dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=0,
            )

            fig.update_xaxes(tick_style, row=1, col=1)
            fig.update_yaxes(
                dict(**tick_style, autorange="reversed", scaleanchor="x1"),
                row=1,
                col=1,
            )
            fig.update_xaxes(tick_style, row=1, col=2)
            fig.update_yaxes(
                dict(**tick_style, autorange="reversed", scaleanchor="x2"),
                row=1,
                col=2,
            )
            fig.update_layout(
                title=dict(
                    text=f"Negativity of graph state: 𝒩 = {negativity:.3f}",
                    subtitle=dict(
                        text=f"edge: {edge}, spectators: ({', '.join(edge_and_spectators[edge])}) = '{sbits}'",
                    ),
                ),
                margin=dict(t=110),
                width=600,
                height=342,
            )

            if plot:
                fig.show(
                    config={
                        "toImageButtonOptions": {
                            "format": "png",
                            "scale": 3,
                        },
                    }
                )

            edge_sbits_result[edge][sbits]["expected_values"] = expected_values
            edge_sbits_result[edge][sbits]["density_matrix"] = rho
            edge_sbits_result[edge][sbits]["partial_transpose"] = rho_pt
            edge_sbits_result[edge][sbits]["negativity"] = negativity
            edge_sbits_result[edge][sbits]["eigenvalues"] = eigvals
            edge_sbits_result[edge][sbits]["figure"] = fig
            edge_sbits_result[edge][sbits]["negativity_std"] = neg_std
            edge_sbits_result[edge][sbits]["negativity_ci"] = (neg_lo, neg_hi)
            # Note: CI is approximately 68% via 16-84th percentiles.

    result = {"best": {edge: {} for edge in target_edges}}

    for edge, sbits_results in edge_sbits_result.items():
        best_result = max(
            sbits_results.values(),
            key=lambda x: x.get("negativity", 0),
        )
        result["best"][edge] = best_result

    result["all"] = edge_sbits_result

    return result


def visualize_graph(
    exp: Experiment,
    G: nx.Graph,
    *,
    title: str | None = None,
    property: str | None = None,
    show_labels: bool | None = None,
    show_data: bool | None = None,
) -> None:
    """
    Visualize a graph with optional edge annotations.

    Parameters
    ----------
    G
        Graph to visualize.
    property
        Edge attribute to display.
    show_labels
        Whether to show edge labels.
    """
    if property is None:
        property = "fidelity"
    if show_labels is None:
        show_labels = False
    if show_data is None:
        show_data = True
    node_values = dict.fromkeys(G.nodes(), 1)
    edge_values = {}
    edge_texts = {}
    if show_data:
        for u, v, data in G.edges(data=True):
            value = data.get(property)
            if property == "fidelity":
                text = f"{value * 1e2:.1f}" if value is not None else "N/A"
            else:
                text = f"{value:.2f}" if value is not None else "N/A"
            if value is not None:
                edge_values[f"{u}-{v}"] = value
                edge_texts[f"{u}-{v}"] = text
    else:
        edge_values = {f"{edge[0]}-{edge[1]}": 1 for edge in G.edges()}

    chip_graph = exp.ctx.quantum_system.chip_graph
    chip_graph.plot_graph_data(
        directed=False,
        title=title or "Chip graph",
        edge_values=edge_values,
        edge_texts=edge_texts if show_labels else None,
        edge_color="turquoise" if not show_data else None,
        node_color="white",
        node_linecolor="ghostwhite",
        node_textcolor="ghostwhite",
        node_overlay=True,
        node_overlay_values=node_values,
        node_overlay_color="ghostwhite",
        node_overlay_linecolor="black",
        node_overlay_textcolor="black",
    )


def measure_graph_state(
    exp: Experiment,
    graph: nx.Graph,
    *,
    mle_fit: bool | None = None,
    use_all_spectator_pattern: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    method: str | None = None,
    reset_awg_and_capunits: bool | None = None,
    n_bootstrap: int | None = None,
    bootstrap_mle: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure graph-state negativities for all edges.

    Parameters
    ----------
    graph
        Graph defining the state to measure.
    mle_fit
        Whether to use MLE reconstruction.
    n_bootstrap
        Bootstrap sample count for error estimates.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="measure_graph_state",
    )
    if mle_fit is None:
        mle_fit = True
    if use_all_spectator_pattern is None:
        use_all_spectator_pattern = True
    if n_shots is None:
        n_shots = 3000
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if method is None:
        method = "execute"
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if n_bootstrap is None:
        n_bootstrap = 200
    if bootstrap_mle is None:
        bootstrap_mle = False
    if plot:
        seq = create_graph_sequence(
            exp,
            graph=graph,
            with_readout_pulses=False,
        )
        seq.plot(
            title=f"Graph state preparation sequence for {len(graph.nodes())} qubits",
            n_samples=1000,
        )

    negativities = {}
    figures = {}
    negativity_errors: dict[tuple[str, str], float] = {}
    rounds = create_measurement_rounds(exp, graph, plot=False)
    for round, target_edges in rounds.items():
        print(f"[{round + 1}/{len(rounds)}] Measuring edges in round #{round + 1}")

        if plot:
            G = nx.Graph()
            for u, v in target_edges:
                G.add_node(u)
                G.add_node(v)
                G.add_edge(u, v)

            node_values = dict.fromkeys(graph.nodes(), 1.0)
            edge_values = {f"{u}-{v}": 1.0 for u, v in graph.edges()}
            edge_overlay_values = {f"{u}-{v}": 1.0 for u, v in G.edges()}

            chip_graph = exp.ctx.quantum_system.chip_graph
            chip_graph.plot_graph_data(
                directed=False,
                title=f"Measurement round : #{round + 1}",
                edge_values=edge_values,
                edge_color="#eef",
                edge_overlay=True,
                edge_overlay_values=edge_overlay_values,
                edge_overlay_color="turquoise",
                node_color="white",
                node_linecolor="ghostwhite",
                node_textcolor="ghostwhite",
                node_overlay=True,
                node_overlay_values=node_values,
                node_overlay_color="ghostwhite",
                node_overlay_linecolor="black",
                node_overlay_textcolor="black",
            )

        result = _measure_graph_state(
            exp,
            graph=graph,
            target_edges=target_edges,
            mle_fit=mle_fit,
            use_all_spectator_pattern=use_all_spectator_pattern,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            method=method,
            reset_awg_and_capunits=reset_awg_and_capunits,
            n_bootstrap=n_bootstrap,
            bootstrap_mle=bootstrap_mle,
        )
        for edge, data in result["best"].items():
            negativities[edge] = data["negativity"]
            figures[edge] = data["figure"]
            if "negativity_std" in data and data["negativity_std"] is not None:
                negativity_errors[edge] = float(data["negativity_std"]) * 2
            data["figure"].show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

    negativities = dict(
        sorted(negativities.items(), key=lambda item: item[1], reverse=False)
    )

    negativities_max = max(negativities.values())
    negativities_min = min(negativities.values())
    negativities_avg = np.mean(list(negativities.values()))
    negativities_std = np.std(list(negativities.values()))
    negativities_med = np.median(list(negativities.values()))

    nonzero_edges = {
        edge: negativity
        for edge, negativity in negativities.items()
        if negativity - negativity_errors.get(edge, 0.0) > 0
    }

    if plot:
        print(f"Statistics of {len(negativities)} edges:")
        print(f"  max: {negativities_max:.3f}")
        print(f"  min: {negativities_min:.3f}")
        print(f"  med: {negativities_med:.3f}")
        print(f"  avg: {negativities_avg:.3f}")
        print(f"  std: {negativities_std:.3f}")
        print("Negativities of edges:")
        for edge, negativity in negativities.items():
            if n_bootstrap:
                print(
                    f"  {edge[0]}-{edge[1]}: {negativity:.3f} ± {negativity_errors.get(edge, 0.0):.3f}"
                )
            else:
                print(f"  {edge[0]}-{edge[1]}: {negativity:.3f}")

        x = [f"{edge[0]}-{edge[1]}" for edge in negativities]
        y = list(negativities.values())
        y_err = [negativity_errors.get(edge, 0.0) for edge in negativities]

        min_y = min(
            0,
            min(
                [
                    negativity - negativity_errors.get(edge, 0.0)
                    for edge, negativity in negativities.items()
                ]
            ),
        )
        max_y = max(
            0.55,
            max(
                [
                    negativity + negativity_errors.get(edge, 0.0)
                    for edge, negativity in negativities.items()
                ]
            ),
        )

        fig = viz.make_figure()
        fig.update_layout(
            title=f"Negativities of {len(graph.nodes())}-qubit graph state",
            xaxis=dict(
                title="Edges",
                title_standoff=25,
                tickangle=90,
                tickmode="array",
                tickvals=list(range(len(x))),
                ticktext=x,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title="Negativity",
                range=[min_y, max_y],
                tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                ticktext=["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
            ),
            width=800,
            height=400,
            margin=dict(l=70, r=70, t=90, b=100),
        )
        fig.add_scatter(
            x=x,
            y=y,
            error_y=dict(
                type="data",
                array=y_err,
            )
            if n_bootstrap
            else None,
        )
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

        nonzero_graph = nx.Graph()
        for edge, negativity in nonzero_edges.items():
            nonzero_graph.add_edge(edge[0], edge[1], negativity=negativity)

        components = nx.connected_components(nonzero_graph)
        n_max = max(len(c) for c in components)

        visualize_graph(
            exp,
            nonzero_graph,
            title=f"Entangled qubits : N (max) = {n_max}",
            property="negativity",
            show_labels=True,
            show_data=True,
        )

    return Result(
        data={
            "negativities": negativities,
            "negativity_errors": negativity_errors,
            "negativities_max": negativities_max,
            "negativities_min": negativities_min,
            "negativities_med": negativities_med,
            "negativities_avg": negativities_avg,
            "negativities_std": negativities_std,
            "nonzero_edges": nonzero_edges,
            "figures": figures,
        },
        figures=figures,
    )


def _canonical_edge(
    exp: Experiment, edge: str | tuple[int | str, int | str]
) -> tuple[str, str]:
    if isinstance(edge, str):
        qubits = tuple(edge.split("-"))
        return (qubits[0], qubits[1])
    else:
        qubits = tuple(edge)
        if isinstance(qubits[0], int):
            qubit0 = exp.ctx.quantum_system.get_qubit(qubits[0]).label
        else:
            qubit0 = qubits[0]
        if isinstance(qubits[1], int):
            qubit1 = exp.ctx.quantum_system.get_qubit(qubits[1]).label
        else:
            qubit1 = qubits[1]
        return (qubit0, qubit1)


def measure_bell_state_fidelities(
    exp: Experiment,
    targets: Collection[str | tuple[int | str, int | str]] | None = None,
    *,
    unavailable_pairs: Collection[str | tuple[int | str, int | str]] | None = None,
    readout_mitigation: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_data: bool | None = None,
    save_path: Path | str | None = None,
    **deprecated_options: Any,
) -> Result:
    """Measure Bell-state fidelities for target pairs."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="measure_bell_state_fidelities",
    )
    if readout_mitigation is None:
        readout_mitigation = True
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = False
    if save_data is None:
        save_data = True
    # TODO: move this to an appropriate location
    fidelities = {}

    if targets is None:
        target_pairs = exp.ctx.cr_pairs
    else:
        target_pairs = [_canonical_edge(exp, target) for target in targets]

    if unavailable_pairs is None:
        unavailable_pairs = []
    else:
        unavailable_pairs = [
            _canonical_edge(exp, target) for target in unavailable_pairs
        ]

    for pair in target_pairs:
        if pair in unavailable_pairs:
            print(f"Skipping unavailable pair: {pair}")
            continue
        try:
            label = f"{pair[0]}-{pair[1]}"
            result = exp.bell_state_tomography(
                *pair,
                readout_mitigation=readout_mitigation,
                n_shots=n_shots,
                shot_interval=shot_interval,
            )
            fidelities[label] = result["fidelity"]
        except Exception as e:
            print(f"Failed for pair {label}: {e}")

    sorted_fidelities = dict(
        sorted(fidelities.items(), key=lambda x: x[1], reverse=True)
    )

    if plot:
        n_pairs = len(sorted_fidelities)
        x = list(sorted_fidelities)
        y = list(sorted_fidelities.values())
        fig = viz.make_figure()
        fig.update_layout(
            title="Fidelities",
            xaxis=dict(
                title="Edges",
                tickangle=45,
                tickmode="array",
                tickvals=list(range(len(x))),
                ticktext=x,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title="Fidelity",
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
                ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"],
            ),
            width=n_pairs * 15 + 150,
            height=400,
            margin=dict(l=70, r=70, t=90, b=100),
        )
        fig.add_bar(x=x, y=y)
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

    if save_data:
        exp.ctx.save_property(
            "bell_state_fidelity",
            sorted_fidelities,
            save_path=save_path,
        )

    return Result(data=sorted_fidelities)


def measure_bell_states(
    exp: Experiment,
    targets: Collection[str | tuple[int | str, int | str]] | None = None,
    *,
    unavailable_pairs: Collection[str | tuple[int | str, int | str]] | None = None,
    control_basis: str | None = None,
    target_basis: str | None = None,
    readout_mitigation: bool | None = None,
    in_parallel: bool | None = None,
    n_cols: int | None = None,
    threshold: float | None = None,
    title: str | None = None,
    plot: bool | None = None,
    plot_round: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    reset_awg_and_capunits_each_time: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """Measure Bell-state populations for target pairs."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="measure_bell_states",
    )
    if control_basis is None:
        control_basis = "Z"
    if target_basis is None:
        target_basis = "Z"
    if readout_mitigation is None:
        readout_mitigation = True
    if in_parallel is None:
        in_parallel = True
    if n_cols is None:
        n_cols = 6
    if threshold is None:
        threshold = 0.0
    if plot is None:
        plot = True
    if plot_round is None:
        plot_round = False
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if reset_awg_and_capunits_each_time is None:
        reset_awg_and_capunits_each_time = True
    if targets is None:
        try:
            fidelities = exp.ctx.load_property("bell_state_fidelity")
            target_pairs = [
                _canonical_edge(exp, label)
                for label, fidelity in fidelities.items()
                if fidelity >= threshold
            ]
        except FileNotFoundError:
            target_pairs = exp.ctx.cr_pairs
    else:
        target_pairs = [_canonical_edge(exp, target) for target in targets]

    if unavailable_pairs is None:
        unavailable_pairs = []
    else:
        unavailable_pairs = [
            _canonical_edge(exp, target) for target in unavailable_pairs
        ]

    all_edges = sorted(
        [
            (pair[0], pair[1])
            for pair in target_pairs
            if pair not in unavailable_pairs
            and f"{pair[0]}-{pair[1]}" in exp.ctx.calib_note.cr_params
        ]
    )

    if reset_awg_and_capunits and not reset_awg_and_capunits_each_time:
        qubits = set()
        for edge in all_edges:
            qubits.add(edge[0])
            qubits.add(edge[1])
        exp.ctx.reset_awg_and_capunits(qubits=qubits)

    n_edges = len(all_edges)
    n_rows = int(np.ceil(n_edges / n_cols))

    edge_indices = {
        edge: (i // n_cols + 1, i % n_cols + 1) for i, edge in enumerate(all_edges)
    }

    fig = viz.make_figure()
    fig.set_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{edge[0]}-{edge[1]}" for edge in all_edges],
    )

    results = {}

    if in_parallel:
        graph = exp.ctx.quantum_system.chip_graph
        rounds = []
        for edges in graph.strong_edge_coloring():
            batch = []
            for edge in edges:
                if edge[0] % 4 in [0, 3]:
                    e = _canonical_edge(exp, (edge[0], edge[1]))
                else:
                    e = _canonical_edge(exp, (edge[1], edge[0]))
                if e in all_edges:
                    batch.append(e)
            rounds.append(batch)

        for round, edges in tqdm(enumerate(rounds), desc="Measuring Bell states"):
            if plot_round:
                G = nx.Graph()
                for u, v in edges:
                    G.add_node(u)
                    G.add_node(v)
                    G.add_edge(u, v)

                edge_values = {f"{u}-{v}": 1.0 for u, v in all_edges}
                edge_overlay_values = {f"{u}-{v}": 1.0 for u, v in G.edges()}

                chip_graph = exp.ctx.quantum_system.chip_graph
                chip_graph.plot_graph_data(
                    directed=False,
                    title=f"Measurement round : {round + 1}",
                    edge_values=edge_values,
                    edge_color="#eef",
                    edge_overlay=True,
                    edge_overlay_values=edge_overlay_values,
                    edge_overlay_color="turquoise",
                    node_color="white",
                    node_linecolor="ghostwhite",
                    node_textcolor="ghostwhite",
                    node_overlay=True,
                    # node_overlay_values=node_values,
                    node_overlay_color="ghostwhite",
                    node_overlay_linecolor="black",
                    node_overlay_textcolor="black",
                )

            with PulseSchedule() as ps:
                for edge in edges:
                    control_qubit, target_qubit = edge
                    # prepare |+⟩|0⟩
                    ps.add(control_qubit, exp.pulse.y90(control_qubit))

                    # create |0⟩|0⟩ + |1⟩|1⟩
                    ps.call(
                        exp.pulse.cnot(
                            control_qubit,
                            target_qubit,
                            only_low_to_high=True,
                        )
                    )

                    # apply the control basis transformation
                    if control_basis == "X":
                        ps.add(control_qubit, exp.pulse.y90m(control_qubit))
                    elif control_basis == "Y":
                        ps.add(control_qubit, exp.pulse.x90(control_qubit))

                    # apply the target basis transformation
                    if target_basis == "X":
                        ps.add(target_qubit, exp.pulse.y90m(target_qubit))
                    elif target_basis == "Y":
                        ps.add(target_qubit, exp.pulse.x90(target_qubit))

            result = exp.measure(
                ps,
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                reset_awg_and_capunits=reset_awg_and_capunits_each_time,
            )

            for edge in edges:
                row, col = edge_indices[edge]

                control_qubit, target_qubit = edge
                basis_labels = result.get_basis_labels(edge)
                prob_dict_raw = result.get_probabilities(edge)
                # Ensure all basis labels are present in the raw probabilities
                prob_dict_raw = {
                    label: prob_dict_raw.get(label, 0) for label in basis_labels
                }
                prob_dict_mitigated = result.get_mitigated_probabilities(edge)

                labels = [f"|{i}⟩" for i in prob_dict_raw]
                prob_arr_raw = np.array(list(prob_dict_raw.values()))
                prob_arr_mitigated = np.array(list(prob_dict_mitigated.values()))

                if readout_mitigation:
                    prob_arr = prob_arr_mitigated
                else:
                    prob_arr = prob_arr_raw

                results[f"{edge[0]}-{edge[1]}"] = {
                    "raw_probabilities": prob_arr_raw,
                    "mitigated_probabilities": prob_arr_mitigated,
                }

                fig.add_bar(
                    x=labels,
                    y=prob_arr,
                    row=row,
                    col=col,
                    marker_color=COLORS[0],
                )
    else:
        for edge in tqdm(all_edges, total=len(all_edges)):
            row, col = edge_indices[edge]
            labels = [f"|{i}⟩" for i in ["00", "01", "10", "11"]]
            result = exp.measure_bell_state(
                *edge,
                n_shots=n_shots,
                plot=False,
                save_image=False,
                reset_awg_and_capunits=reset_awg_and_capunits_each_time,
            )
            if readout_mitigation:
                prob_arr = result["mitigated"]
            else:
                prob_arr = result["raw"]
            results[f"{edge[0]}-{edge[1]}"] = {
                "raw_probabilities": result["raw"],
                "mitigated_probabilities": result["mitigated"],
            }
            fig.add_bar(
                x=labels,
                y=prob_arr,
                row=row,
                col=col,
                marker_color=COLORS[0],
            )

    fig_subtitle = f"{n_edges} pairs, {n_shots} shots"
    if in_parallel:
        fig_subtitle += ", run in parallel"
    else:
        fig_subtitle += ", run sequentially"

    fig.update_layout(
        title=dict(
            text=title or "Bell state measurement",
            subtitle=dict(
                text=fig_subtitle,
                font_size=16,
            ),
            y=0.98,
            yanchor="top",
            font_size=22,
        ),
        height=120 * n_rows + 200,
        width=180 * n_cols + 80,
        showlegend=False,
        margin=dict(l=40, r=40, t=160, b=40),
    )
    fig.update_yaxes(
        range=[0, 0.6],
        tickvals=[0, 0.25, 0.5],
        ticktext=["0", "0.25", "0.5"],
    )
    fig.update_annotations(
        font_size=15,
        yshift=8,
    )
    if plot:
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

    return Result(
        data={
            "data": results,
            "figure": fig,
        },
        figure=fig,
    )
