"""Contributed helpers for thermal excitation characterization."""

from __future__ import annotations

import numpy as np
from qxpulse import PulseSchedule

from qubex import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_SHOTS,
)
from qubex.experiment.models.result import Result

THERMAL_EXCITATION_DEFAULT_SHOTS = int(1024 * DEFAULT_SHOTS)


def measure_thermal_excitation(
    exp: Experiment,
    target: str,
    *,
    n_shots: int = THERMAL_EXCITATION_DEFAULT_SHOTS,
    plot: bool = False,
) -> Result:
    """
    Estimate the thermal excitation probability of a qubit.

    Parameters
    ----------
    target : str
        Target qubit to measure.
    n_shots : int, optional
        Number of measurement shots per sequence.
        Defaults to 1024 times the value of ``DEFAULT_SHOTS``.
    plot : bool, optional
        Whether to plot figures.
    """
    qubit = exp.ctx.resolve_qubit_label(target)
    ge_label = exp.ctx.resolve_ge_label(target)
    ef_label = exp.ctx.resolve_ef_label(target)

    def sequence_sig(ef_rabi_amplitude_scale: float) -> PulseSchedule:
        ef_pulse = exp.pulse.x180(ef_label).scaled(ef_rabi_amplitude_scale)
        with PulseSchedule() as ps:
            ps.add(ef_label, ef_pulse)
            ps.barrier()
            ps.add(ge_label, exp.pulse.x180(ge_label))
        return ps

    def sequence_ref(ef_rabi_amplitude_scale: float) -> PulseSchedule:
        ef_pulse = exp.pulse.x180(ef_label).scaled(ef_rabi_amplitude_scale)
        with PulseSchedule() as ps:
            ps.add(ge_label, exp.pulse.x180(ge_label))
            ps.barrier()
            ps.add(ef_label, ef_pulse)
            ps.barrier()
            ps.add(ge_label, exp.pulse.x180(ge_label))
        return ps

    result_sig_0 = exp.measure(
        sequence=sequence_sig(0),
        mode="avg",
        n_shots=n_shots,
        plot=plot,
    )

    result_sig_1 = exp.measure(
        sequence=sequence_sig(1),
        mode="avg",
        n_shots=n_shots,
        plot=plot,
    )

    result_ref_0 = exp.measure(
        sequence=sequence_ref(0),
        mode="avg",
        n_shots=n_shots,
        plot=plot,
    )

    result_ref_1 = exp.measure(
        sequence=sequence_ref(1),
        mode="avg",
        n_shots=n_shots,
        plot=plot,
    )

    iq_sig_0 = result_sig_0.data[qubit].kerneled
    iq_sig_1 = result_sig_1.data[qubit].kerneled
    iq_ref_0 = result_ref_0.data[qubit].kerneled
    iq_ref_1 = result_ref_1.data[qubit].kerneled

    A_sig = np.abs(iq_sig_1 - iq_sig_0)
    A_ref = np.abs(iq_ref_1 - iq_ref_0)
    total_amplitude = A_sig + A_ref
    if total_amplitude == 0:
        raise ValueError("Cannot estimate thermal excitation with zero signal.")
    p_ex = A_sig / total_amplitude
    print("")
    print(f"{qubit}")
    print(f"A_sig : {A_sig}")
    print(f"A_ref : {A_ref}")
    print(f"p_ex  : {p_ex}")

    return Result(
        data={
            "p_ex": p_ex,
            "signal_amplitude": A_sig,
            "reference_amplitude": A_ref,
        },
    )
