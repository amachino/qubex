"""Contributed helpers for thermal excitation characterization."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from qxpulse import FlatTop, PulseSchedule, Waveform

from qubex import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_SHOTS,
    PI_DURATION,
    PI_RAMPTIME,
)
from qubex.experiment.models.result import Result
from qubex.measurement.models.measure_result import (
    MeasureResult,
)
from qubex.system.target import Target

EF_PI_DURATION = PI_DURATION
EF_PI_RAMPTIME = PI_RAMPTIME

THERMAL_EXCITATION_DEFAULT_SHOTS = int(10 * DEFAULT_SHOTS)


def _build_population_rabi_sequence(
    target: str,
    ef_rabi_amplitude: float,
    pi_pulse: Waveform,
) -> dict[str, PulseSchedule]:
    def sequence0(
        target: str,
        ef_rabi_amplitude: float,
        pi_pulse: Waveform,
    ) -> PulseSchedule:
        ef_label = Target.ef_label(target)

        with PulseSchedule() as ps:
            ps.add(
                ef_label,
                FlatTop(
                    duration=EF_PI_DURATION,
                    amplitude=ef_rabi_amplitude,
                    tau=EF_PI_RAMPTIME,
                ),
            )
            ps.barrier()
            ps.add(target, pi_pulse)
        return ps

    def sequence1(
        target: str,
        ef_rabi_amplitude: float,
        pi_pulse: Waveform,
    ) -> PulseSchedule:
        ef_label = Target.ef_label(target)

        with PulseSchedule() as ps:
            ps.add(target, pi_pulse)
            ps.barrier()
            ps.add(
                ef_label,
                FlatTop(
                    duration=EF_PI_DURATION,
                    amplitude=2 * ef_rabi_amplitude,
                    tau=EF_PI_RAMPTIME,
                ),
            )
            ps.barrier()
            ps.add(target, pi_pulse)
        return ps

    return {
        "sequence0": sequence0(
            target=target,
            ef_rabi_amplitude=ef_rabi_amplitude,
            pi_pulse=pi_pulse,
        ),
        "sequence1": sequence1(
            target=target,
            ef_rabi_amplitude=ef_rabi_amplitude,
            pi_pulse=pi_pulse,
        ),
    }


def thermal_excitation_via_rabi(
    exp: Experiment,
    *,
    target: str,
    n_shots: int = THERMAL_EXCITATION_DEFAULT_SHOTS,
    readout_amplitude: float | None = None,
    plot: bool = False,
) -> Result:
    """
    Estimate the thermal excitation probability of a qubit via ef Rabi oscillations.

    Parameters
    ----------
    target : str
        Target qubit to measure.
    n_shots : int, optional
        Number of measurement shots per sequence.  Defaults to 10 times the value of `DEFAULT_SHOTS``.
    plot : bool, optional
        Whether to plot ef rabi.
    """
    control_amplitude = exp.calc_control_amplitude(target, rabi_rate=0.0125)
    if control_amplitude is not None:
        ef_rabi_amplitude = control_amplitude / np.sqrt(2)
    if ef_rabi_amplitude is None:
        raise ValueError("Failed to determine ef_rabi_amplitude.")

    if readout_amplitude is None:
        readout_amplitudes = None
    else:
        readout_amplitudes = {target: readout_amplitude}

    amplitude_history = defaultdict(list)

    for _ef_rabi_amplitude in [0, ef_rabi_amplitude]:
        sequences = _build_population_rabi_sequence(
            target=target,
            ef_rabi_amplitude=_ef_rabi_amplitude,
            pi_pulse=exp.x180(target),
        )

        result0: MeasureResult = exp.measure(
            sequence=sequences["sequence0"],
            readout_amplitudes=readout_amplitudes,
            mode="avg",
            n_shots=n_shots,
            plot=plot,
        )
        result1: MeasureResult = exp.measure(
            sequence=sequences["sequence1"],
            readout_amplitudes=readout_amplitudes,
            mode="avg",
            n_shots=n_shots,
            plot=plot,
        )

        iq0 = result0.data[target].kerneled
        iq1 = result1.data[target].kerneled
        state_center = exp.state_centers[target][1]
        vec0 = np.abs(iq0 - state_center)
        vec1 = np.abs(iq1 - state_center)
        amplitude_history["0"].append(vec0)
        amplitude_history["1"].append(vec1)

    A_min = np.abs(amplitude_history["0"][0] - amplitude_history["0"][-1])
    A_max = np.abs(amplitude_history["1"][0] - amplitude_history["1"][-1])
    p_ex = A_min / (A_min + A_max)
    print("")
    print(f"{target}")
    print(f"A_min : {A_min}")
    print(f"A_max : {A_max}")
    print(f"p_ex  : {p_ex}")

    return Result(
        data={
            "p_ex": p_ex,
            "rabi_ampl_min": A_min,
            "rabi_ampl_max": A_max,
        },
    )
