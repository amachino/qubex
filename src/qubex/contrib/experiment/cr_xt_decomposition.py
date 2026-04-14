"""CR crosstalk decomposition utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_CR_RAMPTIME,
    DEFAULT_CR_TIME_RANGE,
    DEFAULT_INTERVAL,
)
from qubex.experiment.models import Result
from qubex.pulse import Waveform
from qubex.typing import TargetMap


def decompose_cr_crosstalk(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    cr_amplitude: float | None = None,
    adiabatic_safe_factor: float | None = None,
    max_amplitude: float | None = None,
    x90: TargetMap[Waveform] | None = None,
    shots: int | None = None,
    interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot: bool | None = None,
) -> Result:
    """
    Decompose CR crosstalk into quantum and classical contributions.

    This function performs CR Hamiltonian tomography for a given qubit pair
    and decomposes the measured crosstalk terms (IX, IY) into quantum and
    classical components. The quantum crosstalk is estimated from the CR
    interaction terms (ZX, ZY) using a perturbative relation, while the
    classical crosstalk is obtained as the residual.

    The decomposition is based on:
        IX_quantum = ZX * Δ / alpha_c
        IY_quantum = ZY * Δ / alpha_c

    where Δ is the detuning between the control and target qubits, and
    alpha_c is the anharmonicity of the control qubit.

    Parameters
    ----------
    exp : Experiment
        Experiment object that provides access to calibration routines and
        device configuration.
    control_qubit : str
        Label of the control qubit.
    target_qubit : str
        Label of the target qubit.
    time_range : ArrayLike, optional
        Time points (ns) used in CR Hamiltonian tomography.
    ramptime : float, optional
        Ramp time (ns) of the CR pulse.
    cr_amplitude : float, optional
        Amplitude of the CR drive pulse.
    adiabatic_safe_factor : float, optional
        Scaling factor to limit the CR drive strength relative to detuning.
    max_amplitude : float, optional
        Maximum allowed CR drive amplitude.
    x90 : TargetMap[Waveform], optional
        Pre-calibrated X90 (π/2) pulses for each qubit.
    shots : int, optional
        Number of measurement shots.
    interval : float, optional
        Interval between repeated measurements (ns).
    reset_awg_and_capunits : bool, optional
        Whether to reset AWG and capture units before measurement.
    plot : bool, optional
        If True, display figures and print analysis results.

    Returns
    -------
    Result
        A Result object containing the decomposition results, including:

        - IX_total, IY_total : total crosstalk components
        - IX_quantum, IY_quantum : quantum crosstalk contributions
        - IX_classical, IY_classical : classical crosstalk contributions
        - xt_total, xt_quantum, xt_classical : complex-valued representations
        - f_delta : detuning between qubits (GHz)
        - anharmonicity_control : control qubit anharmonicity (GHz)
        - xt_ratio : ratio |classical| / |quantum|
        - tomography_result : raw CR Hamiltonian tomography result

    Raises
    ------
    ValueError
        If the control qubit anharmonicity is too small to reliably estimate
        quantum crosstalk.

    Notes
    -----
    The anharmonicity of the control qubit is assumed to be known or
    reasonably estimated.

    This method is intended for analysis purposes and does not modify
    calibration parameters.
    """
    if max_amplitude is None:
        max_amplitude = 1.0
    if shots is None:
        shots = CALIBRATION_SHOTS
    if interval is None:
        interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if plot is None:
        plot = True
    if ramptime is None:
        ramptime = _ramptime(
            exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
        )
    if adiabatic_safe_factor is None:
        adiabatic_safe_factor = 0.75

    cr_label = f"{control_qubit}-{target_qubit}"

    f_control = exp.ctx.qubits[control_qubit].frequency
    f_target = exp.ctx.qubits[target_qubit].frequency
    f_delta = f_control - f_target
    max_cr_rabi = adiabatic_safe_factor * np.abs(f_delta)
    max_cr_amplitude = exp.pulse.calc_control_amplitude(control_qubit, max_cr_rabi)
    max_cr_amplitude: float = np.clip(max_cr_amplitude, 0.0, max_amplitude)

    cr_amplitude = cr_amplitude if cr_amplitude is not None else max_cr_amplitude
    if time_range is None:
        time_range = DEFAULT_CR_TIME_RANGE
    time_range = np.array(time_range, dtype=float)

    print(f"Conducting CR Hamiltonian tomography for {cr_label}...")

    result = exp.calibration_service.cr_hamiltonian_tomography(
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        time_range=time_range,
        ramptime=ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=0.0,
        cancel_amplitude=0.0,
        cancel_phase=0.0,
        x90=x90,
        shots=shots,
        interval=interval,
        reset_awg_and_capunits=reset_awg_and_capunits,
        plot=plot,
    )

    coeffs = result["coeffs"]

    IX = coeffs["IX"]
    IY = coeffs["IY"]
    ZX = coeffs["ZX"]
    ZY = coeffs["ZY"]

    alpha_c = exp.ctx.qubits[control_qubit].anharmonicity

    eps = 1e-12

    if np.abs(alpha_c) < eps:
        raise ValueError(
            f"Anharmonicity is too small (alpha_c={alpha_c}). Cannot compute quantum crosstalk."
        )

    IX_quantum = ZX * f_delta / alpha_c
    IY_quantum = ZY * f_delta / alpha_c

    IX_classical = IX - IX_quantum
    IY_classical = IY - IY_quantum

    xt_total = IX + 1j * IY
    xt_quantum = IX_quantum + 1j * IY_quantum
    xt_classical = IX_classical + 1j * IY_classical

    ratio = np.abs(xt_classical) / max(np.abs(xt_quantum), eps)

    if plot:
        print()
        print("Crosstalk decomposition:")
        print("  Anharmonicity (control qubit):")
        print(f"    alpha_c : {alpha_c * 1e3:+.4f} MHz")
        print(
            "    (This value is expected to be measured independently and is used here for estimating quantum crosstalk.)"
        )

        print("  --- IX / IY components ---")
        print(f"    IX total     : {IX * 1e3:+.4f} MHz")
        print(f"    IX quantum   : {IX_quantum * 1e3:+.4f} MHz")
        print(f"    IX classical : {IX_classical * 1e3:+.4f} MHz")

        print(f"    IY total     : {IY * 1e3:+.4f} MHz")
        print(f"    IY quantum   : {IY_quantum * 1e3:+.4f} MHz")
        print(f"    IY classical : {IY_classical * 1e3:+.4f} MHz")

        print("  --- Magnitude ratio ---")
        print(f"    |classical| / |quantum| : {ratio:.3f}")

    return Result(
        data={
            "control_qubit": control_qubit,
            "target_qubit": target_qubit,
            "coeffs": coeffs,
            "IX_total": IX,
            "IY_total": IY,
            "ZX": ZX,
            "ZY": ZY,
            "IX_quantum": IX_quantum,
            "IY_quantum": IY_quantum,
            "IX_classical": IX_classical,
            "IY_classical": IY_classical,
            "xt_total": xt_total,
            "xt_quantum": xt_quantum,
            "xt_classical": xt_classical,
            "f_delta": f_delta,
            "anharmonicity_control": alpha_c,
            "xt_ratio": ratio,
            "tomography_result": result,
        }
    )


def _ramptime(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
) -> float:
    f_ge_control = exp.ctx.qubits[control_qubit].frequency
    f_ef_target = exp.ctx.qubits[target_qubit].control_frequency_ef

    if f_ge_control < f_ef_target:
        return DEFAULT_CR_RAMPTIME
    else:
        return DEFAULT_CR_RAMPTIME * 2
