"""
a experiment library for qubex
"""

import os
from datetime import datetime
from attr import dataclass

import numpy as np
from IPython.display import clear_output

from .measurement import Measurement, QUBE_ID
from .pulse import Pulse, Rcft, Drag
from .params import ampl_hpi_dict, cal_ampl_dict, cal_iq_ratio_dict
from .plot import show_readout_pulse
from .analysis import linear_fit_and_rotate_IQ

AMPL_HPI_RCFT = ampl_hpi_dict[QUBE_ID]
AMPL_HPI_DRAG = cal_ampl_dict
IQ_RATIO_DRAG = cal_iq_ratio_dict


@dataclass
class ExperimentResult:
    """
    An experiment result.
    """

    idx: np.ndarray
    iq: dict[str, list[complex]]
    rotated_iq: dict[str, list[complex]]
    grad: float
    intercept: float


def rabi_experiment() -> ExperimentResult:
    """
    Performs a Rabi measurement.
    """
    m = Measurement()

    iq = {qubit: [] for qubit in m.qubits}

    # sweep the duration of the control RCFT pulses
    idx = np.arange(0, 200, 10)  # ns
    for i in idx:
        # clear figures
        clear_output(wait=True)

        # set the control pulses
        for qubit in m.qubits:
            pulse = Rcft(
                ampl=AMPL_HPI_RCFT[qubit],
                rise=10,
                flat=i,
                fall=10,
            )
            m.set_control_pulse(qubit, pulse)

        m.show_pulse_sequences()

        for qubit, value in m.measure().items():
            pulse = m.ro_return_pulse(qubit)
            show_readout_pulse(qubit, pulse)
            iq[qubit].append(value)

    rotated_iq, grad, intercept = linear_fit_and_rotate_IQ(m.qubits, iq)
    result = ExperimentResult(idx, iq, rotated_iq, grad, intercept)
    save_result(result)
    return result


def repeat_rcft_pi(iteration: int) -> ExperimentResult:
    """
    Repeat RCFT pi pulse.
    """
    m = Measurement()

    iq = {qubit: [] for qubit in m.qubits}

    idx = np.arange(0, iteration)
    for i in idx:
        clear_output(wait=True)
        print(f"idx: {i}")

        for qubit in m.qubits:
            hpi = Rcft(
                ampl=AMPL_HPI_RCFT[qubit],
                rise=10,
                flat=10,
                fall=10,
            )
            pi = Pulse.concat(hpi, hpi)
            m.set_control_pulse(qubit, pi)

        m.show_pulse_sequences()

        for qubit, value in m.measure().items():
            iq[qubit].append(value)

    rotated_iq, grad, intercept = linear_fit_and_rotate_IQ(m.qubits, iq)
    result = ExperimentResult(idx, iq, rotated_iq, grad, intercept)
    save_result(result)
    return result


def repeat_drag_pi(iteration: int) -> ExperimentResult:
    """
    Repeat DRAG pi pulse.
    """
    m = Measurement()

    iq = {qubit: [] for qubit in m.qubits}

    idx = np.arange(0, iteration)
    for i in idx:
        clear_output(wait=True)
        print(f"idx: {i}")

        for qubit in m.qubits:
            hpi = Drag(
                duration=10,
                ampl=AMPL_HPI_DRAG[qubit],
                beta=IQ_RATIO_DRAG,
            )
            pi = Pulse.concat(hpi, hpi)
            m.set_control_pulse(qubit, pi)

        m.show_pulse_sequences()

        for qubit, value in m.measure().items():
            iq[qubit].append(value)

    rotated_iq, grad, intercept = linear_fit_and_rotate_IQ(m.qubits, iq)
    result = ExperimentResult(idx, iq, rotated_iq, grad, intercept)
    save_result(result)
    return result


def save_result(result: ExperimentResult):
    """
    Saves the experiment result.
    """
    now = datetime.now()
    dir_name = now.strftime("%Y/%m/%d/%H%M%S%f")
    path_str = f"./data/{dir_name}/experiment_result.npy"
    dir_path = os.path.dirname(path_str)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.normpath(path_str)
    np.save(path, result)
