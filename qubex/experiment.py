"""
a experiment library for qubex
"""

from attr import dataclass

import numpy as np
from IPython.display import clear_output

from .measurement import Measurement, QUBE_ID
from .pulse import Rcft
from .params import ampl_hpi_dict, cal_ampl_dict, cal_iq_ratio_dict
from .plot import plot_readout_waveform

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


def rabi_experiment():
    """
    Performs a Rabi measurement.
    """
    m = Measurement()

    states = {qubit: [] for qubit in m.qubits}

    # sweep the duration of the control RCFT pulses
    idx = np.arange(0, 200, 10)  # ns
    for i in idx:
        # clear figures
        clear_output(wait=True)

        # set the control waveforms
        for qubit in m.qubits:
            waveform = Rcft(
                ampl=AMPL_HPI_RCFT[qubit],
                rise=10,
                flat=i,
                fall=10,
            )
            m.set_readout_waveform(qubit, waveform)

        m.show_pulse_sequences()

        for qubit, state in m.measure().items():
            waveform = m.readout_rx_waveform(qubit)
            plot_readout_waveform(qubit, waveform)
            states[qubit].append(state)
