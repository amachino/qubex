"""
a plot library for qubex
"""

import numpy as np
import matplotlib.pyplot as plt

from .measurement import READ_SLICE_RANGE
from .waveform import Waveform


def plot_readout_waveform(qubit: str, waveform: Waveform):
    """
    Plots the readout waveform.
    """
    AVG_NUM = 50

    ax = plt.subplot()
    ax.set_title("Detected readout waveform " + qubit)
    ax.set_xlabel("Time / us")
    ax.set_xlim(0, 2.0)
    ax.grid()

    window = np.ones(AVG_NUM)
    mov_avg = np.convolve(waveform.iq, window, mode="valid") / AVG_NUM
    mov_avg = np.append(mov_avg, np.zeros(AVG_NUM - 1))
    time = waveform.time * 1e-3

    ax.plot(time, np.real(mov_avg), label="I")
    ax.plot(time, np.imag(mov_avg), label="Q")

    sliced_time = time[READ_SLICE_RANGE]
    ax.axvspan(
        sliced_time[0],
        sliced_time[-1],
        color="gray",
        alpha=0.1,
    )
    plt.show()
