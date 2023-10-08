"""
a plot library for qubex
"""

import numpy as np
import matplotlib.pyplot as plt

from .measurement import READ_SLICE_RANGE
from .waveform import Waveform
from .analysis import rotate_to_vertical


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


def plot_states_before_after_rotation(data):
    states = np.array(data)
    rotated_states = rotate_to_vertical(data)

    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(states.real, states.imag)
    axs[0].set_title("Before Rotation")

    axs[1].scatter(rotated_states.real, rotated_states.imag)
    axs[1].set_title("After Rotation")

    for ax in axs:
        ax.axis("equal")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True)

    plt.show()


def plot_states_vs_index(data):
    states = np.array(data)
    _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(np.arange(len(states)), states.real)
    ax.plot(np.arange(len(states)), states.imag)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True)

    plt.show()
