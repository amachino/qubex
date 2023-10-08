"""
a module for data analysis of qube experiment
"""

import numpy as np


def rotate_to_vertical(data) -> np.ndarray:
    states = np.array(data)
    fit_params = np.polyfit(states.real, states.imag, 1)
    a, _ = fit_params

    theta = np.arctan(a)
    rotation_angle = np.pi / 2 - theta
    rotated_states = states * np.exp(1j * rotation_angle)

    return rotated_states
