"""
a module for data analysis of qube experiment
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.decomposition import PCA


def principal_components(iq_complex, pca=None):
    iq_complex = np.array(iq_complex)
    iq_vector = np.column_stack([np.real(iq_complex), np.imag(iq_complex)])
    if pca is None:
        pca = PCA(n_components=1)
    results = pca.fit_transform(iq_vector).squeeze()
    return results, pca


def fit_and_find_minimum(x, y, p0=None):
    def cos_func(t, ampl, omega, phi, offset):
        return ampl * np.cos(omega * t + phi) + offset

    if p0 is None:
        p0 = (
            np.abs(np.max(y) - np.min(y)) / 2,
            1 / (x[-1] - x[0]),
            0,
            (np.max(y) + np.min(y)) / 2,
        )

    popt, _ = curve_fit(cos_func, x, y, p0=p0)
    print(
        f"Fitted function: {popt[0]:.3f} * cos({popt[1]:.3f} * t + {popt[2]:.3f}) + {popt[3]:.3f}"
    )

    result = minimize(
        cos_func,
        x0=np.mean(x),
        args=tuple(popt),
        bounds=[(np.min(x), np.max(x))],
    )
    min_x = result.x[0]
    min_y = cos_func(min_x, *popt)

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = cos_func(x_fine, *popt)

    plt.scatter(x, y, label="Data")
    plt.plot(x_fine, y_fine, label="Fit")
    plt.scatter(min_x, min_y, color="red", label="Minimum")
    plt.legend()
    plt.show()

    print(f"Minimum: ({min_x}, {min_y})")

    return min_x, min_y


def rotate_to_vertical(data) -> np.ndarray:
    states = np.array(data)
    fit_params = np.polyfit(states.real, states.imag, 1)
    a, _ = fit_params

    theta = np.arctan(a)
    rotation_angle = np.pi / 2 - theta
    rotated_states = states * np.exp(1j * rotation_angle)

    return rotated_states
