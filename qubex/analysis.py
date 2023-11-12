import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.decomposition import PCA


def func_rabi(
    t: npt.NDArray[np.float64],
    ampl: float,
    omega: float,
    phi: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    return ampl * np.cos(omega * t + phi) + offset


def fit_rabi(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if p0 is None:
        p0 = (
            np.abs(np.max(y) - np.min(y)) / 2,
            2 * np.pi / (x[-1] - x[0]),
            np.pi,
            (np.max(y) + np.min(y)) / 2,
        )

    if bounds is None:
        bounds = (
            (0, 0, 0, -np.inf),
            (np.inf, np.inf, np.pi, np.inf),
        )

    popt, pcov = curve_fit(func_rabi, x, y, p0=p0, bounds=bounds)

    rabi_freq = popt[1] / (2 * np.pi)

    print(f"Rabi frequency: {rabi_freq * 1e3:.3f} MHz")
    print(f"Rabi period: {1 / rabi_freq:.3f} ns")

    print(f"(amplitude, offset): ({popt[0]:.3f}, {popt[3]:.3f})")

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_rabi(x_fine, *popt)

    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, label="Data")
    plt.plot(x_fine, y_fine, label="Fit")
    plt.title(f"Rabi oscillation ({rabi_freq * 1e3:.3f} MHz)")
    plt.xlabel("Time / ns")
    plt.ylabel("Amplitude / a.u.")
    plt.legend()
    plt.grid(True)
    plt.show()

    return popt, pcov


def func_ramsey(
    t: npt.NDArray[np.float64],
    ampl: float,
    tau: float,
    omega: float,
    phi: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    return ampl * np.exp(-t / tau) * np.cos(omega * t + phi) + offset


def fit_ramsey(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if p0 is None:
        p0 = (
            np.abs(np.max(y) - np.min(y)) / 2,
            10_000,
            10 * 2 * np.pi / (x[-1] - x[0]),
            np.pi,
            (np.max(y) + np.min(y)) / 2,
        )

    if bounds is None:
        bounds = (
            (0, 0, 0, 0, -np.inf),
            (np.inf, np.inf, np.inf, np.pi, np.inf),
        )

    popt, pcov = curve_fit(func_ramsey, x, y, p0=p0, bounds=bounds)
    print(
        f"Fitted function: {popt[0]:.3f} * exp(-t/{popt[1]:.3f}) * cos({popt[2]:.3f} * t + {popt[3]:.3f}) + {popt[4]:.3f}"
    )

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_ramsey(x_fine, *popt)

    plt.scatter(x, y, label="Data")
    plt.plot(x_fine, y_fine, label="Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

    return popt, pcov


def func_decay(
    t: npt.NDArray[np.float64],
    ampl: float,
    tau: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    return ampl * np.exp(-t / tau) + offset


def fit_decay(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if p0 is None:
        p0 = (
            np.abs(np.max(y) - np.min(y)) / 2,
            10_000,
            (np.max(y) + np.min(y)) / 2,
        )

    if bounds is None:
        bounds = (
            (0, 0, -np.inf),
            (np.inf, np.inf, np.inf),
        )

    popt, pcov = curve_fit(func_decay, x, y, p0=p0, bounds=bounds)
    print(f"Fitted function: {popt[0]:.3f} * exp(-t/{popt[1]:.3f}) + {popt[2]:.3f}")
    print(f"Decay time: {popt[1] / 1e3:.3f} us")

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_decay(x_fine, *popt)

    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, label="Data")
    plt.plot(x_fine, y_fine, label="Fit")
    plt.title(f"Decay time: {popt[1] / 1e3:.3f} us")
    plt.xlabel("Time / ns")
    plt.ylabel("Readout signal / a.u.")
    plt.semilogx()
    plt.legend()
    plt.grid(True)
    plt.show()

    return popt, pcov


def fit_cos_and_find_minimum(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
) -> tuple[float, float]:
    def cos_func(t, ampl, omega, phi, offset):
        return ampl * np.cos(omega * t + phi) + offset

    if p0 is None:
        p0 = (
            np.abs(np.max(y) - np.min(y)) / 2,
            2 * np.pi / (x[-1] - x[0]),
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
    plt.grid(True)
    plt.show()

    print(f"Minimum: ({min_x}, {min_y})")

    return min_x, min_y


def rotate(
    data: npt.ArrayLike,
    angle: float,
) -> npt.NDArray[np.complex128]:
    points = np.array(data)
    rotated_points = points * np.exp(-1j * angle)
    return rotated_points


def get_angle(
    data: npt.ArrayLike,
) -> float:
    points = np.array(data)

    if len(points) < 2:
        return 0.0

    fit_params = np.polyfit(points.real, points.imag, 1)
    gradient, intercept = fit_params

    theta = np.arctan(gradient)
    angle = theta
    if intercept > 0:
        angle += np.pi / 2
    else:
        angle -= np.pi / 2

    return angle


def principal_components(
    iq_complex: npt.ArrayLike,
    pca=None,
) -> tuple[npt.NDArray[np.float64], PCA]:
    iq_complex = np.array(iq_complex)
    iq_vector = np.column_stack([np.real(iq_complex), np.imag(iq_complex)])
    if pca is None:
        pca = PCA(n_components=1)
    results = pca.fit_transform(iq_vector).squeeze()
    return results, pca


def find_nearest_frequency_combinations(
    target_frequency,
    lo_range=(8000, 12000),
    nco_range=(0, 3000),
    lo_step: int = 500,
    nco_step: int = 375,
) -> tuple[int, list[tuple[int, int]]]:
    # Adjust the start of the range to the nearest multiple of the step using integer division
    lo_start = ((lo_range[0] + lo_step - 1) // lo_step) * lo_step
    nco_start = ((nco_range[0] + nco_step - 1) // nco_step) * nco_step

    # Generate the possible LO and NCO frequencies based on the adjusted ranges and steps
    lo_frequencies = [freq for freq in range(lo_start, lo_range[1] + lo_step, lo_step)]
    nco_frequencies = [
        freq for freq in range(nco_start, nco_range[1] + nco_step, nco_step)
    ]

    # Initialize variables to store the best combinations and minimum difference
    best_combinations = []
    best_frequency = 0
    min_difference = float("inf")

    # Loop through each LO frequency and find the NCO frequency that makes LO - NCO closest to the target
    for lo in lo_frequencies:
        for nco in nco_frequencies:
            lo_minus_nco = lo - nco
            difference = abs(target_frequency - lo_minus_nco)

            if difference < min_difference:
                # Clear the best_combinations list, update the best_frequency, and update the minimum difference
                best_combinations = [(lo, nco)]
                best_frequency = lo_minus_nco
                min_difference = difference
            elif difference == min_difference:
                # Add the new combination to the list
                best_combinations.append((lo, nco))

    print(f"Target frequency: {target_frequency}")
    print(f"Nearest frequency: {best_frequency}")
    print(f"Combinations: {best_combinations}")

    return best_frequency, best_combinations
