"""
Data analysis functions for quantum experiments.
"""

# pylint: disable=unbalanced-tuple-unpacking

# Don't include custom modules in analysis.py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit, minimize  # type: ignore
from sklearn.decomposition import PCA  # type: ignore


def func_cos(
    t: npt.NDArray[np.float64],
    ampl: float,
    omega: float,
    phi: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate the cosine function with amplitude, frequency, phase, and offset.

    Parameters
    ----------
    t : npt.NDArray[np.float64]
        The time points at which to evaluate the cosine function.
    ampl : float
        Amplitude of the cosine wave.
    omega : float
        Angular frequency of the cosine wave.
    phi : float
        Phase shift of the cosine wave.
    offset : float
        Vertical offset of the cosine wave.

    Returns
    -------
    npt.NDArray[np.float64]
        The evaluated cosine function values at each time point.
    """
    return ampl * np.cos(omega * t + phi) + offset


def func_damped_cos(
    t: npt.NDArray[np.float64],
    tau: float,
    ampl: float,
    omega: float,
    phi: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate a damped cosine function with specified parameters.

    Parameters
    ----------
    t : npt.NDArray[np.float64]
        Time points for the function evaluation.
    tau : float
        Time constant of the exponential decay.
    ampl : float
        Amplitude of the cosine wave.
    omega : float
        Angular frequency of the cosine wave.
    phi : float
        Phase shift of the cosine wave.
    offset : float
        Vertical offset of the cosine wave.

    Returns
    -------
    npt.NDArray[np.float64]
        Evaluated damped cosine function values.
    """
    return ampl * np.exp(-t / tau) * np.cos(omega * t + phi) + offset


def fit_rabi(
    times: npt.NDArray[np.int64],
    signals: npt.NDArray[np.complex128],
    wave_count: float = 2.5,
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """
    Fit Rabi oscillation data to a cosine function and plot the results.

    Parameters
    ----------
    times : npt.NDArray[np.int64]
        Array of time points for the Rabi oscillations.
    signals : npt.NDArray[np.complex128]
        Complex signal data corresponding to the Rabi oscillations.
    wave_count : float, optional
        Initial estimate for the number of wave cycles over the time span.

    Returns
    -------
    tuple[float, float, npt.NDArray[np.float64]]
        Phase shift, fluctuation of data, and optimized parameters of fit.
    """
    # Rotate the data to the vertical (Q) axis
    phase_shift = get_angle(signals=signals)
    points = rotate(signals=signals, angle=phase_shift)
    fluctuation = float(np.std(points.real))
    print(f"Phase shift: {phase_shift:.3f} rad, {phase_shift * 180 / np.pi:.3f} deg")
    print(f"Fluctuation: {fluctuation:.3f}")

    x = times
    y = points.imag

    # Estimate the initial parameters
    omega0 = 2 * np.pi / (x[-1] - x[0])
    ampl_est = (np.max(y) - np.min(y)) / 2
    omega_est = wave_count * omega0
    phase_est = np.pi
    offset_est = (np.max(y) + np.min(y)) / 2
    p0 = (ampl_est, omega_est, phase_est, offset_est)

    bounds = (
        (0, 0, 0, -np.inf),
        (np.inf, np.inf, np.pi, np.inf),
    )

    popt, _ = curve_fit(func_cos, x, y, p0=p0, bounds=bounds)

    rabi_freq = popt[1] / (2 * np.pi)

    print(
        f"Fitted function: {popt[0]:.3f} * cos({popt[1]:.3f} * t + {popt[2]:.3f}) + {popt[3]:.3f}"
    )
    print(f"Rabi frequency: {rabi_freq * 1e3:.3f} MHz")
    print(f"Rabi period: {1 / rabi_freq:.3f} ns")

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_cos(x_fine, *popt)

    plt.figure(figsize=(8, 4))
    plt.errorbar(x, y, yerr=fluctuation, label="Data", fmt="o", color="C0")
    plt.plot(x_fine, y_fine, label="Fit", color="C0")
    plt.title(f"Rabi oscillation ({rabi_freq * 1e3:.3f} MHz)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.show()

    return phase_shift, fluctuation, popt


def fit_damped_rabi(
    times: npt.NDArray[np.int64],
    signals: npt.NDArray[np.complex128],
    wave_count: float = 2.5,
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """
    Fit damped Rabi oscillation data to a damped cosine function and plot.

    Parameters
    ----------
    times : npt.NDArray[np.int64]
        Array of time points for the Rabi oscillations.
    signals : npt.NDArray[np.complex128]
        Complex signal data corresponding to the Rabi oscillations.
    wave_count : float, optional
        Estimate for the number of wave cycles over the time span.

    Returns
    -------
    tuple[float, float, npt.NDArray[np.float64]]
        Phase shift, fluctuation of data, and optimized fit parameters.
    """
    # Rotate the data to the vertical (Q) axis
    phase_shift = get_angle(signals=signals)
    points = rotate(signals=signals, angle=phase_shift)
    fluctuation = float(np.std(points.real))
    print(f"Phase shift: {phase_shift:.3f} rad, {phase_shift * 180 / np.pi:.3f} deg")
    print(f"Fluctuation: {fluctuation:.3f}")

    x = times
    y = points.imag

    # Estimate the initial parameters
    omega0 = 2 * np.pi / (x[-1] - x[0])
    ampl_est = (np.max(y) - np.min(y)) / 2
    tau_est = 10_000
    omega_est = wave_count * omega0
    phase_est = np.pi
    offset_est = (np.max(y) + np.min(y)) / 2
    p0 = (ampl_est, tau_est, omega_est, phase_est, offset_est)

    bounds = (
        (0, 0, 0, 0, -np.inf),
        (np.inf, np.inf, np.inf, np.pi, np.inf),
    )

    popt, _ = curve_fit(func_damped_cos, x, y, p0=p0, bounds=bounds)

    rabi_freq = popt[2] / (2 * np.pi)

    print(
        f"Fitted function: {popt[0]:.3f} * exp(-t/{popt[1]:.3f}) * cos({popt[2]:.3f} * t + {popt[3]:.3f}) + {popt[4]:.3f}"
    )
    print(f"Rabi frequency: {rabi_freq * 1e3:.3f} MHz")
    print(f"Rabi period: {1 / rabi_freq:.3f} ns")

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_damped_cos(x_fine, *popt)

    plt.figure(figsize=(8, 4))
    plt.errorbar(x, y, yerr=fluctuation, label="Data", fmt="o", color="C0")
    plt.plot(x_fine, y_fine, label="Fit", color="C0")
    plt.title(f"Rabi oscillation ({rabi_freq * 1e3:.3f} MHz)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.show()

    return phase_shift, fluctuation, popt


def fit_ramsey(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Fit Ramsey fringes using a damped cosine function and plot the results.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Array of time points for the Ramsey fringes.
    y : npt.NDArray[np.float64]
        Amplitude data for the Ramsey fringes.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Optimized fit parameters and covariance of the fit.
    """
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

    popt, pcov = curve_fit(func_damped_cos, x, y, p0=p0, bounds=bounds)
    print(
        f"Fitted function: {popt[0]:.3f} * exp(-t/{popt[1]:.3f}) * cos({popt[2]:.3f} * t + {popt[3]:.3f}) + {popt[4]:.3f}"
    )

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_damped_cos(x_fine, *popt)

    plt.scatter(x, y, label="Data")
    plt.plot(x_fine, y_fine, label="Fit")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.show()

    return popt, pcov


def func_decay(
    t: npt.NDArray[np.float64],
    ampl: float,
    tau: float,
    offset: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate an exponential decay function with given parameters.

    Parameters
    ----------
    t : npt.NDArray[np.float64]
        Time points for the function evaluation.
    ampl : float
        Amplitude of the exponential decay.
    tau : float
        Time constant of the exponential decay.
    offset : float
        Vertical offset of the decay curve.

    Returns
    -------
    npt.NDArray[np.float64]
        Evaluated exponential decay function values.
    """
    return ampl * np.exp(-t / tau) + offset


def fit_decay(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Fit decay data to an exponential decay function and plot the results.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Time points for the decay data.
    y : npt.NDArray[np.float64]
        Amplitude data for the decay.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Optimized fit parameters and covariance of the fit.
    """
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
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.semilogx()
    plt.legend()
    plt.show()

    return popt, pcov


def fit_cos_and_find_minimum(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
) -> tuple[float, float]:
    """
    Fit data to a cosine function and find the minimum of the fit.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Time points for the decay data.
    y : npt.NDArray[np.float64]
        Amplitude data for the decay.
    p0 : optional
        Initial guess for the fitting parameters.

    Returns
    -------
    tuple[float, float]
        Time and amplitude of the minimum of the fit.
    """

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


def fit_and_rotate(
    data: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """
    Rotate complex data points based on the angle determined by linear fit.

    Parameters
    ----------
    data : npt.ArrayLike
        Array of complex data points to be rotated.

    Returns
    -------
    npt.NDArray[np.complex128]
        Rotated complex data points.
    """
    points = np.array(data)
    angle = get_angle(points)
    rotated_points = rotate(points, angle)
    return rotated_points


def rotate(
    signals: npt.ArrayLike,
    angle: float,
) -> npt.NDArray[np.complex128]:
    """
    Rotate complex data points by a specified angle.

    Parameters
    ----------
    signals : npt.ArrayLike
        Array of complex data points to be rotated.
    angle : float
        Angle in radians by which to rotate the data points.

    Returns
    -------
    npt.NDArray[np.complex128]
        Rotated complex data points.
    """
    points = np.array(signals)
    rotated_points = points * np.exp(-1j * angle)
    return rotated_points


def get_angle(
    signals: npt.ArrayLike,
) -> float:
    """
    Determine the angle of a linear fit to the complex data points.

    Parameters
    ----------
    signals : npt.ArrayLike
        Array of complex data points to be rotated.

    Returns
    -------
    float
        Angle in radians of the linear fit to the data points.
    """
    iq_complex = np.array(signals)
    if len(iq_complex) < 2:
        return 0.0
    iq_vector = np.column_stack([iq_complex.real, iq_complex.imag])
    pca = PCA(n_components=1).fit(iq_vector)
    first_component = pca.components_[0]
    gradient = first_component[1] / first_component[0]
    mean = np.mean(iq_vector, axis=0)
    intercept = mean[1] - gradient * mean[0]
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
    """
    Perform PCA on complex IQ data and return the principal components.

    Parameters
    ----------
    iq_complex : npt.ArrayLike
        Array of complex IQ data.
    pca : PCA, optional
        Predefined PCA object, if available.

    Returns
    -------
    tuple[npt.NDArray[np.float64], PCA]
        Principal component values and the PCA object used.
    """
    iq_complex = np.array(iq_complex)
    iq_vector = np.column_stack([np.real(iq_complex), np.imag(iq_complex)])
    if pca is None:
        pca = PCA(n_components=1)
    results = pca.fit_transform(iq_vector).squeeze()
    return results, pca


def fit_chevron(
    center_frequency: float,
    freq_range: npt.NDArray[np.float64],
    time_range: npt.NDArray[np.int64],
    signals: list[npt.NDArray[np.float64]],
):
    """
    Plot Chevron patterns, perform Fourier analysis, and fit the data.

    Parameters
    ----------
    center_frequency : float
        Central frequency around which the Chevron patterns are analyzed.
    freq_range : npt.NDArray[np.float64]
        Frequency range for the Chevron analysis.
    time_range : npt.NDArray[np.int64]
        Time range for the Chevron analysis.
    signals : list[npt.NDArray[np.float64]]
        Signal data for different frequencies in the Chevron analysis.

    Returns
    -------
    None
        The function plots the Chevron patterns and their Fourier analysis.
    """
    time, freq = np.meshgrid(time_range, center_frequency + freq_range * 1e6)
    plt.pcolor(time, freq * 1e-6, signals)
    plt.xlabel("Pulse length (ns)")
    plt.ylabel("Drive frequency (MHz)")
    plt.show()

    length = 2**10
    dt = (time_range[1] - time_range[0]) * 1e-9
    freq_rabi_range = np.linspace(0, 0.5 / dt, length // 2)

    fourier_values = []

    for s in signals:
        s = s - np.average(s)
        signal_zero_filled = np.append(s, np.zeros(length - len(s)))
        fourier = np.abs(np.fft.fft(signal_zero_filled))[: length // 2]
        fourier_values.append(fourier)

    freq_ctrl_range = center_frequency + freq_range * 1e6
    grid_rabi, grid_ctrl = np.meshgrid(freq_rabi_range, freq_ctrl_range)
    plt.pcolor(grid_rabi * 1e-6, grid_ctrl * 1e-6, fourier_values)
    plt.xlabel("Rabi frequency (MHz)")
    plt.ylabel("Drive frequency (MHz)")
    plt.show()

    buf = []
    for f in fourier_values:
        max_index = np.argmax(f)
        buf.append(freq_rabi_range[max_index])
    freq_rabi = np.array(buf)

    def func(f_ctrl, f_rabi, f_reso, coeff):
        return coeff * np.sqrt(f_rabi**2 + (f_ctrl - f_reso) ** 2)

    p0 = [10.0e6, 8000.0e6, 1.0]

    freq_ctrl = center_frequency + freq_range * 1e6

    popt, _ = curve_fit(func, freq_ctrl, freq_rabi, p0=p0, maxfev=100000)

    f_rabi = popt[0]
    f_reso = popt[1]
    coeff = popt[2]  # ge: 1, ef: sqrt(2)

    print(
        f"f_reso = {f_reso * 1e-6:.2f} MHz, f_rabi = {f_rabi * 1e-6:.2f} MHz, coeff = {coeff:.2f}"
    )

    freq_ctrl_fine = np.linspace(np.min(freq_ctrl), np.max(freq_ctrl), 1000)
    freq_rabi_fit = func(freq_ctrl_fine, *popt)

    plt.scatter(freq_ctrl * 1e-6, freq_rabi * 1e-6, label="Data")
    plt.plot(freq_ctrl_fine * 1e-6, freq_rabi_fit * 1e-6, label="Fit")

    plt.xlabel("Drive frequency (MHz)")
    plt.ylabel("Rabi frequency (MHz)")
    plt.legend()
    plt.show()
