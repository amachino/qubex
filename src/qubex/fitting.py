from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit, minimize
from sklearn.decomposition import PCA


@dataclass
class RabiParam:
    """
    Data class representing the parameters of Rabi oscillation.

    ```
    rabi = amplitude * cos(2π * frequency * time + phase) + offset) + noise
    ```

    Attributes
    ----------
    target : str
        Identifier of the target.
    amplitude : float
        Amplitude of the Rabi oscillation.
    frequency : float
        Frequency of the Rabi oscillation.
    phase : float
        Initial phase of the Rabi oscillation.
    offset : float
        Vertical offset of the Rabi oscillation.
    noise : float
        Fluctuation of the Rabi oscillation.
    angle : float
        Angle of the Rabi oscillation.
    """

    target: str
    amplitude: float
    frequency: float
    phase: float
    offset: float
    noise: float
    angle: float


def func_cos(
    t: npt.NDArray[np.float64],
    A: float,
    omega: float,
    phi: float,
    C: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate a cosine function with given parameters.

    Parameters
    ----------
    t : npt.NDArray[np.float64]
        Time points for the function evaluation.
    A : float
        Amplitude of the cosine function.
    omega : float
        Angular frequency of the cosine function.
    phi : float
        Phase offset of the cosine function.
    C : float
        Vertical offset of the cosine function.
    """
    return A * np.cos(omega * t + phi) + C


def func_damped_cos(
    t: npt.NDArray[np.float64],
    A: float,
    omega: float,
    phi: float,
    C: float,
    tau: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate a damped cosine function with given parameters.

    Parameters
    ----------
    t : npt.NDArray[np.float64]
        Time points for the function evaluation.
    A : float
        Amplitude of the cosine function.
    omega : float
        Angular frequency of the cosine function.
    phi : float
        Phase offset of the cosine function.
    C : float
        Vertical offset of the cosine function.
    tau : float
        Time constant of the exponential damping.
    """
    return A * np.exp(-t / tau) * np.cos(omega * t + phi) + C


def func_exp_decay(
    t: npt.NDArray[np.float64],
    A: float,
    tau: float,
    C: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate an exponential decay function with given parameters.

    Parameters
    ----------
    t : npt.NDArray[np.float64]
        Time points for the function evaluation.
    A : float
        Amplitude of the exponential decay.
    tau : float
        Time constant of the exponential decay.
    C : float
        Vertical offset of the exponential decay.
    """
    return A * np.exp(-t / tau) + C


def fit_rabi(
    *,
    target: str,
    times: npt.NDArray[np.float64],
    data: npt.NDArray[np.complex64],
    wave_count: float | None = None,
    plot: bool = True,
    is_damped: bool = False,
) -> RabiParam:
    """
    Fit Rabi oscillation data to a cosine function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    times : npt.NDArray[np.float64]
        Array of time points for the Rabi oscillations.
    data : npt.NDArray[np.complex128]
        Complex signal data corresponding to the Rabi oscillations.
    wave_count : float, optional
        Initial estimate for the number of wave cycles over the time span.
    plot : bool, optional
        Whether to plot the data and the fit.
    is_damped : bool, optional
        Whether to fit the data to a damped cosine function.

    Returns
    -------
    RabiParam
        Data class containing the parameters of the Rabi oscillation.
    """
    print(f"Target: {target}")

    # Rotate the data to the vertical (Q) axis
    angle = get_angle(data)
    rotated = rotate(data, -angle)
    noise = float(np.std(rotated.real))

    x = times
    y = rotated.imag

    # Estimate the initial parameters
    wave_count_est = estimate_wave_count(x, y) if wave_count is None else wave_count
    amplitude_est = (np.max(y) - np.min(y)) / 2
    omega_est = 2 * np.pi * wave_count_est / (x[-1] - x[0])
    phase_est = 0.0
    offset_est = (np.max(y) + np.min(y)) / 2

    try:
        p0: tuple
        bounds: tuple
        if is_damped:
            tau_est = 10_000
            p0 = (amplitude_est, omega_est, phase_est, offset_est, tau_est)
            bounds = (
                (0, 0, 0, -np.inf, 0),
                (np.inf, np.inf, np.pi, np.inf, np.inf),
            )
            popt, _ = curve_fit(func_damped_cos, x, y, p0=p0, bounds=bounds)
            print(
                f"Fitted function: {popt[0]:.3g} * exp(-t/{popt[4]:.3g}) * cos({popt[1]:.3g} * t + {popt[2]:.3g}) + {popt[3]:.3g} ± {noise:.3g}"
            )
        else:
            p0 = (amplitude_est, omega_est, phase_est, offset_est)
            bounds = (
                (0, 0, 0, -np.inf),
                (np.inf, np.inf, np.pi, np.inf),
            )
            popt, _ = curve_fit(func_cos, x, y, p0=p0, bounds=bounds)
            print(
                f"Fitted function: {popt[0]:.3g} * cos({popt[1]:.3g} * t + {popt[2]:.3g}) + {popt[3]:.3g} ± {noise:.3g}"
            )
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return RabiParam(target, 0.0, 0.0, 0.0, 0.0, noise, angle)

    amplitude = popt[0]
    omega = popt[1]
    phase = popt[2]
    offset = popt[3]
    frequency = omega / (2 * np.pi)

    print(f"Phase shift: {angle:.3g} rad, {angle * 180 / np.pi:.3g} deg")
    print(f"Rabi frequency: {frequency * 1e3:.3g} MHz")
    print(f"Rabi period: {1 / frequency:.3g} ns")

    if plot:
        x_fine = np.linspace(np.min(x), np.max(x), 1000)
        y_fine = (
            func_cos(x_fine, *popt) if not is_damped else func_damped_cos(x_fine, *popt)
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_fine,
                y=y_fine,
                mode="lines",
                name="Fit",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data",
                error_y=dict(type="constant", value=noise),
            ),
        )
        fig.update_layout(
            title=(f"Rabi oscillation of {target} : {frequency * 1e3:.3g} MHz"),
            xaxis_title="Drive time (ns)",
            yaxis_title="Amplitude (arb. units)",
        )
        fig.show()

    return RabiParam(
        target=target,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        offset=offset,
        noise=noise,
        angle=angle,
    )


def fit_detuned_rabi(
    *,
    target: str,
    control_frequencies: npt.NDArray[np.float64],
    rabi_frequencies: npt.NDArray[np.float64],
    plot: bool = True,
) -> tuple[float, float]:
    """
    Fit detuned Rabi oscillation data to a cosine function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    control_frequencies : npt.NDArray[np.float64]
        Array of control frequencies for the Rabi oscillations in GHz.
    rabi_frequencies : npt.NDArray[np.float64]
        Rabi frequencies corresponding to the control frequencies in GHz.
    plot : bool, optional
        Whether to plot the data and the fit.

    Returns
    -------
    tuple[float, float]
        Resonance frequency and Rabi frequency of the detuned Rabi oscillation.
    """

    def func(f_control, f_resonance, f_rabi):
        return np.sqrt(f_rabi**2 + (f_control - f_resonance) ** 2)

    popt, _ = curve_fit(func, control_frequencies, rabi_frequencies)

    f_resonance = popt[0]
    f_rabi = popt[1]

    x = control_frequencies
    y = rabi_frequencies * 1e3

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func(x_fine, *popt) * 1e3

    if plot:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_fine,
                y=y_fine,
                mode="lines",
                name="fit",
            )
        )
        fig.add_annotation(
            x=f_resonance,
            y=np.abs(f_rabi * 1e3),
            text=f"min: {f_resonance:.6f} GHz",
            showarrow=True,
            arrowhead=1,
        )
        fig.update_layout(
            title=f"Detuned Rabi oscillation of {target}",
            xaxis_title="Control frequency (GHz)",
            yaxis_title="Rabi frequency (MHz)",
        )
        fig.show()

    print(f"Resonance frequency: {f_resonance:.6f} GHz")

    return f_resonance, f_rabi


def fit_ramsey(
    *,
    target: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
    title: str = "Ramsey fringe",
    xaxis_title: str = "Time (μs)",
    yaxis_title: str = "Amplitude (arb. units)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> tuple[float, float]:
    """
    Fit Ramsey fringe using a damped cosine function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    x : npt.NDArray[np.float64]
        Array of time points for the Ramsey fringe.
    y : npt.NDArray[np.float64]
        Amplitude data for the Ramsey fringe.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.

    Returns
    -------
    tuple[float, float]
        Decay time and frequency of the Ramsey fringe.
    """
    wave_count_est = estimate_wave_count(x, y)
    amplitude_est = (np.max(y) - np.min(y)) / 2
    omega_est = 2 * np.pi * wave_count_est / (x[-1] - x[0])
    phase_est = 0.0
    offset_est = 0.0
    tau_est = 10_000

    if p0 is None:
        p0 = (amplitude_est, omega_est, phase_est, offset_est, tau_est)

    if bounds is None:
        bounds = (
            (0, omega_est * 0.9, 0, -np.inf, 0),
            (np.inf, omega_est * 1.1, np.pi, np.inf, np.inf),
        )

    popt, _ = curve_fit(func_damped_cos, x, y, p0=p0, bounds=bounds)

    A = popt[0]
    omega = popt[1]
    phi = popt[2]
    C = popt[3]
    tau = popt[4]
    f = omega / (2 * np.pi)

    print(
        f"Fitted function: {A:.3g} * exp(-t/{tau:.3g}) * cos({omega:.3g} * t + {phi:.3g}) + {C:.3g}"
    )
    print(f"Decay time: {tau * 1e-3:.3g} μs")
    print(f"Frequency: {f * 1e3:.3g} MHz")

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_damped_cos(x_fine, *popt)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fine * 1e-3,
            y=y_fine,
            mode="lines",
            name="Fit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x * 1e-3,
            y=y,
            mode="markers",
            name="Data",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"τ = {tau * 1e-3:.3g} μs, f = {f * 1e3:.3g} MHz",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )
    fig.show()

    return tau, f


def fit_exp_decay(
    *,
    target: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
    title: str = "Decay time",
    xaxis_title: str = "Time (μs)",
    yaxis_title: str = "Amplitude (arb. units)",
    xaxis_type: Literal["linear", "log"] = "log",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> float:
    """
    Fit decay data to an exponential decay function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
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
    float
        Decay time of the exponential decay in microseconds.
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

    popt, _ = curve_fit(func_exp_decay, x, y, p0=p0, bounds=bounds)
    A = popt[0]
    tau = popt[1]
    C = popt[2]
    print(f"Fitted function: {A:.3g} * exp(-t/{tau:.3g}) + {C:.3g}")
    print(f"Decay time: {tau * 1e-3:.3g} μs")

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_exp_decay(x_fine, *popt)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fine * 1e-3,
            y=y_fine,
            mode="lines",
            name="Fit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x * 1e-3,
            y=y,
            mode="markers",
            name="Data",
        )
    )
    fig.update_layout(
        title=f"{title} = {tau * 1e-3:.3g} μs : {target}",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )
    fig.show()

    return tau


def fit_ampl_calib_data(
    target: str,
    amplitude: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    p0=None,
) -> tuple[float, float]:
    """
    Fit amplitude calibration data to a cosine function and plot the results.

    Parameters
    ----------
    amplitude : npt.NDArray[np.float64]
        Amplitude range for the calibration data.
    data : npt.NDArray[np.float64]
        Measured values for the calibration data.
    p0 : optional
        Initial guess for the fitting parameters.

    Returns
    -------
    tuple[float, float]
        Minimum amplitude and value of the fitted cosine function.
    """

    def cos_func(t, ampl, omega, phi, offset):
        return ampl * np.cos(omega * t + phi) + offset

    if p0 is None:
        p0 = (
            np.abs(np.max(data) - np.min(data)) / 2,
            2 * np.pi / (amplitude[-1] - amplitude[0]),
            np.pi,
            (np.max(data) + np.min(data)) / 2,
        )

    popt, _ = curve_fit(cos_func, amplitude, data, p0=p0)
    print(
        f"Fitted function: {popt[0]:.3g} * cos({popt[1]:.3g} * t + {popt[2]:.3g}) + {popt[3]:.3g}"
    )

    result = minimize(
        cos_func,
        x0=np.mean(amplitude),
        args=tuple(popt),
        bounds=[(np.min(amplitude), np.max(amplitude))],
    )
    min_x = result.x[0]
    min_y = cos_func(min_x, *popt)

    x_fine = np.linspace(np.min(amplitude), np.max(amplitude), 1000)
    y_fine = cos_func(x_fine, *popt)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=y_fine,
            mode="lines",
            name="Fit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=amplitude,
            y=data,
            mode="markers",
            name="Data",
        )
    )
    fig.add_annotation(
        x=min_x,
        y=min_y,
        text=f"min: {min_x:.6g}",
        showarrow=True,
        arrowhead=1,
    )
    fig.update_layout(
        title=f"Amplitude calibration of {target}",
        xaxis_title="Amplitude (arb. units)",
        yaxis_title="Measured value (arb. units)",
    )
    fig.show()

    print(f"Calibrated amplitude: {min_x:.6g}")

    return min_x, min_y


def fit_chevron(
    center_frequency: float,
    freq_range: npt.NDArray[np.float64],
    time_range: npt.NDArray[np.float64],
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
    time_range : npt.NDArray[np.float64]
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


def rotate(
    data: npt.ArrayLike,
    angle: float,
) -> npt.NDArray[np.complex128]:
    """
    Rotate complex data points by a specified angle.

    Parameters
    ----------
    data : npt.ArrayLike
        Array of complex data points to be rotated.
    angle : float
        Angle in radians by which to rotate the data points.

    Returns
    -------
    npt.NDArray[np.complex128]
        Rotated complex data points.
    """
    points = np.array(data)
    rotated_points = points * np.exp(1j * angle)
    return rotated_points


def get_angle(
    data: npt.ArrayLike,
) -> float:
    """
    Determine the angle of a linear fit to the complex data points.

    Parameters
    ----------
    data : npt.ArrayLike
        Array of complex data points to be rotated.

    Returns
    -------
    float
        Angle in radians of the linear fit to the data points.
    """
    data_complex = np.array(data, dtype=np.complex128)
    if len(data_complex) < 2:
        return 0.0
    data_vector = np.column_stack([data_complex.real, data_complex.imag])
    pca = PCA(n_components=1).fit(data_vector)
    first_component = pca.components_[0]
    gradient = first_component[1] / first_component[0]
    mean = np.mean(data_vector, axis=0)
    intercept = mean[1] - gradient * mean[0]
    angle = np.arctan(gradient)
    if intercept > 0:
        angle += np.pi / 2
    else:
        angle -= np.pi / 2
    return angle


def estimate_wave_count(times, data) -> float:
    N = len(times)
    dt = times[1] - times[0]
    F = np.array(fft(data))
    f = np.array(fftfreq(N, dt)[1 : N // 2])
    i = np.argmax(np.abs(F[1 : N // 2]))
    dominant_freq = np.abs(f[i])
    wave_count_est = dominant_freq * (times[-1] - times[0])
    return wave_count_est
