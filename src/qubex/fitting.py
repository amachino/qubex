from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit, minimize  # type: ignore
from sklearn.decomposition import PCA  # type: ignore


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
    times: npt.NDArray[np.int64],
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
    times : npt.NDArray[np.int64]
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
                marker_color="black",
                marker_line_width=2,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data",
                error_y=dict(type="constant", value=noise),
                marker=dict(color="#636EFA", size=5),
            ),
        )
        fig.update_layout(
            title=(f"Rabi oscillation of {target} : {frequency * 1e3:.3g} MHz"),
            xaxis_title="Time (ns)",
            yaxis_title="Amplitude (arb. units)",
            width=600,
            height=300,
            showlegend=True,
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


def fit_ramsey(
    *,
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
        f"Fitted function: {popt[0]:.3g} * exp(-t/{popt[1]:.3g}) * cos({popt[2]:.3g} * t + {popt[3]:.3g}) + {popt[4]:.3g}"
    )

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_damped_cos(x_fine, *popt)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker_color="black",
            marker_size=10,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=y_fine,
            mode="lines",
            name="Fit",
            marker_color="black",
            marker_line_width=2,
        )
    )
    fig.update_layout(
        title="Decay Fit",
        xaxis_title="Time (ns)",
        yaxis_title="Amplitude (arb. units)",
        showlegend=True,
    )
    fig.show()

    return popt, pcov


def fit_exp_decay(
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

    popt, pcov = curve_fit(func_exp_decay, x, y, p0=p0, bounds=bounds)
    print(f"Fitted function: {popt[0]:.3g} * exp(-t/{popt[1]:.3g}) + {popt[2]:.3g}")
    print(f"Decay time: {popt[1] / 1e3:.3g} us")

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_exp_decay(x_fine, *popt)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker_color="black",
            marker_size=10,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=y_fine,
            mode="lines",
            name="Fit",
            marker_color="black",
            marker_line_width=2,
        )
    )
    fig.update_layout(
        title=f"Decay time: {popt[1] / 1e3:.3g} us",
        xaxis_title="Time (ns)",
        yaxis_title="Amplitude (arb. units)",
        showlegend=True,
    )
    fig.show()

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
        f"Fitted function: {popt[0]:.3g} * cos({popt[1]:.3g} * t + {popt[2]:.3g}) + {popt[3]:.3g}"
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

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker_color="black",
            marker_size=10,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=y_fine,
            mode="lines",
            name="Fit",
            marker_color="black",
            marker_line_width=2,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_x],
            y=[min_y],
            mode="markers",
            name="Minimum",
            marker_color="red",
            marker_size=10,
        )
    )
    fig.update_layout(
        title="Fit and Minimum",
        xaxis_title="Time (ns)",
        yaxis_title="Amplitude (arb. units)",
        showlegend=True,
    )
    fig.show()

    print(f"Minimum: ({min_x}, {min_y})")

    return min_x, min_y


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
