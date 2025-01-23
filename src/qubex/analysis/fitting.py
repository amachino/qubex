from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit, least_squares, minimize
from sklearn.decomposition import PCA

COLORS = [
    "#0C5DA5",
    "#00B945",
    "#FF9500",
    "#FF2C00",
    "#845B97",
    "#474747",
    "#9e9e9e",
]


def _plotly_config(filename: str) -> dict:
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"{prefix}_{filename}",
            "scale": 3,
        },
    }


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


def normalize(
    values: npt.NDArray,
    param: RabiParam,
    centers: dict[int, complex] | None = None,
) -> npt.NDArray:
    """
    Normalizes the measured I/Q value.

    Parameters
    ----------
    values : npt.NDArray
        Measured I/Q value.
    param : RabiParam
        Parameters of the Rabi oscillation.
    centers : dict[int, complex], optional
        Centers of the |g> and |e> states.

    Returns
    -------
    npt.NDArray
        Normalized I/Q value.
    """
    # if centers is not None:
    #     p = np.array(values, dtype=np.complex128)
    #     g, e = centers[0], centers[1]
    #     v_ge = e - g
    #     v_gp = p - g
    #     v_gp_proj = np.real(v_gp * np.conj(v_ge)) / np.abs(v_ge)
    #     normalized = 1 - 2 * np.abs(v_gp_proj) / np.abs(v_ge)
    # else:
    #     rotated = values * np.exp(-1j * param.angle)
    #     normalized = (np.imag(rotated) - param.offset) / param.amplitude
    rotated = values * np.exp(-1j * param.angle)
    normalized = (np.imag(rotated) - param.offset) / param.amplitude
    return normalized


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


def func_lorentzian(
    f: npt.NDArray[np.float64],
    A: float,
    f0: float,
    gamma: float,
    C: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate a Lorentzian function with given parameters.

    Parameters
    ----------
    f : npt.NDArray[np.float64]
        Frequency points for the function evaluation.
    A : float
        Amplitude of the Lorentzian function.
    f0 : float
        Central frequency of the Lorentzian function.
    gamma : float
        Width of the Lorentzian function.
    C : float
        Vertical offset of the Lorentzian function.
    """
    return A / (1 + ((f - f0) / gamma) ** 2) + C


def func_sqrt_lorentzian(
    f: npt.NDArray[np.float64],
    A: float,
    f0: float,
    Omega: float,
    C: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate a square root Lorentzian function with given parameters.

    Parameters
    ----------
    f : npt.NDArray[np.float64]
        Frequency points for the function evaluation.
    A : float
        Amplitude of the square root Lorentzian function.
    f0 : float
        Central frequency of the square root Lorentzian function.
    Omega : float
        Width of the square root Lorentzian function.
    C : float
        Vertical offset of the square root Lorentzian function.
    """
    return A / np.sqrt(1 + ((f - f0) / Omega) ** 2) + C


def func_resonance(f, f_r, kappa_ex, kappa_in, A, phi):
    """
    Calculate a resonance function with given parameters.

    Parameters
    ----------
    f : npt.NDArray[np.float64]
        Frequency points for the function evaluation.
    f_r : float
        Resonance frequency.
    kappa_ex : float
        External loss rate.
    kappa_in : float
        Internal loss rate.
    """
    numerator = (f - f_r) * 1j + (-kappa_ex + kappa_in) / 2
    denominator = (f - f_r) * 1j + (kappa_ex + kappa_in) / 2
    return A * np.exp(1j * phi) * (numerator / denominator)


def fit_polynomial(
    *,
    target: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    degree: int,
    plot: bool = True,
    title: str = "Polynomial fit",
    xaxis_title: str = "x",
    yaxis_title: str = "y",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit data to a polynomial function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    x : npt.NDArray[np.float64]
        x values for the data.
    y : npt.NDArray[np.float64]
        y values for the data.
    degree : int
        Degree of the polynomial.
    plot : bool, optional
        Whether to plot the data and the fit.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Label for the x-axis.
    yaxis_title : str, optional
        Label for the y-axis.
    xaxis_type : Literal["linear", "log"], optional
        Type of the x-axis.
    yaxis_type : Literal["linear", "log"], optional
        Type of the y-axis.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """
    popt = np.polyfit(x, y, degree)
    fun = np.poly1d(popt)
    y_fit = fun(x)

    roots = np.roots(fun)
    real_roots = roots[np.isreal(roots)].real
    roots_in_range = real_roots[(real_roots >= np.min(x)) & (real_roots <= np.max(x))]
    try:
        root = roots_in_range[np.argmin(np.abs(roots_in_range))]
    except ValueError:
        print(f"No root found in the range ({np.min(x)}, {np.max(x)}).")
        root = np.nan

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
            x=x,
            y=y_fit,
            mode="lines",
            name="Fit",
        )
    )
    if not np.isnan(root):
        fig.add_annotation(
            x=root,
            y=fun(root),
            text=f"root: {root:.3g}",
            showarrow=True,
            arrowhead=1,
        )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        fig.show(config=_plotly_config(f"polynomial_{target}"))

    return {
        "popt": popt,
        "fun": fun,
        "root": root,
        "fig": fig,
    }


def fit_cosine(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    title: str = "Cosine fit",
    xaxis_title: str = "x",
    yaxis_title: str = "y",
    plot: bool = True,
) -> dict:
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    wave_count_est = estimate_wave_count(x, y)
    amplitude_est = (np.max(y) - np.min(y)) / 2
    omega_est = 2 * np.pi * wave_count_est / (x[-1] - x[0])
    phase_est = 0.0
    offset_est = (np.max(y) + np.min(y)) / 2

    p0 = (amplitude_est, omega_est, phase_est, offset_est)
    bounds = (
        (0, 0, 0, -np.inf),
        (np.inf, np.inf, np.pi, np.inf),
    )
    popt, _ = curve_fit(func_cos, x, y, p0=p0, bounds=bounds)

    amplitude = popt[0]
    omega = popt[1]
    phase = popt[2]
    offset = popt[3]
    frequency = omega / (2 * np.pi)

    if plot:
        x_fine = np.linspace(np.min(x), np.max(x), 1000)
        y_fine = func_cos(x_fine, *popt)

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
            ),
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text=f"f = {frequency:.6f}",
            showarrow=False,
        )
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )
        fig.show(config=_plotly_config("cosine_fit"))

    return {
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
        "offset": offset,
        "popt": popt,
        "fig": fig,
    }


def fit_rabi(
    *,
    target: str,
    times: npt.NDArray[np.float64],
    data: npt.NDArray[np.complex64],
    wave_count: float | None = None,
    plot: bool = True,
    is_damped: bool = False,
    yaxis_title: str | None = None,
    yaxis_range: tuple[float, float] | None = None,
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
    data = np.array(data, dtype=np.complex64)

    # Rotate the data to align the Q axis (|g>: +Q, |e>: -Q)
    if len(data) < 2:
        angle = 0.0
    else:
        data_vec = np.column_stack([data.real, data.imag])
        pca = PCA(n_components=2).fit(data_vec)
        start_point = data_vec[0]
        mean_point = pca.mean_
        data_direction = mean_point - start_point
        principal_component = pca.components_[0]
        dot_product = np.dot(data_direction, principal_component)
        ge_vector = principal_component if dot_product > 0 else -principal_component
        angle_ge = np.arctan2(ge_vector[1], ge_vector[0])
        angle = angle_ge + np.pi / 2

    rotated = data * np.exp(-1j * angle)
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
            # print(
            #     f"Fitted function: {popt[0]:.3g} * exp(-t/{popt[4]:.3g}) * cos({popt[1]:.3g} * t + {popt[2]:.3g}) + {popt[3]:.3g} ± {noise:.3g}"
            # )
        else:
            p0 = (amplitude_est, omega_est, phase_est, offset_est)
            bounds = (
                (0, 0, 0, -np.inf),
                (np.inf, np.inf, np.pi, np.inf),
            )
            popt, _ = curve_fit(func_cos, x, y, p0=p0, bounds=bounds)
            # print(
            #     f"Fitted function: {popt[0]:.3g} * cos({popt[1]:.3g} * t + {popt[2]:.3g}) + {popt[3]:.3g} ± {noise:.3g}"
            # )
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return RabiParam(target, np.nan, np.nan, np.nan, np.nan, noise, angle)

    amplitude = popt[0]
    omega = popt[1]
    phase = popt[2]
    offset = popt[3]
    frequency = omega / (2 * np.pi)

    tau = popt[4] if is_damped else None

    # print(f"Phase shift: {angle:.3g} rad, {angle * 180 / np.pi:.3g} deg")
    print(f"Rabi frequency: {frequency * 1e3:.6g} MHz")
    # print(f"Rabi period: {1 / frequency:.3g} ns")

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
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text=f"f = {frequency * 1e3:.2f} MHz"
            + (f", τ = {tau * 1e-3:.2f} μs" if tau else ""),
            showarrow=False,
        )
        fig.update_layout(
            title=(f"Rabi oscillation : {target}"),
            xaxis_title="Drive duration (ns)",
            yaxis_title=yaxis_title or "Signal (arb. unit)",
            yaxis_range=yaxis_range,
        )
        fig.show(config=_plotly_config(f"rabi_{target}"))

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
) -> dict:
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
    dict
        Fitted parameters and the figure.
    """

    def func(f_control, f_resonance, f_rabi):
        return np.sqrt(f_rabi**2 + (f_control - f_resonance) ** 2)

    try:
        popt, pcov = curve_fit(func, control_frequencies, rabi_frequencies)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    f_resonance, f_rabi = popt
    f_resonance_err, f_rabi_err = np.sqrt(np.diag(pcov))

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
            text=f"min: {f_resonance:.6f} ± {f_resonance_err:.6f} GHz",
            showarrow=True,
            arrowhead=1,
        )
        fig.update_layout(
            title=f"Detuned Rabi oscillation : {target}",
            xaxis_title="Drive frequency (GHz)",
            yaxis_title="Rabi frequency (MHz)",
        )
        fig.show(config=_plotly_config(f"detuned_rabi_{target}"))

    print("Resonance frequency")
    print(f"  {target}: {f_resonance:.6f}")

    return {
        "f_resonance": f_resonance,
        "f_resonance_err": f_resonance_err,
        "f_rabi": f_rabi,
        "f_rabi_err": f_rabi_err,
        "fig": fig,
    }


def fit_ramsey(
    *,
    target: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
    plot: bool = True,
    title: str = "Ramsey fringe",
    xaxis_title: str = "Time (μs)",
    yaxis_title: str = "Signal (arb. unit)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
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
    plot : bool, optional
        Whether to plot the data and the fit.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Label for the x-axis.
    yaxis_title : str, optional
        Label for the y-axis.
    xaxis_type : Literal["linear", "log"], optional
        Type of the x-axis.
    yaxis_type : Literal["linear", "log"], optional
        Type of the y-axis.

    Returns
    -------
    dict
        Fitted parameters and the figure.
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

    try:
        popt, pcov = curve_fit(func_damped_cos, x, y, p0=p0, bounds=bounds)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    A, omega, phi, C, tau = popt
    A_err, omega_err, phi_err, C_err, tau_err = np.sqrt(np.diag(pcov))
    f = omega / (2 * np.pi)
    f_err = omega_err / (2 * np.pi)

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_damped_cos(x_fine, *popt)

    if plot:
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
            text=f"τ = {tau * 1e-3:.2f} ± {tau_err * 1e-3:.2f} μs, f = {f * 1e3:.3f} ± {f_err * 1e3:.3f} MHz",
            showarrow=False,
        )
        fig.update_layout(
            title=f"{title} : {target}",
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis_type=xaxis_type,
            yaxis_type=yaxis_type,
        )
        fig.show(config=_plotly_config(f"ramsey_{target}"))
    else:
        fig = None

    print(f"Target: {target}")
    print("Fit: A * exp(-t / tau) * cos(omega * t + phi) + C")
    print(f"  A = {A:.2f} ± {A_err:.2f}")
    print(f"  omega = {omega * 1e3:.2f} ± {omega_err * 1e3:.2f} rad/μs")
    print(f"  phi = {phi:.2f} ± {phi_err:.2f} rad")
    print(f"  tau = {tau * 1e-3:.2f} ± {tau_err * 1e-3:.2f} μs")
    print(f"  C = {C:.2f} ± {C_err:.2f}")
    print(f"f = {f * 1e3:.3f} ± {f_err * 1e3:.3f} MHz")
    print("")

    return {
        "tau": tau,
        "tau_err": tau_err,
        "f": f,
        "f_err": f_err,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_exp_decay(
    *,
    target: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
    plot: bool = True,
    title: str = "Decay time",
    xaxis_title: str = "Time (μs)",
    yaxis_title: str = "Signal (arb. unit)",
    xaxis_type: Literal["linear", "log"] = "log",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
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
    plot : bool, optional
        Whether to plot the data and the fit.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Label for the x-axis.
    yaxis_title : str, optional
        Label for the y-axis.
    xaxis_type : Literal["linear", "log"], optional
        Type of the x-axis.
    yaxis_type : Literal["linear", "log"], optional
        Type of the y-axis.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """
    if p0 is None:
        tau_guess = 20_000
        p0 = (
            np.abs(np.max(y) - np.min(y)),
            tau_guess,
            np.min(y),
        )

    if bounds is None:
        bounds = (
            (0, 0, -np.inf),
            (np.inf, np.inf, np.inf),
        )

    try:
        popt, pcov = curve_fit(func_exp_decay, x, y, p0=p0, bounds=bounds)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    A, tau, C = popt
    A_err, tau_err, C_err = np.sqrt(np.diag(pcov))

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_exp_decay(x_fine, *popt)

    if plot:
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
            text=f"τ = {tau * 1e-3:.2f} ± {tau_err * 1e-3:.2f} μs",
            showarrow=False,
        )
        fig.update_layout(
            title=f"{title} : {target}",
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis_type=xaxis_type,
            yaxis_type=yaxis_type,
        )
        fig.show(config=_plotly_config(f"exp_decay_{target}"))
    else:
        fig = None

    print(f"Target: {target}")
    print("Fit: A * exp(-t / tau) + C")
    print(f"  A = {A:.2f} ± {A_err:.2f}")
    print(f"  tau = {tau * 1e-3:.2f} ± {tau_err * 1e-3:.2f} μs")
    print(f"  C = {C:.2f} ± {C_err:.2f}")
    print("")

    return {
        "tau": tau,
        "tau_err": tau_err,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_rb(
    *,
    target: str,
    x: npt.NDArray[np.int64],
    y: npt.NDArray[np.float64],
    error_y: npt.NDArray[np.float64] | None = None,
    dimension: int = 2,
    p0=None,
    bounds=None,
    plot: bool = True,
    title: str = "Randomized benchmarking",
    xaxis_title: str = "Number of Cliffords",
    yaxis_title: str = "Normalized signal",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit randomized benchmarking data to an exponential decay function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    x : npt.NDArray[np.float64]
        Time points for the decay data.
    y : npt.NDArray[np.float64]
        Amplitude data for the decay.
    error_y : npt.NDArray[np.float64], optional
        Error data for the decay.
    dimension : int, optional
        Dimension of the Hilbert space.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Label for the x-axis.
    yaxis_title : str, optional
        Label for the y-axis.
    xaxis_type : Literal["linear", "log"], optional
        Type of the x-axis.
    yaxis_type : Literal["linear", "log"], optional
        Type of the y-axis.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """
    if p0 is None:
        p0 = (0.5, 1.0, 0.5)

    if bounds is None:
        bounds = (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        )

    def func_rb(n: npt.NDArray[np.float64], A: float, p: float, C: float):
        return A * p**n + C

    try:
        popt, pcov = curve_fit(func_rb, x, y, p0=p0, bounds=bounds)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    A, p, C = popt
    A_err, p_err, C_err = np.sqrt(np.diag(pcov))

    depolarizing_rate = 1 - p
    avg_gate_error = (dimension - 1) * (1 - p) / dimension
    avg_gate_fidelity = 1 - avg_gate_error
    avg_gate_fidelity_err = (dimension - 1) * p_err / dimension

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_rb(x_fine, *popt)

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
            error_y=dict(type="data", array=error_y),
            mode="markers",
            name="Data",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"F = {avg_gate_fidelity * 100:.3f} ± {avg_gate_fidelity_err * 100:.3f}%",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        fig.show(config=_plotly_config(f"rb_{target}"))

    print(f"Target: {target}")
    print("Fit: A * p^n + C")
    print(f"  A = {A:.6f} ± {A_err:.6f}")
    print(f"  p = {p:.6f} ± {p_err:.6f}")
    print(f"  C = {C:.6f} ± {C_err:.6f}")
    print(f"Depolarizing rate: {depolarizing_rate:.6f}")
    print(f"Average gate error: {avg_gate_error:.6f}")
    print(f"Average gate fidelity: {avg_gate_fidelity:.6f}")
    print("")

    return {
        "A": A,
        "A_err": A_err,
        "p": p,
        "p_err": p_err,
        "C": C,
        "C_err": C_err,
        "depolarizing_rate": depolarizing_rate,
        "avg_gate_error": avg_gate_error,
        "avg_gate_fidelity": avg_gate_fidelity,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def plot_irb(
    *,
    target: str,
    x: npt.NDArray[np.int64],
    y_rb: npt.NDArray[np.float64],
    y_irb: npt.NDArray[np.float64],
    error_y_rb: npt.NDArray[np.float64],
    error_y_irb: npt.NDArray[np.float64],
    A_rb: float,
    A_irb: float,
    p_rb: float,
    p_irb: float,
    C_rb: float,
    C_irb: float,
    gate_fidelity: float,
    title: str = "Interleaved randomized benchmarking",
    xaxis_title: str = "Number of Cliffords",
    yaxis_title: str = "Normalized signal",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> go.Figure:
    """
    Plot interleaved randomized benchmarking data.

    Parameters
    ----------
    target : str
        Identifier of the target.
    x : npt.NDArray[np.float64]
        Time points for the decay data.
    y_rb : npt.NDArray[np.float64]
        Amplitude data for the decay.
    y_irb : npt.NDArray[np.float64]
        Amplitude data for the interleaved decay.
    error_y_rb : npt.NDArray[np.float64]
        Error data for the decay.
    error_y_irb : npt.NDArray[np.float64]
        Error data for the interleaved decay.
    p_rb : float
        Depolarizing parameter of the randomized benchmarking.
    p_irb : float
        Depolarizing parameter of the interleaved randomized benchmarking.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Label for the x-axis.
    yaxis_title : str, optional
        Label for the y-axis.
    xaxis_type : Literal["linear", "log"], optional
        Type of the x-axis.
    yaxis_type : Literal["linear", "log"], optional
        Type of the y-axis.
    """

    def func_rb(n: npt.NDArray[np.float64], A: float, p: float, C: float):
        return A * p**n + C

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_rb_fine = func_rb(x_fine, A_rb, p_rb, C_rb)
    y_irb_fine = func_rb(x_fine, A_irb, p_irb, C_irb)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=y_rb_fine,
            mode="lines",
            name="Reference",
            line=dict(color=COLORS[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_rb,
            error_y=dict(type="data", array=error_y_rb),
            mode="markers",
            marker=dict(color=COLORS[0]),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=y_irb_fine,
            mode="lines",
            name="Interleaved",
            line=dict(color=COLORS[1]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_irb,
            error_y=dict(type="data", array=error_y_irb),
            mode="markers",
            marker=dict(color=COLORS[1]),
            showlegend=False,
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"F = {gate_fidelity * 100:.3f}%",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )
    fig.show(config=_plotly_config(f"irb_{target}"))
    return fig


def fit_ampl_calib_data(
    *,
    target: str,
    amplitude_range: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    p0=None,
    maximize: bool = True,
    plot: bool = True,
    title: str = "Amplitude calibration",
    xaxis_title: str = "Amplitude (arb. unit)",
    yaxis_title: str = "Signal (arb. unit)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit amplitude calibration data to a cosine function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    amplitude_range : npt.NDArray[np.float64]
        Amplitude range for the calibration data.
    data : npt.NDArray[np.float64]
        Measured values for the calibration data.
    p0 : optional
        Initial guess for the fitting parameters.
    maximize : bool, optional
        Whether to maximize or minimize the data.
    plot : bool, optional
        Whether to plot the data and the fit.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """

    if maximize:
        data = -data

    def cos_func(t, ampl, omega, phi, offset):
        return ampl * np.cos(omega * t + phi) + offset

    if p0 is None:
        p0 = (
            np.abs(np.max(data) - np.min(data)) / 2,
            2 * np.pi / (amplitude_range[-1] - amplitude_range[0]),
            np.pi,
            (np.max(data) + np.min(data)) / 2,
        )

    try:
        popt, pcov = curve_fit(cos_func, amplitude_range, data, p0=p0)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {
            "amplitude": np.nan,
        }

    result = minimize(
        cos_func,
        x0=np.mean(amplitude_range),
        args=tuple(popt),
        bounds=[(np.min(amplitude_range), np.max(amplitude_range))],
    )
    min_x = result.x[0]
    min_y = cos_func(min_x, *popt)

    x_fine = np.linspace(np.min(amplitude_range), np.max(amplitude_range), 1000)
    y_fine = cos_func(x_fine, *popt)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=-y_fine if maximize else y_fine,
            mode="lines",
            name="Fit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=amplitude_range,
            y=-data if maximize else data,
            mode="markers",
            name="Data",
        )
    )
    fig.add_annotation(
        x=min_x,
        y=-min_y if maximize else min_y,
        text=f"max: {min_x:.6g}" if maximize else f"min: {min_x:.6g}",
        showarrow=True,
        arrowhead=1,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )
    if plot:
        fig.show(config=_plotly_config(f"ampl_calib_{target}"))

    print(f"Calibrated amplitude: {min_x:.6g}")

    return {
        "amplitude": min_x,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_lorentzian(
    target: str,
    freq_range: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    p0=None,
    plot: bool = True,
    title: str = "Lorentzian fit",
    xaxis_title: str = "Frequency (GHz)",
    yaxis_title: str = "Signal (arb. unit)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit Lorentzian data to a Lorentzian function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    freq_range : npt.NDArray[np.float64]
        Frequency range for the Lorentzian data.
    data : npt.NDArray[np.float64]
        Amplitude data for the Lorentzian data.
    p0 : optional
        Initial guess for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """
    if p0 is None:
        p0 = (
            np.abs(np.max(data) - np.min(data)),
            np.mean(freq_range),
            (np.max(freq_range) - np.min(freq_range)) / 4,
            np.min(data),
        )

    try:
        popt, pcov = curve_fit(func_lorentzian, freq_range, data, p0=p0)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    A, f0, gamma, C = popt
    A_err, f0_err, gamma_err, C_err = np.sqrt(np.diag(pcov))

    x_fine = np.linspace(np.min(freq_range), np.max(freq_range), 1000)
    y_fine = func_lorentzian(x_fine, *popt)

    if plot:
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
                x=freq_range,
                y=data,
                mode="markers",
                name="Data",
            )
        )
        fig.add_annotation(
            x=f0,
            y=A,
            text=f"max: {f0:.6g}",
            showarrow=True,
            arrowhead=1,
        )
        fig.update_layout(
            title=f"{title} : {target}",
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis_type=xaxis_type,
            yaxis_type=yaxis_type,
        )
        fig.show(config=_plotly_config(f"lorentzian_{target}"))

    print("Fit : A / ( 1 + ( (f - f0) / gamma )^2 ) + C")
    print(f"  A = {A:.2f} ± {A_err:.2f}")
    print(f"  f0 = {f0:.6f} ± {f0_err:.6f} GHz")
    print(f"  gamma = {gamma * 1e3:.2f} ± {gamma_err * 1e3:.2f} MHz")
    print(f"  C = {C:.2f} ± {C_err:.2f}")

    return {
        "A": A,
        "f0": f0,
        "gamma": gamma,
        "C": C,
        "A_err": A_err,
        "f0_err": f0_err,
        "gamma_err": gamma_err,
        "C_err": C_err,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_sqrt_lorentzian(
    *,
    target: str,
    freq_range: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    p0=None,
    bounds=None,
    plot: bool = True,
    title: str = "Square root Lorentzian fit",
    xaxis_title: str = "Frequency (GHz)",
    yaxis_title: str = "Signal (arb. unit)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit square root Lorentzian data to a square root Lorentzian function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    freq_range : npt.NDArray[np.float64]
        Frequency range for the square root Lorentzian data.
    data : npt.NDArray[np.float64]
        Amplitude data for the square root Lorentzian data.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """
    if p0 is None:
        p0 = (
            np.min(data) - np.max(data),
            np.mean(freq_range),
            (np.max(freq_range) - np.min(freq_range)) / 4,
            np.max(data),
        )

    if bounds is None:
        bounds = (
            (-np.inf, np.min(freq_range), 0, -np.inf),
            (0, np.max(freq_range), np.inf, np.inf),
        )

    try:
        popt, pcov = curve_fit(
            func_sqrt_lorentzian,
            freq_range,
            data,
            p0=p0,
            bounds=bounds,
        )
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    A, f0, Omega, C = popt
    A_err, f0_err, Omega_err, C_err = np.sqrt(np.diag(pcov))

    x_fine = np.linspace(np.min(freq_range), np.max(freq_range), 1000)
    y_fine = func_sqrt_lorentzian(x_fine, *popt)

    if plot:
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
                x=freq_range,
                y=data,
                mode="markers",
                name="Data",
            )
        )
        fig.update_layout(
            title=f"{title} : {target}",
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis_type=xaxis_type,
            yaxis_type=yaxis_type,
        )
        fig.show(config=_plotly_config(f"sqrt_lorentzian_{target}"))

    print("Fit : A / sqrt{ 1 + ( (f - f0) / Omega )^2 } + C")
    print(f"  A = {A:.2f} ± {A_err:.2f}")
    print(f"  f0 = {f0:.6f} ± {f0_err:.6f} GHz")
    print(f"  Omega = {Omega * 1e3:.2f} ± {Omega_err * 1e3:.2f} MHz")
    print(f"  C = {C:.2f} ± {C_err:.2f}")

    return {
        "A": A,
        "f0": f0,
        "Omega": Omega,
        "C": C,
        "A_err": A_err,
        "f0_err": f0_err,
        "Omega_err": Omega_err,
        "C_err": C_err,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_reflection_coefficient(
    *,
    target: str,
    freq_range: npt.NDArray[np.float64],
    data: npt.NDArray[np.complex128],
    p0=None,
    bounds=None,
    plot: bool = True,
    title: str = "Reflection coefficient",
) -> dict:
    """
    Fit reflection coefficient data and obtain the resonance frequency and loss rates.

    Parameters
    ----------
    target : str
        Identifier of the target.
    freq_range : npt.NDArray[np.float64]
        Frequency range for the reflection coefficient data.
    data : npt.NDArray[np.complex128]
        Complex reflection coefficient data.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """

    if p0 is None:
        p0 = (
            (np.max(freq_range) + np.min(freq_range)) / 2,
            0.005,
            0.0,
            np.mean(np.abs(data)),
            0.0,
        )

    if bounds is None:
        bounds = (
            (np.min(freq_range), 0, 0, 0, -np.pi),
            (np.max(freq_range), 1.0, 1.0, 1.0, np.pi),
        )

    def residuals(params, f, y):
        f_r, kappa_ex, kappa_in, A, phi = params
        y_model = func_resonance(f, f_r, kappa_ex, kappa_in, A, phi)
        return np.hstack([np.real(y_model - y), np.imag(y_model - y)])

    result = least_squares(
        residuals,
        p0,
        bounds=bounds,
        args=(freq_range, data),
    )

    fitted_params = result.x

    f_r, kappa_ex, kappa_in, A, phi = fitted_params

    x_fine = np.linspace(np.min(freq_range), np.max(freq_range), 1000)
    y_fine = func_resonance(x_fine, *fitted_params)

    if plot:
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.5, 0.5],
            row_heights=[1.0, 1.0],
            specs=[
                [{"rowspan": 2}, {}],
                [None, {}],
            ],
            shared_xaxes=False,
            vertical_spacing=0.05,
            horizontal_spacing=0.125,
        )

        fig.add_trace(
            go.Scatter(
                x=np.real(data),
                y=np.imag(data),
                mode="markers",
                name="I/Q (Data)",
                marker=dict(color=COLORS[0]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.real(y_fine),
                y=np.imag(y_fine),
                mode="lines",
                name="I/Q (Fit)",
                marker=dict(color=COLORS[1]),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=freq_range,
                y=np.real(data),
                mode="markers",
                name="Re (Data)",
                marker=dict(color=COLORS[0]),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=x_fine,
                y=np.real(y_fine),
                mode="lines",
                name="Re (Fit)",
                marker=dict(color=COLORS[1]),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=freq_range,
                y=np.imag(data),
                mode="markers",
                name="Im (Data)",
                marker=dict(color=COLORS[0]),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=x_fine,
                y=np.imag(y_fine),
                mode="lines",
                name="Im (Fit)",
                marker=dict(color=COLORS[1]),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"{title} : {target}",
            width=800,
            height=450,
            showlegend=False,
        )

        fig.update_xaxes(
            title_text="Re",
            row=1,
            col=1,
            tickformat=".2g",
            showticklabels=True,
            zeroline=True,
            zerolinecolor="black",
            showgrid=True,
        )
        fig.update_yaxes(
            title_text="Im",
            row=1,
            col=1,
            scaleanchor="x",
            scaleratio=1,
            tickformat=".2g",
            showticklabels=True,
            zeroline=True,
            zerolinecolor="black",
            showgrid=True,
        )
        fig.update_xaxes(
            row=1,
            col=2,
            showticklabels=False,
            matches="x2",
        )
        fig.update_yaxes(
            title_text="Re",
            row=1,
            col=2,
        )
        fig.update_xaxes(
            title_text="Frequency (GHz)",
            row=2,
            col=2,
            matches="x2",
        )
        fig.update_yaxes(
            title_text="Im",
            row=2,
            col=2,
        )
        fig.show()

    print(f"{target}\n--------------------")
    print(f"Resonance frequency:\n  {f_r:.6f} GHz")
    print(f"External loss rate:\n  {kappa_ex * 1e3:.6f} MHz")
    print(f"Internal loss rate:\n  {kappa_in * 1e3:.6f} MHz")
    print("--------------------\n")

    return {
        "f_r": f_r,
        "kappa_ex": kappa_ex,
        "kappa_in": kappa_in,
        "A": A,
        "phi": phi,
        "fig": fig,
    }


def fit_rotation(
    times: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    r0: npt.NDArray[np.float64] = np.array([0, 0, 1]),
    p0=None,
    bounds=None,
    plot: bool = True,
    title: str = "State evolution",
    xlabel: str = "Time (ns)",
    ylabel: str = "Expectation value",
) -> dict:
    """
    Fit 3D rotation data to obtain the rotation coefficients and detuning frequency.

    Parameters
    ----------
    times : npt.NDArray[np.float64]
        Time points for the rotation data.
    data : npt.NDArray[np.float64]
        Expectation value data for the rotation.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.

    Returns
    -------
    dict
        Omega : tuple[float, float, float]
            Rotation coefficients.
        fig : go.Figure
            Plot of the data and the fit.
    """
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Data must be a 2D array with 3 columns.")

    G_x = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    G_y = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    def rotation_matrix(
        t: float,
        omega: float,
        n: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        G = n[0] * G_x + n[1] * G_y + n[2] * G_z
        return np.eye(3) + np.sin(omega * t) * G + (1 - np.cos(omega * t)) * G @ G

    def rotate(
        times: npt.NDArray[np.float64],
        omega: float,
        theta: float,
        phi: float,
        alpha: float,
    ) -> npt.NDArray[np.float64]:
        """
        Simulate the rotation of a state vector.

        Parameters
        ----------
        times : npt.NDArray[np.float64]
            Time points for the rotation.
        omega : float
            Rotation frequency.
        theta : float
            Polar angle of the rotation axis.
        phi : float
            Azimuthal angle of the rotation axis.
        alpha : float
            Decay rate.
        """
        n_x = np.sin(theta) * np.cos(phi)
        n_y = np.sin(theta) * np.sin(phi)
        n_z = np.cos(theta)
        r_t = np.array([rotation_matrix(t, omega, (n_x, n_y, n_z)) @ r0 for t in times])
        decay_factor = np.exp(-alpha * times)
        return decay_factor[:, np.newaxis] * r_t

    def residuals(params, times, data):
        return (rotate(times, *params) - data).flatten()

    if p0 is None:
        N = len(times)
        dt = times[1] - times[0]
        F = np.array(fft(data[:, 2]))
        f = np.array(fftfreq(N, dt)[1 : N // 2])
        i = np.argmax(np.abs(F[1 : N // 2]))
        dominant_freq = np.abs(f[i])
        omega_est = 2 * np.pi * dominant_freq
        theta_est = np.pi / 2
        phi_est = 0.0
        alpha_est = 0.0
        p0 = (omega_est, theta_est, phi_est, alpha_est)

    if bounds is None:
        bounds = (
            (0, 0, -np.pi, 0),
            (np.inf, np.pi, np.pi, 1e-3),
        )

    result = least_squares(
        residuals,
        p0,
        bounds=bounds,
        args=(times, data),
    )

    fitted_params = result.x
    Omega = fitted_params[0]
    theta = fitted_params[1]
    phi = fitted_params[2]
    alpha = fitted_params[3]
    tau = 1 / alpha * 1e-3  # μs
    Omega_x = Omega * np.sin(theta) * np.cos(phi)
    Omega_y = Omega * np.sin(theta) * np.sin(phi)
    Omega_z = Omega * np.cos(theta)
    # print(f"Omega: ({Omega_x:.6f}, {Omega_y:.6f}, {Omega_z:.6f})")

    times_fine = np.linspace(np.min(times), np.max(times), 1000)
    fit = rotate(times_fine, *fitted_params)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=data[:, 0],
            mode="markers",
            name="X (data)",
            marker=dict(color=COLORS[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times_fine,
            y=fit[:, 0],
            mode="lines",
            name="X (fit)",
            line=dict(color=COLORS[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=data[:, 1],
            mode="markers",
            name="Y (data)",
            marker=dict(color=COLORS[1]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times_fine,
            y=fit[:, 1],
            mode="lines",
            name="Y (fit)",
            line=dict(color=COLORS[1]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=data[:, 2],
            mode="markers",
            name="Z (data)",
            marker=dict(color=COLORS[2]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times_fine,
            y=fit[:, 2],
            mode="lines",
            name="Z (fit)",
            line=dict(color=COLORS[2]),
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"τ = {tau:.3f} μs",
        showarrow=False,
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis=dict(range=[-1.1, 1.1]),
    )

    fig3d = go.Figure()
    # data
    fig3d.add_trace(
        go.Scatter3d(
            name="data",
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode="markers",
            marker=dict(size=3),
            hoverinfo="skip",
        )
    )

    # fit
    fig3d.add_trace(
        go.Scatter3d(
            name="fit",
            x=fit[:, 0],
            y=fit[:, 1],
            z=fit[:, 2],
            mode="lines",
            line=dict(width=4),
            hoverinfo="skip",
        )
    )
    # sphere
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = 1
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    fig3d.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.05,
            showscale=False,
            colorscale="gray",
            hoverinfo="skip",
        )
    )
    # layout
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(title="〈X〉", visible=True),
            yaxis=dict(title="〈Y〉", visible=True),
            zaxis=dict(title="〈Z〉", visible=True),
            aspectmode="cube",
        ),
        width=400,
        height=400,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
    )

    if plot:
        fig.show()
        # fig3d.show()

    return {
        "Omega": np.array([Omega_x, Omega_y, Omega_z]),
        "fig": fig,
        "fig3d": fig3d,
    }


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


def estimate_wave_count(times, data) -> float:
    N = len(times)
    dt = times[1] - times[0]
    F = np.array(fft(data))
    f = np.array(fftfreq(N, dt)[1 : N // 2])
    i = np.argmax(np.abs(F[1 : N // 2]))
    dominant_freq = np.abs(f[i])
    wave_count_est = dominant_freq * (times[-1] - times[0])
    return wave_count_est
