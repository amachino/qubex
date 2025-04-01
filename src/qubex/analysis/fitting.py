from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
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

logger = logging.getLogger(__name__)


def _plotly_config(filename: str) -> dict:
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "toImageButtonOptions": {
            "format": "png",
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
    r2 : float
        Coefficient of determination.
    """

    target: str
    amplitude: float
    frequency: float
    phase: float
    offset: float
    noise: float
    angle: float
    r2: float


def normalize(
    values: NDArray,
    param: RabiParam,
) -> NDArray:
    """
    Normalizes the measured I/Q values.

    Parameters
    ----------
    values : NDArray
        Measured I/Q values.
    param : RabiParam
        Parameters of the Rabi oscillation.

    Returns
    -------
    NDArray
        Normalized I/Q values.
    """
    rotated = values * np.exp(-1j * param.angle)
    normalized = (np.imag(rotated) - param.offset) / param.amplitude
    return normalized


def func_cos(
    t: NDArray,
    A: complex,
    omega: float,
    phi: float,
    C: float,
) -> NDArray:
    """
    Calculate a cosine function with given parameters.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time points for the function evaluation.
    A : complex
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
    t: NDArray,
    A: complex,
    omega: float,
    phi: float,
    C: float,
    tau: float,
) -> NDArray:
    """
    Calculate a damped cosine function with given parameters.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time points for the function evaluation.
    A : complex
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


def func_delayed_cos(
    t: NDArray,
    t0: float,
    A: float,
    omega: float,
    C: float,
):
    """
    Calculate a delayed cosine function with given parameters.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time points for the function evaluation.
    t0 : float
        Delay time for the cosine function.
    A : float
        Amplitude of the cosine function.
    omega : float
        Angular frequency of the cosine function.
    C : float
        Vertical offset of the cosine function.
    """
    return np.where(
        t < t0,
        A + C,
        A * np.cos(omega * (t - t0)) + C,
    )


def func_exp_decay(
    t: NDArray,
    A: float,
    tau: float,
    C: float,
) -> NDArray:
    """
    Calculate an exponential decay function with given parameters.

    Parameters
    ----------
    t : NDArray[np.float64]
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
    f: NDArray,
    A: float,
    f0: float,
    gamma: float,
    C: float,
) -> NDArray:
    """
    Calculate a Lorentzian function with given parameters.

    Parameters
    ----------
    f : NDArray[np.float64]
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
    f: NDArray,
    A: float,
    f0: float,
    Omega: float,
    C: float,
) -> NDArray:
    """
    Calculate a square root Lorentzian function with given parameters.

    Parameters
    ----------
    f : NDArray[np.float64]
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


def func_resonance(
    f: NDArray,
    f_r: float,
    kappa_ex: float,
    kappa_in: float,
    A: float,
    phi: float,
) -> NDArray:
    """
    Calculate a resonance function with given parameters.

    Parameters
    ----------
    f : NDArray[np.float64]
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
    x: ArrayLike,
    y: ArrayLike,
    *,
    degree: int,
    plot: bool = True,
    target: str | None = None,
    title: str = "Polynomial fit",
    xlabel: str = "x",
    ylabel: str = "y",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit data to a polynomial function and plot the results.

    Parameters
    ----------
    x : ArrayLike
        x values for the data.
    y : ArrayLike
        y values for the data.
    degree : int
        Degree of the polynomial.
    plot : bool, optional
        Whether to plot the data and the fit.
    target : str, optional
        Identifier of the target.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
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
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

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
        title=f"{title} : {target}" if target else title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        filename = f"fit_polynomial_{target}" if target else "fit_polynomial"
        fig.show(config=_plotly_config(filename))

    return {
        "popt": popt,
        "fun": fun,
        "root": root,
        "fig": fig,
    }


def fit_cosine(
    x: ArrayLike,
    y: ArrayLike,
    *,
    tau_est: float = 10_000,
    is_damped: bool = False,
    target: str | None = None,
    title: str = "Cosine fit",
    xlabel: str = "x",
    ylabel: str = "y",
    plot: bool = True,
) -> dict:
    """
    Fit data to a cosine function and plot the results.

    Parameters
    ----------
    x : ArrayLike
        x values for the data.
    y : ArrayLike
        y values for the data.
    tau_est : float, optional
        Initial guess for the damping time constant.
    is_damped : bool, optional
        Whether to fit the data to a damped cosine function.
    target : str, optional
        Identifier of the target.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    plot : bool, optional
        Whether to plot the data and the fit.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    dt = x[1] - x[0]
    N = len(x)
    f = np.fft.fftfreq(N, dt)[1 : N // 2]
    F = np.fft.fft(y)[1 : N // 2]
    i = np.argmax(np.abs(F))

    # Estimate the initial parameters
    amplitude_est = 2 * np.abs(F[i]) / N
    omega_est = 2 * np.pi * f[i]
    phase_est = np.angle(F[i])
    offset_est = (np.max(y) + np.min(y)) / 2

    logger.debug(
        f"Initial guess: A = {amplitude_est:.3g}, ω = {omega_est:.3g}, φ = {phase_est:.3g}, C = {offset_est:.3g}"
    )

    if is_damped:
        p0 = (amplitude_est, omega_est, phase_est, offset_est, tau_est)
        bounds = (
            (0, 0, -np.pi, -np.inf, 0),
            (np.inf, np.inf, np.pi, np.inf, np.inf),
        )
        popt, pcov = curve_fit(func_damped_cos, x, y, p0=p0, bounds=bounds)
    else:
        p0 = (amplitude_est, omega_est, phase_est, offset_est)
        bounds = (
            (0, 0, -np.pi, -np.inf),
            (np.inf, np.inf, np.pi, np.inf),
        )
        popt, pcov = curve_fit(func_cos, x, y, p0=p0, bounds=bounds)

    if is_damped:
        A, omega, phi, C, tau = popt
        A_err, omega_err, phi_err, C_err, tau_err = np.sqrt(np.diag(pcov))
    else:
        A, omega, phi, C = popt
        A_err, omega_err, phi_err, C_err = np.sqrt(np.diag(pcov))[:4]
        tau, tau_err = None, None

    f = omega / (2 * np.pi)
    f_err = omega_err / (2 * np.pi)

    if is_damped:
        r2 = 1 - np.sum((y - func_damped_cos(x, *popt)) ** 2) / np.sum(
            (y - np.mean(y)) ** 2
        )
    else:
        r2 = 1 - np.sum((y - func_cos(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)

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
        ),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}" if target else title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )

    if plot:
        filename = f"fit_cosine_{target}" if target else "fit_cosine"
        fig.show(config=_plotly_config(filename))

        if target:
            print(f"Target: {target}")
        print("Fit: A * cos(2πft + φ) + C")
        print(f"  A = {A:.3g} ± {A_err:.1g}")
        print(f"  f = {f:.3g} ± {f_err:.1g}")
        print(f"  φ = {phi:.3g} ± {phi_err:.1g}")
        print(f"  C = {C:.3g} ± {C_err:.1g}")
        if is_damped:
            print(f"  τ = {tau:.3g} ± {tau_err:.1g}")
        print(f"  R² = {r2:.3f}")
        print("")

    return {
        "A": A,
        "f": f,
        "phi": phi,
        "C": C,
        "tau": tau,
        "A_err": A_err,
        "f_err": f_err,
        "phi_err": phi_err,
        "C_err": C_err,
        "tau_err": tau_err,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_delayed_cosine(
    x: ArrayLike,
    y: ArrayLike,
    *,
    threshold: float = 0.5,
    p0=None,
    bounds=None,
    plot: bool = True,
    target: str | None = None,
    title: str = "Delayed cosine fit",
    xlabel: str = "x",
    ylabel: str = "y",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit data to a delayed cosine function and plot the results.

    Parameters
    ----------
    x : ArrayLike
        x values for the data.
    y : ArrayLike
        y values for the data.
    threshold : float, optional
        Threshold for the initial guess of t0.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.
    target : str, optional
        Identifier of the target.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
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
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    dt = x[1] - x[0]
    N = len(x)
    f = np.fft.fftfreq(N, dt)[1 : N // 2]
    F = np.fft.fft(y)[1 : N // 2]
    i = np.argmax(np.abs(F))
    amplitude_est = 2 * np.abs(F[i]) / N
    omega_est = 2 * np.pi * f[i]
    offset_est = (np.max(y) + np.min(y)) / 2

    dy = np.abs((y - y[0]))
    dy = dy / np.max(dy)
    idx = np.argmax(dy > threshold)
    t0_est = x[idx]

    logger.error(
        f"Initial guess: A = {amplitude_est:.3g}, ω = {omega_est:.3g}, t0 = {t0_est:.3g}, C = {offset_est:.3g}"
    )

    if p0 is None:
        p0 = (t0_est, amplitude_est, omega_est, offset_est)

    if bounds is None:
        bounds = (
            (0, 0, 0, -np.inf),
            (np.inf, np.inf, np.inf, np.inf),
        )

    try:
        popt, pcov = curve_fit(func_delayed_cos, x, y, p0=p0, bounds=bounds)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    t0, A, omega, C = popt
    t0_err, A_err, omega_err, C_err = np.sqrt(np.diag(pcov))
    f = omega / (2 * np.pi)
    f_err = omega_err / (2 * np.pi)

    r2 = 1 - np.sum((y - func_delayed_cos(x, *popt)) ** 2) / np.sum(
        (y - np.mean(y)) ** 2
    )

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_delayed_cos(x_fine, *popt)

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
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.add_annotation(
        x=t0,
        y=func_delayed_cos(t0, *popt),
        text=f"t0: {t0:.3g}",
        showarrow=True,
        arrowhead=1,
    )
    fig.update_layout(
        title=f"{title} : {target}" if target else title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        filename = f"fit_delayed_cos_{target}" if target else "fit_delayed_cos"
        fig.show(config=_plotly_config(filename))

        if target:
            print(f"Target: {target}")
        print("Fit: A * cos(2πft + φ) + C")
        print(f"  A = {A:.3g} ± {A_err:.1g}")
        print(f"  f = {f:.3g} ± {f_err:.1g}")
        print(f"  t0 = {t0:.3g} ± {t0_err:.1g}")
        print(f"  C = {C:.3g} ± {C_err:.1g}")
        print(f"  R² = {r2:.3f}")
        print("")

    return {
        "t0": t0,
        "f": f,
        "C": C,
        "A": A,
        "t0_err": t0_err,
        "f_err": f_err,
        "C_err": C_err,
        "A_err": A_err,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_exp_decay(
    x: ArrayLike,
    y: ArrayLike,
    *,
    p0=None,
    bounds=None,
    plot: bool = True,
    target: str | None = None,
    title: str = "Decay time",
    xlabel: str = "Time (μs)",
    ylabel: str = "Signal (arb. units)",
    xaxis_type: Literal["linear", "log"] = "log",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit decay data to an exponential decay function and plot the results.

    Parameters
    ----------
    x : ArrayLike
        Time points for the decay data.
    y : ArrayLike
        Amplitude data for the decay.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.
    target : str, optional
        Identifier of the target.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
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
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

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
        return {
            "status": "error",
            "message": "Failed to fit the data.",
        }

    A, tau, C = popt
    A_err, tau_err, C_err = np.sqrt(np.diag(pcov))

    r2 = 1 - np.sum((y - func_exp_decay(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)

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
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"τ = {tau * 1e-3:.1f} ± {tau_err * 1e-3:.1f} μs, R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}" if target else title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        filename = f"fit_exp_decay_{target}" if target else "fit_exp_decay"
        fig.show(config=_plotly_config(filename))

        if target:
            print(f"Target: {target}")
        print("Fit: A * exp(-t/τ) + C")
        print(f"  A = {A:.3g} ± {A_err:.1g}")
        print(f"  τ = {tau * 1e-3:.3g} ± {tau_err * 1e-3:.1g}")
        print(f"  C = {C:.3g} ± {C_err:.1g}")
        print(f"  R² = {r2:.3f}")
        print("")

    result = {
        "A": A,
        "tau": tau,
        "C": C,
        "A_err": A_err,
        "tau_err": tau_err,
        "C_err": C_err,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }

    return {
        "status": "success",
        "message": "Fitting successful.",
        **result,
    }


def fit_lorentzian(
    x: ArrayLike,
    y: ArrayLike,
    *,
    p0=None,
    plot: bool = True,
    target: str | None = None,
    title: str = "Lorentzian fit",
    xlabel: str = "Frequency (GHz)",
    ylabel: str = "Signal (arb. units)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit Lorentzian data to a Lorentzian function and plot the results.

    Parameters
    ----------
    x : ArrayLike
        Frequency range for the Lorentzian data.
    y : ArrayLike
        Amplitude data for the Lorentzian data.
    p0 : optional
        Initial guess for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.
    target : str, optional
        Identifier of the target.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
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
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    if p0 is None:
        p0 = (
            np.abs(np.max(y) - np.min(y)),
            np.mean(x),
            (np.max(x) - np.min(x)) / 4,
            np.min(y),
        )

    try:
        popt, pcov = curve_fit(func_lorentzian, x, y, p0=p0)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    A, f0, gamma, C = popt
    A_err, f0_err, gamma_err, C_err = np.sqrt(np.diag(pcov))

    r2 = 1 - np.sum((y - func_lorentzian(x, *popt)) ** 2) / np.sum(
        (y - np.mean(y)) ** 2
    )

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_lorentzian(x_fine, *popt)

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
        )
    )
    fig.add_annotation(
        x=f0,
        y=func_lorentzian(f0, *popt),
        text=f"max: {f0:.6g}",
        showarrow=True,
        arrowhead=1,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}" if target else title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        filename = f"fit_lorentzian_{target}" if target else "fit_lorentzian"
        fig.show(config=_plotly_config(filename))

        if target:
            print(f"Target: {target}")
        print("Fit : A / [1 + {(f - f0) / γ}^2] + C")
        print(f"  A = {A:.3g} ± {A_err:.1g}")
        print(f"  f0 = {f0:.3g} ± {f0_err:.1g}")
        print(f"  γ = {gamma:.3g} ± {gamma_err:.1g}")
        print(f"  C = {C:.3g} ± {C_err:.2f}")

    return {
        "A": A,
        "f0": f0,
        "gamma": gamma,
        "C": C,
        "A_err": A_err,
        "f0_err": f0_err,
        "gamma_err": gamma_err,
        "C_err": C_err,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_sqrt_lorentzian(
    x: ArrayLike,
    y: ArrayLike,
    *,
    p0=None,
    bounds=None,
    plot: bool = True,
    target: str | None = None,
    title: str = "Square root Lorentzian fit",
    xlabel: str = "Frequency (GHz)",
    ylabel: str = "Signal (arb. units)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit square root Lorentzian data to a square root Lorentzian function and plot the results.

    Parameters
    ----------
    x : ArrayLike
        Frequency range for the square root Lorentzian data.
    y : ArrayLike
        Amplitude data for the square root Lorentzian data.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.
    target : str, optional
        Identifier of the target.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
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
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    if p0 is None:
        p0 = (
            np.min(y) - np.max(y),
            np.mean(x),
            (np.max(x) - np.min(x)) / 4,
            np.max(y),
        )

    if bounds is None:
        bounds = (
            (-np.inf, np.min(x), 0, -np.inf),
            (0, np.max(x), np.inf, np.inf),
        )

    try:
        popt, pcov = curve_fit(
            func_sqrt_lorentzian,
            x,
            y,
            p0=p0,
            bounds=bounds,
        )
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {}

    A, f0, Omega, C = popt
    A_err, f0_err, Omega_err, C_err = np.sqrt(np.diag(pcov))

    r2 = 1 - np.sum((y - func_sqrt_lorentzian(x, *popt)) ** 2) / np.sum(
        (y - np.mean(y)) ** 2
    )

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = func_sqrt_lorentzian(x_fine, *popt)

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
        )
    )
    fig.add_annotation(
        x=f0,
        y=func_sqrt_lorentzian(f0, *popt),
        text=f"max: {f0:.6g}",
        showarrow=True,
        arrowhead=1,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}" if target else title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        filename = f"fit_sqrt_lorentzian_{target}" if target else "fit_sqrt_lorentzian"
        fig.show(config=_plotly_config(filename))

        if target:
            print(f"Target: {target}")
        print("Fit : A / √[1 + {(f - f0) / Ω}^2] + C")
        print(f"  A = {A:.3g} ± {A_err:.1g}")
        print(f"  f0 = {f0:.3g} ± {f0_err:.1g}")
        print(f"  Ω = {Omega:.3g} ± {Omega_err:.1g}")
        print(f"  C = {C:.3g} ± {C_err:.1g}")

    return {
        "A": A,
        "f0": f0,
        "Omega": Omega,
        "C": C,
        "A_err": A_err,
        "f0_err": f0_err,
        "Omega_err": Omega_err,
        "C_err": C_err,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_rabi(
    *,
    target: str,
    times: NDArray,
    data: NDArray,
    tau_est: float = 10_000,
    plot: bool = True,
    is_damped: bool = False,
    ylabel: str | None = None,
    yaxis_range: tuple[float, float] | None = None,
) -> dict:
    """
    Fit Rabi oscillation data to a cosine function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    times : NDArray[np.float64]
        Array of time points for the Rabi oscillations.
    data : NDArray[np.complex64]
        Complex signal data corresponding to the Rabi oscillations.
    tau_est : float, optional
        Initial guess for the damping time constant.
    plot : bool, optional
        Whether to plot the data and the fit.
    is_damped : bool, optional
        Whether to fit the data to a damped cosine function.
    ylabel : str, optional
        Label for the y-axis.
    yaxis_range : tuple[float, float], optional
        Range for the y-axis.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """
    times = np.asarray(times, dtype=np.float64)
    data = np.asarray(data, dtype=np.complex64)

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

    dt = times[1] - times[0]
    N = len(times)
    f = np.fft.fftfreq(N, dt)[1 : N // 2]
    F = np.fft.fft(y)[1 : N // 2]
    i = np.argmax(np.abs(F))

    # Estimate the initial parameters
    amplitude_est = 2 * np.abs(F[i]) / N
    omega_est = 2 * np.pi * f[i]
    phase_est = np.angle(F[i])
    offset_est = (np.max(y) + np.min(y)) / 2

    try:
        p0: tuple
        bounds: tuple
        if is_damped:
            p0 = (amplitude_est, omega_est, phase_est, offset_est, tau_est)
            bounds = (
                (0, 0, -np.inf, -np.inf, 0),
                (np.inf, np.inf, np.pi, np.inf, np.inf),
            )
            popt, pcov = curve_fit(func_damped_cos, x, y, p0=p0, bounds=bounds)
        else:
            p0 = (amplitude_est, omega_est, phase_est, offset_est)
            bounds = (
                (0, 0, -np.inf, -np.inf),
                (np.inf, np.inf, np.pi, np.inf),
            )
            popt, pcov = curve_fit(func_cos, x, y, p0=p0, bounds=bounds)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {
            "status": "error",
            "message": "Failed to fit the data.",
            "rabi_param": RabiParam(
                target=target,
                amplitude=np.nan,
                frequency=np.nan,
                phase=np.nan,
                offset=np.nan,
                noise=noise,
                angle=angle,
                r2=np.nan,
            ),
        }

    if is_damped:
        amplitude, omega, phase, offset, tau = popt
        amplitude_err, omega_err, phase_err, offset_err, tau_err = np.sqrt(
            np.diag(pcov)
        )
    else:
        amplitude, omega, phase, offset = popt
        amplitude_err, omega_err, phase_err, offset_err = np.sqrt(np.diag(pcov))[:4]
        tau, tau_err = None, None

    frequency = omega / (2 * np.pi)
    frequency_err = omega_err / (2 * np.pi)

    if is_damped:
        r2 = 1 - np.sum((y - func_damped_cos(x, *popt)) ** 2) / np.sum(
            (y - np.mean(y)) ** 2
        )
    else:
        r2 = 1 - np.sum((y - func_cos(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)

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
        text=f"R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.update_layout(
        title=(f"Rabi oscillation of {target} : {frequency * 1e3:.2f} MHz"),
        xaxis_title="Drive duration (ns)",
        yaxis_title=ylabel or "Signal (arb. units)",
        yaxis_range=yaxis_range,
    )

    if plot:
        fig.show(config=_plotly_config(f"rabi_{target}"))

        print(f"Target: {target}")
        print(f"Rabi frequency: {frequency * 1e3:.3g} ± {frequency_err * 1e3:.1g} MHz")

    result = {
        "rabi_param": RabiParam(
            target=target,
            amplitude=amplitude,
            frequency=frequency,
            phase=phase,
            offset=offset,
            noise=noise,
            angle=angle,
            r2=r2,
        ),
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
        "offset": offset,
        "tau": tau,
        "amplitude_err": amplitude_err,
        "frequency_err": frequency_err,
        "phase_err": phase_err,
        "offset_err": offset_err,
        "tau_err": tau_err,
        "angle": angle,
        "noise": noise,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }

    if r2 < 0.5:
        print("Error: R² < 0.5")
        return {
            "status": "error",
            "message": "R² < 0.5",
            **result,
        }
    if r2 < 0.9:
        print("Warning: R² < 0.9")
        return {
            "status": "warning",
            "message": "R² < 0.9",
            **result,
        }
    else:
        return {
            "status": "success",
            "message": "Fitting successful",
            **result,
        }


def fit_detuned_rabi(
    *,
    target: str,
    control_frequencies: NDArray,
    rabi_frequencies: NDArray,
    plot: bool = True,
) -> dict:
    """
    Fit detuned Rabi oscillation data to a cosine function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    control_frequencies : NDArray[np.float64]
        Array of control frequencies for the Rabi oscillations in GHz.
    rabi_frequencies : NDArray[np.float64]
        Rabi frequencies corresponding to the control frequencies in GHz.
    plot : bool, optional
        Whether to plot the data and the fit.

    Returns
    -------
    dict
        Fitted parameters and the figure.
    """
    control_frequencies = np.asarray(control_frequencies, dtype=np.float64)
    rabi_frequencies = np.asarray(rabi_frequencies, dtype=np.float64)

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

    r2 = 1 - np.sum((y - func(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)

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
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.update_layout(
        title=f"Detuned Rabi oscillation : {target}",
        xaxis_title="Drive frequency (GHz)",
        yaxis_title="Rabi frequency (MHz)",
    )

    if plot:
        fig.show(config=_plotly_config(f"detuned_rabi_{target}"))

        print("Resonance frequency")
        print(f"  {target}: {f_resonance:.6f}")

    return {
        "f_resonance": f_resonance,
        "f_resonance_err": f_resonance_err,
        "f_rabi": f_rabi,
        "f_rabi_err": f_rabi_err,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_ramsey(
    *,
    target: str,
    times: NDArray,
    data: NDArray,
    tau_est: float = 10_000,
    p0=None,
    bounds=None,
    plot: bool = True,
    title: str = "Ramsey fringe",
    xlabel: str = "Time (μs)",
    ylabel: str = "Signal (arb. units)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit Ramsey fringe using a damped cosine function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    times : NDArray[np.float64]
        Array of time points for the Ramsey fringe.
    data : NDArray[np.float64]
        Amplitude data for the Ramsey fringe.
    tau_est : float, optional
        Initial guess for the damping time constant.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
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
    times = np.asarray(times, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    dt = times[1] - times[0]
    N = len(times)
    f = np.fft.fftfreq(N, dt)[1 : N // 2]
    F = np.fft.fft(data)[1 : N // 2]
    i = np.argmax(np.abs(F))

    amplitude_est = 2 * np.abs(F[i]) / N
    omega_est = 2 * np.pi * f[i]
    phase_est = np.angle(F[i])
    offset_est = (np.max(data) + np.min(data)) / 2

    if p0 is None:
        p0 = (amplitude_est, omega_est, phase_est, offset_est, tau_est)

    if bounds is None:
        bounds = (
            (0, 0, -np.pi, -np.inf, 0),
            (np.inf, np.inf, np.pi, np.inf, np.inf),
        )

    try:
        popt, pcov = curve_fit(func_damped_cos, times, data, p0=p0, bounds=bounds)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {
            "status": "error",
            "message": "Failed to fit the data.",
        }

    A, omega, phi, C, tau = popt
    A_err, omega_err, phi_err, C_err, tau_err = np.sqrt(np.diag(pcov))
    f = omega / (2 * np.pi)
    f_err = omega_err / (2 * np.pi)

    r2 = 1 - np.sum((data - func_damped_cos(times, *popt)) ** 2) / np.sum(
        (data - np.mean(data)) ** 2
    )

    x_fine = np.linspace(np.min(times), np.max(times), 1000)
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
            x=times * 1e-3,
            y=data,
            mode="markers",
            name="Data",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        fig.show(config=_plotly_config(f"ramsey_{target}"))

        print(f"Target: {target}")
        print("Fit: A * exp(-t/τ) * cos(2πft + φ) + C")
        print(f"  A = {A:.3g} ± {A_err:.1g}")
        print(f"  f = {f:.3g} ± {f_err:.1g}")
        print(f"  φ = {phi:.3g} ± {phi_err:.1g}")
        print(f"  τ = {tau:.3g} ± {tau_err:.1g}")
        print(f"  C = {C:.3g} ± {C_err:.1g}")
        print(f"  R² = {r2:.3f}")
        print("")

    result = {
        "A": A,
        "omega": omega,
        "phi": phi,
        "C": C,
        "tau": tau,
        "A_err": A_err,
        "omega_err": omega_err,
        "phi_err": phi_err,
        "C_err": C_err,
        "tau_err": tau_err,
        "f": f,
        "f_err": f_err,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }

    return {
        "status": "success",
        "message": "Fitting successful.",
        **result,
    }


def fit_rb(
    *,
    target: str,
    x: NDArray[np.int64],
    y: NDArray[np.float64],
    error_y: NDArray[np.float64] | None = None,
    dimension: int = 2,
    p0=None,
    bounds=None,
    plot: bool = True,
    title: str = "Randomized benchmarking",
    xlabel: str = "Number of Cliffords",
    ylabel: str = "Normalized signal",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit randomized benchmarking data to an exponential decay function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    x : NDArray[np.float64]
        Number of Cliffords for the randomized benchmarking.
    y : NDArray[np.float64]
        Amplitude data for the randomized benchmarking.
    error_y : NDArray[np.float64], optional
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
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
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

    def func_rb(n: NDArray, A: float, p: float, C: float):
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

    r2 = 1 - np.sum((y - func_rb(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)

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
        text=f"F = {avg_gate_fidelity * 100:.2f} ± {avg_gate_fidelity_err * 100:.2f}%",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    if plot:
        fig.show(config=_plotly_config(f"rb_{target}"))

        print(f"Target: {target}")
        print("Fit: A * p^n + C")
        print(f"  A = {A:.3g} ± {A_err:.1g}")
        print(f"  p = {p:.3g} ± {p_err:.1g}")
        print(f"  C = {C:.3g} ± {C_err:.1g}")
        print(f"  R² = {r2:.3f}")
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
        "avg_gate_fidelity_err": avg_gate_fidelity_err,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def plot_irb(
    *,
    target: str,
    x: NDArray[np.int64],
    y_rb: NDArray[np.float64],
    y_irb: NDArray[np.float64],
    error_y_rb: NDArray[np.float64],
    error_y_irb: NDArray[np.float64],
    A_rb: float,
    A_irb: float,
    p_rb: float,
    p_irb: float,
    C_rb: float,
    C_irb: float,
    gate_fidelity: float,
    gate_fidelity_err: float,
    plot: bool = True,
    title: str = "Interleaved randomized benchmarking",
    xlabel: str = "Number of Cliffords",
    ylabel: str = "Normalized signal",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> go.Figure:
    """
    Plot interleaved randomized benchmarking data.

    Parameters
    ----------
    target : str
        Identifier of the target.
    x : NDArray[np.float64]
        Time points for the decay data.
    y_rb : NDArray[np.float64]
        Amplitude data for the decay.
    y_irb : NDArray[np.float64]
        Amplitude data for the interleaved decay.
    error_y_rb : NDArray[np.float64]
        Error data for the decay.
    error_y_irb : NDArray[np.float64]
        Error data for the interleaved decay.
    A_rb : float
        Amplitude of the randomized benchmarking.
    A_irb : float
        Amplitude of the interleaved randomized benchmarking.
    p_rb : float
        Depolarizing parameter of the randomized benchmarking.
    p_irb : float
        Depolarizing parameter of the interleaved randomized benchmarking.
    C_rb : float
        Offset of the randomized benchmarking.
    C_irb : float
        Offset of the interleaved randomized benchmarking.
    gate_fidelity : float
        Gate fidelity of the randomized benchmarking.
    gate_fidelity_err : float
        Error in the gate fidelity of the randomized benchmarking.
    plot : bool, optional
        Whether to plot the data and the fit.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    xaxis_type : Literal["linear", "log"], optional
        Type of the x-axis.
    yaxis_type : Literal["linear", "log"], optional
        Type of the y-axis.
    """

    def func_rb(n: NDArray[np.float64], A: float, p: float, C: float):
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
        text=f"F = {gate_fidelity * 100:.2f} ± {gate_fidelity_err * 100:.2f}%",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )
    if plot:
        fig.show(config=_plotly_config(f"irb_{target}"))
    return fig


def fit_ampl_calib_data(
    *,
    target: str,
    amplitude_range: NDArray,
    data: NDArray,
    p0=None,
    maximize: bool = True,
    plot: bool = True,
    title: str = "Amplitude calibration",
    xlabel: str = "Amplitude (arb. units)",
    ylabel: str = "Signal (arb. units)",
    xaxis_type: Literal["linear", "log"] = "linear",
    yaxis_type: Literal["linear", "log"] = "linear",
) -> dict:
    """
    Fit amplitude calibration data to a cosine function and plot the results.

    Parameters
    ----------
    target : str
        Identifier of the target.
    amplitude_range : NDArray[np.float64]
        Amplitude range for the calibration data.
    data : NDArray[np.float64]
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
    amplitude_range = np.asarray(amplitude_range, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    if maximize:
        data = -data

    def cos_func(t, ampl, omega, phi, offset):
        return ampl * np.cos(omega * t + phi) + offset

    x = amplitude_range
    y = data
    dt = x[1] - x[0]
    N = len(x)
    f = np.fft.fftfreq(N, dt)[1 : N // 2]
    F = np.fft.fft(y)[1 : N // 2]
    i = np.argmax(np.abs(F))

    amplitude_est = 2 * np.abs(F[i]) / N
    omega_est = 2 * np.pi * f[i]
    phase_est = np.angle(F[i])
    offset_est = (np.max(y) + np.min(y)) / 2

    if p0 is None:
        p0 = (amplitude_est, omega_est, phase_est, offset_est)

    try:
        popt, pcov = curve_fit(cos_func, x, y, p0=p0)
    except RuntimeError:
        print(f"Failed to fit the data for {target}.")
        return {
            "amplitude": np.nan,
            "r2": np.nan,
        }

    result = minimize(
        cos_func,
        x0=np.mean(x),
        args=tuple(popt),
        bounds=[(np.min(x), np.max(x))],
    )
    min_x = result.x[0]
    min_y = cos_func(min_x, *popt)

    r2 = 1 - np.sum((y - cos_func(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)

    x_fine = np.linspace(np.min(x), np.max(x), 1000)
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
            x=x,
            y=-y if maximize else y,
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
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"R² = {r2:.3f}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )
    if plot:
        fig.show(config=_plotly_config(f"ampl_calib_{target}"))

    return {
        "amplitude": min_x,
        "r2": r2,
        "popt": popt,
        "pcov": pcov,
        "fig": fig,
    }


def fit_reflection_coefficient(
    *,
    target: str,
    freq_range: NDArray,
    data: NDArray,
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
    freq_range : NDArray[np.float64]
        Frequency range for the reflection coefficient data.
    data : NDArray[np.complex64]
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
    freq_range = np.asarray(freq_range, dtype=np.float64)
    data = np.asarray(data, dtype=np.complex64)

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

    r2 = 1 - np.sum(residuals(fitted_params, freq_range, data) ** 2) / np.sum(
        np.abs(data - np.mean(data)) ** 2
    )

    x_fine = np.linspace(np.min(freq_range), np.max(freq_range), 1000)
    y_fine = func_resonance(x_fine, *fitted_params)

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

    if plot:
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
        "r2": r2,
        "fig": fig,
    }


def fit_rotation(
    times: NDArray[np.float64],
    data: NDArray[np.float64],
    r0: NDArray[np.float64] = np.array([0, 0, 1]),
    p0=None,
    bounds=None,
    plot: bool = True,
    plot3d: bool = False,
    title: str = "State evolution",
    xlabel: str = "Time (ns)",
    ylabel: str = "Expectation value",
) -> dict:
    """
    Fit 3D rotation data to obtain the rotation coefficients and detuning frequency.

    Parameters
    ----------
    times : NDArray[np.float64]
        Time points for the rotation data.
    data : NDArray[np.float64]
        Expectation value data for the rotation.
    p0 : optional
        Initial guess for the fitting parameters.
    bounds : optional
        Bounds for the fitting parameters.
    plot : bool, optional
        Whether to plot the data and the fit.
    plot3d : bool, optional
        Whether to plot the data and the fit in 3D.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

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
    ) -> NDArray[np.float64]:
        G = n[0] * G_x + n[1] * G_y + n[2] * G_z
        return np.eye(3) + np.sin(omega * t) * G + (1 - np.cos(omega * t)) * G @ G

    def rotate(
        times: NDArray[np.float64],
        omega: float,
        theta: float,
        phi: float,
        alpha: float,
    ) -> NDArray[np.float64]:
        """
        Simulate the rotation of a state vector.

        Parameters
        ----------
        times : NDArray[np.float64]
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
        freq = np.fft.fftfreq(N, dt)[1 : N // 2]
        X = np.fft.fft(data[:, 0])[1 : N // 2]
        Y = np.fft.fft(data[:, 1])[1 : N // 2]
        Z = np.fft.fft(data[:, 2])[1 : N // 2]
        idx = np.argmax(np.abs(X) ** 2 + np.abs(Y) ** 2 + np.abs(Z) ** 2)
        dominant_freq = np.abs(freq[idx])
        F = np.array([X[idx], Y[idx], Z[idx]])
        n = np.cross(np.imag(F), np.real(F))
        n /= np.linalg.norm(n)
        omega_est = 2 * np.pi * dominant_freq
        theta_est = np.arccos(n[2])
        phi_est = np.arctan2(n[1], n[0])
        alpha_est = 0.0
        p0 = (omega_est, theta_est, phi_est, alpha_est)

    if bounds is None:
        bounds = (
            (0, 0, -np.pi, 0),
            (np.inf, np.pi, np.pi, 1e-3),
        )

    logger.info("Fitting rotation data.")
    logger.info(f"Initial guess: {p0}")

    result = least_squares(
        residuals,
        p0,
        bounds=bounds,
        args=(times, data),
    )

    fitted_params = result.x
    F = fitted_params[0]
    theta = fitted_params[1]
    phi = fitted_params[2]
    alpha = fitted_params[3]
    tau = 1 / alpha * 1e-3  # μs
    Omega_x = F * np.sin(theta) * np.cos(phi)
    Omega_y = F * np.sin(theta) * np.sin(phi)
    Omega_z = F * np.cos(theta)

    r2 = 1 - np.sum(residuals(fitted_params, times, data) ** 2) / np.sum(
        np.abs(data - np.mean(data))
    )

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
        text=f"τ = {tau:.3f} μs, R² = {r2:.3f}",
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

    if plot3d:
        fig3d.show()

    return {
        "Omega": np.array([Omega_x, Omega_y, Omega_z]),
        "r2": r2,
        "fig": fig,
        "fig3d": fig3d,
    }


def rotate(
    data: ArrayLike,
    angle: float,
) -> NDArray[np.complex128]:
    """
    Rotate complex data points by a specified angle.

    Parameters
    ----------
    data : ArrayLike
        Array of complex data points to be rotated.
    angle : float
        Angle in radians by which to rotate the data points.

    Returns
    -------
    NDArray[np.complex128]
        Rotated complex data points.
    """
    points = np.array(data)
    rotated_points = points * np.exp(1j * angle)
    return rotated_points
