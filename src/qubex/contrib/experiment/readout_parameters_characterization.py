"""Contributed helpers for readout parameters characterization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import qxvisualizer as viz
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from qubex.analysis import FitResult, FitStatus
from qubex.experiment.experiment import Experiment
from qubex.experiment.models.result import Result


def characterize_readout_parameters(
    exp: Experiment,
    *,
    target: str | None = None,
    frequency_range: NDArray,
    readout_amplitude: float | None = None,
    n_shots: int = 1024,
    save_image: bool = True,
) -> Result:
    """
    Characterize readout parameters by scanning readout frequency.

    Parameters
    ----------
    exp : Experiment
        Qubex Experiment instance
    target : str, optional
        Qubit label
    frequency_range : NDArray
        Range of readout frequencies to scan [GHz]
    readout_amplitude : float, optional
        Amplitude of readout pulse. The unit is a value for Hardware.
    n_shots : int, optional
        Number of shots.
    save_image : bool, optional
        Whether to save the scan result as an image file
    """
    if target is None:
        target = exp.qubit_labels[0]

    if readout_amplitude is None:
        readout_amplitude = 0.01

    result = exp.scan_resonator_frequencies(
        target,
        frequency_range=frequency_range,
        readout_amplitude=readout_amplitude,
        save_image=save_image,
        n_shots=n_shots,
    )
    _mux = target.replace("Q", "")
    mux = int(int(_mux) // 4)
    return Result(
        data={
            "result": result,
            "mux_no": mux,
            "frequency_range": frequency_range,
            "readout_amplitude": readout_amplitude,
        }
    )


def fit_readout_parameters(
    result: Result,
    *,
    f_r: float,
    f_p: float | None = None,
    kappa_p: float | None = None,
    J: float | None = None,
    a: float | None = None,
    b: float | None = None,
    split_freq_width: float = 0.15,
) -> FitResult:
    """
    Fit readout parameters from characterize_readout_parameters output.

    Parameters
    ----------
    kappa_p : float
        Coupling strength between Purcell filter and transmission line [rad/ns]
    gamma_p : float
        Internal loss rate of Purcell filter [rad/ns]
    J : float
        Coupling strength between Purcell filter and resonator [rad/ns]
    gamma_r : float
        Internal loss rate of resonator [rad/ns]
    omega_d : NDArray
        Angular frequency of incident wave [rad/ns]
    omega_p : float
        Angular frequency of Purcell filter [rad/ns]
    omega_r : float
        Angular frequency of resonator [rad/ns]
    """
    scan_result = result.data.get("result", None)
    mux_no = result.data.get("mux_no", None)
    frequency_range = result.data.get("frequency_range", None)
    readout_amplitude = result.data.get("readout_amplitude", None)

    if scan_result is None:
        raise ValueError("result.data['result'] is missing.")
    if frequency_range is None:
        raise ValueError("result.data['frequency_range'] is missing.")

    phases = scan_result.data.get("phases_unwrap", np.nan)

    if a is None:
        a = (phases[-1] - phases[0]) / (frequency_range[-1] - frequency_range[0])
    if b is None:
        b = np.average(phases)
    if f_p is None:
        f_p = f_r
    if kappa_p is None:
        kappa_p = 2 * np.pi * 0.01  # GHz
    if J is None:
        J = 2 * np.pi * 0.01  # GHz

    idx = np.where(
        (frequency_range >= f_r - split_freq_width / 2)
        & (frequency_range <= f_r + split_freq_width / 2)
    )[0]
    _frequency_range = frequency_range[idx]
    _phases = phases[idx]

    bounds_params = [
        [0, 0, 9.5, 9.5, -np.inf, -np.inf],  # Lower bounds
        [np.inf, np.inf, 11.5, 11.5, np.inf, np.inf],  # Upper bounds
    ]

    initial_guess = [kappa_p, J, f_p, f_r, a, b]
    try:
        popt, pcov = curve_fit(
            _fit_func,
            _frequency_range,
            _phases,
            p0=initial_guess,
            bounds=bounds_params,
        )
    except Exception as e:
        return FitResult(
            status=FitStatus.ERROR,
            message=f"Fitting failed: {e}",
        )

    perr = np.sqrt(np.diag(pcov))

    def _calc_r2_score(
        data: NDArray,
        fit_data: NDArray,
    ) -> np.float64:
        ss_res = np.sum((data - fit_data) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        return 1 - (ss_res / ss_tot)

    y_pred = _fit_func(_frequency_range, *popt)
    r2_score = _calc_r2_score(_phases, y_pred)

    fig = viz.make_figure()
    fig.add_trace(
        go.Scatter(
            x=_frequency_range,
            y=_phases,
            mode="markers",
            name="Data",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_frequency_range,
            y=_fit_func(_frequency_range, *popt),
            mode="lines",
            name="Fit",
        )
    )
    fig.add_vline(
        x=popt[2],
        name="Purcell filter",
        showlegend=True,
        line=dict(color="red", dash="dash"),
        annotation=dict(
            text="",
            hovertext=f"purcell: {popt[2]:.8f} GHz",
            showarrow=False,
            hoverlabel=dict(bgcolor="red", font=dict(color="white")),
        ),
    )
    fig.add_vline(
        x=popt[3],
        name="Resonator",
        showlegend=True,
        line=dict(color="green", dash="dash"),
        annotation=dict(
            text="",
            hovertext=f"resonator: {popt[3]:.8f} GHz",
            showarrow=False,
            hoverlabel=dict(bgcolor="green", font=dict(color="white")),
        ),
    )
    fig.update_layout(
        title=dict(
            text="Characterization Readout Parameters",
            subtitle=dict(
                text=(
                    f"mux= {mux_no}, target_freq= {f_r:.2f} GHz, "
                    f"readout ampl = {readout_amplitude}, r2: {r2_score:.3f}"
                )
            ),
        ),
        xaxis_title="Drive frequency [GHz]",
        yaxis_title="Reflection coefficient",
        font=dict(size=14),
    )
    fig.show()

    print("Fitted parameters:")
    print(f"R² score: {r2_score:.4f}")
    print(
        f"purcell filter external linewidth (kappa_p/2π): {popt[0] / (2 * np.pi) * 1e3:.4f} ± {perr[0] / (2 * np.pi) * 1e3:.4f} MHz"
    )
    print(
        f"resonator and purcell coupling (J/2π)         : {popt[1] / (2 * np.pi) * 1e3:.4f} ± {perr[1] / (2 * np.pi) * 1e3:.4f} MHz"
    )
    print(
        f"purcell filter frequency (f_p)                : {popt[2]:.4f} ± {perr[2]:.4f} GHz"
    )
    print(
        f"resonator frequency (f_r)                     : {popt[3]:.4f} ± {perr[3]:.4f} GHz"
    )
    print(
        f"Internal loss for purcell filter (gamma_p/2π) : {0.0} MHz (assumed in fitting)"
    )
    print(
        f"Internal loss for resonator (gamma_r/2π)      : {0.0} MHz (assumed in fitting)"
    )
    print(
        f"a                                             : {popt[4]:.4f} ± {perr[4]:.4f} rad/√GHz"
    )
    print(
        f"attenation coeff (-a / √π / 10 * log_e(10))   : {-popt[4] / np.sqrt(np.pi) / 10 * np.log(10):.4f} ± {perr[4] / np.sqrt(np.pi) / 10 * np.log(10):.4f} /√GHz"
    )
    print(
        f"b                                             : {popt[5]:.4f} ± {perr[5]:.4f} rad"
    )

    data_payload = {
        "kappa_p": popt[0],
        "kappa_p_err": perr[0],
        "J": popt[1],
        "J_err": perr[1],
        "f_p": popt[2],
        "f_p_err": perr[2],
        "f_r": popt[3],
        "f_r_err": perr[3],
        "a": popt[4],
        "a_err": perr[4],
        "b": popt[5],
        "b_err": perr[5],
        "popt": popt,
        "pcov": pcov,
        "perr": perr,
        "r2": r2_score,
        "y_fit": y_pred,
    }
    if r2_score > 0.9:
        status = FitStatus.SUCCESS
        return FitResult(
            status=status,
            message="R² < 0.9",
            data=data_payload,
            figure=fig,
        )
    else:
        status = FitStatus.WARNING
        return FitResult(
            status=status,
            message=f"R² < 0.9: {r2_score:.4f}",
            data=data_payload,
            figure=fig,
        )


def _Gamma(
    kappa_p: float,
    gamma_p: float,
    J: float,
    gamma_r: float,
    omega_d: NDArray,
    omega_p: float,
    omega_r: float,
) -> NDArray:
    """
    Reflection coefficient when Purcell filter is present.

    Parameters
    ----------
    kappa_p : float
        Coupling strength between Purcell filter and transmission line [rad/ns]
    gamma_p : float
        Internal loss rate of Purcell filter [rad/ns]
    J : float
        Coupling strength between Purcell filter and resonator [rad/ns]
    gamma_r : float
        Internal loss rate of resonator [rad/ns]
    omega_d : NDArray
        Angular frequency of incident wave [rad/ns]
    omega_p : float
        Angular frequency of Purcell filter [rad/ns]
    omega_r : float
        Angular frequency of resonator [rad/ns]

    Returns
    -------
    Gamma : complex
        Reflection coefficient
    """
    numerator = 4j * kappa_p * ((omega_r - omega_d) - 1j * gamma_r / 2)
    denominator = (2j * (omega_p - omega_d) + kappa_p + gamma_p) * (
        2j * (omega_r - omega_d) + gamma_r
    ) + 4 * J**2
    return 1 - numerator / denominator


def _fit_func(
    f_d: NDArray,
    kappa_p: float,
    J: float,
    f_p: float,
    f_r: float,
    a: float,
    b: float,
) -> NDArray:
    """
    Fit function for readout parameter characterization.

    Parameters
    ----------
    kappa_p : float
        Coupling strength between Purcell filter and transmission line [rad/ns]
    gamma_p : float
        Internal loss rate of Purcell filter [rad/ns]
    J : float
        Coupling strength between Purcell filter and resonator [rad/ns]
    gamma_r : float
        Internal loss rate of resonator [rad/ns]
    omega_d : NDArray
        Angular frequency of incident wave [rad/ns]
    omega_p : float
        Angular frequency of Purcell filter [rad/ns]
    omega_r : float
        Angular frequency of resonator [rad/ns]
    """
    omega_d = 2 * np.pi * f_d
    omega_p = 2 * np.pi * f_p
    omega_r = 2 * np.pi * f_r
    gamma_purcell = (
        2 * np.pi * 0
    )  # TODO add internal loss rate [GHz] to fitting parameters
    gamma_resonator = (
        2 * np.pi * 0
    )  # TODO add internal loss rate [GHz] to fitting parameters
    angle = np.angle(
        _Gamma(kappa_p, gamma_purcell, J, gamma_resonator, omega_d, omega_p, omega_r)
    )
    return -np.unwrap(angle) + a * np.sqrt(omega_d) + b
