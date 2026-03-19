from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import qxvisualizer as viz
import scipy.optimize as opt
from scipy.optimize import curve_fit

from qubex.experiment.experiment import Experiment
from qubex.experiment.models.result import Result


def characterize_readout_parameters(
    exp: Experiment,
    *,
    mux: int,
    frequency_range: np.ndarray,
    readout_amplitude: float,
    n_shots: int = 1024,
    save_image: bool = True,
) -> CharacterizeReadoutParametersResult:

    target = f"Q{4 * mux}"

    if readout_amplitude is None:
        readout_amplitude = 0.01

    result = exp.scan_resonator_frequencies(
        target,
        frequency_range=frequency_range,
        readout_amplitude=readout_amplitude,
        save_image=save_image,
        n_shots=n_shots,
    )
    return CharacterizeReadoutParametersResult(
        result=result,
        mux=mux,
        frequency_range=frequency_range,
        readout_amplitude=readout_amplitude,
    )


@dataclass
class CharacterizeReadoutParametersResult:
    result: Result
    mux: int
    frequency_range: np.ndarray
    readout_amplitude: float

    @property
    def phases(self) -> np.ndarray:
        return self.result.data.get("phase_unwrap", np.nan)

    @property
    def signals(self) -> np.ndarray:
        return self.result.data.get("signal", np.nan)

    def fit(
        self,
        f_r: float,
        f_p: float | None = None,
        kappa_p: float | None = None,
        J: float | None = None,
        a: float | None = None,  # 1/GHz
        b: float | None = None,  # rad
        split_freq_width: float = 0.15,  # GHz
        mode: Literal["least squares", "curve fit"] = "curve fit",
    ):
        if a is None:
            a = (self.phases[-1] - self.phases[0]) / (
                self.frequency_range[-1] - self.frequency_range[0]
            )
        if b is None:
            b = np.average(self.phases)
        if f_p is None:
            f_p = f_r
        if kappa_p is None:
            kappa_p = 0.01  # GHz
        if J is None:
            J = 0.01  # GHz

        idx = np.where(
            (self.frequency_range >= f_r - split_freq_width)
            & (self.frequency_range <= f_r + split_freq_width)
        )[0]
        _frequency_range = self.frequency_range[idx]
        _phases = self.phases[idx]

        bounds_params = [
            [0, 0, 0, 0, -np.inf, -np.inf],  # Lower bounds
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],  # Upper bounds
        ]

        if mode == "least squares":

            def residuals(params, x, y):
                return y - _fit_func(x, *params)

            initial_guess = [kappa_p, J, f_p, f_r, a, b]
            res = opt.least_squares(
                residuals,
                initial_guess,
                args=(_frequency_range, _phases),
                bounds=bounds_params,
            )
            popt = res.x
        elif mode == "curve fit":
            initial_guess = [kappa_p, J, f_p, f_r, a, b]
            popt, _ = curve_fit(
                _fit_func,
                _frequency_range,
                _phases,
                p0=initial_guess,
                bounds=bounds_params,
            )

        def _calc_r2_score(data, fit_data):
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
        fig.update_layout(
            title=dict(
                text="Characterization Readout Parameters",
                subtitle=dict(
                    text=f"target_freq= {f_r:.2f} GHz, readout ampl = {self.readout_amplitude}, r2: {r2_score:.3f}"
                ),
            ),
            xaxis_title="Drive frequency [GHz]",
            yaxis_title="Reflection coefficient",
            font=dict(size=14),
        )
        fig.show()
        print("Fitted parameters:")
        print(f"R² score: {np.round(r2_score, 6)}")
        print(f"kappa_p/2π: {np.round(popt[0] / (2 * np.pi) * 1e3, 8)} MHz")
        print(f"J/2π: {np.round(popt[1] / (2 * np.pi) * 1e3, 8)} MHz")
        print(f"f_p: {np.round(popt[2], 8)} GHz")
        print(f"f_r: {np.round(popt[3], 8)} GHz")
        print(f"a: {np.round(popt[4], 10)} /GHz")
        print(f"b: {np.round(popt[5], 8)} rad")

        return {
            "popt": popt,
            "r2_score": r2_score,
            "y_pred": y_pred,
        }


def _Gamma(kappa_p, gamma_p, J, gamma_r, f_d, f_p, f_r):
    """
    Reflection coefficient when Purcell filter is present.

    Parameters
    ----------
    kappa_p : float
        Coupling strength between Purcell filter and transmission line [Hz]
    gamma_p : float
        Internal loss rate of Purcell filter [Hz]
    J : float
        Coupling strength between Purcell filter and resonator [Hz]
    gamma_r : float
        Internal loss rate of resonator [Hz]
    f_d : float
        Frequency of incident wave [Hz]
    f_p : float
        Resonant frequency of Purcell filter [Hz]
    f_r : float
        Resonant frequency of resonator [Hz]

    Returns
    -------
    Gamma : complex
        Reflection coefficient
    """
    omega_r = 2 * np.pi * f_r
    omega_p = 2 * np.pi * f_p
    omega_d = 2 * np.pi * f_d
    kappa_p = 2 * np.pi * kappa_p
    gamma_p = 2 * np.pi * gamma_p
    J = 2 * np.pi * J

    numerator = 4j * kappa_p * (2 * np.pi * (omega_r - omega_d) - 1j * gamma_r / 2)
    denominator = (2j * 2 * np.pi * (omega_p - omega_d) + kappa_p + gamma_p) * (
        2j * 2 * np.pi * (omega_r - omega_d) + gamma_r
    ) + 4 * J**2
    return 1 - numerator / denominator


def _fit_func(f_d, kappa_p, J, f_p, f_r, a, b):
    gamma_purcell = 2 * np.pi * 0  # Purcell: filterのinternal loss rate [GHz]
    gamma_resonator = 2 * np.pi * 0  # Resonator: internal loss rate [GHz]
    angle = np.angle(_Gamma(kappa_p, gamma_purcell, J, gamma_resonator, f_d, f_p, f_r))
    return -np.unwrap(angle) + a * f_d + b
