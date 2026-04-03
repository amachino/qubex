from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import qxvisualizer as viz
from scipy.optimize import curve_fit

from qubex.experiment.experiment import Experiment
from qubex.experiment.models.result import Result


def characterize_readout_parameters(
    exp: Experiment,
    *,
    target: str | None = None,
    frequency_range: np.ndarray,
    readout_amplitude: float | None = None,
    n_shots: int = 1024,
    save_image: bool = True,
) -> CharacterizeReadoutParametersResult:

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
        return self.result.data.get("phases_unwrap", np.nan)

    @property
    def signals(self) -> np.ndarray:
        return self.result.data.get("signals", np.nan)

    def fit(
        self,
        *,
        f_r: float,
        f_p: float | None = None,
        kappa_p: float | None = None,
        J: float | None = None,
        a: float | None = None,
        b: float | None = None,
        split_freq_width: float = 0.15,
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
            kappa_p = 2 * np.pi * 0.01  # GHz
        if J is None:
            J = 2 * np.pi * 0.01  # GHz

        idx = np.where(
            (self.frequency_range >= f_r - split_freq_width / 2)
            & (self.frequency_range <= f_r + split_freq_width / 2)
        )[0]
        _frequency_range = self.frequency_range[idx]
        _phases = self.phases[idx]

        bounds_params = [
            [0, 0, 0, 0, -np.inf, -np.inf],  # Lower bounds
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],  # Upper bounds
        ]

        initial_guess = [kappa_p, J, f_p, f_r, a, b]
        popt, pcov = curve_fit(
            _fit_func,
            _frequency_range,
            _phases,
            p0=initial_guess,
            bounds=bounds_params,
        )

        perr = np.sqrt(np.diag(pcov))

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
        fig.add_vline(
            x=popt[2],
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
                    text=f"target_freq= {f_r:.2f} GHz, readout ampl = {self.readout_amplitude}, r2: {r2_score:.3f}"
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
            f"purcell filter external linewidth (kappa_p/2π): {popt[0] / (2 * np.pi) * 1e3:.8f} ± {perr[0] / (2 * np.pi) * 1e3:.8f} MHz"
        )
        print(
            f"resonator and purcell coupling (J/2π)         : {popt[1] / (2 * np.pi) * 1e3:.8f} ± {perr[1] / (2 * np.pi) * 1e3:.8f} MHz"
        )
        print(
            f"purcell filter frequency (f_p)                : {popt[2]:.8f} ± {perr[2]:.8f} GHz"
        )
        print(
            f"resonator frequency (f_r)                     : {popt[3]:.8f} ± {perr[3]:.8f} GHz"
        )
        print(
            f"a                                             : {popt[4]:.8f} ± {perr[4]:.8f} rad/√GHz"
        )
        print(
            f"attenation coeff (-a / √π / 10 * log_e(10))   : {-popt[4] / np.sqrt(np.pi) / 10 * np.log(10):.8f} ± {perr[4] / np.sqrt(np.pi) / 10 * np.log(10):.8f} /√GHz"
        )
        print(
            f"b                                             : {popt[5]:.8f} ± {perr[5]:.8f} rad"
        )

        return {
            "popt": popt,
            "pcov": pcov,
            "perr": perr,
            "r2_score": r2_score,
            "y_pred": y_pred,
        }


def _Gamma(kappa_p, gamma_p, J, gamma_r, omega_d, omega_p, omega_r):
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
    omega_d : float
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


def _fit_func(f_d, kappa_p, J, f_p, f_r, a, b):
    omega_d = 2 * np.pi * f_d
    omega_p = 2 * np.pi * f_p
    omega_r = 2 * np.pi * f_r
    gamma_purcell = 2 * np.pi * 0  # Purcell: filterのinternal loss rate [GHz]
    gamma_resonator = 2 * np.pi * 0  # Resonator: internal loss rate [GHz]
    angle = np.angle(
        _Gamma(kappa_p, gamma_purcell, J, gamma_resonator, omega_d, omega_p, omega_r)
    )
    return -np.unwrap(angle) + a * np.sqrt(omega_d) + b
