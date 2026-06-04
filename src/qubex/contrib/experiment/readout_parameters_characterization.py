"""Contributed helpers for readout parameters characterization."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
import qxvisualizer as viz
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit
from tqdm import tqdm

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


def characterize_coarse_readout_parameters(
    exp: Experiment,
    *,
    target: str | None = None,
    frequency_range: ArrayLike | None = None,
    detuning_range: ArrayLike | None = None,
    readout_amplitudes: ArrayLike | None = None,
    time_range: ArrayLike | None = None,
    n_shots: int = 256,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = True,
) -> Result:
    """
    Characterize readout frequency and amplitude with Rabi-response heatmap.

    Parameters
    ----------
    exp : Experiment
        Qubex experiment instance.
    target : str, optional
        Target qubit label. Defaults to the first configured qubit.
    frequency_range : ArrayLike, optional
        Absolute readout frequencies to sweep in GHz. When omitted, uses the
        current resonator frequency plus `detuning_range`.
    detuning_range : ArrayLike, optional
        Readout-frequency detunings in GHz. Defaults to 11 points from -0.05
        to 0.05 GHz.
    readout_amplitudes : ArrayLike, optional
        Readout pulse amplitudes to sweep. When omitted, uses 7 points from 0
        to the currently configured readout amplitude, clipped to [0, 1].
    time_range : ArrayLike, optional
        Rabi drive time range in ns.
    n_shots : int, optional
        Number of shots for each Rabi experiment.
    shot_interval : float, optional
        Shot interval in ns.
    plot : bool, optional
        Whether to display the heatmap.
    save_image : bool, optional
        Whether to save the heatmap.

    Returns
    -------
    Result
        Rabi response-range heatmap and the maximum-response readout point.

    """
    if target is None:
        target = exp.qubit_labels[0]

    qubit_label = exp.ctx.resolve_qubit_label(target)
    resonator = exp.ctx.resonators[qubit_label]
    read_label = resonator.label
    if frequency_range is None:
        if detuning_range is None:
            detuning_range = np.linspace(-0.025, 0.025, 21)
        detuning_values = np.asarray(detuning_range, dtype=np.float64)
        frequency_values = float(resonator.frequency) + detuning_values
    else:
        frequency_values = np.asarray(frequency_range, dtype=np.float64)
        detuning_values = frequency_values - float(resonator.frequency)

    current_readout_amplitude = float(exp.ctx.params.readout_amplitude[qubit_label])
    if readout_amplitudes is None:
        amplitude_stop = np.clip(current_readout_amplitude, 0.0, 1.0)
        amplitude_values = np.linspace(0.0, amplitude_stop, 7)
    else:
        amplitude_values = np.asarray(readout_amplitudes, dtype=np.float64)
    if time_range is None:
        time_range = range(0, 101, 4)

    if frequency_values.ndim != 1 or frequency_values.size == 0:
        raise ValueError("frequency_range must be a non-empty 1D array.")
    if detuning_values.ndim != 1 or detuning_values.size == 0:
        raise ValueError("detuning_range must be a non-empty 1D array.")
    if amplitude_values.ndim != 1 or amplitude_values.size == 0:
        raise ValueError("readout_amplitudes must be a non-empty 1D array.")
    if np.any((amplitude_values < 0.0) | (amplitude_values > 1.0)):
        raise ValueError("readout_amplitudes must be within [0, 1].")

    heatmap_data = np.full(
        (amplitude_values.size, frequency_values.size),
        np.nan,
        dtype=np.float64,
    )
    rabi_results: list[object] = []
    original_readout_amplitudes = deepcopy(exp.ctx.params.readout_amplitude)
    try:
        with tqdm(
            total=amplitude_values.size * frequency_values.size,
            desc=f"readout rabi {qubit_label}",
        ) as progress:
            for amplitude_index, readout_amplitude in enumerate(amplitude_values):
                exp.ctx.params.readout_amplitude[qubit_label] = float(readout_amplitude)
                for frequency_index, readout_frequency in enumerate(frequency_values):
                    rabi_result = exp.rabi_experiment(
                        time_range=time_range,
                        amplitudes={
                            qubit_label: exp.ctx.params.control_amplitude[qubit_label]
                        },
                        frequencies={read_label: float(readout_frequency)},
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        plot=False,
                        store_params=False,
                    )
                    rabi_results.append(rabi_result)
                    rabi_data = rabi_result.data.get(qubit_label)
                    if rabi_data is None:
                        rabi_data = next(iter(rabi_result.data.values()))
                    heatmap_data[amplitude_index, frequency_index] = (
                        _rabi_response_range(rabi_data.data)
                    )
                    progress.update()
    finally:
        exp.ctx.params.readout_amplitude = original_readout_amplitudes

    finite_mask = np.isfinite(heatmap_data)
    if np.any(finite_mask):
        max_index = int(np.nanargmax(heatmap_data))
        amplitude_index, frequency_index = np.unravel_index(
            max_index,
            heatmap_data.shape,
        )
        optimal_readout_amplitude = float(amplitude_values[amplitude_index])
        optimal_readout_frequency = float(frequency_values[frequency_index])
        optimal_response_range = float(heatmap_data[amplitude_index, frequency_index])
    else:
        optimal_readout_amplitude = float("nan")
        optimal_readout_frequency = float("nan")
        optimal_response_range = float("nan")

    fig = _make_readout_rabi_heatmap_figure(
        target=qubit_label,
        frequency_range=frequency_values,
        readout_amplitudes=amplitude_values,
        heatmap_data=heatmap_data,
        optimal_readout_frequency=optimal_readout_frequency,
        optimal_readout_amplitude=optimal_readout_amplitude,
    )
    if plot:
        fig.show(config=viz.get_config(filename=f"readout_rabi_heatmap_{qubit_label}"))
    if save_image:
        viz.save_figure(
            fig,
            name=f"readout_rabi_heatmap_{qubit_label}",
            width=600,
            height=360,
        )

    return Result(
        data={
            "target": qubit_label,
            "frequency_range": frequency_values,
            "detuning_range": detuning_values,
            "readout_amplitudes": amplitude_values,
            "heatmap_data": heatmap_data,
            "optimal_readout_frequency": optimal_readout_frequency,
            "optimal_readout_amplitude": optimal_readout_amplitude,
            "optimal_response_range": optimal_response_range,
            "optimal_rabi_amplitude": optimal_response_range,
            "rabi_results": rabi_results,
        },
        figure=fig,
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
        f"purcell filter external linewidth (kappa_p): {popt[0] * 1e3:.4f} ± {perr[0] * 1e3:.4f} MHz"
    )
    print(
        f"resonator and purcell coupling (J)         : {popt[1] * 1e3:.4f} ± {perr[1] * 1e3:.4f} MHz"
    )
    print(
        f"purcell filter frequency (f_p)                : {popt[2]:.4f} ± {perr[2]:.4f} GHz"
    )
    print(
        f"resonator frequency (f_r)                     : {popt[3]:.4f} ± {perr[3]:.4f} GHz"
    )
    print(
        f"Internal loss for purcell filter (gamma_p) : {0.0} MHz (assumed in fitting)"
    )
    print(
        f"Internal loss for resonator (gamma_r)      : {0.0} MHz (assumed in fitting)"
    )
    print(
        f"a                                             : {popt[4]:.4f} ± {perr[4]:.4f} 1/√GHz"
    )
    print(
        f"attenation coeff (-a/ 10 * log_e(10))   : {-popt[4] / 10 * np.log(10):.4f} ± {perr[4] / 10 * np.log(10):.4f} /√GHz"
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


def _rabi_response_range(values: ArrayLike) -> float:
    """Return the maximum finite IQ-plane distance in a Rabi trace."""
    value_array = np.asarray(values)
    finite_values = value_array[np.isfinite(value_array)]
    if finite_values.size < 2:
        return float("nan")
    distances = np.abs(finite_values[:, np.newaxis] - finite_values[np.newaxis, :])
    return float(np.max(distances))


def _make_readout_rabi_heatmap_figure(
    *,
    target: str,
    frequency_range: NDArray[np.float64],
    readout_amplitudes: NDArray[np.float64],
    heatmap_data: NDArray[np.float64],
    optimal_readout_frequency: float,
    optimal_readout_amplitude: float,
) -> go.Figure:
    """Return a readout Rabi-amplitude heatmap figure."""
    fig = viz.make_figure()
    fig.add_trace(
        go.Heatmap(
            x=frequency_range,
            y=readout_amplitudes,
            z=heatmap_data,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(
                    text="Rabi response range",
                    side="right",
                )
            ),
        )
    )
    if np.isfinite(optimal_readout_frequency) and np.isfinite(
        optimal_readout_amplitude
    ):
        fig.add_trace(
            go.Scatter(
                x=[optimal_readout_frequency],
                y=[optimal_readout_amplitude],
                mode="markers",
                marker=dict(
                    color="red",
                    size=10,
                    symbol="x",
                    line=dict(color="white", width=1),
                ),
                name="Max",
            )
        )
    fig.update_layout(
        title=f"Readout Rabi response range : {target}",
        xaxis_title="Readout frequency (GHz)",
        yaxis_title="Readout amplitude",
        width=600,
        height=360,
        margin=dict(t=70),
    )
    return fig


def _Gamma(
    omega_kappa_p: float,
    omega_gamma_p: float,
    omega_J: float,
    omega_gamma_r: float,
    omega_d: NDArray,
    omega_p: float,
    omega_r: float,
) -> NDArray:
    """
    Reflection coefficient when Purcell filter is present.

    Parameters
    ----------
    omega_kappa_p : float
        Coupling strength between Purcell filter and transmission line [rad/ns]
    omega_gamma_p : float
        Internal loss rate of Purcell filter [rad/ns]
    omega_J : float
        Coupling strength between Purcell filter and resonator [rad/ns]
    omega_gamma_r : float
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
    numerator = 4j * omega_kappa_p * ((omega_r - omega_d) - 1j * omega_gamma_r / 2)
    denominator = (2j * (omega_p - omega_d) + omega_kappa_p + omega_gamma_p) * (
        2j * (omega_r - omega_d) + omega_gamma_r
    ) + 4 * omega_J**2
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
        Coupling strength between Purcell filter and transmission line [1/ns]
    gamma_p : float
        Internal loss rate of Purcell filter [1/ns]
    J : float
        Coupling strength between Purcell filter and resonator [1/ns]
    gamma_r : float
        Internal loss rate of resonator [1/ns]
    f_d : NDArray
        Angular frequency of incident wave [1/ns]
    f_p : float
        Angular frequency of Purcell filter [1/ns]
    f_r : float
        Angular frequency of resonator [1/ns]
    a: float
        term for attenuation dependent on frequency [/√GHz]
    b: float
        offset. [rad]
    """
    omega_kappa_p = 2 * np.pi * kappa_p
    omega_J = 2 * np.pi * J
    omega_d = 2 * np.pi * f_d
    omega_p = 2 * np.pi * f_p
    omega_r = 2 * np.pi * f_r
    omega_gamma_purcell = (
        2 * np.pi * 0
    )  # TODO add internal loss rate [GHz] to fitting parameters
    omega_gamma_resonator = (
        2 * np.pi * 0
    )  # TODO add internal loss rate [GHz] to fitting parameters
    angle = np.angle(
        _Gamma(
            omega_kappa_p,
            omega_gamma_purcell,
            omega_J,
            omega_gamma_resonator,
            omega_d,
            omega_p,
            omega_r,
        )
    )  # TODO add stack real and imaginary part.
    return -np.unwrap(angle) + np.sqrt(np.pi) * a * np.sqrt(omega_d) + b
