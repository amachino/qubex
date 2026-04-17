"""
Quantum-efficiency characterization helpers.

This module implements reusable contrib experiments for dispersive readout
characterization based on arXiv:1711.05336. The main public APIs are:

- :func:`readout_snr`
- :func:`sweep_readout_snr`
- :func:`measurement_induced_dephasing`
- :func:`measurement_induced_dephasing_experiment`
- :func:`quantum_efficiency_measurement`

The final quantum efficiency is always estimated from fitted scalings,

- ``SNR(epsilon) = a epsilon``
- ``|rho01|(epsilon) = b exp(-epsilon^2 / (2 sigma_m^2))``
- ``eta = a^2 sigma_m^2 / 2``

and is never obtained from a pointwise average over amplitudes.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, TypedDict

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.analysis import FitResult, FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    SAMPLING_PERIOD,
)
from qubex.experiment.models import Result
from qubex.pulse import PulseSchedule

from ._deprecated_options import resolve_shot_options

__all__ = [
    "analyze_quantum_efficiency",
    "compute_readout_snr",
    "measurement_induced_dephasing",
    "measurement_induced_dephasing_experiment",
    "quantum_efficiency_measurement",
    "readout_snr",
    "sweep_readout_snr",
]

ZERO_TOLERANCE = 1e-15
READOUT_SAMPLE_PERIOD_NS = 2.0

SNR_FIGURE_KEY = "snr"
DEPHASING_FIGURE_KEY = "dephasing"
OVERVIEW_FIGURE_KEY = "overview"
RAMSEY_FIGURE_KEY = "ramsey"
PROJECTION_FIGURE_KEY = "projection"
WEIGHT_FIGURE_KEY = "weight"


class RamseyFringeSummary(TypedDict):
    """Typed Ramsey-fringe fit payload."""

    rho01: float
    phi0: float
    offset: float
    cosine: float
    sine: float
    p1: NDArray[np.float64]
    p1_fit: NDArray[np.float64]
    sigma_z: NDArray[np.float64]
    sigma_z_fit: NDArray[np.float64]


class ReadoutSnrSummary(TypedDict):
    """Typed single-amplitude readout-SNR payload."""

    snr: float
    signal: float
    noise: float
    mu_ground: float
    mu_excited: float
    sigma_ground: float
    sigma_excited: float
    weights: NDArray[np.complex128]
    projected_ground: NDArray[np.float64]
    projected_excited: NDArray[np.float64]


def _show_figure(fig: go.Figure, filename: str) -> None:
    """Show one figure with the shared Qubex Plotly config."""
    fig.show(config=viz.get_config(filename=filename))


def _title_with_subtitle(text: str, subtitle: str) -> dict[str, object]:
    """Return one Plotly title payload with a small monospace subtitle."""
    return {
        "text": text,
        "subtitle": {
            "text": subtitle,
            "font": {"size": 11, "family": "monospace"},
        },
    }


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    """Normalize one target selector into a concrete target list."""
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _aggregate_target_results(results: dict[str, Result]) -> Result:
    """Aggregate per-target results while keeping single-target behavior."""
    if len(results) == 1:
        return next(iter(results.values()))

    data: dict[str, object] = {}
    figures: dict[str, go.Figure] = {}
    for target, result in results.items():
        data[target] = result.data
        if result.figure is not None:
            figures[target] = result.figure
        for key, figure in (result.figures or {}).items():
            figures[f"{target}:{key}"] = figure
    return Result(data=data, figures=figures)


def _resolve_plot_options(
    *,
    plot: bool | None,
    save_image: bool | None,
) -> tuple[bool, bool]:
    """Normalize plotting options to explicit booleans."""
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    return plot, save_image


def _normalize_float_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int = 1,
) -> NDArray[np.float64]:
    """Return one finite float array with the requested dimensionality."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"`{name}` must be a {ndim}D array.")
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return array


def _normalize_complex_raw(raw: ArrayLike, *, name: str) -> NDArray[np.complex128]:
    """Normalize one raw IQ payload into shape ``(shots,)`` or ``(shots, samples)``."""
    array = np.asarray(raw)
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")

    if np.iscomplexobj(array):
        normalized = np.asarray(array, dtype=np.complex128)
    elif array.ndim >= 1 and array.shape[-1] == 2:
        normalized = array[..., 0].astype(np.float64) + 1j * array[..., 1].astype(
            np.float64
        )
    else:
        raise ValueError(
            f"`{name}` must be a complex array or a real array with a last axis of length 2."
        )

    if normalized.ndim not in (1, 2):
        raise ValueError(f"`{name}` must have shape (shots,) or (shots, samples).")
    if not np.all(np.isfinite(np.real(normalized))) or not np.all(
        np.isfinite(np.imag(normalized))
    ):
        raise ValueError(f"`{name}` must contain only finite values.")
    return normalized


def _normalize_raw_series(
    raw_series: Sequence[ArrayLike] | ArrayLike,
    *,
    n_amplitudes: int,
    name: str,
) -> list[NDArray[np.complex128]]:
    """Normalize one amplitude-indexed sequence of raw IQ arrays."""
    if isinstance(raw_series, np.ndarray):
        raw_array = raw_series
        if raw_array.dtype == object:
            raw_array = None
        else:
            if raw_array.shape[0] != n_amplitudes:
                raise ValueError(
                    f"`{name}` first dimension must match the number of amplitudes."
                )
            return [
                _normalize_complex_raw(raw_array[index], name=f"{name}[{index}]")
                for index in range(n_amplitudes)
            ]
    if not isinstance(raw_series, Sequence) or len(raw_series) != n_amplitudes:
        raise ValueError(
            f"`{name}` must be a sequence with one entry for each readout amplitude."
        )
    return [
        _normalize_complex_raw(raw_series[index], name=f"{name}[{index}]")
        for index in range(n_amplitudes)
    ]


def _dense_fit_axis(
    values: NDArray[np.float64], *, n_points: int = 500
) -> NDArray[np.float64]:
    """Return one dense axis spanning the input range."""
    return np.linspace(float(np.min(values)), float(np.max(values)), n_points)


def _fit_ramsey_fringe(
    phases: ArrayLike,
    excited_probabilities: ArrayLike,
) -> RamseyFringeSummary:
    """
    Fit one Ramsey fringe and extract ``|rho01|`` and ``phi0``.

    The fit follows the notebook implementation and the Ramsey fringe picture
    in arXiv:1711.05336:

    - ``sigma_z(phi) = c + a cos(phi) + b sin(phi)``
    - ``|rho01| = 0.5 * sqrt(a^2 + b^2)``
    - ``phi0 = atan2(-b, a)``
    """
    phase_array = _normalize_float_array(phases, name="phases")
    probability_array = _normalize_float_array(
        excited_probabilities,
        name="excited_probabilities",
    )
    if len(phase_array) != len(probability_array):
        raise ValueError(
            "`phases` and `excited_probabilities` must have the same length."
        )

    sigma_z = 1.0 - 2.0 * probability_array
    design = np.column_stack(
        [
            np.ones_like(phase_array),
            np.cos(phase_array),
            np.sin(phase_array),
        ]
    )
    offset, cosine, sine = np.linalg.lstsq(design, sigma_z, rcond=None)[0]

    amplitude_sigma_z = float(np.hypot(cosine, sine))
    sigma_z_fit = offset + cosine * np.cos(phase_array) + sine * np.sin(phase_array)

    return {
        "rho01": 0.5 * amplitude_sigma_z,
        "phi0": float(np.arctan2(-sine, cosine)),
        "offset": float(offset),
        "cosine": float(cosine),
        "sine": float(sine),
        "p1": probability_array,
        "p1_fit": 0.5 * (1.0 - sigma_z_fit),
        "sigma_z": sigma_z,
        "sigma_z_fit": sigma_z_fit,
    }


def _measure_ramsey_fringe(
    exp: Experiment,
    target: str,
    *,
    readout_amplitude: float,
    phase_range: NDArray[np.float64],
    n_shots: int | None,
    shot_interval: float | None,
) -> RamseyFringeSummary:
    """Run one weak-measurement Ramsey sweep and fit the resulting fringe."""
    excited_probabilities: list[float] = []
    readout_target = exp.ctx.resolve_read_label(target)

    for phase_shift in phase_range:
        with PulseSchedule() as schedule:
            schedule.add(target, exp.pulse.x90(target))
            schedule.barrier()
            schedule.add(
                readout_target,
                exp.pulse.readout(readout_target, amplitude=readout_amplitude),
            )
            schedule.barrier()
            schedule.add(target, exp.pulse.x90(target).shifted(float(phase_shift)))
            schedule.barrier()
            schedule.add(readout_target, exp.pulse.readout(readout_target))
        measurement = exp.execute(
            schedule=schedule,
            mode="single",
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
        )
        readouts = measurement.data[target]
        final_readout = readouts[len(readouts) - 1]
        probabilities = np.asarray(final_readout.probabilities, dtype=np.float64)
        if probabilities.size < 2:
            raise ValueError(
                "The final readout must contain at least two state probabilities."
            )
        excited_probabilities.append(float(probabilities[1]))

    return _fit_ramsey_fringe(phase_range, excited_probabilities)


def compute_readout_snr(
    raw_ground: ArrayLike,
    raw_excited: ArrayLike,
    *,
    zero_tolerance: float = ZERO_TOLERANCE,
) -> ReadoutSnrSummary:
    """
    Compute the single-shot readout SNR for one readout amplitude.

    For waveform IQ data, the projection weight is the mean state difference
    ``mean_e - mean_g``. For already-integrated IQ data, the projection axis is
    the mean-state separation. The final scalar SNR is

    ``SNR = |mu_e - mu_g| / (0.5 * (sigma_g + sigma_e))``.
    """
    ground = _normalize_complex_raw(raw_ground, name="raw_ground")
    excited = _normalize_complex_raw(raw_excited, name="raw_excited")
    if ground.shape != excited.shape:
        raise ValueError("`raw_ground` and `raw_excited` must have the same shape.")

    if ground.ndim == 1:
        axis = np.mean(excited) - np.mean(ground)
        if abs(axis) < zero_tolerance:
            weights = np.asarray(0.0 + 0.0j, dtype=np.complex128)
            projected_ground = np.zeros_like(np.real(ground), dtype=np.float64)
            projected_excited = np.zeros_like(np.real(excited), dtype=np.float64)
        else:
            normalized_axis = axis / abs(axis)
            weights = np.asarray(normalized_axis, dtype=np.complex128)
            projected_ground = np.real(ground * np.conj(normalized_axis))
            projected_excited = np.real(excited * np.conj(normalized_axis))
    else:
        weights = np.asarray(np.mean(excited, axis=0) - np.mean(ground, axis=0))
        if np.linalg.norm(weights) < zero_tolerance:
            projected_ground = np.zeros(ground.shape[0], dtype=np.float64)
            projected_excited = np.zeros(excited.shape[0], dtype=np.float64)
        else:
            projected_ground = np.real(ground @ np.conj(weights))
            projected_excited = np.real(excited @ np.conj(weights))

    mu_ground = float(np.mean(projected_ground))
    mu_excited = float(np.mean(projected_excited))
    sigma_ground = (
        float(np.std(projected_ground, ddof=1)) if len(projected_ground) > 1 else 0.0
    )
    sigma_excited = (
        float(np.std(projected_excited, ddof=1)) if len(projected_excited) > 1 else 0.0
    )

    signal = abs(mu_excited - mu_ground)
    noise = 0.5 * (sigma_ground + sigma_excited)
    snr = 0.0 if noise < zero_tolerance else signal / noise

    return {
        "snr": float(snr),
        "signal": float(signal),
        "noise": float(noise),
        "mu_ground": mu_ground,
        "mu_excited": mu_excited,
        "sigma_ground": sigma_ground,
        "sigma_excited": sigma_excited,
        "weights": np.asarray(weights, dtype=np.complex128),
        "projected_ground": np.asarray(projected_ground, dtype=np.float64),
        "projected_excited": np.asarray(projected_excited, dtype=np.float64),
    }


def _measure_readout_snr(
    exp: Experiment,
    target: str,
    *,
    readout_amplitude: float,
    n_shots: int,
    shot_interval: float | None,
    readout_duration: float | None,
) -> ReadoutSnrSummary:
    """Acquire the raw single-shot data and compute one readout SNR summary."""
    distributions = exp.measure_state_distribution(
        targets=target,
        n_states=2,
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_duration=readout_duration,
        readout_amplitudes={target: float(readout_amplitude)},
        plot=False,
    )
    ground_raw = distributions[0].data[target].raw
    excited_raw = distributions[1].data[target].raw
    return compute_readout_snr(ground_raw, excited_raw)


def _fit_measurement_induced_dephasing(
    *,
    target: str,
    readout_amplitudes: NDArray[np.float64],
    rho01_values: NDArray[np.float64],
) -> FitResult:
    """
    Fit the weak-measurement Ramsey coherence envelope.

    The fitted model is

    ``|rho01|(epsilon) = b exp(-epsilon^2 / (2 sigma_m^2))``.

    Following arXiv:1711.05336, the same ``sigma_m`` gives

    ``beta_m_fit(epsilon) = epsilon^2 / (2 sigma_m^2)``.
    """
    amplitudes = _normalize_float_array(readout_amplitudes, name="readout_amplitudes")
    rho01 = _normalize_float_array(rho01_values, name="rho01_values")
    if len(amplitudes) != len(rho01):
        raise ValueError(
            "`readout_amplitudes` and `rho01_values` must have the same length."
        )
    if np.any(rho01 <= 0):
        raise ValueError(
            "`rho01_values` must be strictly positive for dephasing fitting."
        )

    def model(
        epsilon: NDArray[np.float64],
        b: float,
        sigma_m: float,
    ) -> NDArray[np.float64]:
        return b * np.exp(-(epsilon**2) / (2.0 * sigma_m**2))

    from scipy.optimize import curve_fit  # lazy import

    initial_b = float(np.max(rho01))
    target_rho01 = initial_b / np.e
    sigma_guess_index = int(np.argmin(np.abs(rho01 - target_rho01)))
    sigma_guess = float(abs(amplitudes[sigma_guess_index]))
    if sigma_guess <= 0:
        sigma_guess = max(float(np.max(np.abs(amplitudes))) / 2.0, 1e-6)

    try:
        popt, pcov = curve_fit(
            model,
            amplitudes,
            rho01,
            p0=[initial_b, sigma_guess],
            bounds=([0.0, 1e-12], [np.inf, np.inf]),
            maxfev=20_000,
        )
    except Exception as exc:
        return FitResult(
            status=FitStatus.ERROR,
            message=f"Measurement-induced dephasing fitting failed: {exc}",
        )

    b_fit = float(popt[0])
    sigma_m_fit = float(popt[1])
    b_err, sigma_m_err = np.sqrt(np.diag(pcov))
    fit_amplitudes = _dense_fit_axis(amplitudes)
    rho01_fit = model(fit_amplitudes, *popt)
    beta_m_raw = -np.log(np.clip(rho01 / b_fit, 1e-12, None))
    beta_m_fit = (fit_amplitudes**2) / (2.0 * sigma_m_fit**2)

    figure = viz.make_figure()
    figure.add_trace(
        go.Scatter(
            x=amplitudes,
            y=rho01,
            mode="markers",
            name="data",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=fit_amplitudes,
            y=rho01_fit,
            mode="lines",
            name="fit",
        )
    )
    figure.update_layout(
        title=f"Measurement-induced dephasing : {target}",
        xaxis_title="Readout amplitude",
        yaxis_title="|ρ<sub>01</sub>|",
        width=600,
        height=400,
        showlegend=True,
    )
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"σ<sub>m</sub> = {sigma_m_fit:.3g}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    return FitResult(
        status=FitStatus.SUCCESS,
        message="Measurement-induced dephasing fitting successful.",
        data={
            "b": b_fit,
            "b_err": float(b_err),
            "sigma_m": sigma_m_fit,
            "sigma_m_err": float(sigma_m_err),
            "popt": popt,
            "pcov": pcov,
            "rho01": rho01,
            "rho01_fit": rho01_fit,
            "fit_amplitudes": fit_amplitudes,
            "beta_m_raw": beta_m_raw,
            "beta_m_fit": beta_m_fit,
            "fig": figure,
        },
        figure=figure,
    )


def _fit_snr_sweep(
    *,
    target: str,
    readout_amplitudes: NDArray[np.float64],
    snr_values: NDArray[np.float64],
) -> FitResult:
    """Fit the origin-constrained SNR scaling ``SNR(epsilon) = a epsilon``."""
    amplitudes = _normalize_float_array(readout_amplitudes, name="readout_amplitudes")
    snr = _normalize_float_array(snr_values, name="snr_values")
    if len(amplitudes) != len(snr):
        raise ValueError(
            "`readout_amplitudes` and `snr_values` must have the same length."
        )

    base_fit = fitting.fit_linear(
        x=amplitudes,
        y=snr,
        intercept=False,
        plot=False,
        target=target,
        title="Readout SNR",
        xlabel="Readout amplitude",
        ylabel="Signal-to-noise ratio",
        xaxis_type="linear",
        yaxis_type="linear",
    )
    if base_fit.status is not FitStatus.SUCCESS:
        return base_fit

    slope = float(base_fit["a"])
    fit_amplitudes = _dense_fit_axis(amplitudes)
    snr_fit = slope * fit_amplitudes
    figure = viz.make_figure()
    figure.add_trace(
        go.Scatter(
            x=amplitudes,
            y=snr,
            mode="markers",
            name="data",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=fit_amplitudes,
            y=snr_fit,
            mode="lines",
            name="fit",
        )
    )
    figure.update_layout(
        title=f"Readout SNR : {target}",
        xaxis_title="Readout amplitude",
        yaxis_title="Signal-to-noise ratio",
        width=600,
        height=400,
        showlegend=True,
    )
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"a = {slope:.3g}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    return FitResult(
        status=FitStatus.SUCCESS,
        message=base_fit.message,
        data={
            **dict(base_fit.data),
            "fit_amplitudes": fit_amplitudes,
            "y_fit": snr_fit,
            "fig": figure,
        },
        figure=figure,
    )


def _build_quantum_efficiency_result(
    *,
    target: str,
    readout_amplitudes: NDArray[np.float64],
    dephasing_result: Result,
    snr_result: Result,
    plot: bool,
    save_image: bool,
    source: Mapping[str, object] | None = None,
) -> Result:
    """Assemble the final fit-based quantum-efficiency result."""
    dephasing_fit = dephasing_result["fit_result"]
    snr_fit = snr_result["fit_result"]

    slope = float(snr_fit["a"])
    sigma_m = float(dephasing_fit["sigma_m"])
    quantum_efficiency = 0.5 * slope**2 * sigma_m**2

    snr_values = np.asarray(snr_result["snr"], dtype=np.float64)
    rho01_values = np.asarray(dephasing_result["rho01"], dtype=np.float64)
    beta_m_raw = np.asarray(dephasing_fit["beta_m_raw"], dtype=np.float64)
    pointwise_eta = np.full_like(readout_amplitudes, np.nan, dtype=np.float64)
    valid = beta_m_raw > ZERO_TOLERANCE
    pointwise_eta[valid] = snr_values[valid] ** 2 / (4.0 * beta_m_raw[valid])

    overview_figure = viz.make_figure()
    overview_figure.set_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    overview_figure.add_trace(
        go.Scatter(
            x=readout_amplitudes,
            y=snr_values,
            mode="markers",
            name="SNR data",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    overview_figure.add_trace(
        go.Scatter(
            x=np.asarray(snr_fit["fit_amplitudes"], dtype=np.float64),
            y=np.asarray(snr_fit["y_fit"], dtype=np.float64),
            mode="lines",
            name="SNR fit",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    overview_figure.add_trace(
        go.Scatter(
            x=readout_amplitudes,
            y=rho01_values,
            mode="markers",
            name="ρ<sub>01</sub> data",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    overview_figure.add_trace(
        go.Scatter(
            x=np.asarray(dephasing_fit["fit_amplitudes"], dtype=np.float64),
            y=np.asarray(dephasing_fit["rho01_fit"], dtype=np.float64),
            mode="lines",
            name="ρ<sub>01</sub> fit",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    overview_figure.update_xaxes(title_text="Readout amplitude", row=1, col=1)
    overview_figure.update_yaxes(
        title_text="|ρ<sub>01</sub>|",
        row=1,
        col=1,
        secondary_y=False,
    )
    overview_figure.update_yaxes(
        title_text="Signal-to-noise ratio",
        row=1,
        col=1,
        secondary_y=True,
    )
    overview_figure.update_layout(
        title=f"Quantum efficiency measurement : {target}",
        width=600,
        height=400,
        showlegend=True,
    )
    overview_figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"η = {quantum_efficiency:.3g}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )
    if plot:
        _show_figure(overview_figure, f"quantum_efficiency_measurement_{target}")
    if save_image:
        viz.save_figure(
            overview_figure,
            name=f"quantum_efficiency_measurement_{target}",
        )
        viz.save_figure(snr_result.get_figure(), name=f"readout_snr_{target}")
        viz.save_figure(
            dephasing_result.get_figure(),
            name=f"measurement_induced_dephasing_{target}",
        )

    data: dict[str, object] = {
        "target": target,
        "readout_amplitudes": readout_amplitudes,
        "quantum_efficiency": float(quantum_efficiency),
        "pointwise_quantum_efficiency": pointwise_eta,
        "snr": {
            "values": np.asarray(snr_result["snr"], dtype=np.float64),
            "signal": np.asarray(snr_result["signal"], dtype=np.float64),
            "noise": np.asarray(snr_result["noise"], dtype=np.float64),
            "projection": snr_result["projection"],
            "fit_result": snr_fit,
        },
        "measurement_induced_dephasing": {
            "rho01": rho01_values,
            "beta_m_raw": beta_m_raw,
            "ramsey": dephasing_result["ramsey"],
            "fit_result": dephasing_fit,
        },
        "ramsey": dephasing_result["ramsey"],
        "snr_result": snr_result,
        "measurement_induced_dephasing_result": dephasing_result,
        "snr_fit": snr_fit,
        "measurement_induced_dephasing_fit": dephasing_fit,
        "fig": overview_figure,
    }
    if source is not None:
        data["source"] = dict(source)

    return Result(
        data=data,
        figure=overview_figure,
        figures={
            OVERVIEW_FIGURE_KEY: overview_figure,
            SNR_FIGURE_KEY: snr_result.get_figure(),
            DEPHASING_FIGURE_KEY: dephasing_result.get_figure(),
        },
    )


def _measurement_induced_dephasing(
    exp: Experiment,
    target: str,
    *,
    readout_amplitude: float,
    phase_range: ArrayLike,
    use_reference: bool,
    n_shots: int,
    shot_interval: float | None,
    plot: bool,
    save_image: bool,
) -> Result:
    """Run the single-target implementation of `measurement_induced_dephasing`."""
    phases = _normalize_float_array(phase_range, name="phase_range")
    measured_summary = _measure_ramsey_fringe(
        exp,
        target,
        readout_amplitude=float(readout_amplitude),
        phase_range=phases,
        n_shots=n_shots,
        shot_interval=shot_interval,
    )
    measured_rho01 = float(measured_summary["rho01"])
    if measured_rho01 <= ZERO_TOLERANCE:
        raise ValueError("Measured Ramsey coherence must be positive.")

    reference_summary: RamseyFringeSummary | None = None
    reference_rho01: float | None = None
    reference_phi0: float | None = None
    beta_m: float | None = None
    if use_reference:
        reference_summary = _measure_ramsey_fringe(
            exp,
            target,
            readout_amplitude=0.0,
            phase_range=phases,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        reference_rho01 = float(reference_summary["rho01"])
        reference_phi0 = float(reference_summary["phi0"])
        if reference_rho01 <= ZERO_TOLERANCE:
            raise ValueError("Reference Ramsey coherence must be positive.")
        beta_m = float(-np.log(np.clip(measured_rho01 / reference_rho01, 1e-12, None)))

    if plot:
        print(f"Measurement-induced dephasing : {target}")
        print(f"  readout_amplitude = {float(readout_amplitude):.6g}")
        print(f"  rho01 = {float(measured_summary['rho01']):.6g}")
        print(f"  phi0 = {float(measured_summary['phi0']):.6g}")
        if reference_summary is None or beta_m is None:
            print("  reference_amplitude = N/A")
            print("  reference_rho01 = N/A")
            print("  reference_phi0 = N/A")
            print("  beta_m = N/A")
        else:
            print("  reference_amplitude = 0")
            print(f"  reference_rho01 = {float(reference_summary['rho01']):.6g}")
            print(f"  reference_phi0 = {float(reference_summary['phi0']):.6g}")
            print(f"  beta_m = {beta_m:.6g}")

    phase_fine = np.linspace(float(np.min(phases)), float(np.max(phases)), 1000)
    tick_start = int(np.floor(np.min(phases) / np.pi))
    tick_stop = int(np.ceil(np.max(phases) / np.pi))
    tickvals = [index * np.pi for index in range(tick_start, tick_stop + 1)]
    ticktext: list[str] = []
    for index in range(tick_start, tick_stop + 1):
        if index == 0:
            ticktext.append("0")
        elif index == 1:
            ticktext.append("π")
        elif index == -1:
            ticktext.append("-π")
        else:
            ticktext.append(f"{index}π")

    def fitted_probability(summary: RamseyFringeSummary) -> NDArray[np.float64]:
        sigma_z = (
            float(summary["offset"])
            + float(summary["cosine"]) * np.cos(phase_fine)
            + float(summary["sine"]) * np.sin(phase_fine)
        )
        return 0.5 * (1.0 - sigma_z)

    figure = viz.make_figure()
    if reference_summary is not None:
        figure.add_trace(
            go.Scatter(
                x=phases,
                y=np.asarray(reference_summary["p1"], dtype=np.float64),
                mode="markers",
                name="reference data",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=phase_fine,
                y=fitted_probability(reference_summary),
                mode="lines",
                name="reference fit",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=phases,
            y=np.asarray(measured_summary["p1"], dtype=np.float64),
            mode="markers",
            name="data",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=phase_fine,
            y=fitted_probability(measured_summary),
            mode="lines",
            name="fit",
        )
    )
    figure.update_layout(
        title=_title_with_subtitle(
            f"Measurement-induced dephasing : {target}",
            f"readout_amplitude={float(readout_amplitude):.6g}",
        ),
        xaxis_title="Phase",
        yaxis_title="Excited-state probability",
        width=600,
        height=400,
        showlegend=True,
    )
    figure.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    if beta_m is not None:
        figure.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text=f"βm = {beta_m:.3g}",
            bgcolor="rgba(255, 255, 255, 0.8)",
            showarrow=False,
        )
    if plot:
        _show_figure(
            figure,
            f"measurement_induced_dephasing_{target}_{float(readout_amplitude):.4g}",
        )
    if save_image:
        viz.save_figure(
            figure,
            name=f"measurement_induced_dephasing_{target}_{float(readout_amplitude):.4g}",
        )

    return Result(
        data={
            "target": target,
            "readout_amplitude": float(readout_amplitude),
            "reference_amplitude": 0.0 if use_reference else None,
            "measurement_induced_dephasing": beta_m,
            "beta_m": beta_m,
            "rho01": measured_rho01,
            "reference_rho01": reference_rho01,
            "phi0": float(measured_summary["phi0"]),
            "reference_phi0": reference_phi0,
            "phase_range": phases,
            "phase_range_over_2pi": phases / (2.0 * np.pi),
            "ramsey": measured_summary,
            "reference_ramsey": reference_summary,
            "fig": figure,
        },
        figure=figure,
        figures={RAMSEY_FIGURE_KEY: figure},
    )


def _readout_snr(
    exp: Experiment,
    target: str,
    *,
    readout_amplitude: float,
    n_shots: int,
    shot_interval: float | None,
    readout_duration: float | None,
    plot: bool,
    save_image: bool,
) -> Result:
    """Run the single-target implementation of `readout_snr`."""
    if n_shots <= 0:
        raise ValueError("`n_shots` must be positive.")

    summary = _measure_readout_snr(
        exp,
        target,
        readout_amplitude=float(readout_amplitude),
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_duration=readout_duration,
    )
    if plot:
        print(f"Readout SNR : {target}")
        print(f"  readout_amplitude = {float(readout_amplitude):.6g}")
        print(f"  mu_ground = {float(summary['mu_ground']):.6g}")
        print(f"  mu_excited = {float(summary['mu_excited']):.6g}")
        print(f"  sigma_ground = {float(summary['sigma_ground']):.6g}")
        print(f"  sigma_excited = {float(summary['sigma_excited']):.6g}")
        print(f"  signal = {float(summary['signal']):.6g}")
        print(f"  noise = {float(summary['noise']):.6g}")
        print(f"  snr = {float(summary['snr']):.6g}")

    projection_figure = viz.make_figure()
    projection_figure.add_trace(
        go.Histogram(
            x=np.asarray(summary["projected_ground"], dtype=np.float64),
            name="|g⟩",
            opacity=0.65,
            nbinsx=40,
        )
    )
    projection_figure.add_trace(
        go.Histogram(
            x=np.asarray(summary["projected_excited"], dtype=np.float64),
            name="|e⟩",
            opacity=0.65,
            nbinsx=40,
        )
    )
    projection_figure.update_layout(
        title=_title_with_subtitle(
            f"Readout SNR : {target}",
            f"readout_amplitude={float(readout_amplitude):.6g}",
        ),
        xaxis_title="Projected response (arb. units)",
        yaxis_title="Count",
        barmode="overlay",
        width=600,
        height=400,
        showlegend=True,
    )
    projection_figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"SNR = {float(summary['snr']):.3g}",
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )

    weight_array = np.asarray(summary["weights"], dtype=np.complex128)
    weight_figure = viz.make_figure()
    if weight_array.ndim == 0 or weight_array.size == 1:
        weight_figure.add_trace(
            go.Bar(
                x=["Re(weight)", "Im(weight)"],
                y=[
                    float(np.real(weight_array).reshape(-1)[0]),
                    float(np.imag(weight_array).reshape(-1)[0]),
                ],
                name="Weight",
            )
        )
        xaxis_title = "Weight component"
    else:
        time_ns = SAMPLING_PERIOD * np.arange(
            weight_array.size,
            dtype=np.float64,
        )
        weight_figure.add_trace(
            go.Scatter(
                x=time_ns,
                y=np.real(weight_array).reshape(-1),
                mode="lines",
                name="Re(weight)",
            )
        )
        weight_figure.add_trace(
            go.Scatter(
                x=time_ns,
                y=np.imag(weight_array).reshape(-1),
                mode="lines",
                name="Im(weight)",
            )
        )
        xaxis_title = "Time (ns)"
    weight_figure.update_layout(
        title=_title_with_subtitle(
            f"Optimal weight function : {target}",
            f"readout_amplitude={float(readout_amplitude):.6g}",
        ),
        xaxis_title=xaxis_title,
        yaxis_title="Weight",
        width=600,
        height=300,
        showlegend=True,
    )
    if plot:
        _show_figure(
            projection_figure,
            f"readout_snr_{target}_{float(readout_amplitude):.4g}",
        )
        _show_figure(
            weight_figure,
            f"readout_weight_{target}_{float(readout_amplitude):.4g}",
        )
    if save_image:
        viz.save_figure(
            projection_figure,
            name=f"readout_snr_{target}_{float(readout_amplitude):.4g}",
        )
        viz.save_figure(
            weight_figure,
            name=f"readout_weight_{target}_{float(readout_amplitude):.4g}",
        )

    return Result(
        data={
            "target": target,
            "readout_amplitude": float(readout_amplitude),
            "snr": float(summary["snr"]),
            "signal": float(summary["signal"]),
            "noise": float(summary["noise"]),
            "weights": np.asarray(summary["weights"], dtype=np.complex128),
            "projection": summary,
            "fig": projection_figure,
        },
        figure=projection_figure,
        figures={
            PROJECTION_FIGURE_KEY: projection_figure,
            WEIGHT_FIGURE_KEY: weight_figure,
        },
    )


def _measurement_induced_dephasing_experiment(
    exp: Experiment,
    target: str,
    *,
    amplitude_range: ArrayLike,
    phase_range: ArrayLike,
    n_shots: int,
    shot_interval: float | None,
    plot: bool,
    save_image: bool,
) -> Result:
    """Run the single-target implementation of `measurement_induced_dephasing_experiment`."""
    amplitudes = _normalize_float_array(amplitude_range, name="amplitude_range")
    phases = _normalize_float_array(phase_range, name="phase_range")
    ramsey_summaries: dict[float, RamseyFringeSummary] = {}
    rho01_values = np.empty(len(amplitudes), dtype=np.float64)

    for index, amplitude in enumerate(amplitudes):
        summary = _measure_ramsey_fringe(
            exp,
            target,
            readout_amplitude=float(amplitude),
            phase_range=phases,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        ramsey_summaries[float(amplitude)] = summary
        rho01_values[index] = float(summary["rho01"])

    fit_result = _fit_measurement_induced_dephasing(
        target=target,
        readout_amplitudes=amplitudes,
        rho01_values=rho01_values,
    )
    if fit_result.status is not FitStatus.SUCCESS:
        raise RuntimeError(
            fit_result.message or "Measurement-induced dephasing fitting failed."
        )

    figure = fit_result.get_figure()
    if plot:
        _show_figure(figure, f"measurement_induced_dephasing_{target}")
    if save_image:
        viz.save_figure(figure, name=f"measurement_induced_dephasing_{target}")

    return Result(
        data={
            "target": target,
            "readout_amplitudes": amplitudes,
            "rho01": rho01_values,
            "beta_m_raw": np.asarray(fit_result["beta_m_raw"], dtype=np.float64),
            "ramsey": ramsey_summaries,
            "fit_result": fit_result,
            "fig": figure,
        },
        figure=figure,
        figures={DEPHASING_FIGURE_KEY: figure},
    )


def _sweep_readout_snr(
    exp: Experiment,
    target: str,
    *,
    amplitude_range: ArrayLike,
    n_shots: int,
    shot_interval: float | None,
    readout_duration: float | None,
    plot: bool,
    save_image: bool,
) -> Result:
    """Run the single-target implementation of `sweep_readout_snr`."""
    amplitudes = _normalize_float_array(amplitude_range, name="amplitude_range")
    if n_shots <= 0:
        raise ValueError("`n_shots` must be positive.")

    projection_summaries: dict[float, ReadoutSnrSummary] = {}
    snr_values = np.empty(len(amplitudes), dtype=np.float64)
    signal_values = np.empty(len(amplitudes), dtype=np.float64)
    noise_values = np.empty(len(amplitudes), dtype=np.float64)

    for index, amplitude in enumerate(amplitudes):
        summary = _measure_readout_snr(
            exp,
            target,
            readout_amplitude=float(amplitude),
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_duration=readout_duration,
        )
        projection_summaries[float(amplitude)] = summary
        snr_values[index] = float(summary["snr"])
        signal_values[index] = float(summary["signal"])
        noise_values[index] = float(summary["noise"])

    fit_result = _fit_snr_sweep(
        target=target,
        readout_amplitudes=amplitudes,
        snr_values=snr_values,
    )
    if fit_result.status is not FitStatus.SUCCESS:
        raise RuntimeError(fit_result.message or "Readout SNR fitting failed.")

    figure = fit_result.get_figure()
    if plot:
        _show_figure(figure, f"readout_snr_{target}")
    if save_image:
        viz.save_figure(figure, name=f"readout_snr_{target}")

    return Result(
        data={
            "target": target,
            "readout_amplitudes": amplitudes,
            "snr": snr_values,
            "signal": signal_values,
            "noise": noise_values,
            "projection": projection_summaries,
            "fit_result": fit_result,
            "fig": figure,
        },
        figure=figure,
        figures={SNR_FIGURE_KEY: figure},
    )


def _quantum_efficiency_measurement(
    exp: Experiment,
    target: str,
    *,
    amplitude_range: ArrayLike,
    phase_range: ArrayLike,
    n_shots: int,
    shot_interval: float | None,
    readout_duration: float | None,
    plot: bool,
    save_image: bool,
) -> Result:
    """Run the single-target implementation of `quantum_efficiency_measurement`."""
    amplitudes = _normalize_float_array(amplitude_range, name="amplitude_range")
    dephasing_result = _measurement_induced_dephasing_experiment(
        exp,
        target,
        amplitude_range=amplitudes,
        phase_range=phase_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
        save_image=False,
    )
    snr_result = _sweep_readout_snr(
        exp,
        target,
        amplitude_range=amplitudes,
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_duration=readout_duration,
        plot=False,
        save_image=False,
    )
    return _build_quantum_efficiency_result(
        target=target,
        readout_amplitudes=amplitudes,
        dephasing_result=dephasing_result,
        snr_result=snr_result,
        plot=plot,
        save_image=save_image,
    )


def readout_snr(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    readout_amplitude: float,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_duration: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure the single-point readout SNR for one or more targets sequentially.

    The returned result contains the projected single-shot histogram as the
    primary figure and the matched-filter weight as the named `"weight"` figure.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="readout_snr",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    plot, save_image = _resolve_plot_options(plot=plot, save_image=save_image)
    results = {
        target: _readout_snr(
            exp,
            target,
            readout_amplitude=readout_amplitude,
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_duration=readout_duration,
            plot=plot,
            save_image=save_image,
        )
        for target in _normalize_targets(exp, targets)
    }
    return _aggregate_target_results(results)


def sweep_readout_snr(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    amplitude_range: ArrayLike,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_duration: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure the readout SNR versus readout amplitude.

    The raw single-shot data are converted into one SNR value per amplitude,
    then fit with ``SNR(epsilon) = a epsilon``. The fitted slope ``a`` is shown
    in the figure because it is the parameter used in
    :func:`quantum_efficiency_measurement`.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="sweep_readout_snr",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    plot, save_image = _resolve_plot_options(plot=plot, save_image=save_image)
    results = {
        target: _sweep_readout_snr(
            exp,
            target,
            amplitude_range=amplitude_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_duration=readout_duration,
            plot=plot,
            save_image=save_image,
        )
        for target in _normalize_targets(exp, targets)
    }
    return _aggregate_target_results(results)


def measurement_induced_dephasing(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    readout_amplitude: float,
    phase_range: ArrayLike,
    use_reference: bool = False,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure the single-point measurement-induced dephasing for one or more targets.

    This function runs a weak-measurement Ramsey sequence at one
    ``readout_amplitude`` and extracts ``rho01`` and ``phi0`` from the fringe
    fit. When ``use_reference`` is true, a second Ramsey fringe is measured at
    zero readout amplitude and

    ``beta_m = -log(|rho01(readout_amplitude)| / |rho01(0)|)``

    is reported.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="measurement_induced_dephasing",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    plot, save_image = _resolve_plot_options(plot=plot, save_image=save_image)
    results = {
        target: _measurement_induced_dephasing(
            exp,
            target,
            readout_amplitude=readout_amplitude,
            phase_range=phase_range,
            use_reference=use_reference,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
            save_image=save_image,
        )
        for target in _normalize_targets(exp, targets)
    }
    return _aggregate_target_results(results)


def measurement_induced_dephasing_experiment(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    amplitude_range: ArrayLike,
    phase_range: ArrayLike,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure measurement-induced dephasing as a function of readout amplitude.

    For each amplitude, this function extracts ``|rho01|`` from a Ramsey fringe
    fit and then fits

    ``|rho01|(epsilon) = b exp(-epsilon^2 / (2 sigma_m^2))``.

    The fitted ``sigma_m`` is shown in the plot and is the parameter combined
    with the SNR slope in :func:`quantum_efficiency_measurement`.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="measurement_induced_dephasing_experiment",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    plot, save_image = _resolve_plot_options(plot=plot, save_image=save_image)
    results = {
        target: _measurement_induced_dephasing_experiment(
            exp,
            target,
            amplitude_range=amplitude_range,
            phase_range=phase_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
            save_image=save_image,
        )
        for target in _normalize_targets(exp, targets)
    }
    return _aggregate_target_results(results)


def quantum_efficiency_measurement(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    amplitude_range: ArrayLike,
    phase_range: ArrayLike,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_duration: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure the quantum efficiency for one or more targets sequentially.

    This wrapper runs

    - :func:`sweep_readout_snr`
    - :func:`measurement_induced_dephasing_experiment`

    and computes the final fit-based quantum efficiency as

    ``eta = a^2 sigma_m^2 / 2``.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="quantum_efficiency_measurement",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    plot, save_image = _resolve_plot_options(plot=plot, save_image=save_image)
    results = {
        target: _quantum_efficiency_measurement(
            exp,
            target,
            amplitude_range=amplitude_range,
            phase_range=phase_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_duration=readout_duration,
            plot=plot,
            save_image=save_image,
        )
        for target in _normalize_targets(exp, targets)
    }
    return _aggregate_target_results(results)


def analyze_quantum_efficiency(
    exp: object,
    target: str,
    *,
    readout_amplitudes: ArrayLike,
    ramsey_phases: ArrayLike,
    ramsey_excited_probabilities: ArrayLike,
    ground_state_raw: Sequence[ArrayLike] | ArrayLike,
    excited_state_raw: Sequence[ArrayLike] | ArrayLike,
    plot: bool = True,
) -> Result:
    """
    Analyze notebook-style quantum-efficiency data without running the hardware.

    Parameters
    ----------
    exp
        Unused placeholder kept for consistency with experiment-style APIs.
    target
        Target qubit label.
    readout_amplitudes
        Readout amplitudes used for both the Ramsey and SNR sweeps.
    ramsey_phases
        Phase sweep used for each weak-measurement Ramsey fringe.
    ramsey_excited_probabilities
        Excited-state probabilities with shape
        ``(len(readout_amplitudes), len(ramsey_phases))``.
    ground_state_raw, excited_state_raw
        Raw IQ samples for the single-shot SNR extraction at each amplitude.
    plot
        Whether to display the overview figure.
    """
    del exp
    amplitudes = _normalize_float_array(
        readout_amplitudes,
        name="readout_amplitudes",
    )
    phases = _normalize_float_array(ramsey_phases, name="ramsey_phases")
    probabilities = _normalize_float_array(
        ramsey_excited_probabilities,
        name="ramsey_excited_probabilities",
        ndim=2,
    )
    ground_raw_series = _normalize_raw_series(
        ground_state_raw,
        n_amplitudes=len(amplitudes),
        name="ground_state_raw",
    )
    excited_raw_series = _normalize_raw_series(
        excited_state_raw,
        n_amplitudes=len(amplitudes),
        name="excited_state_raw",
    )
    expected_shape = (len(amplitudes), len(phases))
    if probabilities.shape != expected_shape:
        raise ValueError(
            "`ramsey_excited_probabilities` must have shape "
            "(len(readout_amplitudes), len(ramsey_phases))."
        )
    if len(ground_raw_series) != len(amplitudes):
        raise ValueError("`ground_state_raw` must match the number of amplitudes.")
    if len(excited_raw_series) != len(amplitudes):
        raise ValueError("`excited_state_raw` must match the number of amplitudes.")
    for index, (ground_raw, excited_raw) in enumerate(
        zip(ground_raw_series, excited_raw_series, strict=True)
    ):
        if ground_raw.ndim != excited_raw.ndim:
            raise ValueError(
                f"`ground_state_raw[{index}]` and `excited_state_raw[{index}]` must have the same dimensionality."
            )
        if ground_raw.shape != excited_raw.shape:
            raise ValueError(
                f"`ground_state_raw[{index}]` and `excited_state_raw[{index}]` must have the same shape."
            )

    ramsey_summaries: dict[float, RamseyFringeSummary] = {}
    projection_summaries: dict[float, ReadoutSnrSummary] = {}
    for index, amplitude in enumerate(amplitudes):
        ramsey_summaries[float(amplitude)] = _fit_ramsey_fringe(
            phases,
            probabilities[index],
        )
        projection_summaries[float(amplitude)] = compute_readout_snr(
            ground_raw_series[index],
            excited_raw_series[index],
        )

    rho01_values = np.array(
        [
            float(ramsey_summaries[float(amplitude)]["rho01"])
            for amplitude in amplitudes
        ],
        dtype=np.float64,
    )
    snr_values = np.array(
        [
            float(projection_summaries[float(amplitude)]["snr"])
            for amplitude in amplitudes
        ],
        dtype=np.float64,
    )
    signal_values = np.array(
        [
            float(projection_summaries[float(amplitude)]["signal"])
            for amplitude in amplitudes
        ],
        dtype=np.float64,
    )
    noise_values = np.array(
        [
            float(projection_summaries[float(amplitude)]["noise"])
            for amplitude in amplitudes
        ],
        dtype=np.float64,
    )

    dephasing_fit = _fit_measurement_induced_dephasing(
        target=target,
        readout_amplitudes=amplitudes,
        rho01_values=rho01_values,
    )
    if dephasing_fit.status is not FitStatus.SUCCESS:
        raise RuntimeError(
            dephasing_fit.message or "Measurement-induced dephasing fitting failed."
        )

    snr_fit = _fit_snr_sweep(
        target=target,
        readout_amplitudes=amplitudes,
        snr_values=snr_values,
    )
    if snr_fit.status is not FitStatus.SUCCESS:
        raise RuntimeError(snr_fit.message or "Readout SNR fitting failed.")

    dephasing_result = Result(
        data={
            "target": target,
            "readout_amplitudes": amplitudes,
            "rho01": rho01_values,
            "beta_m_raw": np.asarray(dephasing_fit["beta_m_raw"], dtype=np.float64),
            "ramsey": ramsey_summaries,
            "fit_result": dephasing_fit,
            "fig": dephasing_fit.get_figure(),
        },
        figure=dephasing_fit.get_figure(),
        figures={DEPHASING_FIGURE_KEY: dephasing_fit.get_figure()},
    )
    snr_result = Result(
        data={
            "target": target,
            "readout_amplitudes": amplitudes,
            "snr": snr_values,
            "signal": signal_values,
            "noise": noise_values,
            "projection": projection_summaries,
            "fit_result": snr_fit,
            "fig": snr_fit.get_figure(),
        },
        figure=snr_fit.get_figure(),
        figures={SNR_FIGURE_KEY: snr_fit.get_figure()},
    )
    return _build_quantum_efficiency_result(
        target=target,
        readout_amplitudes=amplitudes,
        dephasing_result=dephasing_result,
        snr_result=snr_result,
        plot=plot,
        save_image=False,
        source={
            "ramsey_phases": phases,
            "ramsey": ramsey_summaries,
            "ground_state_raw": ground_raw_series,
            "excited_state_raw": excited_raw_series,
        },
    )
