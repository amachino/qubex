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


class GaussianHistogramComponentSummary(TypedDict):
    """Typed one-component Gaussian summary extracted from a histogram fit."""

    mu: float
    sigma: float
    weight: float


class DoubleGaussianHistogramFitSummary(TypedDict):
    """Typed double-Gaussian histogram fit payload."""

    status: str
    message: str
    bin_edges: NDArray[np.float64]
    bin_centers: NDArray[np.float64]
    counts: NDArray[np.float64]
    fit_counts: NDArray[np.float64]
    fit_axis: NDArray[np.float64]
    total_curve: NDArray[np.float64]
    main_curve: NDArray[np.float64]
    spurious_curve: NDArray[np.float64]
    main_component: GaussianHistogramComponentSummary
    spurious_component: GaussianHistogramComponentSummary


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
    ground_fit: DoubleGaussianHistogramFitSummary
    excited_fit: DoubleGaussianHistogramFitSummary


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


def _gaussian_pdf(
    x: ArrayLike,
    *,
    mu: float,
    sigma: float,
    zero_tolerance: float = ZERO_TOLERANCE,
) -> NDArray[np.float64]:
    """Return one normalized Gaussian PDF on the input axis."""
    axis = np.asarray(x, dtype=np.float64)
    sigma_safe = max(float(sigma), zero_tolerance)
    normalized = (axis - float(mu)) / sigma_safe
    return np.exp(-0.5 * normalized**2) / (np.sqrt(2.0 * np.pi) * sigma_safe)


def _resolve_histogram_bin_edges(
    values: ArrayLike,
    *,
    min_bins: int = 20,
    max_bins: int = 80,
) -> NDArray[np.float64]:
    """Return one finite, increasing histogram edge array."""
    samples = _normalize_float_array(values, name="histogram_values")
    lower = float(np.min(samples))
    upper = float(np.max(samples))
    if upper - lower < ZERO_TOLERANCE:
        half_span = max(abs(lower), 1.0)
        return np.linspace(
            lower - half_span,
            lower + half_span,
            min_bins + 1,
            dtype=np.float64,
        )

    edges = np.histogram_bin_edges(samples, bins="fd")
    n_bins = len(edges) - 1
    if n_bins < min_bins:
        edges = np.histogram_bin_edges(samples, bins=min_bins)
    elif n_bins > max_bins:
        edges = np.histogram_bin_edges(samples, bins=max_bins)
    edges = np.asarray(edges, dtype=np.float64)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        return np.linspace(lower, upper, min_bins + 1, dtype=np.float64)
    return edges


def _fallback_double_gaussian_histogram_fit(
    values: NDArray[np.float64],
    *,
    bin_edges: NDArray[np.float64],
    message: str,
) -> DoubleGaussianHistogramFitSummary:
    """Return one direct-statistics fallback when the double-Gaussian fit fails."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sigma = max(sigma, bin_width / np.sqrt(12.0), ZERO_TOLERANCE)
    fit_axis = np.linspace(float(edges[0]), float(edges[-1]), 500, dtype=np.float64)
    total_curve = len(values) * bin_width * _gaussian_pdf(fit_axis, mu=mu, sigma=sigma)
    fit_counts = len(values) * bin_width * _gaussian_pdf(centers, mu=mu, sigma=sigma)
    zero_curve = np.zeros_like(fit_axis)
    return {
        "status": "fallback",
        "message": message,
        "bin_edges": edges,
        "bin_centers": centers.astype(np.float64),
        "counts": counts.astype(np.float64),
        "fit_counts": fit_counts.astype(np.float64),
        "fit_axis": fit_axis,
        "total_curve": total_curve.astype(np.float64),
        "main_curve": total_curve.astype(np.float64),
        "spurious_curve": zero_curve.astype(np.float64),
        "main_component": {
            "mu": mu,
            "sigma": sigma,
            "weight": 1.0,
        },
        "spurious_component": {
            "mu": mu,
            "sigma": sigma,
            "weight": 0.0,
        },
    }


def _gaussian_logpdf(
    x: ArrayLike,
    *,
    mu: float,
    sigma: float,
    zero_tolerance: float = ZERO_TOLERANCE,
) -> NDArray[np.float64]:
    """Return one Gaussian log-PDF on the input axis."""
    axis = np.asarray(x, dtype=np.float64)
    sigma_safe = max(float(sigma), zero_tolerance)
    normalized = (axis - float(mu)) / sigma_safe
    return -0.5 * normalized**2 - np.log(np.sqrt(2.0 * np.pi) * sigma_safe)


def _gaussian_component(
    *,
    mu: float,
    sigma: float,
    weight: float,
) -> GaussianHistogramComponentSummary:
    """Return one typed Gaussian component payload."""
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "weight": float(weight),
    }


def _gaussian_component_curve(
    x: NDArray[np.float64],
    *,
    sample_count: int,
    bin_width: float,
    component: GaussianHistogramComponentSummary,
) -> NDArray[np.float64]:
    """Return one histogram-space Gaussian component curve."""
    return (
        sample_count
        * bin_width
        * component["weight"]
        * _gaussian_pdf(
            x,
            mu=component["mu"],
            sigma=component["sigma"],
        )
    )


def _resolve_double_gaussian_components(
    component_1: GaussianHistogramComponentSummary,
    component_2: GaussianHistogramComponentSummary,
) -> tuple[GaussianHistogramComponentSummary, GaussianHistogramComponentSummary]:
    """Return components ordered as main then spurious."""
    if component_1["weight"] >= component_2["weight"]:
        return component_1, component_2
    return component_2, component_1


def _independent_double_gaussian_candidate_parameters(
    *,
    samples: NDArray[np.float64],
    centers: NDArray[np.float64],
    counts: NDArray[np.float64],
    sigma_floor: float,
    sample_std: float,
) -> list[NDArray[np.float64]]:
    """Return candidate initial parameters for the independent histogram fit."""
    q25, q50, q75 = np.quantile(samples, [0.25, 0.5, 0.75])
    lower_half = samples[samples <= q50]
    upper_half = samples[samples > q50]
    mu_1_init = float(np.mean(lower_half)) if lower_half.size else float(q25)
    mu_2_init = float(np.mean(upper_half)) if upper_half.size else float(q75)
    if abs(mu_2_init - mu_1_init) < ZERO_TOLERANCE:
        mu_1_init = float(q25)
        mu_2_init = float(q75)

    sigma_1_init = max(
        float(np.std(lower_half, ddof=1)) if lower_half.size > 1 else sample_std / 2.0,
        sigma_floor,
    )
    sigma_2_init = max(
        float(np.std(upper_half, ddof=1)) if upper_half.size > 1 else sample_std / 2.0,
        sigma_floor,
    )
    dominant_center = float(centers[int(np.argmax(counts))])
    weight_init = (
        0.8
        if abs(mu_1_init - dominant_center) <= abs(mu_2_init - dominant_center)
        else 0.2
    )
    return [
        np.array(
            [weight_init, mu_1_init, sigma_1_init, mu_2_init, sigma_2_init],
            dtype=np.float64,
        ),
        np.array(
            [1.0 - weight_init, mu_1_init, sigma_1_init, mu_2_init, sigma_2_init],
            dtype=np.float64,
        ),
        np.array(
            [1.0 - weight_init, mu_2_init, sigma_2_init, mu_1_init, sigma_1_init],
            dtype=np.float64,
        ),
    ]


def _fit_independent_double_gaussian_model(
    x: NDArray[np.float64],
    *,
    sample_count: int,
    bin_width: float,
    weight: float,
    mu_1: float,
    sigma_1: float,
    mu_2: float,
    sigma_2: float,
) -> NDArray[np.float64]:
    """Return the histogram-space two-Gaussian model curve."""
    component_1 = _gaussian_component(mu=mu_1, sigma=sigma_1, weight=weight)
    component_2 = _gaussian_component(
        mu=mu_2,
        sigma=sigma_2,
        weight=1.0 - weight,
    )
    return _gaussian_component_curve(
        x,
        sample_count=sample_count,
        bin_width=bin_width,
        component=component_1,
    ) + _gaussian_component_curve(
        x,
        sample_count=sample_count,
        bin_width=bin_width,
        component=component_2,
    )


def _independent_double_gaussian_bounds(
    *,
    samples: NDArray[np.float64],
    sigma_floor: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return lower/upper parameter bounds for the independent histogram fit."""
    span = max(float(np.max(samples) - np.min(samples)), sigma_floor)
    lower_bounds = np.array(
        [
            1e-6,
            float(np.min(samples)) - 2.0 * span,
            sigma_floor,
            float(np.min(samples)) - 2.0 * span,
            sigma_floor,
        ],
        dtype=np.float64,
    )
    upper_bounds = np.array(
        [
            1.0 - 1e-6,
            float(np.max(samples)) + 2.0 * span,
            max(10.0 * span, 10.0 * sigma_floor),
            float(np.max(samples)) + 2.0 * span,
            max(10.0 * span, 10.0 * sigma_floor),
        ],
        dtype=np.float64,
    )
    return lower_bounds, upper_bounds


def _build_double_gaussian_histogram_fit_summary(
    values: NDArray[np.float64],
    *,
    bin_edges: NDArray[np.float64],
    status: str,
    message: str,
    main_component: GaussianHistogramComponentSummary,
    spurious_component: GaussianHistogramComponentSummary,
) -> DoubleGaussianHistogramFitSummary:
    """Build one histogram-fit payload from explicit Gaussian components."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    fit_axis = np.linspace(float(edges[0]), float(edges[-1]), 500, dtype=np.float64)
    sample_count = len(values)
    main_curve = _gaussian_component_curve(
        fit_axis,
        sample_count=sample_count,
        bin_width=bin_width,
        component=main_component,
    )
    spurious_curve = _gaussian_component_curve(
        fit_axis,
        sample_count=sample_count,
        bin_width=bin_width,
        component=spurious_component,
    )
    fit_counts = _gaussian_component_curve(
        centers,
        sample_count=sample_count,
        bin_width=bin_width,
        component=main_component,
    ) + _gaussian_component_curve(
        centers,
        sample_count=sample_count,
        bin_width=bin_width,
        component=spurious_component,
    )
    return {
        "status": status,
        "message": message,
        "bin_edges": edges.astype(np.float64),
        "bin_centers": centers.astype(np.float64),
        "counts": counts.astype(np.float64),
        "fit_counts": fit_counts.astype(np.float64),
        "fit_axis": fit_axis,
        "total_curve": (main_curve + spurious_curve).astype(np.float64),
        "main_curve": main_curve.astype(np.float64),
        "spurious_curve": spurious_curve.astype(np.float64),
        "main_component": main_component,
        "spurious_component": spurious_component,
    }


def _fit_double_gaussian_histogram(
    values: ArrayLike,
    *,
    name: str,
    bin_edges: NDArray[np.float64] | None = None,
) -> DoubleGaussianHistogramFitSummary:
    """
    Fit one histogram independently with the sum of two Gaussian functions.

    This helper is kept as a fallback for cases where the coupled ground/excited
    constrained fit is unavailable or fails to converge.
    """
    samples = _normalize_float_array(values, name=name)
    if bin_edges is None:
        edges = _resolve_histogram_bin_edges(samples)
    else:
        edges = _normalize_float_array(bin_edges, name=f"{name}_bin_edges")
        if len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"`{name}_bin_edges` must be a strictly increasing array.")

    counts, _ = np.histogram(samples, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    if len(samples) < 4 or len(centers) < 2:
        return _fallback_double_gaussian_histogram_fit(
            samples,
            bin_edges=edges,
            message="Insufficient data for double-Gaussian histogram fitting.",
        )

    sample_std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    if sample_std < ZERO_TOLERANCE:
        return _fallback_double_gaussian_histogram_fit(
            samples,
            bin_edges=edges,
            message="Degenerate projected data prevented double-Gaussian fitting.",
        )

    sigma_floor = max(bin_width / np.sqrt(12.0), sample_std / 20.0, 1e-6)
    lower_bounds, upper_bounds = _independent_double_gaussian_bounds(
        samples=samples,
        sigma_floor=sigma_floor,
    )
    candidate_parameters = _independent_double_gaussian_candidate_parameters(
        samples=samples,
        centers=centers,
        counts=counts.astype(np.float64),
        sigma_floor=sigma_floor,
        sample_std=sample_std,
    )

    from scipy.optimize import curve_fit  # lazy import

    sigma_counts = np.sqrt(np.maximum(counts.astype(np.float64), 1.0))
    sample_count = len(samples)

    def model(
        x: NDArray[np.float64],
        weight: float,
        mu_1: float,
        sigma_1: float,
        mu_2: float,
        sigma_2: float,
    ) -> NDArray[np.float64]:
        return _fit_independent_double_gaussian_model(
            x,
            sample_count=sample_count,
            bin_width=bin_width,
            weight=weight,
            mu_1=mu_1,
            sigma_1=sigma_1,
            mu_2=mu_2,
            sigma_2=sigma_2,
        )

    best_parameters: NDArray[np.float64] | None = None
    best_cost = np.inf
    last_error: Exception | None = None
    for initial in candidate_parameters:
        try:
            popt, _ = curve_fit(
                model,
                centers,
                counts.astype(np.float64),
                p0=np.clip(initial, lower_bounds, upper_bounds),
                bounds=(lower_bounds, upper_bounds),
                sigma=sigma_counts,
                absolute_sigma=True,
                maxfev=50_000,
            )
        except Exception as exc:
            last_error = exc
            continue

        residual = (
            counts.astype(np.float64)
            - _fit_independent_double_gaussian_model(
                centers,
                sample_count=sample_count,
                bin_width=bin_width,
                weight=float(popt[0]),
                mu_1=float(popt[1]),
                sigma_1=float(popt[2]),
                mu_2=float(popt[3]),
                sigma_2=float(popt[4]),
            )
        ) / sigma_counts
        cost = float(np.sum(residual**2))
        if cost < best_cost:
            best_cost = cost
            best_parameters = np.asarray(popt, dtype=np.float64)

    if best_parameters is None:
        detail = f": {last_error}" if last_error is not None else "."
        return _fallback_double_gaussian_histogram_fit(
            samples,
            bin_edges=edges,
            message=f"Double-Gaussian histogram fitting failed for {name}{detail}",
        )

    weight_1 = float(best_parameters[0])
    component_1 = _gaussian_component(
        mu=float(best_parameters[1]),
        sigma=float(best_parameters[2]),
        weight=weight_1,
    )
    component_2 = _gaussian_component(
        mu=float(best_parameters[3]),
        sigma=float(best_parameters[4]),
        weight=1.0 - weight_1,
    )
    main_component, spurious_component = _resolve_double_gaussian_components(
        component_1,
        component_2,
    )
    return _build_double_gaussian_histogram_fit_summary(
        samples,
        bin_edges=edges,
        status="success",
        message=f"Double-Gaussian histogram fitting successful for {name}.",
        main_component=main_component,
        spurious_component=spurious_component,
    )


def _independent_double_gaussian_fit_pair(
    ground_samples: NDArray[np.float64],
    excited_samples: NDArray[np.float64],
    *,
    bin_edges: NDArray[np.float64],
) -> tuple[DoubleGaussianHistogramFitSummary, DoubleGaussianHistogramFitSummary]:
    """Return independent histogram fits for the projected ground/excited data."""
    return (
        _fit_double_gaussian_histogram(
            ground_samples,
            name="projected_ground",
            bin_edges=bin_edges,
        ),
        _fit_double_gaussian_histogram(
            excited_samples,
            name="projected_excited",
            bin_edges=bin_edges,
        ),
    )


def _coupled_double_gaussian_negative_log_likelihood(
    params: NDArray[np.float64],
    *,
    ground_samples: NDArray[np.float64],
    excited_samples: NDArray[np.float64],
) -> float:
    """Return the coupled constrained double-Gaussian negative log-likelihood."""
    center = float(params[0])
    gap = float(params[1])
    sigma_ground = float(params[2])
    sigma_excited = float(params[3])
    p_ground_to_excited = float(params[4])
    p_excited_to_ground = float(params[5])
    mu_ground = center - 0.5 * gap
    mu_excited = center + 0.5 * gap

    log_ground = np.logaddexp(
        np.log1p(-p_ground_to_excited)
        + _gaussian_logpdf(
            ground_samples,
            mu=mu_ground,
            sigma=sigma_ground,
        ),
        np.log(p_ground_to_excited)
        + _gaussian_logpdf(
            ground_samples,
            mu=mu_excited,
            sigma=sigma_excited,
        ),
    )
    log_excited = np.logaddexp(
        np.log(p_excited_to_ground)
        + _gaussian_logpdf(
            excited_samples,
            mu=mu_ground,
            sigma=sigma_ground,
        ),
        np.log1p(-p_excited_to_ground)
        + _gaussian_logpdf(
            excited_samples,
            mu=mu_excited,
            sigma=sigma_excited,
        ),
    )
    return float(-(np.sum(log_ground) + np.sum(log_excited)))


def _coupled_double_gaussian_initial_parameters(
    *,
    ground_samples: NDArray[np.float64],
    excited_samples: NDArray[np.float64],
    sigma_floor: float,
    max_gap: float,
    max_sigma: float,
) -> list[NDArray[np.float64]]:
    """Return candidate initial parameters for the coupled constrained fit."""
    ground_mean = float(np.mean(ground_samples))
    excited_mean = float(np.mean(excited_samples))
    center_init = 0.5 * (ground_mean + excited_mean)
    gap_init = max(abs(excited_mean - ground_mean), sigma_floor)
    midpoint_init = center_init
    ground_std = max(
        float(np.std(ground_samples, ddof=1)) if len(ground_samples) > 1 else 0.0,
        sigma_floor,
    )
    excited_std = max(
        float(np.std(excited_samples, ddof=1)) if len(excited_samples) > 1 else 0.0,
        sigma_floor,
    )
    weight_ground_init = float(np.mean(ground_samples > midpoint_init))
    weight_excited_init = float(np.mean(excited_samples < midpoint_init))
    weight_ground_init = float(np.clip(weight_ground_init, 1e-6, 0.25))
    weight_excited_init = float(np.clip(weight_excited_init, 1e-6, 0.25))
    return [
        np.array(
            [
                center_init,
                gap_init,
                ground_std,
                excited_std,
                weight_ground_init,
                weight_excited_init,
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                center_init,
                max(gap_init * 0.75, sigma_floor),
                ground_std,
                excited_std,
                max(weight_ground_init / 2.0, 1e-6),
                max(weight_excited_init / 2.0, 1e-6),
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                center_init,
                min(gap_init * 1.25, max_gap),
                min(max(ground_std * 0.8, sigma_floor), max_sigma),
                min(max(excited_std * 0.8, sigma_floor), max_sigma),
                min(max(weight_ground_init * 2.0, 1e-6), 0.1),
                min(max(weight_excited_init * 2.0, 1e-6), 0.1),
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                center_init,
                gap_init,
                ground_std,
                excited_std,
                0.02,
                0.02,
            ],
            dtype=np.float64,
        ),
    ]


def _coupled_double_gaussian_fit_message(*, success: bool, detail: str = "") -> str:
    """Return one user-facing message for the coupled constrained fit."""
    if success:
        return (
            "Coupled constrained double-Gaussian histogram fitting successful for "
            "projected ground/excited data."
        )
    return (
        "Coupled constrained double-Gaussian fitting failed for projected "
        f"readout data{detail} Falling back to independent fits."
    )


def _fit_coupled_double_gaussian_histograms(
    ground_values: ArrayLike,
    excited_values: ArrayLike,
    *,
    bin_edges: NDArray[np.float64] | None = None,
) -> tuple[DoubleGaussianHistogramFitSummary, DoubleGaussianHistogramFitSummary]:
    """
    Fit ground/excited projected histograms with one coupled constrained model.

    The model follows the Fig. 1(c) interpretation in arXiv:1711.05336:

    - prepared ``|g>`` data are a mixture of the main ``|g>`` Gaussian and a
      spurious ``|e>`` Gaussian
    - prepared ``|e>`` data are a mixture of the main ``|e>`` Gaussian and a
      spurious ``|g>`` Gaussian

    This ties the spurious component of one prepared state to the main Gaussian
    of the opposite prepared state, instead of fitting both histograms
    independently.
    """
    ground_samples = _normalize_float_array(ground_values, name="projected_ground")
    excited_samples = _normalize_float_array(excited_values, name="projected_excited")
    combined_samples = np.concatenate([ground_samples, excited_samples])
    if bin_edges is None:
        edges = _resolve_histogram_bin_edges(combined_samples)
    else:
        edges = _normalize_float_array(bin_edges, name="projected_bin_edges")
        if len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(
                "`projected_bin_edges` must be a strictly increasing array."
            )

    if len(ground_samples) < 4 or len(excited_samples) < 4:
        return _independent_double_gaussian_fit_pair(
            ground_samples,
            excited_samples,
            bin_edges=edges,
        )

    combined_std = (
        float(np.std(combined_samples, ddof=1)) if len(combined_samples) > 1 else 0.0
    )
    if combined_std < ZERO_TOLERANCE:
        return (
            _fallback_double_gaussian_histogram_fit(
                ground_samples,
                bin_edges=edges,
                message="Degenerate projected ground data prevented constrained fitting.",
            ),
            _fallback_double_gaussian_histogram_fit(
                excited_samples,
                bin_edges=edges,
                message="Degenerate projected excited data prevented constrained fitting.",
            ),
        )

    bin_width = float(edges[1] - edges[0])
    sigma_floor = max(bin_width / np.sqrt(12.0), combined_std / 100.0, 1e-6)
    span = max(float(np.max(combined_samples) - np.min(combined_samples)), sigma_floor)
    max_sigma = max(10.0 * span, 10.0 * sigma_floor)
    max_gap = max(4.0 * span, 2.0 * sigma_floor)
    weight_upper = 0.499999

    from scipy.optimize import minimize  # lazy import

    bounds = [
        (
            float(np.min(combined_samples)) - 2.0 * span,
            float(np.max(combined_samples)) + 2.0 * span,
        ),
        (sigma_floor, max_gap),
        (sigma_floor, max_sigma),
        (sigma_floor, max_sigma),
        (1e-6, weight_upper),
        (1e-6, weight_upper),
    ]
    initial_parameters = _coupled_double_gaussian_initial_parameters(
        ground_samples=ground_samples,
        excited_samples=excited_samples,
        sigma_floor=sigma_floor,
        max_gap=max_gap,
        max_sigma=max_sigma,
    )

    def objective(params: NDArray[np.float64]) -> float:
        return _coupled_double_gaussian_negative_log_likelihood(
            params,
            ground_samples=ground_samples,
            excited_samples=excited_samples,
        )

    best_result = None
    last_error: Exception | None = None
    for initial in initial_parameters:
        try:
            result = minimize(
                objective,
                np.clip(
                    initial,
                    [bound[0] for bound in bounds],
                    [bound[1] for bound in bounds],
                ),
                method="L-BFGS-B",
                bounds=bounds,
            )
        except Exception as exc:
            last_error = exc
            continue

        if result.success:
            if best_result is None or float(result.fun) < float(best_result.fun):
                best_result = result

    if best_result is None:
        detail = f": {last_error}" if last_error is not None else "."
        fallback_message = _coupled_double_gaussian_fit_message(
            success=False,
            detail=detail,
        )
        ground_fit, excited_fit = _independent_double_gaussian_fit_pair(
            ground_samples,
            excited_samples,
            bin_edges=edges,
        )
        return (
            {
                **ground_fit,
                "status": "fallback",
                "message": fallback_message,
            },
            {
                **excited_fit,
                "status": "fallback",
                "message": fallback_message,
            },
        )

    (
        center_fit,
        gap_fit,
        sigma_ground_fit,
        sigma_excited_fit,
        p_ground_to_excited_fit,
        p_excited_to_ground_fit,
    ) = np.asarray(best_result.x, dtype=np.float64)
    mu_ground_fit = float(center_fit - 0.5 * gap_fit)
    mu_excited_fit = float(center_fit + 0.5 * gap_fit)
    sigma_ground_value = float(sigma_ground_fit)
    sigma_excited_value = float(sigma_excited_fit)
    p_ground_to_excited = float(p_ground_to_excited_fit)
    p_excited_to_ground = float(p_excited_to_ground_fit)
    success_message = _coupled_double_gaussian_fit_message(success=True)

    ground_fit = _build_double_gaussian_histogram_fit_summary(
        ground_samples,
        bin_edges=edges,
        status="success",
        message=success_message,
        main_component=_gaussian_component(
            mu=mu_ground_fit,
            sigma=sigma_ground_value,
            weight=1.0 - p_ground_to_excited,
        ),
        spurious_component=_gaussian_component(
            mu=mu_excited_fit,
            sigma=sigma_excited_value,
            weight=p_ground_to_excited,
        ),
    )
    excited_fit = _build_double_gaussian_histogram_fit_summary(
        excited_samples,
        bin_edges=edges,
        status="success",
        message=success_message,
        main_component=_gaussian_component(
            mu=mu_excited_fit,
            sigma=sigma_excited_value,
            weight=1.0 - p_excited_to_ground,
        ),
        spurious_component=_gaussian_component(
            mu=mu_ground_fit,
            sigma=sigma_ground_value,
            weight=p_excited_to_ground,
        ),
    )
    return ground_fit, excited_fit


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
    the mean-state separation. The projected ``|g>`` and ``|e>`` histograms are
    then fit jointly with one constrained double-Gaussian model, following the
    Fig. 1(c) interpretation in arXiv:1711.05336: the spurious component in one
    prepared state is tied to the main Gaussian of the opposite prepared state.
    The final scalar SNR is

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

    common_bin_edges = _resolve_histogram_bin_edges(
        np.concatenate([projected_ground, projected_excited]),
    )
    ground_fit, excited_fit = _fit_coupled_double_gaussian_histograms(
        projected_ground,
        projected_excited,
        bin_edges=common_bin_edges,
    )

    mu_ground = float(ground_fit["main_component"]["mu"])
    mu_excited = float(excited_fit["main_component"]["mu"])
    sigma_ground = float(ground_fit["main_component"]["sigma"])
    sigma_excited = float(excited_fit["main_component"]["sigma"])

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
        "ground_fit": ground_fit,
        "excited_fit": excited_fit,
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

    ground_fit = summary["ground_fit"]
    excited_fit = summary["excited_fit"]
    common_edges = np.asarray(ground_fit["bin_edges"], dtype=np.float64)
    xbins = dict(
        start=float(common_edges[0]),
        end=float(common_edges[-1]),
        size=float(common_edges[1] - common_edges[0]),
    )
    ground_color = "#1f77b4"
    excited_color = "#d62728"

    projection_figure = viz.make_figure()
    projection_figure.add_trace(
        go.Histogram(
            x=np.asarray(summary["projected_ground"], dtype=np.float64),
            name="|g⟩",
            opacity=0.55,
            marker_color=ground_color,
            xbins=xbins,
        )
    )
    projection_figure.add_trace(
        go.Histogram(
            x=np.asarray(summary["projected_excited"], dtype=np.float64),
            name="|e⟩",
            opacity=0.55,
            marker_color=excited_color,
            xbins=xbins,
        )
    )
    projection_figure.add_trace(
        go.Scatter(
            x=np.asarray(ground_fit["fit_axis"], dtype=np.float64),
            y=np.asarray(ground_fit["total_curve"], dtype=np.float64),
            mode="lines",
            name="|g⟩ fit",
            line={"color": ground_color, "width": 2},
        )
    )
    projection_figure.add_trace(
        go.Scatter(
            x=np.asarray(ground_fit["fit_axis"], dtype=np.float64),
            y=np.asarray(ground_fit["main_curve"], dtype=np.float64),
            mode="lines",
            name="|g⟩ main",
            line={"color": ground_color, "dash": "dash", "width": 1.5},
        )
    )
    projection_figure.add_trace(
        go.Scatter(
            x=np.asarray(ground_fit["fit_axis"], dtype=np.float64),
            y=np.asarray(ground_fit["spurious_curve"], dtype=np.float64),
            mode="lines",
            name="|g⟩ spurious",
            line={"color": ground_color, "dash": "dot", "width": 1.5},
        )
    )
    projection_figure.add_trace(
        go.Scatter(
            x=np.asarray(excited_fit["fit_axis"], dtype=np.float64),
            y=np.asarray(excited_fit["total_curve"], dtype=np.float64),
            mode="lines",
            name="|e⟩ fit",
            line={"color": excited_color, "width": 2},
        )
    )
    projection_figure.add_trace(
        go.Scatter(
            x=np.asarray(excited_fit["fit_axis"], dtype=np.float64),
            y=np.asarray(excited_fit["main_curve"], dtype=np.float64),
            mode="lines",
            name="|e⟩ main",
            line={"color": excited_color, "dash": "dash", "width": 1.5},
        )
    )
    projection_figure.add_trace(
        go.Scatter(
            x=np.asarray(excited_fit["fit_axis"], dtype=np.float64),
            y=np.asarray(excited_fit["spurious_curve"], dtype=np.float64),
            mode="lines",
            name="|e⟩ spurious",
            line={"color": excited_color, "dash": "dot", "width": 1.5},
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
                f"`ground_state_raw[{index}]` and "
                f"`excited_state_raw[{index}]` must have the same dimensionality."
            )
        if ground_raw.shape != excited_raw.shape:
            raise ValueError(
                f"`ground_state_raw[{index}]` and "
                f"`excited_state_raw[{index}]` must have the same shape."
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
