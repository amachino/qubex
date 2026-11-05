"""
Characterize continuous-time transitions in the GEF manifold.

The workflow prepares G, E, and F, measures their population trajectories, fits
a shared six-rate continuous-time Markov generator, and derives relaxation
times relevant to the EF transition.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.linalg import expm

from qubex.contrib.experiment.gef_population_estimation import (
    GefPopulationCalibration,
    measure_gef_populations,
)
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.pulse import Blank, PulseSchedule

from .readout_mist import (
    PROTOCOL_ORDER,
    STATE_ORDER,
    TRANSITION_MATRIX_INDICES,
    TRANSITION_ORDER,
    FitGoodnessOfFit,
    GefPopulationFitDiagnostic,
    GefPopulationSeries,
    ResidualBlock,
    covariance_whitener,
    fit_full_gls,
    whiten_predictions,
)

DEFAULT_TIME_RANGE_NS = tuple(
    float(value)
    for value in np.concatenate(
        ([0.0], np.logspace(np.log10(100.0), np.log10(200_000.0), 31))
    )
)
DEFAULT_GEF_MEASUREMENT_N_SHOTS = 4096
DEFAULT_GEF_CALIBRATION_N_SHOTS = 8192
_RATE_COUNT = len(TRANSITION_ORDER)
_PARAMETER_COUNT = _RATE_COUNT + 2 * len(PROTOCOL_ORDER)
_INITIAL_ANCHORS = {"initial_g": 0, "initial_e": 1, "initial_f": 2}
_ESTIMABILITY_RELATIVE_TOLERANCE = 1e-7
_PHYSICAL_PARAMETER_ORDER = (
    *TRANSITION_ORDER,
    *(f"{protocol}_{state}" for protocol in PROTOCOL_ORDER for state in STATE_ORDER),
)
_DERIVED_TIME_RATES = {
    "t1_f_to_e": ("f_to_e",),
    "f_state_exit_lifetime": ("f_to_e", "f_to_g"),
    "ef_pair_relaxation_time": ("f_to_e", "e_to_f"),
}
AcquisitionOrder = Literal["interleaved", "randomized", "grouped"]


@dataclass(frozen=True)
class GefTransitionRateEstimate:
    """Store one transition-rate estimate in inverse microseconds."""

    value_per_us: float
    standard_error_per_us: float | None
    estimable: bool
    unit: str = "1/us"


@dataclass(frozen=True)
class GefRelaxationTimeEstimate:
    """Store one derived relaxation-time estimate in nanoseconds."""

    value: float
    standard_error: float | None
    estimable: bool
    unit: str = "ns"


@dataclass(frozen=True)
class GefRelaxationMeasurements:
    """Store G/E/F population trajectories on one continuous time grid."""

    time_ns: NDArray[np.float64]
    protocols: dict[str, GefPopulationSeries]

    def __post_init__(self) -> None:
        """Validate and normalize the shared time grid and trajectories."""
        time_ns = np.asarray(self.time_ns, dtype=np.float64)
        if time_ns.ndim != 1 or not np.all(np.isfinite(time_ns)):
            raise ValueError("time_ns must be a finite one-dimensional array.")
        if (
            time_ns.size < 5
            or time_ns[0] != 0.0
            or np.any(time_ns < 0.0)
            or np.any(np.diff(time_ns) <= 0.0)
        ):
            raise ValueError(
                "time_ns must contain at least five strictly increasing points "
                "beginning with zero."
            )
        if set(self.protocols) != set(PROTOCOL_ORDER):
            raise ValueError(
                "protocols must contain exactly initial_g, initial_e, initial_f."
            )
        for protocol, series in self.protocols.items():
            if series.population.shape[0] != time_ns.size:
                raise ValueError(f"{protocol} data must match time_ns.")
        object.__setattr__(self, "time_ns", time_ns)


@dataclass(frozen=True)
class GefRelaxationFit:
    """Store one six-rate continuous-time GEF fit and its diagnostics."""

    success: bool
    generator_matrix_per_us: NDArray[np.float64]
    transition_rates: dict[str, GefTransitionRateEstimate]
    derived_times_ns: dict[str, GefRelaxationTimeEstimate]
    initial_populations: dict[str, NDArray[np.float64]]
    fitted_populations: dict[str, NDArray[np.float64]]
    parameter_order: tuple[str, ...]
    parameter_covariance: NDArray[np.float64]
    parameter_estimable: dict[str, bool]
    goodness_of_fit: FitGoodnessOfFit
    whitened_residual: NDArray[np.float64]
    mahalanobis_residuals: dict[str, NDArray[np.float64]]
    message: str
    objective: float
    jacobian_rank: int
    jacobian_nullity: int
    jacobian_singular_values: NDArray[np.float64]
    state_order: tuple[str, str, str] = STATE_ORDER
    rate_unit: str = "1/us"
    time_unit: str = "ns"


def _simplex_from_logits(
    logits: NDArray[np.float64], *, anchor: int
) -> NDArray[np.float64]:
    destinations = tuple(index for index in range(3) if index != anchor)
    weights = np.ones(3, dtype=np.float64)
    weights[list(destinations)] = np.exp(logits)
    return weights / np.sum(weights)


def _simplex_logits(population: ArrayLike, *, anchor: int) -> NDArray[np.float64]:
    values = np.clip(np.asarray(population, dtype=np.float64), 1e-10, None)
    values /= np.sum(values)
    destinations = tuple(index for index in range(3) if index != anchor)
    return np.log(values[list(destinations)] / values[anchor])


def _generator_from_rates(rates: NDArray[np.float64]) -> NDArray[np.float64]:
    generator = np.zeros((3, 3), dtype=np.float64)
    for rate, (_, final_state, initial_state) in zip(
        rates, TRANSITION_MATRIX_INDICES, strict=True
    ):
        generator[final_state, initial_state] = rate
    for initial_state in range(3):
        generator[initial_state, initial_state] = -np.sum(generator[:, initial_state])
    return generator


def _decode_parameters(
    parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    if parameters.shape != (_PARAMETER_COUNT,):
        raise ValueError(
            f"The GEF relaxation model requires {_PARAMETER_COUNT} parameters."
        )
    rates = np.exp(parameters[:_RATE_COUNT])
    initial = {
        protocol: _simplex_from_logits(
            parameters[_RATE_COUNT + 2 * index : _RATE_COUNT + 2 * index + 2],
            anchor=_INITIAL_ANCHORS[protocol],
        )
        for index, protocol in enumerate(PROTOCOL_ORDER)
    }
    return _generator_from_rates(rates), initial


def _encode_parameters(
    rates: ArrayLike,
    initial: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    rate_values = np.clip(np.asarray(rates, dtype=np.float64), 1e-12, None)
    return np.concatenate(
        (
            np.log(rate_values),
            *(
                _simplex_logits(initial[protocol], anchor=_INITIAL_ANCHORS[protocol])
                for protocol in PROTOCOL_ORDER
            ),
        )
    )


def _predictions(
    parameters: NDArray[np.float64], time_ns: NDArray[np.float64]
) -> dict[str, NDArray[np.float64]]:
    generator, initial = _decode_parameters(parameters)
    propagators = tuple(expm(generator * (time * 1e-3)) for time in time_ns)
    return {
        protocol: np.stack(
            [propagator @ initial[protocol] for propagator in propagators]
        )
        for protocol in PROTOCOL_ORDER
    }


def _physical_vector(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    generator, initial = _decode_parameters(parameters)
    rates = np.asarray(
        [
            generator[final_state, initial_state]
            for _, final_state, initial_state in TRANSITION_MATRIX_INDICES
        ],
        dtype=np.float64,
    )
    return np.concatenate((rates, *(initial[name] for name in PROTOCOL_ORDER)))


def _numerical_jacobian(
    function: Callable[[NDArray[np.float64]], ArrayLike],
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    baseline = np.atleast_1d(np.asarray(function(parameters), dtype=np.float64))
    jacobian = np.empty((baseline.size, parameters.size), dtype=np.float64)
    for index in range(parameters.size):
        step = 1e-5 * max(1.0, abs(float(parameters[index])))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        jacobian[:, index] = (
            np.atleast_1d(function(upper)) - np.atleast_1d(function(lower))
        ) / (2.0 * step)
    return jacobian


def _gradient_is_estimable(
    gradient: NDArray[np.float64], null_space: NDArray[np.float64]
) -> bool:
    norm = float(np.linalg.norm(gradient))
    if norm == 0.0:
        return False
    if null_space.size == 0:
        return True
    return float(np.linalg.norm(null_space @ gradient)) <= (
        _ESTIMABILITY_RELATIVE_TOLERANCE * norm
    )


def _initializations(
    measurements: GefRelaxationMeasurements,
) -> tuple[NDArray[np.float64], ...]:
    zero_index = 0
    first_positive = 1
    initial = {
        protocol: measurements.protocols[protocol].population[zero_index]
        for protocol in PROTOCOL_ORDER
    }
    p0 = np.column_stack([initial[protocol] for protocol in PROTOCOL_ORDER])
    p1 = np.column_stack(
        [
            measurements.protocols[protocol].population[first_positive]
            for protocol in PROTOCOL_ORDER
        ]
    )
    delta_time_us = measurements.time_ns[first_positive] * 1e-3
    approximate_generator = ((p1 - p0) @ np.linalg.pinv(p0)) / delta_time_us
    estimated_rates = np.asarray(
        [
            max(float(approximate_generator[final_state, initial_state]), 1e-6)
            for _, final_state, initial_state in TRANSITION_MATRIX_INDICES
        ]
    )
    rate_starts = (
        estimated_rates,
        np.full(_RATE_COUNT, 0.001),
        np.full(_RATE_COUNT, 0.01),
        np.full(_RATE_COUNT, 0.1),
    )
    return tuple(_encode_parameters(rates, initial) for rates in rate_starts)


def _failed_fit(message: str) -> GefRelaxationFit:
    rates = {
        name: GefTransitionRateEstimate(float("nan"), None, False)
        for name in TRANSITION_ORDER
    }
    times = {
        name: GefRelaxationTimeEstimate(float("nan"), None, False)
        for name in _DERIVED_TIME_RATES
    }
    goodness = FitGoodnessOfFit(
        0, float("nan"), float("nan"), float("nan"), float("nan")
    )
    return GefRelaxationFit(
        success=False,
        generator_matrix_per_us=np.full((3, 3), np.nan),
        transition_rates=rates,
        derived_times_ns=times,
        initial_populations={},
        fitted_populations={},
        parameter_order=_PHYSICAL_PARAMETER_ORDER,
        parameter_covariance=np.full(
            (len(_PHYSICAL_PARAMETER_ORDER), len(_PHYSICAL_PARAMETER_ORDER)), np.nan
        ),
        parameter_estimable=dict.fromkeys(_PHYSICAL_PARAMETER_ORDER, False),
        goodness_of_fit=goodness,
        whitened_residual=np.empty(0),
        mahalanobis_residuals={},
        message=message,
        objective=float("nan"),
        jacobian_rank=0,
        jacobian_nullity=_PARAMETER_COUNT,
        jacobian_singular_values=np.empty(0),
    )


def _derived_time_estimate(
    function: Callable[[NDArray[np.float64]], float],
    parameters: NDArray[np.float64],
    covariance: NDArray[np.float64],
    null_space: NDArray[np.float64],
) -> GefRelaxationTimeEstimate:
    value = float(function(parameters))
    gradient = _numerical_jacobian(function, parameters)[0]
    estimable = _gradient_is_estimable(gradient, null_space)
    variance = float(gradient @ covariance @ gradient)
    standard_error = (
        float(np.sqrt(max(variance, 0.0)))
        if estimable and np.isfinite(variance)
        else None
    )
    return GefRelaxationTimeEstimate(value, standard_error, estimable)


def _inverse_rate_time_function(
    *rate_names: str,
) -> Callable[[NDArray[np.float64]], float]:
    """Return a function mapping fit parameters to an inverse-rate time in ns."""
    rate_indices = tuple(TRANSITION_ORDER.index(name) for name in rate_names)

    def evaluate(parameters: NDArray[np.float64]) -> float:
        rates = _physical_vector(parameters)[:_RATE_COUNT]
        return float(1_000.0 / np.sum(rates[list(rate_indices)]))

    return evaluate


def analyze_gef_relaxation(
    measurements: GefRelaxationMeasurements,
    *,
    covariance_rcond: float = 1e-12,
) -> GefRelaxationFit:
    """
    Fit a shared continuous-time six-rate model to G/E/F trajectories.

    Parameters
    ----------
    measurements
        G/E/F population trajectories on a shared time grid in ns.
    covariance_rcond
        Relative covariance eigenvalue cutoff in `[0, 1)`.

    Returns
    -------
    GefRelaxationFit
        Six rates, derived EF times, fitted populations, and GLS diagnostics.
    """
    covariance_rcond = _validate_covariance_rcond(covariance_rcond)
    blocks = [
        ResidualBlock(
            observed=measurements.protocols[protocol].population[index],
            whitener=covariance_whitener(
                measurements.protocols[protocol].covariance[index],
                rcond=covariance_rcond,
            ),
        )
        for protocol in PROTOCOL_ORDER
        for index in range(measurements.time_ns.size)
    ]

    def block_predictions(parameters: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        predicted = _predictions(parameters, measurements.time_ns)
        return [
            predicted[protocol][index]
            for protocol in PROTOCOL_ORDER
            for index in range(measurements.time_ns.size)
        ]

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return whiten_predictions(block_predictions(parameters), blocks)

    block_sizes = tuple(block.whitener.shape[0] for block in blocks)
    fitted = fit_full_gls(
        residual,
        _initializations(measurements),
        parameter_count=_PARAMETER_COUNT,
        residual_block_sizes=block_sizes,
    )
    if not fitted.success:
        return _failed_fit(fitted.message)

    physical = _physical_vector(fitted.parameters)
    transform = _numerical_jacobian(_physical_vector, fitted.parameters)
    raw_covariance = transform @ fitted.covariance @ transform.T
    raw_covariance = 0.5 * (raw_covariance + raw_covariance.T)
    estimable = {
        name: _gradient_is_estimable(transform[index], fitted.null_space)
        for index, name in enumerate(_PHYSICAL_PARAMETER_ORDER)
    }
    covariance = raw_covariance.copy()
    unresolved = [
        index
        for index, name in enumerate(_PHYSICAL_PARAMETER_ORDER)
        if not estimable[name]
    ]
    covariance[unresolved, :] = np.nan
    covariance[:, unresolved] = np.nan
    rate_estimates = {
        name: GefTransitionRateEstimate(
            value_per_us=float(physical[index]),
            standard_error_per_us=(
                float(np.sqrt(max(raw_covariance[index, index], 0.0)))
                if estimable[name] and np.isfinite(raw_covariance[index, index])
                else None
            ),
            estimable=estimable[name],
        )
        for index, name in enumerate(TRANSITION_ORDER)
    }
    derived = {
        name: _derived_time_estimate(
            _inverse_rate_time_function(*rate_names),
            fitted.parameters,
            fitted.covariance,
            fitted.null_space,
        )
        for name, rate_names in _DERIVED_TIME_RATES.items()
    }
    generator, initial = _decode_parameters(fitted.parameters)
    predicted = _predictions(fitted.parameters, measurements.time_ns)
    offsets = np.cumsum((0, *block_sizes))
    block_norms = np.asarray(
        [
            np.linalg.norm(fitted.whitened_residual[start:stop])
            for start, stop in pairwise(offsets)
        ]
    )
    point_count = measurements.time_ns.size
    mahalanobis = {
        protocol: block_norms[index * point_count : (index + 1) * point_count]
        for index, protocol in enumerate(PROTOCOL_ORDER)
    }
    return GefRelaxationFit(
        success=True,
        generator_matrix_per_us=generator,
        transition_rates=rate_estimates,
        derived_times_ns=derived,
        initial_populations=initial,
        fitted_populations=predicted,
        parameter_order=_PHYSICAL_PARAMETER_ORDER,
        parameter_covariance=covariance,
        parameter_estimable=estimable,
        goodness_of_fit=fitted.goodness_of_fit,
        whitened_residual=fitted.whitened_residual,
        mahalanobis_residuals=mahalanobis,
        message=fitted.message,
        objective=fitted.objective,
        jacobian_rank=fitted.jacobian_rank,
        jacobian_nullity=fitted.jacobian_nullity,
        jacobian_singular_values=fitted.jacobian_singular_values,
    )


def _validate_covariance_rcond(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("covariance_rcond must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError("covariance_rcond must be in [0, 1).")
    return normalized


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 2:
        raise ValueError(f"{name} must be at least two.")
    return int(value)


def _finite_positive(value: float, name: str) -> float:
    """Validate one finite positive real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return normalized


def _optional_seed(value: int | None) -> int | None:
    """Validate an optional nonnegative acquisition seed."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("acquisition_seed must be an integer or None.")
    if value < 0:
        raise ValueError("acquisition_seed must be nonnegative.")
    return int(value)


def _validate_acquisition_order(value: str) -> AcquisitionOrder:
    """Validate the schedule submission order."""
    if value not in ("interleaved", "randomized", "grouped"):
        raise ValueError(
            "acquisition_order must be 'interleaved', 'randomized', or 'grouped'."
        )
    return cast(AcquisitionOrder, value)


def _normalize_targets(
    exp: Experiment, targets: Collection[str] | str | None
) -> list[str]:
    if targets is None:
        values = list(exp.ctx.qubit_labels)
    elif isinstance(targets, str):
        values = [targets]
    else:
        values = list(targets)
    resolved = [exp.ctx.resolve_qubit_label(target) for target in values]
    if not resolved:
        raise ValueError("targets must contain at least one qubit.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("targets must not contain duplicates.")
    return resolved


def _resolve_time_range(
    exp: Experiment, time_range: ArrayLike | None
) -> NDArray[np.float64]:
    supplied = DEFAULT_TIME_RANGE_NS if time_range is None else time_range
    object_values = np.asarray(supplied, dtype=object)
    if object_values.ndim != 1:
        raise ValueError("time_range must be a one-dimensional array.")
    if any(isinstance(value, (bool, np.bool_)) for value in object_values):
        raise TypeError("time_range must contain real numbers, not booleans.")
    values = np.asarray(supplied, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            "time_range must be a finite one-dimensional nonnegative array."
        )
    sampling_period = float(exp.ctx.measurement.sampling_period)
    discretized = exp.ctx.util.discretize_time_range(
        values,
        sampling_period=sampling_period,
    )
    if discretized is None:
        raise ValueError("time_range could not be discretized.")
    discretized_values = np.asarray(discretized, dtype=np.float64)
    if (
        discretized_values.ndim != 1
        or not np.all(np.isfinite(discretized_values))
        or np.any(discretized_values < 0.0)
    ):
        raise ValueError(
            "Discretized time_range must be finite, one-dimensional, and nonnegative."
        )
    resolved = np.unique(discretized_values)
    if resolved.size < 5 or resolved[0] != 0.0:
        raise ValueError(
            "time_range must retain at least five unique points including zero."
        )
    return resolved


def _build_preparation(
    exp: Experiment,
    targets: Sequence[str],
    protocol: str,
    wait_ns: float,
) -> PulseSchedule:
    if protocol not in PROTOCOL_ORDER:
        raise ValueError(f"Unknown preparation protocol `{protocol}`.")
    ge_labels = {target: exp.ctx.resolve_ge_label(target) for target in targets}
    ef_labels = {target: exp.ctx.resolve_ef_label(target) for target in targets}
    with PulseSchedule([*ge_labels.values(), *ef_labels.values()]) as schedule:
        if protocol in ("initial_e", "initial_f"):
            for target in targets:
                schedule.add(ge_labels[target], exp.pulse.x180(ge_labels[target]))
            schedule.barrier()
        if protocol == "initial_f":
            for target in targets:
                schedule.add(ef_labels[target], exp.pulse.x180(ef_labels[target]))
            schedule.barrier()
        if wait_ns > 0.0:
            for target in targets:
                schedule.add(ge_labels[target], Blank(wait_ns))
            schedule.barrier()
    return schedule


def _schedule_name(protocol: str, time_ns: float) -> str:
    time_text = np.format_float_positional(float(time_ns), unique=True, trim="-")
    return f"{protocol}/t_ns={time_text}"


def _build_schedules(
    exp: Experiment,
    targets: Sequence[str],
    time_ns: NDArray[np.float64],
    *,
    acquisition_order: AcquisitionOrder,
    acquisition_seed: int | None,
) -> tuple[dict[str, PulseSchedule], dict[str, tuple[str, ...]]]:
    by_protocol = {
        protocol: tuple(_schedule_name(protocol, time) for time in time_ns)
        for protocol in PROTOCOL_ORDER
    }
    schedules = {
        _schedule_name(protocol, time): _build_preparation(
            exp, targets, protocol, float(time)
        )
        for protocol in PROTOCOL_ORDER
        for time in time_ns
    }
    if acquisition_order == "grouped":
        names = [name for protocol in PROTOCOL_ORDER for name in by_protocol[protocol]]
    else:
        points = []
        for index in range(time_ns.size):
            offset = index % len(PROTOCOL_ORDER)
            protocols = (*PROTOCOL_ORDER[offset:], *PROTOCOL_ORDER[:offset])
            points.extend((protocol, index) for protocol in protocols)
        if acquisition_order == "randomized":
            np.random.default_rng(acquisition_seed).shuffle(points)
        names = [by_protocol[protocol][index] for protocol, index in points]
    return {name: schedules[name] for name in names}, by_protocol


def _extract_series(
    acquisition: Result,
    target: str,
    names: Sequence[str],
) -> GefPopulationSeries:
    data = cast(Mapping[str, Any], acquisition.data)
    fits = cast(Mapping[str, Mapping[str, Any]], data["fits"])
    selected = [fits[name][target] for name in names]
    return GefPopulationSeries(
        population=np.stack([fit.population for fit in selected]),
        covariance=np.stack([fit.population_covariance for fit in selected]),
        standard_error=np.stack([fit.population_standard_error for fit in selected]),
        population_unconstrained=np.stack(
            [fit.population_unconstrained for fit in selected]
        ),
        fit_diagnostics=tuple(
            GefPopulationFitDiagnostic(
                objective=float(fit.objective),
                design_rank=int(fit.design_rank),
                design_condition_number=float(fit.design_condition_number),
                residual=np.asarray(fit.residual, dtype=np.float64),
                message=str(fit.message),
            )
            for fit in selected
        ),
    )


def _population_figure(
    target: str,
    measurements: GefRelaxationMeasurements,
    fit: GefRelaxationFit,
) -> go.Figure:
    figure = make_subplots(rows=1, cols=3, subplot_titles=PROTOCOL_ORDER)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for column, protocol in enumerate(PROTOCOL_ORDER, start=1):
        series = measurements.protocols[protocol]
        for state_index, state in enumerate(STATE_ORDER):
            figure.add_trace(
                go.Scatter(
                    x=measurements.time_ns * 1e-3,
                    y=series.population[:, state_index],
                    error_y={"array": series.standard_error[:, state_index]},
                    mode="markers",
                    name=f"P{state}",
                    legendgroup=state,
                    showlegend=column == 1,
                    marker={"color": colors[state_index]},
                ),
                row=1,
                col=column,
            )
            if fit.success:
                figure.add_trace(
                    go.Scatter(
                        x=measurements.time_ns * 1e-3,
                        y=fit.fitted_populations[protocol][:, state_index],
                        mode="lines",
                        name=f"P{state} fit",
                        legendgroup=state,
                        showlegend=False,
                        line={"color": colors[state_index]},
                    ),
                    row=1,
                    col=column,
                )
    figure.update_layout(title=f"GEF relaxation: {target}")
    figure.update_xaxes(title_text="Time (us)")
    figure.update_yaxes(title_text="Population", range=[0.0, 1.0])
    return figure


def _rate_figure(target: str, fit: GefRelaxationFit) -> go.Figure:
    values = fit.generator_matrix_per_us.copy()
    np.fill_diagonal(values, np.nan)
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=STATE_ORDER,
            y=STATE_ORDER,
            colorbar={"title": "1/us"},
            hovertemplate="initial=%{x}<br>final=%{y}<br>rate=%{z:.4g} 1/us<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"GEF transition rates: {target}",
        xaxis_title="Initial state",
        yaxis_title="Final state",
    )
    return figure


def characterize_gef_relaxation(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    time_range: ArrayLike | None = None,
    calibration: Mapping[str, GefPopulationCalibration] | None = None,
    gef_measurement_n_shots: int = DEFAULT_GEF_MEASUREMENT_N_SHOTS,
    gef_calibration_n_shots: int = DEFAULT_GEF_CALIBRATION_N_SHOTS,
    shot_interval: float | None = None,
    covariance_rcond: float = 1e-12,
    acquisition_order: AcquisitionOrder = "interleaved",
    acquisition_seed: int | None = 0,
    enable_tqdm: bool = True,
    plot: bool = True,
) -> Result:
    """
    Measure and fit all six continuous-time transition rates in the GEF manifold.

    Parameters
    ----------
    exp
        Experiment used for pulse construction and acquisition.
    targets
        Qubits to characterize simultaneously. Defaults to all qubits.
    time_range
        Relaxation delays in ns. At least five unique points including zero are
        required after timing-grid discretization.
    calibration
        Optional reusable GEF population calibration.
    gef_measurement_n_shots
        Shots for each GEF population configuration.
    gef_calibration_n_shots
        Shots for each GEF calibration configuration when calibration is omitted.
    shot_interval
        Interval between shots in ns, or the configured default.
    covariance_rcond
        Relative covariance eigenvalue cutoff in `[0, 1)`.
    acquisition_order
        Submission order for protocol and delay points.
    acquisition_seed
        Nonnegative seed used for randomized acquisition order.
    enable_tqdm
        Whether acquisition services display progress.
    plot
        Whether generated figures are displayed.

    Returns
    -------
    Result
        Measurements, six-rate fits, derived EF times, raw GEF data, and figures.

    Notes
    -----
    This function performs hardware acquisition. Rates are reported in `1/us`.
    `t1_f_to_e` is `1 / k_f_to_e`; `f_state_exit_lifetime` also includes direct
    `f_to_g` decay. Local GLS uncertainty is conditional on the fixed GEF
    calibration.
    """
    for name, value in (("enable_tqdm", enable_tqdm), ("plot", plot)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")
    target_list = _normalize_targets(exp, targets)
    resolved_time = _resolve_time_range(exp, time_range)
    measurement_shots = _positive_integer(
        gef_measurement_n_shots, "gef_measurement_n_shots"
    )
    calibration_shots = (
        _positive_integer(gef_calibration_n_shots, "gef_calibration_n_shots")
        if calibration is None
        else None
    )
    covariance_rcond = _validate_covariance_rcond(covariance_rcond)
    resolved_order = _validate_acquisition_order(acquisition_order)
    seed = _optional_seed(acquisition_seed)
    configured_interval = resolve_measurement_defaults(
        exp.ctx.experiment_system.measurement_defaults
    ).execution.shot_interval_ns
    interval = _finite_positive(
        configured_interval if shot_interval is None else shot_interval,
        "shot_interval",
    )
    schedules, names_by_protocol = _build_schedules(
        exp,
        target_list,
        resolved_time,
        acquisition_order=resolved_order,
        acquisition_seed=seed,
    )
    acquisition = measure_gef_populations(
        exp,
        targets=target_list,
        sequences=schedules,
        calibration=calibration,
        n_shots=measurement_shots,
        calibration_n_shots=calibration_shots,
        shot_interval=interval,
        covariance_rcond=covariance_rcond,
        n_bootstrap=0,
        enable_tqdm=enable_tqdm,
        progress_label="GEF relaxation",
    )
    measurements = {
        target: GefRelaxationMeasurements(
            time_ns=resolved_time,
            protocols={
                protocol: _extract_series(
                    acquisition, target, names_by_protocol[protocol]
                )
                for protocol in PROTOCOL_ORDER
            },
        )
        for target in target_list
    }
    analysis = {
        target: analyze_gef_relaxation(
            measurements[target], covariance_rcond=covariance_rcond
        )
        for target in target_list
    }
    figures: dict[str, go.Figure] = {}
    for target in target_list:
        figures[f"{target}/populations"] = _population_figure(
            target, measurements[target], analysis[target]
        )
        figures[f"{target}/rates"] = _rate_figure(target, analysis[target])
    if plot:
        for figure in figures.values():
            figure.show()
    acquisition_data = cast(Mapping[str, Any], acquisition.data)
    return Result(
        data={
            "measurements": measurements,
            "analysis": analysis,
            "raw_data": {
                "gef_calibration": acquisition_data["calibration"],
                "raw_iq": acquisition_data["raw_iq"],
                "moment_summaries": acquisition_data["moment_summaries"],
                "population_fits": acquisition_data["fits"],
            },
            "measurement_options": {
                "time_ns": resolved_time,
                "gef_measurement_n_shots": measurement_shots,
                "gef_calibration_n_shots": calibration_shots,
                "shot_interval_ns": interval,
                "covariance_rcond": covariance_rcond,
                "acquisition_order": resolved_order,
                "acquisition_seed": seed,
                "submitted_sequence_order": tuple(schedules),
                "enable_tqdm": enable_tqdm,
                "plot": plot,
            },
            "analysis_options": {
                "model": "continuous_time_three_state_markov_generator",
                "matrix_convention": "Q[final_state, initial_state]",
                "rate_unit": "1/us",
                "initial_condition_treatment": "latent_nuisance_with_t0_observation",
                "uncertainty": "unscaled_local_gls_conditional_on_gef_calibration",
            },
        },
        figure=figures[f"{target_list[0]}/populations"],
        figures=figures,
    )


__all__ = [
    "GefRelaxationFit",
    "GefRelaxationMeasurements",
    "GefRelaxationTimeEstimate",
    "GefTransitionRateEstimate",
    "analyze_gef_relaxation",
    "characterize_gef_relaxation",
]
