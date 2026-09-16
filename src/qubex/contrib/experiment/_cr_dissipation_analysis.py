"""Physical forward analysis for CR dissipation characterization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ._cr_dissipation_fit import (
    ResidualBlock,
    covariance_whitener,
    fit_gls_candidate,
    scalar_residual_blocks,
    select_aicc_candidate,
    whiten_predictions,
)
from ._cr_dissipation_pulses import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    PROTOCOL_D,
    ProtocolSchedules,
    ZX90Descriptor,
    semantic_zx90,
)
from ._cr_dissipation_simulation import (
    X_CONTROL,
    Y_TARGET,
    CrNoiseRates,
    compose_channel,
    leakage_aware_average_fidelity,
    noiseless_channel,
    propagate_fidelity_uncertainty,
    simulate_repeated_observable,
    state_density,
)
from ._cr_dissipation_types import (
    CandidateFit,
    ControlPopulationRateFit,
    CrDissipationAnalysis,
    CrDissipationFidelityEstimate,
    CrDissipationFidelityLimits,
    CrDissipationFits,
    CrDissipationIdlePrediction,
    CrDissipationMeasurements,
    CrDissipationRateEstimate,
    CrDissipationRateStatus,
    CrDissipationWarning,
    DecayRateFit,
    ExchangeRateFit,
    GefPopulationSeries,
    IdleNoiseParameters,
    PhysicalForwardDephasingFit,
)

POPULATION_MINIMUM_CHANGE = 0.01
LEAKAGE_MINIMUM_CHANGE = 0.003
CHANGE_SIGMA_THRESHOLD = 3.0
MODEL_SELECTION_DELTA_AICC = 6.0
PARAMETER_SIGNIFICANCE_THRESHOLD = 2.0
BOUNDARY_FRACTION_CHANGE_THRESHOLD = 0.25
PSD_RELATIVE_TOLERANCE = 1e-10

_CONTROL_RATE_ORDER = (
    "control_e_to_g",
    "control_g_to_e",
    "control_e_to_f",
    "control_f_to_e",
)


@dataclass(frozen=True)
class _VectorObservation:
    protocol: str
    point_index: int
    component_indices: tuple[int, ...]
    block: ResidualBlock


@dataclass(frozen=True)
class _IdleEquivalentBaseline:
    predictions: dict[str, CrDissipationIdlePrediction]
    equivalent_rates: dict[str, float]
    target_t1rho_by_protocol: dict[str, float]
    raw_cd_predictions: dict[str, NDArray[np.float64]]


def _select_significant_nested_candidate(
    candidates: Mapping[str, CandidateFit],
    null_values: Mapping[str, float],
) -> CandidateFit | None:
    """
    Select by AICc, then iteratively remove sub-2-sigma parameters.

    Each simplification is restricted to a successful nested candidate which
    retains every parameter that was significant in the current model.  The
    loop is finite because every accepted replacement has fewer parameters.
    """
    selected = select_aicc_candidate(
        candidates,
        delta_threshold=MODEL_SELECTION_DELTA_AICC,
    )
    while selected is not None and selected.parameter_names:
        significant: set[str] = set()
        for index, name in enumerate(selected.parameter_names):
            error = float(selected.standard_errors[index])
            distance = abs(float(selected.parameters[index]) - null_values[name])
            if (
                np.isfinite(error)
                and error > 0.0
                and distance / error >= PARAMETER_SIGNIFICANCE_THRESHOLD
            ):
                significant.add(name)
        selected_names = set(selected.parameter_names)
        if significant == selected_names:
            break
        nested = {
            name: candidate
            for name, candidate in candidates.items()
            if candidate.success
            and set(candidate.parameter_names) < selected_names
            and significant.issubset(candidate.parameter_names)
        }
        replacement = select_aicc_candidate(
            nested,
            delta_threshold=MODEL_SELECTION_DELTA_AICC,
        )
        if replacement is None:
            break
        selected = replacement
    return selected


def computational_polarization(
    series: GefPopulationSeries,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return leakage-normalized `(Pg-Pe)/(Pg+Pe)` and delta-method SE."""
    population = np.asarray(series.population, dtype=np.float64)
    covariance = np.asarray(series.covariance, dtype=np.float64)
    denominator = population[:, 0] + population[:, 1]
    values = np.full(population.shape[0], np.nan, dtype=np.float64)
    errors = np.full_like(values, np.nan)
    valid = np.isfinite(denominator) & (denominator > 0.0)
    values[valid] = (population[valid, 0] - population[valid, 1]) / denominator[valid]
    for index in np.flatnonzero(valid):
        pg, pe = population[index, :2]
        total = pg + pe
        gradient = np.array([2.0 * pe / total**2, -2.0 * pg / total**2, 0.0])
        cov = covariance[index]
        if cov.shape == (3, 3) and np.all(np.isfinite(cov)):
            variance = float(gradient @ cov @ gradient)
            if variance >= -1e-15:
                errors[index] = np.sqrt(max(variance, 0.0))
    return values, errors


def nonnormalized_ge_expectation(
    series: GefPopulationSeries,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return signed non-normalized `Pg-Pe` and its full-covariance SE."""
    population = np.asarray(series.population, dtype=np.float64)
    difference = population[:, 0] - population[:, 1]
    sign = 1.0 if difference.size == 0 or difference[0] >= 0.0 else -1.0
    values = sign * difference
    errors = np.full(values.shape, np.nan, dtype=np.float64)
    vector = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    for index, covariance in enumerate(np.asarray(series.covariance, dtype=np.float64)):
        if covariance.shape == (3, 3) and np.all(np.isfinite(covariance)):
            variance = float(vector @ covariance @ vector)
            if variance >= -1e-15:
                errors[index] = np.sqrt(max(variance, 0.0))
    return values, errors, sign


def three_level_rate_matrix(
    e_to_g: float,
    g_to_e: float,
    e_to_f: float,
    f_to_e: float,
) -> NDArray[np.float64]:
    """Return the column-population generator for adjacent qutrit exchange."""
    return np.array(
        [
            [-g_to_e, e_to_g, 0.0],
            [g_to_e, -(e_to_g + e_to_f), f_to_e],
            [0.0, e_to_f, -f_to_e],
        ],
        dtype=np.float64,
    )


def control_population_trajectory(
    repetition_counts: ArrayLike,
    initial_population: ArrayLike,
    rates: Mapping[str, float],
    *,
    cr_lobe_duration_ns: float,
    blank_duration_ns: float,
    idle_e_to_g_per_ns: float,
) -> NDArray[np.float64]:
    """Propagate A/B populations through `4n` segment-wise un-echoed units."""
    cr_matrix = three_level_rate_matrix(
        rates["control_e_to_g"],
        rates["control_g_to_e"],
        rates["control_e_to_f"],
        rates["control_f_to_e"],
    )
    idle_matrix = three_level_rate_matrix(idle_e_to_g_per_ns, 0.0, 0.0, 0.0)
    cr_step = expm(cr_matrix * cr_lobe_duration_ns)
    blank_step = expm(idle_matrix * blank_duration_ns)
    unit = blank_step @ cr_step @ blank_step @ cr_step
    initial = np.asarray(initial_population, dtype=np.float64)
    counts = np.asarray(repetition_counts, dtype=np.int64).reshape(-1)
    return np.asarray(
        [np.linalg.matrix_power(unit, 4 * int(count)) @ initial for count in counts],
        dtype=np.float64,
    )


def target_t1rho_trajectory(
    repetition_counts: ArrayLike,
    initial_value: float,
    rate_per_ns: float,
    x_infinity: float,
    *,
    cr_lobe_duration_ns: float,
    blank_duration_ns: float,
    idle_transverse_rate_per_ns: float,
) -> NDArray[np.float64]:
    """Propagate target X through actual CR/blank segment ordering."""
    cr_factor = np.exp(-rate_per_ns * cr_lobe_duration_ns)
    blank_factor = np.exp(-idle_transverse_rate_per_ns * blank_duration_ns)

    def one_unit(value: float) -> float:
        value = x_infinity + (value - x_infinity) * cr_factor
        value *= blank_factor
        value = x_infinity + (value - x_infinity) * cr_factor
        return value * blank_factor

    output = []
    counts = np.asarray(repetition_counts, dtype=np.int64).reshape(-1)
    for count in counts:
        value = float(initial_value)
        for _ in range(4 * int(count)):
            value = one_unit(value)
        output.append(value)
    return np.asarray(output, dtype=np.float64)


def target_exchange_trajectory(
    repetition_counts: ArrayLike,
    initial_pf: float,
    leakage_rate_per_ns: float,
    seepage_rate_per_ns: float,
    *,
    cr_lobe_duration_ns: float,
) -> NDArray[np.float64]:
    """Propagate target f population through the eight CR lobes per block."""
    counts = np.asarray(repetition_counts, dtype=np.float64).reshape(-1)
    total = leakage_rate_per_ns + seepage_rate_per_ns
    if total == 0.0:
        return np.full(counts.size, initial_pf, dtype=np.float64)
    equilibrium = leakage_rate_per_ns / total
    active_time = counts * 8.0 * cr_lobe_duration_ns
    return equilibrium + (initial_pf - equilibrium) * np.exp(-total * active_time)


def _rate_upper_bound(active_time_ns: NDArray[np.float64]) -> float:
    positive = active_time_ns[np.isfinite(active_time_ns) & (active_time_ns > 0.0)]
    return 1.0 if positive.size == 0 else max(1e-8, 50.0 / float(np.min(positive)))


def _control_observations(
    measurements: CrDissipationMeasurements,
    *,
    covariance_rcond: float,
) -> tuple[tuple[_VectorObservation, ...], tuple[str, ...]]:
    observations: list[_VectorObservation] = []
    excluded: list[str] = []
    for protocol in (PROTOCOL_A, PROTOCOL_B):
        series = getattr(measurements, protocol).actual.control_gef
        if series is None:
            raise ValueError(f"{protocol} is missing control GEF data.")
        for index in range(1, measurements.repetition_counts.size):
            observed = np.asarray(series.population[index, 1:3], dtype=np.float64)
            covariance = np.asarray(
                series.covariance[index][1:3, 1:3], dtype=np.float64
            )
            label = f"{protocol}[{index}]"
            try:
                whitener, _ = covariance_whitener(covariance, rcond=covariance_rcond)
                observations.append(
                    _VectorObservation(
                        protocol,
                        index,
                        (1, 2),
                        ResidualBlock(observed, whitener, label),
                    )
                )
                continue
            except ValueError:
                pass
            errors = np.asarray(series.standard_error[index, 1:3], dtype=np.float64)
            usable = np.flatnonzero(
                np.isfinite(errors) & (errors > 0.0) & np.isfinite(observed)
            )
            if usable.size:
                observations.append(
                    _VectorObservation(
                        protocol,
                        index,
                        tuple(int(value + 1) for value in usable),
                        ResidualBlock(
                            observed[usable],
                            np.diag(1.0 / errors[usable]),
                            label,
                        ),
                    )
                )
            else:
                excluded.append(label)
    return tuple(observations), tuple(excluded)


def _change_detected(
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    minimum_change: float,
    include_initial_error: bool,
) -> bool:
    difference = np.abs(values[1:] - reference[1:])
    denominator = errors[1:].copy()
    if include_initial_error:
        denominator = np.sqrt(np.square(denominator) + errors[0] ** 2)
    valid = np.isfinite(difference) & np.isfinite(denominator) & (denominator > 0.0)
    if not np.any(valid):
        return False
    return bool(
        np.max(difference[valid]) >= minimum_change
        and np.max(difference[valid] / denominator[valid]) >= CHANGE_SIGMA_THRESHOLD
    )


def _simplex_boundary_override(
    series: GefPopulationSeries,
    component: int,
    *,
    minimum_change: float,
) -> bool:
    """Keep dynamic candidates when unconstrained/bootstrapped boundary data move."""
    constrained = np.asarray(series.population[:, component], dtype=np.float64)
    if not np.any(np.isclose(constrained, 0.0, atol=1e-10)):
        return False
    unconstrained = np.asarray(
        series.population_unconstrained[:, component],
        dtype=np.float64,
    )
    if unconstrained.size > 1 and np.all(np.isfinite(unconstrained)):
        if (
            float(np.max(np.abs(unconstrained[1:] - unconstrained[0])))
            >= minimum_change
        ):
            return True
    if not series.bootstrap:
        return False
    initial = series.bootstrap[0]
    initial_boundary_fraction = getattr(initial, "boundary_fraction", None)
    for item in series.bootstrap[1:]:
        initial_interval = getattr(initial, "confidence_interval", None)
        interval = getattr(item, "confidence_interval", None)
        if initial_interval is not None and interval is not None:
            initial_interval = np.asarray(initial_interval, dtype=np.float64)[
                :, component
            ]
            interval = np.asarray(interval, dtype=np.float64)[:, component]
            if np.all(np.isfinite(initial_interval)) and np.all(np.isfinite(interval)):
                if (
                    interval[0] > initial_interval[1]
                    or interval[1] < initial_interval[0]
                ):
                    return True
        # Boundary occupancy is only an auxiliary signal: require the
        # constrained trace itself to look flat and pinned to the simplex.
        boundary_fraction = getattr(item, "boundary_fraction", None)
        if initial_boundary_fraction is None or boundary_fraction is None:
            continue
        initial_fraction = np.asarray(initial_boundary_fraction, dtype=np.float64)
        fraction = np.asarray(boundary_fraction, dtype=np.float64)
        if (
            initial_fraction.shape == (3,)
            and fraction.shape == (3,)
            and np.isfinite(initial_fraction[component])
            and np.isfinite(fraction[component])
            and float(np.ptp(constrained)) < minimum_change
            and abs(fraction[component] - initial_fraction[component])
            >= BOUNDARY_FRACTION_CHANGE_THRESHOLD
        ):
            return True
    return False


def fit_control_populations(
    measurements: CrDissipationMeasurements,
    descriptor: ZX90Descriptor,
    control_idle: IdleNoiseParameters,
    *,
    covariance_rcond: float,
    force_all_candidates: bool = False,
) -> ControlPopulationRateFit:
    """Jointly fit the A/B three-state control model and select submodels."""
    observations, excluded = _control_observations(
        measurements,
        covariance_rcond=covariance_rcond,
    )
    n_observations = sum(item.block.rank for item in observations)
    initial_by_protocol = {}
    for protocol in (PROTOCOL_A, PROTOCOL_B):
        series = getattr(measurements, protocol).actual.control_gef
        if series is None:
            raise ValueError(f"{protocol} is missing control GEF data.")
        initial_by_protocol[protocol] = np.asarray(
            series.population[0], dtype=np.float64
        )
    counts = measurements.repetition_counts
    idle_rate = control_idle.relaxation_rate_per_ns
    null_rates: dict[str, float] = dict.fromkeys(_CONTROL_RATE_ORDER, 0.0)
    null_rates["control_e_to_g"] = idle_rate
    null_predictions = {
        protocol: control_population_trajectory(
            counts,
            initial,
            null_rates,
            cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
            blank_duration_ns=descriptor.echo_slot_duration_ns,
            idle_e_to_g_per_ns=idle_rate,
        )
        for protocol, initial in initial_by_protocol.items()
    }
    ge_dynamic = force_all_candidates
    ef_dynamic = force_all_candidates
    for protocol in (PROTOCOL_A, PROTOCOL_B):
        series = getattr(measurements, protocol).actual.control_gef
        if series is None:
            raise ValueError(f"{protocol} is missing control GEF data.")
        ge_dynamic |= _change_detected(
            series.population[:, 1],
            series.standard_error[:, 1],
            null_predictions[protocol][:, 1],
            minimum_change=POPULATION_MINIMUM_CHANGE,
            include_initial_error=False,
        )
        ef_dynamic |= _change_detected(
            series.population[:, 2],
            series.standard_error[:, 2],
            np.full(counts.size, series.population[0, 2]),
            minimum_change=LEAKAGE_MINIMUM_CHANGE,
            include_initial_error=True,
        )
        ef_dynamic |= _simplex_boundary_override(
            series,
            2,
            minimum_change=LEAKAGE_MINIMUM_CHANGE,
        )
    g_models = {
        "G0": ({"control_e_to_g": idle_rate, "control_g_to_e": 0.0}, ()),
        "G1": ({"control_g_to_e": 0.0}, ("control_e_to_g",)),
        "G1b": ({"control_e_to_g": idle_rate}, ("control_g_to_e",)),
        "G2": ({}, ("control_e_to_g", "control_g_to_e")),
    }
    f_models = {
        "F0": ({"control_e_to_f": 0.0, "control_f_to_e": 0.0}, ()),
        "F1": ({"control_f_to_e": 0.0}, ("control_e_to_f",)),
        "F1b": ({"control_e_to_f": 0.0}, ("control_f_to_e",)),
        "F2": ({}, ("control_e_to_f", "control_f_to_e")),
    }
    retained_g = tuple(g_models) if ge_dynamic else ("G0",)
    retained_f = tuple(f_models) if ef_dynamic else ("F0",)
    upper = _rate_upper_bound(getattr(measurements, PROTOCOL_A).cr_active_time_ns)
    candidates: dict[str, CandidateFit] = {}
    candidate_rates: dict[str, dict[str, float]] = {}
    candidate_predictions: dict[str, dict[str, NDArray[np.float64]]] = {}
    blocks = tuple(item.block for item in observations)

    for g_name in retained_g:
        for f_name in retained_f:
            name = f"{g_name} x {f_name}"
            fixed = {**g_models[g_name][0], **f_models[f_name][0]}
            free_names = (*g_models[g_name][1], *f_models[f_name][1])

            def rates_from(
                parameters: NDArray[np.float64],
                *,
                fixed_values: Mapping[str, float] = fixed,
                names: tuple[str, ...] = free_names,
            ) -> dict[str, float]:
                rates = dict(fixed_values)
                rates.update(zip(names, parameters, strict=True))
                return rates

            def trajectories(
                parameters: NDArray[np.float64],
            ) -> dict[str, NDArray[np.float64]]:
                rates = rates_from(parameters)
                return {
                    protocol: control_population_trajectory(
                        counts,
                        initial,
                        rates,
                        cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
                        blank_duration_ns=descriptor.echo_slot_duration_ns,
                        idle_e_to_g_per_ns=idle_rate,
                    )
                    for protocol, initial in initial_by_protocol.items()
                }

            def block_predictions(
                parameters: NDArray[np.float64],
            ) -> list[NDArray[np.float64]]:
                predicted = trajectories(parameters)
                return [
                    predicted[item.protocol][item.point_index, item.component_indices]
                    for item in observations
                ]

            def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
                return whiten_predictions(block_predictions(parameters), blocks)

            def prediction(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
                return (
                    np.concatenate(block_predictions(parameters))
                    if observations
                    else np.empty(0)
                )

            initial = np.full(len(free_names), max(idle_rate, 1e-5), dtype=np.float64)
            candidate = fit_gls_candidate(
                name=name,
                parameter_names=free_names,
                initial=initial,
                bounds=(np.zeros(len(free_names)), np.full(len(free_names), upper)),
                residual=residual,
                prediction=prediction,
                n_observations=n_observations,
            )
            candidates[name] = candidate
            if candidate.success:
                candidate_rates[name] = rates_from(candidate.parameters)
                candidate_predictions[name] = trajectories(candidate.parameters)

    selected = _select_significant_nested_candidate(
        candidates,
        null_rates,
    )
    if selected is None:
        return _failed_control_fit(candidates, excluded)
    rates = candidate_rates[selected.name]
    g_name, f_name = selected.name.split(" x ")
    statuses = {
        "control_e_to_g": (
            CrDissipationRateStatus.UNRESOLVED_ASSUMED_IDLE
            if g_name in ("G0", "G1b")
            else CrDissipationRateStatus.RESOLVED
        ),
        "control_g_to_e": (
            CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED
            if g_name in ("G0", "G1")
            else CrDissipationRateStatus.RESOLVED
        ),
        "control_e_to_f": (
            CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED
            if f_name in ("F0", "F1b")
            else CrDissipationRateStatus.RESOLVED
        ),
        "control_f_to_e": (
            CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED
            if f_name in ("F0", "F1")
            else CrDissipationRateStatus.RESOLVED
        ),
    }
    covariance = np.zeros((4, 4), dtype=np.float64)
    for left, left_name in enumerate(_CONTROL_RATE_ORDER):
        if left_name not in selected.parameter_names:
            continue
        source_left = selected.parameter_names.index(left_name)
        for right, right_name in enumerate(_CONTROL_RATE_ORDER):
            if right_name in selected.parameter_names:
                source_right = selected.parameter_names.index(right_name)
                covariance[left, right] = selected.covariance[source_left, source_right]
    return ControlPopulationRateFit(
        success=True,
        selected_model=selected.name,
        rates_per_ns=rates,
        statuses=statuses,
        covariance=covariance,
        parameter_order=_CONTROL_RATE_ORDER,
        candidates=candidates,
        fitted_populations=candidate_predictions[selected.name],
        message=(
            "Final ordinary-GLS joint fit selected."
            + (f" Excluded points: {', '.join(excluded)}." if excluded else "")
        ),
    )


def _failed_control_fit(
    candidates: dict[str, CandidateFit], excluded: Sequence[str]
) -> ControlPopulationRateFit:
    return ControlPopulationRateFit(
        success=False,
        selected_model="fit_failed",
        rates_per_ns={},
        statuses=dict.fromkeys(
            _CONTROL_RATE_ORDER,
            CrDissipationRateStatus.FIT_FAILED,
        ),
        covariance=np.full((4, 4), np.nan),
        parameter_order=_CONTROL_RATE_ORDER,
        candidates=candidates,
        fitted_populations={},
        message="No identifiable joint GLS candidate. "
        + (f"Excluded points: {', '.join(excluded)}." if excluded else ""),
    )


def fit_target_t1rho(
    measurements: CrDissipationMeasurements,
    protocol: str,
    descriptor: ZX90Descriptor,
    target_idle: IdleNoiseParameters,
    *,
    idle_equivalent_rate_per_ns: float,
) -> DecayRateFit:
    """Fit one A/B target T1rho rate against its precomputed idle equivalent."""
    data = getattr(measurements, protocol).actual
    if data.target_x_comp is None or data.target_x_comp_standard_error is None:
        raise ValueError(f"{protocol} is missing target X data.")
    counts = measurements.repetition_counts
    blocks = scalar_residual_blocks(
        data.target_x_comp[1:],
        data.target_x_comp_standard_error[1:],
    )
    retained_indices = [
        index
        for index in range(1, counts.size)
        if np.isfinite(data.target_x_comp[index])
        and np.isfinite(data.target_x_comp_standard_error[index])
        and data.target_x_comp_standard_error[index] > 0.0
    ]
    n_observations = sum(block.rank for block in blocks)
    initial_value = float(data.target_x_comp[0])
    active = getattr(measurements, protocol).cr_active_time_ns
    upper = _rate_upper_bound(active)

    def trajectory(rate: float, x_infinity: float) -> NDArray[np.float64]:
        return target_t1rho_trajectory(
            counts,
            initial_value,
            rate,
            x_infinity,
            cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
            blank_duration_ns=descriptor.echo_slot_duration_ns,
            idle_transverse_rate_per_ns=target_idle.transverse_rate_per_ns,
        )

    def make_candidate(name: str, fixed_rate: float | None) -> CandidateFit:
        names = (
            ("x_infinity",) if fixed_rate is not None else ("rate_per_ns", "x_infinity")
        )

        def unpack(parameters: NDArray[np.float64]) -> tuple[float, float]:
            return (
                (fixed_rate, float(parameters[0]))
                if fixed_rate is not None
                else (float(parameters[0]), float(parameters[1]))
            )

        def predictions(parameters: NDArray[np.float64]) -> list[NDArray[np.float64]]:
            rate, x_inf = unpack(parameters)
            values = trajectory(rate, x_inf)
            return [np.array([values[index]]) for index in retained_indices]

        return fit_gls_candidate(
            name=name,
            parameter_names=names,
            initial=np.array([0.0])
            if fixed_rate is not None
            else np.array([max(fixed_rate or 0.0, 1e-5), 0.0]),
            bounds=(
                np.array([-1.5]) if fixed_rate is not None else np.array([0.0, -1.5]),
                np.array([1.5]) if fixed_rate is not None else np.array([upper, 1.5]),
            ),
            residual=lambda parameters: whiten_predictions(
                predictions(parameters), blocks
            ),
            prediction=lambda parameters: np.concatenate(predictions(parameters)),
            n_observations=n_observations,
        )

    null = make_candidate("idle_equivalent", idle_equivalent_rate_per_ns)
    free = make_candidate("free", None)
    candidates = {null.name: null, free.name: free}
    if not null.success or not free.success:
        return DecayRateFit(
            False,
            "fit_failed",
            float("nan"),
            None,
            CrDissipationRateStatus.FIT_FAILED,
            float("nan"),
            candidates,
            np.full(counts.shape, np.nan),
            "T1rho null/free GLS comparison failed.",
        )
    rate = float(free.parameters[0])
    error = float(free.standard_errors[0])
    delta = (
        float(null.aicc - free.aicc)
        if null.aicc is not None and free.aicc is not None
        else -np.inf
    )
    significance = (
        abs(rate - idle_equivalent_rate_per_ns) / error
        if error > 0.0 and np.isfinite(error)
        else 0.0
    )
    if (
        delta >= MODEL_SELECTION_DELTA_AICC
        and significance >= PARAMETER_SIGNIFICANCE_THRESHOLD
    ):
        selected = free
        status = CrDissipationRateStatus.RESOLVED
        selected_rate = rate
        selected_error: float | None = error
        x_infinity = float(free.parameters[1])
    else:
        selected = null
        status = CrDissipationRateStatus.UNRESOLVED_ASSUMED_IDLE
        selected_rate = idle_equivalent_rate_per_ns
        selected_error = None
        x_infinity = float(null.parameters[0])
    return DecayRateFit(
        True,
        selected.name,
        selected_rate,
        selected_error,
        status,
        x_infinity,
        candidates,
        trajectory(selected_rate, x_infinity),
        "Compared a free total CR-active rate with the precomputed idle equivalent.",
    )


def fit_target_exchange(
    measurements: CrDissipationMeasurements,
    protocol: str,
    descriptor: ZX90Descriptor,
    *,
    force_all_candidates: bool = False,
) -> ExchangeRateFit:
    """Fit L0/L1/L1b/L2 to one target f-population series."""
    series = getattr(measurements, protocol).actual.target_gef
    if series is None:
        raise ValueError(f"{protocol} is missing target GEF data.")
    counts = measurements.repetition_counts
    values = np.asarray(series.population[:, 2], dtype=np.float64)
    errors = np.asarray(series.standard_error[:, 2], dtype=np.float64)
    retained = [
        index
        for index in range(1, counts.size)
        if np.isfinite(values[index])
        and np.isfinite(errors[index])
        and errors[index] > 0.0
    ]
    blocks = scalar_residual_blocks(values[retained], errors[retained])
    n_observations = sum(block.rank for block in blocks)
    dynamic = force_all_candidates or _change_detected(
        values,
        errors,
        np.full(values.shape, values[0]),
        minimum_change=LEAKAGE_MINIMUM_CHANGE,
        include_initial_error=True,
    )
    dynamic |= _simplex_boundary_override(
        series,
        2,
        minimum_change=LEAKAGE_MINIMUM_CHANGE,
    )
    models = {
        "L0": (0.0, 0.0, ()),
        "L1": (None, 0.0, ("leakage_rate_per_ns",)),
        "L1b": (0.0, None, ("seepage_rate_per_ns",)),
        "L2": (None, None, ("leakage_rate_per_ns", "seepage_rate_per_ns")),
    }
    retained_models = tuple(models) if dynamic else ("L0",)
    upper = _rate_upper_bound(getattr(measurements, protocol).cr_active_time_ns)
    candidates: dict[str, CandidateFit] = {}
    trajectories: dict[str, NDArray[np.float64]] = {}
    rates: dict[str, tuple[float, float]] = {}
    for name in retained_models:
        fixed_leak, fixed_seep, parameter_names = models[name]

        def unpack(
            parameters: NDArray[np.float64],
            *,
            leak_fixed: float | None = fixed_leak,
            seep_fixed: float | None = fixed_seep,
        ) -> tuple[float, float]:
            iterator = iter(parameters)
            leak = float(next(iterator)) if leak_fixed is None else leak_fixed
            seep = float(next(iterator)) if seep_fixed is None else seep_fixed
            return leak, seep

        def trajectory(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            leak, seep = unpack(parameters)
            return target_exchange_trajectory(
                counts,
                values[0],
                leak,
                seep,
                cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
            )

        def predictions(parameters: NDArray[np.float64]) -> list[NDArray[np.float64]]:
            predicted = trajectory(parameters)
            return [np.array([predicted[index]]) for index in retained]

        candidate = fit_gls_candidate(
            name=name,
            parameter_names=parameter_names,
            initial=np.full(len(parameter_names), 1e-5),
            bounds=(
                np.zeros(len(parameter_names)),
                np.full(len(parameter_names), upper),
            ),
            residual=lambda parameters: whiten_predictions(
                predictions(parameters), blocks
            ),
            prediction=lambda parameters: np.concatenate(predictions(parameters)),
            n_observations=n_observations,
        )
        candidates[name] = candidate
        if candidate.success:
            rates[name] = unpack(candidate.parameters)
            trajectories[name] = trajectory(candidate.parameters)
    selected = _select_significant_nested_candidate(
        candidates,
        {"leakage_rate_per_ns": 0.0, "seepage_rate_per_ns": 0.0},
    )
    if selected is None:
        return _failed_exchange_fit(candidates, counts.size)
    leak, seep = rates[selected.name]
    leak_free = "leakage_rate_per_ns" in selected.parameter_names
    seep_free = "seepage_rate_per_ns" in selected.parameter_names
    leak_error = (
        float(
            selected.standard_errors[
                selected.parameter_names.index("leakage_rate_per_ns")
            ]
        )
        if leak_free
        else None
    )
    seep_error = (
        float(
            selected.standard_errors[
                selected.parameter_names.index("seepage_rate_per_ns")
            ]
        )
        if seep_free
        else None
    )
    covariance = np.zeros((2, 2), dtype=np.float64)
    for left, left_name in enumerate(("leakage_rate_per_ns", "seepage_rate_per_ns")):
        if left_name in selected.parameter_names:
            source_left = selected.parameter_names.index(left_name)
            for right, right_name in enumerate(
                ("leakage_rate_per_ns", "seepage_rate_per_ns")
            ):
                if right_name in selected.parameter_names:
                    source_right = selected.parameter_names.index(right_name)
                    covariance[left, right] = selected.covariance[
                        source_left, source_right
                    ]
    return ExchangeRateFit(
        True,
        selected.name,
        leak,
        seep,
        leak_error,
        seep_error,
        CrDissipationRateStatus.RESOLVED
        if leak_free
        else CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED,
        CrDissipationRateStatus.RESOLVED
        if seep_free
        else CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED,
        covariance,
        candidates,
        trajectories[selected.name],
        "Selected a reduced target leakage/seepage model by ordinary-GLS AICc.",
    )


def _failed_exchange_fit(
    candidates: dict[str, CandidateFit], size: int
) -> ExchangeRateFit:
    return ExchangeRateFit(
        False,
        "fit_failed",
        float("nan"),
        float("nan"),
        None,
        None,
        CrDissipationRateStatus.FIT_FAILED,
        CrDissipationRateStatus.FIT_FAILED,
        np.full((2, 2), np.nan),
        candidates,
        np.full(size, np.nan),
        "No identifiable target exchange candidate.",
    )


def fit_physical_dephasing(
    values: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    repetition_counts: NDArray[np.int64],
    block_operations: Sequence[Any],
    *,
    initial_density: NDArray[np.complex128],
    observable: NDArray[np.complex128],
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    fixed_rates: CrNoiseRates,
    fitted_role: str,
    force_free: bool = False,
) -> PhysicalForwardDephasingFit:
    """Fit C/D additional pure dephasing with affine SPAM nuisance terms."""
    blocks = scalar_residual_blocks(values, standard_errors)
    retained = [
        index
        for index in range(values.size)
        if np.isfinite(values[index])
        and np.isfinite(standard_errors[index])
        and standard_errors[index] > 0.0
    ]
    n_observations = sum(block.rank for block in blocks)
    maximum_count = max(1, int(np.max(repetition_counts)))
    duration = (
        sum(
            float(getattr(operation, "duration_ns", 0.0))
            for operation in block_operations
            if getattr(operation, "cr_active", False)
        )
        * maximum_count
    )
    upper = max(1e-8, 50.0 / max(duration, 1.0))
    cache: dict[tuple[float, bool], NDArray[np.float64]] = {}

    def simulated(rate: float, signed: bool = False) -> NDArray[np.float64]:
        key = (float(rate), signed)
        if key not in cache:
            kwargs: dict[str, float] = {}
            rates = fixed_rates
            if fitted_role == "control":
                rates = replace(fixed_rates, control_pure_dephasing=max(rate, 0.0))
                if signed:
                    kwargs["signed_control_pure_dephasing"] = rate
            else:
                rates = replace(
                    fixed_rates,
                    target_rotating_frame_pure_dephasing=max(rate, 0.0),
                )
                if signed:
                    kwargs["signed_target_pure_dephasing"] = rate
            cache[key] = simulate_repeated_observable(
                block_operations,
                repetition_counts,
                initial_density,
                observable,
                control_idle=control_idle,
                target_idle=target_idle,
                cr_rates=rates,
                include_leakage=True,
                **kwargs,
            )
        return cache[key]

    def make_candidate(
        name: str, *, free_rate: bool, signed: bool = False
    ) -> CandidateFit:
        names = (
            ("spam_amplitude", "spam_offset")
            if not free_rate
            else (
                "pure_dephasing_rate_per_ns",
                "spam_amplitude",
                "spam_offset",
            )
        )

        def unpack(parameters: NDArray[np.float64]) -> tuple[float, float, float]:
            return (
                (0.0, float(parameters[0]), float(parameters[1]))
                if not free_rate
                else (float(parameters[0]), float(parameters[1]), float(parameters[2]))
            )

        def curve(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            rate, amplitude, offset = unpack(parameters)
            return amplitude * simulated(rate, signed=signed) + offset

        def predictions(parameters: NDArray[np.float64]) -> list[NDArray[np.float64]]:
            result = curve(parameters)
            return [np.array([result[index]]) for index in retained]

        if free_rate:
            initial = np.array([min(1e-5, upper / 10.0), 1.0, 0.0])
            lower_rate = -upper if signed else 0.0
            bounds = (np.array([lower_rate, -2.0, -2.0]), np.array([upper, 2.0, 2.0]))
        else:
            initial = np.array([1.0, 0.0])
            bounds = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
        return fit_gls_candidate(
            name=name,
            parameter_names=names,
            initial=initial,
            bounds=bounds,
            residual=lambda parameters: whiten_predictions(
                predictions(parameters), blocks
            ),
            prediction=lambda parameters: curve(parameters),
            n_observations=n_observations,
            max_nfev=300,
        )

    free = make_candidate("free_nonnegative", free_rate=True)
    if force_free:
        candidates = {free.name: free}
        if not free.success:
            return _failed_physical_fit(candidates, values.size)
        return PhysicalForwardDephasingFit(
            True,
            free.name,
            float(free.parameters[0]),
            float(free.standard_errors[0]),
            CrDissipationRateStatus.RESOLVED,
            float(free.parameters[1]),
            float(free.parameters[2]),
            candidates,
            np.asarray(free.prediction, dtype=np.float64),
            None,
            None,
            "Free constrained estimator used for non-recursive idle baseline.",
        )
    null = make_candidate("zero_additional_dephasing", free_rate=False)
    diagnostic = make_candidate("signed_diagnostic", free_rate=True, signed=True)
    candidates = {null.name: null, free.name: free, diagnostic.name: diagnostic}
    if not null.success or not free.success:
        return _failed_physical_fit(candidates, values.size)
    error = float(free.standard_errors[0])
    delta = (
        float(null.aicc - free.aicc)
        if null.aicc is not None and free.aicc is not None
        else -np.inf
    )
    significance = (
        float(free.parameters[0]) / error if error > 0.0 and np.isfinite(error) else 0.0
    )
    if (
        delta >= MODEL_SELECTION_DELTA_AICC
        and significance >= PARAMETER_SIGNIFICANCE_THRESHOLD
    ):
        selected = free
        rate = float(free.parameters[0])
        rate_error: float | None = error
        status = CrDissipationRateStatus.RESOLVED
        amplitude = float(free.parameters[1])
        offset = float(free.parameters[2])
    else:
        selected = null
        rate = 0.0
        rate_error = None
        status = CrDissipationRateStatus.CONSISTENT_WITH_ZERO
        amplitude = float(null.parameters[0])
        offset = float(null.parameters[1])
    unconstrained_rate = float(diagnostic.parameters[0]) if diagnostic.success else None
    unconstrained_error = (
        float(diagnostic.standard_errors[0]) if diagnostic.success else None
    )
    if (
        unconstrained_rate is not None
        and unconstrained_error is not None
        and np.isfinite(unconstrained_error)
        and unconstrained_rate < -2.0 * unconstrained_error
    ):
        status = CrDissipationRateStatus.INCONSISTENT_RATE_DECOMPOSITION
        rate = 0.0
        rate_error = None
    return PhysicalForwardDephasingFit(
        True,
        selected.name,
        rate,
        rate_error,
        status,
        amplitude,
        offset,
        candidates,
        amplitude * simulated(rate) + offset,
        unconstrained_rate,
        unconstrained_error,
        "A/B nominal dissipation was fixed while additional pure dephasing was tested.",
    )


def _failed_physical_fit(
    candidates: dict[str, CandidateFit], size: int
) -> PhysicalForwardDephasingFit:
    return PhysicalForwardDephasingFit(
        False,
        "fit_failed",
        float("nan"),
        None,
        CrDissipationRateStatus.FIT_FAILED,
        float("nan"),
        float("nan"),
        candidates,
        np.full(size, np.nan),
        None,
        None,
        "Physical-forward GLS fit failed or was underdetermined.",
    )


def _dependency_failed_physical_fit(
    message: str, size: int
) -> PhysicalForwardDephasingFit:
    fit = _failed_physical_fit({}, size)
    return replace(fit, message=message)


def _aggregate_status(
    left: CrDissipationRateStatus,
    right: CrDissipationRateStatus,
    *,
    unresolved: CrDissipationRateStatus,
) -> CrDissipationRateStatus:
    if CrDissipationRateStatus.FIT_FAILED in (left, right):
        return CrDissipationRateStatus.FIT_FAILED
    if left == right == CrDissipationRateStatus.RESOLVED:
        return CrDissipationRateStatus.RESOLVED
    if left == right == unresolved:
        return unresolved
    return CrDissipationRateStatus.PARTIALLY_UNRESOLVED


def _aggregate_value_error(
    left_value: float,
    left_error: float | None,
    right_value: float,
    right_error: float | None,
) -> tuple[float, float | None]:
    value = 0.5 * (left_value + right_value)
    errors = [
        error
        for error in (left_error, right_error)
        if error is not None and np.isfinite(error)
    ]
    return value, (
        0.5 * float(np.sqrt(sum(error**2 for error in errors))) if errors else None
    )


def _control_state_dependence_warning(
    quantity: str,
    left_value: float,
    left_error: float | None,
    left_status: CrDissipationRateStatus,
    right_value: float,
    right_error: float | None,
    right_status: CrDissipationRateStatus,
) -> CrDissipationWarning | None:
    """Return a diagnostic warning for a resolved A/B rate discrepancy."""
    if left_status != right_status or left_status != CrDissipationRateStatus.RESOLVED:
        return None
    if left_error is None or right_error is None:
        return None
    values = np.array([left_value, left_error, right_value, right_error])
    if not np.all(np.isfinite(values)) or left_error < 0.0 or right_error < 0.0:
        return None
    combined_error = float(np.hypot(left_error, right_error))
    difference = abs(left_value - right_value)
    if difference <= 2.0 * combined_error:
        return None
    return CrDissipationWarning(
        "target_rate_control_state_dependence",
        f"{quantity} differs between control-ground and control-excited protocols "
        f"by {difference:.3g} 1/ns (> 2 combined standard errors).",
        (quantity,),
    )


def _target_control_state_dependence_warnings(
    t1_a: DecayRateFit,
    t1_b: DecayRateFit,
    leak_a: ExchangeRateFit,
    leak_b: ExchangeRateFit,
) -> tuple[CrDissipationWarning, ...]:
    """Compare resolved target rates without changing their central estimates."""
    candidates = (
        _control_state_dependence_warning(
            "target_t1rho",
            t1_a.rate_per_ns,
            t1_a.rate_standard_error_per_ns,
            t1_a.status,
            t1_b.rate_per_ns,
            t1_b.rate_standard_error_per_ns,
            t1_b.status,
        ),
        _control_state_dependence_warning(
            "target_leakage",
            leak_a.leakage_rate_per_ns,
            leak_a.leakage_standard_error_per_ns,
            leak_a.leakage_status,
            leak_b.leakage_rate_per_ns,
            leak_b.leakage_standard_error_per_ns,
            leak_b.leakage_status,
        ),
        _control_state_dependence_warning(
            "target_seepage",
            leak_a.seepage_rate_per_ns,
            leak_a.seepage_standard_error_per_ns,
            leak_a.seepage_status,
            leak_b.seepage_rate_per_ns,
            leak_b.seepage_standard_error_per_ns,
            leak_b.seepage_status,
        ),
    )
    return tuple(item for item in candidates if item is not None)


def _lifetime(
    rate: float, error: float | None
) -> tuple[float, tuple[float, float] | None]:
    if not np.isfinite(rate) or rate < 0.0:
        return float("nan"), None
    if rate == 0.0:
        return float("inf"), None
    lifetime = 1.0 / rate
    if error is None or not np.isfinite(error):
        return lifetime, None
    low = 1.0 / (rate + error) if rate + error > 0.0 else float("inf")
    high = 1.0 / (rate - error) if rate > error else float("inf")
    return lifetime, (low, high)


def _estimate(
    value: float,
    error: float | None,
    idle_value: float,
    status: CrDissipationRateStatus,
    sources: tuple[str, ...],
    message: str | None = None,
) -> CrDissipationRateEstimate:
    lifetime, interval = _lifetime(value, error)
    idle_lifetime, _ = _lifetime(idle_value, None)
    return CrDissipationRateEstimate(
        value,
        error,
        lifetime,
        interval,
        idle_value,
        idle_lifetime,
        status,
        sources,
        message,
    )


def _pure_dephasing_component(
    transverse_rate: float,
    longitudinal_rate: float,
) -> float:
    """Return the nonnegative pure-dephasing part of a transverse rate."""
    return max(0.0, transverse_rate - 0.5 * longitudinal_rate)


def _control_standard_error(fit: ControlPopulationRateFit, name: str) -> float | None:
    if not fit.success or fit.statuses[name] != CrDissipationRateStatus.RESOLVED:
        return None
    index = fit.parameter_order.index(name)
    variance = fit.covariance[index, index]
    return (
        float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0.0 else None
    )


def _fit_idle_control_rates(
    measurements: CrDissipationMeasurements,
    descriptor: ZX90Descriptor,
    control_idle: IdleNoiseParameters,
    synthetic: Mapping[str, NDArray[np.float64]],
    *,
    covariance_rcond: float,
) -> dict[str, float]:
    """Fit the full four-rate model once, without selection or status logic."""
    observations, _ = _control_observations(
        measurements, covariance_rcond=covariance_rcond
    )
    blocks = tuple(
        replace(
            item.block,
            observed=synthetic[item.protocol][item.point_index, item.component_indices],
        )
        for item in observations
    )
    counts = measurements.repetition_counts
    initials = {
        protocol: synthetic[protocol][0] for protocol in (PROTOCOL_A, PROTOCOL_B)
    }

    def trajectories(parameters: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        rates: dict[str, float] = {
            name: float(value)
            for name, value in zip(_CONTROL_RATE_ORDER, parameters, strict=True)
        }
        return {
            protocol: control_population_trajectory(
                counts,
                initial,
                rates,
                cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
                blank_duration_ns=descriptor.echo_slot_duration_ns,
                idle_e_to_g_per_ns=control_idle.relaxation_rate_per_ns,
            )
            for protocol, initial in initials.items()
        }

    def block_predictions(parameters: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        predicted = trajectories(parameters)
        return [
            predicted[item.protocol][item.point_index, item.component_indices]
            for item in observations
        ]

    upper = _rate_upper_bound(getattr(measurements, PROTOCOL_A).cr_active_time_ns)
    initial = np.array(
        [control_idle.relaxation_rate_per_ns, 0.0, 0.0, 0.0], dtype=np.float64
    )
    candidate = fit_gls_candidate(
        name="idle_full_control",
        parameter_names=_CONTROL_RATE_ORDER,
        initial=initial,
        bounds=(np.zeros(4), np.full(4, upper)),
        residual=lambda parameters: whiten_predictions(
            block_predictions(parameters), blocks
        ),
        prediction=lambda parameters: np.concatenate(block_predictions(parameters)),
        n_observations=sum(block.rank for block in blocks),
    )
    if not candidate.success:
        # An exact boundary solution can have singular local covariance even
        # though its constrained central value is known from zero residual.
        if (
            np.linalg.norm(whiten_predictions(block_predictions(initial), blocks))
            <= 1e-10
        ):
            return {
                name: float(value)
                for name, value in zip(_CONTROL_RATE_ORDER, initial, strict=True)
            }
        return dict.fromkeys(_CONTROL_RATE_ORDER, float("nan"))
    return dict(zip(_CONTROL_RATE_ORDER, candidate.parameters, strict=True))


def _fit_idle_target_t1rho(
    synthetic: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    counts: NDArray[np.int64],
    descriptor: ZX90Descriptor,
    target_idle: IdleNoiseParameters,
    active_time_ns: NDArray[np.float64],
) -> float:
    """Apply the free constrained A/B T1rho estimator without a null model."""
    retained = (
        np.flatnonzero(
            np.isfinite(synthetic[1:])
            & np.isfinite(standard_errors[1:])
            & (standard_errors[1:] > 0.0)
        )
        + 1
    )
    blocks = scalar_residual_blocks(synthetic[retained], standard_errors[retained])

    def curve(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return target_t1rho_trajectory(
            counts,
            float(synthetic[0]),
            float(parameters[0]),
            float(parameters[1]),
            cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
            blank_duration_ns=descriptor.echo_slot_duration_ns,
            idle_transverse_rate_per_ns=target_idle.transverse_rate_per_ns,
        )

    candidate = fit_gls_candidate(
        name="idle_free_t1rho",
        parameter_names=("rate_per_ns", "x_infinity"),
        initial=np.array([target_idle.transverse_rate_per_ns, 0.0]),
        bounds=(
            np.array([0.0, -1.5]),
            np.array([_rate_upper_bound(active_time_ns), 1.5]),
        ),
        residual=lambda parameters: whiten_predictions(
            [np.array([curve(parameters)[index]]) for index in retained], blocks
        ),
        prediction=curve,
        n_observations=sum(block.rank for block in blocks),
    )
    if candidate.success:
        return float(candidate.parameters[0])
    initial = np.array([target_idle.transverse_rate_per_ns, 0.0])
    residual = whiten_predictions(
        [np.array([curve(initial)[index]]) for index in retained], blocks
    )
    return float(initial[0]) if np.linalg.norm(residual) <= 1e-10 else float("nan")


def _fit_idle_target_exchange(
    synthetic: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    counts: NDArray[np.int64],
    descriptor: ZX90Descriptor,
    active_time_ns: NDArray[np.float64],
) -> tuple[float, float]:
    """Apply the free constrained two-way exchange estimator without selection."""
    retained = (
        np.flatnonzero(
            np.isfinite(synthetic[1:])
            & np.isfinite(standard_errors[1:])
            & (standard_errors[1:] > 0.0)
        )
        + 1
    )
    blocks = scalar_residual_blocks(synthetic[retained], standard_errors[retained])

    def curve(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return target_exchange_trajectory(
            counts,
            float(synthetic[0]),
            float(parameters[0]),
            float(parameters[1]),
            cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
        )

    candidate = fit_gls_candidate(
        name="idle_free_exchange",
        parameter_names=("leakage_rate_per_ns", "seepage_rate_per_ns"),
        initial=np.full(2, 1e-12),
        bounds=(np.zeros(2), np.full(2, _rate_upper_bound(active_time_ns))),
        residual=lambda parameters: whiten_predictions(
            [np.array([curve(parameters)[index]]) for index in retained], blocks
        ),
        prediction=curve,
        n_observations=sum(block.rank for block in blocks),
    )
    if not candidate.success:
        boundary = np.zeros(2)
        residual = whiten_predictions(
            [np.array([curve(boundary)[index]]) for index in retained], blocks
        )
        if np.linalg.norm(residual) <= 1e-10:
            return 0.0, 0.0
        return float("nan"), float("nan")
    return float(candidate.parameters[0]), float(candidate.parameters[1])


def _extract_idle_equivalent_baseline(
    measurements: CrDissipationMeasurements,
    schedules: ProtocolSchedules,
    descriptor: ZX90Descriptor,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    *,
    covariance_rcond: float,
) -> _IdleEquivalentBaseline:
    """Generate and fit the non-recursive idle-only A/B/C/D baseline once."""
    counts = measurements.repetition_counts
    idle_control_rates: dict[str, float] = dict.fromkeys(_CONTROL_RATE_ORDER, 0.0)
    idle_control_rates["control_e_to_g"] = control_idle.relaxation_rate_per_ns
    synthetic_control: dict[str, NDArray[np.float64]] = {}
    synthetic_x: dict[str, NDArray[np.float64]] = {}
    synthetic_pf: dict[str, NDArray[np.float64]] = {}
    for protocol in (PROTOCOL_A, PROTOCOL_B):
        data = getattr(measurements, protocol).actual
        if data.control_gef is None or data.target_gef is None:
            raise ValueError(f"{protocol} is missing primary GEF data.")
        synthetic_control[protocol] = control_population_trajectory(
            counts,
            data.control_gef.population[0],
            idle_control_rates,
            cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
            blank_duration_ns=descriptor.echo_slot_duration_ns,
            idle_e_to_g_per_ns=control_idle.relaxation_rate_per_ns,
        )
        if data.target_x_comp is None:
            raise ValueError(f"{protocol} is missing target X data.")
        synthetic_x[protocol] = target_t1rho_trajectory(
            counts,
            float(data.target_x_comp[0]),
            target_idle.transverse_rate_per_ns,
            0.0,
            cr_lobe_duration_ns=descriptor.cr_lobe_duration_ns,
            blank_duration_ns=descriptor.echo_slot_duration_ns,
            idle_transverse_rate_per_ns=target_idle.transverse_rate_per_ns,
        )
        synthetic_pf[protocol] = np.full(counts.shape, data.target_gef.population[0, 2])

    equivalent = _fit_idle_control_rates(
        measurements,
        descriptor,
        control_idle,
        synthetic_control,
        covariance_rcond=covariance_rcond,
    )
    target_t1rho_by_protocol: dict[str, float] = {}
    target_exchange_by_protocol: dict[str, tuple[float, float]] = {}
    for protocol in (PROTOCOL_A, PROTOCOL_B):
        data = getattr(measurements, protocol).actual
        if data.target_x_comp_standard_error is None or data.target_gef is None:
            raise ValueError(f"{protocol} is missing target uncertainty data.")
        target_t1rho_by_protocol[protocol] = _fit_idle_target_t1rho(
            synthetic_x[protocol],
            data.target_x_comp_standard_error,
            counts,
            descriptor,
            target_idle,
            getattr(measurements, protocol).cr_active_time_ns,
        )
        target_exchange_by_protocol[protocol] = _fit_idle_target_exchange(
            synthetic_pf[protocol],
            data.target_gef.standard_error[:, 2],
            counts,
            descriptor,
            getattr(measurements, protocol).cr_active_time_ns,
        )
    equivalent["target_t1rho"] = float(
        np.mean(tuple(target_t1rho_by_protocol.values()))
    )
    equivalent["target_leakage"] = float(
        np.mean([value[0] for value in target_exchange_by_protocol.values()])
    )
    equivalent["target_seepage"] = float(
        np.mean([value[1] for value in target_exchange_by_protocol.values()])
    )
    baseline_rates = CrNoiseRates(
        control_e_to_g=equivalent["control_e_to_g"],
        control_g_to_e=equivalent["control_g_to_e"],
        control_e_to_f=equivalent["control_e_to_f"],
        control_f_to_e=equivalent["control_f_to_e"],
        target_t1rho=equivalent["target_t1rho"],
        target_leakage=equivalent["target_leakage"],
        target_seepage=equivalent["target_seepage"],
    )
    c_raw = simulate_repeated_observable(
        schedules.semantic_blocks[PROTOCOL_C],
        counts,
        state_density("+x", "+x"),
        X_CONTROL,
        control_idle=control_idle,
        target_idle=target_idle,
        cr_rates=None,
        include_leakage=False,
    )
    d_raw = simulate_repeated_observable(
        schedules.semantic_blocks[PROTOCOL_D],
        counts,
        state_density("g", "+y"),
        Y_TARGET,
        control_idle=control_idle,
        target_idle=target_idle,
        cr_rates=None,
        include_leakage=False,
    )
    c_data = getattr(measurements, PROTOCOL_C).actual
    d_data = getattr(measurements, PROTOCOL_D).actual
    if c_data.primary_standard_error is None or d_data.primary_standard_error is None:
        raise ValueError("C/D primary standard errors are required.")
    c_baseline_fit = fit_physical_dephasing(
        c_raw,
        c_data.primary_standard_error,
        counts,
        schedules.semantic_blocks[PROTOCOL_C],
        initial_density=state_density("+x", "+x"),
        observable=X_CONTROL,
        control_idle=control_idle,
        target_idle=target_idle,
        fixed_rates=baseline_rates,
        fitted_role="control",
        force_free=True,
    )
    d_baseline_fit = fit_physical_dephasing(
        d_raw,
        d_data.primary_standard_error,
        counts,
        schedules.semantic_blocks[PROTOCOL_D],
        initial_density=state_density("g", "+y"),
        observable=Y_TARGET,
        control_idle=control_idle,
        target_idle=target_idle,
        fixed_rates=baseline_rates,
        fitted_role="target",
        force_free=True,
    )
    c_phi = c_baseline_fit.rate_per_ns if c_baseline_fit.success else float("nan")
    d_phi = d_baseline_fit.rate_per_ns if d_baseline_fit.success else float("nan")
    equivalent["control_transverse"] = (
        0.5 * (equivalent["control_e_to_g"] + equivalent["control_g_to_e"]) + c_phi
    )
    equivalent["target_t2rho"] = 0.5 * equivalent["target_t1rho"] + d_phi
    predictions: dict[str, CrDissipationIdlePrediction] = {}
    for protocol in (PROTOCOL_A, PROTOCOL_B):
        predictions[protocol] = CrDissipationIdlePrediction(
            getattr(measurements, protocol).elapsed_time_ns,
            {
                "control_population": synthetic_control[protocol],
                "target_x_comp": synthetic_x[protocol],
                "target_pf": synthetic_pf[protocol],
            },
            dict(equivalent),
        )
    return _IdleEquivalentBaseline(
        predictions,
        equivalent,
        target_t1rho_by_protocol,
        {PROTOCOL_C: c_raw, PROTOCOL_D: d_raw},
    )


def _apply_idle_prediction_spam(
    baseline: _IdleEquivalentBaseline,
    measurements: CrDissipationMeasurements,
    c_fit: PhysicalForwardDephasingFit,
    d_fit: PhysicalForwardDephasingFit,
) -> dict[str, CrDissipationIdlePrediction]:
    predictions = dict(baseline.predictions)
    for protocol, actual_fit, phi_name in (
        (PROTOCOL_C, c_fit, "control_pure_dephasing"),
        (PROTOCOL_D, d_fit, "target_rotating_frame_pure_dephasing"),
    ):
        raw = baseline.raw_cd_predictions[protocol]
        plotted = (
            actual_fit.spam_amplitude * raw + actual_fit.spam_offset
            if actual_fit.success
            else raw
        )
        predictions[protocol] = CrDissipationIdlePrediction(
            getattr(measurements, protocol).elapsed_time_ns,
            {"primary_expectation": plotted},
            {
                **baseline.equivalent_rates,
                phi_name: (
                    _pure_dephasing_component(
                        baseline.equivalent_rates["control_transverse"],
                        baseline.equivalent_rates["control_e_to_g"]
                        + baseline.equivalent_rates["control_g_to_e"],
                    )
                    if protocol == PROTOCOL_C
                    else _pure_dephasing_component(
                        baseline.equivalent_rates["target_t2rho"],
                        baseline.equivalent_rates["target_t1rho"],
                    )
                ),
            },
        )
    return predictions


def _fidelity_limits(
    descriptor: ZX90Descriptor,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    nominal: Mapping[str, float],
    statuses: Mapping[str, CrDissipationRateStatus],
    primitive_covariance: NDArray[np.float64],
    primitive_order: Sequence[str],
) -> tuple[CrDissipationFidelityLimits, tuple[CrDissipationWarning, ...]]:
    operations = semantic_zx90(descriptor)
    ideal = noiseless_channel(operations)
    idle_channel = compose_channel(
        operations,
        control_idle=control_idle,
        target_idle=target_idle,
        cr_rates=None,
        include_leakage=False,
    )
    idle_value = leakage_aware_average_fidelity(idle_channel, ideal)[0]
    idle_estimate = CrDissipationFidelityEstimate(idle_value, None, True, None)
    coherence_dependencies = (
        "control_e_to_g",
        "control_g_to_e",
        "target_t1rho",
        "control_pure_dephasing",
        "target_rotating_frame_pure_dephasing",
    )
    dissipative_dependencies = (
        *coherence_dependencies,
        "control_e_to_f",
        "control_f_to_e",
        "target_leakage",
        "target_seepage",
    )
    fixed_statuses = {
        CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED,
        CrDissipationRateStatus.UNRESOLVED_ASSUMED_IDLE,
        CrDissipationRateStatus.CONSISTENT_WITH_ZERO,
        CrDissipationRateStatus.INCONSISTENT_RATE_DECOMPOSITION,
    }
    # Restrict fallback metadata to primitive simulator inputs.  A partially
    # unresolved primitive with conditional variance still participates in the
    # sigma-point vector and therefore is not described as fixed.
    fixed = tuple(
        name for name in primitive_order if statuses.get(name) in fixed_statuses
    )
    fallbacks = {name: nominal[name] for name in fixed if name in nominal}
    warnings: list[CrDissipationWarning] = []

    def unavailable(message: str) -> CrDissipationFidelityEstimate:
        return CrDissipationFidelityEstimate(float("nan"), None, False, None, message)

    def rates_from(mapping: Mapping[str, float]) -> CrNoiseRates:
        merged = dict(nominal)
        merged.update(mapping)
        return CrNoiseRates(
            control_e_to_g=merged["control_e_to_g"],
            control_g_to_e=merged["control_g_to_e"],
            control_e_to_f=merged["control_e_to_f"],
            control_f_to_e=merged["control_f_to_e"],
            control_pure_dephasing=merged["control_pure_dephasing"],
            target_t1rho=merged["target_t1rho"],
            target_leakage=merged["target_leakage"],
            target_seepage=merged["target_seepage"],
            target_rotating_frame_pure_dephasing=merged[
                "target_rotating_frame_pure_dephasing"
            ],
        )

    def evaluate(mapping: Mapping[str, float], include_leakage: bool) -> float:
        channel = compose_channel(
            operations,
            control_idle=control_idle,
            target_idle=target_idle,
            cr_rates=rates_from(mapping),
            include_leakage=include_leakage,
        )
        return leakage_aware_average_fidelity(channel, ideal)[0]

    def one_limit(
        dependencies: Sequence[str], include_leakage: bool
    ) -> CrDissipationFidelityEstimate:
        failed = [
            name
            for name in dependencies
            if statuses.get(name) == CrDissipationRateStatus.FIT_FAILED
        ]
        if failed:
            return unavailable("Required rates failed: " + ", ".join(failed))
        value = evaluate({}, include_leakage)
        uncertain_indices = [
            index
            for index, name in enumerate(primitive_order)
            if name in dependencies
            and name not in fixed
            and np.isfinite(primitive_covariance[index, index])
            and primitive_covariance[index, index] > 0.0
        ]
        if not uncertain_indices:
            return CrDissipationFidelityEstimate(
                value, 0.0, True, "conditional_statistical"
            )
        uncertain_names = [primitive_order[index] for index in uncertain_indices]
        covariance = primitive_covariance[np.ix_(uncertain_indices, uncertain_indices)]
        central = {name: nominal[name] for name in uncertain_names}
        positive = tuple(name for name in uncertain_names if central[name] > 0.0)
        standard_error, covariance_ok = propagate_fidelity_uncertainty(
            central,
            covariance,
            lambda rates: evaluate(rates, include_leakage),
            positive_parameters=positive,
            psd_relative_tolerance=PSD_RELATIVE_TOLERANCE,
        )
        if not covariance_ok:
            warnings.append(
                CrDissipationWarning(
                    "covariance_inconsistency",
                    "A materially non-PSD covariance prevented fidelity uncertainty propagation.",
                    ("cr_on_fidelity",),
                )
            )
        return CrDissipationFidelityEstimate(
            value,
            standard_error,
            True,
            "conditional_statistical" if covariance_ok else None,
            None
            if covariance_ok
            else "Central fidelity is available; uncertainty is not.",
        )

    coherence = one_limit(coherence_dependencies, False)
    dissipative = one_limit(dissipative_dependencies, True)
    return (
        CrDissipationFidelityLimits(
            idle_estimate, coherence, dissipative, fixed, fallbacks
        ),
        tuple(warnings),
    )


def analyze_cr_dissipation(
    measurements: CrDissipationMeasurements,
    descriptor: ZX90Descriptor,
    schedules: ProtocolSchedules,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    *,
    covariance_rcond: float,
    initial_warnings: Sequence[CrDissipationWarning] = (),
) -> CrDissipationAnalysis:
    """Run the complete conditional v11 analysis on processed measurements."""
    warnings = list(initial_warnings)
    idle_baseline = _extract_idle_equivalent_baseline(
        measurements,
        schedules,
        descriptor,
        control_idle,
        target_idle,
        covariance_rcond=covariance_rcond,
    )
    control_fit = fit_control_populations(
        measurements,
        descriptor,
        control_idle,
        covariance_rcond=covariance_rcond,
    )
    t1_a = fit_target_t1rho(
        measurements,
        PROTOCOL_A,
        descriptor,
        target_idle,
        idle_equivalent_rate_per_ns=idle_baseline.target_t1rho_by_protocol[PROTOCOL_A],
    )
    t1_b = fit_target_t1rho(
        measurements,
        PROTOCOL_B,
        descriptor,
        target_idle,
        idle_equivalent_rate_per_ns=idle_baseline.target_t1rho_by_protocol[PROTOCOL_B],
    )
    leak_a = fit_target_exchange(measurements, PROTOCOL_A, descriptor)
    leak_b = fit_target_exchange(measurements, PROTOCOL_B, descriptor)
    warnings.extend(
        _target_control_state_dependence_warnings(t1_a, t1_b, leak_a, leak_b)
    )
    t1_status = _aggregate_status(
        t1_a.status,
        t1_b.status,
        unresolved=CrDissipationRateStatus.UNRESOLVED_ASSUMED_IDLE,
    )
    target_t1rho, target_t1rho_error = _aggregate_value_error(
        t1_a.rate_per_ns,
        t1_a.rate_standard_error_per_ns,
        t1_b.rate_per_ns,
        t1_b.rate_standard_error_per_ns,
    )
    leak_status = _aggregate_status(
        leak_a.leakage_status,
        leak_b.leakage_status,
        unresolved=CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED,
    )
    seep_status = _aggregate_status(
        leak_a.seepage_status,
        leak_b.seepage_status,
        unresolved=CrDissipationRateStatus.NOMINAL_ZERO_UNRESOLVED,
    )
    target_leak, target_leak_error = _aggregate_value_error(
        leak_a.leakage_rate_per_ns,
        leak_a.leakage_standard_error_per_ns,
        leak_b.leakage_rate_per_ns,
        leak_b.leakage_standard_error_per_ns,
    )
    target_seep, target_seep_error = _aggregate_value_error(
        leak_a.seepage_rate_per_ns,
        leak_a.seepage_standard_error_per_ns,
        leak_b.seepage_rate_per_ns,
        leak_b.seepage_standard_error_per_ns,
    )
    nominal: dict[str, float] = {}
    statuses: dict[str, CrDissipationRateStatus] = {}
    if control_fit.success:
        nominal.update(control_fit.rates_per_ns)
        statuses.update(control_fit.statuses)
    else:
        statuses.update(control_fit.statuses)
    nominal.update(
        {
            "target_t1rho": target_t1rho,
            "target_leakage": target_leak,
            "target_seepage": target_seep,
        }
    )
    statuses.update(
        {
            "target_t1rho": t1_status,
            "target_leakage": leak_status,
            "target_seepage": seep_status,
        }
    )
    required_ab = (
        *_CONTROL_RATE_ORDER,
        "target_t1rho",
        "target_leakage",
        "target_seepage",
    )
    dependency_failure = any(
        statuses[name] == CrDissipationRateStatus.FIT_FAILED for name in required_ab
    )
    counts = measurements.repetition_counts
    c_data = getattr(measurements, PROTOCOL_C).actual
    d_data = getattr(measurements, PROTOCOL_D).actual
    if c_data.primary_expectation is None or c_data.primary_standard_error is None:
        raise ValueError("Protocol C primary data are required.")
    if d_data.primary_expectation is None or d_data.primary_standard_error is None:
        raise ValueError("Protocol D primary data are required.")
    if dependency_failure:
        message = "Skipped because at least one required A/B primitive rate failed."
        c_fit = _dependency_failed_physical_fit(message, counts.size)
        d_fit = _dependency_failed_physical_fit(message, counts.size)
    else:
        fixed = CrNoiseRates(
            control_e_to_g=nominal["control_e_to_g"],
            control_g_to_e=nominal["control_g_to_e"],
            control_e_to_f=nominal["control_e_to_f"],
            control_f_to_e=nominal["control_f_to_e"],
            target_t1rho=nominal["target_t1rho"],
            target_leakage=nominal["target_leakage"],
            target_seepage=nominal["target_seepage"],
        )
        c_fit = fit_physical_dephasing(
            c_data.primary_expectation,
            c_data.primary_standard_error,
            counts,
            schedules.semantic_blocks[PROTOCOL_C],
            initial_density=state_density("+x", "+x"),
            observable=X_CONTROL,
            control_idle=control_idle,
            target_idle=target_idle,
            fixed_rates=fixed,
            fitted_role="control",
        )
        d_fit = fit_physical_dephasing(
            d_data.primary_expectation,
            d_data.primary_standard_error,
            counts,
            schedules.semantic_blocks[PROTOCOL_D],
            initial_density=state_density("g", "+y"),
            observable=Y_TARGET,
            control_idle=control_idle,
            target_idle=target_idle,
            fixed_rates=fixed,
            fitted_role="target",
        )
    nominal["control_pure_dephasing"] = (
        c_fit.rate_per_ns if c_fit.success else float("nan")
    )
    nominal["target_rotating_frame_pure_dephasing"] = (
        d_fit.rate_per_ns if d_fit.success else float("nan")
    )
    statuses["control_pure_dephasing"] = c_fit.status
    statuses["target_rotating_frame_pure_dephasing"] = d_fit.status
    for fit, output in ((c_fit, "control_transverse"), (d_fit, "target_t2rho")):
        if fit.status == CrDissipationRateStatus.INCONSISTENT_RATE_DECOMPOSITION:
            warnings.append(
                CrDissipationWarning(
                    "inconsistent_rate_decomposition",
                    "A signed diagnostic fit required significantly negative pure dephasing; the physical projection uses zero.",
                    (output,),
                )
            )

    idle_predictions = _apply_idle_prediction_spam(
        idle_baseline, measurements, c_fit, d_fit
    )
    idle_equivalent = idle_baseline.equivalent_rates
    failed_fits = tuple(
        name
        for name, fit in (
            ("control_population_ab", control_fit),
            ("target_a_t1rho", t1_a),
            ("target_a_leakage", leak_a),
            ("target_b_t1rho", t1_b),
            ("target_b_leakage", leak_b),
            ("control_pure_dephasing_c", c_fit),
            ("target_rotating_frame_pure_dephasing_d", d_fit),
        )
        if not fit.success
    )
    if failed_fits:
        warnings.append(
            CrDissipationWarning(
                "fit_failed",
                "One or more fits failed; independent outputs were retained.",
                failed_fits,
            )
        )
    invalid_idle_equivalents = tuple(
        name for name, value in idle_equivalent.items() if not np.isfinite(value)
    )
    if invalid_idle_equivalents:
        warnings.append(
            CrDissipationWarning(
                "idle_equivalent_fit_failed",
                "One or more synthetic idle-equivalent rates were not identifiable.",
                invalid_idle_equivalents,
            )
        )
    primary_rates: dict[str, CrDissipationRateEstimate] = {}
    for name in _CONTROL_RATE_ORDER:
        value = nominal.get(name, float("nan"))
        primary_rates[name] = _estimate(
            value,
            _control_standard_error(control_fit, name),
            idle_equivalent[name],
            statuses[name],
            (PROTOCOL_A, PROTOCOL_B),
        )
    primary_rates["target_t1rho"] = _estimate(
        target_t1rho,
        target_t1rho_error,
        idle_equivalent["target_t1rho"],
        t1_status,
        (PROTOCOL_A, PROTOCOL_B),
    )
    primary_rates["target_leakage"] = _estimate(
        target_leak,
        target_leak_error,
        idle_equivalent["target_leakage"],
        leak_status,
        (PROTOCOL_A, PROTOCOL_B),
    )
    primary_rates["target_seepage"] = _estimate(
        target_seep,
        target_seep_error,
        idle_equivalent["target_seepage"],
        seep_status,
        (PROTOCOL_A, PROTOCOL_B),
    )
    control_dependencies = (
        statuses["control_e_to_g"],
        statuses["control_g_to_e"],
        c_fit.status,
    )
    target_dependencies = (t1_status, d_fit.status)
    control_transverse_status = _derived_status(control_dependencies)
    target_t2rho_status = _derived_status(target_dependencies)
    control_transverse = (
        0.5 * (nominal["control_e_to_g"] + nominal["control_g_to_e"])
        + nominal["control_pure_dephasing"]
        if control_transverse_status != CrDissipationRateStatus.FIT_FAILED
        else float("nan")
    )
    target_t2rho = (
        0.5 * nominal["target_t1rho"] + nominal["target_rotating_frame_pure_dephasing"]
        if target_t2rho_status != CrDissipationRateStatus.FIT_FAILED
        else float("nan")
    )
    nominal["control_transverse"] = control_transverse
    nominal["target_t2rho"] = target_t2rho
    statuses["control_transverse"] = control_transverse_status
    statuses["target_t2rho"] = target_t2rho_status
    control_variance = 0.0
    if control_fit.success:
        indices = [
            control_fit.parameter_order.index(name)
            for name in ("control_e_to_g", "control_g_to_e")
        ]
        control_variance = 0.25 * float(
            np.sum(control_fit.covariance[np.ix_(indices, indices)])
        )
    if c_fit.rate_standard_error_per_ns is not None:
        control_variance += c_fit.rate_standard_error_per_ns**2
    target_variance = 0.25 * (target_t1rho_error or 0.0) ** 2
    if d_fit.rate_standard_error_per_ns is not None:
        target_variance += d_fit.rate_standard_error_per_ns**2
    primary_rates["control_transverse"] = _estimate(
        control_transverse,
        np.sqrt(max(control_variance, 0.0))
        if np.isfinite(control_transverse)
        else None,
        idle_equivalent["control_transverse"],
        control_transverse_status,
        (PROTOCOL_A, PROTOCOL_B, PROTOCOL_C),
    )
    primary_rates["target_t2rho"] = _estimate(
        target_t2rho,
        np.sqrt(max(target_variance, 0.0)) if np.isfinite(target_t2rho) else None,
        idle_equivalent["target_t2rho"],
        target_t2rho_status,
        (PROTOCOL_A, PROTOCOL_B, PROTOCOL_D),
    )
    derived = {
        "control_pure_dephasing": _estimate(
            nominal["control_pure_dephasing"],
            c_fit.rate_standard_error_per_ns,
            _pure_dephasing_component(
                idle_equivalent["control_transverse"],
                idle_equivalent["control_e_to_g"] + idle_equivalent["control_g_to_e"],
            ),
            c_fit.status,
            (PROTOCOL_C,),
        ),
        "target_rotating_frame_pure_dephasing": _estimate(
            nominal["target_rotating_frame_pure_dephasing"],
            d_fit.rate_standard_error_per_ns,
            _pure_dephasing_component(
                idle_equivalent["target_t2rho"],
                idle_equivalent["target_t1rho"],
            ),
            d_fit.status,
            (PROTOCOL_D,),
        ),
    }
    primitive_order = (
        *_CONTROL_RATE_ORDER,
        "target_t1rho",
        "target_leakage",
        "target_seepage",
        "control_pure_dephasing",
        "target_rotating_frame_pure_dephasing",
    )
    primitive_covariance = np.zeros((len(primitive_order), len(primitive_order)))
    if control_fit.success:
        primitive_covariance[:4, :4] = control_fit.covariance
    if leak_a.success and leak_b.success:
        primitive_covariance[5:7, 5:7] = (leak_a.covariance + leak_b.covariance) / 4.0
    for index, error in (
        (4, target_t1rho_error),
        (5, target_leak_error),
        (6, target_seep_error),
        (7, c_fit.rate_standard_error_per_ns),
        (8, d_fit.rate_standard_error_per_ns),
    ):
        primitive_covariance[index, index] = 0.0 if error is None else error**2
    fidelity, fidelity_warnings = _fidelity_limits(
        descriptor,
        control_idle,
        target_idle,
        nominal,
        statuses,
        primitive_covariance,
        primitive_order,
    )
    warnings.extend(fidelity_warnings)
    fits = CrDissipationFits(control_fit, t1_a, leak_a, t1_b, leak_b, c_fit, d_fit)
    return CrDissipationAnalysis(
        rates=primary_rates,
        derived_rates=derived,
        nominal_rates_for_forward_model={
            name: nominal[name]
            for name in (
                *_CONTROL_RATE_ORDER,
                "target_t1rho",
                "target_leakage",
                "target_seepage",
                "control_pure_dephasing",
                "target_rotating_frame_pure_dephasing",
            )
            if np.isfinite(nominal.get(name, np.nan))
        },
        fidelity=fidelity,
        fits=fits,
        idle_predictions=idle_predictions,
        warnings=tuple(warnings),
        metadata={
            "uncertainty_conditioning": "idle_coherence_and_gef_calibration_fixed",
            "cross_protocol_covariance_approximation": "block_diagonal",
            "idle_equivalent_extraction": "non_recursive_synthetic_baseline",
            "cr_sign_dependent_dissipation": False,
        },
    )


def _derived_status(
    dependencies: Sequence[CrDissipationRateStatus],
) -> CrDissipationRateStatus:
    if CrDissipationRateStatus.FIT_FAILED in dependencies:
        return CrDissipationRateStatus.FIT_FAILED
    if all(status == CrDissipationRateStatus.RESOLVED for status in dependencies):
        return CrDissipationRateStatus.RESOLVED
    return CrDissipationRateStatus.PARTIALLY_UNRESOLVED
