"""Estimate transmon GEF populations from full second moments of IQ shots."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag

from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, PulseSchedule

_CALIBRATION_STEPS = {
    "c1": (),
    "c2": ("ef",),
    "c3": ("ge",),
    "c4": ("ge", "ef"),
    "c5": ("ef", "ge"),
    "c6": ("ge", "ef", "ge"),
}
_MEASUREMENT_STEPS = {
    "s1": (),
    "s4": ("ge", "ef"),
    "s5": ("ef", "ge"),
}
_FEATURE_COUNT = 5
_STATE_COUNT = 3
_DEFAULT_N_BOOTSTRAP = 1000
_DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
_MIN_BOOTSTRAP_SUCCESS_RATE = 0.8
_SIMPLEX_BOUNDARY_TOLERANCE = 1e-10


class _PopulationNotIdentifiableError(ValueError):
    """Indicate that calibrated features cannot identify GEF populations."""


@dataclass(frozen=True)
class IQMomentSummary:
    """
    Store the mean and covariance of the five IQ moment features.

    Attributes
    ----------
    mean
        Mean of `[I, Q, I², IQ, Q²]`, with shape `(5,)`.
    covariance
        Estimated covariance of `mean`, with shape `(5, 5)`.
    n_shots
        Number of single-shot IQ samples used in the estimate.
    """

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    n_shots: int

    def __post_init__(self) -> None:
        """Validate and normalize the stored arrays."""
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if mean.shape != (_FEATURE_COUNT,):
            raise ValueError("mean must have shape (5,).")
        if covariance.shape != (_FEATURE_COUNT, _FEATURE_COUNT):
            raise ValueError("covariance must have shape (5, 5).")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("IQ moment summary values must be finite.")
        if isinstance(self.n_shots, bool) or not isinstance(self.n_shots, Integral):
            raise TypeError("n_shots must be an integer.")
        if self.n_shots < 2:
            raise ValueError("n_shots must be at least two.")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric.")
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(np.min(eigenvalues)) < -1e-12 * eigenvalue_scale:
            raise ValueError("covariance must be positive semidefinite.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "n_shots", int(self.n_shots))


@dataclass(frozen=True)
class GefPopulationCalibration:
    """
    Store six-configuration calibration data for one qubit.

    Attributes
    ----------
    target
        Canonical qubit label.
    state_features
        Reconstructed GEF state features ordered as g, e, f, with shape `(3, 5)`.
    summaries
        Moment summaries keyed by `c1` through `c6`.
    raw_iq
        Single-shot complex IQ arrays keyed by calibration configuration.
    analyzer_duration
        Common duration of the padded calibration analyzers in ns.
    """

    target: str
    state_features: NDArray[np.float64]
    summaries: dict[str, IQMomentSummary]
    raw_iq: dict[str, NDArray[np.complex128]]
    analyzer_duration: float


@dataclass(frozen=True)
class GefPopulationFit:
    """
    Store a constrained GLS population estimate and diagnostics.

    Attributes
    ----------
    population
        Constrained population `[P_g, P_e, P_f]`.
    population_unconstrained
        GLS estimate before physical probability constraints are applied.
    objective
        GLS residual objective at the constrained solution.
    residual
        Residual vector ordered by the S1, S4, and S5 feature blocks.
    success
        Whether the constrained optimizer reported success.
    message
        Optimizer status message.
    design_rank
        Rank of the covariance-weighted design matrix.
    design_condition_number
        Condition number of the covariance-weighted design matrix.
    """

    population: NDArray[np.float64]
    population_unconstrained: NDArray[np.float64]
    objective: float
    residual: NDArray[np.float64]
    success: bool
    message: str
    design_rank: int
    design_condition_number: float


@dataclass(frozen=True)
class GefPopulationBootstrap:
    """
    Store bootstrap population samples and uncertainty diagnostics.

    Attributes
    ----------
    point_estimate
        Constrained estimate from the original IQ shots, ordered as g, e, f.
    unconstrained_point_estimate
        GLS estimate from the original shots before probability constraints.
    samples
        Constrained bootstrap estimates with shape `(n_resamples, 3)`. Failed
        fits are represented by rows of NaNs.
    unconstrained_samples
        Unconstrained bootstrap estimates with shape `(n_resamples, 3)`.
    standard_error
        Sample standard deviation of successful constrained estimates.
    confidence_interval
        Marginal percentile interval with lower and upper rows, shape `(2, 3)`.
    bias
        Bootstrap mean minus `point_estimate` for each state.
    boundary_fraction
        Fraction of successful estimates on the zero boundary for each state.
    requested_resamples
        Number of requested bootstrap resamples.
    successful_resamples
        Number of resamples with a successful fit.
    success_rate
        Successful fraction, or `None` when bootstrap is disabled.
    confidence_level
        Marginal percentile confidence level.
    minimum_success_rate
        Minimum successful fraction required for available uncertainty.
    seed
        Random seed used for resampling, or `None` for nondeterministic sampling.
    unavailable_reason
        Reason the uncertainty should not be used, or `None` when available.
    """

    point_estimate: NDArray[np.float64]
    unconstrained_point_estimate: NDArray[np.float64]
    samples: NDArray[np.float64]
    unconstrained_samples: NDArray[np.float64]
    standard_error: NDArray[np.float64] | None
    confidence_interval: NDArray[np.float64] | None
    bias: NDArray[np.float64] | None
    boundary_fraction: NDArray[np.float64] | None
    requested_resamples: int
    successful_resamples: int
    success_rate: float | None
    confidence_level: float
    minimum_success_rate: float
    seed: int | None
    unavailable_reason: str | None


def summarize_iq_shots(iq: ArrayLike) -> IQMomentSummary:
    """
    Compute full second-moment features from complex single-shot IQ data.

    Parameters
    ----------
    iq
        One-dimensional complex IQ array with at least two finite shots.

    Returns
    -------
    IQMomentSummary
        Mean features and the estimated covariance of their mean.

    Raises
    ------
    ValueError
        Raised when the input is not one-dimensional, contains fewer than two
        shots, or contains nonfinite values.
    """
    shots = _normalize_iq_shots(iq, name="iq")

    i_values = shots.real
    q_values = shots.imag
    features = np.column_stack(
        (
            i_values,
            q_values,
            i_values**2,
            i_values * q_values,
            q_values**2,
        )
    )
    mean = np.mean(features, axis=0)
    covariance = np.cov(features, rowvar=False, ddof=1) / shots.size
    return IQMomentSummary(
        mean=mean,
        covariance=covariance,
        n_shots=int(shots.size),
    )


def reconstruct_gef_state_features(
    calibration_summaries: Mapping[str, IQMomentSummary],
) -> NDArray[np.float64]:
    """
    Reconstruct state-specific IQ features from six calibration configurations.

    Parameters
    ----------
    calibration_summaries
        Moment summaries keyed by `c1` through `c6`.

    Returns
    -------
    NDArray[np.float64]
        State features ordered as g, e, f, with shape `(3, 5)`.

    Raises
    ------
    ValueError
        Raised when a required calibration configuration is missing.
    """
    _require_configurations(calibration_summaries, _CALIBRATION_STEPS)
    y_c1 = calibration_summaries["c1"].mean
    y_c2 = calibration_summaries["c2"].mean
    y_c3 = calibration_summaries["c3"].mean
    y_c4 = calibration_summaries["c4"].mean
    y_c5 = calibration_summaries["c5"].mean
    y_c6 = calibration_summaries["c6"].mean

    sum_ge = y_c1 + y_c3
    sum_gf = y_c2 + y_c4
    sum_ef = y_c5 + y_c6
    h_g = 0.5 * (sum_ge + sum_gf - sum_ef)
    h_e = 0.5 * (sum_ge + sum_ef - sum_gf)
    h_f = 0.5 * (sum_gf + sum_ef - sum_ge)
    return np.stack((h_g, h_e, h_f))


def fit_gef_population(
    summaries: Mapping[str, IQMomentSummary],
    state_features: ArrayLike,
    *,
    covariance_rcond: float = 1e-12,
) -> GefPopulationFit:
    """
    Fit a physical GEF population using constrained generalized least squares.

    Parameters
    ----------
    summaries
        Unknown-state moment summaries keyed by `s1`, `s4`, and `s5`.
    state_features
        Calibrated state features ordered as g, e, f, with shape `(3, 5)`.
    covariance_rcond
        Relative cutoff used by covariance pseudo-inverses. Must be in `[0, 1)`.

    Returns
    -------
    GefPopulationFit
        Constrained and unconstrained population estimates with fit diagnostics.

    Raises
    ------
    ValueError
        Raised for missing configurations, invalid state features, or an invalid
        pseudo-inverse cutoff.
    """
    _require_configurations(summaries, _MEASUREMENT_STEPS)
    calibrated_features = np.asarray(state_features, dtype=np.float64)
    if calibrated_features.shape != (_STATE_COUNT, _FEATURE_COUNT):
        raise ValueError("state_features must have shape (3, 5).")
    if not np.all(np.isfinite(calibrated_features)):
        raise ValueError("state_features must be finite.")
    covariance_rcond = _validate_covariance_rcond(covariance_rcond)

    h_g, h_e, h_f = calibrated_features
    design = np.block(
        [
            [h_g[:, None], h_e[:, None], h_f[:, None]],
            [h_f[:, None], h_g[:, None], h_e[:, None]],
            [h_e[:, None], h_f[:, None], h_g[:, None]],
        ]
    )
    observation = np.concatenate([summaries[name].mean for name in _MEASUREMENT_STEPS])
    covariance = block_diag(
        *(summaries[name].covariance for name in _MEASUREMENT_STEPS)
    )
    covariance = 0.5 * (covariance + covariance.T)

    feature_values = np.vstack(
        [
            calibrated_features,
            *(summaries[name].mean for name in _MEASUREMENT_STEPS),
        ]
    )
    feature_scale = np.max(np.abs(feature_values), axis=0)
    feature_scale = np.where(feature_scale > np.finfo(float).eps, feature_scale, 1.0)
    stacked_scale = np.tile(feature_scale, len(_MEASUREMENT_STEPS))
    scaled_design = design / stacked_scale[:, None]
    scaled_observation = observation / stacked_scale
    scaled_covariance = covariance / np.outer(stacked_scale, stacked_scale)
    weight = _covariance_precision(scaled_covariance, rcond=covariance_rcond)

    eigenvalues, eigenvectors = np.linalg.eigh(weight)
    sqrt_weight = (
        eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    ) @ eigenvectors.T
    weighted_design = sqrt_weight @ scaled_design
    simplex_tangent = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    if np.linalg.matrix_rank(weighted_design @ simplex_tangent) < 2:
        raise _PopulationNotIdentifiableError(
            "GEF populations are not identifiable from the calibrated features "
            "and measurement covariance."
        )

    normal_matrix = scaled_design.T @ weight @ scaled_design
    normal_vector = scaled_design.T @ weight @ scaled_observation
    population_unconstrained = (
        np.linalg.pinv(
            normal_matrix,
            rcond=covariance_rcond,
            hermitian=True,
        )
        @ normal_vector
    )

    def objective(population: NDArray[np.float64]) -> float:
        residual = scaled_observation - scaled_design @ population
        return float(residual @ weight @ residual)

    population = _solve_probability_simplex_gls(
        normal_matrix,
        normal_vector,
        objective=objective,
        rcond=covariance_rcond,
    )
    residual = observation - design @ population

    design_rank = int(np.linalg.matrix_rank(weighted_design))
    design_condition_number = (
        float(np.linalg.cond(weighted_design))
        if design_rank == _STATE_COUNT
        else float("inf")
    )

    return GefPopulationFit(
        population=population,
        population_unconstrained=np.asarray(
            population_unconstrained,
            dtype=np.float64,
        ),
        objective=objective(population),
        residual=residual,
        success=True,
        message="Solved by probability-simplex active-set enumeration.",
        design_rank=design_rank,
        design_condition_number=design_condition_number,
    )


def bootstrap_gef_populations(
    calibration: Mapping[str, GefPopulationCalibration],
    raw_iq: Mapping[str, Mapping[str, Mapping[str, ArrayLike]]],
    *,
    n_resamples: int = _DEFAULT_N_BOOTSTRAP,
    seed: int | None = 0,
    confidence_level: float = _DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
    covariance_rcond: float = 1e-12,
) -> dict[str, dict[str, GefPopulationBootstrap]]:
    """
    Estimate GEF population uncertainty by resampling raw IQ shots.

    Parameters
    ----------
    calibration
        Calibration data keyed by target. Every calibration must contain raw IQ
        shots for `c1` through `c6` when bootstrap is enabled.
    raw_iq
        Unknown-state IQ shots grouped by sequence, then `s1`/`s4`/`s5`
        configuration, then target.
    n_resamples
        Nonnegative number of bootstrap resamples. Defaults to 1000. Set to zero
        to return point estimates with disabled uncertainty.
    seed
        Nonnegative random seed. Defaults to zero for reproducibility. Pass
        `None` for nondeterministic resampling.
    confidence_level
        Marginal percentile confidence level strictly between zero and one.
        Defaults to 0.95.
    covariance_rcond
        Relative cutoff used by covariance pseudo-inverses. Must be in `[0, 1)`.

    Returns
    -------
    dict[str, dict[str, GefPopulationBootstrap]]
        Bootstrap results grouped by input sequence and target.

    Raises
    ------
    TypeError
        Raised when a bootstrap count, seed, or confidence level has an invalid
        type.
    ValueError
        Raised for invalid options, missing IQ configurations or targets,
        incompatible paired shot counts, or invalid IQ shots.

    Notes
    -----
    Each configuration is resampled independently. A calibration resample is
    shared by every input sequence, while simultaneously measured targets share
    shot indices within each configuration. This preserves the correlations
    needed for later sequence and target comparisons. Failed fits remain as NaN
    rows and are not retried. For multiple targets, each configuration must come
    from one simultaneous acquisition. The intervals describe finite-shot
    uncertainty under independent sampling; they do not include drift or model
    and pulse-calibration errors.
    """
    resolved_n_resamples = _validate_nonnegative_integer(
        n_resamples,
        name="n_resamples",
    )
    resolved_seed = _validate_optional_seed(seed, name="seed")
    resolved_confidence_level = _validate_open_unit_interval(
        confidence_level,
        name="confidence_level",
    )
    covariance_rcond = _validate_covariance_rcond(covariance_rcond)

    target_list = list(calibration)
    if not target_list:
        raise ValueError("calibration must contain at least one target.")
    calibration_by_target = _validate_calibration(target_list, calibration)
    measurement_iq = _normalize_bootstrap_measurement_iq(raw_iq, target_list)

    point_fits = {
        sequence_name: {
            target: fit_gef_population(
                {
                    name: summarize_iq_shots(configurations[name][target])
                    for name in _MEASUREMENT_STEPS
                },
                calibration_by_target[target].state_features,
                covariance_rcond=covariance_rcond,
            )
            for target in target_list
        }
        for sequence_name, configurations in measurement_iq.items()
    }
    samples = {
        sequence_name: {
            target: np.full(
                (resolved_n_resamples, _STATE_COUNT),
                np.nan,
                dtype=np.float64,
            )
            for target in target_list
        }
        for sequence_name in measurement_iq
    }
    unconstrained_samples = {
        sequence_name: {
            target: np.full(
                (resolved_n_resamples, _STATE_COUNT),
                np.nan,
                dtype=np.float64,
            )
            for target in target_list
        }
        for sequence_name in measurement_iq
    }

    if resolved_n_resamples > 0:
        calibration_iq = _normalize_bootstrap_calibration_iq(
            calibration_by_target,
            target_list,
        )
        generator = np.random.default_rng(resolved_seed)
        for resample_index in range(resolved_n_resamples):
            state_features = _resample_calibration_features(
                calibration_iq,
                target_list,
                generator,
            )
            for sequence_name, configurations in measurement_iq.items():
                resampled_summaries = _resample_measurement_summaries(
                    configurations,
                    target_list,
                    generator,
                )
                for target in target_list:
                    try:
                        fit = fit_gef_population(
                            {
                                name: resampled_summaries[name][target]
                                for name in _MEASUREMENT_STEPS
                            },
                            state_features[target],
                            covariance_rcond=covariance_rcond,
                        )
                    except (_PopulationNotIdentifiableError, np.linalg.LinAlgError):
                        continue
                    samples[sequence_name][target][resample_index] = fit.population
                    unconstrained_samples[sequence_name][target][resample_index] = (
                        fit.population_unconstrained
                    )

    return {
        sequence_name: {
            target: _summarize_population_bootstrap(
                point_fits[sequence_name][target],
                samples[sequence_name][target],
                unconstrained_samples[sequence_name][target],
                requested_resamples=resolved_n_resamples,
                confidence_level=resolved_confidence_level,
                seed=resolved_seed,
            )
            for target in target_list
        }
        for sequence_name in measurement_iq
    }


def calibrate_gef_population(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    n_shots: int | None = None,
    shot_interval: float | None = None,
) -> dict[str, GefPopulationCalibration]:
    """
    Measure six thermal-state permutations and calibrate GEF IQ features.

    Parameters
    ----------
    exp
        Experiment used for pulse generation and single-shot measurements.
    targets
        Qubit label or labels to calibrate simultaneously.
    n_shots
        Number of shots per calibration configuration. Defaults to
        `CALIBRATION_SHOTS`.
    shot_interval
        Interval between shots in ns. Defaults to `DEFAULT_INTERVAL`.

    Returns
    -------
    dict[str, GefPopulationCalibration]
        Calibration data keyed by canonical qubit label.

    Notes
    -----
    This function performs six hardware measurements. It assumes the initial
    population is `(p, q, 0)` and that the calibrated GE and EF pi pulses act as
    ideal population swaps.
    """
    target_list = _normalize_targets(exp, targets)
    resolved_n_shots = _resolve_shot_count(
        n_shots,
        name="n_shots",
        default=CALIBRATION_SHOTS,
    )
    resolved_shot_interval = _resolve_positive_real(
        shot_interval,
        name="shot_interval",
        default=DEFAULT_INTERVAL,
    )
    analyzers = _build_padded_analyzers(exp, target_list, _CALIBRATION_STEPS)
    raw_iq, summaries = _measure_configurations(
        exp,
        target_list,
        analyzers,
        n_shots=resolved_n_shots,
        shot_interval=resolved_shot_interval,
    )
    analyzer_duration = next(iter(analyzers.values())).duration
    return {
        target: GefPopulationCalibration(
            target=target,
            state_features=reconstruct_gef_state_features(
                {name: summaries[name][target] for name in _CALIBRATION_STEPS}
            ),
            summaries={name: summaries[name][target] for name in _CALIBRATION_STEPS},
            raw_iq={name: raw_iq[name][target] for name in _CALIBRATION_STEPS},
            analyzer_duration=analyzer_duration,
        )
        for target in target_list
    }


def measure_gef_populations(
    exp: Experiment,
    targets: Collection[str] | str,
    sequences: Mapping[str, PulseSchedule] | Sequence[PulseSchedule],
    *,
    calibration: Mapping[str, GefPopulationCalibration] | None = None,
    n_shots: int | None = None,
    calibration_n_shots: int | None = None,
    shot_interval: float | None = None,
    covariance_rcond: float = 1e-12,
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
    bootstrap_seed: int | None = 0,
    bootstrap_confidence_level: float = _DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
) -> Result:
    """
    Calibrate and estimate GEF populations after arbitrary pulse schedules.

    Parameters
    ----------
    exp
        Experiment used for pulse generation and single-shot measurements.
    targets
        Qubit label or labels whose marginal GEF populations are estimated.
    sequences
        Named mapping or ordered sequence of state-preparation schedules. Ordered
        inputs receive names such as `sequence_0`.
    calibration
        Optional prior calibration keyed by canonical qubit label. When omitted,
        six calibration configurations are measured before the input sequences.
    n_shots
        Number of shots per S1/S4/S5 configuration. Defaults to `DEFAULT_SHOTS`.
    calibration_n_shots
        Number of shots per calibration configuration. Defaults to
        `CALIBRATION_SHOTS` and is ignored when `calibration` is provided.
    shot_interval
        Interval between shots in ns. Defaults to `DEFAULT_INTERVAL`.
    covariance_rcond
        Relative cutoff used by covariance pseudo-inverses. Must be in `[0, 1)`.
    n_bootstrap
        Number of raw-shot bootstrap resamples. Defaults to 1000. Set to zero
        to disable bootstrap uncertainty estimation.
    bootstrap_seed
        Nonnegative bootstrap seed. Defaults to zero for reproducibility. Pass
        `None` for nondeterministic resampling.
    bootstrap_confidence_level
        Marginal percentile confidence level strictly between zero and one.
        Defaults to 0.95.

    Returns
    -------
    Result
        Populations, fit diagnostics, bootstrap uncertainty, calibration data,
        raw IQ shots, and moment summaries grouped by input sequence and target.

    Notes
    -----
    This function performs six calibration measurements when needed, followed by
    three hardware measurements per input sequence. The analyzer configurations
    are right-padded so readout starts at a common time within each group.
    Populations are modeled only in the g/e/f subspace and sum to one. The method
    assumes negligible initial f population during calibration, ideal GE/EF
    population swaps, and stable readout response throughout the run.
    """
    target_list = _normalize_targets(exp, targets)
    named_sequences = _normalize_sequences(sequences)
    resolved_n_shots = _resolve_shot_count(
        n_shots,
        name="n_shots",
        default=DEFAULT_SHOTS,
    )
    resolved_shot_interval = _resolve_positive_real(
        shot_interval,
        name="shot_interval",
        default=DEFAULT_INTERVAL,
    )
    covariance_rcond = _validate_covariance_rcond(covariance_rcond)
    resolved_n_bootstrap = _validate_nonnegative_integer(
        n_bootstrap,
        name="n_bootstrap",
    )
    resolved_bootstrap_seed = _validate_optional_seed(
        bootstrap_seed,
        name="bootstrap_seed",
    )
    resolved_bootstrap_confidence_level = _validate_open_unit_interval(
        bootstrap_confidence_level,
        name="bootstrap_confidence_level",
    )

    if calibration is None:
        resolved_calibration_n_shots = _resolve_shot_count(
            calibration_n_shots,
            name="calibration_n_shots",
            default=CALIBRATION_SHOTS,
        )
        calibration_by_target = calibrate_gef_population(
            exp,
            target_list,
            n_shots=resolved_calibration_n_shots,
            shot_interval=resolved_shot_interval,
        )
    else:
        calibration_by_target = _validate_calibration(
            target_list,
            calibration,
        )
    if resolved_n_bootstrap > 0:
        _normalize_bootstrap_calibration_iq(
            calibration_by_target,
            target_list,
        )

    analyzers = _build_padded_analyzers(exp, target_list, _MEASUREMENT_STEPS)
    populations: dict[str, dict[str, NDArray[np.float64]]] = {}
    fits: dict[str, dict[str, GefPopulationFit]] = {}
    raw_iq: dict[str, dict[str, dict[str, NDArray[np.complex128]]]] = {}
    all_summaries: dict[str, dict[str, dict[str, IQMomentSummary]]] = {}

    for sequence_name, preparation in named_sequences.items():
        schedules = {
            configuration: _append_analyzer(preparation, analyzer)
            for configuration, analyzer in analyzers.items()
        }
        sequence_raw_iq, sequence_summaries = _measure_configurations(
            exp,
            target_list,
            schedules,
            n_shots=resolved_n_shots,
            shot_interval=resolved_shot_interval,
        )
        sequence_fits = {
            target: fit_gef_population(
                {name: sequence_summaries[name][target] for name in _MEASUREMENT_STEPS},
                calibration_by_target[target].state_features,
                covariance_rcond=covariance_rcond,
            )
            for target in target_list
        }
        populations[sequence_name] = {
            target: sequence_fits[target].population for target in target_list
        }
        fits[sequence_name] = sequence_fits
        raw_iq[sequence_name] = sequence_raw_iq
        all_summaries[sequence_name] = sequence_summaries

    bootstrap = bootstrap_gef_populations(
        calibration_by_target,
        raw_iq,
        n_resamples=resolved_n_bootstrap,
        seed=resolved_bootstrap_seed,
        confidence_level=resolved_bootstrap_confidence_level,
        covariance_rcond=covariance_rcond,
    )

    return Result(
        data={
            "targets": tuple(target_list),
            "sequence_names": tuple(named_sequences),
            "state_order": ("g", "e", "f"),
            "calibration_configuration_order": tuple(_CALIBRATION_STEPS),
            "measurement_configuration_order": tuple(_MEASUREMENT_STEPS),
            "populations": populations,
            "fits": fits,
            "calibration": calibration_by_target,
            "raw_iq": raw_iq,
            "moment_summaries": all_summaries,
            "bootstrap": bootstrap,
            "measurement_options": {
                "n_shots": resolved_n_shots,
                "calibration_n_shots": (
                    None
                    if calibration is not None
                    else next(iter(calibration_by_target.values()))
                    .summaries["c1"]
                    .n_shots
                ),
                "shot_interval": resolved_shot_interval,
                "covariance_rcond": covariance_rcond,
                "n_bootstrap": resolved_n_bootstrap,
                "bootstrap_seed": resolved_bootstrap_seed,
                "bootstrap_confidence_level": resolved_bootstrap_confidence_level,
            },
        }
    )


def _require_configurations(
    values: Mapping[str, Any],
    configurations: Mapping[str, tuple[str, ...]],
) -> None:
    """Require all named measurement configurations."""
    missing = [name for name in configurations if name not in values]
    if missing:
        raise ValueError(f"Missing measurement configurations: {', '.join(missing)}.")


def _solve_probability_simplex_gls(
    normal_matrix: NDArray[np.float64],
    normal_vector: NDArray[np.float64],
    *,
    objective: Callable[[NDArray[np.float64]], float],
    rcond: float,
) -> NDArray[np.float64]:
    """Solve the three-state convex GLS problem by enumerating simplex faces."""
    candidates: list[NDArray[np.float64]] = []
    for active_count in range(1, _STATE_COUNT + 1):
        for active_tuple in combinations(range(_STATE_COUNT), active_count):
            active = np.asarray(active_tuple, dtype=np.int64)
            if active_count == 1:
                candidate = np.zeros(_STATE_COUNT, dtype=np.float64)
                candidate[active[0]] = 1.0
                candidates.append(candidate)
                continue

            active_normal = normal_matrix[np.ix_(active, active)]
            kkt = np.block(
                [
                    [active_normal, np.ones((active_count, 1))],
                    [np.ones((1, active_count)), np.zeros((1, 1))],
                ]
            )
            rhs = np.concatenate((normal_vector[active], [1.0]))
            solution = np.linalg.pinv(kkt, rcond=rcond) @ rhs
            active_population = solution[:active_count]
            if (
                np.all(active_population >= -1e-10)
                and abs(float(np.sum(active_population)) - 1.0) <= 1e-8
            ):
                candidate = np.zeros(_STATE_COUNT, dtype=np.float64)
                candidate[active] = np.clip(active_population, 0.0, None)
                candidate /= np.sum(candidate)
                candidates.append(candidate)

    return min(candidates, key=objective)


def _covariance_precision(
    covariance: NDArray[np.float64],
    *,
    rcond: float,
) -> NDArray[np.float64]:
    """Return a positive-semidefinite covariance pseudo-inverse."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(np.max(eigenvalues))
    if largest <= 0.0:
        return np.zeros_like(covariance)
    cutoff = rcond * largest
    inverse_eigenvalues = np.zeros_like(eigenvalues)
    retained = eigenvalues > cutoff
    inverse_eigenvalues[retained] = 1.0 / eigenvalues[retained]
    return np.asarray(
        (eigenvectors * inverse_eigenvalues) @ eigenvectors.T,
        dtype=np.float64,
    )


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str,
) -> list[str]:
    """Resolve a nonempty collection of unique canonical qubit labels."""
    requested = [targets] if isinstance(targets, str) else list(targets)
    if not requested:
        raise ValueError("targets must contain at least one qubit label.")
    normalized = [exp.ctx.resolve_qubit_label(target) for target in requested]
    if len(set(normalized)) != len(normalized):
        raise ValueError("targets must resolve to unique qubit labels.")
    return normalized


def _normalize_sequences(
    sequences: Mapping[str, PulseSchedule] | Sequence[PulseSchedule],
) -> dict[str, PulseSchedule]:
    """Normalize named or ordered state-preparation schedules."""
    if isinstance(sequences, Mapping):
        normalized = dict(sequences)
    else:
        normalized = {
            f"sequence_{index}": sequence for index, sequence in enumerate(sequences)
        }
    if not normalized:
        raise ValueError("sequences must contain at least one pulse schedule.")
    for name, sequence in normalized.items():
        if not isinstance(name, str) or not name:
            raise ValueError("sequence names must be nonempty strings.")
        if not isinstance(sequence, PulseSchedule):
            raise TypeError(f"Sequence `{name}` must be a PulseSchedule.")
        if not sequence.is_valid():
            raise ValueError(f"Sequence `{name}` is not a valid PulseSchedule.")
    return normalized


def _resolve_shot_count(value: int | None, *, name: str, default: int) -> int:
    """Resolve and validate a shot-count option."""
    resolved = default if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, Integral):
        raise TypeError(f"{name} must be an integer of at least two.")
    if resolved < 2:
        raise ValueError(f"{name} must be an integer of at least two.")
    return int(resolved)


def _resolve_positive_real(
    value: float | None,
    *,
    name: str,
    default: float,
) -> float:
    """Resolve and validate a positive finite real-valued option."""
    candidate = default if value is None else value
    if isinstance(candidate, bool) or not isinstance(candidate, Real):
        raise TypeError(f"{name} must be a real number.")
    resolved = float(candidate)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return resolved


def _validate_nonnegative_integer(value: object, *, name: str) -> int:
    """Validate and return a nonnegative integer option."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return resolved


def _validate_optional_seed(value: object, *, name: str) -> int | None:
    """Validate and return an optional nonnegative random seed."""
    if value is None:
        return None
    return _validate_nonnegative_integer(value, name=name)


def _validate_open_unit_interval(value: object, *, name: str) -> float:
    """Validate and return a real number strictly between zero and one."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number strictly between zero and one.")
    resolved = float(value)
    if not np.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ValueError(f"{name} must be strictly between zero and one.")
    return resolved


def _validate_covariance_rcond(value: object) -> float:
    """Validate the covariance pseudo-inverse cutoff."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("covariance_rcond must be a real number in [0, 1).")
    resolved = float(value)
    if not np.isfinite(resolved) or not 0.0 <= resolved < 1.0:
        raise ValueError("covariance_rcond must be in [0, 1).")
    return resolved


def _normalize_bootstrap_calibration_iq(
    calibration: Mapping[str, GefPopulationCalibration],
    targets: list[str],
) -> dict[str, dict[str, NDArray[np.complex128]]]:
    """Validate calibration IQ shots and group them by configuration."""
    grouped: dict[str, dict[str, NDArray[np.complex128]]] = {}
    for configuration in _CALIBRATION_STEPS:
        grouped[configuration] = {}
        for target in targets:
            target_iq = calibration[target].raw_iq
            if configuration not in target_iq:
                raise ValueError(
                    f"Calibration for `{target}` is missing `{configuration}` raw IQ."
                )
            grouped[configuration][target] = _normalize_iq_shots(
                target_iq[configuration],
                name=f"Calibration `{configuration}` IQ for `{target}`",
            )
        _require_paired_shot_counts(
            grouped[configuration],
            description=f"Calibration `{configuration}`",
        )
    return grouped


def _normalize_bootstrap_measurement_iq(
    raw_iq: Mapping[str, Mapping[str, Mapping[str, ArrayLike]]],
    targets: list[str],
) -> dict[str, dict[str, dict[str, NDArray[np.complex128]]]]:
    """Validate unknown-state bootstrap IQ shots."""
    if not raw_iq:
        raise ValueError("raw_iq must contain at least one input sequence.")
    normalized: dict[str, dict[str, dict[str, NDArray[np.complex128]]]] = {}
    for sequence_name, configurations in raw_iq.items():
        if not isinstance(sequence_name, str) or not sequence_name:
            raise ValueError("raw_iq sequence names must be nonempty strings.")
        _require_configurations(configurations, _MEASUREMENT_STEPS)
        normalized[sequence_name] = {}
        for configuration in _MEASUREMENT_STEPS:
            iq_by_target = configurations[configuration]
            missing = [target for target in targets if target not in iq_by_target]
            if missing:
                raise ValueError(
                    f"Measurement `{sequence_name}`/`{configuration}` is missing "
                    f"targets: {', '.join(missing)}."
                )
            normalized[sequence_name][configuration] = {
                target: _normalize_iq_shots(
                    iq_by_target[target],
                    name=(
                        f"Measurement `{sequence_name}`/`{configuration}` IQ for "
                        f"`{target}`"
                    ),
                )
                for target in targets
            }
            _require_paired_shot_counts(
                normalized[sequence_name][configuration],
                description=f"Measurement `{sequence_name}`/`{configuration}`",
            )
    return normalized


def _normalize_iq_shots(
    iq: ArrayLike,
    *,
    name: str,
) -> NDArray[np.complex128]:
    """Validate and normalize one single-shot IQ array."""
    shots = np.asarray(iq, dtype=np.complex128)
    if shots.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of complex shots.")
    if shots.size < 2:
        raise ValueError(f"{name} must contain at least two shots.")
    if not np.all(np.isfinite(shots)):
        raise ValueError(f"{name} shots must be finite.")
    return shots


def _require_paired_shot_counts(
    iq_by_target: Mapping[str, NDArray[np.complex128]],
    *,
    description: str,
) -> None:
    """Require equal shot counts for simultaneously measured targets."""
    shot_counts = {len(iq) for iq in iq_by_target.values()}
    if len(shot_counts) != 1:
        raise ValueError(
            f"{description} targets must have equal shot counts for paired resampling."
        )


def _resample_calibration_features(
    calibration_iq: Mapping[str, Mapping[str, NDArray[np.complex128]]],
    targets: list[str],
    generator: np.random.Generator,
) -> dict[str, NDArray[np.float64]]:
    """Resample all calibration configurations once for one replicate."""
    summaries: dict[str, dict[str, IQMomentSummary]] = {
        target: {} for target in targets
    }
    for configuration in _CALIBRATION_STEPS:
        iq_by_target = calibration_iq[configuration]
        n_shots = len(iq_by_target[targets[0]])
        indices = generator.integers(0, n_shots, size=n_shots, dtype=np.int64)
        for target in targets:
            summaries[target][configuration] = summarize_iq_shots(
                iq_by_target[target][indices]
            )
    return {
        target: reconstruct_gef_state_features(summaries[target]) for target in targets
    }


def _resample_measurement_summaries(
    configurations: Mapping[str, Mapping[str, NDArray[np.complex128]]],
    targets: list[str],
    generator: np.random.Generator,
) -> dict[str, dict[str, IQMomentSummary]]:
    """Resample one sequence while preserving simultaneous-target pairing."""
    summaries: dict[str, dict[str, IQMomentSummary]] = {}
    for configuration in _MEASUREMENT_STEPS:
        iq_by_target = configurations[configuration]
        n_shots = len(iq_by_target[targets[0]])
        indices = generator.integers(0, n_shots, size=n_shots, dtype=np.int64)
        summaries[configuration] = {
            target: summarize_iq_shots(iq_by_target[target][indices])
            for target in targets
        }
    return summaries


def _summarize_population_bootstrap(
    point_fit: GefPopulationFit,
    samples: NDArray[np.float64],
    unconstrained_samples: NDArray[np.float64],
    *,
    requested_resamples: int,
    confidence_level: float,
    seed: int | None,
) -> GefPopulationBootstrap:
    """Summarize successful bootstrap population estimates."""
    successful = np.all(np.isfinite(samples), axis=1) & np.all(
        np.isfinite(unconstrained_samples),
        axis=1,
    )
    successful_samples = samples[successful]
    n_successful = len(successful_samples)
    success_rate = (
        float(n_successful / requested_resamples) if requested_resamples > 0 else None
    )

    standard_error: NDArray[np.float64] | None = None
    confidence_interval: NDArray[np.float64] | None = None
    bias: NDArray[np.float64] | None = None
    boundary_fraction: NDArray[np.float64] | None = None
    unavailable_reason: str | None = None
    if requested_resamples == 0:
        unavailable_reason = "disabled"
    elif n_successful < 2:
        unavailable_reason = "fewer_than_two_successful_resamples"
    else:
        alpha = (1.0 - confidence_level) / 2.0
        standard_error = np.std(successful_samples, axis=0, ddof=1)
        confidence_interval = np.quantile(
            successful_samples,
            [alpha, 1.0 - alpha],
            axis=0,
        )
        bias = np.mean(successful_samples, axis=0) - point_fit.population
        boundary_fraction = np.mean(
            successful_samples <= _SIMPLEX_BOUNDARY_TOLERANCE,
            axis=0,
        )
        if success_rate is not None and success_rate < _MIN_BOOTSTRAP_SUCCESS_RATE:
            unavailable_reason = "success_rate_below_threshold"

    return GefPopulationBootstrap(
        point_estimate=point_fit.population,
        unconstrained_point_estimate=point_fit.population_unconstrained,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        bias=bias,
        boundary_fraction=boundary_fraction,
        requested_resamples=requested_resamples,
        successful_resamples=n_successful,
        success_rate=success_rate,
        confidence_level=confidence_level,
        minimum_success_rate=_MIN_BOOTSTRAP_SUCCESS_RATE,
        seed=seed,
        unavailable_reason=unavailable_reason,
    )


def _build_padded_analyzers(
    exp: Experiment,
    targets: list[str],
    configurations: Mapping[str, tuple[str, ...]],
) -> dict[str, PulseSchedule]:
    """Build analyzer schedules and right-pad them to a common duration."""
    schedules = {
        name: _build_analyzer(exp, targets, steps)
        for name, steps in configurations.items()
    }
    common_duration = max(schedule.duration for schedule in schedules.values())
    return {
        name: schedule.padded(common_duration, pad_side="right")
        for name, schedule in schedules.items()
    }


def _build_analyzer(
    exp: Experiment,
    targets: list[str],
    steps: tuple[str, ...],
) -> PulseSchedule:
    """Build simultaneous GE/EF population-permutation pulses."""
    ge_labels = {target: exp.ctx.resolve_ge_label(target) for target in targets}
    ef_labels = {target: exp.ctx.resolve_ef_label(target) for target in targets}
    with PulseSchedule() as schedule:
        if not steps:
            for target in targets:
                schedule.add(ge_labels[target], Blank(0))
        for transition in steps:
            labels = ge_labels if transition == "ge" else ef_labels
            for target in targets:
                label = labels[target]
                schedule.add(label, exp.pulse.x180(label))
            schedule.barrier()
    return schedule


def _append_analyzer(
    preparation: PulseSchedule,
    analyzer: PulseSchedule,
) -> PulseSchedule:
    """Append a copied analyzer to a copied state-preparation schedule."""
    with PulseSchedule() as schedule:
        schedule.call(preparation, copy=True)
        schedule.barrier()
        schedule.call(analyzer, copy=True)
    return schedule


def _measure_configurations(
    exp: Experiment,
    targets: list[str],
    schedules: Mapping[str, PulseSchedule],
    *,
    n_shots: int,
    shot_interval: float,
) -> tuple[
    dict[str, dict[str, NDArray[np.complex128]]],
    dict[str, dict[str, IQMomentSummary]],
]:
    """Measure configuration schedules and summarize target IQ shots."""
    raw_iq: dict[str, dict[str, NDArray[np.complex128]]] = {}
    summaries: dict[str, dict[str, IQMomentSummary]] = {}
    for name, schedule in schedules.items():
        result = exp.measurement_service.measure(
            sequence=schedule,
            mode="single",
            n_shots=n_shots,
            shot_interval=shot_interval,
            time_integration=True,
            state_classification=False,
            plot=False,
        )
        raw_iq[name] = {}
        summaries[name] = {}
        for target in targets:
            if target not in result.data:
                raise ValueError(
                    f"Measurement configuration `{name}` did not return `{target}`."
                )
            iq = np.asarray(result.data[target].kerneled, dtype=np.complex128)
            if iq.ndim == 0:
                iq = np.atleast_1d(iq)
            raw_iq[name][target] = iq
            summaries[name][target] = summarize_iq_shots(iq)
    return raw_iq, summaries


def _validate_calibration(
    targets: list[str],
    calibration: Mapping[str, GefPopulationCalibration],
) -> dict[str, GefPopulationCalibration]:
    """Validate reusable calibration data for all requested targets."""
    missing = [target for target in targets if target not in calibration]
    if missing:
        raise ValueError(f"Calibration is missing targets: {', '.join(missing)}.")
    selected = {target: calibration[target] for target in targets}
    for target, value in selected.items():
        if not isinstance(value, GefPopulationCalibration):
            raise TypeError(f"Calibration for `{target}` has an invalid type.")
        if value.target != target:
            raise ValueError(
                f"Calibration target `{value.target}` does not match `{target}`."
            )
        features = np.asarray(value.state_features)
        if features.shape != (_STATE_COUNT, _FEATURE_COUNT) or not np.all(
            np.isfinite(features)
        ):
            raise ValueError(
                f"Calibration state_features for `{target}` must be finite with shape (3, 5)."
            )
    return selected


__all__ = [
    "GefPopulationBootstrap",
    "GefPopulationCalibration",
    "GefPopulationFit",
    "IQMomentSummary",
    "bootstrap_gef_populations",
    "calibrate_gef_population",
    "fit_gef_population",
    "measure_gef_populations",
    "reconstruct_gef_state_features",
    "summarize_iq_shots",
]
