"""
Characterize measurement-induced state transitions from a readout cycle.

The module constructs repeated candidate and blank-reference schedules, estimates
GEF populations, fits a shared column-stochastic transition matrix, and returns
rank-aware local GLS uncertainty and objective residual diagnostics. Optional
profile-likelihood bounds and reduced-model comparisons are separate
postprocessing features.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import chi2

from qubex.analysis import fitting
from qubex.contrib.experiment.gef_population_estimation import measure_gef_populations
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.pulse import Blank, PulseSchedule

# Data records and public conventions

STATE_ORDER = ("g", "e", "f")
PROTOCOL_ORDER = ("initial_g", "initial_e", "initial_f")
TRANSITION_MATRIX_INDICES = (
    ("g_to_e", 1, 0),
    ("e_to_g", 0, 1),
    ("g_to_f", 2, 0),
    ("e_to_f", 2, 1),
    ("f_to_g", 0, 2),
    ("f_to_e", 1, 2),
)
TRANSITION_ORDER = tuple(name for name, _, _ in TRANSITION_MATRIX_INDICES)
_PHYSICAL_PARAMETER_ORDER = (
    *TRANSITION_ORDER,
    *(f"{protocol}_{state}" for protocol in PROTOCOL_ORDER for state in STATE_ORDER),
)
_DIAGNOSTIC_CAPTURE_INDEX = -1


@dataclass(frozen=True)
class GefPopulationFitDiagnostic:
    """Store numerical diagnostics for one GEF population estimate."""

    objective: float
    design_rank: int
    design_condition_number: float
    residual: NDArray[np.float64]
    message: str

    def __post_init__(self) -> None:
        """Normalize the residual and validate scalar diagnostics."""
        residual = np.asarray(self.residual, dtype=np.float64)
        if residual.ndim != 1 or not np.all(np.isfinite(residual)):
            raise ValueError("GEF fit residual must be a finite one-dimensional array.")
        if not np.isfinite(self.objective) or self.objective < 0.0:
            raise ValueError("GEF fit objective must be finite and nonnegative.")
        if self.design_rank < 0:
            raise ValueError("GEF fit design_rank must be nonnegative.")
        if np.isnan(self.design_condition_number) or self.design_condition_number < 0.0:
            raise ValueError(
                "GEF fit design_condition_number must be nonnegative and not NaN."
            )
        object.__setattr__(self, "residual", residual)


@dataclass(frozen=True)
class GefPopulationSeries:
    """Store one processed GEF population trajectory."""

    population: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    population_unconstrained: NDArray[np.float64]
    fit_diagnostics: tuple[GefPopulationFitDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Normalize arrays and require one consistent repetition grid."""
        population = np.asarray(self.population, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        standard_error = np.asarray(self.standard_error, dtype=np.float64)
        unconstrained = np.asarray(self.population_unconstrained, dtype=np.float64)
        if population.ndim != 2 or population.shape[1:] != (3,):
            raise ValueError("population must have shape (n_points, 3).")
        point_count = population.shape[0]
        if covariance.shape != (point_count, 3, 3):
            raise ValueError("covariance must have shape (n_points, 3, 3).")
        if standard_error.shape != population.shape:
            raise ValueError("standard_error must match population shape.")
        if unconstrained.shape != population.shape:
            raise ValueError("population_unconstrained must match population shape.")
        if not all(
            np.all(np.isfinite(values))
            for values in (population, covariance, standard_error, unconstrained)
        ):
            raise ValueError("GEF population series arrays must be finite.")
        if np.any(population < -1e-10) or not np.allclose(
            np.sum(population, axis=1),
            1.0,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("population rows must lie on the probability simplex.")
        if np.any(standard_error < 0.0):
            raise ValueError("standard_error must be nonnegative.")
        if self.fit_diagnostics and len(self.fit_diagnostics) != point_count:
            raise ValueError("fit_diagnostics must be empty or match n_points.")
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "population_unconstrained", unconstrained)


@dataclass(frozen=True)
class ReadoutMistConditionMeasurements:
    """Store actual and optional blank-reference data for one preparation."""

    actual: GefPopulationSeries
    reference: GefPopulationSeries | None


@dataclass(frozen=True)
class ReadoutMistMeasurements:
    """Store G/E/F trajectories on a shared repetition grid."""

    repetition_counts: NDArray[np.int64]
    protocols: dict[str, ReadoutMistConditionMeasurements]

    def __post_init__(self) -> None:
        """Require all three preparations on one shared repetition grid."""
        counts = np.asarray(self.repetition_counts)
        if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError(
                "repetition_counts must be a one-dimensional integer array."
            )
        if set(self.protocols) != set(PROTOCOL_ORDER):
            raise ValueError(
                "protocols must contain exactly initial_g, initial_e, initial_f."
            )
        if (
            counts.size < 2
            or counts[0] != 0
            or 1 not in counts
            or np.any(counts < 0)
            or np.any(np.diff(counts) <= 0)
        ):
            raise ValueError(
                "repetition_counts must be increasing, nonnegative, and contain 0 and 1."
            )
        reference_presence = tuple(
            self.protocols[protocol].reference is not None
            for protocol in PROTOCOL_ORDER
        )
        if any(reference_presence) and not all(reference_presence):
            raise ValueError(
                "Reference data must be present for all protocols or none."
            )
        for protocol in PROTOCOL_ORDER:
            measured = self.protocols[protocol]
            if measured.actual.population.shape[0] != counts.size:
                raise ValueError(
                    f"{protocol} actual data must match repetition_counts."
                )
            if (
                measured.reference is not None
                and measured.reference.population.shape[0] != counts.size
            ):
                raise ValueError(
                    f"{protocol} reference data must match repetition_counts."
                )
        object.__setattr__(
            self, "repetition_counts", counts.astype(np.int64, copy=False)
        )


@dataclass(frozen=True)
class ProbabilityEstimate:
    """
    Store one probability with local GLS uncertainty and an optional profile bound.

    `estimable` records whether the probability is insensitive to local Jacobian
    null directions. Profile bounds are advanced, one-sided diagnostics. All
    uncertainty is conditional on the fixed GEF calibration.
    """

    value: float
    standard_error: float | None
    estimable: bool
    upper_bound: float | None = None
    upper_bound_confidence_level: float | None = None
    upper_bound_unavailable_reason: str | None = None
    unit: str = "1/readout_cycle"


@dataclass(frozen=True)
class FitGoodnessOfFit:
    """
    Store objective fit diagnostics without classifying fit quality.

    Degrees of freedom are the number of whitened residual components minus the
    numerical Jacobian rank.
    `weighted_residual_rms` is `sqrt(chi_squared / degrees_of_freedom)`.
    """

    degrees_of_freedom: int
    chi_squared: float
    reduced_chi_squared: float
    weighted_residual_rms: float
    maximum_absolute_whitened_residual: float


@dataclass(frozen=True)
class ReadoutMistMatrixFit:
    """
    Store one full or reduced stochastic transition-matrix fit.

    `parameter_covariance` follows `parameter_order`. Rows and columns are NaN
    when that physical quantity is not locally estimable; `parameter_estimable`
    records the same decision explicitly.
    """

    success: bool
    transition_matrix: NDArray[np.float64]
    transition_probabilities: dict[str, ProbabilityEstimate]
    derived_metrics: dict[str, ProbabilityEstimate]
    initial_populations: dict[str, NDArray[np.float64]]
    fitted_populations: dict[str, NDArray[np.float64]]
    parameter_order: tuple[str, ...]
    parameter_covariance: NDArray[np.float64]
    goodness_of_fit: FitGoodnessOfFit
    whitened_residual: NDArray[np.float64]
    message: str
    model_name: str = "full"
    free_parameter_count: int = 12
    objective: float = float("nan")
    jacobian_rank: int = 0
    jacobian_nullity: int = 0
    jacobian_singular_values: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    parameter_estimable: dict[str, bool] = field(default_factory=dict)
    state_order: tuple[str, str, str] = STATE_ORDER
    probability_unit: str = "1/readout_cycle"


@dataclass(frozen=True)
class ReadoutMistModelCandidate:
    """Store one candidate model fit and its AICc comparison values."""

    fit: ReadoutMistMatrixFit
    aicc: float
    delta_aicc: float


@dataclass(frozen=True)
class ReadoutMistModelComparison:
    """Store candidate fits and AICc differences without selecting a model."""

    candidates: dict[str, ReadoutMistModelCandidate]


@dataclass(frozen=True)
class ReadoutMistAnalysis:
    """
    Store actual/reference fits and optional advanced diagnostics.

    `reference_difference` is the signed elementwise difference between the
    independently fitted actual and blank-reference transition probabilities.
    It is diagnostic only and is not a readout-only transition probability.
    """

    actual: ReadoutMistMatrixFit
    reference: ReadoutMistMatrixFit | None
    reference_difference: dict[str, float] | None
    actual_model_comparison: ReadoutMistModelComparison | None = None
    reference_model_comparison: ReadoutMistModelComparison | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Pulse schedule construction

AcquisitionOrder = Literal["interleaved", "randomized", "grouped"]


@dataclass(frozen=True)
class ReadoutMistSchedules:
    """Store schedules in semantic and submitted acquisition order."""

    by_condition: dict[str, tuple[PulseSchedule, ...]]
    preparations: dict[str, PulseSchedule]
    submitted: dict[str, PulseSchedule]
    names_by_condition: dict[str, tuple[str, ...]]
    acquisition_order: tuple[str, ...]
    cycle_duration_ns: float


def _finite_nonnegative(value: float, name: str) -> float:
    """Validate one finite nonnegative duration."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return normalized


def build_default_readout_cycle(
    exp: Experiment,
    target: str,
    *,
    cycle_gap_ns: float,
) -> PulseSchedule:
    """Build the configured readout drive followed by the complete trailing gap."""
    gap = _finite_nonnegative(cycle_gap_ns, "cycle_gap_ns")
    readout_label = exp.ctx.resolve_read_label(target)
    readout = exp.pulse.readout(target)
    with PulseSchedule([readout_label]) as cycle:
        cycle.add(readout_label, readout)
        if gap > 0.0:
            cycle.add(
                readout_label,
                Blank(gap, sampling_period=readout.sampling_period),
            )
    return cycle


def validate_readout_cycle(cycle: PulseSchedule) -> PulseSchedule:
    """Validate that an override is a finite positive complete cycle."""
    if not isinstance(cycle, PulseSchedule):
        raise TypeError("readout_cycle must be a PulseSchedule or None.")
    if not cycle.is_valid():
        raise ValueError("readout_cycle must be a valid PulseSchedule.")
    if not np.isfinite(cycle.duration) or cycle.duration <= 0.0:
        raise ValueError("readout_cycle must have positive finite duration.")
    return cycle.copy()


def build_reference_cycle(candidate: PulseSchedule) -> PulseSchedule:
    """Zero every waveform while retaining the complete cycle timing and metadata."""
    return candidate.scaled(0.0)


def _preparations(exp: Experiment, target: str) -> dict[str, PulseSchedule]:
    """Build G, E, and F preparations using calibrated GE/EF pi pulses."""
    ge_label = exp.ctx.resolve_ge_label(target)
    ef_label = exp.ctx.resolve_ef_label(target)
    preparations: dict[str, PulseSchedule] = {}
    with PulseSchedule([ge_label, ef_label]) as ground:
        pass
    preparations["initial_g"] = ground
    with PulseSchedule([ge_label, ef_label]) as excited:
        excited.add(ge_label, exp.pulse.x180(ge_label))
        excited.barrier()
    preparations["initial_e"] = excited
    with PulseSchedule([ge_label, ef_label]) as second_excited:
        second_excited.add(ge_label, exp.pulse.x180(ge_label))
        second_excited.barrier()
        second_excited.add(ef_label, exp.pulse.x180(ef_label))
        second_excited.barrier()
    preparations["initial_f"] = second_excited
    return preparations


def _logical_order(
    repetition_counts: Sequence[int],
    order: AcquisitionOrder,
    seed: int | None,
) -> list[tuple[str, int]]:
    """Return deterministic grouped/interleaved or seeded randomized points."""
    if order == "grouped":
        return [
            (protocol, index)
            for protocol in PROTOCOL_ORDER
            for index in range(len(repetition_counts))
        ]
    points = []
    for index in range(len(repetition_counts)):
        offset = index % len(PROTOCOL_ORDER)
        protocols = (*PROTOCOL_ORDER[offset:], *PROTOCOL_ORDER[:offset])
        points.extend((protocol, index) for protocol in protocols)
    if order == "randomized":
        generator = np.random.default_rng(seed)
        generator.shuffle(points)
    elif order != "interleaved":
        raise ValueError(
            "acquisition_order must be 'interleaved', 'randomized', or 'grouped'."
        )
    return points


def build_mist_schedules(
    exp: Experiment,
    target: str,
    cycle: PulseSchedule,
    repetition_counts: Sequence[int],
    *,
    prefix: str = "actual",
    acquisition_order: AcquisitionOrder = "interleaved",
    acquisition_seed: int | None = 0,
) -> ReadoutMistSchedules:
    """Build G/E/F schedules while preserving the trailing gap in every cycle."""
    preparations = _preparations(exp, target)
    by_condition: dict[str, tuple[PulseSchedule, ...]] = {}
    names_by_condition: dict[str, tuple[str, ...]] = {}
    lookup: dict[tuple[str, int], tuple[str, PulseSchedule]] = {}
    for protocol in PROTOCOL_ORDER:
        schedules = []
        names = []
        for index, count in enumerate(repetition_counts):
            with PulseSchedule() as schedule:
                schedule.call(preparations[protocol], copy=True)
                if count > 0:
                    schedule.barrier()
                    schedule.call(cycle.repeated(int(count)), copy=True)
            name = f"{prefix}/{protocol}/n={int(count)}"
            schedules.append(schedule)
            names.append(name)
            lookup[protocol, index] = (name, schedule)
        by_condition[protocol] = tuple(schedules)
        names_by_condition[protocol] = tuple(names)
    ordered_points = _logical_order(
        repetition_counts,
        acquisition_order,
        acquisition_seed,
    )
    submitted = {lookup[point][0]: lookup[point][1] for point in ordered_points}
    return ReadoutMistSchedules(
        by_condition=by_condition,
        preparations=preparations,
        submitted=submitted,
        names_by_condition=names_by_condition,
        acquisition_order=tuple(submitted),
        cycle_duration_ns=float(cycle.duration),
    )


def _interleave_acquisition_schedules(
    actual: ReadoutMistSchedules,
    reference: ReadoutMistSchedules | None,
) -> dict[str, PulseSchedule]:
    """Return one acquisition mapping with adjacent actual/reference pairs."""
    if reference is None:
        return dict(actual.submitted)
    if len(actual.submitted) != len(reference.submitted):
        raise ValueError("Actual and reference schedule counts must match.")

    submitted: dict[str, PulseSchedule] = {}
    for (actual_name, actual_schedule), (
        reference_name,
        reference_schedule,
    ) in zip(actual.submitted.items(), reference.submitted.items(), strict=True):
        if actual_name.removeprefix("actual/") != reference_name.removeprefix(
            "reference/"
        ):
            raise ValueError("Actual and reference acquisition orders must match.")
        submitted[actual_name] = actual_schedule
        submitted[reference_name] = reference_schedule
    return submitted


def _align_reference_acquisition_order(
    actual: ReadoutMistSchedules,
    reference: ReadoutMistSchedules,
) -> ReadoutMistSchedules:
    """Align reference points to the one logical order sampled for actual points."""
    reference_by_point = {
        name.removeprefix("reference/"): schedule
        for name, schedule in reference.submitted.items()
    }
    actual_points = tuple(name.removeprefix("actual/") for name in actual.submitted)
    if len(reference_by_point) != len(actual_points) or set(reference_by_point) != set(
        actual_points
    ):
        raise ValueError("Actual and reference schedule points must match.")
    submitted = {
        f"reference/{point}": reference_by_point[point] for point in actual_points
    }
    return replace(
        reference,
        submitted=submitted,
        acquisition_order=tuple(submitted),
    )


# Numerical fitting helpers

_EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class ResidualBlock:
    """Store one population observation and its covariance whitener."""

    observed: NDArray[np.float64]
    whitener: NDArray[np.float64]


@dataclass(frozen=True)
class FullFitResult:
    """
    Store one nonlinear GLS result.

    `success` records optimizer convergence. For a rank-deficient Jacobian,
    `covariance` is the pseudo-covariance on the identifiable subspace and
    `null_space` records the unresolved parameter directions.
    """

    success: bool
    parameters: NDArray[np.float64]
    covariance: NDArray[np.float64]
    whitened_residual: NDArray[np.float64]
    goodness_of_fit: FitGoodnessOfFit
    message: str
    objective: float
    jacobian_rank: int
    jacobian_nullity: int
    jacobian_singular_values: NDArray[np.float64]
    null_space: NDArray[np.float64]


def covariance_whitener(
    covariance: ArrayLike,
    *,
    rcond: float,
) -> NDArray[np.float64]:
    """Return a PSD pseudo-inverse square root for resolved covariance modes."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        raise ValueError("population covariance must be a finite 3x3 matrix.")
    scale = max(float(np.max(np.abs(cov))), _EPS)
    if not np.allclose(
        cov,
        cov.T,
        rtol=1e-7,
        atol=float(max(1e-15, scale * 1e-10)),
    ):
        raise ValueError("population covariance must be symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (cov + cov.T))
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), _EPS)
    cutoff = max(rcond, 0.0) * spectral_scale
    if float(np.min(eigenvalues)) < -max(cutoff, 1e-15 * spectral_scale):
        raise ValueError("population covariance must be positive semidefinite.")
    retained = eigenvalues > cutoff
    if not np.any(retained):
        raise ValueError("population covariance has no resolved positive mode.")
    return np.asarray(
        (eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])).T,
        dtype=np.float64,
    )


def whiten_predictions(
    predictions: Sequence[ArrayLike],
    blocks: Sequence[ResidualBlock],
) -> NDArray[np.float64]:
    """Concatenate covariance-whitened prediction residuals."""
    if len(predictions) != len(blocks):
        raise ValueError("predictions must match residual blocks.")
    residuals = []
    for prediction, block in zip(predictions, blocks, strict=True):
        predicted = np.asarray(prediction, dtype=np.float64)
        if predicted.shape != (3,):
            raise ValueError("each population prediction must have shape (3,).")
        residuals.append(block.whitener @ (predicted - block.observed))
    return np.concatenate(residuals)


def assess_goodness_of_fit(
    residual: ArrayLike,
    *,
    n_parameters: int,
) -> FitGoodnessOfFit:
    """Compute objective diagnostics from shot-noise-whitened residuals."""
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    degrees_of_freedom = int(values.size - n_parameters)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return FitGoodnessOfFit(
            degrees_of_freedom=degrees_of_freedom,
            chi_squared=float("nan"),
            reduced_chi_squared=float("nan"),
            weighted_residual_rms=float("nan"),
            maximum_absolute_whitened_residual=float("nan"),
        )
    chi_squared = float(values @ values)
    maximum = float(np.max(np.abs(values)))
    if degrees_of_freedom <= 0:
        return FitGoodnessOfFit(
            degrees_of_freedom=degrees_of_freedom,
            chi_squared=chi_squared,
            reduced_chi_squared=float("nan"),
            weighted_residual_rms=float("nan"),
            maximum_absolute_whitened_residual=maximum,
        )
    reduced = chi_squared / degrees_of_freedom
    return FitGoodnessOfFit(
        degrees_of_freedom=degrees_of_freedom,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced,
        weighted_residual_rms=float(np.sqrt(reduced)),
        maximum_absolute_whitened_residual=maximum,
    )


def _jacobian_diagnostics(
    jacobian: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    int,
    NDArray[np.float64],
    float,
]:
    """Return pseudo-covariance, singular spectrum, rank, null space, and cutoff."""
    if jacobian.ndim != 2:
        size = jacobian.shape[-1] if jacobian.ndim > 0 else 0
        return (
            np.full((size, size), np.nan),
            np.empty(0, dtype=np.float64),
            0,
            np.eye(size, dtype=np.float64),
            float("nan"),
        )
    parameter_count = jacobian.shape[1]
    if parameter_count == 0:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0,
            np.empty((0, 0), dtype=np.float64),
            0.0,
        )
    if not np.all(np.isfinite(jacobian)):
        return (
            np.full((parameter_count, parameter_count), np.nan),
            np.full(min(jacobian.shape), np.nan),
            0,
            np.eye(parameter_count, dtype=np.float64),
            float("nan"),
        )
    try:
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=True)
    except np.linalg.LinAlgError:
        return (
            np.full((parameter_count, parameter_count), np.nan),
            np.full(min(jacobian.shape), np.nan),
            0,
            np.eye(parameter_count, dtype=np.float64),
            float("nan"),
        )
    largest = float(np.max(singular_values)) if singular_values.size else 0.0
    tolerance = max(
        1e-12 * largest,
        _EPS * max(jacobian.shape) * largest,
    )
    rank = (
        int(np.count_nonzero(singular_values > tolerance))
        if np.all(np.isfinite(singular_values)) and largest > 0.0
        else 0
    )
    resolved_vt = vt[:rank]
    covariance = (
        (resolved_vt.T / np.square(singular_values[:rank])) @ resolved_vt
        if rank > 0
        else np.zeros((parameter_count, parameter_count), dtype=np.float64)
    )
    covariance = np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64)
    return (
        covariance,
        np.asarray(singular_values, dtype=np.float64),
        rank,
        np.asarray(vt[rank:], dtype=np.float64),
        tolerance,
    )


def _run_fit(
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initial: NDArray[np.float64],
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    *,
    loss: str,
) -> OptimizeResult | None:
    """Run one bounded least-squares attempt and normalize numerical failure."""
    try:
        return least_squares(
            residual,
            initial,
            bounds=bounds,
            loss=loss,
            f_scale=1.0,
            max_nfev=5000,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def fit_full_gls(
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initializations: Sequence[ArrayLike],
    *,
    parameter_count: int,
) -> FullFitResult:
    """
    Fit robust initializations followed by final ordinary GLS.

    A converged central estimate remains successful for a rank-deficient
    Jacobian. In that case, `covariance` contains the pseudo-covariance on the
    identifiable subspace; callers must use `null_space` when deciding whether
    a derived physical quantity has a meaningful local uncertainty. Residual
    degrees of freedom use the numerical Jacobian rank rather than the nominal
    parameter count.
    """
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, Integral):
        raise TypeError("parameter_count must be a positive integer.")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be a positive integer.")
    parameter_count = int(parameter_count)
    starts = tuple(np.asarray(value, dtype=np.float64) for value in initializations)
    if not starts:
        raise ValueError("initializations must contain at least one parameter vector.")
    if any(value.shape != (parameter_count,) for value in starts):
        raise ValueError("each initialization must match parameter_count.")
    if any(not np.all(np.isfinite(value)) for value in starts):
        raise ValueError("initializations must be finite.")
    lower = np.full(parameter_count, -30.0)
    upper = np.full(parameter_count, 30.0)
    solutions: list[tuple[float, OptimizeResult, NDArray[np.float64]]] = []
    for initial in starts:
        robust = _run_fit(residual, initial, (lower, upper), loss="soft_l1")
        final_initial = (
            np.asarray(robust.x, dtype=np.float64)
            if robust is not None and np.all(np.isfinite(robust.x))
            else initial
        )
        final = _run_fit(residual, final_initial, (lower, upper), loss="linear")
        if final is None or not final.success or not np.all(np.isfinite(final.x)):
            continue
        whitened = np.asarray(final.fun, dtype=np.float64).reshape(-1)
        if whitened.size and np.all(np.isfinite(whitened)):
            solutions.append((float(whitened @ whitened), final, whitened))
    if not solutions:
        empty = np.full(parameter_count, np.nan)
        goodness = assess_goodness_of_fit(np.empty(0), n_parameters=parameter_count)
        return FullFitResult(
            success=False,
            parameters=empty,
            covariance=np.full((parameter_count, parameter_count), np.nan),
            whitened_residual=np.empty(0),
            goodness_of_fit=goodness,
            message="All nonlinear GLS initializations failed.",
            objective=float("nan"),
            jacobian_rank=0,
            jacobian_nullity=parameter_count,
            jacobian_singular_values=np.empty(0, dtype=np.float64),
            null_space=np.eye(parameter_count, dtype=np.float64),
        )
    _, final, whitened = min(solutions, key=lambda item: item[0])
    jacobian = np.asarray(final.jac, dtype=np.float64)
    covariance, singular_values, rank, null_space, _ = _jacobian_diagnostics(jacobian)
    covariance_available = bool(
        np.all(np.isfinite(covariance)) and rank == parameter_count
    )
    return FullFitResult(
        success=True,
        parameters=np.asarray(final.x, dtype=np.float64),
        covariance=covariance,
        whitened_residual=whitened,
        goodness_of_fit=assess_goodness_of_fit(
            whitened,
            n_parameters=rank,
        ),
        message=(
            str(final.message)
            if covariance_available
            else (
                f"{final.message} Central parameters converged, but some parameters "
                "are not locally identifiable; the reported covariance is "
                "restricted to the identifiable parameter subspace."
            )
        ),
        objective=float(whitened @ whitened),
        jacobian_rank=rank,
        jacobian_nullity=parameter_count - rank,
        jacobian_singular_values=singular_values,
        null_space=null_space,
    )


# Transition-matrix analysis

_PARAMETER_COUNT = 12
_INITIAL_ANCHORS = {"initial_g": 0, "initial_e": 1, "initial_f": 2}
_ESTIMABILITY_RELATIVE_TOLERANCE = 1e-7


@dataclass(frozen=True)
class _TransitionModelSpec:
    """Define fixed off-diagonal transition probabilities for one fit model."""

    name: str
    fixed_probabilities: tuple[tuple[str, float], ...] = ()


_FULL_MODEL = _TransitionModelSpec("full")
_ADJACENT_ONLY_MODEL = _TransitionModelSpec(
    "adjacent_only",
    (("g_to_f", 0.0), ("f_to_g", 0.0)),
)
_NO_LEAKAGE_MODEL = _TransitionModelSpec(
    "no_leakage",
    (("g_to_f", 0.0), ("e_to_f", 0.0)),
)
_REDUCED_MODEL_SPECS = (_ADJACENT_ONLY_MODEL, _NO_LEAKAGE_MODEL)


def _transition_name(final_state: int, initial_state: int) -> str:
    """Return the public name for one off-diagonal transition."""
    return next(
        name
        for name, final, initial in TRANSITION_MATRIX_INDICES
        if final == final_state and initial == initial_state
    )


def _model_transition_parameter_count(spec: _TransitionModelSpec) -> int:
    """Return the number of free off-diagonal transition parameters."""
    return len(TRANSITION_ORDER) - len(spec.fixed_probabilities)


def _decode_model_parameters(
    parameters: NDArray[np.float64],
    spec: _TransitionModelSpec,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Decode one model parameter vector into a transition matrix and initials."""
    fixed = dict(spec.fixed_probabilities)
    if len(fixed) != len(spec.fixed_probabilities) or any(
        name not in TRANSITION_ORDER for name in fixed
    ):
        raise ValueError(f"Model `{spec.name}` contains invalid fixed transitions.")
    if any(
        not np.isfinite(value) or not 0.0 <= value < 1.0 for value in fixed.values()
    ):
        raise ValueError(
            f"Model `{spec.name}` fixed probabilities must be finite and in [0, 1)."
        )
    transition_parameter_count = _model_transition_parameter_count(spec)
    expected = transition_parameter_count + 2 * len(PROTOCOL_ORDER)
    if parameters.shape != (expected,):
        raise ValueError(f"Model `{spec.name}` requires {expected} parameters.")
    matrix = np.zeros((3, 3), dtype=np.float64)
    offset = 0
    for initial_state in range(3):
        destinations = tuple(index for index in range(3) if index != initial_state)
        fixed_in_column = {
            destination: fixed[_transition_name(destination, initial_state)]
            for destination in destinations
            if _transition_name(destination, initial_state) in fixed
        }
        fixed_sum = float(sum(fixed_in_column.values()))
        if fixed_sum >= 1.0:
            raise ValueError(
                f"Model `{spec.name}` fixes at least one column to probability >= 1."
            )
        remaining = 1.0 - fixed_sum
        free_destinations = tuple(
            destination
            for destination in destinations
            if destination not in fixed_in_column
        )
        logits = parameters[offset : offset + len(free_destinations)]
        offset += len(free_destinations)
        weights = np.exp(logits)
        denominator = 1.0 + float(np.sum(weights))
        matrix[initial_state, initial_state] = remaining / denominator
        for destination, probability in fixed_in_column.items():
            matrix[destination, initial_state] = probability
        for destination, weight in zip(free_destinations, weights, strict=True):
            matrix[destination, initial_state] = remaining * float(weight) / denominator
    initial = {
        protocol: _simplex_from_logits(
            parameters[offset + 2 * index : offset + 2 * index + 2],
            anchor=_INITIAL_ANCHORS[protocol],
        )
        for index, protocol in enumerate(PROTOCOL_ORDER)
    }
    return matrix, initial


def _encode_model_parameters(
    transition: NDArray[np.float64],
    initial: Mapping[str, NDArray[np.float64]],
    spec: _TransitionModelSpec,
) -> NDArray[np.float64]:
    """Encode physical starting values for a full, reduced, or profile model."""
    fixed = dict(spec.fixed_probabilities)
    values: list[float] = []
    for initial_state in range(3):
        destinations = tuple(index for index in range(3) if index != initial_state)
        fixed_sum = sum(
            fixed.get(_transition_name(destination, initial_state), 0.0)
            for destination in destinations
        )
        remaining = 1.0 - fixed_sum
        free_destinations = tuple(
            destination
            for destination in destinations
            if _transition_name(destination, initial_state) not in fixed
        )
        free_and_stay = np.array(
            [
                transition[destination, initial_state]
                for destination in free_destinations
            ]
            + [transition[initial_state, initial_state]],
            dtype=np.float64,
        )
        free_and_stay = np.clip(free_and_stay, 1e-12, None)
        free_and_stay *= remaining / np.sum(free_and_stay)
        stay = free_and_stay[-1]
        values.extend(np.log(free_and_stay[:-1] / stay))
    for protocol in PROTOCOL_ORDER:
        values.extend(
            _simplex_logits(initial[protocol], anchor=_INITIAL_ANCHORS[protocol])
        )
    return np.clip(np.asarray(values, dtype=np.float64), -25.0, 25.0)


def _validate_covariance_rcond(value: float, *, name: str) -> float:
    """Validate a relative covariance cutoff without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f"{name} must be in [0, 1).")
    return normalized


def _simplex_from_logits(
    logits: NDArray[np.float64],
    *,
    anchor: int,
) -> NDArray[np.float64]:
    """Map two log ratios to a three-component simplex with one anchor."""
    destinations = tuple(index for index in range(3) if index != anchor)
    weights = np.ones(3, dtype=np.float64)
    weights[list(destinations)] = np.exp(logits)
    return weights / np.sum(weights)


def _simplex_logits(
    population: NDArray[np.float64],
    *,
    anchor: int,
) -> NDArray[np.float64]:
    """Return clipped log ratios relative to one simplex anchor."""
    clipped = np.clip(np.asarray(population, dtype=np.float64), 1e-10, None)
    clipped /= np.sum(clipped)
    destinations = tuple(index for index in range(3) if index != anchor)
    return np.log(clipped[list(destinations)] / clipped[anchor])


def _model_predictions(
    parameters: NDArray[np.float64],
    repetition_counts: NDArray[np.int64],
    spec: _TransitionModelSpec,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate population trajectories for one transition-model specification."""
    transition, initial_populations = _decode_model_parameters(parameters, spec)
    propagators = {
        int(count): np.linalg.matrix_power(transition, int(count))
        for count in repetition_counts
    }
    return {
        protocol: np.stack(
            [
                propagators[int(count)] @ initial_populations[protocol]
                for count in repetition_counts
            ]
        )
        for protocol in PROTOCOL_ORDER
    }


def _project_simplex(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project one finite vector onto the probability simplex."""
    values = np.asarray(vector, dtype=np.float64)
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, values.size + 1)
    active = sorted_values - cumulative / indices > 0.0
    rho = int(np.nonzero(active)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    return np.maximum(values - threshold, 0.0)


def _transition_initial_guess(
    populations: Mapping[str, NDArray[np.float64]],
    repetition_counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Estimate a one-step matrix from measured zero/one trajectories."""
    n0 = int(np.nonzero(repetition_counts == 0)[0][0])
    n1 = int(np.nonzero(repetition_counts == 1)[0][0])
    p0 = np.column_stack([populations[name][n0] for name in PROTOCOL_ORDER])
    p1 = np.column_stack([populations[name][n1] for name in PROTOCOL_ORDER])
    transition = p1 @ np.linalg.pinv(p0)
    transition = np.column_stack(
        [_project_simplex(transition[:, index]) for index in range(3)]
    )
    values = []
    for initial_state in range(3):
        destinations = tuple(index for index in range(3) if index != initial_state)
        column = np.clip(transition[:, initial_state], 1e-10, None)
        values.extend(np.log(column[list(destinations)] / column[initial_state]))
    for protocol in PROTOCOL_ORDER:
        values.extend(
            _simplex_logits(
                populations[protocol][n0],
                anchor=_INITIAL_ANCHORS[protocol],
            )
        )
    return np.clip(np.asarray(values, dtype=np.float64), -25.0, 25.0)


def _identity_biased_guess(
    populations: Mapping[str, NDArray[np.float64]],
    repetition_counts: NDArray[np.int64],
    transition_probability: float,
) -> NDArray[np.float64]:
    """Build a stable alternative initialization around an identity map."""
    n0 = int(np.nonzero(repetition_counts == 0)[0][0])
    stay = 1.0 - 2.0 * transition_probability
    transition_logits = np.full(
        6,
        np.log(transition_probability / stay),
        dtype=np.float64,
    )
    initial_logits = np.concatenate(
        [
            _simplex_logits(
                populations[protocol][n0],
                anchor=_INITIAL_ANCHORS[protocol],
            )
            for protocol in PROTOCOL_ORDER
        ]
    )
    return np.concatenate((transition_logits, initial_logits))


def _model_physical_vector(
    parameters: NDArray[np.float64],
    spec: _TransitionModelSpec,
) -> NDArray[np.float64]:
    """Return public physical quantities for one model parameter vector."""
    matrix, initial_populations = _decode_model_parameters(parameters, spec)
    transitions = np.array(
        [
            matrix[final_state, initial_state]
            for _, final_state, initial_state in TRANSITION_MATRIX_INDICES
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        (
            transitions,
            *(initial_populations[name] for name in PROTOCOL_ORDER),
        )
    )


def _numerical_jacobian(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a centered finite-difference Jacobian."""
    baseline = np.asarray(function(parameters), dtype=np.float64)
    jacobian = np.empty((baseline.size, parameters.size), dtype=np.float64)
    for index in range(parameters.size):
        step = 1e-5 * max(1.0, abs(float(parameters[index])))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        jacobian[:, index] = (function(upper) - function(lower)) / (2.0 * step)
    return jacobian


def _estimate(
    value: float,
    variance: float,
    *,
    estimable: bool,
) -> ProbabilityEstimate:
    """Build a probability estimate with local GLS uncertainty."""
    standard_error = (
        float(np.sqrt(max(variance, 0.0)))
        if estimable and np.isfinite(variance)
        else None
    )
    return ProbabilityEstimate(
        value=float(value),
        standard_error=standard_error,
        estimable=estimable,
    )


def _derived_estimate(
    physical: NDArray[np.float64],
    parameter_covariance: NDArray[np.float64],
    physical_transform: NDArray[np.float64],
    null_space: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> ProbabilityEstimate:
    """Propagate uncertainty for an independently tested linear metric."""
    parameter_gradient = weights @ physical_transform[: len(TRANSITION_ORDER)]
    estimable = _gradient_is_estimable(parameter_gradient, null_space)
    return _estimate(
        float(weights @ physical[:6]),
        (
            float(parameter_gradient @ parameter_covariance @ parameter_gradient)
            if estimable
            else float("nan")
        ),
        estimable=estimable,
    )


def _gradient_is_estimable(
    gradient: NDArray[np.float64],
    null_space: NDArray[np.float64],
) -> bool:
    """Return whether a scalar quantity is insensitive to local null directions."""
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm == 0.0:
        return False
    if null_space.size == 0:
        return True
    unresolved = float(np.linalg.norm(null_space @ gradient))
    return unresolved <= _ESTIMABILITY_RELATIVE_TOLERANCE * gradient_norm


def _failed_fit(
    message: str,
    *,
    model_name: str = "full",
    free_parameter_count: int = _PARAMETER_COUNT,
) -> ReadoutMistMatrixFit:
    """Build a stable unavailable fit result."""
    estimates = {
        name: ProbabilityEstimate(
            value=float("nan"),
            standard_error=None,
            estimable=False,
        )
        for name in TRANSITION_ORDER
    }
    goodness = FitGoodnessOfFit(
        degrees_of_freedom=0,
        chi_squared=float("nan"),
        reduced_chi_squared=float("nan"),
        weighted_residual_rms=float("nan"),
        maximum_absolute_whitened_residual=float("nan"),
    )
    return ReadoutMistMatrixFit(
        success=False,
        transition_matrix=np.full((3, 3), np.nan),
        transition_probabilities=estimates,
        derived_metrics={},
        initial_populations={},
        fitted_populations={},
        parameter_order=_PHYSICAL_PARAMETER_ORDER,
        parameter_covariance=np.full(
            (len(_PHYSICAL_PARAMETER_ORDER), len(_PHYSICAL_PARAMETER_ORDER)),
            np.nan,
        ),
        goodness_of_fit=goodness,
        whitened_residual=np.empty(0),
        message=message,
        model_name=model_name,
        free_parameter_count=free_parameter_count,
        jacobian_nullity=free_parameter_count,
        parameter_estimable=dict.fromkeys(_PHYSICAL_PARAMETER_ORDER, False),
    )


def _condition_series_and_blocks(
    measurements: ReadoutMistMeasurements,
    *,
    reference: bool,
    covariance_rcond: float,
) -> tuple[dict[str, GefPopulationSeries], list[ResidualBlock]] | None:
    """Collect one condition's series and matching covariance whiteners."""
    series = {}
    for protocol in PROTOCOL_ORDER:
        condition = measurements.protocols[protocol]
        value = condition.reference if reference else condition.actual
        if value is None:
            return None
        series[protocol] = value
    blocks = [
        ResidualBlock(
            observed=series[protocol].population[index],
            whitener=covariance_whitener(
                series[protocol].covariance[index],
                rcond=covariance_rcond,
            ),
        )
        for protocol in PROTOCOL_ORDER
        for index in range(measurements.repetition_counts.size)
    ]
    return series, blocks


def _model_initializations(
    populations: Mapping[str, NDArray[np.float64]],
    repetition_counts: NDArray[np.int64],
    spec: _TransitionModelSpec,
    preferred: tuple[NDArray[np.float64], Mapping[str, NDArray[np.float64]]] | None,
) -> tuple[NDArray[np.float64], ...]:
    """Build physical-consistent starts for a full or constrained model."""
    full_starts = (
        _transition_initial_guess(populations, repetition_counts),
        _identity_biased_guess(populations, repetition_counts, 1e-3),
        _identity_biased_guess(populations, repetition_counts, 1e-2),
    )
    physical_starts: list[
        tuple[NDArray[np.float64], Mapping[str, NDArray[np.float64]]]
    ] = [
        _decode_model_parameters(parameters, _FULL_MODEL) for parameters in full_starts
    ]
    if preferred is not None:
        physical_starts.insert(0, preferred)
    return tuple(
        _encode_model_parameters(transition, initial, spec)
        for transition, initial in physical_starts
    )


def _fit_model_condition(
    measurements: ReadoutMistMeasurements,
    *,
    reference: bool,
    covariance_rcond: float,
    spec: _TransitionModelSpec,
    preferred: tuple[NDArray[np.float64], Mapping[str, NDArray[np.float64]]]
    | None = None,
) -> ReadoutMistMatrixFit:
    """Fit one condition with a full, reduced, or profile model."""
    prepared = _condition_series_and_blocks(
        measurements,
        reference=reference,
        covariance_rcond=covariance_rcond,
    )
    parameter_count = _model_transition_parameter_count(spec) + 2 * len(PROTOCOL_ORDER)
    if prepared is None:
        return _failed_fit(
            "Reference populations were not acquired.",
            model_name=spec.name,
            free_parameter_count=parameter_count,
        )
    series, blocks = prepared
    populations = {name: series[name].population for name in PROTOCOL_ORDER}

    def block_predictions(parameters: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        predicted = _model_predictions(parameters, measurements.repetition_counts, spec)
        return [
            predicted[protocol][index]
            for protocol in PROTOCOL_ORDER
            for index in range(measurements.repetition_counts.size)
        ]

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return whiten_predictions(block_predictions(parameters), blocks)

    initializations = _model_initializations(
        populations,
        measurements.repetition_counts,
        spec,
        preferred,
    )
    fitted = fit_full_gls(
        residual,
        initializations,
        parameter_count=parameter_count,
    )
    if not fitted.success:
        return _failed_fit(
            fitted.message,
            model_name=spec.name,
            free_parameter_count=parameter_count,
        )
    physical_function = lambda parameters: _model_physical_vector(parameters, spec)
    physical = physical_function(fitted.parameters)
    transform = _numerical_jacobian(physical_function, fitted.parameters)
    raw_covariance = transform @ fitted.covariance @ transform.T
    raw_covariance = 0.5 * (raw_covariance + raw_covariance.T)
    fixed = dict(spec.fixed_probabilities)
    estimable = {
        name: (
            name not in fixed
            and _gradient_is_estimable(transform[index], fitted.null_space)
        )
        for index, name in enumerate(_PHYSICAL_PARAMETER_ORDER)
    }
    covariance = raw_covariance.copy()
    unresolved_indices = [
        index
        for index, name in enumerate(_PHYSICAL_PARAMETER_ORDER)
        if not estimable[name]
    ]
    covariance[unresolved_indices, :] = np.nan
    covariance[:, unresolved_indices] = np.nan
    transition_estimates = {}
    for index, name in enumerate(TRANSITION_ORDER):
        if name in fixed:
            transition_estimates[name] = ProbabilityEstimate(
                value=float(physical[index]),
                standard_error=None,
                estimable=False,
            )
        else:
            transition_estimates[name] = _estimate(
                physical[index],
                raw_covariance[index, index] if estimable[name] else float("nan"),
                estimable=estimable[name],
            )
    transition_index = {name: index for index, name in enumerate(TRANSITION_ORDER)}

    def weights(**values: float) -> NDArray[np.float64]:
        """Build a linear derived-metric vector in public transition order."""
        result = np.zeros(len(TRANSITION_ORDER), dtype=np.float64)
        for name, value in values.items():
            result[transition_index[name]] = value
        return result

    derived_weights = {
        "leakage_from_g": weights(g_to_f=1.0),
        "leakage_from_e": weights(e_to_f=1.0),
        "average_computational_leakage": weights(g_to_f=0.5, e_to_f=0.5),
        "seepage_from_f": weights(f_to_g=1.0, f_to_e=1.0),
    }
    derived = {
        name: _derived_estimate(
            physical,
            fitted.covariance,
            transform,
            fitted.null_space,
            weights,
        )
        for name, weights in derived_weights.items()
    }
    transition_matrix, initial_populations = _decode_model_parameters(
        fitted.parameters, spec
    )
    predicted = _model_predictions(
        fitted.parameters, measurements.repetition_counts, spec
    )
    return ReadoutMistMatrixFit(
        success=True,
        transition_matrix=transition_matrix,
        transition_probabilities=transition_estimates,
        derived_metrics=derived,
        initial_populations=initial_populations,
        fitted_populations=predicted,
        parameter_order=_PHYSICAL_PARAMETER_ORDER,
        parameter_covariance=covariance,
        goodness_of_fit=fitted.goodness_of_fit,
        whitened_residual=fitted.whitened_residual,
        message=fitted.message,
        model_name=spec.name,
        free_parameter_count=parameter_count,
        objective=fitted.objective,
        jacobian_rank=fitted.jacobian_rank,
        jacobian_nullity=fitted.jacobian_nullity,
        jacobian_singular_values=fitted.jacobian_singular_values,
        parameter_estimable=estimable,
    )


def _fit_condition(
    measurements: ReadoutMistMeasurements,
    *,
    reference: bool,
    covariance_rcond: float,
) -> ReadoutMistMatrixFit:
    """Fit actual or reference trajectories with the full transition model."""
    return _fit_model_condition(
        measurements,
        reference=reference,
        covariance_rcond=covariance_rcond,
        spec=_FULL_MODEL,
    )


def _validate_profile_confidence_level(value: float) -> float:
    """Validate a one-sided profile-likelihood confidence level."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("profile_confidence_level must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.5 < normalized < 1.0:
        raise ValueError("profile_confidence_level must be strictly between 0.5 and 1.")
    return normalized


def _aicc(fit: ReadoutMistMatrixFit) -> float:
    """Return AICc for a Gaussian GLS fit, omitting model-independent constants."""
    observation_count = fit.whitened_residual.size
    parameter_count = fit.free_parameter_count
    if (
        not fit.success
        or not np.isfinite(fit.objective)
        or observation_count <= parameter_count + 1
    ):
        return float("inf")
    return float(
        fit.objective
        + 2.0 * parameter_count
        + 2.0
        * parameter_count
        * (parameter_count + 1)
        / (observation_count - parameter_count - 1)
    )


def _compare_transition_models(
    measurements: ReadoutMistMeasurements,
    *,
    reference: bool,
    covariance_rcond: float,
    full_fit: ReadoutMistMatrixFit,
) -> ReadoutMistModelComparison:
    """Compare the full fit with the predefined physically useful reductions."""
    preferred = (
        (
            full_fit.transition_matrix,
            full_fit.initial_populations,
        )
        if full_fit.success
        else None
    )
    fits = {"full": full_fit}
    for spec in _REDUCED_MODEL_SPECS:
        fits[spec.name] = _fit_model_condition(
            measurements,
            reference=reference,
            covariance_rcond=covariance_rcond,
            spec=spec,
            preferred=preferred,
        )
    scores = {name: _aicc(fit) for name, fit in fits.items()}
    finite_scores = [score for score in scores.values() if np.isfinite(score)]
    minimum = min(finite_scores) if finite_scores else float("inf")
    candidates = {
        name: ReadoutMistModelCandidate(
            fit=fit,
            aicc=scores[name],
            delta_aicc=(
                scores[name] - minimum if np.isfinite(minimum) else float("inf")
            ),
        )
        for name, fit in fits.items()
    }
    return ReadoutMistModelComparison(candidates=candidates)


def _profile_upper_bound(
    measurements: ReadoutMistMeasurements,
    *,
    reference: bool,
    covariance_rcond: float,
    full_fit: ReadoutMistMatrixFit,
    transition_name: str,
    confidence_level: float,
) -> tuple[float | None, str | None]:
    """Find a one-sided conditional profile-likelihood upper bound."""
    if not full_fit.success:
        return None, "central_fit_failed"
    estimate = full_fit.transition_probabilities[transition_name]
    lower = float(np.clip(estimate.value, 0.0, 1.0 - 1e-9))
    threshold = float(chi2.ppf(2.0 * confidence_level - 1.0, df=1))
    preferred = (full_fit.transition_matrix, full_fit.initial_populations)
    cache: dict[float, float] = {}

    def delta(probability: float) -> float:
        key = float(probability)
        if key not in cache:
            profile = _fit_model_condition(
                measurements,
                reference=reference,
                covariance_rcond=covariance_rcond,
                spec=_TransitionModelSpec(
                    f"profile_{transition_name}",
                    ((transition_name, key),),
                ),
                preferred=preferred,
            )
            cache[key] = (
                max(0.0, profile.objective - full_fit.objective)
                if profile.success and np.isfinite(profile.objective)
                else float("nan")
            )
        return cache[key]

    left = lower
    right = max(left + 1e-6, 2.0 * max(left, 1e-6))
    right = min(right, 1.0 - 1e-9)
    right_delta = delta(right)
    if not np.isfinite(right_delta):
        return None, "profile_fit_failed"
    while right_delta < threshold and right < 1.0 - 1e-9:
        left = right
        right = min(max(2.0 * right, right + 1e-4), 1.0 - 1e-9)
        right_delta = delta(right)
        if not np.isfinite(right_delta):
            return None, "profile_fit_failed"
    if right_delta < threshold:
        return None, "not_constrained_below_physical_limit"
    for _ in range(20):
        midpoint = 0.5 * (left + right)
        midpoint_delta = delta(midpoint)
        if not np.isfinite(midpoint_delta):
            return None, "profile_fit_failed"
        if midpoint_delta < threshold:
            left = midpoint
        else:
            right = midpoint
    return float(right), None


def _add_profile_upper_bounds(
    measurements: ReadoutMistMeasurements,
    *,
    reference: bool,
    covariance_rcond: float,
    fit: ReadoutMistMatrixFit,
    confidence_level: float,
) -> ReadoutMistMatrixFit:
    """Attach one-sided profile bounds to every transition probability."""
    estimates = dict(fit.transition_probabilities)
    for name, estimate in estimates.items():
        upper_bound, reason = _profile_upper_bound(
            measurements,
            reference=reference,
            covariance_rcond=covariance_rcond,
            full_fit=fit,
            transition_name=name,
            confidence_level=confidence_level,
        )
        estimates[name] = replace(
            estimate,
            upper_bound=upper_bound,
            upper_bound_confidence_level=(
                confidence_level if upper_bound is not None else None
            ),
            upper_bound_unavailable_reason=reason,
        )
    return replace(fit, transition_probabilities=estimates)


def analyze_readout_mist(
    measurements: ReadoutMistMeasurements,
    *,
    covariance_rcond: float = 1e-12,
    compute_profile_upper_bounds: bool = False,
    profile_confidence_level: float = 0.95,
    fit_reduced_models: bool = False,
) -> ReadoutMistAnalysis:
    """
    Fit full MIST dynamics with optional advanced postprocessing.

    Parameters
    ----------
    measurements : ReadoutMistMeasurements
        Actual and optional blank-reference GEF population trajectories.
    covariance_rcond : float, optional
        Relative covariance eigenvalue cutoff in `[0, 1)`.
    compute_profile_upper_bounds : bool, optional
        Whether to add one-sided profile-likelihood upper bounds to all six
        transition probabilities.
    profile_confidence_level : float, optional
        Confidence level strictly between 0.5 and 1 for profile bounds.
    fit_reduced_models : bool, optional
        Whether to fit the predefined `adjacent_only` and `no_leakage`
        candidates and report their AICc differences.

    Returns
    -------
    ReadoutMistAnalysis
        Full-model fits and any requested advanced postprocessing results.

    Notes
    -----
    Profile upper bounds and unscaled local covariance treat the supplied GEF
    calibration as fixed. Set the advanced options only when profile bounds or
    AICc candidate comparisons are needed. The primary `actual` and `reference`
    results always use the full six-transition model, and model selection is left
    to the caller.
    """
    for name, value in (
        ("compute_profile_upper_bounds", compute_profile_upper_bounds),
        ("fit_reduced_models", fit_reduced_models),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")
    confidence_level = _validate_profile_confidence_level(profile_confidence_level)
    covariance_rcond = _validate_covariance_rcond(
        covariance_rcond,
        name="covariance_rcond",
    )
    actual = _fit_condition(
        measurements,
        reference=False,
        covariance_rcond=covariance_rcond,
    )
    has_reference = measurements.protocols[PROTOCOL_ORDER[0]].reference is not None
    reference = (
        _fit_condition(
            measurements,
            reference=True,
            covariance_rcond=covariance_rcond,
        )
        if has_reference
        else None
    )
    if compute_profile_upper_bounds:
        actual = _add_profile_upper_bounds(
            measurements,
            reference=False,
            covariance_rcond=covariance_rcond,
            fit=actual,
            confidence_level=confidence_level,
        )
        if reference is not None:
            reference = _add_profile_upper_bounds(
                measurements,
                reference=True,
                covariance_rcond=covariance_rcond,
                fit=reference,
                confidence_level=confidence_level,
            )
    actual_comparison = (
        _compare_transition_models(
            measurements,
            reference=False,
            covariance_rcond=covariance_rcond,
            full_fit=actual,
        )
        if fit_reduced_models
        else None
    )
    reference_comparison = (
        _compare_transition_models(
            measurements,
            reference=True,
            covariance_rcond=covariance_rcond,
            full_fit=reference,
        )
        if fit_reduced_models and reference is not None
        else None
    )
    difference = (
        {
            name: actual.transition_probabilities[name].value
            - reference.transition_probabilities[name].value
            for name in TRANSITION_ORDER
        }
        if reference is not None and reference.success and actual.success
        else None
    )
    return ReadoutMistAnalysis(
        actual=actual,
        reference=reference,
        reference_difference=difference,
        actual_model_comparison=actual_comparison,
        reference_model_comparison=reference_comparison,
        metadata={
            "model": "discrete_time_column_stochastic_transition_matrix",
            "matrix_convention": "T[final_state, initial_state]",
            "state_order": STATE_ORDER,
            "probability_unit": "1/readout_cycle",
            "initial_condition_treatment": "latent_nuisance_with_n0_observation",
            "uncertainty": "unscaled_local_gls_conditional_on_gef_calibration",
            "reference_interpretation": (
                "duration_matched_blank_cycle_diagnostic" if has_reference else None
            ),
            "profile_upper_bounds": compute_profile_upper_bounds,
            "profile_confidence_level": (
                confidence_level if compute_profile_upper_bounds else None
            ),
            "reduced_model_comparison": fit_reduced_models,
        },
    )


# Acquisition, presentation, and public workflow

DEFAULT_REPETITION_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_GEF_MEASUREMENT_N_SHOTS = 4096
DEFAULT_GEF_CALIBRATION_N_SHOTS = 8192


def _validate_repetition_counts(values: Sequence[int]) -> tuple[int, ...]:
    """Validate the ordered MIST repetition grid."""
    resolved = tuple(values)
    if len(resolved) < 5:
        raise ValueError("repetition_counts must contain at least five points.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in resolved
    ):
        raise TypeError("repetition_counts must contain integers only.")
    normalized = tuple(int(value) for value in resolved)
    if normalized[0] != 0:
        raise ValueError("repetition_counts must start at zero.")
    if 1 not in normalized:
        raise ValueError("repetition_counts must contain one.")
    if any(right <= left for left, right in pairwise(normalized)):
        raise ValueError("repetition_counts must be unique and strictly increasing.")
    return normalized


def _positive_integer(value: int, name: str) -> int:
    """Validate a shot count without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 2:
        raise ValueError(f"{name} must be at least two.")
    return int(value)


def _optional_seed(value: int | None) -> int | None:
    """Validate an optional nonnegative acquisition seed."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("acquisition_seed must be a nonnegative integer or None.")
    if value < 0:
        raise ValueError("acquisition_seed must be nonnegative.")
    return int(value)


def _finite_positive(value: float, name: str) -> float:
    """Validate a finite positive real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return normalized


def _extract_series(
    result: Result,
    schedules: ReadoutMistSchedules,
    target: str,
    protocol: str,
) -> GefPopulationSeries:
    """Convert named GEF fits into one ordered population trajectory."""
    data = cast(Mapping[str, Any], result.data)
    fits = cast(Mapping[str, Mapping[str, Any]], data["fits"])
    selected = [fits[name][target] for name in schedules.names_by_condition[protocol]]
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


def _build_measurements(
    counts: tuple[int, ...],
    target: str,
    result: Result,
    actual_schedules: ReadoutMistSchedules,
    reference_schedules: ReadoutMistSchedules | None,
) -> ReadoutMistMeasurements:
    """Build fixed-order measurement records from GEF result mappings."""
    protocols = {
        protocol: ReadoutMistConditionMeasurements(
            actual=_extract_series(result, actual_schedules, target, protocol),
            reference=(
                None
                if reference_schedules is None
                else _extract_series(
                    result,
                    reference_schedules,
                    target,
                    protocol,
                )
            ),
        )
        for protocol in PROTOCOL_ORDER
    }
    return ReadoutMistMeasurements(
        repetition_counts=np.asarray(counts, dtype=np.int64),
        protocols=protocols,
    )


def _select_acquisition_data(
    data: Mapping[str, Any],
    key: str,
    schedules: ReadoutMistSchedules | None,
) -> dict[str, Any] | None:
    """Select one condition from a combined acquisition result."""
    if schedules is None:
        return None
    values = cast(Mapping[str, Any], data[key])
    return {name: values[name] for name in schedules.submitted}


def _fitted_populations_on_integer_grid(
    fit: ReadoutMistMatrixFit,
    point_count: int,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate fitted populations at every integer cycle from zero onward."""
    trajectories = {}
    for protocol in PROTOCOL_ORDER:
        values = np.empty((point_count, len(STATE_ORDER)), dtype=np.float64)
        values[0] = fit.initial_populations[protocol]
        for index in range(1, point_count):
            values[index] = fit.transition_matrix @ values[index - 1]
        trajectories[protocol] = values
    return trajectories


def _fit_diagnostic_text(fit: ReadoutMistMatrixFit) -> str:
    """Format objective fit metrics without assigning a quality category."""
    if not fit.success:
        return f"optimizer=failed, rank={fit.jacobian_rank}/{fit.free_parameter_count}"
    goodness = fit.goodness_of_fit
    return (
        f"optimizer=success, χ²/DoF={goodness.reduced_chi_squared:.3g}, "
        f"WRMS={goodness.weighted_residual_rms:.3g}, "
        f"max|r_w|={goodness.maximum_absolute_whitened_residual:.3g}, "
        f"rank={fit.jacobian_rank}/{fit.free_parameter_count}"
    )


def _population_figure(
    measurements: ReadoutMistMeasurements,
    analysis: ReadoutMistAnalysis,
) -> go.Figure:
    """Plot observations and integer-cycle fitted trajectories by preparation."""
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Initial g", "Initial e", "Initial f"),
        vertical_spacing=0.08,
    )
    colors = fitting.COLORS[:3]
    counts = measurements.repetition_counts
    fit_counts = np.arange(int(counts[-1]) + 1, dtype=np.int64)
    actual_fit = (
        _fitted_populations_on_integer_grid(analysis.actual, fit_counts.size)
        if analysis.actual.success
        else {}
    )
    reference_fit = (
        _fitted_populations_on_integer_grid(analysis.reference, fit_counts.size)
        if analysis.reference is not None and analysis.reference.success
        else {}
    )
    for row, protocol in enumerate(PROTOCOL_ORDER, start=1):
        measured = measurements.protocols[protocol]
        for state_index, state in enumerate(STATE_ORDER):
            figure.add_trace(
                go.Scatter(
                    x=counts,
                    y=measured.actual.population[:, state_index],
                    error_y={
                        "type": "data",
                        "array": measured.actual.standard_error[:, state_index],
                        "visible": True,
                    },
                    mode="markers",
                    marker={"color": colors[state_index]},
                    name=f"actual P{state}",
                    legendgroup=f"actual-{state}",
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
            if analysis.actual.success:
                figure.add_trace(
                    go.Scatter(
                        x=fit_counts,
                        y=actual_fit[protocol][:, state_index],
                        mode="lines",
                        line={"color": colors[state_index]},
                        name=f"fit P{state}",
                        legendgroup=f"fit-{state}",
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )
            if measured.reference is not None:
                figure.add_trace(
                    go.Scatter(
                        x=counts,
                        y=measured.reference.population[:, state_index],
                        error_y={
                            "type": "data",
                            "array": measured.reference.standard_error[:, state_index],
                            "visible": True,
                        },
                        mode="markers",
                        marker={
                            "color": colors[state_index],
                            "symbol": "circle-open",
                        },
                        name=f"blank P{state}",
                        legendgroup=f"blank-{state}",
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )
                if analysis.reference is not None and analysis.reference.success:
                    figure.add_trace(
                        go.Scatter(
                            x=fit_counts,
                            y=reference_fit[protocol][:, state_index],
                            mode="lines",
                            line={"color": colors[state_index], "dash": "dot"},
                            name=f"blank fit P{state}",
                            legendgroup=f"blank-fit-{state}",
                            showlegend=row == 1,
                        ),
                        row=row,
                        col=1,
                    )
        figure.update_yaxes(
            title_text="GEF population", range=[-0.03, 1.03], row=row, col=1
        )
    figure.update_xaxes(title_text="Readout repetitions (n)", row=3, col=1)
    figure.update_layout(
        title=(
            "Readout-induced state transitions"
            f"<br><sup>Actual: {_fit_diagnostic_text(analysis.actual)}"
            + (
                f" | Reference: {_fit_diagnostic_text(analysis.reference)}"
                if analysis.reference is not None
                else ""
            )
            + "</sup>"
        ),
        template="plotly_white",
        height=850,
    )
    return figure


def _transition_matrix_standard_error(
    fit: ReadoutMistMatrixFit,
) -> NDArray[np.float64]:
    """Propagate fitted transition uncertainty to every matrix element."""
    errors = np.full((3, 3), np.nan, dtype=np.float64)
    indices = {name: index for index, name in enumerate(TRANSITION_ORDER)}
    outgoing: dict[int, tuple[str, str]] = {
        0: ("g_to_e", "g_to_f"),
        1: ("e_to_g", "e_to_f"),
        2: ("f_to_g", "f_to_e"),
    }
    for name, final_state, initial_state in TRANSITION_MATRIX_INDICES:
        standard_error = fit.transition_probabilities[name].standard_error
        if standard_error is not None:
            errors[final_state, initial_state] = standard_error
    for initial_state, names in outgoing.items():
        selected = [indices[name] for name in names]
        covariance = fit.parameter_covariance[np.ix_(selected, selected)]
        if np.all(np.isfinite(covariance)):
            variance = float(np.sum(covariance))
            errors[initial_state, initial_state] = np.sqrt(max(variance, 0.0))
    return errors


def _transition_matrix_text(fit: ReadoutMistMatrixFit) -> NDArray[np.str_]:
    """Format transition probabilities and available standard errors."""
    errors = _transition_matrix_standard_error(fit)
    return np.asarray(
        [
            [
                (
                    f"{value:.5f}<br>± {errors[row, column]:.2g}"
                    if np.isfinite(errors[row, column])
                    else f"{value:.5f}"
                )
                for column, value in enumerate(matrix_row)
            ]
            for row, matrix_row in enumerate(fit.transition_matrix)
        ],
        dtype=str,
    )


def _matrix_figure(analysis: ReadoutMistAnalysis) -> go.Figure:
    """Compare actual/reference matrices on a low-probability-focused scale."""
    conditions = [
        (
            f"Actual<br><sup>{_fit_diagnostic_text(analysis.actual)}</sup>",
            analysis.actual,
        )
    ]
    if analysis.reference is not None:
        conditions.append(
            (
                f"Reference<br><sup>{_fit_diagnostic_text(analysis.reference)}</sup>",
                analysis.reference,
            )
        )
    figure = make_subplots(
        rows=1,
        cols=len(conditions),
        subplot_titles=tuple(label for label, _ in conditions),
        horizontal_spacing=0.12,
    )
    for column, (_, fit) in enumerate(conditions, start=1):
        if fit.success:
            figure.add_trace(
                go.Heatmap(
                    z=fit.transition_matrix,
                    x=STATE_ORDER,
                    y=STATE_ORDER,
                    text=_transition_matrix_text(fit),
                    texttemplate="%{text}",
                    coloraxis="coloraxis",
                    hovertemplate=(
                        "Final %{y}<br>Initial %{x}<br>%{text}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        else:
            axis_suffix = "" if column == 1 else str(column)
            figure.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{axis_suffix} domain",
                yref=f"y{axis_suffix} domain",
                text="Fit unavailable",
                showarrow=False,
            )
        figure.update_xaxes(title_text="Initial state", row=1, col=column)
        figure.update_yaxes(autorange="reversed", row=1, col=column)
    figure.update_yaxes(title_text="Final state", row=1, col=1)
    figure.update_layout(
        title="Transition matrix T[final, initial] (estimate ± local SE)",
        template="plotly_white",
        width=900 if len(conditions) == 2 else 520,
        coloraxis={
            "colorscale": "Blues",
            "cmin": 0.0,
            "cmax": 0.1,
            "colorbar": {"title": "Probability<br>(saturated ≥ 0.1)"},
        },
    )
    return figure


def characterize_readout_mist(
    exp: Experiment,
    target: str,
    *,
    repetition_counts: Sequence[int] | None = None,
    readout_cycle: PulseSchedule | None = None,
    cycle_gap_ns: float = 0.0,
    measure_reference: bool = True,
    gef_measurement_n_shots: int = DEFAULT_GEF_MEASUREMENT_N_SHOTS,
    gef_calibration_n_shots: int = DEFAULT_GEF_CALIBRATION_N_SHOTS,
    shot_interval: float | None = None,
    gef_covariance_rcond: float = 1e-12,
    acquisition_order: Literal["interleaved", "randomized", "grouped"] = "interleaved",
    acquisition_seed: int | None = 0,
    enable_tqdm: bool = True,
    plot: bool = True,
) -> Result:
    """
    Characterize effective GEF transitions caused by one complete readout cycle.

    Parameters
    ----------
    exp : Experiment
        Experiment used for calibrated pulse construction and acquisition.
    target : str
        Target qubit whose GEF transition matrix is measured.
    repetition_counts : Sequence[int] | None, optional
        Strictly increasing counts beginning with zero and containing one.
        At least five points are required. Defaults to
        `(0, 1, 2, 4, 8, 16, 32, 64, 128, 256)`.
    readout_cycle : PulseSchedule | None, optional
        Complete candidate cycle containing drive, optional active depletion,
        and its trailing gap. When omitted, use the configured readout pulse
        followed by `cycle_gap_ns`.
    cycle_gap_ns : float, optional
        Trailing gap for the automatically built configured cycle, in ns.
        Must remain zero when `readout_cycle` is supplied.
    measure_reference : bool, optional
        Whether to fit a duration-matched all-blank cycle. Actual and reference
        schedules are submitted as adjacent pairs in the same population batch.
    gef_measurement_n_shots : int, optional
        Shots for each of the three diagnostic GEF permutations.
    gef_calibration_n_shots : int, optional
        Shots for each of the six GEF calibration configurations.
    shot_interval : float | None, optional
        Shot interval in ns, or the configured measurement default.
    gef_covariance_rcond : float, optional
        Relative covariance eigenvalue cutoff in `[0, 1)`.
    acquisition_order : {"interleaved", "randomized", "grouped"}, optional
        Logical G/E/F and repetition-point submission order. When reference
        measurement is enabled, each logical point is submitted as an adjacent
        actual/reference pair.
    acquisition_seed : int | None, optional
        Seed used only for randomized acquisition order.
    enable_tqdm : bool, optional
        Whether acquisition services show progress labels and bars. Enabled by
        default.
    plot : bool, optional
        Whether to construct and show figures. If false, the returned `figures`
        mapping is empty. This function does not print a fit summary.

    Returns
    -------
    Result
        Measurements, full six-transition fits, raw GEF data, timing, and figures.

    Notes
    -----
    This function performs hardware acquisition. Reported probabilities use
    `T[final_state, initial_state]` and units of `1/readout_cycle`. They are
    effective cycle-boundary probabilities rather than microscopic rates.
    Uncertainty is unscaled local GLS covariance conditional on the shared GEF
    calibration; local null-space directions are assessed separately for every
    reported quantity. Reduced chi-squared, whitened-residual RMS, maximum
    absolute whitened residual, optimizer success, and Jacobian rank are returned
    without assigning a fit-quality category. Calibration uncertainty is not
    included.
    The blank reference zeros the entire candidate cycle while preserving its
    timing. Actual and reference schedules are interleaved in one population
    batch to reduce sensitivity to acquisition drift. A backend may split the
    batch according to its schedule-packing support, while preserving submission
    order. The actual and reference transition matrices are fitted independently;
    their difference is diagnostic and is not subtracted as an estimate of
    readout-only transitions. The final GEF discriminator uses the calibrated
    configured readout, independently of `readout_cycle`. If the candidate cycle
    also creates captures, only the last capture—the final GEF diagnostic
    readout—is used for population estimation and retained as raw IQ.
    Population-fit curves are evaluated at every integer cycle through the largest
    requested repetition. Transition-matrix heatmaps show local standard errors
    when available and saturate their shared color scale at probability 0.1.
    Consequently, diagnostic-readout backaction and population outside the closed
    GEF subspace remain model limitations. Raw IQ, unconstrained populations, and
    GEF fit diagnostics are retained for caller-side assessment.
    """
    for name, value in (
        ("measure_reference", measure_reference),
        ("enable_tqdm", enable_tqdm),
        ("plot", plot),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")
    counts = _validate_repetition_counts(
        DEFAULT_REPETITION_COUNTS if repetition_counts is None else repetition_counts
    )
    measurement_shots = _positive_integer(
        gef_measurement_n_shots,
        "gef_measurement_n_shots",
    )
    calibration_shots = _positive_integer(
        gef_calibration_n_shots,
        "gef_calibration_n_shots",
    )
    covariance_rcond = _validate_covariance_rcond(
        gef_covariance_rcond,
        name="gef_covariance_rcond",
    )
    seed = _optional_seed(acquisition_seed)
    configured_interval = resolve_measurement_defaults(
        exp.ctx.experiment_system.measurement_defaults
    ).execution.shot_interval_ns
    interval = _finite_positive(
        configured_interval if shot_interval is None else shot_interval,
        "shot_interval",
    )
    resolved_target = exp.ctx.resolve_qubit_label(target)
    if readout_cycle is not None and cycle_gap_ns != 0.0:
        raise ValueError(
            "cycle_gap_ns must be zero when readout_cycle supplies the complete cycle."
        )
    cycle = (
        build_default_readout_cycle(exp, resolved_target, cycle_gap_ns=cycle_gap_ns)
        if readout_cycle is None
        else validate_readout_cycle(readout_cycle)
    )
    actual_schedules = build_mist_schedules(
        exp,
        resolved_target,
        cycle,
        counts,
        prefix="actual",
        acquisition_order=cast(AcquisitionOrder, acquisition_order),
        acquisition_seed=seed,
    )
    reference_schedules = None
    reference_cycle = None
    if measure_reference:
        reference_cycle = build_reference_cycle(cycle)
        reference_schedules = build_mist_schedules(
            exp,
            resolved_target,
            reference_cycle,
            counts,
            prefix="reference",
            acquisition_order=cast(AcquisitionOrder, acquisition_order),
            acquisition_seed=seed,
        )
        reference_schedules = _align_reference_acquisition_order(
            actual_schedules,
            reference_schedules,
        )
    submitted_schedules = _interleave_acquisition_schedules(
        actual_schedules,
        reference_schedules,
    )
    acquisition_result = measure_gef_populations(
        exp,
        targets=[resolved_target],
        sequences=submitted_schedules,
        n_shots=measurement_shots,
        calibration_n_shots=calibration_shots,
        shot_interval=interval,
        covariance_rcond=covariance_rcond,
        n_bootstrap=0,
        enable_tqdm=enable_tqdm,
        progress_label=(
            "readout MIST actual/reference"
            if measure_reference
            else "readout MIST actual"
        ),
        capture_index=_DIAGNOSTIC_CAPTURE_INDEX,
    )
    acquisition_data = cast(Mapping[str, Any], acquisition_result.data)
    calibration = acquisition_data["calibration"]
    measurements = _build_measurements(
        counts,
        resolved_target,
        acquisition_result,
        actual_schedules,
        reference_schedules,
    )
    analysis = analyze_readout_mist(
        measurements,
        covariance_rcond=covariance_rcond,
    )
    figures = (
        {"populations": _population_figure(measurements, analysis)} if plot else {}
    )
    if plot:
        figures["transition_matrix"] = _matrix_figure(analysis)
    if plot:
        for figure in figures.values():
            figure.show()
    measurement_options = {
        "repetition_counts": counts,
        "gef_measurement_n_shots": measurement_shots,
        "gef_calibration_n_shots": calibration_shots,
        "shot_interval_ns": interval,
        "measure_reference": measure_reference,
        "gef_covariance_rcond": covariance_rcond,
        "diagnostic_capture_index": _DIAGNOSTIC_CAPTURE_INDEX,
        "acquisition_order": acquisition_order,
        "acquisition_seed": seed,
        "reference_acquisition": (
            "same_batch_adjacent_pairs" if measure_reference else None
        ),
        "submitted_sequence_order": tuple(submitted_schedules),
        "enable_tqdm": enable_tqdm,
        "plot": plot,
    }
    return Result(
        data={
            "measurements": measurements,
            "analysis": analysis,
            "raw_data": {
                "gef_calibration": calibration,
                "actual_raw_iq": _select_acquisition_data(
                    acquisition_data,
                    "raw_iq",
                    actual_schedules,
                ),
                "actual_moment_summaries": _select_acquisition_data(
                    acquisition_data,
                    "moment_summaries",
                    actual_schedules,
                ),
                "actual_population_fits": _select_acquisition_data(
                    acquisition_data,
                    "fits",
                    actual_schedules,
                ),
                "reference_raw_iq": _select_acquisition_data(
                    acquisition_data,
                    "raw_iq",
                    reference_schedules,
                ),
                "reference_moment_summaries": _select_acquisition_data(
                    acquisition_data,
                    "moment_summaries",
                    reference_schedules,
                ),
                "reference_population_fits": _select_acquisition_data(
                    acquisition_data,
                    "fits",
                    reference_schedules,
                ),
            },
            "measurement_options": measurement_options,
            "analysis_options": {
                "model": "full_six_transition_discrete_time_matrix",
                "calibration_bootstrap": False,
                "uncertainty": "unscaled_local_gls_conditional_on_gef_calibration",
            },
            "readout_cycle_timing": {
                "duration_ns": float(cycle.duration),
                "labels": tuple(cycle.labels),
                "frequencies": dict(cycle.get_frequencies()),
                "source": "configured" if readout_cycle is None else "override",
                "cycle_gap_ns": float(cycle_gap_ns) if readout_cycle is None else None,
                "trailing_gap_in_final_cycle": True,
                "reference": "complete_duration_matched_blank"
                if measure_reference
                else None,
                "reference_duration_ns": (
                    None if reference_cycle is None else float(reference_cycle.duration)
                ),
            },
            "schedules": {
                "actual": actual_schedules,
                "reference": reference_schedules,
            },
        },
        figure=figures.get("populations"),
        figures=figures,
    )


__all__ = ["analyze_readout_mist", "characterize_readout_mist"]
