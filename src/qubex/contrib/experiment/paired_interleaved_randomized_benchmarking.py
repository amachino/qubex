"""
Run statistically paired interleaved randomized benchmarking.

The numerical fit intentionally remains a private, workflow-local prototype in
`qubex.contrib`. It should move to `qxfitting` if this workflow is promoted from
contrib or a second production caller needs the same fit API.
"""

from __future__ import annotations

import asyncio
import logging
import math
import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from numbers import Real
from typing import Any, Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from qxpulse import PulseSchedule, Waveform
from scipy.optimize import OptimizeWarning, curve_fit

import qubex.visualization as viz
from qubex.clifford.clifford import Clifford
from qubex.core.async_bridge import get_shared_async_bridge
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_N_CLIFFORDS_1Q,
    DEFAULT_MAX_N_CLIFFORDS_2Q,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
)
from qubex.experiment.models import Result
from qubex.measurement import MeasurementResult, MeasurementResultConverter
from qubex.typing import TargetMap

_Protocol = Literal["reference", "interleaved"]
_ErrorBar = Literal["sem", "std"] | None
_InterleavedOverride = (
    Waveform | PulseSchedule | TargetMap[Waveform | PulseSchedule] | None
)
_X90Override = Waveform | TargetMap[Waveform] | None
_ZX90Override = PulseSchedule | TargetMap[PulseSchedule] | None
_ResolvedInterleavedWaveform = Waveform | PulseSchedule | None

_DEFAULT_BOOTSTRAP_SAMPLES = 2_000
_DEFAULT_BOOTSTRAP_SEED = 0
_DEFAULT_ACQUISITION_SEED: int | None = None
_DEFAULT_PAIRS_PER_SWEEP: int | None = None
_DEFAULT_FIXED_N_CLIFFORDS_2Q = (0, 1, 2, 4, 8, 16, 32, 64, 128)
_DEFAULT_FIXED_N_CLIFFORDS_1Q = (0, 16, 32, 64, 128, 256, 512, 1024, 2048)
_DEFAULT_SWEEP_TIMEOUT_SECONDS = 1800.0
_DEFAULT_AUTO_RANGE_REMAINING_FRACTION = 0.10
_DEFAULT_MAIN_REMAINING_FRACTIONS = (0.90, 0.70, 0.50, 0.30, 0.15, 0.05, 0.02)
_MIN_RECOMMENDED_TRIALS = 20
_MIN_RECOMMENDED_CLIFFORD_POINTS = 6
_MIN_AUTO_RANGE_FIT_POINTS = 6
_MIN_MAIN_GRID_POINTS = 6
_MIN_PILOT_DECAY_SIGNIFICANCE = 3.0
_MIN_PILOT_AMPLITUDE = 0.05
_MIN_PILOT_R_SQUARED = 0.5
_PILOT_BOUND_ATOL = 2e-4
_MIN_ABSOLUTE_LOG_DECAY = 1e-12
_MIN_BOOTSTRAP_SUCCESS_RATE = 0.8
_MIN_BOOTSTRAP_SAMPLES_PER_STRATUM = 2
_MAX_BOOTSTRAP_ANY_BOUND_FRACTION = 0.10
_MAX_BOOTSTRAP_P_BOUND_FRACTION = 0.05
_RAW_SCHEMA_VERSION = 1
_EMPIRICAL_SEM_FLOOR_FRACTION = 0.25
_ALL_ZERO_SEM_FLOOR = 1e-3
_PROTOCOL_REFERENCE: _Protocol = "reference"
_PROTOCOL_INTERLEAVED: _Protocol = "interleaved"

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Internal data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _DecayFitResult:
    """Store one pure numerical weighted RB-decay fit."""

    amplitude: float
    decay_parameter: float
    offset: float
    covariance: NDArray[np.float64] | None
    standard_errors: NDArray[np.float64] | None
    predicted: NDArray[np.float64]
    residuals: NDArray[np.float64]
    weighted_residuals: NDArray[np.float64]
    chi_square: float
    reduced_chi_square: float | None
    r_squared: float | None
    parameters_at_bound: tuple[str, ...]

    @property
    def covariance_valid(self) -> bool:
        """Return whether covariance-derived diagnostics are available."""
        return self.covariance is not None and self.standard_errors is not None


@dataclass(frozen=True)
class _PilotDecayFitResult:
    """Store one fixed-offset fit used only to estimate a pilot decay scale."""

    amplitude: float
    decay_parameter: float
    offset: float
    predicted: NDArray[np.float64]
    residuals: NDArray[np.float64]
    r_squared: float | None
    parameters_at_bound: tuple[str, ...]

    @property
    def amplitude_lower_bound_hit(self) -> bool:
        """Return whether the fitted pilot contrast is effectively zero."""
        return bool(np.isclose(self.amplitude, 0.0, atol=_PILOT_BOUND_ATOL, rtol=0.0))

    @property
    def amplitude_upper_bound_hit(self) -> bool:
        """Return whether the fitted contrast reaches its physical maximum."""
        return bool(
            np.isclose(
                self.amplitude,
                1.0 - self.offset,
                atol=_PILOT_BOUND_ATOL,
                rtol=0.0,
            )
        )


@dataclass(frozen=True)
class _PilotArmAssessment:
    """Store one pilot arm's fit, observed decay, and validity decision."""

    fit: _PilotDecayFitResult | None
    decay_significance: float
    invalid_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Return whether this arm can guide stopping and main-grid design."""
        return not self.invalid_reasons


@dataclass(frozen=True)
class _PilotDecayAssessment:
    """Store paired pilot-fit quality without reusing main-fit inference."""

    reference: _PilotArmAssessment
    interleaved: _PilotArmAssessment

    @property
    def valid_fits(
        self,
    ) -> tuple[_PilotDecayFitResult, _PilotDecayFitResult] | None:
        """Return both fits only when both arms have observable valid decay."""
        if not self.reference.valid or not self.interleaved.valid:
            return None
        reference_fit = self.reference.fit
        interleaved_fit = self.interleaved.fit
        if reference_fit is None or interleaved_fit is None:
            return None
        return reference_fit, interleaved_fit

    @property
    def blocking_reason(self) -> str | None:
        """Return the primary reason this assessment cannot stop the pilot."""
        reasons = (
            *self.reference.invalid_reasons,
            *self.interleaved.invalid_reasons,
        )
        if not reasons:
            return None
        if any(reason != "insufficient_observed_decay" for reason in reasons):
            return "fit_unusable"
        return "insufficient_observed_decay"


@dataclass(frozen=True)
class _TargetSpec:
    """Store the resolved target kind and measured qubit labels."""

    is_two_qubit: bool
    dimension: int
    qubits: tuple[str, ...]


@dataclass(frozen=True)
class _TrialStatistics:
    """Store per-length statistics and numerical fitting weights."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    sem: NDArray[np.float64]
    sem_used: NDArray[np.float64]
    sem_floor_count: int


@dataclass(frozen=True)
class _TrialMoments:
    """Store per-length moments before SEM regularization."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    sem: NDArray[np.float64]


@dataclass(frozen=True)
class _PairedStatistics:
    """Store both arms regularized with one common SEM floor."""

    reference: _TrialStatistics
    interleaved: _TrialStatistics
    sem_floor: float
    fit_covariance_scale_empirical: bool


@dataclass(frozen=True)
class _BootstrapResult:
    """Store successful paired-bootstrap estimates and execution counts."""

    seed: int | None
    n_requested: int
    fidelities: NDArray[np.float64]
    p_reference: NDArray[np.float64]
    p_interleaved: NDArray[np.float64]
    p_correlation: float | None
    stratum_counts: tuple[dict[str, int], ...]
    reference_any_bound_count: int
    interleaved_any_bound_count: int
    reference_p_bound_count: int
    interleaved_p_bound_count: int

    @property
    def n_success(self) -> int:
        """Return the number of successful bootstrap replicates."""
        return len(self.fidelities)

    @property
    def execution_skipped(self) -> bool:
        """Return whether invalid strata prevented bootstrap fitting."""
        return self.n_requested > 0 and not self.resampling_valid

    @property
    def n_attempted(self) -> int:
        """Return the number of bootstrap fits actually attempted."""
        return 0 if self.execution_skipped else self.n_requested

    @property
    def n_failed(self) -> int:
        """Return the number of attempted bootstrap replicates that failed."""
        return self.n_attempted - self.n_success

    @property
    def success_rate(self) -> float | None:
        """Return the successful fraction among attempted bootstrap replicates."""
        if self.n_attempted == 0:
            return None
        return self.n_success / self.n_attempted

    @property
    def optimizer_valid(self) -> bool:
        """Return whether enough bootstrap replicates converged."""
        success_rate = self.success_rate
        return (
            self.n_success >= 2
            and success_rate is not None
            and success_rate >= _MIN_BOOTSTRAP_SUCCESS_RATE
        )

    @property
    def resampling_valid(self) -> bool:
        """Return whether every AB/BA stratum can vary under resampling."""
        return all(
            counts["reference_first"] >= _MIN_BOOTSTRAP_SAMPLES_PER_STRATUM
            and counts["interleaved_first"] >= _MIN_BOOTSTRAP_SAMPLES_PER_STRATUM
            for counts in self.stratum_counts
        )

    @property
    def invalid_reasons(self) -> tuple[str, ...]:
        """Return the reason primary bootstrap uncertainty is unavailable."""
        if self.n_requested == 0:
            return ("bootstrap_disabled",)
        if not self.resampling_valid:
            return ("insufficient_samples_per_stratum",)
        if not self.optimizer_valid:
            return ("insufficient_optimizer_success",)
        return ()

    @property
    def valid(self) -> bool:
        """Return whether resampling and optimization permit primary uncertainty."""
        return self.resampling_valid and self.optimizer_valid

    def _bound_fraction(self, count: int) -> float | None:
        """Return a bound-hit fraction among successful replicates."""
        if self.n_success == 0:
            return None
        return count / self.n_success

    @property
    def reference_any_bound_fraction(self) -> float | None:
        """Return the reference any-parameter bound-hit fraction."""
        return self._bound_fraction(self.reference_any_bound_count)

    @property
    def interleaved_any_bound_fraction(self) -> float | None:
        """Return the interleaved any-parameter bound-hit fraction."""
        return self._bound_fraction(self.interleaved_any_bound_count)

    @property
    def reference_p_bound_fraction(self) -> float | None:
        """Return the reference decay-parameter bound-hit fraction."""
        return self._bound_fraction(self.reference_p_bound_count)

    @property
    def interleaved_p_bound_fraction(self) -> float | None:
        """Return the interleaved decay-parameter bound-hit fraction."""
        return self._bound_fraction(self.interleaved_p_bound_count)

    @property
    def parameter_bound_quality_valid(self) -> bool | None:
        """Return whether successful fits avoid excessive parameter-bound hits."""
        if self.n_requested == 0 or self.n_success == 0:
            return None
        fractions = (
            self.reference_any_bound_fraction,
            self.interleaved_any_bound_fraction,
            self.reference_p_bound_fraction,
            self.interleaved_p_bound_fraction,
        )
        if any(value is None for value in fractions):
            return None
        reference_any, interleaved_any, reference_p, interleaved_p = cast(
            tuple[float, float, float, float],
            fractions,
        )
        return (
            reference_any <= _MAX_BOOTSTRAP_ANY_BOUND_FRACTION
            and interleaved_any <= _MAX_BOOTSTRAP_ANY_BOUND_FRACTION
            and reference_p <= _MAX_BOOTSTRAP_P_BOUND_FRACTION
            and interleaved_p <= _MAX_BOOTSTRAP_P_BOUND_FRACTION
        )

    @property
    def fidelity_std(self) -> float | None:
        """Return sample SD across successful estimates for diagnostics."""
        if self.n_success < 2:
            return None
        return float(np.std(self.fidelities, ddof=1))

    def _fidelity_interval(
        self, quantiles: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Return one percentile interval across successful estimates."""
        if self.n_success < 2:
            return None
        bounds = np.quantile(self.fidelities, quantiles)
        return float(bounds[0]), float(bounds[1])

    @property
    def fidelity_ci68(self) -> tuple[float, float] | None:
        """Return the diagnostic 68% percentile interval."""
        return self._fidelity_interval((0.16, 0.84))

    @property
    def fidelity_ci95(self) -> tuple[float, float] | None:
        """Return the diagnostic 95% percentile interval."""
        return self._fidelity_interval((0.025, 0.975))

    def to_payload(self) -> dict[str, object]:
        """Return the public bootstrap result payload."""
        return {
            "seed": self.seed,
            "n_requested": self.n_requested,
            "n_attempted": self.n_attempted,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "success_rate": self.success_rate,
            "optimizer_valid": self.optimizer_valid,
            "resampling_valid": self.resampling_valid,
            "execution_skipped": self.execution_skipped,
            "valid": self.valid,
            "invalid_reasons": self.invalid_reasons,
            "minimum_samples_per_stratum": _MIN_BOOTSTRAP_SAMPLES_PER_STRATUM,
            "parameter_bound_quality_valid": self.parameter_bound_quality_valid,
            "parameter_bound_quality_thresholds": {
                "any_bound_fraction_max": _MAX_BOOTSTRAP_ANY_BOUND_FRACTION,
                "p_bound_fraction_max": _MAX_BOOTSTRAP_P_BOUND_FRACTION,
            },
            "fidelity_std": self.fidelity_std,
            "fidelity_ci68": self.fidelity_ci68,
            "fidelity_ci95": self.fidelity_ci95,
            "fidelities": self.fidelities,
            "p_reference": self.p_reference,
            "p_interleaved": self.p_interleaved,
            "p_correlation": self.p_correlation,
            "resampling_method": "ab_ba_stratified_paired",
            "stratum_counts": self.stratum_counts,
            "reference_any_bound_count": self.reference_any_bound_count,
            "reference_any_bound_fraction": self.reference_any_bound_fraction,
            "interleaved_any_bound_count": self.interleaved_any_bound_count,
            "interleaved_any_bound_fraction": self.interleaved_any_bound_fraction,
            "reference_p_bound_count": self.reference_p_bound_count,
            "reference_p_bound_fraction": self.reference_p_bound_fraction,
            "interleaved_p_bound_count": self.interleaved_p_bound_count,
            "interleaved_p_bound_fraction": self.interleaved_p_bound_fraction,
        }


@dataclass(frozen=True)
class _RawIRBData:
    """Store validated raw paired data and its metadata."""

    n_cliffords: NDArray[np.int64]
    reference_trials: NDArray[np.float64]
    interleaved_trials: NDArray[np.float64]
    first_protocols: NDArray[np.str_]
    dimension: int
    acquisition: Mapping[str, object]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _AnalysisOptions:
    """Store validated options shared by analysis entry points."""

    n_bootstrap: int
    bootstrap_seed: int | None
    sem_floor: float | None
    error_bar: _ErrorBar
    plot: bool
    save_image: bool


@dataclass(frozen=True)
class _PairPlan:
    """Describe one randomized acquisition pair."""

    length_index: int
    trial_index: int
    n_cliffords: int
    seed: int
    first_protocol: _Protocol

    @property
    def protocols(self) -> tuple[_Protocol, _Protocol]:
        """Return the adjacent protocol order for this pair."""
        if self.first_protocol == _PROTOCOL_REFERENCE:
            return (_PROTOCOL_REFERENCE, _PROTOCOL_INTERLEAVED)
        return (_PROTOCOL_INTERLEAVED, _PROTOCOL_REFERENCE)


@dataclass(frozen=True)
class _SingleAcquisitionContext:
    """Store validated settings reused across single-target acquisitions."""

    target_spec: _TargetSpec
    interleaved_clifford: Clifford
    interleaved_clifford_name: str
    interleaved_waveform: _ResolvedInterleavedWaveform
    pairs_per_sweep: int | None
    x90: _X90Override
    zx90: PulseSchedule | None
    mitigate_readout: bool
    n_shots: int
    shot_interval: float
    time_integration: bool
    sweep_timeout: float


@dataclass(frozen=True)
class _AutoRangeOutcome:
    """Store a selected main grid and auditable pilot metadata."""

    selected_main_grid: NDArray[np.int64]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _ParallelAutoRangeOutcome:
    """Store one shared grid and per-target pilot metadata."""

    selected_main_grid: NDArray[np.int64]
    metadata_by_target: dict[str, dict[str, object]]


@dataclass(frozen=True)
class _MainGridSelection:
    """Store one unthinned, interleaved-first main-grid decision."""

    reference_grid: NDArray[np.int64]
    reference_tail_grid: NDArray[np.int64]
    interleaved_grid: NDArray[np.int64]
    selected_grid: NDArray[np.int64]
    supplemental_integer_grid: NDArray[np.int64]
    maximum_anchor_added: bool
    fallback_used: bool
    fallback_reason: str | None


# -----------------------------------------------------------------------------
# Numerical primitives, validation, and target resolution
# -----------------------------------------------------------------------------


def _rb_decay(
    n_cliffords: NDArray[np.float64],
    amplitude: float,
    decay_parameter: float,
    offset: float,
) -> NDArray[np.float64]:
    """Evaluate the single-exponential RB survival model."""
    return amplitude * decay_parameter**n_cliffords + offset


def _validate_positive_integer(value: object, *, name: str) -> int:
    """Return a strictly positive integer after validation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"`{name}` must be a positive integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return resolved


def _validate_nonnegative_integer(value: object, *, name: str) -> int:
    """Return a nonnegative integer after validation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"`{name}` must be a nonnegative integer.")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"`{name}` must be a nonnegative integer.")
    return resolved


def _resolve_paired_n_trials(value: object) -> int:
    """Resolve the main paired-trial count and enforce analyzable data."""
    resolved = _validate_positive_integer(
        DEFAULT_RB_N_TRIALS if value is None else value,
        name="n_trials",
    )
    if resolved < 2:
        raise ValueError("`n_trials` must be at least 2 for paired analysis.")
    return resolved


def _validate_optional_rng_seed(value: object, *, name: str) -> int | None:
    """Return a NumPy-compatible optional RNG seed."""
    if value is None:
        return None
    resolved = _validate_nonnegative_integer(value, name=name)
    if resolved >= 2**64:
        raise ValueError(f"`{name}` must be smaller than 2**64.")
    return resolved


def _validate_positive_real(value: object, *, name: str) -> float:
    """Return a positive finite floating-point value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a positive real number.")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"`{name}` must be positive and finite.")
    return resolved


def _validate_open_unit_interval(value: object, *, name: str) -> float:
    """Return a finite floating-point value strictly between zero and one."""
    resolved = _validate_positive_real(value, name=name)
    if resolved >= 1.0:
        raise ValueError(f"`{name}` must be smaller than 1.")
    return resolved


def _resolve_main_remaining_fractions(
    values: Collection[float],
) -> tuple[float, ...]:
    """Validate the descending contrast targets used for main-grid design."""
    if isinstance(values, (set, frozenset)):
        raise TypeError(
            "`main_remaining_fractions` must preserve order; use a list, "
            "tuple, or array instead of a set."
        )
    if isinstance(values, Mapping):
        raise TypeError(
            "`main_remaining_fractions` must be a one-dimensional collection "
            "of values, not a mapping."
        )
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError("`main_remaining_fractions` must be a collection of reals.")
    if len(values) == 0:
        raise ValueError("`main_remaining_fractions` must not be empty.")
    resolved = tuple(
        _validate_open_unit_interval(
            value,
            name=f"main_remaining_fractions[{index}]",
        )
        for index, value in enumerate(values)
    )
    if any(left <= right for left, right in pairwise(resolved)):
        raise ValueError("`main_remaining_fractions` must be strictly decreasing.")
    return resolved


def _validate_boolean(value: object, *, name: str) -> bool:
    """Return a Boolean without silently accepting truthy objects."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"`{name}` must be a boolean.")
    return bool(value)


def _normalize_targets(targets: Sequence[str] | str) -> tuple[str, ...]:
    """Return a nonempty ordered tuple of distinct target labels."""
    if isinstance(targets, str):
        resolved = (targets,)
    elif isinstance(targets, (set, frozenset)):
        raise TypeError(
            "`targets` must preserve order; use a list or tuple instead of a set."
        )
    elif isinstance(targets, Sequence):
        resolved = tuple(targets)
    else:
        raise TypeError("`targets` must be a target label or an ordered sequence.")
    if not resolved:
        raise ValueError("`targets` must contain at least one target label.")
    if not all(isinstance(target, str) for target in resolved):
        raise TypeError("Every entry in `targets` must be a string.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("`targets` must not contain duplicate labels.")
    return cast(tuple[str, ...], resolved)


def _split_optional_seed(
    value: object,
    *,
    count: int,
    name: str,
) -> tuple[int | None, ...]:
    """Split one deterministic master seed into independent child streams."""
    resolved = _validate_optional_rng_seed(value, name=name)
    if count == 1 or resolved is None:
        return (resolved,) * count
    children = np.random.SeedSequence(resolved).spawn(count)
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)


def _explicit_seeds_by_target(
    seeds: ArrayLike | Mapping[str, ArrayLike] | None,
    *,
    targets: tuple[str, ...],
) -> dict[str, ArrayLike | None]:
    """Resolve optional fixed seed matrices without sharing one across targets."""
    if seeds is None:
        return dict.fromkeys(targets)
    if isinstance(seeds, Mapping):
        missing = [target for target in targets if target not in seeds]
        if missing:
            raise ValueError(
                "`seeds` is missing target entries: " + ", ".join(missing) + "."
            )
        resolved: dict[str, ArrayLike | None] = {
            target: seeds[target] for target in targets
        }
        missing_matrices = [
            target for target, matrix in resolved.items() if matrix is None
        ]
        if missing_matrices:
            raise TypeError(
                "`seeds` must provide an explicit matrix for each target; got "
                f"None for {', '.join(missing_matrices)}."
            )
        return resolved
    if len(targets) != 1:
        raise ValueError(
            "Multi-target explicit `seeds` must map each target to its own matrix."
        )
    return {targets[0]: seeds}


def _type_names(expected_types: tuple[type, ...]) -> str:
    """Return a readable union of runtime types for validation errors."""
    return " or ".join(expected_type.__name__ for expected_type in expected_types)


def _require_target_mapping_entry(
    value: Mapping[str, Any],
    target: str,
    *,
    name: str,
    expected_types: tuple[type, ...],
) -> Any:
    """Return and type-check one explicit target override without fallback."""
    if target not in value:
        raise ValueError(f"`{name}` is missing an override for target `{target}`.")
    resolved = value[target]
    if not isinstance(resolved, expected_types):
        raise TypeError(
            f"`{name}[{target!r}]` must be {_type_names(expected_types)}; "
            f"got {type(resolved).__name__}."
        )
    return resolved


def _validate_waveform_mapping(
    value: Mapping[str, Any],
    *,
    required_keys: Collection[str],
    name: str,
) -> None:
    """Require waveform-valued entries for every requested physical-qubit key."""
    missing = [key for key in required_keys if key not in value]
    if missing:
        raise ValueError(
            f"`{name}` is missing waveform overrides for: {', '.join(missing)}."
        )
    for key in required_keys:
        resolved = value[key]
        if not isinstance(resolved, Waveform):
            raise TypeError(
                f"`{name}[{key!r}]` must be Waveform; got {type(resolved).__name__}."
            )


def _validate_multi_target_schedule_overrides(
    targets: tuple[str, ...],
    target_specs: Mapping[str, _TargetSpec],
    *,
    interleaved_waveform: _InterleavedOverride,
    zx90: _ZX90Override,
) -> None:
    """Reject target-specific schedules that would be reused across multiple targets."""
    if len(targets) <= 1:
        return
    if isinstance(interleaved_waveform, PulseSchedule):
        raise ValueError(  # noqa: TRY004 - valid for one target, invalid here
            "Multi-target paired IRB cannot reuse one `interleaved_waveform` "
            "PulseSchedule across targets; pass a TargetMap keyed by target instead."
        )
    two_qubit_targets = tuple(
        target for target in targets if target_specs[target].is_two_qubit
    )
    if len(two_qubit_targets) > 1 and isinstance(zx90, PulseSchedule):
        raise ValueError(
            "Multi-target paired IRB with more than one CR target cannot reuse one "
            "`zx90` PulseSchedule; pass a TargetMap keyed by CR target instead."
        )


def _resolve_interleaved_waveform_override(
    target: str,
    target_spec: _TargetSpec,
    value: _InterleavedOverride,
) -> _ResolvedInterleavedWaveform:
    """Resolve one target's interleaved-gate implementation."""
    expected_types = (
        (PulseSchedule,) if target_spec.is_two_qubit else (Waveform, PulseSchedule)
    )
    if isinstance(value, Mapping):
        return cast(
            _ResolvedInterleavedWaveform,
            _require_target_mapping_entry(
                value,
                target,
                name="interleaved_waveform",
                expected_types=expected_types,
            ),
        )
    if value is not None and not isinstance(value, expected_types):
        raise TypeError(
            "`interleaved_waveform` must be "
            f"{_type_names(expected_types)} for target `{target}`; "
            f"got {type(value).__name__}."
        )
    return value


def _resolve_zx90_override(
    target: str,
    target_spec: _TargetSpec,
    value: _ZX90Override,
) -> PulseSchedule | None:
    """Resolve a CR-target ZX90 schedule and discard it for 1Q targets."""
    if not target_spec.is_two_qubit:
        return None
    if isinstance(value, Mapping):
        return cast(
            PulseSchedule,
            _require_target_mapping_entry(
                value,
                target,
                name="zx90",
                expected_types=(PulseSchedule,),
            ),
        )
    if value is not None and not isinstance(value, PulseSchedule):
        raise TypeError(
            f"`zx90` must be PulseSchedule for CR target `{target}`; "
            f"got {type(value).__name__}."
        )
    return value


def _resolve_x90_override(
    target: str,
    target_spec: _TargetSpec,
    value: _X90Override,
) -> _X90Override:
    """Resolve a 1Q target entry or validate a CR physical-qubit map."""
    if target_spec.is_two_qubit:
        if isinstance(value, Mapping):
            _validate_waveform_mapping(
                value,
                required_keys=target_spec.qubits,
                name="x90",
            )
        elif value is not None:
            raise TypeError(
                "`x90` must be a physical-qubit waveform mapping for CR target "
                f"`{target}`; got {type(value).__name__}."
            )
        return value
    if isinstance(value, Mapping):
        return cast(
            Waveform,
            _require_target_mapping_entry(
                value,
                target,
                name="x90",
                expected_types=(Waveform,),
            ),
        )
    if value is not None and not isinstance(value, Waveform):
        raise TypeError(
            f"`x90` must be Waveform for 1Q target `{target}`; "
            f"got {type(value).__name__}."
        )
    return value


def _resolve_analysis_options(
    *,
    n_bootstrap: object,
    bootstrap_seed: object,
    sem_floor: object,
    error_bar: object,
    plot: object,
    save_image: object,
) -> _AnalysisOptions:
    """Validate analysis options before numerical work or hardware acquisition."""
    if error_bar is not None and not isinstance(error_bar, str):
        raise TypeError("`error_bar` must be a string or None.")
    if error_bar not in (None, "sem", "std"):
        raise ValueError("`error_bar` must be 'sem', 'std', or None.")
    resolved_sem_floor = (
        None
        if sem_floor is None
        else _validate_positive_real(sem_floor, name="sem_floor")
    )
    return _AnalysisOptions(
        n_bootstrap=_validate_nonnegative_integer(
            n_bootstrap,
            name="n_bootstrap",
        ),
        bootstrap_seed=_validate_optional_rng_seed(
            bootstrap_seed,
            name="bootstrap_seed",
        ),
        sem_floor=resolved_sem_floor,
        error_bar=cast(_ErrorBar, error_bar),
        plot=_validate_boolean(plot, name="plot"),
        save_image=_validate_boolean(save_image, name="save_image"),
    )


def _default_n_cliffords(maximum: int) -> NDArray[np.int64]:
    """Return zero followed by powers of two through the fixed maximum."""
    maximum = _validate_positive_integer(maximum, name="max_n_cliffords")
    if maximum < 4:
        raise ValueError("`max_n_cliffords` must be at least 4.")
    if maximum > np.iinfo(np.int64).max:
        raise ValueError("`max_n_cliffords` must fit in a signed 64-bit integer.")
    values = [0]
    value = 1
    while value <= maximum:
        values.append(value)
        value *= 2
    return np.asarray(values, dtype=np.int64)


def _auto_range_candidate_grid(maximum: int) -> NDArray[np.int64]:
    """Return power-of-two pilot lengths plus the exact exploration ceiling."""
    grid = _default_n_cliffords(maximum)
    if int(grid[-1]) != maximum:
        grid = np.append(grid, np.int64(maximum))
    if len(grid) < _MIN_AUTO_RANGE_FIT_POINTS:
        raise ValueError(
            "Active auto-range requires `max_n_cliffords` to provide at least "
            "six pilot lengths; increase it or set `auto_range=False`."
        )
    return grid


def _validate_range_choice(
    n_cliffords_range: ArrayLike | None,
    max_n_cliffords: int | None,
) -> None:
    """Reject the two mutually exclusive Clifford-range controls."""
    if n_cliffords_range is not None and max_n_cliffords is not None:
        raise ValueError(
            "Specify only one of `n_cliffords_range` and `max_n_cliffords`."
        )


def _resolve_n_cliffords(
    n_cliffords_range: ArrayLike | None,
    *,
    maximum: int,
) -> NDArray[np.int64]:
    """Return a validated fixed Clifford-length grid."""
    if n_cliffords_range is None:
        return _default_n_cliffords(maximum)
    raw = np.asarray(n_cliffords_range)
    if raw.ndim != 1:
        raise ValueError("`n_cliffords_range` must be one-dimensional.")
    if raw.size < 4:
        raise ValueError("`n_cliffords_range` must contain at least four lengths.")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("`n_cliffords_range` must contain integers.")
    resolved = raw.astype(np.int64)
    if np.any(resolved < 0):
        raise ValueError("`n_cliffords_range` must contain nonnegative lengths.")
    if np.any(np.diff(resolved) <= 0):
        raise ValueError("`n_cliffords_range` must be strictly increasing.")
    return resolved


def _generated_seed_matrix(
    shape: tuple[int, int],
    *,
    sequence_seed: int | None,
) -> NDArray[np.int64]:
    """Generate independent uint32 seeds, retrying the negligible collisions."""
    rng = np.random.default_rng(sequence_seed)
    size = int(np.prod(shape))
    while True:
        seeds = rng.integers(0, 2**32, size=size, dtype=np.uint32).astype(np.int64)
        if len(np.unique(seeds)) == size:
            return seeds.reshape(shape)


def _resolve_seed_matrix(
    seeds: ArrayLike | None,
    *,
    shape: tuple[int, int],
    sequence_seed: int | None,
) -> NDArray[np.int64]:
    """Return exactly one sequence seed per Clifford-length/trial cell."""
    if seeds is not None and sequence_seed is not None:
        raise ValueError("Specify only one of `seeds` and `sequence_seed`.")
    if seeds is None:
        return _generated_seed_matrix(shape, sequence_seed=sequence_seed)
    raw = np.asarray(seeds)
    if raw.shape != shape:
        raise ValueError(f"`seeds` must have shape {shape}; got {raw.shape}.")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("`seeds` must contain integers.")
    resolved = raw.astype(np.int64)
    if np.any(resolved < 0) or np.any(resolved >= 2**32):
        raise ValueError("`seeds` values must be in the uint32 range.")
    if len(np.unique(resolved)) != resolved.size:
        raise ValueError("`seeds` must contain distinct values for every cell.")
    return resolved


def _build_pair_plan(
    n_cliffords: NDArray[np.int64],
    seeds: NDArray[np.int64],
    *,
    acquisition_seed: int | None,
) -> list[_PairPlan]:
    """Randomize pairs globally after balancing AB/BA within every length."""
    rng = np.random.default_rng(acquisition_seed)
    unordered_plans: list[_PairPlan] = []
    n_trials = seeds.shape[1]
    extra_reference_first = np.zeros(len(n_cliffords), dtype=np.bool_)
    if n_trials % 2 == 1:
        n_extra_reference = len(n_cliffords) // 2
        if len(n_cliffords) % 2 == 1:
            n_extra_reference += int(rng.integers(0, 2))
        extra_reference_first[:n_extra_reference] = True
        rng.shuffle(extra_reference_first)
    for length_index, n_clifford in enumerate(n_cliffords):
        n_reference_first = n_trials // 2 + int(extra_reference_first[length_index])
        first_protocols: list[_Protocol] = [
            *(_PROTOCOL_REFERENCE for _ in range(n_reference_first)),
            *(_PROTOCOL_INTERLEAVED for _ in range(n_trials - n_reference_first)),
        ]
        rng.shuffle(first_protocols)
        unordered_plans.extend(
            _PairPlan(
                length_index=length_index,
                trial_index=trial_index,
                n_cliffords=int(n_clifford),
                seed=int(seeds[length_index, trial_index]),
                first_protocol=first_protocol,
            )
            for trial_index, first_protocol in enumerate(first_protocols)
        )
    pair_order = rng.permutation(len(unordered_plans))
    return [unordered_plans[int(index)] for index in pair_order]


def _resolve_interleaved_clifford(
    exp: Experiment,
    interleaved_clifford: str | Clifford,
) -> tuple[Clifford, str]:
    """Resolve a public Clifford name without changing the legacy service."""
    if isinstance(interleaved_clifford, str):
        resolved = exp.benchmarking_service.clifford.get(interleaved_clifford)
        if resolved is None:
            raise ValueError(f"Invalid Clifford: {interleaved_clifford}")
        if not isinstance(resolved, Clifford):
            raise TypeError(
                "The benchmarking-service Clifford registry must contain "
                "Clifford instances."
            )
        return resolved, interleaved_clifford
    if not isinstance(interleaved_clifford, Clifford):
        raise TypeError("`interleaved_clifford` must be a Clifford or its name.")
    return interleaved_clifford, interleaved_clifford.name


def _validate_clifford_arity(
    interleaved_clifford: Clifford,
    *,
    target_spec: _TargetSpec,
) -> None:
    """Require the Clifford and target to act on the same number of qubits."""
    expected_arity = 2 if target_spec.is_two_qubit else 1
    operator_arities = {len(operator) for operator in interleaved_clifford.map}
    if operator_arities != {expected_arity}:
        target_kind = "two-qubit" if target_spec.is_two_qubit else "one-qubit"
        raise ValueError(
            f"`interleaved_clifford` must be a {target_kind} Clifford for this target."
        )


def _resolve_target(exp: Experiment, target: str) -> _TargetSpec:
    """Validate one target and return its measurement specification."""
    if not isinstance(target, str):
        raise TypeError("`target` must be a single target label string.")
    target_object = exp.experiment_system.get_target(target)
    is_two_qubit = bool(target_object.is_cr)
    if not is_two_qubit:
        qubit = exp.experiment_system.resolve_qubit_label(target)
        if not isinstance(qubit, str):
            raise ValueError(f"Could not resolve a qubit label for `{target}`.")
        exp.pulse.validate_rabi_params([target])
        return _TargetSpec(is_two_qubit=False, dimension=2, qubits=(qubit,))

    cr_params = getattr(getattr(exp.ctx, "calib_note", None), "cr_params", {})
    if target not in cr_params:
        raise ValueError(f"CR parameters for `{target}` are not stored.")
    qubits = tuple(exp.ctx.cr_pair(target))
    if (
        len(qubits) != 2
        or not all(isinstance(qubit, str) for qubit in qubits)
        or len(set(qubits)) != 2
    ):
        raise ValueError(f"CR target `{target}` must resolve to two distinct qubits.")
    classifiers = exp.ctx.classifiers
    missing = [qubit for qubit in qubits if classifiers.get(qubit) is None]
    if missing:
        raise ValueError(f"Classifier not found for {', '.join(missing)}.")
    return _TargetSpec(is_two_qubit=True, dimension=4, qubits=qubits)


def _resolve_target_maximum(
    target_spec: _TargetSpec,
    max_n_cliffords: int | None,
) -> int:
    """Return the target-specific default or a validated explicit maximum."""
    if max_n_cliffords is not None:
        return _validate_positive_integer(
            max_n_cliffords,
            name="max_n_cliffords",
        )
    return (
        DEFAULT_MAX_N_CLIFFORDS_2Q
        if target_spec.is_two_qubit
        else DEFAULT_MAX_N_CLIFFORDS_1Q
    )


def _default_fixed_n_cliffords(
    target_spec: _TargetSpec,
) -> NDArray[np.int64]:
    """Return the default fixed Clifford grid for one target kind."""
    values = (
        _DEFAULT_FIXED_N_CLIFFORDS_2Q
        if target_spec.is_two_qubit
        else _DEFAULT_FIXED_N_CLIFFORDS_1Q
    )
    return np.asarray(values, dtype=np.int64)


def _resolve_fixed_n_cliffords(
    target_spec: _TargetSpec,
    n_cliffords_range: ArrayLike | None,
    max_n_cliffords: int | None,
) -> NDArray[np.int64]:
    """Resolve explicit/default fixed-range Clifford lengths for one target."""
    if n_cliffords_range is None and max_n_cliffords is None:
        return _default_fixed_n_cliffords(target_spec)
    maximum = _resolve_target_maximum(target_spec, max_n_cliffords)
    return _resolve_n_cliffords(n_cliffords_range, maximum=maximum)


def _resolve_pairs_per_sweep(value: object) -> int | None:
    """Return a positive pair chunk size, or None to submit every pair at once."""
    if value is None:
        return None
    return _validate_positive_integer(value, name="pairs_per_sweep")


def _pair_chunk_size(context: _SingleAcquisitionContext, n_pairs: int) -> int:
    """Return the number of adjacent pairs submitted in one sweep call."""
    return n_pairs if context.pairs_per_sweep is None else context.pairs_per_sweep


def _resolve_target_acquisition(
    exp: Experiment,
    target: str,
    *,
    interleaved_clifford: str | Clifford,
    interleaved_waveform: _InterleavedOverride,
    pairs_per_sweep: object,
    x90: _X90Override,
    zx90: _ZX90Override,
    mitigate_readout: object,
    n_shots: object,
    shot_interval: object,
    time_integration: object,
    sweep_timeout: object,
    target_spec: _TargetSpec | None = None,
) -> _SingleAcquisitionContext:
    """Resolve one target and return its validated acquisition context."""
    resolved_target_spec = (
        _resolve_target(exp, target) if target_spec is None else target_spec
    )
    resolved_clifford, clifford_name = _resolve_interleaved_clifford(
        exp,
        interleaved_clifford,
    )
    _validate_clifford_arity(resolved_clifford, target_spec=resolved_target_spec)
    context = _SingleAcquisitionContext(
        target_spec=resolved_target_spec,
        interleaved_clifford=resolved_clifford,
        interleaved_clifford_name=clifford_name,
        interleaved_waveform=_resolve_interleaved_waveform_override(
            target,
            resolved_target_spec,
            interleaved_waveform,
        ),
        pairs_per_sweep=_resolve_pairs_per_sweep(pairs_per_sweep),
        x90=_resolve_x90_override(target, resolved_target_spec, x90),
        zx90=_resolve_zx90_override(target, resolved_target_spec, zx90),
        mitigate_readout=_validate_boolean(
            mitigate_readout,
            name="mitigate_readout",
        ),
        n_shots=_validate_positive_integer(n_shots, name="n_shots"),
        shot_interval=_validate_positive_real(
            shot_interval,
            name="shot_interval",
        ),
        time_integration=_validate_boolean(
            time_integration,
            name="time_integration",
        ),
        sweep_timeout=_validate_positive_real(
            sweep_timeout,
            name="sweep_timeout",
        ),
    )
    return context


def _resolve_target_acquisitions(
    exp: Experiment,
    targets: tuple[str, ...],
    target_specs: Mapping[str, _TargetSpec],
    *,
    interleaved_clifford: str | Clifford,
    interleaved_waveform: _InterleavedOverride,
    pairs_per_sweep: object,
    x90: _X90Override,
    zx90: _ZX90Override,
    mitigate_readout: object,
    n_shots: object,
    shot_interval: object,
    time_integration: object,
    sweep_timeout: object,
) -> dict[str, _SingleAcquisitionContext]:
    """Resolve validated acquisition contexts for an ordered target group."""
    return {
        target: _resolve_target_acquisition(
            exp,
            target,
            interleaved_clifford=interleaved_clifford,
            interleaved_waveform=interleaved_waveform,
            pairs_per_sweep=pairs_per_sweep,
            x90=x90,
            zx90=zx90,
            mitigate_readout=mitigate_readout,
            n_shots=n_shots,
            shot_interval=shot_interval,
            time_integration=time_integration,
            sweep_timeout=sweep_timeout,
            target_spec=target_specs[target],
        )
        for target in targets
    }


def _rb_schedule_for_protocol(
    exp: Experiment,
    target: str,
    context: _SingleAcquisitionContext,
    *,
    n_cliffords: int,
    seed: int,
    protocol: _Protocol,
) -> PulseSchedule:
    """Build one reference or interleaved RB schedule for a resolved target."""
    is_interleaved = protocol == _PROTOCOL_INTERLEAVED
    return exp.rb_sequence(
        target,
        n=n_cliffords,
        x90=context.x90,
        zx90=context.zx90,
        interleaved_waveform=(context.interleaved_waveform if is_interleaved else None),
        interleaved_clifford=(context.interleaved_clifford if is_interleaved else None),
        seed=seed,
    )


def _pair_chunk_points(
    chunk: Sequence[_PairPlan],
) -> list[tuple[_PairPlan, _Protocol]]:
    """Expand each pair into its two adjacent protocol measurement points."""
    return [(pair, protocol) for pair in chunk for protocol in pair.protocols]


def _raw_target_payload(
    *,
    context: _SingleAcquisitionContext,
    n_cliffords: NDArray[np.int64],
    seeds: NDArray[np.int64],
    sequence_seed: int | None,
    acquisition_seed: int | None,
    pair_plan: Sequence[_PairPlan],
    pair_chunks: tuple[dict[str, object], ...],
    reference_trials: NDArray[np.float64],
    interleaved_trials: NDArray[np.float64],
    batching_semantics: str,
    acquisition_extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the shared raw-result schema for one target."""
    planned_order = tuple(
        {
            "pair_order_index": pair_order_index,
            "length_index": pair.length_index,
            "trial_index": pair.trial_index,
            "n_cliffords": pair.n_cliffords,
            "seed": int(seeds[pair.length_index, pair.trial_index]),
            "first_protocol": pair.first_protocol,
        }
        for pair_order_index, pair in enumerate(pair_plan)
    )
    acquisition: dict[str, object] = {
        "n_cliffords": n_cliffords.copy(),
        "n_trials": seeds.shape[1],
        "n_shots": context.n_shots,
        "shot_interval": context.shot_interval,
        "time_integration": context.time_integration,
        "seeds": seeds.copy(),
        "sequence_seed": sequence_seed,
        "acquisition_seed": acquisition_seed,
        "planned_order": planned_order,
        "pair_chunks": pair_chunks,
        "pairs_per_sweep": context.pairs_per_sweep,
        "n_sweep_calls": len(pair_chunks),
        "n_measurement_points": 2 * seeds.size,
        "timestamp_granularity": "sweep_chunk",
        "batching_semantics": batching_semantics,
    }
    if acquisition_extras is not None:
        acquisition.update(acquisition_extras)

    return {
        "reference": {"trials": reference_trials},
        "interleaved": {"trials": interleaved_trials},
        "acquisition": acquisition,
        "metadata": {
            "schema_version": _RAW_SCHEMA_VERSION,
            "dimension": context.target_spec.dimension,
            "is_two_qubit": context.target_spec.is_two_qubit,
            "qubits": context.target_spec.qubits,
            "interleaved_clifford": context.interleaved_clifford_name,
            "mitigate_readout": (
                context.mitigate_readout if context.target_spec.is_two_qubit else False
            ),
        },
    }


# -----------------------------------------------------------------------------
# Hardware acquisition and raw-result construction
# -----------------------------------------------------------------------------


def _point_kerneled_data(
    exp: Experiment,
    target: str,
    point_result: MeasurementResult,
) -> NDArray[Any]:
    """Return integrated data from the final capture of a one-qubit target."""
    resolve_read_label = getattr(exp.experiment_system, "resolve_read_label", None)
    readout_target = (
        str(resolve_read_label(target)) if callable(resolve_read_label) else target
    )
    if readout_target in point_result.data:
        data_target = readout_target
    elif target in point_result.data:
        data_target = target
    else:
        raise KeyError(
            f"Neither `{readout_target}` nor `{target}` exists in measurement "
            f"result targets {list(point_result.data)}."
        )
    captures = point_result.data[data_target]
    if not captures:
        raise ValueError(f"Measurement result for `{data_target}` has no captures.")
    capture = captures[-1]
    capture_data = np.asarray(capture.data)
    capture_config = getattr(capture, "config", None)
    if capture_config is not None and not capture_config.time_integration:
        integration_axis = None if capture_config.shot_averaging else -1
        return np.asarray(np.sum(capture_data, axis=integration_axis))
    return capture_data


def _extract_survival_probability(
    exp: Experiment,
    target: str,
    point_result: MeasurementResult,
    *,
    is_two_qubit: bool,
    qubits: tuple[str, ...],
    mitigate_readout: bool,
) -> float:
    """Convert one sweep-point result into computational-state survival."""
    if not is_two_qubit:
        kerneled_data = _point_kerneled_data(exp, target, point_result)
        normalized = exp.pulse.rabi_params[target].normalize(kerneled_data)
        probability = float(np.real(np.mean(np.asarray(normalized)))) / 2.0 + 0.5
    else:
        measure_result = MeasurementResultConverter.to_measure_result(
            point_result,
            index=-1,
            classifiers=exp.ctx.classifiers,
        )
        probabilities = (
            measure_result.get_mitigated_probabilities(qubits)
            if mitigate_readout
            else measure_result.get_probabilities(qubits)
        )
        try:
            probability = float(probabilities["00"])
        except KeyError:
            raise ValueError(
                "The two-qubit measurement did not produce `P(00)`."
            ) from None
    if not np.isfinite(probability):
        raise ValueError("The measured survival probability must be finite.")
    return probability


def _iso_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


async def _measure_paired_irb_async(
    exp: Experiment,
    target: str,
    *,
    context: _SingleAcquisitionContext,
    n_cliffords: NDArray[np.int64],
    seeds: NDArray[np.int64],
    sequence_seed: int | None,
    acquisition_seed: int | None,
) -> Result:
    """Measure randomized pair chunks and retain their honest timing metadata."""
    pair_plan = _build_pair_plan(
        n_cliffords,
        seeds,
        acquisition_seed=acquisition_seed,
    )
    reference_trials = np.full(seeds.shape, np.nan, dtype=np.float64)
    interleaved_trials = np.full(seeds.shape, np.nan, dtype=np.float64)
    chunk_records: list[dict[str, object]] = []
    exp.ctx.reset_awg_and_capunits(qubits=set(context.target_spec.qubits))

    chunk_size = _pair_chunk_size(context, len(pair_plan))
    for chunk_start in range(0, len(pair_plan), chunk_size):
        chunk = pair_plan[chunk_start : chunk_start + chunk_size]
        points = _pair_chunk_points(chunk)

        def schedule(
            point_index: int,
            *,
            _points: list[tuple[_PairPlan, _Protocol]] = points,
        ) -> PulseSchedule:
            pair, protocol = _points[int(point_index)]
            return _rb_schedule_for_protocol(
                exp,
                target,
                context,
                n_cliffords=pair.n_cliffords,
                seed=pair.seed,
                protocol=protocol,
            )

        started_at = _iso_now()
        sweep = await asyncio.wait_for(
            exp.measurement_service.run_sweep_measurement(
                schedule,
                sweep_values=np.arange(len(points), dtype=np.int64),
                n_shots=context.n_shots,
                shot_interval=context.shot_interval,
                shot_averaging=not context.target_spec.is_two_qubit,
                time_integration=context.time_integration,
                state_classification=False,
                final_measurement=True,
                plot=False,
                enable_tqdm=False,
            ),
            timeout=context.sweep_timeout,
        )
        completed_at = _iso_now()
        if len(sweep.results) != len(points):
            raise RuntimeError(
                "Sweep result count does not match the requested paired points: "
                f"expected {len(points)}, got {len(sweep.results)}."
            )
        for point_result, (pair, protocol) in zip(
            sweep.results,
            points,
            strict=True,
        ):
            probability = _extract_survival_probability(
                exp,
                target,
                point_result,
                is_two_qubit=context.target_spec.is_two_qubit,
                qubits=context.target_spec.qubits,
                mitigate_readout=context.mitigate_readout,
            )
            destination = (
                reference_trials
                if protocol == _PROTOCOL_REFERENCE
                else interleaved_trials
            )
            destination[pair.length_index, pair.trial_index] = probability
        chunk_records.append(
            {
                "pair_order_indices": tuple(
                    range(chunk_start, chunk_start + len(chunk))
                ),
                "n_points": len(points),
                "started_at": started_at,
                "completed_at": completed_at,
            }
        )

    if not np.all(np.isfinite(reference_trials)) or not np.all(
        np.isfinite(interleaved_trials)
    ):
        raise RuntimeError("Paired IRB acquisition produced non-finite trial data.")
    payload = _raw_target_payload(
        context=context,
        n_cliffords=n_cliffords,
        seeds=seeds,
        sequence_seed=sequence_seed,
        acquisition_seed=acquisition_seed,
        pair_plan=pair_plan,
        pair_chunks=tuple(chunk_records),
        reference_trials=reference_trials,
        interleaved_trials=interleaved_trials,
        batching_semantics="delegated_to_run_sweep_measurement",
    )
    return Result(data={target: payload})


def _run_single_target_acquisition(
    exp: Experiment,
    target: str,
    *,
    context: _SingleAcquisitionContext,
    n_cliffords: NDArray[np.int64],
    seeds: NDArray[np.int64],
    sequence_seed: int | None,
    acquisition_seed: int | None,
) -> Result:
    """Run one resolved single-target acquisition through the async bridge."""
    chunk_size = _pair_chunk_size(context, seeds.size)
    n_pair_chunks = math.ceil(seeds.size / chunk_size)
    bridge_timeout = context.sweep_timeout * n_pair_chunks + 30.0
    bridge = get_shared_async_bridge(key="experiment")
    return bridge.run(
        lambda: _measure_paired_irb_async(
            exp,
            target,
            context=context,
            n_cliffords=n_cliffords,
            seeds=seeds,
            sequence_seed=sequence_seed,
            acquisition_seed=acquisition_seed,
        ),
        timeout=bridge_timeout,
    )


def _validate_parallel_target_specs(
    target_specs: Mapping[str, _TargetSpec],
) -> bool:
    """Validate homogeneous, physically disjoint parallel targets."""
    target_kinds = {target_spec.is_two_qubit for target_spec in target_specs.values()}
    if len(target_kinds) != 1:
        raise ValueError("Parallel paired IRB cannot mix 1Q and 2Q targets.")
    occupied_qubits: set[str] = set()
    for target, target_spec in target_specs.items():
        overlap = occupied_qubits.intersection(target_spec.qubits)
        if overlap:
            raise ValueError(
                "Parallel paired IRB targets must not share physical qubits; "
                f"`{target}` overlaps on {', '.join(sorted(overlap))}."
            )
        occupied_qubits.update(target_spec.qubits)
    return next(iter(target_kinds))


async def _measure_paired_irb_parallel_async(
    exp: Experiment,
    targets: tuple[str, ...],
    *,
    contexts: Mapping[str, _SingleAcquisitionContext],
    n_cliffords: NDArray[np.int64],
    seeds_by_target: Mapping[str, NDArray[np.int64]],
    sequence_seeds: Mapping[str, int | None],
    acquisition_seed: int | None,
) -> Result:
    """Acquire one shared point stream with independent per-target sequences."""
    first_context = contexts[targets[0]]
    pair_plan = _build_pair_plan(
        n_cliffords,
        seeds_by_target[targets[0]],
        acquisition_seed=acquisition_seed,
    )
    reference_trials = {
        target: np.full(seeds_by_target[target].shape, np.nan, dtype=np.float64)
        for target in targets
    }
    interleaved_trials = {
        target: np.full(seeds_by_target[target].shape, np.nan, dtype=np.float64)
        for target in targets
    }
    chunk_records: list[dict[str, object]] = []
    reset_qubits = {
        qubit for context in contexts.values() for qubit in context.target_spec.qubits
    }
    exp.ctx.reset_awg_and_capunits(qubits=reset_qubits)

    chunk_size = _pair_chunk_size(first_context, len(pair_plan))
    for chunk_start in range(0, len(pair_plan), chunk_size):
        chunk = pair_plan[chunk_start : chunk_start + chunk_size]
        points = _pair_chunk_points(chunk)

        def schedule(
            point_index: int,
            *,
            _points: list[tuple[_PairPlan, _Protocol]] = points,
        ) -> PulseSchedule:
            pair, protocol = _points[int(point_index)]
            target_schedules = [
                _rb_schedule_for_protocol(
                    exp,
                    target,
                    contexts[target],
                    n_cliffords=pair.n_cliffords,
                    seed=int(
                        seeds_by_target[target][
                            pair.length_index,
                            pair.trial_index,
                        ]
                    ),
                    protocol=protocol,
                )
                for target in targets
            ]
            maximum_duration = max(
                target_schedule.duration for target_schedule in target_schedules
            )
            with PulseSchedule() as combined:
                for target_schedule in target_schedules:
                    combined.call(
                        target_schedule.padded(
                            total_duration=maximum_duration,
                            pad_side="left",
                            deepcopy=False,
                        )
                    )
            return combined

        started_at = _iso_now()
        sweep = await asyncio.wait_for(
            exp.measurement_service.run_sweep_measurement(
                schedule,
                sweep_values=np.arange(len(points), dtype=np.int64),
                n_shots=first_context.n_shots,
                shot_interval=first_context.shot_interval,
                shot_averaging=not first_context.target_spec.is_two_qubit,
                time_integration=first_context.time_integration,
                state_classification=False,
                final_measurement=True,
                plot=False,
                enable_tqdm=False,
            ),
            timeout=first_context.sweep_timeout,
        )
        completed_at = _iso_now()
        if len(sweep.results) != len(points):
            raise RuntimeError(
                "Parallel sweep result count does not match the requested paired "
                f"points: expected {len(points)}, got {len(sweep.results)}."
            )
        for point_result, (pair, protocol) in zip(
            sweep.results,
            points,
            strict=True,
        ):
            for target in targets:
                context = contexts[target]
                probability = _extract_survival_probability(
                    exp,
                    target,
                    point_result,
                    is_two_qubit=context.target_spec.is_two_qubit,
                    qubits=context.target_spec.qubits,
                    mitigate_readout=context.mitigate_readout,
                )
                destination = (
                    reference_trials[target]
                    if protocol == _PROTOCOL_REFERENCE
                    else interleaved_trials[target]
                )
                destination[pair.length_index, pair.trial_index] = probability
        chunk_records.append(
            {
                "pair_order_indices": tuple(
                    range(chunk_start, chunk_start + len(chunk))
                ),
                "n_points": len(points),
                "started_at": started_at,
                "completed_at": completed_at,
            }
        )

    result_data: dict[str, object] = {}
    for target in targets:
        if not np.all(np.isfinite(reference_trials[target])) or not np.all(
            np.isfinite(interleaved_trials[target])
        ):
            raise RuntimeError(
                f"Parallel paired IRB produced non-finite data for `{target}`."
            )
        target_seeds = seeds_by_target[target]
        context = contexts[target]
        result_data[target] = _raw_target_payload(
            context=context,
            n_cliffords=n_cliffords,
            seeds=target_seeds,
            sequence_seed=sequence_seeds[target],
            acquisition_seed=acquisition_seed,
            pair_plan=pair_plan,
            pair_chunks=tuple(dict(record) for record in chunk_records),
            reference_trials=reference_trials[target],
            interleaved_trials=interleaved_trials[target],
            batching_semantics="parallel_targets_per_measurement_point",
            acquisition_extras={
                "in_parallel": True,
                "parallel_targets": targets,
                "schedule_alignment": "left_padded_to_group_max_duration",
            },
        )
    return Result(data=result_data)


def _run_parallel_acquisition(
    exp: Experiment,
    targets: tuple[str, ...],
    *,
    contexts: Mapping[str, _SingleAcquisitionContext],
    n_cliffords: NDArray[np.int64],
    seeds_by_target: Mapping[str, NDArray[np.int64]],
    sequence_seeds: Mapping[str, int | None],
    acquisition_seed: int | None,
) -> Result:
    """Run one resolved parallel acquisition through the async bridge."""
    first_context = contexts[targets[0]]
    n_pairs = seeds_by_target[targets[0]].size
    chunk_size = _pair_chunk_size(first_context, n_pairs)
    n_pair_chunks = math.ceil(n_pairs / chunk_size)
    bridge_timeout = first_context.sweep_timeout * n_pair_chunks + 30.0
    bridge = get_shared_async_bridge(key="experiment")
    return bridge.run(
        lambda: _measure_paired_irb_parallel_async(
            exp,
            targets,
            contexts=contexts,
            n_cliffords=n_cliffords,
            seeds_by_target=seeds_by_target,
            sequence_seeds=sequence_seeds,
            acquisition_seed=acquisition_seed,
        ),
        timeout=bridge_timeout,
    )


def measure_paired_irb(
    exp: Experiment,
    target: str,
    *,
    interleaved_clifford: str | Clifford,
    interleaved_waveform: Waveform | PulseSchedule | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    sequence_seed: int | None = None,
    acquisition_seed: int | None = _DEFAULT_ACQUISITION_SEED,
    pairs_per_sweep: int | None = _DEFAULT_PAIRS_PER_SWEEP,
    max_n_cliffords: int | None = None,
    x90: Waveform | TargetMap[Waveform] | None = None,
    zx90: PulseSchedule | None = None,
    mitigate_readout: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    time_integration: bool = True,
    sweep_timeout: float = _DEFAULT_SWEEP_TIMEOUT_SECONDS,
) -> Result:
    """
    Acquire raw reference/interleaved trials in randomized adjacent pairs.

    Parameters
    ----------
    exp : Experiment
        Configured Qubex experiment.
    target : str
        One 1Q or CR target label. This low-level primitive is single-target.
    interleaved_clifford : str | Clifford
        Clifford inserted after every random Clifford in the interleaved arm.
    interleaved_waveform : Waveform | PulseSchedule | None, optional
        Physical implementation of the interleaved gate. A 1Q target accepts a
        `Waveform` or a target-containing `PulseSchedule`; a 2Q target requires
        a `PulseSchedule`. Required when the existing RB sequence builder
        cannot infer it from `interleaved_clifford`.
    n_cliffords_range : ArrayLike | None, optional
        Fixed, strictly increasing Clifford lengths. At least four are required.
        Four or five points produce an analysis warning because the decay fit
        has little residual freedom. Mutually exclusive with `max_n_cliffords`.
        When omitted with no `max_n_cliffords`, use the target-specific fixed
        default grid: `[0, 1, 2, 4, 8, 16, 32, 64, 128]` for 2Q targets or
        `[0, 16, 32, 64, 128, 256, 512, 1024, 2048]` for 1Q targets.
    n_trials : int | None, optional
        Paired sequence trials per length. Must be at least 2 and defaults to
        30.
    seeds : ArrayLike | None, optional
        Explicit uint32 seed matrix with shape `(n_lengths, n_trials)`. Every
        cell must contain a distinct seed.
    sequence_seed : int | None, optional
        Master seed used to generate the seed matrix. Mutually exclusive with
        `seeds`. Defaults to `None`, which generates a fresh matrix.
    acquisition_seed : int | None, optional
        Seed for global pair order and balanced AB/BA order. Defaults to `None`,
        which draws a fresh randomized order for each acquisition.
    pairs_per_sweep : int | None, optional
        Number of complete adjacent pairs submitted in each measurement call.
        `None` submits all pairs in one measurement call and is the default.
        Pass a positive integer to split the acquisition into smaller sweep calls.
    max_n_cliffords : int | None, optional
        Upper bound for power-of-two lengths in the default grid. Mutually
        exclusive with `n_cliffords_range`. Defaults to the target-specific
        experiment constant.
    x90 : Waveform | TargetMap[Waveform] | None, optional
        Optional pi/2 pulse override used by the RB sequence builder. Use a
        `Waveform` or one-target mapping for 1Q and a two-qubit mapping for 2Q.
    zx90 : PulseSchedule | None, optional
        Optional ZX90 schedule override for a CR target. Ignored for 1Q.
    mitigate_readout : bool, optional
        Apply fixed-classifier readout mitigation to 2Q `P(00)`. Ignored for 1Q.
        Defaults to `True`.
    n_shots : int | None, optional
        Shots per sweep point. Defaults to the experiment constant.
    shot_interval : float | None, optional
        Shot interval in ns. Defaults to the experiment constant.
    time_integration : bool, optional
        Integrate readout data over time. Defaults to `True`.
    sweep_timeout : float, optional
        Timeout in seconds for each pair-preserving sweep chunk. Defaults to
        1800 seconds.

    Returns
    -------
    Result
        Raw paired trials, seed matrix, planned acquisition order, and chunk
        timestamps. Call `analyze_paired_irb` to fit without remeasurement.

    Raises
    ------
    TypeError
        Raised when an argument has an incompatible type.
    ValueError
        Raised when the target, Clifford, length grid, seeds, or measurement
        options are invalid.
    TimeoutError
        Raised when a pair-preserving sweep chunk exceeds `sweep_timeout`.
    RuntimeError
        Raised when the measurement backend returns incomplete or invalid data.

    Notes
    -----
    This function performs hardware acquisition and resets the AWG and capture
    units for the resolved qubits once before the first sweep chunk.

    The backend may batch or schedule-pack points according to its capabilities.
    Qubex therefore records honest chunk timestamps, not unavailable per-point
    hardware timestamps. Increasing `pairs_per_sweep` reduces service-call
    overhead while preserving adjacent reference/interleaved points inside
    every pair.

    The default `pairs_per_sweep=None` submits every randomized adjacent pair in
    one measurement call. Pass a positive integer to limit the number of pairs
    per call while preserving reference/interleaved adjacency inside each pair.

    The returned target payload contains `reference["trials"]` and
    `interleaved["trials"]` matrices with shape `(n_lengths, n_trials)`, plus
    `acquisition` and schema-versioned `metadata` mappings needed for later
    analysis.
    """
    context = _resolve_target_acquisition(
        exp,
        target,
        interleaved_clifford=interleaved_clifford,
        interleaved_waveform=interleaved_waveform,
        pairs_per_sweep=pairs_per_sweep,
        x90=x90,
        zx90=zx90,
        mitigate_readout=mitigate_readout,
        n_shots=DEFAULT_SHOTS if n_shots is None else n_shots,
        shot_interval=DEFAULT_INTERVAL if shot_interval is None else shot_interval,
        time_integration=time_integration,
        sweep_timeout=sweep_timeout,
    )
    resolved_trials = _resolve_paired_n_trials(n_trials)
    _validate_range_choice(n_cliffords_range, max_n_cliffords)
    n_cliffords = _resolve_fixed_n_cliffords(
        context.target_spec,
        n_cliffords_range,
        max_n_cliffords,
    )
    resolved_sequence_seed = _validate_optional_rng_seed(
        sequence_seed,
        name="sequence_seed",
    )
    seed_matrix = _resolve_seed_matrix(
        seeds,
        shape=(len(n_cliffords), resolved_trials),
        sequence_seed=resolved_sequence_seed,
    )
    resolved_acquisition_seed = _validate_optional_rng_seed(
        acquisition_seed,
        name="acquisition_seed",
    )
    return _run_single_target_acquisition(
        exp,
        target,
        context=context,
        n_cliffords=n_cliffords,
        seeds=seed_matrix,
        sequence_seed=resolved_sequence_seed,
        acquisition_seed=resolved_acquisition_seed,
    )


def _trial_moments(trials: NDArray[np.float64]) -> _TrialMoments:
    """Calculate mean, sample SD, and SEM for each Clifford length."""
    mean = np.mean(trials, axis=1)
    std = np.std(trials, axis=1, ddof=1)
    # NumPy's two-pass variance can leave round-off residue for a row whose
    # values are exactly identical. Preserve the statistical zero explicitly.
    constant_rows = np.all(trials == trials[:, :1], axis=1)
    std = np.where(constant_rows, 0.0, std)
    sem = std / math.sqrt(trials.shape[1])
    return _TrialMoments(mean=mean, std=std, sem=sem)


def _resolve_sem_floor(
    reference_sem: NDArray[np.float64],
    interleaved_sem: NDArray[np.float64],
    requested_floor: float | None,
) -> float:
    """Resolve one common empirical or user-specified SEM floor for both arms."""
    if requested_floor is not None:
        return requested_floor
    combined = np.concatenate((reference_sem, interleaved_sem))
    positive = combined[combined > 0.0]
    if len(positive) == 0:
        # With no empirical scatter, equal finite weights are preferable to
        # machine-epsilon weights. The absolute scale affects covariance only;
        # the weighted point estimate is unchanged when every weight is equal.
        return _ALL_ZERO_SEM_FLOOR
    empirical_floor = float(_EMPIRICAL_SEM_FLOOR_FRACTION * np.median(positive))
    return empirical_floor if empirical_floor > 0.0 else _ALL_ZERO_SEM_FLOOR


def _regularize_trial_moments(
    moments: _TrialMoments,
    *,
    sem_floor: float,
) -> _TrialStatistics:
    """Apply a resolved SEM floor to one arm's raw trial moments."""
    sem_used = np.maximum(moments.sem, sem_floor)
    return _TrialStatistics(
        mean=moments.mean,
        std=moments.std,
        sem=moments.sem,
        sem_used=sem_used,
        sem_floor_count=int(np.count_nonzero(moments.sem < sem_floor)),
    )


def _paired_trial_statistics(
    reference_trials: NDArray[np.float64],
    interleaved_trials: NDArray[np.float64],
    *,
    sem_floor: float | None,
) -> _PairedStatistics:
    """Calculate paired statistics using a common resolved SEM floor."""
    reference_moments = _trial_moments(reference_trials)
    interleaved_moments = _trial_moments(interleaved_trials)
    resolved_floor = _resolve_sem_floor(
        reference_moments.sem,
        interleaved_moments.sem,
        sem_floor,
    )
    combined_sem = np.concatenate((reference_moments.sem, interleaved_moments.sem))
    # An automatic floor is estimated from the positive SEM values themselves.
    # A fixed floor supplies no empirical covariance scale when it replaces all
    # observed SEM values, even if those values are nonzero.
    fit_covariance_scale_empirical = bool(
        np.any(combined_sem > 0.0)
        if sem_floor is None
        else np.any(combined_sem >= resolved_floor)
    )
    return _PairedStatistics(
        reference=_regularize_trial_moments(
            reference_moments,
            sem_floor=resolved_floor,
        ),
        interleaved=_regularize_trial_moments(
            interleaved_moments,
            sem_floor=resolved_floor,
        ),
        sem_floor=resolved_floor,
        fit_covariance_scale_empirical=fit_covariance_scale_empirical,
    )


def _initial_fit_parameters(
    n_cliffords: NDArray[np.int64],
    mean: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Return an interior starting point for bounded RB fitting."""
    epsilon = 1e-6
    offset = float(np.clip(np.min(mean), epsilon, 1.0 - epsilon))
    amplitude = float(np.clip(mean[0] - offset, epsilon, 1.0 - epsilon))
    length_scale = max(float(n_cliffords[-1]), 1.0)
    initial_p = float(np.exp(-1.0 / length_scale))
    return amplitude, initial_p, offset


def _fit_rb_decay(
    n_cliffords: NDArray[np.int64],
    mean: NDArray[np.float64],
    sem_used: NDArray[np.float64],
) -> _DecayFitResult:
    """Fit `A * p**m + C` with absolute SEM weights and no plotting."""
    if not (
        np.all(np.isfinite(mean))
        and np.all(np.isfinite(sem_used))
        and np.all(sem_used > 0)
    ):
        raise ValueError(
            "RB fit inputs must be finite and SEM weights must be positive."
        )
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always", OptimizeWarning)
        parameters, covariance = curve_fit(
            _rb_decay,
            n_cliffords.astype(np.float64),
            mean,
            p0=_initial_fit_parameters(n_cliffords, mean),
            sigma=sem_used,
            absolute_sigma=True,
            bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            maxfev=20_000,
        )
    parameters = np.asarray(parameters, dtype=np.float64)
    raw_covariance = np.asarray(covariance, dtype=np.float64)
    if not np.all(np.isfinite(parameters)):
        raise RuntimeError("RB fit returned non-finite parameters.")
    covariance_valid = bool(
        raw_covariance.shape == (3, 3)
        and np.all(np.isfinite(raw_covariance))
        and np.all(np.diag(raw_covariance) >= 0)
        and not any(
            isinstance(warning.message, OptimizeWarning) for warning in fit_warnings
        )
    )
    covariance = raw_covariance if covariance_valid else None
    standard_errors = np.sqrt(np.diag(raw_covariance)) if covariance_valid else None
    predicted = _rb_decay(n_cliffords.astype(np.float64), *parameters)
    residuals = mean - predicted
    weighted_residuals = residuals / sem_used
    chi_square = float(np.sum(weighted_residuals**2))
    degrees_of_freedom = len(n_cliffords) - len(parameters)
    reduced_chi_square = (
        chi_square / degrees_of_freedom if degrees_of_freedom > 0 else None
    )
    centered_sum_squares = float(np.sum((mean - np.mean(mean)) ** 2))
    r_squared = (
        1.0 - float(np.sum(residuals**2)) / centered_sum_squares
        if centered_sum_squares > 0
        else None
    )
    names = ("A", "p", "C")
    at_bound = tuple(
        name
        for name, value in zip(names, parameters, strict=True)
        if np.isclose(value, 0.0, atol=1e-6, rtol=0.0)
        or np.isclose(value, 1.0, atol=1e-6, rtol=0.0)
    )
    return _DecayFitResult(
        amplitude=float(parameters[0]),
        decay_parameter=float(parameters[1]),
        offset=float(parameters[2]),
        covariance=covariance,
        standard_errors=standard_errors,
        predicted=predicted,
        residuals=residuals,
        weighted_residuals=weighted_residuals,
        chi_square=chi_square,
        reduced_chi_square=reduced_chi_square,
        r_squared=r_squared,
        parameters_at_bound=at_bound,
    )


def _fit_pilot_decay(
    n_cliffords: NDArray[np.int64],
    mean: NDArray[np.float64],
    *,
    dimension: int,
) -> _PilotDecayFitResult:
    """Fit a coarse pilot decay with the asymptote fixed to `1 / dimension`."""
    if dimension not in (2, 4):
        raise ValueError("Pilot RB dimension must be 2 or 4.")
    if n_cliffords.ndim != 1 or mean.shape != n_cliffords.shape:
        raise ValueError("Pilot RB lengths and means must be matching 1D arrays.")
    if not np.all(np.isfinite(mean)):
        raise ValueError("Pilot RB means must be finite.")

    offset = 1.0 / dimension
    maximum_amplitude = 1.0 - offset
    epsilon = 1e-6
    initial_amplitude = float(
        np.clip(mean[0] - offset, epsilon, maximum_amplitude - epsilon)
    )

    def fixed_offset_decay(
        lengths: NDArray[np.float64],
        amplitude: float,
        decay_parameter: float,
    ) -> NDArray[np.float64]:
        return _rb_decay(lengths, amplitude, decay_parameter, offset)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        parameters, _ = curve_fit(
            fixed_offset_decay,
            n_cliffords.astype(np.float64),
            mean,
            p0=(initial_amplitude, 0.98),
            bounds=((0.0, 0.0), (maximum_amplitude, 1.0)),
            maxfev=20_000,
        )
    parameters = np.asarray(parameters, dtype=np.float64)
    if not np.all(np.isfinite(parameters)):
        raise RuntimeError("Pilot RB fit returned non-finite parameters.")
    predicted = fixed_offset_decay(
        n_cliffords.astype(np.float64),
        float(parameters[0]),
        float(parameters[1]),
    )
    residuals = mean - predicted
    centered_sum_squares = float(np.sum((mean - np.mean(mean)) ** 2))
    r_squared = (
        1.0 - float(np.sum(residuals**2)) / centered_sum_squares
        if centered_sum_squares > np.finfo(np.float64).eps
        else None
    )
    at_bound = tuple(
        name
        for name, value, upper in zip(
            ("A", "p"),
            parameters,
            (maximum_amplitude, 1.0),
            strict=True,
        )
        if np.isclose(value, 0.0, atol=_PILOT_BOUND_ATOL, rtol=0.0)
        or np.isclose(value, upper, atol=_PILOT_BOUND_ATOL, rtol=0.0)
    )
    return _PilotDecayFitResult(
        amplitude=float(parameters[0]),
        decay_parameter=float(parameters[1]),
        offset=offset,
        predicted=predicted,
        residuals=residuals,
        r_squared=r_squared,
        parameters_at_bound=at_bound,
    )


def _observed_decay_significance(
    mean: NDArray[np.float64],
    sem: NDArray[np.float64],
) -> float:
    """Return endpoint decay divided by its independent-SEM uncertainty."""
    if mean.ndim != 1 or sem.shape != mean.shape or len(mean) < 2:
        raise ValueError("Pilot means and SEM values must be matching 1D arrays.")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(sem))):
        raise ValueError("Pilot means and SEM values must be finite.")
    uncertainty = float(math.hypot(sem[0], sem[-1]))
    observed_decay = float(mean[0] - mean[-1])
    if uncertainty > 0.0:
        return observed_decay / uncertainty
    if observed_decay > 0.0:
        return math.inf
    if observed_decay < 0.0:
        return -math.inf
    return 0.0


def _pilot_fit_payload(fit: _PilotDecayFitResult) -> dict[str, object]:
    """Convert one fixed-offset pilot fit into an auditable payload."""
    return {
        "model": "A * p**m + 1/d",
        "A": fit.amplitude,
        "p": fit.decay_parameter,
        "C": fit.offset,
        "predicted": fit.predicted,
        "residuals": fit.residuals,
        "r_squared": fit.r_squared,
        "parameters_at_bound": fit.parameters_at_bound,
        "amplitude_lower_bound_hit": fit.amplitude_lower_bound_hit,
        "amplitude_upper_bound_hit": fit.amplitude_upper_bound_hit,
        "weighting": "unweighted",
    }


def _pilot_model_payload(
    *,
    dimension: int,
    unmitigated_two_qubit_readout: bool,
) -> dict[str, object]:
    """Describe the pilot-only fixed-offset approximation and its caveat."""
    caveat = (
        "Asymmetric assignment errors can shift the observed unmitigated 2Q "
        "asymptote away from 1/d."
        if unmitigated_two_qubit_readout
        else None
    )
    return {
        "model": "A * p**m + 1/d",
        "fixed_offset": 1.0 / dimension,
        "purpose": "range_selection_only",
        "used_for_main_fit": False,
        "unmitigated_2q_readout_caveat": caveat,
    }


def _fit_payload(fit: _DecayFitResult) -> dict[str, object]:
    """Convert a private fit value into a stable result payload."""
    standard_errors = fit.standard_errors
    return {
        "model": "A * p**m + C",
        "A": fit.amplitude,
        "p": fit.decay_parameter,
        "C": fit.offset,
        "covariance": fit.covariance,
        "standard_errors": standard_errors,
        "A_err": float(standard_errors[0]) if standard_errors is not None else None,
        "p_err": float(standard_errors[1]) if standard_errors is not None else None,
        "C_err": float(standard_errors[2]) if standard_errors is not None else None,
        "predicted": fit.predicted,
        "residuals": fit.residuals,
        "weighted_residuals": fit.weighted_residuals,
        "chi_square": fit.chi_square,
        "reduced_chi_square": fit.reduced_chi_square,
        "r_squared": fit.r_squared,
        "parameters_at_bound": fit.parameters_at_bound,
        "covariance_valid": fit.covariance_valid,
        "absolute_sigma": True,
    }


def _curve_payload(
    trials: NDArray[np.float64],
    statistics: _TrialStatistics,
    fit: _DecayFitResult,
) -> dict[str, object]:
    """Build one reference or interleaved analysis payload."""
    return {
        "trials": trials,
        "mean": statistics.mean,
        "std": statistics.std,
        "sem": statistics.sem,
        "sem_used": statistics.sem_used,
        "fit": _fit_payload(fit),
    }


def _gate_fidelity(
    p_reference: float,
    p_interleaved: float,
    *,
    dimension: int,
) -> float:
    """Calculate standard IRB gate fidelity without physical-range clipping."""
    if not np.isfinite(p_reference) or not np.isfinite(p_interleaved):
        raise ValueError("Decay parameters must be finite.")
    if p_reference == 0.0:
        raise ValueError("Reference decay parameter is zero.")
    factor = (dimension - 1.0) / dimension
    return float(1.0 - factor * (1.0 - p_interleaved / p_reference))


def _stratified_paired_resample_indices(
    first_protocols: NDArray[np.str_],
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample paired trial indices within each length and AB/BA stratum."""
    if first_protocols.ndim != 2:
        raise ValueError("Bootstrap protocol strata must be two-dimensional.")
    indices = np.empty(first_protocols.shape, dtype=np.int64)
    for length_index, protocol_row in enumerate(first_protocols):
        for protocol in (_PROTOCOL_REFERENCE, _PROTOCOL_INTERLEAVED):
            stratum_indices = np.flatnonzero(protocol_row == protocol)
            if len(stratum_indices) == 0:
                raise ValueError(
                    "Each Clifford length must contain both AB/BA bootstrap strata."
                )
            indices[length_index, stratum_indices] = rng.choice(
                stratum_indices,
                size=len(stratum_indices),
                replace=True,
            )
    return indices


def _bootstrap_stratum_counts(
    first_protocols: NDArray[np.str_],
) -> tuple[dict[str, int], ...]:
    """Return the retained reference-first and interleaved-first counts."""
    return tuple(
        {
            "reference_first": int(
                np.count_nonzero(protocol_row == _PROTOCOL_REFERENCE)
            ),
            "interleaved_first": int(
                np.count_nonzero(protocol_row == _PROTOCOL_INTERLEAVED)
            ),
        }
        for protocol_row in first_protocols
    )


def _bootstrap_analysis(
    n_cliffords: NDArray[np.int64],
    reference_trials: NDArray[np.float64],
    interleaved_trials: NDArray[np.float64],
    first_protocols: NDArray[np.str_],
    *,
    dimension: int,
    n_bootstrap: int,
    bootstrap_seed: int | None,
    sem_floor: float | None,
) -> _BootstrapResult:
    """Repeat the paired statistics, fit, and fidelity chain within AB/BA strata."""
    stratum_counts = _bootstrap_stratum_counts(first_protocols)
    resampling_valid = all(
        counts["reference_first"] >= _MIN_BOOTSTRAP_SAMPLES_PER_STRATUM
        and counts["interleaved_first"] >= _MIN_BOOTSTRAP_SAMPLES_PER_STRATUM
        for counts in stratum_counts
    )
    if n_bootstrap == 0 or not resampling_valid:
        empty = np.asarray([], dtype=np.float64)
        return _BootstrapResult(
            seed=bootstrap_seed,
            n_requested=n_bootstrap,
            fidelities=empty,
            p_reference=empty.copy(),
            p_interleaved=empty.copy(),
            p_correlation=None,
            stratum_counts=stratum_counts,
            reference_any_bound_count=0,
            interleaved_any_bound_count=0,
            reference_p_bound_count=0,
            interleaved_p_bound_count=0,
        )

    rng = np.random.default_rng(bootstrap_seed)
    fidelities: list[float] = []
    p_reference_samples: list[float] = []
    p_interleaved_samples: list[float] = []
    reference_any_bound_count = 0
    interleaved_any_bound_count = 0
    reference_p_bound_count = 0
    interleaved_p_bound_count = 0
    for _ in range(n_bootstrap):
        indices = _stratified_paired_resample_indices(
            first_protocols,
            rng=rng,
        )
        row_indices = np.arange(len(n_cliffords))[:, None]
        reference_sample = reference_trials[row_indices, indices]
        interleaved_sample = interleaved_trials[row_indices, indices]
        try:
            statistics = _paired_trial_statistics(
                reference_sample,
                interleaved_sample,
                sem_floor=sem_floor,
            )
            reference_fit = _fit_rb_decay(
                n_cliffords,
                statistics.reference.mean,
                statistics.reference.sem_used,
            )
            interleaved_fit = _fit_rb_decay(
                n_cliffords,
                statistics.interleaved.mean,
                statistics.interleaved.sem_used,
            )
            fidelity = _gate_fidelity(
                reference_fit.decay_parameter,
                interleaved_fit.decay_parameter,
                dimension=dimension,
            )
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            continue
        if not np.isfinite(fidelity):
            continue
        fidelities.append(fidelity)
        p_reference_samples.append(reference_fit.decay_parameter)
        p_interleaved_samples.append(interleaved_fit.decay_parameter)
        reference_any_bound_count += int(bool(reference_fit.parameters_at_bound))
        interleaved_any_bound_count += int(bool(interleaved_fit.parameters_at_bound))
        reference_p_bound_count += int("p" in reference_fit.parameters_at_bound)
        interleaved_p_bound_count += int("p" in interleaved_fit.parameters_at_bound)

    fidelity_array = np.asarray(fidelities, dtype=np.float64)
    p_reference_array = np.asarray(p_reference_samples, dtype=np.float64)
    p_interleaved_array = np.asarray(p_interleaved_samples, dtype=np.float64)
    n_success = len(fidelity_array)
    correlation: float | None = None
    if (
        n_success >= 2
        and np.std(p_reference_array) > 0
        and np.std(p_interleaved_array) > 0
    ):
        candidate = float(np.corrcoef(p_reference_array, p_interleaved_array)[0, 1])
        if np.isfinite(candidate):
            correlation = candidate
    return _BootstrapResult(
        seed=bootstrap_seed,
        n_requested=n_bootstrap,
        fidelities=fidelity_array,
        p_reference=p_reference_array,
        p_interleaved=p_interleaved_array,
        p_correlation=correlation,
        stratum_counts=stratum_counts,
        reference_any_bound_count=reference_any_bound_count,
        interleaved_any_bound_count=interleaved_any_bound_count,
        reference_p_bound_count=reference_p_bound_count,
        interleaved_p_bound_count=interleaved_p_bound_count,
    )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    """Narrow a result payload node to a string-keyed mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    return cast(Mapping[str, object], value)


def _validated_dimension(metadata: Mapping[str, object]) -> int:
    """Return a valid dimension consistent with optional target-kind metadata."""
    try:
        dimension = _validate_positive_integer(
            metadata["dimension"],
            name="metadata.dimension",
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "Raw paired IRB metadata must contain a valid dimension."
        ) from None
    if dimension not in (2, 4):
        raise ValueError("Paired IRB dimension must be 2 or 4.")
    if "is_two_qubit" not in metadata:
        return dimension
    try:
        is_two_qubit = _validate_boolean(
            metadata["is_two_qubit"],
            name="metadata.is_two_qubit",
        )
    except TypeError:
        raise ValueError(
            "Raw paired IRB metadata must contain a valid is_two_qubit flag."
        ) from None
    expected_dimension = 4 if is_two_qubit else 2
    if dimension != expected_dimension:
        raise ValueError(
            "`metadata.dimension` is inconsistent with `metadata.is_two_qubit`."
        )
    return dimension


def _validated_trial_matrices(
    reference: Mapping[str, object],
    interleaved: Mapping[str, object],
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]]:
    """Return finite, shape-compatible paired trial matrices."""
    reference_trials = np.asarray(reference.get("trials"), dtype=np.float64)
    interleaved_trials = np.asarray(interleaved.get("trials"), dtype=np.float64)
    if reference_trials.ndim != 2 or interleaved_trials.ndim != 2:
        raise ValueError("Reference and interleaved trials must be two-dimensional.")
    if reference_trials.shape != interleaved_trials.shape:
        raise ValueError("Reference and interleaved trials must have the same shape.")
    if reference_trials.shape[1] < 2:
        raise ValueError("Paired IRB analysis requires at least two trials.")
    if not np.all(np.isfinite(reference_trials)) or not np.all(
        np.isfinite(interleaved_trials)
    ):
        raise ValueError("Raw paired trials must be finite.")
    shape = (reference_trials.shape[0], reference_trials.shape[1])
    return reference_trials, interleaved_trials, shape


def _validated_first_protocols(
    acquisition: Mapping[str, object],
    *,
    shape: tuple[int, int],
) -> NDArray[np.str_]:
    """Reconstruct and validate per-cell AB/BA strata from the acquisition plan."""
    raw_plan = acquisition.get("planned_order")
    if isinstance(raw_plan, (str, bytes)) or not isinstance(raw_plan, Sequence):
        raise TypeError("Acquisition `planned_order` must be a sequence.")
    if len(raw_plan) != int(np.prod(shape)):
        raise ValueError(
            "Acquisition `planned_order` must contain one entry per trial cell."
        )
    first_protocols = np.full(shape, "", dtype="<U11")
    for pair_order_index, raw_pair in enumerate(raw_plan):
        pair = _mapping(
            raw_pair,
            name=f"acquisition.planned_order[{pair_order_index}]",
        )
        length_index = _validate_nonnegative_integer(
            pair.get("length_index"),
            name=f"planned_order[{pair_order_index}].length_index",
        )
        trial_index = _validate_nonnegative_integer(
            pair.get("trial_index"),
            name=f"planned_order[{pair_order_index}].trial_index",
        )
        if length_index >= shape[0] or trial_index >= shape[1]:
            raise ValueError("Acquisition `planned_order` contains an invalid cell.")
        if first_protocols[length_index, trial_index]:
            raise ValueError("Acquisition `planned_order` contains a duplicate cell.")
        first_protocol = pair.get("first_protocol")
        if not isinstance(first_protocol, str) or first_protocol not in (
            _PROTOCOL_REFERENCE,
            _PROTOCOL_INTERLEAVED,
        ):
            raise ValueError(
                "Each acquisition-plan entry must have a valid `first_protocol`."
            )
        first_protocols[length_index, trial_index] = cast(str, first_protocol)
    for row in first_protocols:
        n_reference_first = int(np.count_nonzero(row == _PROTOCOL_REFERENCE))
        n_interleaved_first = int(np.count_nonzero(row == _PROTOCOL_INTERLEAVED))
        if abs(n_reference_first - n_interleaved_first) > 1:
            raise ValueError(
                "Acquisition `planned_order` must balance AB/BA within each length."
            )
    return first_protocols


def _extract_raw_payload(
    raw_result: Result | Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    """Return the sole target and its raw acquisition payload."""
    data: Mapping[str, object]
    if isinstance(raw_result, Result):
        data = raw_result.data
    elif isinstance(raw_result, Mapping):
        data = raw_result
    else:
        raise TypeError("`raw_result` must be a Result or mapping.")
    if len(data) != 1:
        raise ValueError("Paired IRB analysis requires exactly one target entry.")
    target = next(iter(data))
    if not isinstance(target, str):
        raise TypeError("The paired IRB target key must be a string.")
    return target, _mapping(data[target], name=f"raw_result[{target!r}]")


def _validated_raw_data(
    target_payload: Mapping[str, object],
) -> _RawIRBData:
    """Read and validate paired matrices plus acquisition metadata."""
    reference = _mapping(target_payload.get("reference"), name="reference")
    interleaved = _mapping(target_payload.get("interleaved"), name="interleaved")
    acquisition = _mapping(target_payload.get("acquisition"), name="acquisition")
    metadata = _mapping(target_payload.get("metadata"), name="metadata")
    schema_version = metadata.get("schema_version")
    if (
        isinstance(schema_version, (bool, np.bool_))
        or not isinstance(schema_version, (int, np.integer))
        or int(schema_version) != _RAW_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported paired IRB schema_version "
            f"{schema_version!r}; expected {_RAW_SCHEMA_VERSION}."
        )
    dimension = _validated_dimension(metadata)
    reference_trials, interleaved_trials, trial_shape = _validated_trial_matrices(
        reference,
        interleaved,
    )
    raw_n_cliffords = acquisition.get("n_cliffords")
    if raw_n_cliffords is None:
        raise ValueError("Acquisition metadata must contain `n_cliffords`.")
    n_cliffords = _resolve_n_cliffords(
        cast(ArrayLike, raw_n_cliffords),
        maximum=DEFAULT_MAX_N_CLIFFORDS_1Q,
    )
    if len(n_cliffords) != reference_trials.shape[0]:
        raise ValueError(
            "The number of Clifford lengths must match the trial-matrix rows."
        )
    raw_seeds = acquisition.get("seeds")
    if raw_seeds is None:
        raise ValueError("Acquisition metadata must contain `seeds`.")
    _resolve_seed_matrix(
        cast(ArrayLike, raw_seeds),
        shape=trial_shape,
        sequence_seed=None,
    )
    raw_n_trials = acquisition.get("n_trials")
    if raw_n_trials is not None:
        resolved_n_trials = _validate_positive_integer(
            raw_n_trials,
            name="acquisition.n_trials",
        )
        if resolved_n_trials != reference_trials.shape[1]:
            raise ValueError(
                "Acquisition `n_trials` must match the trial-matrix columns."
            )
    first_protocols = _validated_first_protocols(
        acquisition,
        shape=trial_shape,
    )
    return _RawIRBData(
        n_cliffords=n_cliffords,
        reference_trials=reference_trials,
        interleaved_trials=interleaved_trials,
        first_protocols=first_protocols,
        dimension=dimension,
        acquisition=acquisition,
        metadata=metadata,
    )


def _bootstrap_warning_messages(bootstrap: _BootstrapResult) -> list[str]:
    """Return bootstrap design, optimizer, and bound-hit diagnostics."""
    messages: list[str] = []
    if bootstrap.n_requested > 0:
        if not bootstrap.resampling_valid:
            messages.append(
                "Primary paired-bootstrap uncertainty is unavailable because at "
                "least two samples per AB/BA stratum are required at every "
                "Clifford length. Bootstrap fitting was skipped."
            )
        elif bootstrap.n_success < 2:
            messages.append(
                "Fewer than two paired bootstrap replicates produced valid fits; "
                "uncertainty intervals are unavailable."
            )
        elif not bootstrap.optimizer_valid:
            messages.append(
                f"Fewer than {_MIN_BOOTSTRAP_SUCCESS_RATE:.0%} of paired bootstrap "
                "replicates produced valid fits; "
                "bootstrap spread summaries are diagnostic only and primary "
                "uncertainty is unavailable."
            )
    for protocol, any_fraction, p_fraction in (
        (
            _PROTOCOL_REFERENCE,
            bootstrap.reference_any_bound_fraction,
            bootstrap.reference_p_bound_fraction,
        ),
        (
            _PROTOCOL_INTERLEAVED,
            bootstrap.interleaved_any_bound_fraction,
            bootstrap.interleaved_p_bound_fraction,
        ),
    ):
        if (
            any_fraction is not None
            and p_fraction is not None
            and (
                any_fraction > _MAX_BOOTSTRAP_ANY_BOUND_FRACTION
                or p_fraction > _MAX_BOOTSTRAP_P_BOUND_FRACTION
            )
        ):
            messages.append(
                f"The {protocol} bootstrap fit has excessive parameter-bound "
                f"hits (any={any_fraction:.1%}, p={p_fraction:.1%}); bootstrap "
                "fit quality is unreliable."
            )
    return messages


def _curve_fit_warning_messages(
    reference_fit: _DecayFitResult,
    interleaved_fit: _DecayFitResult,
) -> list[str]:
    """Return bound, covariance, and goodness-of-fit warnings for both arms."""
    messages: list[str] = []
    for protocol, fit in (
        (_PROTOCOL_REFERENCE, reference_fit),
        (_PROTOCOL_INTERLEAVED, interleaved_fit),
    ):
        if fit.parameters_at_bound:
            messages.append(
                f"The {protocol} fit has parameters at a bound: "
                f"{', '.join(fit.parameters_at_bound)}."
            )
        if not fit.covariance_valid:
            messages.append(f"The {protocol} fit covariance is unavailable.")
    return messages


def _analysis_warnings(
    *,
    n_trials: int,
    n_clifford_points: int,
    reference_statistics: _TrialStatistics,
    interleaved_statistics: _TrialStatistics,
    fit_covariance_scale_empirical: bool,
    reference_fit: _DecayFitResult,
    interleaved_fit: _DecayFitResult,
    gate_fidelity: float,
    bootstrap: _BootstrapResult,
) -> list[str]:
    """Collect and emit fit and bootstrap diagnostics requiring attention."""
    messages: list[str] = []
    if n_trials < _MIN_RECOMMENDED_TRIALS:
        messages.append(
            f"n_trials={n_trials} is below the recommended minimum of "
            f"{_MIN_RECOMMENDED_TRIALS}."
        )
    if n_clifford_points < _MIN_RECOMMENDED_CLIFFORD_POINTS:
        messages.append(
            f"Only {n_clifford_points} Clifford-length points were provided; "
            f"at least {_MIN_RECOMMENDED_CLIFFORD_POINTS} are recommended for "
            "the three-parameter decay fit."
        )
    floor_count = (
        reference_statistics.sem_floor_count + interleaved_statistics.sem_floor_count
    )
    if floor_count:
        messages.append(f"The SEM floor was applied at {floor_count} points.")
    if not fit_covariance_scale_empirical:
        messages.append(
            "No empirical SEM values contribute to the effective fit weights; "
            "the fit-covariance scale is non-empirical and covariance-derived "
            "fidelity uncertainty is unavailable."
        )
    messages.extend(_curve_fit_warning_messages(reference_fit, interleaved_fit))
    if not 0.0 <= gate_fidelity <= 1.0:
        messages.append(
            "The finite-sample gate-fidelity estimate lies outside [0, 1] and "
            "was intentionally not clipped."
        )
    messages.extend(_bootstrap_warning_messages(bootstrap))
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=4)
    return messages


def _fit_covariance_fidelity_error(
    reference_fit: _DecayFitResult,
    interleaved_fit: _DecayFitResult,
    *,
    dimension: int,
) -> float | None:
    """Propagate independent fit covariance as a diagnostic only."""
    reference_errors = reference_fit.standard_errors
    interleaved_errors = interleaved_fit.standard_errors
    if reference_errors is None or interleaved_errors is None:
        return None
    p_reference = reference_fit.decay_parameter
    p_interleaved = interleaved_fit.decay_parameter
    factor = (dimension - 1.0) / dimension
    p_reference_error = float(reference_errors[1])
    p_interleaved_error = float(interleaved_errors[1])
    variance = factor**2 * (
        (p_interleaved_error / p_reference) ** 2
        + (p_reference_error * p_interleaved / p_reference**2) ** 2
    )
    if not np.isfinite(variance) or variance < 0:
        return None
    return float(math.sqrt(variance))


def _make_figure(
    target: str,
    n_cliffords: NDArray[np.int64],
    reference_statistics: _TrialStatistics,
    interleaved_statistics: _TrialStatistics,
    reference_fit: _DecayFitResult,
    interleaved_fit: _DecayFitResult,
    *,
    gate_fidelity: float,
    gate_fidelity_error: float | None,
    gate_fidelity_ci95: tuple[float, float] | None,
    error_bar: _ErrorBar,
    interleaved_clifford_name: str | None,
) -> go.Figure:
    """Build a paired IRB figure in the established Qubex IRB style."""
    figure = viz.make_figure()
    dense_x = np.linspace(float(n_cliffords[0]), float(n_cliffords[-1]), 1_000)
    for name, color, statistics, fit in (
        ("Reference", viz.COLORS[0], reference_statistics, reference_fit),
        ("Interleaved", viz.COLORS[1], interleaved_statistics, interleaved_fit),
    ):
        if error_bar == "sem":
            error_values = statistics.sem
        elif error_bar == "std":
            error_values = statistics.std
        else:
            error_values = None
        dense_y = _rb_decay(
            dense_x,
            fit.amplitude,
            fit.decay_parameter,
            fit.offset,
        )
        figure.add_trace(
            go.Scatter(
                x=dense_x,
                y=dense_y,
                mode="lines",
                name=name,
                line={"color": color},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=n_cliffords,
                y=statistics.mean,
                mode="markers",
                error_y={"type": "data", "array": error_values}
                if error_values is not None
                else None,
                marker={"color": color},
                showlegend=False,
            )
        )
    uncertainty_text = (
        "unavailable"
        if gate_fidelity_error is None
        else f"{100.0 * gate_fidelity_error:.4f} pp"
    )
    ci_text = (
        "unavailable"
        if gate_fidelity_ci95 is None
        else (
            f"[{100.0 * gate_fidelity_ci95[0]:.4f}, "
            f"{100.0 * gate_fidelity_ci95[1]:.4f}] %"
        )
    )
    reference_chi_text = (
        "unavailable"
        if reference_fit.reduced_chi_square is None
        else f"{reference_fit.reduced_chi_square:.3g}"
    )
    interleaved_chi_text = (
        "unavailable"
        if interleaved_fit.reduced_chi_square is None
        else f"{interleaved_fit.reduced_chi_square:.3g}"
    )
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=(
            f"F = {100.0 * gate_fidelity:.4f} %<br>"
            f"bootstrap σ = {uncertainty_text}<br>"
            f"95% CI = {ci_text}<br>"
            f"reduced χ² (ref / IRB) = "
            f"{reference_chi_text} / {interleaved_chi_text}"
        ),
        showarrow=False,
        align="right",
    )
    gate_name = f" of {interleaved_clifford_name}" if interleaved_clifford_name else ""
    figure.update_layout(
        title=f"Paired interleaved randomized benchmarking{gate_name} : {target}",
        xaxis_title="Number of Cliffords",
        xaxis_type="linear",
        yaxis_title="Normalized signal",
        yaxis_type="linear",
    )
    return figure


def _analyze_paired_irb(
    raw_result: Result | Mapping[str, object],
    *,
    options: _AnalysisOptions,
) -> Result:
    """Run paired IRB analysis with options validated by a public entry point."""
    target, target_payload = _extract_raw_payload(raw_result)
    raw_data = _validated_raw_data(target_payload)
    dimension = raw_data.dimension

    statistics = _paired_trial_statistics(
        raw_data.reference_trials,
        raw_data.interleaved_trials,
        sem_floor=options.sem_floor,
    )
    reference_statistics = statistics.reference
    interleaved_statistics = statistics.interleaved
    reference_fit = _fit_rb_decay(
        raw_data.n_cliffords,
        reference_statistics.mean,
        reference_statistics.sem_used,
    )
    interleaved_fit = _fit_rb_decay(
        raw_data.n_cliffords,
        interleaved_statistics.mean,
        interleaved_statistics.sem_used,
    )
    gate_fidelity = _gate_fidelity(
        reference_fit.decay_parameter,
        interleaved_fit.decay_parameter,
        dimension=dimension,
    )
    bootstrap = _bootstrap_analysis(
        raw_data.n_cliffords,
        raw_data.reference_trials,
        raw_data.interleaved_trials,
        raw_data.first_protocols,
        dimension=dimension,
        n_bootstrap=options.n_bootstrap,
        bootstrap_seed=options.bootstrap_seed,
        sem_floor=options.sem_floor,
    )
    if bootstrap.valid:
        gate_fidelity_error = bootstrap.fidelity_std
        gate_fidelity_ci68 = bootstrap.fidelity_ci68
        gate_fidelity_ci95 = bootstrap.fidelity_ci95
        uncertainty_method = "paired_bootstrap"
    else:
        gate_fidelity_error = None
        gate_fidelity_ci68 = None
        gate_fidelity_ci95 = None
        if bootstrap.n_requested > 0 and not bootstrap.resampling_valid:
            uncertainty_method = "paired_bootstrap_resampling_invalid"
        elif bootstrap.n_success >= 2:
            uncertainty_method = "paired_bootstrap_unreliable"
        else:
            uncertainty_method = "unavailable"
    warning_messages = _analysis_warnings(
        n_trials=raw_data.reference_trials.shape[1],
        n_clifford_points=len(raw_data.n_cliffords),
        reference_statistics=reference_statistics,
        interleaved_statistics=interleaved_statistics,
        fit_covariance_scale_empirical=(statistics.fit_covariance_scale_empirical),
        reference_fit=reference_fit,
        interleaved_fit=interleaved_fit,
        gate_fidelity=gate_fidelity,
        bootstrap=bootstrap,
    )
    reference_payload = _curve_payload(
        raw_data.reference_trials,
        reference_statistics,
        reference_fit,
    )
    interleaved_payload = _curve_payload(
        raw_data.interleaved_trials,
        interleaved_statistics,
        interleaved_fit,
    )
    fit_covariance_error = (
        _fit_covariance_fidelity_error(
            reference_fit,
            interleaved_fit,
            dimension=dimension,
        )
        if statistics.fit_covariance_scale_empirical
        else None
    )
    parameter_bound_quality_valid = not (
        reference_fit.parameters_at_bound or interleaved_fit.parameters_at_bound
    )
    analyzed_payload: dict[str, object] = {
        "gate_fidelity": gate_fidelity,
        "gate_error": 1.0 - gate_fidelity,
        "gate_fidelity_err": gate_fidelity_error,
        "gate_fidelity_ci68": gate_fidelity_ci68,
        "gate_fidelity_ci95": gate_fidelity_ci95,
        "reference": reference_payload,
        "interleaved": interleaved_payload,
        "acquisition": dict(raw_data.acquisition),
        "metadata": dict(raw_data.metadata),
        "uncertainty_method": uncertainty_method,
        "bootstrap": bootstrap.to_payload(),
        "diagnostics": {
            "warnings": tuple(warning_messages),
            "parameter_bound_quality_valid": parameter_bound_quality_valid,
            "reference_p_at_bound": "p" in reference_fit.parameters_at_bound,
            "interleaved_p_at_bound": "p" in interleaved_fit.parameters_at_bound,
            "bootstrap_parameter_bound_quality_valid": (
                bootstrap.parameter_bound_quality_valid
            ),
            "bootstrap_resampling_valid": bootstrap.resampling_valid,
            "bootstrap_optimizer_valid": bootstrap.optimizer_valid,
            "bootstrap_parameters_at_bound": {
                "reference": {
                    "any_count": bootstrap.reference_any_bound_count,
                    "any_fraction": bootstrap.reference_any_bound_fraction,
                    "p_count": bootstrap.reference_p_bound_count,
                    "p_fraction": bootstrap.reference_p_bound_fraction,
                },
                "interleaved": {
                    "any_count": bootstrap.interleaved_any_bound_count,
                    "any_fraction": bootstrap.interleaved_any_bound_fraction,
                    "p_count": bootstrap.interleaved_p_bound_count,
                    "p_fraction": bootstrap.interleaved_p_bound_fraction,
                },
            },
            "sem_floor": statistics.sem_floor,
            "sem_floor_mode": ("automatic" if options.sem_floor is None else "fixed"),
            "sem_floor_applied": {
                "reference": reference_statistics.sem_floor_count,
                "interleaved": interleaved_statistics.sem_floor_count,
            },
            "parameters_at_bound": {
                "reference": reference_fit.parameters_at_bound,
                "interleaved": interleaved_fit.parameters_at_bound,
            },
            "reduced_chi_square": {
                "reference": reference_fit.reduced_chi_square,
                "interleaved": interleaved_fit.reduced_chi_square,
            },
            "r_squared": {
                "reference": reference_fit.r_squared,
                "interleaved": interleaved_fit.r_squared,
            },
            "gate_fidelity_err_fit_covariance": fit_covariance_error,
            "fit_covariance_scale_empirical": (
                statistics.fit_covariance_scale_empirical
            ),
            "uncertainty_scope": "paired statistical bootstrap only",
            "irb_model_uncertainty_included": False,
            "readout_calibration_uncertainty_included": False,
        },
    }
    figure: go.Figure | None = None
    if options.plot or options.save_image:
        raw_interleaved_name = raw_data.metadata.get("interleaved_clifford")
        figure = _make_figure(
            target,
            raw_data.n_cliffords,
            reference_statistics,
            interleaved_statistics,
            reference_fit,
            interleaved_fit,
            gate_fidelity=gate_fidelity,
            gate_fidelity_error=gate_fidelity_error,
            gate_fidelity_ci95=gate_fidelity_ci95,
            error_bar=options.error_bar,
            interleaved_clifford_name=(
                raw_interleaved_name if isinstance(raw_interleaved_name, str) else None
            ),
        )
        if options.save_image:
            viz.save_figure(
                figure,
                name=f"paired_interleaved_randomized_benchmarking_{target}",
            )
        if options.plot:
            figure.show()
    return Result(
        data={target: analyzed_payload},
        figure=figure,
        figures={target: figure} if figure is not None else None,
    )


def analyze_paired_irb(
    raw_result: Result | Mapping[str, object],
    *,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int | None = _DEFAULT_BOOTSTRAP_SEED,
    sem_floor: float | None = None,
    error_bar: Literal["sem", "std"] | None = "sem",
    plot: bool = False,
    save_image: bool = False,
) -> Result:
    """
    Analyze saved paired IRB trials without repeating hardware measurement.

    Parameters
    ----------
    raw_result : Result | Mapping[str, object]
        Measurement-only result returned by `measure_paired_irb`.
    n_bootstrap : int, optional
        Number of paired bootstrap replicates. Defaults to 2000. Zero disables
        bootstrap uncertainty while retaining the weighted point estimate. At
        least two samples in every AB/BA stratum, two successful replicates,
        and an 80% success rate are required to report primary uncertainty.
    bootstrap_seed : int | None, optional
        Reproducible bootstrap RNG seed. Defaults to 0.
    sem_floor : float | None, optional
        Common positive SEM floor for both weighted fits. With the default
        `None`, it is 25% of the median positive SEM across both arms. If every
        SEM is zero, a probability-scale fallback of 0.001 is used. Pass a
        positive value to override this automatic floor. If that fixed value
        replaces every empirical SEM, covariance-derived fidelity uncertainty
        is unavailable.
    error_bar : {"sem", "std"} | None, optional
        Error bars used in the optional figure. Use `None` to disable them.
        Defaults to `"sem"`.
    plot : bool, optional
        Display the analysis figure. Defaults to `False`.
    save_image : bool, optional
        Save the analysis figure through Qubex visualization. Defaults to
        `False`.

    Returns
    -------
    Result
        Full-data weighted fits, unclipped gate fidelity, paired-bootstrap
        uncertainty and intervals, raw trials, and diagnostics.

    Raises
    ------
    TypeError
        Raised when `raw_result` or an option has an incompatible type.
    ValueError
        Raised when the raw result schema or an analysis option is invalid.
    RuntimeError
        Raised when either full-data decay fit does not converge.

    Notes
    -----
    Raw input must use paired IRB schema version 1, including one
    `acquisition.planned_order` entry per trial cell. When
    `metadata.is_two_qubit` is present, it must agree with `metadata.dimension`.
    Setting `save_image=True` writes the generated figure through Qubex
    visualization.

    Bootstrap resampling is independent at each length and stratified by the
    actual reference-first/interleaved-first counts in `acquisition.planned_order`.
    It uses the same sampled trial indices for the reference and interleaved
    matrices. Each stratum must contain at least two samples; bootstrap fitting
    is skipped entirely when that condition is not met. Data with two or three
    total trials still produce a point estimate, but cannot produce primary
    stratified-bootstrap uncertainty. Readout calibration and IRB model/systematic
    uncertainty are treated as fixed and are not included in the reported
    statistical interval.

    The returned target payload keeps the raw trials and adds `mean`, `std`,
    `sem`, and `fit` under each arm. Primary uncertainty fields are derived from
    the paired bootstrap only when every stratum has at least two samples and
    at least 80% of replicates succeed. When fitting is attempted, bootstrap
    spread summaries remain available as diagnostics if optimizer validity
    fails. Resampling validity, execution status, optimizer reliability, and
    bootstrap parameter-bound quality are reported separately; bound-hit
    samples remain in the distribution. Fit covariance propagation is
    diagnostic only.

    `bootstrap["parameter_bound_quality_valid"]` uses a 5% decay-parameter
    bound-hit limit and a 10% any-parameter bound-hit limit among successful
    replicates. It is independent of `bootstrap["optimizer_valid"]`; exceeding
    a bound-hit limit emits a warning but does not suppress primary uncertainty.
    Full-data `diagnostics["parameter_bound_quality_valid"]` is `False` when
    either fit has any parameter at a bound, while the point fidelity remains
    available.

    With automatic SEM regularization, `diagnostics["sem_floor"]` is the
    full-data floor and `diagnostics["sem_floor_mode"]` is `"automatic"`.
    Each bootstrap replicate recomputes its floor from its resampled trials.
    When every trial SEM is zero, or a fixed floor replaces every empirical
    SEM, covariance-derived fidelity uncertainty is unavailable and
    `diagnostics["fit_covariance_scale_empirical"]` is `False`. Full-data and
    bootstrap parameter-bound quality flags are also retained in `diagnostics`
    without suppressing the point fidelity.
    """
    options = _resolve_analysis_options(
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        sem_floor=sem_floor,
        error_bar=error_bar,
        plot=plot,
        save_image=save_image,
    )
    return _analyze_paired_irb(raw_result, options=options)


# -----------------------------------------------------------------------------
# Auto-range pilot and adaptive main-grid selection
# -----------------------------------------------------------------------------


def _combine_pilot_results(
    target: str,
    stage_results: Sequence[Result],
    *,
    sequence_seed: int | None,
    acquisition_seed: int | None,
    stage_acquisition_seeds: Sequence[int | None],
) -> dict[str, object]:
    """Combine adaptive pilot stages into one schema-compatible raw payload."""
    if not stage_results:
        raise ValueError("At least one pilot stage result is required.")
    payloads = [
        _mapping(stage.data[target], name=f"pilot_result[{target!r}]")
        for stage in stage_results
    ]
    acquisitions = [
        _mapping(payload["acquisition"], name="pilot acquisition")
        for payload in payloads
    ]
    reference_trials = np.concatenate(
        [
            np.asarray(
                _mapping(payload["reference"], name="pilot reference")["trials"],
                dtype=np.float64,
            )
            for payload in payloads
        ],
        axis=0,
    )
    interleaved_trials = np.concatenate(
        [
            np.asarray(
                _mapping(payload["interleaved"], name="pilot interleaved")["trials"],
                dtype=np.float64,
            )
            for payload in payloads
        ],
        axis=0,
    )
    n_cliffords = np.concatenate(
        [
            np.asarray(acquisition["n_cliffords"], dtype=np.int64)
            for acquisition in acquisitions
        ]
    )
    seeds = np.concatenate(
        [
            np.asarray(acquisition["seeds"], dtype=np.int64)
            for acquisition in acquisitions
        ],
        axis=0,
    )
    planned_order: list[dict[str, object]] = []
    pair_chunks: list[dict[str, object]] = []
    length_offset = 0
    pair_offset = 0
    for payload, acquisition in zip(payloads, acquisitions, strict=True):
        stage_rows = np.asarray(
            _mapping(payload["reference"], name="pilot reference")["trials"]
        ).shape[0]
        stage_plan = cast(Sequence[object], acquisition["planned_order"])
        for raw_pair in stage_plan:
            pair = dict(_mapping(raw_pair, name="pilot planned-order entry"))
            pair["pair_order_index"] = len(planned_order)
            pair["length_index"] = int(cast(int, pair["length_index"])) + length_offset
            planned_order.append(pair)
        for raw_chunk in cast(Sequence[object], acquisition["pair_chunks"]):
            chunk = dict(_mapping(raw_chunk, name="pilot pair chunk"))
            chunk["pair_order_indices"] = tuple(
                int(cast(int, index)) + pair_offset
                for index in cast(Sequence[object], chunk["pair_order_indices"])
            )
            pair_chunks.append(chunk)
        length_offset += int(stage_rows)
        pair_offset += len(stage_plan)

    first_acquisition = dict(acquisitions[0])
    first_acquisition.update(
        {
            "n_cliffords": n_cliffords,
            "n_trials": reference_trials.shape[1],
            "seeds": seeds,
            "sequence_seed": sequence_seed,
            "acquisition_seed": acquisition_seed,
            "stage_acquisition_seeds": tuple(stage_acquisition_seeds),
            "planned_order": tuple(planned_order),
            "pair_chunks": tuple(pair_chunks),
            "n_sweep_calls": sum(
                int(cast(int, acquisition["n_sweep_calls"]))
                for acquisition in acquisitions
            ),
            "n_measurement_points": sum(
                int(cast(int, acquisition["n_measurement_points"]))
                for acquisition in acquisitions
            ),
            "adaptive_stage_count": len(stage_results),
        }
    )
    metadata = dict(_mapping(payloads[0]["metadata"], name="pilot metadata"))
    metadata["pilot"] = True
    return {
        "reference": {"trials": reference_trials},
        "interleaved": {"trials": interleaved_trials},
        "acquisition": first_acquisition,
        "metadata": metadata,
    }


def _assess_pilot_arm(
    n_cliffords: NDArray[np.int64],
    moments: _TrialMoments,
    *,
    dimension: int,
) -> _PilotArmAssessment:
    """Fit and validate one pilot arm, including observed endpoint decay."""
    significance = _observed_decay_significance(moments.mean, moments.sem)
    try:
        fit = _fit_pilot_decay(
            n_cliffords,
            moments.mean,
            dimension=dimension,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        fit = None

    invalid_reasons: list[str] = []
    if fit is None:
        invalid_reasons.append("fit_failed")
    else:
        if not 0.0 < fit.decay_parameter < 1.0:
            invalid_reasons.append("invalid_decay_parameter")
        if "p" in fit.parameters_at_bound:
            invalid_reasons.append("decay_parameter_at_bound")
        if fit.amplitude < _MIN_PILOT_AMPLITUDE:
            invalid_reasons.append("pathological_amplitude")
        if fit.r_squared is None or fit.r_squared < _MIN_PILOT_R_SQUARED:
            invalid_reasons.append("poor_decay_shape")
    if significance < _MIN_PILOT_DECAY_SIGNIFICANCE:
        invalid_reasons.append("insufficient_observed_decay")
    return _PilotArmAssessment(
        fit=fit,
        decay_significance=significance,
        invalid_reasons=tuple(invalid_reasons),
    )


def _pilot_arm_quality_payload(
    assessment: _PilotArmAssessment,
) -> dict[str, object]:
    """Return one concise pilot-specific fit-quality record."""
    fit = assessment.fit
    return {
        "p": fit.decay_parameter if fit is not None else None,
        "A": fit.amplitude if fit is not None else None,
        "C": fit.offset if fit is not None else None,
        "r_squared": fit.r_squared if fit is not None else None,
        "parameters_at_bound": fit.parameters_at_bound if fit is not None else (),
        "amplitude_lower_bound_hit": (
            fit.amplitude_lower_bound_hit if fit is not None else False
        ),
        "amplitude_upper_bound_hit": (
            fit.amplitude_upper_bound_hit if fit is not None else False
        ),
        "minimum_amplitude": _MIN_PILOT_AMPLITUDE,
        "minimum_r_squared": _MIN_PILOT_R_SQUARED,
        "decay_significance": assessment.decay_significance,
        "minimum_decay_significance": _MIN_PILOT_DECAY_SIGNIFICANCE,
        "valid": assessment.valid,
        "invalid_reasons": assessment.invalid_reasons,
    }


def _pilot_assessment_payload(
    assessment: _PilotDecayAssessment,
) -> dict[str, object]:
    """Return target-level pilot quality diagnostics for both arms."""
    return {
        "reference": _pilot_arm_quality_payload(assessment.reference),
        "interleaved": _pilot_arm_quality_payload(assessment.interleaved),
    }


def _assess_pilot_decay(
    pilot_payload: Mapping[str, object],
) -> _PilotDecayAssessment:
    """Assess fixed-offset paired pilot fits independently of main analysis."""
    acquisition = _mapping(pilot_payload["acquisition"], name="pilot acquisition")
    n_cliffords = np.asarray(acquisition["n_cliffords"], dtype=np.int64)
    if len(n_cliffords) < _MIN_AUTO_RANGE_FIT_POINTS:
        raise ValueError("Pilot assessment requires at least six Clifford lengths.")
    reference_trials = np.asarray(
        _mapping(pilot_payload["reference"], name="pilot reference")["trials"],
        dtype=np.float64,
    )
    interleaved_trials = np.asarray(
        _mapping(pilot_payload["interleaved"], name="pilot interleaved")["trials"],
        dtype=np.float64,
    )
    dimension = _validated_dimension(
        _mapping(pilot_payload["metadata"], name="pilot metadata")
    )
    return _PilotDecayAssessment(
        reference=_assess_pilot_arm(
            n_cliffords,
            _trial_moments(reference_trials),
            dimension=dimension,
        ),
        interleaved=_assess_pilot_arm(
            n_cliffords,
            _trial_moments(interleaved_trials),
            dimension=dimension,
        ),
    )


def _remaining_contrast_fraction(
    fits: tuple[_PilotDecayFitResult, _PilotDecayFitResult],
    *,
    n_cliffords: int,
) -> float:
    """Return the slower arm's fitted contrast fraction at one length."""
    return max(
        fits[0].decay_parameter ** n_cliffords,
        fits[1].decay_parameter ** n_cliffords,
    )


def _contrast_target_grid(
    decay_parameter: float,
    remaining_fractions: Collection[float],
    *,
    maximum: int,
) -> tuple[NDArray[np.int64], bool]:
    """Map target contrast fractions to bounded integer Clifford lengths."""
    if not np.isfinite(decay_parameter) or not 0.0 < decay_parameter < 1.0:
        raise ValueError("The pilot decay parameter must be finite and in (0, 1).")
    log_decay = math.log(decay_parameter)
    if not np.isfinite(log_decay) or abs(log_decay) < _MIN_ABSOLUTE_LOG_DECAY:
        raise ValueError("The pilot decay parameter is too close to 1.")

    candidates: set[int] = set()
    maximum_anchor_added = False
    for remaining_fraction in remaining_fractions:
        estimated_length = math.log(remaining_fraction) / log_decay
        if not np.isfinite(estimated_length):
            raise ValueError("A contrast target produced a non-finite length.")
        if estimated_length > maximum:
            maximum_anchor_added = True
            continue
        rounded_length = round(estimated_length)
        if 0 < rounded_length <= maximum:
            candidates.add(rounded_length)
    if maximum_anchor_added:
        candidates.add(maximum)
    return np.asarray(sorted(candidates), dtype=np.int64), maximum_anchor_added


def _parallel_shared_main_grid(
    selections: Mapping[str, _MainGridSelection],
) -> NDArray[np.int64]:
    """Build the unthinned union needed by every parallel target."""
    return np.asarray(
        sorted(
            {
                int(length)
                for selection in selections.values()
                for length in selection.selected_grid
            }
        ),
        dtype=np.int64,
    )


def _fill_short_main_grid(
    grid: NDArray[np.int64],
    *,
    maximum: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Add well-separated integer lengths when contrast targets round together.

    This is only used after a valid pilot fit. For very fast decay, several
    contrast targets can round to the same early Clifford length.
    """
    if len(grid) >= _MIN_MAIN_GRID_POINTS or maximum < 5:
        return grid, np.asarray([], dtype=np.int64)

    selected = {int(value) for value in grid}
    upper = min(maximum, max(_MIN_MAIN_GRID_POINTS - 1, int(grid[-1])))
    added: list[int] = []
    if upper not in selected:
        selected.add(upper)
        added.append(upper)
    while len(selected) < _MIN_MAIN_GRID_POINTS:
        ordered = sorted(selected)
        candidates = {
            min(
                right - 1,
                max(
                    left + 1,
                    round(math.expm1((math.log1p(left) + math.log1p(right)) / 2.0)),
                ),
            )
            for left, right in pairwise(ordered)
            if right - left > 1
        }
        if not candidates:
            break
        next_value = max(
            candidates,
            key=lambda value: (
                min(
                    abs(math.log1p(value) - math.log1p(current)) for current in selected
                ),
                -value,
            ),
        )
        selected.add(next_value)
        added.append(next_value)
    return (
        np.asarray(sorted(selected), dtype=np.int64),
        np.asarray(sorted(added), dtype=np.int64),
    )


def _combine_main_grid_candidates(
    interleaved_grid: NDArray[np.int64],
    reference_grid: NDArray[np.int64],
    *,
    maximum: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Add only reference candidates beyond the interleaved grid's range."""
    covered_maximum = max(
        1,
        int(interleaved_grid[-1]) if interleaved_grid.size else 1,
    )
    reference_tail = reference_grid[reference_grid > covered_maximum].copy()
    selected = np.unique(
        np.concatenate(
            (
                np.asarray([0, 1], dtype=np.int64),
                interleaved_grid,
                reference_tail,
            )
        )
    )
    return selected[(selected >= 0) & (selected <= maximum)], reference_tail


def _fallback_main_grid(
    fallback_grid: NDArray[np.int64],
    *,
    reason: str,
) -> _MainGridSelection:
    """Return an exploration-grid fallback with an auditable reason."""
    return _MainGridSelection(
        reference_grid=np.asarray([], dtype=np.int64),
        reference_tail_grid=np.asarray([], dtype=np.int64),
        interleaved_grid=np.asarray([], dtype=np.int64),
        selected_grid=fallback_grid.copy(),
        supplemental_integer_grid=np.asarray([], dtype=np.int64),
        maximum_anchor_added=False,
        fallback_used=True,
        fallback_reason=reason,
    )


def _select_decay_adapted_main_grid(
    fits: tuple[_PilotDecayFitResult, _PilotDecayFitResult] | None,
    *,
    remaining_fractions: tuple[float, ...],
    maximum: int,
    fallback_grid: NDArray[np.int64],
) -> _MainGridSelection:
    """Select an interleaved-first grid extended only by the reference tail."""
    if fits is None:
        return _fallback_main_grid(fallback_grid, reason="pilot_fit_unusable")
    try:
        reference_grid, reference_capped = _contrast_target_grid(
            fits[0].decay_parameter,
            remaining_fractions,
            maximum=maximum,
        )
        interleaved_grid, interleaved_capped = _contrast_target_grid(
            fits[1].decay_parameter,
            remaining_fractions,
            maximum=maximum,
        )
    except ValueError as error:
        return _fallback_main_grid(
            fallback_grid,
            reason=f"invalid_pilot_decay: {error}",
        )

    selected_grid, reference_tail_grid = _combine_main_grid_candidates(
        interleaved_grid,
        reference_grid,
        maximum=maximum,
    )

    selected_grid, supplemental_integer_grid = _fill_short_main_grid(
        selected_grid,
        maximum=maximum,
    )
    return _MainGridSelection(
        reference_grid=reference_grid,
        reference_tail_grid=reference_tail_grid,
        interleaved_grid=interleaved_grid,
        selected_grid=selected_grid,
        supplemental_integer_grid=supplemental_integer_grid,
        maximum_anchor_added=reference_capped or interleaved_capped,
        fallback_used=False,
        fallback_reason=None,
    )


def _run_single_target_auto_range(
    exp: Experiment,
    target: str,
    *,
    context: _SingleAcquisitionContext,
    candidate_grid: NDArray[np.int64],
    maximum: int,
    pilot_n_trials: int,
    remaining_fraction_threshold: float,
    main_remaining_fractions: tuple[float, ...],
    pilot_sequence_seed: int | None,
    pilot_acquisition_seed: int | None,
) -> _AutoRangeOutcome:
    """Select a fresh-main grid from incrementally extended paired pilot data."""
    pilot_seeds = _generated_seed_matrix(
        (len(candidate_grid), pilot_n_trials),
        sequence_seed=pilot_sequence_seed,
    )
    stage_seed_values = _split_optional_seed(
        pilot_acquisition_seed,
        count=len(candidate_grid),
        name="pilot_acquisition_seed",
    )
    stage_results: list[Result] = []
    used_stage_seeds: list[int | None] = []
    stage_diagnostics: list[dict[str, object]] = []
    measured_count = 0
    pilot_payload: dict[str, object] | None = None
    pilot_assessment: _PilotDecayAssessment | None = None
    pilot_fits: tuple[_PilotDecayFitResult, _PilotDecayFitResult] | None = None
    stop_reason = "max_n_cliffords_reached"
    stop_detail = "fit_unusable"
    while measured_count < len(candidate_grid):
        next_count = (
            min(_MIN_AUTO_RANGE_FIT_POINTS, len(candidate_grid))
            if measured_count == 0
            else measured_count + 1
        )
        stage_slice = slice(measured_count, next_count)
        stage_seed = stage_seed_values[len(stage_results)]
        stage_results.append(
            _run_single_target_acquisition(
                exp,
                target,
                context=context,
                n_cliffords=candidate_grid[stage_slice],
                seeds=pilot_seeds[stage_slice],
                sequence_seed=pilot_sequence_seed,
                acquisition_seed=stage_seed,
            )
        )
        used_stage_seeds.append(stage_seed)
        measured_count = next_count
        pilot_payload = _combine_pilot_results(
            target,
            stage_results,
            sequence_seed=pilot_sequence_seed,
            acquisition_seed=pilot_acquisition_seed,
            stage_acquisition_seeds=used_stage_seeds,
        )
        current_maximum = int(candidate_grid[measured_count - 1])
        remaining_fraction: float | None = None
        fit_quality: dict[str, object] | None = None
        if measured_count >= _MIN_AUTO_RANGE_FIT_POINTS:
            pilot_assessment = _assess_pilot_decay(pilot_payload)
            fit_quality = _pilot_assessment_payload(pilot_assessment)
            pilot_fits = pilot_assessment.valid_fits
        if pilot_fits is None:
            stop_detail = (
                pilot_assessment.blocking_reason
                if pilot_assessment is not None
                else "fit_unusable"
            ) or "fit_unusable"
        else:
            remaining_fraction = _remaining_contrast_fraction(
                pilot_fits,
                n_cliffords=current_maximum,
            )
            if remaining_fraction <= remaining_fraction_threshold:
                stop_reason = "decay_threshold_reached"
                stop_detail = "criteria_satisfied"
            else:
                stop_detail = "remaining_fraction_above_threshold"
        stage_diagnostics.append(
            {
                "max_n_cliffords": current_maximum,
                "remaining_fraction": remaining_fraction,
                "decision": stop_detail,
                "fit_quality": fit_quality,
            }
        )
        if stop_reason == "decay_threshold_reached":
            break

    if pilot_payload is None:
        raise RuntimeError("Paired-IRB pilot acquisition produced no data.")
    if stop_reason == "max_n_cliffords_reached":
        warnings.warn(
            f"Paired-IRB auto-range reached the maximum Clifford length "
            f"{int(candidate_grid[-1])} before satisfying the decay criterion "
            f"({stop_detail}).",
            RuntimeWarning,
            stacklevel=3,
        )
    pilot_grid = candidate_grid[:measured_count].copy()
    main_grid_selection = _select_decay_adapted_main_grid(
        pilot_fits,
        remaining_fractions=main_remaining_fractions,
        maximum=maximum,
        fallback_grid=candidate_grid,
    )
    if main_grid_selection.fallback_used:
        warnings.warn(
            "Paired-IRB main-grid selection is falling back to the pilot "
            f"candidate grid: {main_grid_selection.fallback_reason}.",
            RuntimeWarning,
            stacklevel=3,
        )
    if main_grid_selection.maximum_anchor_added:
        warnings.warn(
            "Paired-IRB main-grid tail target extends beyond `max_n_cliffords`; "
            f"using the maximum Clifford length {maximum} as the tail anchor. "
            "Increase `max_n_cliffords` if a deeper decay tail is required.",
            RuntimeWarning,
            stacklevel=3,
        )
    reference_fit, interleaved_fit = (
        (
            pilot_assessment.reference.fit,
            pilot_assessment.interleaved.fit,
        )
        if pilot_assessment is not None
        else (None, None)
    )
    return _AutoRangeOutcome(
        selected_main_grid=main_grid_selection.selected_grid,
        metadata={
            "mode": "decay_adaptive",
            "enabled": True,
            "pilot_grid": pilot_grid.copy(),
            "candidate_pilot_grid": candidate_grid.copy(),
            "pilot_n_trials": pilot_n_trials,
            "pilot_remaining_fraction": remaining_fraction_threshold,
            "pilot_model": _pilot_model_payload(
                dimension=context.target_spec.dimension,
                unmitigated_two_qubit_readout=(
                    context.target_spec.is_two_qubit and not context.mitigate_readout
                ),
            ),
            "pilot_p_reference": (
                reference_fit.decay_parameter if reference_fit is not None else None
            ),
            "pilot_p_interleaved": (
                interleaved_fit.decay_parameter if interleaved_fit is not None else None
            ),
            "main_remaining_fractions": main_remaining_fractions,
            "supplemental_integer_grid": (
                main_grid_selection.supplemental_integer_grid
            ),
            "candidate_reference_grid": main_grid_selection.reference_grid,
            "reference_tail_grid": main_grid_selection.reference_tail_grid,
            "candidate_interleaved_grid": main_grid_selection.interleaved_grid,
            "selected_main_grid": main_grid_selection.selected_grid,
            "max_n_cliffords": maximum,
            "fallback_used": main_grid_selection.fallback_used,
            "fallback_reason": main_grid_selection.fallback_reason,
            "maximum_anchor_added": main_grid_selection.maximum_anchor_added,
            "pilot_stop_reason": stop_reason,
            "pilot_stop_detail": stop_detail,
            "pilot_fit_quality": (
                _pilot_assessment_payload(pilot_assessment)
                if pilot_assessment is not None
                else None
            ),
            "pilot_stage_diagnostics": tuple(stage_diagnostics),
            "pilot_sequence_seed": pilot_sequence_seed,
            "main_sequence_seed": None,
            "pilot_acquisition_seed": pilot_acquisition_seed,
            "main_acquisition_seed": None,
            "pilot_reference_fit": (
                _pilot_fit_payload(reference_fit) if reference_fit is not None else None
            ),
            "pilot_interleaved_fit": (
                _pilot_fit_payload(interleaved_fit)
                if interleaved_fit is not None
                else None
            ),
            "pilot_raw_result": pilot_payload,
        },
    )


def _warn_parallel_main_grid_selection(
    target: str,
    selection: _MainGridSelection,
    *,
    maximum: int,
) -> None:
    """Warn about fallback or clipped tail selection for one parallel target."""
    if selection.fallback_used:
        warnings.warn(
            "Parallel paired-IRB main-grid selection is falling back to the "
            f"pilot candidate grid for `{target}`: {selection.fallback_reason}.",
            RuntimeWarning,
            stacklevel=4,
        )
    if selection.maximum_anchor_added:
        warnings.warn(
            "Parallel paired-IRB main-grid tail target for "
            f"`{target}` extends beyond `max_n_cliffords`; using the maximum "
            f"Clifford length {maximum} as the tail anchor. Increase "
            "`max_n_cliffords` if a deeper decay tail is required.",
            RuntimeWarning,
            stacklevel=4,
        )


def _run_parallel_auto_range(
    exp: Experiment,
    targets: tuple[str, ...],
    *,
    contexts: Mapping[str, _SingleAcquisitionContext],
    candidate_grid: NDArray[np.int64],
    maximum: int,
    pilot_n_trials: int,
    remaining_fraction_threshold: float,
    main_remaining_fractions: tuple[float, ...],
    pilot_sequence_seeds: Mapping[str, int | None],
    pilot_acquisition_seed: int | None,
) -> _ParallelAutoRangeOutcome:
    """Select one shared range from incrementally extended parallel pilots."""
    pilot_seeds_by_target = {
        target: _generated_seed_matrix(
            (len(candidate_grid), pilot_n_trials),
            sequence_seed=pilot_sequence_seeds[target],
        )
        for target in targets
    }
    stage_seed_values = _split_optional_seed(
        pilot_acquisition_seed,
        count=len(candidate_grid),
        name="pilot_acquisition_seed",
    )
    stage_results: list[Result] = []
    used_stage_seeds: list[int | None] = []
    measured_count = 0
    pilot_payloads: dict[str, dict[str, object]] = {}
    pilot_assessments: dict[str, _PilotDecayAssessment] = {}
    pilot_fits: dict[
        str,
        tuple[_PilotDecayFitResult, _PilotDecayFitResult] | None,
    ] = {}
    stage_diagnostics_by_target: dict[str, list[dict[str, object]]] = {
        target: [] for target in targets
    }
    stop_details = dict.fromkeys(targets, "fit_unusable")
    stop_reason = "max_n_cliffords_reached"
    while measured_count < len(candidate_grid):
        next_count = (
            min(_MIN_AUTO_RANGE_FIT_POINTS, len(candidate_grid))
            if measured_count == 0
            else measured_count + 1
        )
        stage_slice = slice(measured_count, next_count)
        stage_seed = stage_seed_values[len(stage_results)]
        stage_results.append(
            _run_parallel_acquisition(
                exp,
                targets,
                contexts=contexts,
                n_cliffords=candidate_grid[stage_slice],
                seeds_by_target={
                    target: values[stage_slice]
                    for target, values in pilot_seeds_by_target.items()
                },
                sequence_seeds=pilot_sequence_seeds,
                acquisition_seed=stage_seed,
            )
        )
        used_stage_seeds.append(stage_seed)
        measured_count = next_count
        pilot_payloads = {
            target: _combine_pilot_results(
                target,
                stage_results,
                sequence_seed=pilot_sequence_seeds[target],
                acquisition_seed=pilot_acquisition_seed,
                stage_acquisition_seeds=used_stage_seeds,
            )
            for target in targets
        }
        if measured_count >= _MIN_AUTO_RANGE_FIT_POINTS:
            pilot_assessments = {
                target: _assess_pilot_decay(payload)
                for target, payload in pilot_payloads.items()
            }
            pilot_fits = {
                target: assessment.valid_fits
                for target, assessment in pilot_assessments.items()
            }
        current_maximum = int(candidate_grid[measured_count - 1])
        all_targets_satisfied = True
        for target in targets:
            assessment = pilot_assessments.get(target)
            fits = pilot_fits.get(target)
            remaining_fraction: float | None = None
            if fits is None:
                detail = (
                    assessment.blocking_reason
                    if assessment is not None
                    else "fit_unusable"
                ) or "fit_unusable"
            else:
                remaining_fraction = _remaining_contrast_fraction(
                    fits,
                    n_cliffords=current_maximum,
                )
                detail = (
                    "criteria_satisfied"
                    if remaining_fraction <= remaining_fraction_threshold
                    else "remaining_fraction_above_threshold"
                )
            stop_details[target] = detail
            all_targets_satisfied &= detail == "criteria_satisfied"
            stage_diagnostics_by_target[target].append(
                {
                    "max_n_cliffords": current_maximum,
                    "remaining_fraction": remaining_fraction,
                    "decision": detail,
                    "fit_quality": (
                        _pilot_assessment_payload(assessment)
                        if assessment is not None
                        else None
                    ),
                }
            )
        if all_targets_satisfied:
            stop_reason = "decay_threshold_reached"
            break

    if stop_reason == "max_n_cliffords_reached":
        warnings.warn(
            f"Parallel paired-IRB auto-range reached the maximum Clifford length "
            f"{int(candidate_grid[-1])} before every target satisfied the decay "
            f"criterion ({stop_details}).",
            RuntimeWarning,
            stacklevel=3,
        )
    pilot_grid = candidate_grid[:measured_count].copy()
    selections = {
        target: _select_decay_adapted_main_grid(
            pilot_fits.get(target),
            remaining_fractions=main_remaining_fractions,
            maximum=maximum,
            fallback_grid=candidate_grid,
        )
        for target in targets
    }
    for target, selection in selections.items():
        _warn_parallel_main_grid_selection(
            target,
            selection,
            maximum=maximum,
        )
    shared_main_grid = _parallel_shared_main_grid(selections)
    metadata_by_target: dict[str, dict[str, object]] = {}
    for target in targets:
        assessment = pilot_assessments.get(target)
        reference_fit, interleaved_fit = (
            (assessment.reference.fit, assessment.interleaved.fit)
            if assessment is not None
            else (None, None)
        )
        selection = selections[target]
        metadata_by_target[target] = {
            "mode": "decay_adaptive",
            "enabled": True,
            "pilot_grid": pilot_grid.copy(),
            "candidate_pilot_grid": candidate_grid.copy(),
            "pilot_n_trials": pilot_n_trials,
            "pilot_remaining_fraction": remaining_fraction_threshold,
            "pilot_model": _pilot_model_payload(
                dimension=contexts[target].target_spec.dimension,
                unmitigated_two_qubit_readout=(
                    contexts[target].target_spec.is_two_qubit
                    and not contexts[target].mitigate_readout
                ),
            ),
            "pilot_p_reference": (
                reference_fit.decay_parameter if reference_fit is not None else None
            ),
            "pilot_p_interleaved": (
                interleaved_fit.decay_parameter if interleaved_fit is not None else None
            ),
            "main_remaining_fractions": main_remaining_fractions,
            "supplemental_integer_grid": selection.supplemental_integer_grid.copy(),
            "candidate_reference_grid": selection.reference_grid.copy(),
            "reference_tail_grid": selection.reference_tail_grid.copy(),
            "candidate_interleaved_grid": selection.interleaved_grid.copy(),
            "target_candidate_main_grid": selection.selected_grid.copy(),
            "selected_main_grid": shared_main_grid.copy(),
            "max_n_cliffords": maximum,
            "fallback_used": selection.fallback_used,
            "fallback_reason": selection.fallback_reason,
            "maximum_anchor_added": selection.maximum_anchor_added,
            "pilot_stop_reason": stop_reason,
            "pilot_stop_detail": stop_details[target],
            "pilot_fit_quality": (
                _pilot_assessment_payload(assessment)
                if assessment is not None
                else None
            ),
            "pilot_stage_diagnostics": tuple(stage_diagnostics_by_target[target]),
            "parallel_shared_grid": True,
            "pilot_sequence_seed": pilot_sequence_seeds[target],
            "main_sequence_seed": None,
            "pilot_acquisition_seed": pilot_acquisition_seed,
            "main_acquisition_seed": None,
            "pilot_reference_fit": (
                _pilot_fit_payload(reference_fit) if reference_fit is not None else None
            ),
            "pilot_interleaved_fit": (
                _pilot_fit_payload(interleaved_fit)
                if interleaved_fit is not None
                else None
            ),
            "pilot_raw_result": pilot_payloads[target],
        }
    return _ParallelAutoRangeOutcome(
        selected_main_grid=shared_main_grid,
        metadata_by_target=metadata_by_target,
    )


# -----------------------------------------------------------------------------
# High-level workflow orchestration and reporting
# -----------------------------------------------------------------------------


def _fixed_range_metadata(
    n_cliffords: NDArray[np.int64],
    *,
    stop_reason: Literal["explicit_range", "auto_range_disabled"],
) -> dict[str, object]:
    """Return high-level range metadata when no pilot was required."""
    return {
        "mode": ("explicit" if stop_reason == "explicit_range" else "fixed_default"),
        "enabled": False,
        "pilot_grid": np.asarray([], dtype=np.int64),
        "candidate_pilot_grid": np.asarray([], dtype=np.int64),
        "pilot_n_trials": None,
        "pilot_remaining_fraction": None,
        "pilot_model": None,
        "pilot_p_reference": None,
        "pilot_p_interleaved": None,
        "main_remaining_fractions": None,
        "supplemental_integer_grid": np.asarray([], dtype=np.int64),
        "candidate_reference_grid": np.asarray([], dtype=np.int64),
        "reference_tail_grid": np.asarray([], dtype=np.int64),
        "candidate_interleaved_grid": np.asarray([], dtype=np.int64),
        "selected_main_grid": n_cliffords.copy(),
        "max_n_cliffords": int(n_cliffords[-1]),
        "fallback_used": False,
        "fallback_reason": None,
        "maximum_anchor_added": False,
        "pilot_stop_reason": stop_reason,
        "pilot_stop_detail": None,
        "pilot_fit_quality": None,
        "pilot_stage_diagnostics": (),
        "pilot_reference_fit": None,
        "pilot_interleaved_fit": None,
        "pilot_raw_result": None,
    }


def _validate_serial_explicit_seed_matrices(
    targets: tuple[str, ...],
    *,
    contexts: Mapping[str, _SingleAcquisitionContext],
    seeds_by_target: Mapping[str, ArrayLike | None],
    n_cliffords_range: ArrayLike | None,
    n_trials: int,
    max_n_cliffords: int | None,
) -> None:
    """Validate every serial seed matrix before the first hardware call."""
    for target in targets:
        explicit_seed_matrix = seeds_by_target[target]
        if explicit_seed_matrix is None:
            continue
        target_spec = contexts[target].target_spec
        n_cliffords = _resolve_fixed_n_cliffords(
            target_spec,
            n_cliffords_range,
            max_n_cliffords,
        )
        _resolve_seed_matrix(
            explicit_seed_matrix,
            shape=(len(n_cliffords), n_trials),
            sequence_seed=None,
        )


def _log_paired_irb_summary(target: str, payload: Mapping[str, object]) -> None:
    """Log only the primary paired-IRB point estimate and bootstrap interval."""
    reference_fit = _mapping(
        _mapping(payload["reference"], name="reference")["fit"],
        name="reference.fit",
    )
    interleaved_fit = _mapping(
        _mapping(payload["interleaved"], name="interleaved")["fit"],
        name="interleaved.fit",
    )
    gate_fidelity = float(cast(float, payload["gate_fidelity"]))
    gate_fidelity_error = cast(float | None, payload["gate_fidelity_err"])
    gate_fidelity_ci95 = cast(
        tuple[float, float] | None,
        payload["gate_fidelity_ci95"],
    )
    logger.info("Paired IRB: %s", target)
    logger.info("p_ref = %.6f", float(cast(float, reference_fit["p"])))
    logger.info("p_irb = %.6f", float(cast(float, interleaved_fit["p"])))
    if gate_fidelity_error is None:
        logger.info("Gate fidelity = %.4f %%", 100.0 * gate_fidelity)
    else:
        logger.info(
            "Gate fidelity = %.4f ± %.4f %%",
            100.0 * gate_fidelity,
            100.0 * gate_fidelity_error,
        )
    if gate_fidelity_ci95 is None:
        logger.info("95%% bootstrap CI = unavailable")
    else:
        logger.info(
            "95%% bootstrap CI = [%.4f, %.4f] %%",
            100.0 * gate_fidelity_ci95[0],
            100.0 * gate_fidelity_ci95[1],
        )


def _analyze_workflow_target(
    raw_result: Result,
    target: str,
    *,
    grid_selection: Mapping[str, object],
    options: _AnalysisOptions,
) -> tuple[dict[str, object], go.Figure | None]:
    """Analyze one target and attach workflow-level grid metadata."""
    analyzed = _analyze_paired_irb(
        Result(data={target: raw_result.data[target]}),
        options=options,
    )
    payload = cast(dict[str, object], analyzed.data[target])
    payload["grid_selection"] = grid_selection
    _log_paired_irb_summary(target, payload)
    return payload, analyzed.figure


def _run_parallel_workflow(
    exp: Experiment,
    targets: tuple[str, ...],
    *,
    interleaved_clifford: str | Clifford,
    interleaved_waveform: _InterleavedOverride,
    n_cliffords_range: ArrayLike | None,
    use_auto_range: bool,
    pilot_n_trials: int,
    remaining_fraction_threshold: float,
    main_remaining_fractions: tuple[float, ...],
    n_trials: int,
    seeds: ArrayLike | Mapping[str, ArrayLike] | None,
    sequence_seed: int | None,
    acquisition_seed: int | None,
    pairs_per_sweep: int | None,
    max_n_cliffords: int | None,
    x90: _X90Override,
    zx90: _ZX90Override,
    mitigate_readout: bool,
    n_shots: int | None,
    shot_interval: float | None,
    time_integration: bool,
    sweep_timeout: float,
    analysis_options: _AnalysisOptions,
) -> Result:
    """Run fixed-range or adaptive paired IRB with shared parallel points."""
    target_specs = {target: _resolve_target(exp, target) for target in targets}
    _validate_parallel_target_specs(target_specs)
    _validate_multi_target_schedule_overrides(
        targets,
        target_specs,
        interleaved_waveform=interleaved_waveform,
        zx90=zx90,
    )
    contexts = _resolve_target_acquisitions(
        exp,
        targets,
        target_specs,
        interleaved_clifford=interleaved_clifford,
        interleaved_waveform=interleaved_waveform,
        pairs_per_sweep=pairs_per_sweep,
        x90=x90,
        zx90=zx90,
        mitigate_readout=mitigate_readout,
        n_shots=DEFAULT_SHOTS if n_shots is None else n_shots,
        shot_interval=DEFAULT_INTERVAL if shot_interval is None else shot_interval,
        time_integration=time_integration,
        sweep_timeout=sweep_timeout,
    )
    resolved_maximum = _resolve_target_maximum(
        contexts[targets[0]].target_spec,
        max_n_cliffords,
    )

    if use_auto_range:
        sequence_streams = _split_optional_seed(
            sequence_seed,
            count=2 * len(targets),
            name="sequence_seed",
        )
        pilot_sequence_seeds = {
            target: sequence_streams[2 * index] for index, target in enumerate(targets)
        }
        main_sequence_seeds = {
            target: sequence_streams[2 * index + 1]
            for index, target in enumerate(targets)
        }
        acquisition_streams = _split_optional_seed(
            acquisition_seed,
            count=2,
            name="acquisition_seed",
        )
        auto_outcome = _run_parallel_auto_range(
            exp,
            targets,
            contexts=contexts,
            candidate_grid=_auto_range_candidate_grid(resolved_maximum),
            maximum=resolved_maximum,
            pilot_n_trials=pilot_n_trials,
            remaining_fraction_threshold=remaining_fraction_threshold,
            main_remaining_fractions=main_remaining_fractions,
            pilot_sequence_seeds=pilot_sequence_seeds,
            pilot_acquisition_seed=acquisition_streams[0],
        )
        selected_n_cliffords = auto_outcome.selected_main_grid
        range_metadata = auto_outcome.metadata_by_target
        main_acquisition_seed = acquisition_streams[1]
        for target in targets:
            range_metadata[target]["main_sequence_seed"] = main_sequence_seeds[target]
            range_metadata[target]["main_acquisition_seed"] = main_acquisition_seed
    else:
        selected_n_cliffords = _resolve_fixed_n_cliffords(
            contexts[targets[0]].target_spec,
            n_cliffords_range,
            max_n_cliffords,
        )
        main_sequence_values = _split_optional_seed(
            sequence_seed,
            count=len(targets),
            name="sequence_seed",
        )
        main_sequence_seeds = dict(zip(targets, main_sequence_values, strict=True))
        main_acquisition_seed = _validate_optional_rng_seed(
            acquisition_seed,
            name="acquisition_seed",
        )
        stop_reason = (
            "explicit_range" if n_cliffords_range is not None else "auto_range_disabled"
        )
        range_metadata = {
            target: _fixed_range_metadata(
                selected_n_cliffords,
                stop_reason=stop_reason,
            )
            for target in targets
        }

    explicit_seeds = _explicit_seeds_by_target(seeds, targets=targets)
    main_seeds_by_target = {
        target: _resolve_seed_matrix(
            explicit_seeds[target],
            shape=(len(selected_n_cliffords), n_trials),
            sequence_seed=main_sequence_seeds[target],
        )
        for target in targets
    }
    raw_result = _run_parallel_acquisition(
        exp,
        targets,
        contexts=contexts,
        n_cliffords=selected_n_cliffords,
        seeds_by_target=main_seeds_by_target,
        sequence_seeds=main_sequence_seeds,
        acquisition_seed=main_acquisition_seed,
    )
    data: dict[str, object] = {}
    figures: dict[str, go.Figure] = {}
    for target in targets:
        target_payload, figure = _analyze_workflow_target(
            raw_result,
            target,
            grid_selection=range_metadata[target],
            options=analysis_options,
        )
        data[target] = target_payload
        if figure is not None:
            figures[target] = figure
    primary_figure = figures.get(targets[0]) if len(targets) == 1 else None
    return Result(data=data, figure=primary_figure, figures=figures or None)


def _run_serial_workflow(
    exp: Experiment,
    targets: tuple[str, ...],
    *,
    interleaved_clifford: str | Clifford,
    interleaved_waveform: _InterleavedOverride,
    n_cliffords_range: ArrayLike | None,
    use_auto_range: bool,
    pilot_n_trials: int,
    remaining_fraction_threshold: float,
    main_remaining_fractions: tuple[float, ...],
    n_trials: int,
    seeds: ArrayLike | Mapping[str, ArrayLike] | None,
    sequence_seed: int | None,
    acquisition_seed: int | None,
    pairs_per_sweep: int | None,
    max_n_cliffords: int | None,
    x90: _X90Override,
    zx90: _ZX90Override,
    mitigate_readout: bool,
    n_shots: int | None,
    shot_interval: float | None,
    time_integration: bool,
    sweep_timeout: float,
    analysis_options: _AnalysisOptions,
) -> Result:
    """Run fixed-range or adaptive paired IRB independently by target."""
    target_specs = {target: _resolve_target(exp, target) for target in targets}
    _validate_multi_target_schedule_overrides(
        targets,
        target_specs,
        interleaved_waveform=interleaved_waveform,
        zx90=zx90,
    )
    contexts = _resolve_target_acquisitions(
        exp,
        targets,
        target_specs,
        interleaved_clifford=interleaved_clifford,
        interleaved_waveform=interleaved_waveform,
        pairs_per_sweep=pairs_per_sweep,
        x90=x90,
        zx90=zx90,
        mitigate_readout=mitigate_readout,
        n_shots=DEFAULT_SHOTS if n_shots is None else n_shots,
        shot_interval=DEFAULT_INTERVAL if shot_interval is None else shot_interval,
        time_integration=time_integration,
        sweep_timeout=sweep_timeout,
    )
    seeds_by_target = _explicit_seeds_by_target(seeds, targets=targets)
    streams_per_target = 2 if use_auto_range else 1
    sequence_streams = _split_optional_seed(
        sequence_seed,
        count=len(targets) * streams_per_target,
        name="sequence_seed",
    )
    acquisition_streams = _split_optional_seed(
        acquisition_seed,
        count=len(targets) * streams_per_target,
        name="acquisition_seed",
    )
    if seeds is not None:
        _validate_serial_explicit_seed_matrices(
            targets,
            contexts=contexts,
            seeds_by_target=seeds_by_target,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            max_n_cliffords=max_n_cliffords,
        )

    data: dict[str, object] = {}
    figures: dict[str, go.Figure] = {}
    for target_index, target in enumerate(targets):
        context = contexts[target]
        stream_start = target_index * streams_per_target
        if use_auto_range:
            pilot_sequence_seed = sequence_streams[stream_start]
            main_sequence_seed = sequence_streams[stream_start + 1]
            pilot_acquisition_seed = acquisition_streams[stream_start]
            main_acquisition_seed = acquisition_streams[stream_start + 1]
            resolved_maximum = _resolve_target_maximum(
                context.target_spec,
                max_n_cliffords,
            )
            auto_range_outcome = _run_single_target_auto_range(
                exp,
                target,
                context=context,
                candidate_grid=_auto_range_candidate_grid(resolved_maximum),
                maximum=resolved_maximum,
                pilot_n_trials=pilot_n_trials,
                remaining_fraction_threshold=remaining_fraction_threshold,
                main_remaining_fractions=main_remaining_fractions,
                pilot_sequence_seed=pilot_sequence_seed,
                pilot_acquisition_seed=pilot_acquisition_seed,
            )
            selected_n_cliffords = auto_range_outcome.selected_main_grid
            range_metadata = auto_range_outcome.metadata
            range_metadata["main_sequence_seed"] = main_sequence_seed
            range_metadata["main_acquisition_seed"] = main_acquisition_seed
        else:
            main_sequence_seed = sequence_streams[stream_start]
            main_acquisition_seed = acquisition_streams[stream_start]
            selected_n_cliffords = _resolve_fixed_n_cliffords(
                context.target_spec,
                n_cliffords_range,
                max_n_cliffords,
            )
            range_metadata = _fixed_range_metadata(
                selected_n_cliffords,
                stop_reason=(
                    "explicit_range"
                    if n_cliffords_range is not None
                    else "auto_range_disabled"
                ),
            )

        main_seeds = _resolve_seed_matrix(
            seeds_by_target[target],
            shape=(len(selected_n_cliffords), n_trials),
            sequence_seed=main_sequence_seed,
        )
        raw_result = _run_single_target_acquisition(
            exp,
            target,
            context=context,
            n_cliffords=selected_n_cliffords,
            seeds=main_seeds,
            sequence_seed=main_sequence_seed,
            acquisition_seed=main_acquisition_seed,
        )
        target_payload, figure = _analyze_workflow_target(
            raw_result,
            target,
            grid_selection=range_metadata,
            options=analysis_options,
        )
        data[target] = target_payload
        if figure is not None:
            figures[target] = figure

    primary_figure = figures.get(targets[0]) if len(targets) == 1 else None
    return Result(data=data, figure=primary_figure, figures=figures or None)


def paired_interleaved_randomized_benchmarking(
    exp: Experiment,
    targets: Sequence[str] | str,
    *,
    interleaved_clifford: str | Clifford,
    interleaved_waveform: (
        Waveform | PulseSchedule | TargetMap[Waveform | PulseSchedule] | None
    ) = None,
    n_cliffords_range: ArrayLike | None = None,
    auto_range: bool = False,
    pilot_n_trials: int = 6,
    auto_range_remaining_fraction: float = _DEFAULT_AUTO_RANGE_REMAINING_FRACTION,
    main_remaining_fractions: Collection[float] = _DEFAULT_MAIN_REMAINING_FRACTIONS,
    n_trials: int | None = None,
    seeds: ArrayLike | Mapping[str, ArrayLike] | None = None,
    sequence_seed: int | None = None,
    acquisition_seed: int | None = _DEFAULT_ACQUISITION_SEED,
    bootstrap_seed: int | None = _DEFAULT_BOOTSTRAP_SEED,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    pairs_per_sweep: int | None = _DEFAULT_PAIRS_PER_SWEEP,
    max_n_cliffords: int | None = None,
    x90: Waveform | TargetMap[Waveform] | None = None,
    zx90: PulseSchedule | TargetMap[PulseSchedule] | None = None,
    mitigate_readout: bool = True,
    in_parallel: bool = False,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    time_integration: bool = True,
    sweep_timeout: float = _DEFAULT_SWEEP_TIMEOUT_SECONDS,
    sem_floor: float | None = None,
    error_bar: Literal["sem", "std"] | None = "sem",
    plot: bool | None = None,
    save_image: bool | None = None,
) -> Result:
    """
    Measure and analyze paired IRB for one or more targets.

    This high-level workflow adds serial or parallel multi-target orchestration
    and optional pilot-based range selection around the single-target paired
    acquisition and analysis primitives.

    Parameters
    ----------
    exp : Experiment
        Configured Qubex experiment.
    targets : Sequence[str] | str
        One target label or a nonempty ordered sequence of distinct 1Q or CR
        target labels. Sequence order determines RNG-stream assignment, serial
        execution, and result order. Sets and frozensets are rejected.
    interleaved_clifford : str | Clifford
        Clifford inserted in the interleaved arm.
    interleaved_waveform : Waveform | PulseSchedule | TargetMap | None, optional
        Optional physical implementation of the interleaved gate. A 1Q target
        accepts a `Waveform` or a target-containing `PulseSchedule`; a 2Q
        target requires a `PulseSchedule`. A target map is resolved strictly by
        target, and every requested target must be present. A single
        `PulseSchedule` is rejected for multi-target calls because schedules are
        target-specific. Required when the RB sequence builder cannot infer the
        implementation from `interleaved_clifford`.
    n_cliffords_range : ArrayLike | None, optional
        Explicit, strictly increasing Clifford-length grid shared by both arms.
        At least four lengths are required. When supplied, it takes precedence
        over `auto_range` and is mutually exclusive with `max_n_cliffords`. With
        the default `auto_range=False` and no explicit maximum, use
        `[0, 1, 2, 4, 8, 16, 32, 64, 128]` for 2Q targets or
        `[0, 16, 32, 64, 128, 256, 512, 1024, 2048]` for 1Q targets.
    auto_range : bool, optional
        When `True` and no explicit range is supplied, use a paired pilot to
        select a fresh main-measurement grid. The pilot uses an unweighted
        two-parameter decay fit with `C=1/d`, an R-squared threshold of 0.5,
        and the observed-decay safeguard. When `False`, use the fixed grid from
        `n_cliffords_range`, the target-specific default fixed grid when neither
        range nor maximum is supplied, or a power-of-two grid through an
        explicitly supplied `max_n_cliffords`. Defaults to `False`.
    pilot_n_trials : int, optional
        Paired trials per pilot length. When auto-range is active, this must be
        at least 4 and pilot stopping is not evaluated until at least six
        Clifford lengths are available. Ignored when no pilot is run. Defaults
        to 6.
    auto_range_remaining_fraction : float, optional
        Auto-range stopping threshold strictly between 0 and 1. The pilot stops
        when `max(p_ref**m, p_irb**m)` is no greater than this value at its
        current maximum length, both fits are valid, and both arms have endpoint
        decay significance of at least 3. Ignored when no pilot is run. Defaults
        to 0.10 so the pilot observes the decay well into its tail before main-grid
        selection.
    main_remaining_fractions : Collection[float], optional
        Ordered, strictly decreasing contrast fractions used to place the
        interleaved main-grid lengths while auto-range is active. Reference
        candidates are added only beyond the interleaved grid's maximum; all
        selected lengths measure both arms. Sets and frozensets are rejected.
        Ignored when no pilot is run. Defaults to
        `(0.90, 0.70, 0.50, 0.30, 0.15, 0.05, 0.02)` so the main fit reaches
        about 2% fitted interleaved contrast without subsequently thinning the
        candidate union.
    n_trials : int | None, optional
        Paired trials per main-measurement length. Must be at least 2 and
        defaults to 30. Two or three trials permit the point estimate but not
        primary AB/BA-stratified bootstrap uncertainty.
    seeds : ArrayLike | Mapping[str, ArrayLike] | None, optional
        Explicit uint32 main seed matrix with shape `(n_lengths, n_trials)`. A
        multi-target call requires one matrix per target. Explicit matrices are
        not allowed with active auto-range because the selected length count is
        initially unknown.
    sequence_seed : int | None, optional
        Master seed for deterministic target-specific seed matrices. Auto-range
        splits it into independent pilot and main streams. Mutually exclusive
        with `seeds`. `None` generates fresh matrices.
    acquisition_seed : int | None, optional
        Master seed for global pair order and balanced AB/BA order. Auto-range
        splits it into independent pilot and main streams. `None` draws fresh
        randomized orders.
    bootstrap_seed : int | None, optional
        Paired-bootstrap RNG seed. Defaults to 0.
    n_bootstrap : int, optional
        Bootstrap replicate count. Defaults to 2000. At least two successful
        replicates, an 80% success rate, and two samples in every AB/BA stratum
        are required to report primary uncertainty.
    pairs_per_sweep : int | None, optional
        Complete adjacent pairs submitted per measurement call. `None` submits
        all pairs in one measurement call and is the default. Pass a positive
        integer to split the acquisition into smaller sweep calls.
    max_n_cliffords : int | None, optional
        Exact final pilot candidate for auto-range, or an upper bound for the
        fixed power-of-two grid when auto-range is disabled. Mutually exclusive
        with an explicit range. Active auto-range requires this ceiling to
        produce at least six pilot lengths (the smallest accepted value is 9).
        Defaults to the target-specific experiment constant.
    x90 : Waveform | TargetMap[Waveform] | None, optional
        Optional pi/2 pulse override. A 1Q target map is resolved by target. For
        a CR target, the physical-qubit map is passed intact to the RB sequence
        builder.
    zx90 : PulseSchedule | TargetMap[PulseSchedule] | None, optional
        Optional ZX90 schedule override. A target map is resolved strictly by CR
        target, and every requested CR target must be present. A single schedule
        is accepted for one CR target but rejected when more than one CR target
        is requested. The value is ignored for 1Q sequence construction.
    mitigate_readout : bool, optional
        Apply fixed readout mitigation for 2Q survival. Ignored for 1Q. Defaults
        to `True`. With `False`, asymmetric assignment errors can shift the
        observed pilot asymptote away from its fixed approximation `C=0.25`.
    in_parallel : bool, optional
        Execute every target in each shared measurement point. Parallel groups
        must contain only 1Q targets or only disjoint 2Q targets. Their AB/BA
        order is shared and shorter schedules are left-padded. The shared grid
        is the unthinned union of every target's proposed grid. Defaults to
        `False` for serial target execution.
    n_shots : int | None, optional
        Shots per sweep point. Defaults to the experiment constant.
    shot_interval : float | None, optional
        Shot interval in ns. Defaults to the experiment constant.
    time_integration : bool, optional
        Integrate readout data over time. Defaults to `True`.
    sweep_timeout : float, optional
        Timeout in seconds for each sweep chunk. Defaults to 1800 seconds.
    sem_floor : float | None, optional
        Common positive SEM floor for both weighted main fits. It does not
        affect the unweighted pilot fit. With the default `None`, use 25% of the
        paired median positive SEM, or 0.001 if every SEM is zero. Pass a
        positive value to override the automatic floor. If it replaces every
        empirical SEM, covariance-derived fidelity uncertainty is unavailable.
    error_bar : {"sem", "std"} | None, optional
        Optional figure error bars. Use `None` to disable them. Defaults to
        `"sem"`.
    plot : bool | None, optional
        Display each result figure. `None` preserves the legacy-facing default
        of `True`; explicit `False` disables display.
    save_image : bool | None, optional
        Save each result figure through Qubex visualization. `None` preserves
        the legacy-facing default of `True`; explicit `False` disables saving.

    Returns
    -------
    Result
        Target-keyed paired raw data, fits, fidelities, uncertainty, range
        metadata, and diagnostics. `figures` contains target-keyed figures when
        plotting or saving is enabled.

    Raises
    ------
    TypeError
        Raised when an argument has an incompatible type.
    ValueError
        Raised when a measurement or analysis option is invalid.
    TimeoutError
        Raised when a measurement sweep chunk times out.
    RuntimeError
        Raised when measurement data are invalid or full-data fitting fails.

    Notes
    -----
    Auto-range measures an incremental paired pilot at zero, powers of two, and
    the exact non-power-of-two exploration ceiling when needed. It maps the
    requested remaining-contrast fractions through the interleaved decay first,
    then adds only reference candidates beyond that grid's maximum. Zero and one
    are included, and the resulting union is not merged or thinned before the
    fresh main acquisition.
    Pilot trials are retained under `grid_selection["pilot_raw_result"]` but
    never enter the main fit or bootstrap. Serial targets choose grids
    independently; parallel targets use the unthinned union of all target grids.

    The pilot-only fixed-offset model assumes asymptotic survival `1/d`. For
    unmitigated 2Q readout, asymmetric assignment errors can make the observed
    asymptote differ from 0.25. This approximation selects the main grid only;
    the final main fit keeps `C` free.

    This function performs hardware acquisition and resets the AWG and capture
    units for each serial acquisition or parallel acquisition stage. A target
    failure is raised immediately; use the legacy benchmark-suite wrappers when
    best-effort continuation across targets is required. Use `measure_paired_irb`
    and `analyze_paired_irb` separately for side-effect-free reanalysis.

    The default `pairs_per_sweep=None` submits all randomized adjacent pairs in
    one measurement-service call. Pass a positive integer to split the
    acquisition into smaller calls while preserving within-pair adjacency.
    """
    resolved_targets = _normalize_targets(targets)
    _validate_range_choice(n_cliffords_range, max_n_cliffords)
    resolved_auto_range = _validate_boolean(auto_range, name="auto_range")
    resolved_in_parallel = _validate_boolean(in_parallel, name="in_parallel")
    resolved_n_trials = _resolve_paired_n_trials(n_trials)
    use_auto_range = n_cliffords_range is None and resolved_auto_range
    if use_auto_range:
        resolved_pilot_trials = _validate_positive_integer(
            pilot_n_trials,
            name="pilot_n_trials",
        )
        if resolved_pilot_trials < 4:
            raise ValueError("`pilot_n_trials` must be at least 4.")
        resolved_remaining_fraction = _validate_open_unit_interval(
            auto_range_remaining_fraction,
            name="auto_range_remaining_fraction",
        )
        resolved_main_remaining_fractions = _resolve_main_remaining_fractions(
            main_remaining_fractions
        )
    else:
        # Pilot-only options are intentionally ignored when no pilot is run.
        resolved_pilot_trials = 6
        resolved_remaining_fraction = _DEFAULT_AUTO_RANGE_REMAINING_FRACTION
        resolved_main_remaining_fractions = _DEFAULT_MAIN_REMAINING_FRACTIONS
    if use_auto_range and seeds is not None:
        raise ValueError("`auto_range=True` cannot be combined with explicit `seeds`.")
    resolved_plot = True if plot is None else _validate_boolean(plot, name="plot")
    resolved_save_image = (
        True if save_image is None else _validate_boolean(save_image, name="save_image")
    )
    analysis_options = _resolve_analysis_options(
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        sem_floor=sem_floor,
        error_bar=error_bar,
        plot=resolved_plot,
        save_image=resolved_save_image,
    )
    if seeds is not None and sequence_seed is not None:
        raise ValueError("Specify only one of `seeds` and `sequence_seed`.")
    if resolved_in_parallel:
        return _run_parallel_workflow(
            exp,
            resolved_targets,
            interleaved_clifford=interleaved_clifford,
            interleaved_waveform=interleaved_waveform,
            n_cliffords_range=n_cliffords_range,
            use_auto_range=use_auto_range,
            pilot_n_trials=resolved_pilot_trials,
            remaining_fraction_threshold=resolved_remaining_fraction,
            main_remaining_fractions=resolved_main_remaining_fractions,
            n_trials=resolved_n_trials,
            seeds=seeds,
            sequence_seed=sequence_seed,
            acquisition_seed=acquisition_seed,
            pairs_per_sweep=pairs_per_sweep,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            mitigate_readout=mitigate_readout,
            n_shots=n_shots,
            shot_interval=shot_interval,
            time_integration=time_integration,
            sweep_timeout=sweep_timeout,
            analysis_options=analysis_options,
        )
    return _run_serial_workflow(
        exp,
        resolved_targets,
        interleaved_clifford=interleaved_clifford,
        interleaved_waveform=interleaved_waveform,
        n_cliffords_range=n_cliffords_range,
        use_auto_range=use_auto_range,
        pilot_n_trials=resolved_pilot_trials,
        remaining_fraction_threshold=resolved_remaining_fraction,
        main_remaining_fractions=resolved_main_remaining_fractions,
        n_trials=resolved_n_trials,
        seeds=seeds,
        sequence_seed=sequence_seed,
        acquisition_seed=acquisition_seed,
        pairs_per_sweep=pairs_per_sweep,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        zx90=zx90,
        mitigate_readout=mitigate_readout,
        n_shots=n_shots,
        shot_interval=shot_interval,
        time_integration=time_integration,
        sweep_timeout=sweep_timeout,
        analysis_options=analysis_options,
    )


__all__ = [
    "analyze_paired_irb",
    "measure_paired_irb",
    "paired_interleaved_randomized_benchmarking",
]
