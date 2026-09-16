"""Internal data models for CR dissipation characterization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CrDissipationRateStatus(str, Enum):
    """Represent the resolution status of one reported rate."""

    RESOLVED = "resolved"
    NOMINAL_ZERO_UNRESOLVED = "nominal_zero_unresolved"
    PARTIALLY_UNRESOLVED = "partially_unresolved"
    UNRESOLVED_ASSUMED_IDLE = "unresolved_assumed_idle"
    CONSISTENT_WITH_ZERO = "consistent_with_zero"
    INCONSISTENT_RATE_DECOMPOSITION = "inconsistent_rate_decomposition"
    FIT_FAILED = "fit_failed"


@dataclass(frozen=True)
class IdleNoiseParameters:
    """Store fixed idle coherence inputs in ns."""

    t1_ns: float
    t2_echo_ns: float

    def __post_init__(self) -> None:
        """Validate positive finite lifetimes or positive infinity."""
        for name in ("t1_ns", "t2_echo_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            normalized = float(value)
            if np.isnan(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} must be positive; +inf is permitted.")
            object.__setattr__(self, name, normalized)

    @property
    def relaxation_rate_per_ns(self) -> float:
        """Return the idle e-to-g rate in 1/ns."""
        return 0.0 if np.isinf(self.t1_ns) else 1.0 / self.t1_ns

    @property
    def transverse_rate_per_ns(self) -> float:
        """Return the supplied echo transverse rate in 1/ns."""
        return 0.0 if np.isinf(self.t2_echo_ns) else 1.0 / self.t2_echo_ns

    @property
    def pure_dephasing_rate_per_ns(self) -> float:
        """Return physical idle pure dephasing in 1/ns."""
        return max(
            0.0,
            self.transverse_rate_per_ns - 0.5 * self.relaxation_rate_per_ns,
        )


@dataclass(frozen=True)
class GefPopulationSeries:
    """Store processed GEF populations and conditional uncertainty."""

    population: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    population_unconstrained: NDArray[np.float64]
    fit_diagnostics: tuple[dict[str, Any], ...] = ()
    bootstrap: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CrDissipationProtocolData:
    """Store one actual, reference, or diagnostic protocol data series."""

    control_gef: GefPopulationSeries | None = None
    target_gef: GefPopulationSeries | None = None
    target_x_comp: NDArray[np.float64] | None = None
    target_x_comp_standard_error: NDArray[np.float64] | None = None
    primary_expectation: NDArray[np.float64] | None = None
    primary_standard_error: NDArray[np.float64] | None = None
    components: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    component_standard_errors: dict[str, NDArray[np.float64]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class CrDissipationProtocolMeasurements:
    """Store resolved timing and data for one protocol."""

    elapsed_time_ns: NDArray[np.float64]
    cr_active_time_ns: NDArray[np.float64]
    actual: CrDissipationProtocolData
    reference: CrDissipationProtocolData | None
    orthogonal: CrDissipationProtocolData | None


@dataclass(frozen=True)
class CrDissipationMeasurements:
    """Store processed measurements for the four CR protocols."""

    repetition_counts: NDArray[np.int64]
    cr_gate_counts: NDArray[np.int64]
    control_ground_cr_population: CrDissipationProtocolMeasurements
    control_excited_cr_population: CrDissipationProtocolMeasurements
    control_cr_transverse_echo: CrDissipationProtocolMeasurements
    target_cr_rotating_frame_echo: CrDissipationProtocolMeasurements


@dataclass(frozen=True)
class CandidateFit:
    """Store one final ordinary-GLS candidate fit."""

    name: str
    success: bool
    parameter_names: tuple[str, ...]
    parameters: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    prediction: NDArray[np.float64]
    chi_squared: float
    reduced_chi_squared: float
    aicc: float | None
    n_observations: int
    n_parameters: int
    message: str


@dataclass(frozen=True)
class ControlPopulationRateFit:
    """Store the selected joint A/B control-population fit."""

    success: bool
    selected_model: str
    rates_per_ns: dict[str, float]
    statuses: dict[str, CrDissipationRateStatus]
    covariance: NDArray[np.float64]
    parameter_order: tuple[str, ...]
    candidates: dict[str, CandidateFit]
    fitted_populations: dict[str, NDArray[np.float64]]
    message: str


@dataclass(frozen=True)
class DecayRateFit:
    """Store one selected target T1rho fit."""

    success: bool
    selected_model: str
    rate_per_ns: float
    rate_standard_error_per_ns: float | None
    status: CrDissipationRateStatus
    x_infinity: float
    candidates: dict[str, CandidateFit]
    fitted_values: NDArray[np.float64]
    message: str


@dataclass(frozen=True)
class ExchangeRateFit:
    """Store one selected target leakage/seepage fit."""

    success: bool
    selected_model: str
    leakage_rate_per_ns: float
    seepage_rate_per_ns: float
    leakage_standard_error_per_ns: float | None
    seepage_standard_error_per_ns: float | None
    leakage_status: CrDissipationRateStatus
    seepage_status: CrDissipationRateStatus
    covariance: NDArray[np.float64]
    candidates: dict[str, CandidateFit]
    fitted_values: NDArray[np.float64]
    message: str


@dataclass(frozen=True)
class PhysicalForwardDephasingFit:
    """Store one C/D physical-forward pure-dephasing fit."""

    success: bool
    selected_model: str
    rate_per_ns: float
    rate_standard_error_per_ns: float | None
    status: CrDissipationRateStatus
    spam_amplitude: float
    spam_offset: float
    candidates: dict[str, CandidateFit]
    fitted_values: NDArray[np.float64]
    unconstrained_rate_per_ns: float | None
    unconstrained_standard_error_per_ns: float | None
    message: str


@dataclass(frozen=True)
class CrDissipationFits:
    """Store all selected fits used by the analysis."""

    control_population_ab: ControlPopulationRateFit
    target_a_t1rho: DecayRateFit
    target_a_leakage: ExchangeRateFit
    target_b_t1rho: DecayRateFit
    target_b_leakage: ExchangeRateFit
    control_pure_dephasing_c: PhysicalForwardDephasingFit
    target_rotating_frame_pure_dephasing_d: PhysicalForwardDephasingFit


@dataclass(frozen=True)
class CrDissipationRateEstimate:
    """Store a user-facing CR-active rate and equivalent lifetime."""

    rate_per_ns: float
    rate_standard_error_per_ns: float | None
    lifetime_ns: float
    lifetime_interval_1sigma_ns: tuple[float, float] | None
    idle_equivalent_rate_per_ns: float
    idle_equivalent_lifetime_ns: float
    status: CrDissipationRateStatus
    source_protocols: tuple[str, ...]
    message: str | None = None


@dataclass(frozen=True)
class CrDissipationFidelityEstimate:
    """Store one dissipative fidelity limit and uncertainty."""

    value: float
    standard_error: float | None
    available: bool
    uncertainty_type: str | None
    message: str | None = None


@dataclass(frozen=True)
class CrDissipationFidelityLimits:
    """Store the three requested fidelity limits."""

    idle_coherence_limited: CrDissipationFidelityEstimate
    cr_on_coherence_limited: CrDissipationFidelityEstimate
    cr_on_dissipative_limited: CrDissipationFidelityEstimate
    fixed_unresolved_rates: tuple[str, ...]
    fallback_rates_per_ns: dict[str, float]


@dataclass(frozen=True)
class CrDissipationIdlePrediction:
    """Store an actual-sequence idle-only predicted curve."""

    elapsed_time_ns: NDArray[np.float64]
    observables: dict[str, NDArray[np.float64]]
    equivalent_rates: dict[str, float]


@dataclass(frozen=True)
class CrDissipationWarning:
    """Store one structured diagnostic warning."""

    code: str
    message: str
    affected_outputs: tuple[str, ...]


@dataclass(frozen=True)
class CrDissipationAnalysis:
    """Store rates, fits, predictions, fidelity limits, and warnings."""

    rates: dict[str, CrDissipationRateEstimate]
    derived_rates: dict[str, CrDissipationRateEstimate]
    nominal_rates_for_forward_model: dict[str, float]
    fidelity: CrDissipationFidelityLimits
    fits: CrDissipationFits
    idle_predictions: dict[str, CrDissipationIdlePrediction]
    warnings: tuple[CrDissipationWarning, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
