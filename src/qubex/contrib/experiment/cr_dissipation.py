"""
Fast dissipation characterization of decoherence and leakage during cross-resonance pulses.

The workflow is intentionally diagnostic rather than a strict parameter-
identification pipeline.  It measures four physically named protocols:

`control_ground_cr_population` / `control_excited_cr_population`
    Repeated non-echoed blocks with the control prepared in `|0>` or `|1>`.
    Each positive CR lobe is followed by a control-pi-sized blank.
    GEF readout provides control populations, target X polarization, and target
    F-state population.

`control_cr_transverse_echo`
    Echoed ZX90 sequence whose primary observable is control X.

`target_cr_rotating_frame_echo`
    Four-ZX90 rotating-frame echo whose primary observable is target Y. The target
    virtual-Z frame update is applied to both the target and CR channels.

Standard references preserve the internal control-pi slots. CR-active windows
are either disabled or replaced by a calibrated target IX45 with the CR
envelope. Optional orthogonal Pauli measurements are acquired only for
diagnosis. Uncertainties are local analytic approximations and no raw-shot
bootstrap is run.

The primary public entry points are `characterize_cr_dissipation`,
`analyze_cr_dissipation`, and `plot_cr_dissipation`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.linalg import expm
from scipy.optimize import OptimizeResult, least_squares
from tqdm.auto import tqdm

from qubex.analysis import fitting
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.pulse import Blank, FlatTop, PulseSchedule, Waveform
from qubex.visualization import COLORS

from .gef_population_estimation import calibrate_gef_population, measure_gef_populations

if TYPE_CHECKING:
    from .gef_population_estimation import GefPopulationCalibration, GefPopulationFit

# ---------------------------------------------------------------------------
# Public constants and small data models
# ---------------------------------------------------------------------------

DEFAULT_N_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55)
DEFAULT_N_SHOTS = 4096
DEFAULT_CALIBRATION_N_SHOTS = 8192
DEFAULT_REFERENCE_CALIBRATION_N_SHOTS = 2048
_REFERENCE_CALIBRATION_NOTE_KEY = "cr_dissipation_reference_ix45"

_BASIS = Literal["X", "Y", "Z"]
_CONTROL_STATE_PROTOCOL = Literal[
    "control_ground_cr_population", "control_excited_cr_population"
]
_ECHO_PROTOCOL = Literal["control_cr_transverse_echo", "target_cr_rotating_frame_echo"]
_EXCHANGE_MODEL = Literal["stable", "outward_only", "inward_only", "two_way"]
_CHANGE_STATUS = Literal["stable", "changing", "invalid"]
_FIT_QUALITY = Literal["stable", "good", "fair", "poor", "failed"]
_POPULATION_WEIGHTING = Literal[
    "full_covariance",
    "component_se_diagonal",
    "mixed_component_se_empirical",
    "empirical_scale",
    "not_used_stable",
    "not_used_invalid_initial",
]
_ROBUST_LOSSES = {"linear", "soft_l1", "huber", "cauchy", "arctan"}

_CONTROL_GROUND = "control_ground_cr_population"
_CONTROL_EXCITED = "control_excited_cr_population"
_CONTROL_TRANSVERSE_ECHO = "control_cr_transverse_echo"
_TARGET_ROTATING_FRAME_ECHO = "target_cr_rotating_frame_echo"
_CONTROL_STATE_PROTOCOLS = (_CONTROL_GROUND, _CONTROL_EXCITED)
_ECHO_PROTOCOLS = (_CONTROL_TRANSVERSE_ECHO, _TARGET_ROTATING_FRAME_ECHO)
_PROTOCOL_LABELS = {
    _CONTROL_GROUND: "Control initialized in |g>",
    _CONTROL_EXCITED: "Control initialized in |e>",
    _CONTROL_TRANSVERSE_ECHO: "Control CR transverse echo",
    _TARGET_ROTATING_FRAME_ECHO: "Target CR rotating-frame echo",
}
_STATE_NAMES = ("g", "e", "f")

_MIN_LEAKAGE_AXIS_UPPER = 0.01
_LEAKAGE_AXIS_HEADROOM = 1.15
_EPS = float(np.finfo(float).eps)
_PROBABILITY_TOLERANCE = 1e-6
_NAN_BLOCH_FACTORS: tuple[float, float, float] = (
    float("nan"),
    float("nan"),
    float("nan"),
)


@dataclass(frozen=True)
class ZX90Timing:
    """
    Store Qubex ZX90 timing metadata in ns.

    `cr_duration` is the schedule's raw metadata value: one lobe for an echoed
    ZX90 and the total CR-active duration for an un-echoed schedule call.
    """

    cr_duration: float
    echo: bool
    total_duration: float

    def __post_init__(self) -> None:
        """Validate the ZX90 timing metadata."""
        if not isinstance(self.echo, bool):
            raise TypeError("echo must be boolean.")
        for name, value in (
            ("cr_duration", self.cr_duration),
            ("total_duration", self.total_duration),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.cr_active_duration > self.total_duration + 1e-9:
            raise ValueError("CR-active duration cannot exceed total ZX90 duration.")

    @property
    def cr_active_duration(self) -> float:
        """Return the CR-active time contained in one ZX90 schedule."""
        return (2.0 if self.echo else 1.0) * self.cr_duration


@dataclass(frozen=True)
class CrShapedIx45Calibration:
    """Calibration of one target IX45 lobe shaped like the CR envelope."""

    amplitude: float
    duration: float
    ramptime: float
    ramp_type: str
    sampling_period: float
    r_squared: float
    timestamp: str

    def __post_init__(self) -> None:
        """Validate the cached calibration payload."""
        for name, value in (
            ("amplitude", self.amplitude),
            ("duration", self.duration),
            ("ramptime", self.ramptime),
            ("sampling_period", self.sampling_period),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 < self.amplitude <= 1.0:
            raise ValueError("amplitude must be in (0, 1].")
        if self.duration <= 0.0 or self.sampling_period <= 0.0:
            raise ValueError("duration and sampling_period must be positive.")
        if self.ramptime < 0.0:
            raise ValueError("ramptime must be nonnegative.")
        if not isinstance(self.ramp_type, str):
            raise TypeError("ramp_type must be a string.")
        if isinstance(self.r_squared, bool) or not isinstance(self.r_squared, Real):
            raise TypeError("r_squared must be a real number.")
        if not np.isfinite(self.r_squared) and not np.isnan(self.r_squared):
            raise ValueError("r_squared must be finite or NaN.")
        if not isinstance(self.timestamp, str):
            raise TypeError("timestamp must be a string.")

        for name in (
            "amplitude",
            "duration",
            "ramptime",
            "sampling_period",
            "r_squared",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def fingerprint(self) -> tuple[float, float, str, float]:
        """Return the pulse-shape fields that determine cache compatibility."""
        return (self.duration, self.ramptime, self.ramp_type, self.sampling_period)


@dataclass(frozen=True)
class IdleNoiseParameters:
    """Idle T1/T2 values in ns used by decay corrections and fidelity."""

    t1: float
    t2_echo: float
    t2_star: float | None = None

    def __post_init__(self) -> None:
        """Validate the idle lifetime inputs."""
        for name, value in (("t1", self.t1), ("t2_echo", self.t2_echo)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            if np.isnan(value) or value <= 0:
                raise ValueError(f"{name} must be positive; +inf is permitted.")
            object.__setattr__(self, name, float(value))
        if self.t2_star is not None:
            if (
                isinstance(self.t2_star, bool)
                or not isinstance(self.t2_star, Real)
                or np.isnan(self.t2_star)
                or self.t2_star <= 0
            ):
                raise ValueError("t2_star must be positive; +inf is permitted.")
            object.__setattr__(self, "t2_star", float(self.t2_star))
        if not np.isinf(self.t1) and (
            np.isinf(self.t2_echo) or self.t2_echo > 2.0 * self.t1
        ):
            warnings.warn(
                "t2_echo exceeds 2*t1; decay correction and fidelity estimation "
                "will cap T2 at the physical 2*T1 limit.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def effective_t2_echo(self) -> float:
        """Return physical T2 for corrections and fidelity, capped at 2*T1."""
        return min(float(self.t2_echo), 2.0 * float(self.t1))

    @property
    def effective_t2_star(self) -> float:
        """Return T2-star for unrefocused blanks, falling back to T2-echo."""
        value = self.t2_echo if self.t2_star is None else self.t2_star
        return min(float(value), 2.0 * float(self.t1))


@dataclass(frozen=True)
class ChangeAssessment:
    """Lightweight decision on whether one measured curve changes or is unusable."""

    status: _CHANGE_STATUS
    constant_value: float
    robust_range: float
    noise_scale: float
    range_signal_to_noise: float
    reduced_chi_squared: float
    minimum_change: float
    change_sigma_threshold: float


@dataclass(frozen=True)
class ExponentialDecayFit:
    """Offset-exponential fit, ``offset + amplitude * exp(-rate * t)``."""

    assessment: ChangeAssessment
    success: bool
    quality: _FIT_QUALITY
    message: str
    rate: float
    rate_standard_error: float
    time_constant: float
    time_constant_standard_error: float
    amplitude: float
    offset: float
    fitted_values: NDArray[np.float64]
    curve_times: NDArray[np.float64]
    curve_values: NDArray[np.float64]
    r_squared: float
    reduced_chi_squared: float


@dataclass(frozen=True)
class PopulationExchangeFit:
    """
    Store a minimal effective model for an F-state population trajectory.

    Stable traces use zero rates. Changing traces compare the appropriate
    directional one-way model with a two-way exchange model and retain the
    smallest model supported by a BIC-like robust-cost score.
    """

    assessment: ChangeAssessment
    model: _EXCHANGE_MODEL
    success: bool
    quality: _FIT_QUALITY
    message: str
    outward_rate: float
    inward_rate: float
    outward_rate_standard_error: float
    inward_rate_standard_error: float
    exchange_time_constant: float
    p0: float
    p_inf: float
    fitted_values: NDArray[np.float64]
    curve_times: NDArray[np.float64]
    curve_values: NDArray[np.float64]
    r_squared: float
    reduced_chi_squared: float


@dataclass(frozen=True)
class ControlPopulationRateFit:
    """Reduced three-level control-population fit for ground/excited preparation."""

    success: bool
    quality: _FIT_QUALITY
    message: str
    active_rates: tuple[str, ...]
    rates: Mapping[str, float]
    rate_standard_errors: Mapping[str, float]
    time_constants: Mapping[str, float]
    covariance: NDArray[np.float64]
    assessments: Mapping[str, ChangeAssessment]
    fitted_populations: Mapping[str, NDArray[np.float64]]
    curve_times: NDArray[np.float64]
    curve_populations: Mapping[str, NDArray[np.float64]]
    r_squared: float
    reduced_chi_squared: float
    population_weighting: _POPULATION_WEIGHTING


@dataclass(frozen=True)
class CrDissipationFidelityEstimate:
    """Dissipation-limited fidelity and leakage-correlation bounds."""

    available: bool
    message: str
    estimated_fidelity: float
    estimated_fidelity_standard_error: float
    idle_coherence_limit: float
    average_leakage: float
    control_lambdas: tuple[float, float, float]
    target_lambdas: tuple[float, float, float]
    computational_survival: float
    parameter_values: Mapping[str, float]
    parameter_standard_errors: Mapping[str, float]
    metadata: Mapping[str, object]

    @property
    def estimated_dissipation_limited_fidelity(self) -> float:
        """Return the nominal dissipation-limited average gate fidelity."""
        return self.estimated_fidelity

    @property
    def computational_survival_nominal(self) -> float:
        """Return survival assuming independent control and target leakage."""
        return self.computational_survival

    def _metadata_float(self, key: str) -> float:
        value = self.metadata.get(key, np.nan)
        return float(value) if isinstance(value, Real) else float("nan")

    @property
    def computational_survival_lower(self) -> float:
        """Return the Fréchet lower bound without assuming leakage independence."""
        return self._metadata_float("computational_survival_lower")

    @property
    def computational_survival_upper(self) -> float:
        """Return the Fréchet upper bound without assuming leakage independence."""
        return self._metadata_float("computational_survival_upper")

    @property
    def leakage_correlation_fidelity_lower(self) -> float:
        """Return fidelity rescaled by the lower survival bound."""
        return self._metadata_float("leakage_correlation_fidelity_lower")

    @property
    def leakage_correlation_fidelity_upper(self) -> float:
        """Return fidelity rescaled by the upper survival bound."""
        return self._metadata_float("leakage_correlation_fidelity_upper")


@dataclass(frozen=True)
class CrDissipationMeasurements:
    """
    Store processed measurements sufficient for offline dissipation characterization analysis.

    `total_times` and `cr_active_times` are in ns. Populations and Pauli
    expectations are dimensionless.  Analysis therefore reports rates in
    `1/ns` and time constants in ns. Each population row must lie on the G/E/F
    probability simplex. `population_covariances` optionally stores the full
    analytic 3x3 covariance for each population row.
    """

    control_qubit: str
    target_qubit: str
    n_values: tuple[int, ...]
    total_times: Mapping[str, NDArray[np.float64]]
    cr_active_times: Mapping[str, NDArray[np.float64]]
    populations: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
    population_standard_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
    target_x_standard_errors: Mapping[str, Mapping[str, NDArray[np.float64]]]
    pauli_expectations: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
    pauli_standard_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
    diagnostic_components_measured: bool
    population_covariances: (
        Mapping[
            str,
            Mapping[str, Mapping[str, NDArray[np.float64]]],
        ]
        | None
    ) = None

    @property
    def target_x(self) -> dict[str, dict[str, NDArray[np.float64]]]:
        """Return target X from the two control-state GEF measurements."""
        result: dict[str, dict[str, NDArray[np.float64]]] = {}
        for protocol in _CONTROL_STATE_PROTOCOLS:
            result[protocol] = {}
            for kind in ("actual", "reference"):
                population = np.asarray(
                    self.populations[protocol][kind][self.target_qubit],
                    dtype=np.float64,
                )
                denominator = population[:, 0] + population[:, 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    result[protocol][kind] = np.where(
                        denominator > _EPS,
                        (population[:, 0] - population[:, 1]) / denominator,
                        np.nan,
                    )
        return result

    @property
    def cr_pulse_counts(self) -> tuple[int, ...]:
        """Return the common number of ZX90 schedule calls, `4*n`."""
        return tuple(4 * n for n in self.n_values)


@dataclass(frozen=True)
class CrDissipationAnalysis:
    """
    Store lightweight dissipation characterization fits, summaries, and the fidelity estimate.

    `rate_differences_from_reference` stores signed reference differences.
    For the control-state protocols these are direct `actual-reference`
    differences on the common CR-active time axis. For the two echo protocols
    they are `(actual_total_rate - reference_total_rate) / CR_duty_cycle`.
    They are reported for interpretation and are not substituted for the
    nominal dissipation rates.
    """

    control_rate_fits: Mapping[str, ControlPopulationRateFit]
    target_t1rho_fits: Mapping[str, Mapping[str, ExponentialDecayFit]]
    target_leakage_fits: Mapping[str, Mapping[str, PopulationExchangeFit]]
    transverse_echo_fits: Mapping[str, Mapping[str, ExponentialDecayFit]]
    cr_active_rates: Mapping[str, float]
    cr_active_rate_standard_errors: Mapping[str, float]
    decay_times: Mapping[str, float]
    rate_differences_from_reference: Mapping[str, float]
    fidelity: CrDissipationFidelityEstimate
    quality_flags: Mapping[str, str]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _PauliMeasurement:
    expectation: float
    standard_error: float


# ---------------------------------------------------------------------------
# Validation and schedule helpers
# ---------------------------------------------------------------------------


def _diagnostic_key(
    protocol: _CONTROL_STATE_PROTOCOL,
    qubit_role: Literal["control", "target"],
) -> str:
    """Return the storage key for optional orthogonal Pauli diagnostics."""
    return f"{protocol}_{qubit_role}_diagnostic"


def _validate_n_values(n_values: Sequence[int] | None) -> tuple[int, ...]:
    values = DEFAULT_N_VALUES if n_values is None else tuple(n_values)
    if len(values) < 3:
        raise ValueError("n_values must contain at least three values.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError("n_values must contain only integers.")
    normalized = tuple(int(value) for value in values)
    if normalized[0] != 0:
        raise ValueError("n_values must begin at zero.")
    if any(value < 0 for value in normalized):
        raise ValueError("n_values must be nonnegative.")
    if any(right <= left for left, right in pairwise(normalized)):
        raise ValueError("n_values must be strictly increasing and unique.")
    return normalized


def _resolve_shot_count(value: int | None, *, default: int, name: str) -> int:
    resolved = default if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, Integral):
        raise TypeError(f"{name} must be an integer of at least two.")
    if resolved < 2:
        raise ValueError(f"{name} must be at least two.")
    return int(resolved)


def _positive_real(value: float | None, *, default: float, name: str) -> float:
    resolved = default if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, Real):
        raise TypeError(f"{name} must be a positive finite real number.")
    resolved = float(resolved)
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return resolved


def _nonnegative_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a nonnegative finite real number.")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be a nonnegative finite real number.")
    return resolved


def _optional_probability(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be in [0, 1] or None.")
    resolved = float(value)
    if not np.isfinite(resolved) or not 0 <= resolved <= 1:
        raise ValueError(f"{name} must be in [0, 1] or None.")
    return resolved


def _validate_robust_loss(value: str) -> str:
    """Return a supported scipy least-squares loss name."""
    if not isinstance(value, str):
        raise TypeError("robust_loss must be a string.")
    if value not in _ROBUST_LOSSES:
        allowed = ", ".join(sorted(_ROBUST_LOSSES))
        raise ValueError(f"robust_loss must be one of: {allowed}.")
    return value


def _extract_zx90_timing(gate: PulseSchedule) -> ZX90Timing:
    try:
        raw_cr_duration = gate.cr_duration  # type: ignore[attr-defined]
        echo = gate.echo  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ValueError(
            "ZX90 schedule must expose Qubex `cr_duration` and `echo` metadata."
        ) from exc
    if isinstance(raw_cr_duration, bool) or not isinstance(raw_cr_duration, Real):
        raise TypeError("ZX90 `cr_duration` metadata must be a real number.")
    cr_duration = float(raw_cr_duration)
    if not isinstance(echo, bool):
        raise TypeError("ZX90 `echo` metadata must be boolean.")
    if not np.isfinite(cr_duration) or cr_duration <= 0:
        raise ValueError("ZX90 `cr_duration` must be positive and finite.")
    raw_total_duration = gate.duration
    if isinstance(raw_total_duration, bool) or not isinstance(raw_total_duration, Real):
        raise TypeError("ZX90 schedule duration must be a real number.")
    total_duration = float(raw_total_duration)
    if not np.isfinite(total_duration) or total_duration <= 0:
        raise ValueError("ZX90 schedule duration must be positive and finite.")
    active_duration = (2.0 if echo else 1.0) * cr_duration
    if active_duration > total_duration + 1e-9:
        raise ValueError(
            "ZX90 CR-active duration exceeds the total schedule duration "
            f"({active_duration} ns > {total_duration} ns)."
        )
    return ZX90Timing(
        cr_duration=cr_duration,
        echo=echo,
        total_duration=total_duration,
    )


def _flat_top_unit_area(pulse: FlatTop) -> float:
    """Return the discrete area of a unit-amplitude copy of a FlatTop pulse."""
    unit = FlatTop(
        duration=pulse.duration,
        amplitude=1.0,
        tau=pulse.tau,
        beta=pulse.beta,
        type=pulse.type,
        sampling_period=pulse.sampling_period,
    )
    area = float(np.sum(unit.real) * unit.sampling_period)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Pulse envelope must have a positive finite area.")
    return area


def _initial_cr_shaped_ix45_amplitude(
    calibrated_x90: FlatTop,
    cr_envelope: FlatTop,
) -> float:
    """Estimate CR-shaped IX45 amplitude by pulse-area conversion from X90."""
    return float(
        0.5
        * calibrated_x90.amplitude
        * _flat_top_unit_area(calibrated_x90)
        / _flat_top_unit_area(cr_envelope)
    )


def _make_cr_shaped_ix45(
    cr_envelope: FlatTop,
    amplitude: float,
    *,
    allow_zero: bool = False,
) -> FlatTop:
    """Create a target pulse with CR geometry; zero is allowed only for a sweep."""
    if not isinstance(allow_zero, bool):
        raise TypeError("allow_zero must be boolean.")
    if isinstance(amplitude, bool) or not isinstance(amplitude, Real):
        raise TypeError("reference_ix45_amplitude must be a real number.")
    resolved_amplitude = float(amplitude)
    lower = 0.0 if allow_zero else np.nextafter(0.0, 1.0)
    if not np.isfinite(resolved_amplitude) or not lower <= resolved_amplitude <= 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"reference_ix45_amplitude must be in {interval}.")
    return FlatTop(
        duration=cr_envelope.duration,
        amplitude=resolved_amplitude,
        tau=cr_envelope.tau,
        type=cr_envelope.type,
        sampling_period=cr_envelope.sampling_period,
    )


def _ix45_fingerprint(cr_envelope: FlatTop) -> tuple[float, float, str, float]:
    """Return the cache fingerprint for a CR-shaped IX45 pulse."""
    return (
        float(cr_envelope.duration),
        float(cr_envelope.tau),
        str(cr_envelope.type),
        float(cr_envelope.sampling_period),
    )


def _calibrate_cr_shaped_ix45(
    exp: Experiment,
    target_qubit: str,
    cr_envelope: FlatTop,
    *,
    initial_amplitude: float,
    n_shots: int,
    shot_interval: float,
    n_points: int = 21,
    n_rotations: int = 2,
    r2_threshold: float = 0.5,
    plot: bool = False,
) -> CrShapedIx45Calibration:
    """Calibrate the actual ``IX45 -> IX45`` reference unit as one X90."""
    if n_points < 5:
        raise ValueError("reference_calibration_n_points must be at least five.")
    if n_rotations < 1:
        raise ValueError("reference_calibration_n_rotations must be positive.")
    span = 0.5 / n_rotations
    lower = float(np.clip(initial_amplitude * (1.0 - span), 0.0, 1.0))
    upper = float(np.clip(initial_amplitude * (1.0 + span), 0.0, 1.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        amplitudes = np.linspace(0.0, 1.0, n_points)
    else:
        amplitudes = np.linspace(lower, upper, n_points)

    def sequence(amplitude: float) -> dict[str, Waveform]:
        ix45 = _make_cr_shaped_ix45(cr_envelope, amplitude, allow_zero=True)
        return {target_qubit: ix45.repeated(2)}

    sweep = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=amplitudes,
        repetitions=4 * n_rotations,
        shots=n_shots,
        interval=shot_interval,
        plot=False,
    ).data[target_qubit]
    fit = fitting.fit_ampl_calib_data(
        target=target_qubit,
        amplitude_range=amplitudes,
        data=np.asarray(sweep.normalized, dtype=np.float64),
        plot=plot,
        title="CR-shaped IX45 pair calibration",
        ylabel="Normalized signal",
    )
    fitted_amplitude = float(fit["amplitude"])
    r_squared = float(fit["r2"])
    if (
        not np.isfinite(fitted_amplitude)
        or not 0.0 < fitted_amplitude <= 1.0
        or not np.isfinite(r_squared)
        or r_squared < r2_threshold
    ):
        raise RuntimeError(
            "CR-shaped IX45 calibration failed quality validation: "
            f"amplitude={fitted_amplitude!r}, r_squared={r_squared!r}, "
            f"required_r_squared={r2_threshold}."
        )
    return CrShapedIx45Calibration(
        amplitude=fitted_amplitude,
        duration=float(cr_envelope.duration),
        ramptime=float(cr_envelope.tau),
        ramp_type=str(cr_envelope.type),
        sampling_period=float(cr_envelope.sampling_period),
        r_squared=r_squared,
        # CalibrationNote validates cache age with this legacy timestamp format.
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _resolve_cr_shaped_ix45_calibration(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    cr_envelope: FlatTop,
    *,
    amplitude: float | None,
    valid_days: int | None,
    force: bool,
    n_shots: int,
    shot_interval: float,
    n_points: int,
    n_rotations: int,
    r2_threshold: float,
    plot: bool,
) -> CrShapedIx45Calibration:
    """Load a compatible IX45 calibration or acquire and cache a new one."""
    fingerprint = _ix45_fingerprint(cr_envelope)
    if amplitude is not None:
        return CrShapedIx45Calibration(
            amplitude=_positive_real(
                amplitude, default=1.0, name="reference_ix45_amplitude"
            ),
            duration=fingerprint[0],
            ramptime=fingerprint[1],
            ramp_type=fingerprint[2],
            sampling_period=fingerprint[3],
            r_squared=float("nan"),
            timestamp="user-supplied",
        )

    key = f"{control_qubit}-{target_qubit}"
    note = exp.ctx.calib_note
    cached = None
    if not force:
        try:
            cached = note.get_property(
                _REFERENCE_CALIBRATION_NOTE_KEY,
                key,
                valid_days,
            )
        except AttributeError:
            # CalibrationNote.get_property currently raises when the property
            # category itself has never been created. That is a normal first-use
            # cache miss, not a failed calibration.
            cached = None
        except (KeyError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Ignoring malformed cached CR-shaped IX45 calibration: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    if isinstance(cached, Mapping):
        try:
            candidate = CrShapedIx45Calibration(**cached)
        except (TypeError, ValueError):
            candidate = None
        if (
            candidate is not None
            and candidate.fingerprint == fingerprint
            and np.isfinite(candidate.amplitude)
            and 0.0 < candidate.amplitude <= 1.0
            and np.isfinite(candidate.r_squared)
            and candidate.r_squared >= r2_threshold
        ):
            return candidate

    hpi = exp.pulse.get_hpi_pulse(target_qubit)
    if not isinstance(hpi, FlatTop):
        raise TypeError("The calibrated target hpi pulse must be a FlatTop pulse.")
    calibrated = _calibrate_cr_shaped_ix45(
        exp,
        target_qubit,
        cr_envelope,
        initial_amplitude=_initial_cr_shaped_ix45_amplitude(hpi, cr_envelope),
        n_shots=n_shots,
        shot_interval=shot_interval,
        n_points=n_points,
        n_rotations=n_rotations,
        r2_threshold=r2_threshold,
        plot=plot,
    )
    note.put_property(_REFERENCE_CALIBRATION_NOTE_KEY, key, calibrated.__dict__)
    return calibrated


def _build_un_echoed_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    blank_duration: float = 0.0,
) -> PulseSchedule:
    """Build ``C+ -> B_pi,c -> C+ -> B_pi,c`` for protocols A and B."""
    lobe = exp.pulse.zx90(control_qubit, target_qubit, echo=False)
    timing = _extract_zx90_timing(lobe)
    if timing.echo:
        raise ValueError("exp.pulse.zx90(..., echo=False) returned an echoed gate.")
    blank = _reference_unit(
        (control_qubit, f"{control_qubit}-{target_qubit}", target_qubit),
        blank_duration,
        frequencies=lobe.get_frequencies(),
    )
    with PulseSchedule() as gate:
        gate.call(lobe, copy=True)
        gate.call(blank, copy=True)
        gate.call(lobe, copy=True)
        gate.call(blank, copy=True)
    gate.set_frequencies(lobe.get_frequencies())
    gate.cr_duration = 2 * timing.cr_duration  # type: ignore[attr-defined]
    gate.echo = False  # type: ignore[attr-defined]
    return gate


def _echo_pi_slot_duration(zx90_echo: PulseSchedule) -> float:
    """Return one embedded control-pi slot, including its margins and gaps."""
    timing = _extract_zx90_timing(zx90_echo)
    if not timing.echo:
        raise ValueError("zx90_echo must be echoed.")
    duration = (timing.total_duration - timing.cr_active_duration) / 2.0
    if duration < -1e-9:
        raise ValueError("Echo-pi slot duration cannot be negative.")
    return max(0.0, duration)


def _build_cr_shaped_ix45_lobe(
    labels: Sequence[str],
    target_qubit: str,
    pulse: FlatTop,
    *,
    frequencies: Mapping[str, float | None],
) -> PulseSchedule:
    """Build one CR-duration target IX45 lobe on the three protocol channels."""
    return _reference_unit(
        labels,
        pulse.duration,
        pulse_target=target_qubit,
        pulse=pulse,
        frequencies=frequencies,
    )


def _build_non_echoed_reference_unit(
    labels: Sequence[str],
    target_qubit: str,
    ix45: FlatTop,
    blank_duration: float,
    *,
    frequencies: Mapping[str, float | None],
) -> PulseSchedule:
    """Build the A/B reference with IX45 lobes and preserved pi-sized blanks."""
    lobe = _build_cr_shaped_ix45_lobe(
        labels, target_qubit, ix45, frequencies=frequencies
    )
    blank = _reference_unit(labels, blank_duration, frequencies=frequencies)
    with PulseSchedule() as unit:
        unit.call(lobe, copy=True)
        unit.call(blank, copy=True)
        unit.call(lobe, copy=True)
        unit.call(blank, copy=True)
    unit.set_frequencies(frequencies)
    return unit


def _build_echoed_reference_unit(
    zx90_echo: PulseSchedule,
    control_qubit: str,
    target_qubit: str,
    ix45: FlatTop | None,
) -> PulseSchedule:
    """Replace CR-active lobes while preserving both internal control-pi slots."""
    labels = (control_qubit, f"{control_qubit}-{target_qubit}", target_qubit)
    frequencies = zx90_echo.get_frequencies()
    lobe_duration = _extract_zx90_timing(zx90_echo).cr_duration
    if ix45 is None:
        lobe = _reference_unit(labels, lobe_duration, frequencies=frequencies)
    else:
        lobe = _build_cr_shaped_ix45_lobe(
            labels, target_qubit, ix45, frequencies=frequencies
        )
    pi_pulse = getattr(zx90_echo, "pi_pulse", None)
    slot_duration = _echo_pi_slot_duration(zx90_echo)
    if pi_pulse is None:
        raise ValueError(
            "The echoed ZX90 must expose `pi_pulse` to preserve its reference timing."
        )
    if pi_pulse.duration > slot_duration + 1e-9:
        raise ValueError("Embedded control pi pulse is longer than its inferred slot.")
    left_margin = max(0.0, (slot_duration - pi_pulse.duration) / 2.0)
    right_margin = max(0.0, slot_duration - pi_pulse.duration - left_margin)

    def add_pi_slot(schedule: PulseSchedule) -> None:
        schedule.add(control_qubit, Blank(left_margin))
        schedule.add(control_qubit, pi_pulse)
        schedule.add(control_qubit, Blank(right_margin))

    with PulseSchedule(list(labels)) as reference:
        reference.call(lobe, copy=True)
        reference.barrier()
        add_pi_slot(reference)
        reference.barrier()
        reference.call(lobe, copy=True)
        reference.barrier()
        add_pi_slot(reference)
    if not np.isclose(reference.duration, zx90_echo.duration):
        raise ValueError(
            "Reconstructed reference does not match the echoed ZX90 duration; "
            f"inferred pi slot was {slot_duration} ns."
        )
    reference.set_frequencies(frequencies)
    return reference


def _reference_unit(
    labels: Sequence[str],
    duration: float,
    *,
    pulse_target: str | None = None,
    pulse: Waveform | None = None,
    frequencies: Mapping[str, float | None] | None = None,
) -> PulseSchedule:
    """Create a duration-matched reference unit with matching frequency metadata."""
    if (pulse_target is None) != (pulse is None):
        raise ValueError("pulse_target and pulse must be provided together.")
    with PulseSchedule(list(labels)) as schedule:
        if pulse_target is not None and pulse is not None:
            schedule.add(pulse_target, pulse)
        else:
            schedule.add(labels[0], Blank(0))
    if schedule.duration > duration and not np.isclose(schedule.duration, duration):
        raise ValueError(
            "Reference single-qubit pulse is longer than the ZX90 schedule "
            f"({schedule.duration} ns > {duration} ns)."
        )
    reference = schedule.padded(duration, pad_side="right")
    if frequencies is not None:
        reference.set_frequencies(frequencies)
    return reference


def _require_matched_duration(
    actual: PulseSchedule,
    reference: PulseSchedule,
    *,
    name: str,
) -> None:
    """Reject an actual/reference pair with unequal evolution duration."""
    if not np.isclose(actual.duration, reference.duration, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{name} actual/reference durations differ "
            f"({actual.duration} ns != {reference.duration} ns)."
        )


def _state_preparation(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    control_state: Literal["0", "1", "+"],
    target_state: Literal["0", "+"],
) -> PulseSchedule:
    with PulseSchedule([control_qubit, target_qubit]) as schedule:
        schedule.add(
            control_qubit,
            exp.pulse.get_pulse_for_state(control_qubit, control_state),
        )
        schedule.add(
            target_qubit,
            exp.pulse.get_pulse_for_state(target_qubit, target_state),
        )
        schedule.barrier()
    return schedule


def _control_state_base_sequence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    control_state: Literal["0", "1"],
    evolution: PulseSchedule,
) -> PulseSchedule:
    """Build one control-state preparation/evolution sequence before analysis."""
    preparation = _state_preparation(
        exp,
        control_qubit,
        target_qubit,
        control_state,
        "+",
    )
    with PulseSchedule() as schedule:
        schedule.call(preparation, copy=True)
        schedule.call(evolution, copy=True)
    schedule.set_frequencies(evolution.get_frequencies())
    return schedule


def _control_state_gef_sequence(
    exp: Experiment,
    base: PulseSchedule,
    target_qubit: str,
) -> PulseSchedule:
    """Append target -Y90 so GEF readout yields target X polarization."""
    schedule = base.copy()
    with schedule:
        schedule.barrier()
        schedule.add(target_qubit, exp.pulse.y90m(target_qubit))
    return schedule


def _control_cr_transverse_echo_block(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    zx90: PulseSchedule,
) -> PulseSchedule:
    """Build ``U-XI-U-(YI+IX)-U-XI-U-YI`` to monitor control X."""
    with PulseSchedule() as block:
        block.call(zx90, copy=True)
        block.barrier()
        block.add(control_qubit, exp.pulse.x180(control_qubit))
        block.barrier()

        block.call(zx90, copy=True)
        block.barrier()
        block.add(control_qubit, exp.pulse.y180(control_qubit))
        block.add(target_qubit, exp.pulse.x180(target_qubit))
        block.barrier()

        block.call(zx90, copy=True)
        block.barrier()
        block.add(control_qubit, exp.pulse.x180(control_qubit))
        block.barrier()

        block.call(zx90, copy=True)
        block.barrier()
        block.add(control_qubit, exp.pulse.y180(control_qubit))
    block.set_frequencies(zx90.get_frequencies())
    return block


def _target_cr_rotating_frame_echo_block(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    zx90: PulseSchedule,
) -> PulseSchedule:
    """
    Build the four-ZX90 target rotating-frame echo block used to monitor target Y.

    Target virtual Z must be applied to both the target channel and the CR
    channel so the relative CR/cancel/rotary phase is preserved.
    """
    cr_label = f"{control_qubit}-{target_qubit}"
    z180 = exp.pulse.z180()
    with PulseSchedule() as block:
        block.call(zx90, copy=True)
        block.barrier()
        block.add(target_qubit, exp.pulse.y180(target_qubit))
        block.barrier()
        block.call(zx90, copy=True)
        block.barrier()
        block.add(target_qubit, z180)
        block.add(cr_label, z180)
        block.barrier()
        block.call(zx90, copy=True)
        block.barrier()
        block.add(target_qubit, exp.pulse.y180(target_qubit))
        block.barrier()
        block.call(zx90, copy=True)
        block.barrier()
        block.add(target_qubit, z180)
        block.add(cr_label, z180)
    block.set_frequencies(zx90.get_frequencies())
    return block


def _echo_protocol_sequence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    protocol: _ECHO_PROTOCOL,
    evolution: PulseSchedule,
) -> PulseSchedule:
    if protocol == _CONTROL_TRANSVERSE_ECHO:
        preparation = _state_preparation(exp, control_qubit, target_qubit, "+", "+")
    else:
        preparation = _state_preparation(exp, control_qubit, target_qubit, "0", "0")
        with preparation:
            preparation.add(target_qubit, exp.pulse.x90m(target_qubit))
    with PulseSchedule() as schedule:
        schedule.call(preparation, copy=True)
        schedule.call(evolution, copy=True)
    schedule.set_frequencies(evolution.get_frequencies())
    return schedule


def _append_pauli_analyzer(
    exp: Experiment,
    sequence: PulseSchedule,
    target: str,
    basis: _BASIS,
) -> PulseSchedule:
    measurement = sequence.copy()
    with measurement:
        analyzer: Waveform | None = None
        if basis == "X":
            analyzer = exp.pulse.y90m(target)
        elif basis == "Y":
            analyzer = exp.pulse.x90(target)
        if analyzer is not None:
            measurement.barrier()
            measurement.add(target, analyzer)
    return measurement


# ---------------------------------------------------------------------------
# Lightweight statistics and fits
# ---------------------------------------------------------------------------


def _finite_errors(
    errors: ArrayLike | None,
    shape: tuple[int, ...],
) -> NDArray[np.float64] | None:
    """Return usable positive SEs, replacing isolated invalid entries pragmatically."""
    if errors is None:
        return None
    array = np.asarray(errors, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"standard-error shape {array.shape} does not match {shape}.")
    valid = np.isfinite(array) & (array > 0)
    if not np.any(valid):
        return None
    typical = float(np.median(array[valid]))
    floor = max(typical * 1e-3, 1e-12)
    replacement = max(typical, floor)
    return np.where(valid, np.maximum(array, floor), replacement)


def _validated_population_covariance(
    covariance: ArrayLike,
    *,
    rcond: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Return a finite, nearly symmetric, positive-semidefinite 3x3 covariance.

    Tiny antisymmetric components can arise from floating-point roundoff and are
    symmetrized here. Tiny negative eigenvalues within tolerance are accepted and
    clipped to zero later when constructing the whitener. Resolved asymmetry or a
    resolved negative eigenvalue is treated as malformed covariance data so callers
    can fall back to component-wise standard errors instead of silently
    manufacturing an overconfident uncertainty.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        raise ValueError("population covariance must be a finite 3x3 matrix.")

    entry_scale = max(float(np.max(np.abs(cov))), _EPS)
    symmetry_atol = max(1e-15, 1e-10 * entry_scale)
    if not np.allclose(cov, cov.T, rtol=1e-7, atol=symmetry_atol):
        raise ValueError(
            "population covariance must be symmetric within numerical tolerance."
        )

    cov = 0.5 * (cov + cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    scale = max(float(np.max(np.abs(eigenvalues))), _EPS)
    cutoff = max(float(rcond), 0.0) * scale
    negative_tolerance = max(cutoff, 1e-15 * scale, _EPS)
    if float(np.min(eigenvalues)) < -negative_tolerance:
        raise ValueError("population covariance must be positive semidefinite.")
    return np.asarray(cov, dtype=np.float64)


def _population_covariance_whitener(
    covariance: ArrayLike,
    *,
    rcond: float = 1e-12,
) -> tuple[NDArray[np.float64], int]:
    """
    Return a pseudo-inverse square-root whitener and its effective rank.

    G/E/F population estimates obey Pg + Pe + Pf = 1, so their covariance is
    generally singular.  Whitening therefore uses only resolved covariance
    eigenmodes instead of treating the three population components as
    independent measurements.
    """
    cov = _validated_population_covariance(covariance, rcond=rcond)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    scale = max(float(np.max(np.abs(eigenvalues))), _EPS)
    cutoff = max(float(rcond), 0.0) * scale

    eigenvalues = np.maximum(eigenvalues, 0.0)
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise ValueError("population covariance has no resolved positive mode.")
    whitener = (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])).T
    return np.asarray(whitener, dtype=np.float64), int(np.count_nonzero(keep))


def _weighted_constant(
    values: NDArray[np.float64], errors: NDArray[np.float64] | None
) -> float:
    if errors is None:
        return float(np.mean(values))
    weights = 1.0 / np.square(errors)
    return float(np.sum(weights * values) / np.sum(weights))


def _reduced_chi_squared(
    observed: NDArray[np.float64],
    fitted: NDArray[np.float64],
    errors: NDArray[np.float64] | None,
    parameter_count: int,
) -> float:
    residual = np.asarray(observed - fitted, dtype=np.float64)
    if errors is None:
        # Without externally estimated point uncertainties this is not a true
        # chi-squared statistic.  Leave it undefined and let R^2 drive the
        # lightweight fit-quality label.
        return float("nan")
    dof = residual.size - parameter_count
    if dof <= 0:
        # A saturated/over-parameterized fit has no residual degrees of freedom,
        # so a reduced chi-squared statistic is not defined.
        return float("nan")
    return float(np.sum(np.square(residual / errors)) / dof)


def _r_squared(observed: NDArray[np.float64], fitted: NDArray[np.float64]) -> float:
    residual = float(np.sum(np.square(observed - fitted)))
    total = float(np.sum(np.square(observed - np.mean(observed))))
    if total <= _EPS:
        return 1.0 if residual <= _EPS else float("nan")
    return 1.0 - residual / total


def _assess_change(
    times: ArrayLike,
    values: ArrayLike,
    standard_errors: ArrayLike | None,
    *,
    minimum_change: float,
    change_sigma_threshold: float,
) -> ChangeAssessment:
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or y.shape != t.shape or t.size < 3:
        raise ValueError(
            "change assessment requires matching 1D arrays with >=3 points."
        )
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("change assessment inputs must be finite.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("change-assessment times must be strictly increasing.")
    errors = _finite_errors(standard_errors, y.shape)
    constant = _weighted_constant(y, errors)
    constant_curve = np.full_like(y, constant)
    reduced_chi2 = _reduced_chi_squared(y, constant_curve, errors, 1)

    # A percentile range is less sensitive to one bad point than max-min while
    # remaining intuitive for the default ten-point sweep.
    q10, q90 = np.percentile(y, [10.0, 90.0])
    robust_range = float(abs(q90 - q10))
    if errors is None:
        if y.size >= 3:
            step_noise = np.diff(y)
            noise_scale = float(
                np.median(np.abs(step_noise - np.median(step_noise))) / 0.6745
            )
        else:  # pragma: no cover
            noise_scale = 0.0
        noise_scale = max(noise_scale, 1e-6)
    else:
        noise_scale = max(float(np.median(errors)), 1e-12)
    snr = robust_range / noise_scale
    changing = robust_range >= minimum_change and snr >= change_sigma_threshold

    return ChangeAssessment(
        status="changing" if changing else "stable",
        constant_value=constant,
        robust_range=robust_range,
        noise_scale=noise_scale,
        range_signal_to_noise=snr,
        reduced_chi_squared=reduced_chi2,
        minimum_change=minimum_change,
        change_sigma_threshold=change_sigma_threshold,
    )


def _rate_upper_bound(times: NDArray[np.float64]) -> float:
    """Return a generous rate ceiling set by the finest sampled time step."""
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    if positive_steps.size == 0:
        return 1.0
    return float(20.0 / max(float(np.min(positive_steps)), 1e-12))


def _run_robust_least_squares(
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    x0: NDArray[np.float64],
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    robust_loss: str,
    max_nfev: int,
) -> tuple[OptimizeResult | None, str | None]:
    """Run one bounded robust fit without letting numerical failures abort a dissipation characterization."""
    try:
        optimization = least_squares(
            residual,
            x0=x0,
            bounds=bounds,
            loss=robust_loss,
            f_scale=1.0,
            max_nfev=max_nfev,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return None, f"Numerical fit failed: {exc}"
    else:
        return optimization, None


def _approximate_covariance(
    optimization: OptimizeResult,
    residual_count: int,
    parameter_count: int,
    *,
    absolute_errors: bool,
) -> NDArray[np.float64]:
    """
    Estimate covariance when fitted parameters are locally identifiable.

    A pseudo-inverse by itself can return finite diagonal entries even when the
    Jacobian is rank-deficient.  That would misleadingly attach finite standard
    errors to parameters that are not independently resolved.  Treat such local
    uncertainty as unresolved instead, while leaving the point estimate intact.
    """
    if parameter_count == 0:
        return np.empty((0, 0), dtype=np.float64)
    jac = np.asarray(optimization.jac, dtype=np.float64)
    if jac.ndim != 2 or jac.shape[1] != parameter_count or not np.all(np.isfinite(jac)):
        return np.full((parameter_count, parameter_count), np.nan)
    try:
        _, singular_values, vt = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full((parameter_count, parameter_count), np.nan)
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        return np.full((parameter_count, parameter_count), np.nan)
    largest = float(np.max(singular_values))
    if largest <= _EPS:
        return np.full((parameter_count, parameter_count), np.nan)
    rank_tolerance = max(
        1e-12 * largest,
        np.finfo(np.float64).eps * max(jac.shape) * largest,
    )
    effective_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if effective_rank < parameter_count:
        return np.full((parameter_count, parameter_count), np.nan)

    # Build (J^T J)^-1 from the *same* singular values used for the rank test.
    # Forming J^T J first squares the condition number; applying another rcond
    # cutoff there can therefore discard a direction that the rank test above
    # deliberately retained (for example s_min / s_max = 1e-8).  Direct SVD
    # construction keeps the identifiability criterion and covariance estimate
    # numerically consistent.
    try:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            inverse = (vt.T / np.square(singular_values)) @ vt
    except (FloatingPointError, ValueError):
        return np.full((parameter_count, parameter_count), np.nan)
    if inverse.shape != (parameter_count, parameter_count) or not np.all(
        np.isfinite(inverse)
    ):
        return np.full((parameter_count, parameter_count), np.nan)
    inverse = 0.5 * (inverse + inverse.T)
    if absolute_errors:
        return np.asarray(inverse, dtype=np.float64)
    dof = residual_count - parameter_count
    if dof <= 0:
        return np.full((parameter_count, parameter_count), np.nan)
    # Scipy least_squares cost is 1/2 sum(rho(residual**2)); this remains a
    # local diagnostic approximation when a robust loss modifies the residuals.
    scale = 2.0 * float(optimization.cost) / dof
    return np.asarray(inverse * scale, dtype=np.float64)


def _fit_quality(success: bool, r_squared: float, reduced_chi2: float) -> _FIT_QUALITY:
    if not success:
        return "failed"
    if np.isfinite(reduced_chi2) and reduced_chi2 > 20.0:
        return "poor"
    if (
        np.isfinite(r_squared)
        and r_squared >= 0.95
        and (np.isnan(reduced_chi2) or reduced_chi2 <= 5.0)
    ):
        return "good"
    if (
        np.isfinite(r_squared)
        and r_squared >= 0.75
        and (np.isnan(reduced_chi2) or reduced_chi2 <= 20.0)
    ):
        return "fair"
    return "poor"


def _rate_to_lifetime(rate: float, rate_error: float) -> tuple[float, float]:
    if not np.isfinite(rate):
        return float("nan"), float("nan")
    if rate <= 0:
        return float("inf"), float("nan")
    lifetime = 1.0 / rate
    error = (
        abs(rate_error) / rate**2
        if np.isfinite(rate_error) and rate_error >= 0
        else float("nan")
    )
    return float(lifetime), float(error)


def _fit_exponential_decay(
    times: ArrayLike,
    values: ArrayLike,
    standard_errors: ArrayLike | None,
    *,
    minimum_change: float,
    change_sigma_threshold: float,
    robust_loss: str,
) -> ExponentialDecayFit:
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    errors = _finite_errors(standard_errors, y.shape)

    # A derived observable can legitimately become undefined even when the raw
    # measurements are finite.  The main example is target X = (Pg-Pe)/(Pg+Pe)
    # when essentially all target population has leaked to |f>, so Pg+Pe -> 0.
    # Treat that particular fit as unresolved instead of aborting the entire
    # dissipation analysis and losing otherwise useful relaxation/leakage diagnostics.
    if not np.all(np.isfinite(y)):
        if t.ndim != 1 or y.shape != t.shape or t.size < 3:
            raise ValueError(
                "decay fit requires matching 1D time/value arrays with >=3 points."
            )
        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("decay-fit times must be finite and strictly increasing.")
        finite = np.isfinite(y)
        finite_errors = None if errors is None else errors[finite]
        if np.count_nonzero(finite) >= 3:
            finite_assessment = _assess_change(
                t[finite],
                y[finite],
                finite_errors,
                minimum_change=minimum_change,
                change_sigma_threshold=change_sigma_threshold,
            )
            assessment_values = (
                finite_assessment.constant_value,
                finite_assessment.robust_range,
                finite_assessment.noise_scale,
                finite_assessment.range_signal_to_noise,
                finite_assessment.reduced_chi_squared,
            )
        else:
            assessment_values = (float("nan"),) * 5
        assessment = ChangeAssessment(
            status="invalid",
            constant_value=assessment_values[0],
            robust_range=assessment_values[1],
            noise_scale=assessment_values[2],
            range_signal_to_noise=assessment_values[3],
            reduced_chi_squared=assessment_values[4],
            minimum_change=minimum_change,
            change_sigma_threshold=change_sigma_threshold,
        )
        dense_t = np.linspace(float(t[0]), float(t[-1]), 400)
        return ExponentialDecayFit(
            assessment=assessment,
            success=False,
            quality="failed",
            message=(
                "Decay observable contains non-finite values; this fit is left "
                "unresolved so the remaining dissipation analysis can continue."
            ),
            rate=float("nan"),
            rate_standard_error=float("nan"),
            time_constant=float("nan"),
            time_constant_standard_error=float("nan"),
            amplitude=float("nan"),
            offset=float("nan"),
            fitted_values=np.full_like(y, np.nan),
            curve_times=dense_t,
            curve_values=np.full_like(dense_t, np.nan),
            r_squared=float("nan"),
            reduced_chi_squared=float("nan"),
        )

    assessment = _assess_change(
        t,
        y,
        errors,
        minimum_change=minimum_change,
        change_sigma_threshold=change_sigma_threshold,
    )
    elapsed = t - t[0]
    dense_t = np.linspace(float(t[0]), float(t[-1]), 400)
    dense_elapsed = dense_t - t[0]
    if assessment.status == "stable":
        fitted = np.full_like(y, assessment.constant_value)
        curve = np.full_like(dense_t, assessment.constant_value)
        return ExponentialDecayFit(
            assessment=assessment,
            success=True,
            quality="stable",
            message="No resolvable change; nominal rate fixed to zero.",
            rate=0.0,
            rate_standard_error=float("nan"),
            time_constant=float("inf"),
            time_constant_standard_error=float("nan"),
            amplitude=0.0,
            offset=assessment.constant_value,
            fitted_values=fitted,
            curve_times=dense_t,
            curve_values=curve,
            r_squared=_r_squared(y, fitted),
            reduced_chi_squared=assessment.reduced_chi_squared,
        )

    span = max(float(t[-1] - t[0]), 1.0)
    tail_count = max(2, y.size // 3)
    offset0 = float(np.mean(y[-tail_count:]))
    amplitude0 = float(y[0] - offset0)
    rate0 = 1.0 / max(0.4 * span, 1.0)
    scale = errors if errors is not None else np.full_like(y, max(np.std(y), 1e-3))

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        offset, amplitude, rate = parameters
        model = offset + amplitude * np.exp(-rate * elapsed)
        return (model - y) / scale

    optimization, numerical_error = _run_robust_least_squares(
        residual,
        x0=np.array([offset0, amplitude0, rate0], dtype=np.float64),
        bounds=(
            np.array([-2.0, -4.0, 0.0], dtype=np.float64),
            np.array([2.0, 4.0, _rate_upper_bound(t)], dtype=np.float64),
        ),
        robust_loss=robust_loss,
        max_nfev=1500,
    )
    if optimization is None:
        return ExponentialDecayFit(
            assessment=assessment,
            success=False,
            quality="failed",
            message=cast(str, numerical_error),
            rate=float("nan"),
            rate_standard_error=float("nan"),
            time_constant=float("nan"),
            time_constant_standard_error=float("nan"),
            amplitude=float("nan"),
            offset=float("nan"),
            fitted_values=np.full_like(y, np.nan),
            curve_times=dense_t,
            curve_values=np.full_like(dense_t, np.nan),
            r_squared=float("nan"),
            reduced_chi_squared=float("nan"),
        )
    params = np.asarray(optimization.x, dtype=np.float64)
    finite_solution = params.shape == (3,) and np.all(np.isfinite(params))
    if not finite_solution:
        fitted = np.full_like(y, np.nan)
        return ExponentialDecayFit(
            assessment=assessment,
            success=False,
            quality="failed",
            message="Numerical fit did not return finite parameters.",
            rate=float("nan"),
            rate_standard_error=float("nan"),
            time_constant=float("nan"),
            time_constant_standard_error=float("nan"),
            amplitude=float("nan"),
            offset=float("nan"),
            fitted_values=fitted,
            curve_times=dense_t,
            curve_values=np.full_like(dense_t, np.nan),
            r_squared=float("nan"),
            reduced_chi_squared=float("nan"),
        )

    offset, amplitude, rate = map(float, params)
    fitted = offset + amplitude * np.exp(-rate * elapsed)
    curve = offset + amplitude * np.exp(-rate * dense_elapsed)
    covariance = _approximate_covariance(
        optimization,
        y.size,
        3,
        absolute_errors=errors is not None,
    )
    rate_error = (
        float(np.sqrt(max(covariance[2, 2], 0.0)))
        if covariance.shape == (3, 3) and np.isfinite(covariance[2, 2])
        else float("nan")
    )
    lifetime, lifetime_error = _rate_to_lifetime(rate, rate_error)
    r2 = _r_squared(y, fitted)
    chi2 = _reduced_chi_squared(y, fitted, errors, 3)
    quality = _fit_quality(True, r2, chi2)
    message = str(optimization.message)
    if y.size - 3 <= 0:
        quality = "poor"
        message = (
            "Fit has no residual degrees of freedom; fit quality cannot be "
            "assessed. " + message
        )
    if not optimization.success:
        message = "Finite diagnostic fit returned despite optimizer warning: " + message
        quality = "poor"
    return ExponentialDecayFit(
        assessment=assessment,
        success=True,
        quality=quality,
        message=message,
        rate=rate,
        rate_standard_error=rate_error,
        time_constant=lifetime,
        time_constant_standard_error=lifetime_error,
        amplitude=amplitude,
        offset=offset,
        fitted_values=np.asarray(fitted, dtype=np.float64),
        curve_times=dense_t,
        curve_values=np.asarray(curve, dtype=np.float64),
        r_squared=r2,
        reduced_chi_squared=chi2,
    )


def _fit_population_exchange(
    times: ArrayLike,
    f_population: ArrayLike,
    standard_errors: ArrayLike | None,
    *,
    minimum_change: float,
    change_sigma_threshold: float,
    robust_loss: str,
) -> PopulationExchangeFit:
    """
    Fit a minimal effective F-population exchange model.

    A stable trace fixes both rates to zero. For a changing trace, the fit
    compares the direction-compatible one-way model with a two-way model and
    uses a BIC-like robust-cost score to prefer the smallest adequate model.
    The goal is a robust diagnostic metric rather than unique microscopic rates.
    """
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(f_population, dtype=np.float64)
    errors = _finite_errors(standard_errors, y.shape)
    assessment = _assess_change(
        t,
        y,
        errors,
        minimum_change=minimum_change,
        change_sigma_threshold=change_sigma_threshold,
    )
    elapsed = t - t[0]
    dense_t = np.linspace(float(t[0]), float(t[-1]), 400)
    dense_elapsed = dense_t - t[0]
    if assessment.status == "stable":
        fitted = np.full_like(y, assessment.constant_value)
        return PopulationExchangeFit(
            assessment=assessment,
            model="stable",
            success=True,
            quality="stable",
            message="No resolvable F-state change; nominal leakage/seepage fixed to zero.",
            outward_rate=0.0,
            inward_rate=0.0,
            outward_rate_standard_error=float("nan"),
            inward_rate_standard_error=float("nan"),
            exchange_time_constant=float("inf"),
            p0=assessment.constant_value,
            p_inf=assessment.constant_value,
            fitted_values=fitted,
            curve_times=dense_t,
            curve_values=np.full_like(dense_t, assessment.constant_value),
            r_squared=_r_squared(y, fitted),
            reduced_chi_squared=assessment.reduced_chi_squared,
        )

    edge_count = max(1, y.size // 3)
    net_change = float(np.mean(y[-edge_count:]) - np.mean(y[:edge_count]))
    direction_threshold = 0.5 * minimum_change
    if net_change > direction_threshold:
        preferred_model: _EXCHANGE_MODEL = "outward_only"
    elif net_change < -direction_threshold:
        preferred_model = "inward_only"
    else:
        preferred_model = "two_way"

    scale = (
        errors if errors is not None else np.full_like(y, max(float(np.std(y)), 1e-3))
    )
    p0_guess = float(np.clip(y[0], 0.0, 1.0))
    p_inf_guess = float(np.clip(np.mean(y[-edge_count:]), 0.0, 1.0))
    gamma_guess = 1.0 / max(0.4 * float(elapsed[-1]), 1.0)
    rate_upper = _rate_upper_bound(t)

    def fit_candidate(
        model: _EXCHANGE_MODEL,
    ) -> (
        tuple[
            float,
            _EXCHANGE_MODEL,
            OptimizeResult,
            Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
        ]
        | None
    ):
        if model == "outward_only":

            def evaluate(
                parameters: NDArray[np.float64],
                x: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                p0, gamma = parameters
                return 1.0 - (1.0 - p0) * np.exp(-gamma * x)

            x0 = np.array([p0_guess, gamma_guess], dtype=np.float64)
            lower = np.array([0.0, 0.0], dtype=np.float64)
            upper = np.array([1.0, rate_upper], dtype=np.float64)
        elif model == "inward_only":

            def evaluate(
                parameters: NDArray[np.float64],
                x: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                p0, gamma = parameters
                return p0 * np.exp(-gamma * x)

            x0 = np.array([p0_guess, gamma_guess], dtype=np.float64)
            lower = np.array([0.0, 0.0], dtype=np.float64)
            upper = np.array([1.0, rate_upper], dtype=np.float64)
        else:

            def evaluate(
                parameters: NDArray[np.float64],
                x: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                p0, p_inf, gamma = parameters
                return p_inf + (p0 - p_inf) * np.exp(-gamma * x)

            x0 = np.array([p0_guess, p_inf_guess, gamma_guess], dtype=np.float64)
            lower = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            upper = np.array([1.0, 1.0, rate_upper], dtype=np.float64)

        def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            return (evaluate(parameters, elapsed) - y) / scale

        optimization, _ = _run_robust_least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            robust_loss=robust_loss,
            max_nfev=1500,
        )
        if optimization is None:
            return None
        parameters = np.asarray(optimization.x, dtype=np.float64)
        if parameters.shape != x0.shape or not np.all(np.isfinite(parameters)):
            return None
        # One-way exchange is nested at p_inf=1 or p_inf=0 in the two-way
        # model. A BIC-like robust-cost score keeps the reduced model unless a
        # resolved finite asymptote materially improves the fit.
        score = 2.0 * float(optimization.cost) + parameters.size * np.log(y.size)
        return score, model, optimization, evaluate

    candidate_models = (
        (preferred_model,)
        if preferred_model == "two_way"
        else (preferred_model, "two_way")
    )
    candidates = [
        candidate
        for model_name in candidate_models
        if (candidate := fit_candidate(model_name)) is not None
    ]
    if not candidates:
        return PopulationExchangeFit(
            assessment=assessment,
            model=preferred_model,
            success=False,
            quality="failed",
            message="All numerical exchange-model fits failed.",
            outward_rate=float("nan"),
            inward_rate=float("nan"),
            outward_rate_standard_error=float("nan"),
            inward_rate_standard_error=float("nan"),
            exchange_time_constant=float("nan"),
            p0=float("nan"),
            p_inf=float("nan"),
            fitted_values=np.full_like(y, np.nan),
            curve_times=dense_t,
            curve_values=np.full_like(dense_t, np.nan),
            r_squared=float("nan"),
            reduced_chi_squared=float("nan"),
        )
    successful_candidates = [
        candidate for candidate in candidates if candidate[2].success
    ]
    _, model, optimization, evaluate = min(
        successful_candidates or candidates,
        key=lambda item: item[0],
    )
    params = np.asarray(optimization.x, dtype=np.float64)

    fitted = evaluate(params, elapsed)
    curve = evaluate(params, dense_elapsed)
    covariance = _approximate_covariance(
        optimization,
        y.size,
        params.size,
        absolute_errors=errors is not None,
    )
    if model == "outward_only":
        p0, gamma = map(float, params)
        p_inf = 1.0
        outward = gamma
        inward = 0.0
        outward_error = (
            float(np.sqrt(max(float(covariance[1, 1]), 0.0)))
            if covariance.shape == (2, 2) and np.isfinite(covariance[1, 1])
            else float("nan")
        )
        inward_error = float("nan")
    elif model == "inward_only":
        p0, gamma = map(float, params)
        p_inf = 0.0
        outward = 0.0
        inward = gamma
        outward_error = float("nan")
        inward_error = (
            float(np.sqrt(max(float(covariance[1, 1]), 0.0)))
            if covariance.shape == (2, 2) and np.isfinite(covariance[1, 1])
            else float("nan")
        )
    else:
        p0, p_inf, gamma = map(float, params)
        outward = p_inf * gamma
        inward = (1.0 - p_inf) * gamma
        if covariance.shape == (3, 3) and np.all(np.isfinite(covariance)):
            grad_out = np.array([0.0, gamma, p_inf], dtype=np.float64)
            grad_in = np.array([0.0, -gamma, 1.0 - p_inf], dtype=np.float64)
            outward_error = float(
                np.sqrt(max(float(grad_out @ covariance @ grad_out), 0.0))
            )
            inward_error = float(
                np.sqrt(max(float(grad_in @ covariance @ grad_in), 0.0))
            )
        else:
            outward_error = inward_error = float("nan")

    r2 = _r_squared(y, fitted)
    chi2 = _reduced_chi_squared(y, fitted, errors, params.size)
    quality = _fit_quality(True, r2, chi2)
    message = f"Selected {model} exchange model. {optimization.message}"
    if y.size - params.size <= 0:
        quality = "poor"
        message = (
            "Fit has no residual degrees of freedom; fit quality cannot be "
            "assessed. " + message
        )
    if not optimization.success:
        quality = "poor"
        message = "Finite diagnostic fit returned despite optimizer warning: " + message
    return PopulationExchangeFit(
        assessment=assessment,
        model=model,
        success=True,
        quality=quality,
        message=message,
        outward_rate=float(outward),
        inward_rate=float(inward),
        outward_rate_standard_error=outward_error,
        inward_rate_standard_error=inward_error,
        exchange_time_constant=float("inf") if gamma <= 0 else 1.0 / gamma,
        p0=float(p0),
        p_inf=float(p_inf),
        fitted_values=np.asarray(fitted, dtype=np.float64),
        curve_times=dense_t,
        curve_values=np.asarray(curve, dtype=np.float64),
        r_squared=r2,
        reduced_chi_squared=chi2,
    )


def _three_level_rate_matrix(
    gamma_e_to_g: float,
    gamma_g_to_e: float,
    gamma_f_to_e: float,
    gamma_e_to_f: float,
) -> NDArray[np.float64]:
    return np.array(
        [
            [-gamma_g_to_e, gamma_e_to_g, 0.0],
            [gamma_g_to_e, -(gamma_e_to_g + gamma_e_to_f), gamma_f_to_e],
            [0.0, gamma_e_to_f, -gamma_f_to_e],
        ],
        dtype=np.float64,
    )


def _population_trajectory(
    times: NDArray[np.float64],
    initial: NDArray[np.float64],
    rates: Mapping[str, float],
) -> NDArray[np.float64]:
    matrix = _three_level_rate_matrix(
        rates["gamma_e_to_g"],
        rates["gamma_g_to_e"],
        rates["gamma_f_to_e"],
        rates["gamma_e_to_f"],
    )
    return np.stack([expm(matrix * float(time)) @ initial for time in times])


def _fit_control_population_rates(
    times_ground: ArrayLike,
    times_excited: ArrayLike,
    populations_ground: ArrayLike,
    populations_excited: ArrayLike,
    errors_ground: ArrayLike | None,
    errors_excited: ArrayLike | None,
    covariances_ground: ArrayLike | None = None,
    covariances_excited: ArrayLike | None = None,
    *,
    population_minimum_change: float,
    leakage_minimum_change: float,
    change_sigma_threshold: float,
    robust_loss: str,
) -> ControlPopulationRateFit:
    t_ground = np.asarray(times_ground, dtype=np.float64)
    t_excited = np.asarray(times_excited, dtype=np.float64)
    p_ground = np.asarray(populations_ground, dtype=np.float64)
    p_excited = np.asarray(populations_excited, dtype=np.float64)
    if p_ground.shape != (t_ground.size, 3) or p_excited.shape != (t_excited.size, 3):
        raise ValueError("Control-state populations must have shape (n_points, 3).")
    e_ground = (
        None if errors_ground is None else np.asarray(errors_ground, dtype=np.float64)
    )
    e_excited = (
        None if errors_excited is None else np.asarray(errors_excited, dtype=np.float64)
    )
    c_ground = (
        None
        if covariances_ground is None
        else np.asarray(covariances_ground, dtype=np.float64)
    )
    c_excited = (
        None
        if covariances_excited is None
        else np.asarray(covariances_excited, dtype=np.float64)
    )
    for name, covariance, n_points in (
        ("covariances_ground", c_ground, t_ground.size),
        ("covariances_excited", c_excited, t_excited.size),
    ):
        if covariance is not None and covariance.shape != (n_points, 3, 3):
            raise ValueError(f"{name} must have shape ({n_points}, 3, 3).")

    assessments: dict[str, ChangeAssessment] = {}
    for protocol, times, populations, errors in (
        (_CONTROL_GROUND, t_ground, p_ground, e_ground),
        (_CONTROL_EXCITED, t_excited, p_excited, e_excited),
    ):
        for index, state in enumerate(_STATE_NAMES):
            threshold = (
                leakage_minimum_change if state == "f" else population_minimum_change
            )
            state_errors = None if errors is None else errors[:, index]
            assessments[f"{protocol}_P{state}"] = _assess_change(
                times,
                populations[:, index],
                state_errors,
                minimum_change=threshold,
                change_sigma_threshold=change_sigma_threshold,
            )

    def edge_delta(values: NDArray[np.float64]) -> float:
        count = max(1, values.size // 3)
        return float(np.mean(values[-count:]) - np.mean(values[:count]))

    active: list[str] = []
    half_population_threshold = 0.5 * population_minimum_change
    ground_pg_delta = edge_delta(p_ground[:, 0])
    ground_pe_delta = edge_delta(p_ground[:, 1])
    excited_pg_delta = edge_delta(p_excited[:, 0])

    # Ground preparation is mainly sensitive to g->e; excited preparation is
    # mainly sensitive to e->g.  Direction checks reduce false activation when
    # Pe changes only because population is leaking to F.
    ground_ge_changed = any(
        assessments[f"{_CONTROL_GROUND}_P{s}"].status == "changing" for s in ("g", "e")
    )
    if ground_ge_changed and (
        ground_pg_delta < -half_population_threshold
        or ground_pe_delta > half_population_threshold
    ):
        active.append("gamma_g_to_e")

    excited_ge_changed = any(
        assessments[f"{_CONTROL_EXCITED}_P{s}"].status == "changing" for s in ("g", "e")
    )
    if excited_ge_changed and excited_pg_delta > half_population_threshold:
        active.append("gamma_e_to_g")

    f_changes = [
        assessments[f"{protocol}_Pf"].status == "changing"
        for protocol in _CONTROL_STATE_PROTOCOLS
    ]
    if any(f_changes):
        deltas = np.array([edge_delta(p_ground[:, 2]), edge_delta(p_excited[:, 2])])
        half_threshold = 0.5 * leakage_minimum_change
        outward = bool(np.max(deltas) > half_threshold)
        inward = bool(np.min(deltas) < -half_threshold)
        if not outward and not inward:
            # Direction-ambiguous F dynamics: allow both rates rather than fail.
            outward = inward = True
        if outward:
            active.append("gamma_e_to_f")
        if inward:
            active.append("gamma_f_to_e")

    # If G/E populations are visibly changing but the directional heuristics did
    # not activate any transition and F dynamics are also unresolved, release
    # the preparation-relevant G/E rate rather than incorrectly label the whole
    # control trace as stable.
    ge_active = any(name in active for name in ("gamma_e_to_g", "gamma_g_to_e"))
    ef_active = any(name in active for name in ("gamma_e_to_f", "gamma_f_to_e"))
    if not ge_active and not ef_active:
        if ground_ge_changed:
            active.append("gamma_g_to_e")
        if excited_ge_changed:
            active.append("gamma_e_to_g")

    rate_names: tuple[str, ...] = (
        "gamma_e_to_g",
        "gamma_g_to_e",
        "gamma_f_to_e",
        "gamma_e_to_f",
    )
    active = [name for name in rate_names if name in active]
    initial_ground = np.clip(p_ground[0], 0.0, 1.0)
    initial_excited = np.clip(p_excited[0], 0.0, 1.0)
    initial_ground_sum = float(np.sum(initial_ground))
    initial_excited_sum = float(np.sum(initial_excited))
    if initial_ground_sum <= _EPS or initial_excited_sum <= _EPS:
        dense = np.linspace(0.0, max(float(t_ground[-1]), float(t_excited[-1])), 400)
        return ControlPopulationRateFit(
            success=False,
            quality="failed",
            message=(
                "Control-rate fit unavailable because the clipped n=0 G/E/F "
                "population vector has zero total weight."
            ),
            active_rates=(),
            rates={name: float("nan") for name in rate_names},
            rate_standard_errors={name: float("nan") for name in rate_names},
            time_constants={name: float("nan") for name in rate_names},
            covariance=np.full((4, 4), np.nan, dtype=np.float64),
            assessments=assessments,
            fitted_populations={
                _CONTROL_GROUND: np.full_like(p_ground, np.nan),
                _CONTROL_EXCITED: np.full_like(p_excited, np.nan),
            },
            curve_times=dense,
            curve_populations={
                _CONTROL_GROUND: np.full((dense.size, 3), np.nan),
                _CONTROL_EXCITED: np.full((dense.size, 3), np.nan),
            },
            r_squared=float("nan"),
            reduced_chi_squared=float("nan"),
            population_weighting="not_used_invalid_initial",
        )
    initial_ground = initial_ground / initial_ground_sum
    initial_excited = initial_excited / initial_excited_sum

    if not active:
        rates: dict[str, float] = dict.fromkeys(rate_names, 0.0)
        fitted = {
            _CONTROL_GROUND: np.tile(initial_ground, (t_ground.size, 1)),
            _CONTROL_EXCITED: np.tile(initial_excited, (t_excited.size, 1)),
        }
        dense = np.linspace(0.0, max(float(t_ground[-1]), float(t_excited[-1])), 400)
        curves = {
            _CONTROL_GROUND: np.tile(initial_ground, (dense.size, 1)),
            _CONTROL_EXCITED: np.tile(initial_excited, (dense.size, 1)),
        }
        # n=0 defines the fixed initial population and is not reused as a
        # goodness-of-fit data point.
        observed = np.concatenate([p_ground[1:].ravel(), p_excited[1:].ravel()])
        predicted = np.concatenate(
            [
                fitted[_CONTROL_GROUND][1:].ravel(),
                fitted[_CONTROL_EXCITED][1:].ravel(),
            ]
        )
        return ControlPopulationRateFit(
            success=True,
            quality="stable",
            message="No resolvable control-population dynamics; all nominal rates fixed to zero.",
            active_rates=(),
            rates=rates,
            rate_standard_errors={name: float("nan") for name in rate_names},
            time_constants={name: float("inf") for name in rate_names},
            covariance=np.full((4, 4), np.nan, dtype=np.float64),
            assessments=assessments,
            fitted_populations=fitted,
            curve_times=dense,
            curve_populations=curves,
            r_squared=_r_squared(observed, predicted),
            reduced_chi_squared=float("nan"),
            population_weighting="not_used_stable",
        )

    positive_times = np.concatenate([t_ground[t_ground > 0], t_excited[t_excited > 0]])
    time_scale = float(np.max(positive_times)) if positive_times.size else 1.0
    rate0 = 1.0 / max(0.5 * time_scale, 1.0)
    positive_steps = np.concatenate([np.diff(t_ground), np.diff(t_excited)])
    positive_steps = positive_steps[positive_steps > 0]
    min_step = float(np.min(positive_steps)) if positive_steps.size else time_scale
    upper = max(20.0 / max(min_step, 1e-12), rate0 * 100.0)

    weighted_errors_ground = (
        None if e_ground is None else _finite_errors(e_ground, p_ground.shape)
    )
    weighted_errors_excited = (
        None if e_excited is None else _finite_errors(e_excited, p_excited.shape)
    )

    # n=0 is used only to define the initial population vector.  Fits and fit
    # quality start at n=1 so that the same measurement is not counted twice.
    fit_ground = slice(1, None)
    fit_excited = slice(1, None)

    ground_whiteners: list[NDArray[np.float64]] | None = None
    excited_whiteners: list[NDArray[np.float64]] | None = None
    ground_effective_count = 3 * max(t_ground.size - 1, 0)
    excited_effective_count = 3 * max(t_excited.size - 1, 0)
    if c_ground is not None and c_excited is not None:
        ground_whiteners = []
        excited_whiteners = []
        ground_effective_count = 0
        excited_effective_count = 0
        try:
            for covariance in c_ground[fit_ground]:
                whitener, rank = _population_covariance_whitener(covariance)
                ground_whiteners.append(whitener)
                ground_effective_count += rank
            for covariance in c_excited[fit_excited]:
                whitener, rank = _population_covariance_whitener(covariance)
                excited_whiteners.append(whitener)
                excited_effective_count += rank
        except ValueError:
            # Keep offline analysis usable when optional covariance data are
            # incomplete or malformed.
            ground_whiteners = None
            excited_whiteners = None

    scale_ground = (
        weighted_errors_ground
        if weighted_errors_ground is not None
        else np.full_like(p_ground, max(float(np.std(p_ground)), 1e-3))
    )
    scale_excited = (
        weighted_errors_excited
        if weighted_errors_excited is not None
        else np.full_like(p_excited, max(float(np.std(p_excited)), 1e-3))
    )
    use_full_covariance = ground_whiteners is not None and excited_whiteners is not None
    if use_full_covariance:
        population_weighting: _POPULATION_WEIGHTING = "full_covariance"
    elif weighted_errors_ground is not None and weighted_errors_excited is not None:
        population_weighting = "component_se_diagonal"
    elif weighted_errors_ground is None and weighted_errors_excited is None:
        population_weighting = "empirical_scale"
    else:
        population_weighting = "mixed_component_se_empirical"
    if not use_full_covariance:
        ground_effective_count = 3 * max(t_ground.size - 1, 0)
        excited_effective_count = 3 * max(t_excited.size - 1, 0)

    def unpack(
        parameters: NDArray[np.float64], active_rates: Sequence[str]
    ) -> dict[str, float]:
        rates: dict[str, float] = dict.fromkeys(rate_names, 0.0)
        for name, value in zip(active_rates, parameters, strict=True):
            rates[name] = float(value)
        return rates

    def residual(
        parameters: NDArray[np.float64], active_rates: Sequence[str]
    ) -> NDArray[np.float64]:
        rates = unpack(parameters, active_rates)
        model_ground = _population_trajectory(t_ground, initial_ground, rates)
        model_excited = _population_trajectory(t_excited, initial_excited, rates)
        delta_ground = model_ground[fit_ground] - p_ground[fit_ground]
        delta_excited = model_excited[fit_excited] - p_excited[fit_excited]
        if use_full_covariance:
            if (
                ground_whiteners is None or excited_whiteners is None
            ):  # pragma: no cover
                raise RuntimeError(
                    "Full-covariance weighting was selected without whiteners."
                )
            return np.concatenate(
                [
                    *(
                        whitener @ delta
                        for whitener, delta in zip(
                            ground_whiteners, delta_ground, strict=True
                        )
                    ),
                    *(
                        whitener @ delta
                        for whitener, delta in zip(
                            excited_whiteners, delta_excited, strict=True
                        )
                    ),
                ]
            )
        return np.concatenate(
            [
                (delta_ground / scale_ground[fit_ground]).ravel(),
                (delta_excited / scale_excited[fit_excited]).ravel(),
            ]
        )

    def fit_candidate(
        active_rates: tuple[str, ...],
    ) -> tuple[float, tuple[str, ...], OptimizeResult] | None:
        x0 = np.full(len(active_rates), rate0, dtype=np.float64)

        def candidate_residual(
            parameters: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return residual(parameters, active_rates)

        optimization, _ = _run_robust_least_squares(
            candidate_residual,
            x0=x0,
            bounds=(np.zeros_like(x0), np.full_like(x0, upper)),
            robust_loss=robust_loss,
            max_nfev=2000,
        )
        if optimization is None:
            return None
        parameters = np.asarray(optimization.x, dtype=np.float64)
        if parameters.shape != x0.shape or not np.all(np.isfinite(parameters)):
            return None
        residual_count = ground_effective_count + excited_effective_count
        score = 2.0 * float(optimization.cost) + len(active_rates) * np.log(
            max(residual_count, 1)
        )
        return score, active_rates, optimization

    reduced_active = tuple(active)
    candidate_rate_sets = [reduced_active]
    ef_rates = {"gamma_e_to_f", "gamma_f_to_e"}
    if any(f_changes) and len(ef_rates.intersection(reduced_active)) == 1:
        candidate_rate_sets.append(
            tuple(
                name
                for name in rate_names
                if name in reduced_active or name in ef_rates
            )
        )
    candidates = [
        candidate
        for active_rates in candidate_rate_sets
        if (candidate := fit_candidate(active_rates)) is not None
    ]
    if not candidates:
        nan_rates = {
            name: float("nan") if name in active else 0.0 for name in rate_names
        }
        return ControlPopulationRateFit(
            success=False,
            quality="failed",
            message="All numerical control-rate model fits failed.",
            active_rates=tuple(active),
            rates=nan_rates,
            rate_standard_errors={name: float("nan") for name in rate_names},
            time_constants={
                name: float("nan") if name in active else float("inf")
                for name in rate_names
            },
            covariance=np.full((4, 4), np.nan),
            assessments=assessments,
            fitted_populations={
                _CONTROL_GROUND: np.full_like(p_ground, np.nan),
                _CONTROL_EXCITED: np.full_like(p_excited, np.nan),
            },
            curve_times=np.linspace(
                0.0, max(float(t_ground[-1]), float(t_excited[-1])), 400
            ),
            curve_populations={
                _CONTROL_GROUND: np.full((400, 3), np.nan),
                _CONTROL_EXCITED: np.full((400, 3), np.nan),
            },
            r_squared=float("nan"),
            reduced_chi_squared=float("nan"),
            population_weighting=population_weighting,
        )
    successful_candidates = [
        candidate for candidate in candidates if candidate[2].success
    ]
    _, selected_active, optimization = min(
        successful_candidates or candidates,
        key=lambda item: item[0],
    )
    active = list(selected_active)
    params = np.asarray(optimization.x, dtype=np.float64)

    rates = unpack(params, active)
    fitted = {
        _CONTROL_GROUND: _population_trajectory(t_ground, initial_ground, rates),
        _CONTROL_EXCITED: _population_trajectory(t_excited, initial_excited, rates),
    }
    dense = np.linspace(0.0, max(float(t_ground[-1]), float(t_excited[-1])), 400)
    curves = {
        _CONTROL_GROUND: _population_trajectory(dense, initial_ground, rates),
        _CONTROL_EXCITED: _population_trajectory(dense, initial_excited, rates),
    }
    covariance_active = _approximate_covariance(
        optimization,
        residual_count=ground_effective_count + excited_effective_count,
        parameter_count=len(active),
        absolute_errors=(
            use_full_covariance
            or (
                weighted_errors_ground is not None
                and weighted_errors_excited is not None
            )
        ),
    )
    # Inactive rates are nominal zero because they were not resolved, not
    # parameters known with zero variance. Keep their covariance entries NaN.
    covariance = np.full((4, 4), np.nan, dtype=np.float64)
    errors = {name: float("nan") for name in rate_names}
    index_map = {name: i for i, name in enumerate(rate_names)}
    if covariance_active.shape == (len(active), len(active)):
        for i, name_i in enumerate(active):
            for j, name_j in enumerate(active):
                covariance[index_map[name_i], index_map[name_j]] = covariance_active[
                    i, j
                ]
            variance = covariance_active[i, i]
            if np.isfinite(variance):
                errors[name_i] = float(np.sqrt(max(float(variance), 0.0)))
    time_constants = {
        name: _rate_to_lifetime(rates[name], errors[name])[0] for name in rate_names
    }
    observed = np.concatenate(
        [p_ground[fit_ground].ravel(), p_excited[fit_excited].ravel()]
    )
    predicted = np.concatenate(
        [
            fitted[_CONTROL_GROUND][fit_ground].ravel(),
            fitted[_CONTROL_EXCITED][fit_excited].ravel(),
        ]
    )
    r2 = _r_squared(observed, predicted)
    if use_full_covariance:
        whitened = residual(params, active)
        dof = whitened.size - len(active)
        chi2 = float(np.sum(np.square(whitened)) / dof) if dof > 0 else float("nan")
    else:
        combined_errors = None
        if weighted_errors_ground is not None and weighted_errors_excited is not None:
            combined_errors = np.concatenate(
                [
                    weighted_errors_ground[fit_ground].ravel(),
                    weighted_errors_excited[fit_excited].ravel(),
                ]
            )
        chi2 = _reduced_chi_squared(
            observed,
            predicted,
            combined_errors,
            len(active),
        )
    quality = _fit_quality(True, r2, chi2)
    message = f"Selected rates: {', '.join(active)}. {optimization.message}"
    if not optimization.success:
        quality = "poor"
        message = "Finite diagnostic fit returned despite optimizer warning: " + message
    return ControlPopulationRateFit(
        success=True,
        quality=quality,
        message=message,
        active_rates=tuple(active),
        rates=rates,
        rate_standard_errors=errors,
        time_constants=time_constants,
        covariance=covariance,
        assessments=assessments,
        fitted_populations=fitted,
        curve_times=dense,
        curve_populations=curves,
        r_squared=r2,
        reduced_chi_squared=chi2,
        population_weighting=population_weighting,
    )


# ---------------------------------------------------------------------------
# Fidelity estimate
# ---------------------------------------------------------------------------


def _load_idle_noise(
    exp: Experiment,
    control: str,
    target: str,
    idle_t1: Mapping[str, float] | None,
    idle_t2_echo: Mapping[str, float] | None,
    idle_t2_star: Mapping[str, float] | None = None,
) -> tuple[IdleNoiseParameters, IdleNoiseParameters]:
    """Load idle lifetimes, using T2-echo only for missing stored T2-star values."""
    t1 = (
        exp.ctx.system_manager.config_loader.load_param_data("t1")
        if idle_t1 is None
        else idle_t1
    )
    t2 = (
        exp.ctx.system_manager.config_loader.load_param_data("t2_echo")
        if idle_t2_echo is None
        else idle_t2_echo
    )
    t2_star_loaded_from_storage = idle_t2_star is None
    if idle_t2_star is not None:
        t2_star = idle_t2_star
    else:
        try:
            t2_star = exp.ctx.system_manager.config_loader.load_param_data("t2_star")
        except (
            AttributeError,
            FileNotFoundError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            warnings.warn(
                "Stored t2_star is unavailable; using t2_echo for unrefocused "
                "A/B blanks.",
                RuntimeWarning,
                stacklevel=2,
            )
            t2_star = t2

    def t2_star_for(qubit: str) -> float:
        try:
            return t2_star[qubit]
        except KeyError:
            if not t2_star_loaded_from_storage:
                raise
            warnings.warn(
                f"Stored t2_star is unavailable for {qubit}; using t2_echo for "
                "its unrefocused A/B blanks.",
                RuntimeWarning,
                stacklevel=2,
            )
            return t2[qubit]

    try:
        control_noise = IdleNoiseParameters(
            t1[control], t2[control], t2_star_for(control)
        )
        target_noise = IdleNoiseParameters(t1[target], t2[target], t2_star_for(target))
    except KeyError as exc:
        raise ValueError(
            f"Idle coherence data missing for qubit {exc.args[0]}."
        ) from exc
    else:
        return control_noise, target_noise


def _single_qubit_entanglement_fidelity(lx: float, ly: float, lz: float) -> float:
    return float(np.clip((1.0 + lx + ly + lz) / 4.0, 0.0, 1.0))


def _rough_fidelity_from_parameters(
    *,
    gate_total_duration: float,
    gate_cr_duration: float,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    control_xy_rate: float,
    control_z_rate: float,
    target_t1rho_rate: float,
    target_t2rho_rate: float,
    control_leakage_rate: float,
    target_leakage_rate: float,
) -> tuple[float, float, tuple[float, float, float], tuple[float, float, float]]:
    t_cr = max(float(gate_cr_duration), 0.0)
    t_off = max(float(gate_total_duration) - t_cr, 0.0)

    # Dissipation rates are nominally zero when a change is not resolved.
    # For fidelity only, idle decoherence is used as the fallback for those
    # nominal-zero rates.  Positive measured rates are kept as measured, even
    # when they are lower than the corresponding idle rate.
    control_xy_active = (
        1.0 / control_idle.effective_t2_echo
        if control_xy_rate <= 0.0
        else control_xy_rate
    )
    control_z_active = (
        1.0 / control_idle.t1 if control_z_rate <= 0.0 else control_z_rate
    )
    target_x_active = (
        1.0 / target_idle.effective_t2_echo
        if target_t1rho_rate <= 0.0
        else target_t1rho_rate
    )
    target_y_active = (
        1.0 / target_idle.effective_t2_echo
        if target_t2rho_rate <= 0.0
        else target_t2rho_rate
    )
    target_z_active = (
        1.0 / target_idle.t1 if target_t2rho_rate <= 0.0 else target_t2rho_rate
    )

    control_lx = np.exp(
        -control_xy_active * t_cr - t_off / control_idle.effective_t2_echo
    )
    control_ly = control_lx
    control_lz = np.exp(-control_z_active * t_cr - t_off / control_idle.t1)

    target_lx = np.exp(-target_x_active * t_cr - t_off / target_idle.effective_t2_echo)
    target_ly = np.exp(-target_y_active * t_cr - t_off / target_idle.effective_t2_echo)
    target_lz = np.exp(-target_z_active * t_cr - t_off / target_idle.t1)

    fe_control = _single_qubit_entanglement_fidelity(control_lx, control_ly, control_lz)
    fe_target = _single_qubit_entanglement_fidelity(target_lx, target_ly, target_lz)
    fe_pair = fe_control * fe_target

    # Control e->f leakage is weighted by an average 1/2 computational e
    # occupation.  Target outward leakage is an effective rate from the full
    # computational manifold.  Seepage is ignored over one short gate.
    survival = float(
        np.exp(
            -t_cr
            * (0.5 * max(control_leakage_rate, 0.0) + max(target_leakage_rate, 0.0))
        )
    )
    fidelity = survival * (4.0 * fe_pair + 1.0) / 5.0
    return (
        float(np.clip(fidelity, 0.0, 1.0)),
        float(np.clip(survival, 0.0, 1.0)),
        (float(control_lx), float(control_ly), float(control_lz)),
        (float(target_lx), float(target_ly), float(target_lz)),
    )


def _idle_fidelity_limit(
    gate_duration: float,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
) -> float:
    cxy = np.exp(-gate_duration / control_idle.effective_t2_echo)
    cz = np.exp(-gate_duration / control_idle.t1)
    txy = np.exp(-gate_duration / target_idle.effective_t2_echo)
    tz = np.exp(-gate_duration / target_idle.t1)
    fe_c = _single_qubit_entanglement_fidelity(cxy, cxy, cz)
    fe_t = _single_qubit_entanglement_fidelity(txy, txy, tz)
    return float(np.clip((4.0 * fe_c * fe_t + 1.0) / 5.0, 0.0, 1.0))


def _fidelity_standard_error(
    parameter_values: Mapping[str, float],
    parameter_errors: Mapping[str, float],
    evaluator: Callable[[Mapping[str, float]], float],
) -> float:
    """Propagate available local rate errors without inventing zero uncertainty."""
    baseline = float(evaluator(parameter_values))
    variance = 0.0
    propagated = False
    for name, value in parameter_values.items():
        sigma = float(parameter_errors.get(name, float("nan")))
        if not np.isfinite(sigma) or sigma < 0:
            # A nominal stable-zero rate is unresolved below the dissipation characterization
            # threshold, not known exactly. Its missing uncertainty therefore
            # leaves the aggregate fidelity uncertainty unresolved as well.
            return float("nan")
        if sigma == 0:
            continue
        propagated = True
        step = max(abs(value) * 1e-4, sigma * 1e-2, 1e-10)
        shifted = dict(parameter_values)
        shifted[name] = max(0.0, value + step)
        derivative = (float(evaluator(shifted)) - baseline) / step
        variance += derivative**2 * sigma**2
    return float(np.sqrt(max(variance, 0.0))) if propagated else float("nan")


def _optional_population_covariance(
    population_covariances: Mapping[
        str, Mapping[str, Mapping[str, NDArray[np.float64]]]
    ]
    | None,
    protocol: str,
    kind: str,
    qubit: str,
    *,
    n_points: int,
) -> NDArray[np.float64] | None:
    """
    Return one optional covariance stack, or `None` when unusable or missing.

    Full population covariance is a best-effort refinement for offline analysis.
    Partially populated measurement objects are therefore allowed to
    omit any protocol/kind/qubit entry.  Malformed entries likewise fall back to
    component-wise standard errors rather than aborting the entire dissipation characterization.
    """
    if population_covariances is None:
        return None
    try:
        value = population_covariances[protocol][kind][qubit]
    except (KeyError, TypeError):
        return None
    try:
        covariance = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if covariance.shape != (n_points, 3, 3):
        return None
    return covariance


def _validate_dissipation_measurements(measurements: CrDissipationMeasurements) -> None:
    """Validate the array structure required by offline dissipation analysis."""
    if measurements.control_qubit == measurements.target_qubit:
        raise ValueError("control_qubit and target_qubit must differ.")
    n_values = _validate_n_values(measurements.n_values)
    n_points = len(n_values)
    if not isinstance(measurements.diagnostic_components_measured, bool):
        raise TypeError("diagnostic_components_measured must be boolean.")

    def array(
        name: str,
        value: ArrayLike,
        shape: tuple[int, ...],
        *,
        errors: bool = False,
        allow_nan: bool = False,
    ) -> NDArray[np.float64]:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {result.shape}.")
        if errors:
            valid = np.isnan(result) | (np.isfinite(result) & (result >= 0))
        elif allow_nan:
            valid = np.isnan(result) | np.isfinite(result)
        else:
            valid = np.isfinite(result)
        if not np.all(valid):
            requirement = (
                "nonnegative finite values or NaN"
                if errors
                else "finite values or NaN"
                if allow_nan
                else "finite values"
            )
            raise ValueError(f"{name} must contain {requirement}.")
        return result

    for protocol in (*_CONTROL_STATE_PROTOCOLS, *_ECHO_PROTOCOLS):
        total = array(
            f"total_times[{protocol}]", measurements.total_times[protocol], (n_points,)
        )
        active = array(
            f"cr_active_times[{protocol}]",
            measurements.cr_active_times[protocol],
            (n_points,),
        )
        if (
            not np.isclose(total[0], 0.0, atol=1e-12, rtol=0.0)
            or not np.isclose(active[0], 0.0, atol=1e-12, rtol=0.0)
            or np.any(np.diff(total) <= 0)
            or np.any(np.diff(active) <= 0)
        ):
            raise ValueError(
                f"{protocol} time axes must start at zero and increase strictly."
            )
        if np.any(active < 0) or np.any(active > total + 1e-9):
            raise ValueError(f"{protocol} CR-active times must lie within total times.")
        duty_cycle = active[1:] / total[1:]
        if not np.allclose(duty_cycle, duty_cycle[0], rtol=1e-9, atol=1e-12):
            raise ValueError(f"{protocol} must have a constant CR-active duty cycle.")

    for protocol in (*_CONTROL_STATE_PROTOCOLS, *_ECHO_PROTOCOLS):
        if protocol not in measurements.populations:
            if protocol in _ECHO_PROTOCOLS:
                continue
            raise ValueError(f"populations[{protocol}] is required.")
        for kind in ("actual", "reference"):
            for qubit in (measurements.control_qubit, measurements.target_qubit):
                population = array(
                    f"populations[{protocol}][{kind}][{qubit}]",
                    measurements.populations[protocol][kind][qubit],
                    (n_points, 3),
                )
                if (
                    np.any(population < -_PROBABILITY_TOLERANCE)
                    or np.any(population > 1.0 + _PROBABILITY_TOLERANCE)
                    or not np.allclose(
                        np.sum(population, axis=1),
                        1.0,
                        rtol=0.0,
                        atol=_PROBABILITY_TOLERANCE,
                    )
                ):
                    raise ValueError(
                        f"populations[{protocol}][{kind}][{qubit}] must contain "
                        "vectors on the G/E/F probability simplex."
                    )
                array(
                    f"population_standard_errors[{protocol}][{kind}][{qubit}]",
                    measurements.population_standard_errors[protocol][kind][qubit],
                    (n_points, 3),
                    errors=True,
                )
            if protocol in _CONTROL_STATE_PROTOCOLS:
                array(
                    f"target_x_standard_errors[{protocol}][{kind}]",
                    measurements.target_x_standard_errors[protocol][kind],
                    (n_points,),
                    errors=True,
                )
            # Full population covariance is optional and best-effort.  Missing
            # target covariance, partial legacy mappings, malformed shapes, and
            # unusable matrices must not abort offline analysis; the control-rate
            # fit falls back to component-wise SE weighting as needed.

    for protocol, primary in (
        (_CONTROL_TRANSVERSE_ECHO, "X"),
        (_TARGET_ROTATING_FRAME_ECHO, "Y"),
    ):
        for kind in ("actual", "reference"):
            array(
                f"pauli_expectations[{protocol}][{primary}][{kind}]",
                measurements.pauli_expectations[protocol][primary][kind],
                (n_points,),
                allow_nan=True,
            )
            array(
                f"pauli_standard_errors[{protocol}][{primary}][{kind}]",
                measurements.pauli_standard_errors[protocol][primary][kind],
                (n_points,),
                errors=True,
            )

    if measurements.diagnostic_components_measured:
        diagnostic_specs = {
            _CONTROL_TRANSVERSE_ECHO: ("Y", "Z"),
            _TARGET_ROTATING_FRAME_ECHO: ("X", "Z"),
            _diagnostic_key(_CONTROL_GROUND, "control"): ("X", "Y"),
            _diagnostic_key(_CONTROL_GROUND, "target"): ("Y", "Z"),
            _diagnostic_key(_CONTROL_EXCITED, "control"): ("X", "Y"),
            _diagnostic_key(_CONTROL_EXCITED, "target"): ("Y", "Z"),
        }
        for key, components in diagnostic_specs.items():
            for component in components:
                for kind in ("actual", "reference"):
                    array(
                        f"pauli_expectations[{key}][{component}][{kind}]",
                        measurements.pauli_expectations[key][component][kind],
                        (n_points,),
                        allow_nan=True,
                    )
                    array(
                        f"pauli_standard_errors[{key}][{component}][{kind}]",
                        measurements.pauli_standard_errors[key][component][kind],
                        (n_points,),
                        errors=True,
                    )


# ---------------------------------------------------------------------------
# Offline dissipation analysis
# ---------------------------------------------------------------------------


def _mean_rate_and_error(fits: Sequence[ExponentialDecayFit]) -> tuple[float, float]:
    """
    Average control-ground/control-excited rates only when both fits are resolved.

    Stable fits contribute their nominal zero rate, but their local uncertainty
    remains unresolved (`NaN`).  A numerical fit failure is not silently replaced
    by the other preparation, because doing so would make the aggregate rate and
    rough fidelity optimistic.
    """
    if not fits or any(not fit.success or not np.isfinite(fit.rate) for fit in fits):
        return float("nan"), float("nan")
    rate = float(np.mean([fit.rate for fit in fits]))
    local_errors = [fit.rate_standard_error for fit in fits]
    error = (
        float(np.sqrt(np.sum(np.square(local_errors))) / len(fits))
        if all(np.isfinite(value) and value >= 0 for value in local_errors)
        else float("nan")
    )
    return rate, error


def _mean_exchange_rate_and_error(
    fits: Sequence[PopulationExchangeFit],
    direction: Literal["outward", "inward"],
) -> tuple[float, float]:
    """
    Average control-ground/control-excited exchange rates when both fits are resolved.

    Stable fits are fixed to nominal zero while their local uncertainty remains
    unresolved (`NaN`).  A failed fit keeps the aggregate quantity unresolved
    rather than silently discarding one control preparation.
    """
    if not fits or any(not fit.success for fit in fits):
        return float("nan"), float("nan")
    if direction == "outward":
        values = [fit.outward_rate for fit in fits]
        raw_errors = [fit.outward_rate_standard_error for fit in fits]
    else:
        values = [fit.inward_rate for fit in fits]
        raw_errors = [fit.inward_rate_standard_error for fit in fits]
    if not all(np.isfinite(value) for value in values):
        return float("nan"), float("nan")
    local_errors = list(raw_errors)
    value = float(np.mean(values))
    error = (
        float(np.sqrt(np.sum(np.square(local_errors))) / len(fits))
        if all(np.isfinite(item) and item >= 0 for item in local_errors)
        else float("nan")
    )
    return value, error


def _cr_active_rate_from_transverse_echo_fits(
    measurements: CrDissipationMeasurements,
    protocol: _ECHO_PROTOCOL,
    actual: ExponentialDecayFit,
    reference: ExponentialDecayFit,
    *,
    idle_rate: float | None,
) -> tuple[float, float, float]:
    """
    Convert total-sequence decay into a rough CR-active rate.

    The model assumes the measured total decay rate is a duty-cycle mixture of
    one effective CR-active rate and the independently measured idle rate
    during CR-off time. If idle data are unavailable, the reference rate is
    retained as a documented fallback.
    The third return value is the signed actual-minus-reference excess rate,
    normalized to CR-active time.
    """
    valid_idle_rate = (
        idle_rate is not None and np.isfinite(idle_rate) and idle_rate >= 0.0
    )
    if not actual.success or not np.isfinite(actual.rate):
        return float("nan"), float("nan"), float("nan")
    if not valid_idle_rate and (
        not reference.success or not np.isfinite(reference.rate)
    ):
        return float("nan"), float("nan"), float("nan")
    total = np.asarray(measurements.total_times[protocol], dtype=np.float64)
    active = np.asarray(measurements.cr_active_times[protocol], dtype=np.float64)
    positive = total > 0
    if not np.any(positive):
        return 0.0, float("nan"), 0.0
    duty_values = active[positive] / total[positive]
    duty = float(np.median(duty_values))
    if not np.isfinite(duty) or duty <= 0 or duty > 1 + 1e-9:
        return float("nan"), float("nan"), float("nan")
    duty = min(duty, 1.0)
    off_rate = float(idle_rate) if valid_idle_rate else float(reference.rate)
    raw = (actual.rate - (1.0 - duty) * off_rate) / duty
    if not np.isfinite(raw):
        return float("nan"), float("nan"), float("nan")
    # Keep a nonnegative dissipation rate for lifetimes/fidelity, but preserve the
    # signed duty-cycle-corrected difference as diagnostic information.
    difference = (
        float((actual.rate - reference.rate) / duty)
        if reference.success and np.isfinite(reference.rate)
        else float("nan")
    )
    cr_rate = max(0.0, float(raw))

    def local_error(fit: ExponentialDecayFit) -> float:
        return (
            float(fit.rate_standard_error)
            if np.isfinite(fit.rate_standard_error) and fit.rate_standard_error >= 0
            else float("nan")
        )

    aerr = local_error(actual)
    if valid_idle_rate:
        error = float(aerr / duty) if np.isfinite(aerr) else float("nan")
    else:
        rerr = local_error(reference)
        error = (
            float(np.sqrt(aerr**2 + ((1.0 - duty) * rerr) ** 2) / duty)
            if np.isfinite(aerr) and np.isfinite(rerr)
            else float("nan")
        )
    return cr_rate, error, difference


def _rate_difference(actual: float, reference: float) -> float:
    """Return signed actual-reference without converting NaN into zero."""
    if not np.isfinite(actual) or not np.isfinite(reference):
        return float("nan")
    return float(actual - reference)


def _subtract_known_rate(rate: float, known_contribution: float) -> float:
    """Subtract a fixed contribution without converting an unresolved rate to zero."""
    if not np.isfinite(rate) or not np.isfinite(known_contribution):
        return float("nan")
    return max(0.0, float(rate - known_contribution))


def _nonnegative_rate_difference(total: float, contribution: float) -> float:
    """Return a nonnegative residual rate while preserving unresolved inputs."""
    if not np.isfinite(total) or not np.isfinite(contribution):
        return float("nan")
    return max(0.0, float(total - contribution))


def analyze_cr_dissipation(
    measurements: CrDissipationMeasurements,
    *,
    population_minimum_change: float = 0.01,
    leakage_minimum_change: float = 0.003,
    pauli_minimum_change: float = 0.05,
    change_sigma_threshold: float = 3.0,
    robust_loss: str = "soft_l1",
    zx90_echo_timing: ZX90Timing | None = None,
    control_idle_noise: IdleNoiseParameters | None = None,
    target_idle_noise: IdleNoiseParameters | None = None,
    estimate_fidelity: bool = True,
    estimate_fidelity_uncertainty: bool = True,
) -> CrDissipationAnalysis:
    """
    Analyze saved CR-pulse dissipation characterization measurements without hardware access.

    Parameters
    ----------
    measurements
        Processed measurements returned by `characterize_cr_dissipation`
        or an equivalent offline data set.
    population_minimum_change
        Minimum resolvable absolute control G/E population change before a
        transition rate is allowed to vary.
    leakage_minimum_change
        Minimum resolvable absolute F-state population change before an
        effective leakage/seepage rate is allowed to vary.
    pauli_minimum_change
        Minimum resolvable absolute Pauli-expectation change before an
        exponential decay rate is fitted.
    change_sigma_threshold
        Required robust-range-to-point-SE ratio for classifying a curve as
        changing.  A curve must pass both this threshold and its absolute
        minimum-change threshold.
    robust_loss
        Loss passed to `scipy.optimize.least_squares` for changing curves.
    zx90_echo_timing, control_idle_noise, target_idle_noise
        Echo timing and independently measured idle noise. Idle T1/T2 are used
        to remove blank/CR-off contributions from the reported CR-active rates;
        they are also required for the optional fidelity estimate. If idle
        noise is omitted, echo rates fall back to reference-based correction.
    estimate_fidelity
        Whether to calculate the diagnostic fidelity estimate when its inputs
        and all required dissipation rates are available.
    estimate_fidelity_uncertainty
        Whether to propagate local fit standard errors to the diagnostic
        fidelity with a fast diagonal finite-difference approximation. The
        result remains unresolved if any required rate error is unavailable.

    Returns
    -------
    CrDissipationAnalysis
        Lightweight rate fits, approximate time constants, signed rate
        differences from the duration-matched references, quality flags, and the
        optional rough fidelity estimate.

    Notes
    -----
    Stable curves are assigned a nominal zero rate and are not nonlinearly fit.
    A derived decay observable containing non-finite values is marked invalid and
    that fit alone is left unresolved so independent diagnostics remain usable.
    Changing curves use small robust models.  A finite fit is retained even
    when its residual quality is poor; the quality flag communicates that fact.
    Optional orthogonal diagnostic components are deliberately excluded from
    every fit. Echo-protocol rates are CR-active-equivalent diagnostics, not
    microscopic Lindblad parameters. Their references preserve the internal
    echoed-ZX90 control-pi slots. Independently measured idle T2 is used as the
    nominal CR-off decay; the reference fit is only a fallback. The
    control-population fit
    fixes each initial population vector to the clipped, normalized `n=0`
    measurement and does not propagate its uncertainty separately.
    Population rows outside the G/E/F probability simplex are rejected.
    """
    _validate_dissipation_measurements(measurements)
    for name, value in (
        ("estimate_fidelity", estimate_fidelity),
        ("estimate_fidelity_uncertainty", estimate_fidelity_uncertainty),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")
    if zx90_echo_timing is not None:
        if not isinstance(zx90_echo_timing, ZX90Timing):
            raise TypeError("zx90_echo_timing must be ZX90Timing or None.")
        if not zx90_echo_timing.echo:
            raise ValueError("zx90_echo_timing must describe an echoed ZX90.")
    for name, value in (
        ("control_idle_noise", control_idle_noise),
        ("target_idle_noise", target_idle_noise),
    ):
        if value is not None and not isinstance(value, IdleNoiseParameters):
            raise TypeError(f"{name} must be IdleNoiseParameters or None.")
    robust_loss = _validate_robust_loss(robust_loss)

    population_minimum_change = _nonnegative_real(
        population_minimum_change, name="population_minimum_change"
    )
    leakage_minimum_change = _nonnegative_real(
        leakage_minimum_change, name="leakage_minimum_change"
    )
    pauli_minimum_change = _nonnegative_real(
        pauli_minimum_change, name="pauli_minimum_change"
    )
    change_sigma_threshold = _positive_real(
        change_sigma_threshold, default=3.0, name="change_sigma_threshold"
    )

    control_fits: dict[str, ControlPopulationRateFit] = {}
    target_t1rho_fits: dict[str, dict[str, ExponentialDecayFit]] = {
        _CONTROL_GROUND: {},
        _CONTROL_EXCITED: {},
    }
    target_leakage_fits: dict[str, dict[str, PopulationExchangeFit]] = {
        _CONTROL_GROUND: {},
        _CONTROL_EXCITED: {},
    }

    target_x = measurements.target_x
    for kind in ("actual", "reference"):
        control_fits[kind] = _fit_control_population_rates(
            measurements.cr_active_times[_CONTROL_GROUND],
            measurements.cr_active_times[_CONTROL_EXCITED],
            measurements.populations[_CONTROL_GROUND][kind][measurements.control_qubit],
            measurements.populations[_CONTROL_EXCITED][kind][
                measurements.control_qubit
            ],
            measurements.population_standard_errors[_CONTROL_GROUND][kind][
                measurements.control_qubit
            ],
            measurements.population_standard_errors[_CONTROL_EXCITED][kind][
                measurements.control_qubit
            ],
            _optional_population_covariance(
                measurements.population_covariances,
                _CONTROL_GROUND,
                kind,
                measurements.control_qubit,
                n_points=len(measurements.n_values),
            ),
            _optional_population_covariance(
                measurements.population_covariances,
                _CONTROL_EXCITED,
                kind,
                measurements.control_qubit,
                n_points=len(measurements.n_values),
            ),
            population_minimum_change=population_minimum_change,
            leakage_minimum_change=leakage_minimum_change,
            change_sigma_threshold=change_sigma_threshold,
            robust_loss=robust_loss,
        )
        for protocol in _CONTROL_STATE_PROTOCOLS:
            target_t1rho_fits[protocol][kind] = _fit_exponential_decay(
                measurements.cr_active_times[protocol],
                target_x[protocol][kind],
                measurements.target_x_standard_errors[protocol][kind],
                minimum_change=pauli_minimum_change,
                change_sigma_threshold=change_sigma_threshold,
                robust_loss=robust_loss,
            )
            target_population = measurements.populations[protocol][kind][
                measurements.target_qubit
            ]
            target_errors = measurements.population_standard_errors[protocol][kind][
                measurements.target_qubit
            ]
            target_leakage_fits[protocol][kind] = _fit_population_exchange(
                measurements.cr_active_times[protocol],
                target_population[:, 2],
                target_errors[:, 2],
                minimum_change=leakage_minimum_change,
                change_sigma_threshold=change_sigma_threshold,
                robust_loss=robust_loss,
            )

    transverse_echo_fits: dict[str, dict[str, ExponentialDecayFit]] = {
        _CONTROL_TRANSVERSE_ECHO: {},
        _TARGET_ROTATING_FRAME_ECHO: {},
    }
    for protocol, component in (
        (_CONTROL_TRANSVERSE_ECHO, "X"),
        (_TARGET_ROTATING_FRAME_ECHO, "Y"),
    ):
        for kind in ("actual", "reference"):
            transverse_echo_fits[protocol][kind] = _fit_exponential_decay(
                measurements.total_times[protocol],
                measurements.pauli_expectations[protocol][component][kind],
                measurements.pauli_standard_errors[protocol][component][kind],
                minimum_change=pauli_minimum_change,
                change_sigma_threshold=change_sigma_threshold,
                robust_loss=robust_loss,
            )

    actual_control = control_fits["actual"]
    reference_control = control_fits["reference"]
    target_t1rho_rate, target_t1rho_error = _mean_rate_and_error(
        [
            target_t1rho_fits[_CONTROL_GROUND]["actual"],
            target_t1rho_fits[_CONTROL_EXCITED]["actual"],
        ]
    )
    target_leak, target_leak_error = _mean_exchange_rate_and_error(
        [
            target_leakage_fits[_CONTROL_GROUND]["actual"],
            target_leakage_fits[_CONTROL_EXCITED]["actual"],
        ],
        "outward",
    )
    target_seep, target_seep_error = _mean_exchange_rate_and_error(
        [
            target_leakage_fits[_CONTROL_GROUND]["actual"],
            target_leakage_fits[_CONTROL_EXCITED]["actual"],
        ],
        "inward",
    )

    control_xy, control_xy_error, control_xy_difference = (
        _cr_active_rate_from_transverse_echo_fits(
            measurements,
            _CONTROL_TRANSVERSE_ECHO,
            transverse_echo_fits[_CONTROL_TRANSVERSE_ECHO]["actual"],
            transverse_echo_fits[_CONTROL_TRANSVERSE_ECHO]["reference"],
            idle_rate=(
                None
                if control_idle_noise is None
                else 1.0 / control_idle_noise.effective_t2_echo
            ),
        )
    )
    target_t2rho_rate, target_t2rho_error, target_t2rho_difference = (
        _cr_active_rate_from_transverse_echo_fits(
            measurements,
            _TARGET_ROTATING_FRAME_ECHO,
            transverse_echo_fits[_TARGET_ROTATING_FRAME_ECHO]["actual"],
            transverse_echo_fits[_TARGET_ROTATING_FRAME_ECHO]["reference"],
            idle_rate=(
                None
                if target_idle_noise is None
                else 1.0 / target_idle_noise.effective_t2_echo
            ),
        )
    )
    active_axis = np.asarray(
        measurements.cr_active_times[_CONTROL_GROUND], dtype=np.float64
    )
    total_axis = np.asarray(measurements.total_times[_CONTROL_GROUND], dtype=np.float64)
    positive_active = active_axis > 0.0
    blank_to_active = (
        float(
            np.median(
                (total_axis[positive_active] - active_axis[positive_active])
                / active_axis[positive_active]
            )
        )
        if np.any(positive_active)
        else 0.0
    )
    control_down = actual_control.rates["gamma_e_to_g"]
    target_t1rho_rate_corrected = target_t1rho_rate
    if control_idle_noise is not None:
        control_down = _subtract_known_rate(
            control_down,
            blank_to_active / control_idle_noise.t1,
        )
    if target_idle_noise is not None:
        target_t1rho_rate_corrected = _subtract_known_rate(
            target_t1rho_rate,
            blank_to_active / target_idle_noise.effective_t2_star,
        )
    control_population_transverse = 0.5 * (
        control_down + actual_control.rates["gamma_g_to_e"]
    )
    control_phi = _nonnegative_rate_difference(
        control_xy, control_population_transverse
    )
    target_phi_rho = _nonnegative_rate_difference(
        target_t2rho_rate, 0.5 * target_t1rho_rate_corrected
    )
    control_phi_error = float(
        np.hypot(
            control_xy_error,
            0.5
            * np.hypot(
                actual_control.rate_standard_errors["gamma_e_to_g"],
                actual_control.rate_standard_errors["gamma_g_to_e"],
            ),
        )
    )
    target_phi_rho_error = float(np.hypot(target_t2rho_error, 0.5 * target_t1rho_error))

    rate_names = (
        "gamma_control_e_to_g",
        "gamma_control_g_to_e",
        "gamma_control_f_to_e",
        "gamma_control_e_to_f",
        "gamma_target_t1rho",
        "gamma_target_leakage",
        "gamma_target_seepage",
        "gamma_control_transverse",
        "gamma_target_t2rho",
        "gamma_control_phi_cr",
        "gamma_target_phi_rho_cr",
    )
    cr_rates = {
        "gamma_control_e_to_g": control_down,
        "gamma_control_g_to_e": actual_control.rates["gamma_g_to_e"],
        "gamma_control_f_to_e": actual_control.rates["gamma_f_to_e"],
        "gamma_control_e_to_f": actual_control.rates["gamma_e_to_f"],
        "gamma_target_t1rho": target_t1rho_rate_corrected,
        "gamma_target_leakage": target_leak,
        "gamma_target_seepage": target_seep,
        "gamma_control_transverse": control_xy,
        "gamma_target_t2rho": target_t2rho_rate,
        "gamma_control_phi_cr": control_phi,
        "gamma_target_phi_rho_cr": target_phi_rho,
    }
    cr_errors = {
        "gamma_control_e_to_g": actual_control.rate_standard_errors["gamma_e_to_g"],
        "gamma_control_g_to_e": actual_control.rate_standard_errors["gamma_g_to_e"],
        "gamma_control_f_to_e": actual_control.rate_standard_errors["gamma_f_to_e"],
        "gamma_control_e_to_f": actual_control.rate_standard_errors["gamma_e_to_f"],
        "gamma_target_t1rho": target_t1rho_error,
        "gamma_target_leakage": target_leak_error,
        "gamma_target_seepage": target_seep_error,
        "gamma_control_transverse": control_xy_error,
        "gamma_target_t2rho": target_t2rho_error,
        "gamma_control_phi_cr": control_phi_error,
        "gamma_target_phi_rho_cr": target_phi_rho_error,
    }
    decay_times = {
        name: _rate_to_lifetime(cr_rates[name], cr_errors[name])[0]
        for name in rate_names
    }

    reference_target_t1rho_rate = _mean_rate_and_error(
        [
            target_t1rho_fits[_CONTROL_GROUND]["reference"],
            target_t1rho_fits[_CONTROL_EXCITED]["reference"],
        ]
    )[0]
    reference_target_leak = _mean_exchange_rate_and_error(
        [
            target_leakage_fits[_CONTROL_GROUND]["reference"],
            target_leakage_fits[_CONTROL_EXCITED]["reference"],
        ],
        "outward",
    )[0]
    reference_target_seep = _mean_exchange_rate_and_error(
        [
            target_leakage_fits[_CONTROL_GROUND]["reference"],
            target_leakage_fits[_CONTROL_EXCITED]["reference"],
        ],
        "inward",
    )[0]
    rate_differences_from_reference = {
        "gamma_control_e_to_g": _rate_difference(
            actual_control.rates["gamma_e_to_g"],
            reference_control.rates["gamma_e_to_g"],
        ),
        "gamma_control_g_to_e": _rate_difference(
            actual_control.rates["gamma_g_to_e"],
            reference_control.rates["gamma_g_to_e"],
        ),
        "gamma_control_f_to_e": _rate_difference(
            actual_control.rates["gamma_f_to_e"],
            reference_control.rates["gamma_f_to_e"],
        ),
        "gamma_control_e_to_f": _rate_difference(
            actual_control.rates["gamma_e_to_f"],
            reference_control.rates["gamma_e_to_f"],
        ),
        "gamma_target_t1rho": _rate_difference(
            target_t1rho_rate, reference_target_t1rho_rate
        ),
        "gamma_target_leakage": _rate_difference(target_leak, reference_target_leak),
        "gamma_target_seepage": _rate_difference(target_seep, reference_target_seep),
        "gamma_control_transverse": control_xy_difference,
        "gamma_target_t2rho": target_t2rho_difference,
        # No like-for-like pure-dephasing reference is identified by this
        # effective model.  Reporting the absolute rates as differences would
        # be dimensionally valid but semantically misleading.
        "gamma_control_phi_cr": float("nan"),
        "gamma_target_phi_rho_cr": float("nan"),
    }

    if estimate_fidelity:
        initial_fidelity_message = "Rough fidelity requested but ZX90 timing and idle T1/T2 inputs are incomplete."
    else:
        initial_fidelity_message = "Rough fidelity estimate disabled."
    fidelity = CrDissipationFidelityEstimate(
        available=False,
        message=initial_fidelity_message,
        estimated_fidelity=float("nan"),
        estimated_fidelity_standard_error=float("nan"),
        idle_coherence_limit=float("nan"),
        average_leakage=float("nan"),
        control_lambdas=_NAN_BLOCH_FACTORS,
        target_lambdas=_NAN_BLOCH_FACTORS,
        computational_survival=float("nan"),
        parameter_values={},
        parameter_standard_errors={},
        metadata={"diagnostic_only": True},
    )
    if estimate_fidelity and all(
        value is not None
        for value in (zx90_echo_timing, control_idle_noise, target_idle_noise)
    ):
        timing = cast(ZX90Timing, zx90_echo_timing)
        control_idle = cast(IdleNoiseParameters, control_idle_noise)
        target_idle = cast(IdleNoiseParameters, target_idle_noise)
        raw_parameter_values = {
            "control_xy_rate": control_xy,
            "control_z_rate": (control_down + actual_control.rates["gamma_g_to_e"]),
            "target_t1rho_rate": target_t1rho_rate_corrected,
            "target_t2rho_rate": target_t2rho_rate,
            "control_leakage_rate": actual_control.rates["gamma_e_to_f"],
            "target_leakage_rate": target_leak,
        }
        if not all(np.isfinite(value) for value in raw_parameter_values.values()):
            missing = ", ".join(
                name
                for name, value in raw_parameter_values.items()
                if not np.isfinite(value)
            )
            fidelity = CrDissipationFidelityEstimate(
                available=False,
                message=(
                    "Rough fidelity unavailable because required dissipation rates are "
                    f"unresolved: {missing}."
                ),
                estimated_fidelity=float("nan"),
                estimated_fidelity_standard_error=float("nan"),
                idle_coherence_limit=_idle_fidelity_limit(
                    timing.total_duration, control_idle, target_idle
                ),
                average_leakage=float("nan"),
                control_lambdas=_NAN_BLOCH_FACTORS,
                target_lambdas=_NAN_BLOCH_FACTORS,
                computational_survival=float("nan"),
                parameter_values=raw_parameter_values,
                parameter_standard_errors={},
                metadata={"diagnostic_only": True},
            )
        else:
            parameter_values = {
                name: max(float(value), 0.0)
                for name, value in raw_parameter_values.items()
            }
            cov = np.asarray(actual_control.covariance, dtype=np.float64)
            if cov.shape == (4, 4) and np.all(np.isfinite(cov[[0, 1]][:, [0, 1]])):
                grad = np.array([1.0, 1.0])
                control_z_var = float(grad @ cov[np.ix_([0, 1], [0, 1])] @ grad)
                control_z_error = float(np.sqrt(max(control_z_var, 0.0)))
            else:

                def local_control_rate_error(rate_name: str) -> float:
                    if rate_name not in actual_control.active_rates:
                        return float("nan")
                    error = actual_control.rate_standard_errors[rate_name]
                    return (
                        float(error)
                        if np.isfinite(error) and error >= 0
                        else float("nan")
                    )

                ge_errors = [
                    local_control_rate_error(rate_name)
                    for rate_name in ("gamma_e_to_g", "gamma_g_to_e")
                ]
                control_z_error = (
                    float(np.sqrt(np.sum(np.square(ge_errors))))
                    if all(np.isfinite(value) for value in ge_errors)
                    else float("nan")
                )
            parameter_errors = {
                "control_xy_rate": control_xy_error,
                "control_z_rate": control_z_error,
                "target_t1rho_rate": target_t1rho_error,
                "target_t2rho_rate": target_t2rho_error,
                "control_leakage_rate": actual_control.rate_standard_errors[
                    "gamma_e_to_f"
                ],
                "target_leakage_rate": target_leak_error,
            }

            def evaluator(parameters: Mapping[str, float]) -> float:
                return _rough_fidelity_from_parameters(
                    gate_total_duration=timing.total_duration,
                    gate_cr_duration=timing.cr_active_duration,
                    control_idle=control_idle,
                    target_idle=target_idle,
                    **parameters,
                )[0]

            value, survival, c_lambda, t_lambda = _rough_fidelity_from_parameters(
                gate_total_duration=timing.total_duration,
                gate_cr_duration=timing.cr_active_duration,
                control_idle=control_idle,
                target_idle=target_idle,
                **parameter_values,
            )
            control_survival = float(
                np.exp(
                    -0.5
                    * parameter_values["control_leakage_rate"]
                    * timing.cr_active_duration
                )
            )
            target_survival = float(
                np.exp(
                    -parameter_values["target_leakage_rate"] * timing.cr_active_duration
                )
            )
            survival_lower = max(0.0, control_survival + target_survival - 1.0)
            survival_upper = min(control_survival, target_survival)
            coherence_factor = value / survival if survival > _EPS else float("nan")
            fidelity_error = (
                _fidelity_standard_error(parameter_values, parameter_errors, evaluator)
                if estimate_fidelity_uncertainty
                else float("nan")
            )
            fidelity = CrDissipationFidelityEstimate(
                available=True,
                message=(
                    "Diagnostic local-noise fidelity estimate from measured effective rates; "
                    "not a strict gate-fidelity bound."
                ),
                estimated_fidelity=value,
                estimated_fidelity_standard_error=fidelity_error,
                idle_coherence_limit=_idle_fidelity_limit(
                    timing.total_duration, control_idle, target_idle
                ),
                average_leakage=1.0 - survival,
                control_lambdas=c_lambda,
                target_lambdas=t_lambda,
                computational_survival=survival,
                parameter_values=parameter_values,
                parameter_standard_errors=parameter_errors,
                metadata={
                    "diagnostic_only": True,
                    "model": "independent_local_bloch_contractions_plus_uniform_leakage",
                    "control_xy_from": "control_cr_transverse_echo actual decay with CR-off idle-T2 correction",
                    "target_transverse_from": "target_cr_rotating_frame_echo actual decay with CR-off idle-T2 correction",
                    "target_x_from": "mean A/B target-X decay with blank idle-T2-star correction",
                    "control_z_from": "A/B control g<->e rates with blank idle-T1 correction",
                    "stable_curves_are_nominal_zero": True,
                    "idle_decoherence_used_for_cr_off_correction": True,
                    "subthreshold_rate_uncertainty_unresolved": True,
                    "positive_measured_rates_are_not_floored_by_idle_rates": True,
                    "idle_t2_capped_at_2t1_for_fidelity": True,
                    "idle_noise_uncertainty_not_propagated": True,
                    "fit_covariance_cross_terms_ignored_except_control_ge_sum": True,
                    "target_y_decay_assumed_equal_to_target_z_decay": True,
                    "control_y_decay_assumed_equal_to_control_x_decay": True,
                    "cr_sign_dependence_not_resolved": True,
                    "target_seepage_ignored_over_one_gate": True,
                    "control_leakage_weighted_by_half_e_occupation": True,
                    "possible_leakage_decoherence_overlap": True,
                    "coherent_gate_errors_not_included": True,
                    "echo_rates_are_cr_active_equivalent_not_strict_cr_only": True,
                    "echo_reference_preserves_internal_zx90_echo_pi": True,
                    "control_computational_survival": control_survival,
                    "target_computational_survival": target_survival,
                    "computational_survival_lower": survival_lower,
                    "computational_survival_upper": survival_upper,
                    "leakage_correlation_fidelity_lower": coherence_factor
                    * survival_lower,
                    "leakage_correlation_fidelity_upper": coherence_factor
                    * survival_upper,
                    "survival_bounds_are_frechet_not_statistical_confidence": True,
                },
            )

    quality_flags: dict[str, str] = {
        "control_actual": actual_control.quality,
        "control_reference": reference_control.quality,
    }
    for protocol in _CONTROL_STATE_PROTOCOLS:
        for kind in ("actual", "reference"):
            quality_flags[f"target_x_{protocol}_{kind}"] = target_t1rho_fits[protocol][
                kind
            ].quality
            quality_flags[f"target_f_{protocol}_{kind}"] = target_leakage_fits[
                protocol
            ][kind].quality
    for protocol in _ECHO_PROTOCOLS:
        for kind in ("actual", "reference"):
            quality_flags[f"{protocol}_{kind}"] = transverse_echo_fits[protocol][
                kind
            ].quality
    return CrDissipationAnalysis(
        control_rate_fits=control_fits,
        target_t1rho_fits=target_t1rho_fits,
        target_leakage_fits=target_leakage_fits,
        transverse_echo_fits=transverse_echo_fits,
        cr_active_rates=cr_rates,
        cr_active_rate_standard_errors=cr_errors,
        decay_times=decay_times,
        rate_differences_from_reference=rate_differences_from_reference,
        fidelity=fidelity,
        quality_flags=quality_flags,
        metadata={
            "rate_unit": "1/ns",
            "time_unit": "ns",
            "analysis_goal": "effective_rate_decomposition",
            "robust_loss": robust_loss,
            "population_minimum_change": population_minimum_change,
            "leakage_minimum_change": leakage_minimum_change,
            "pauli_minimum_change": pauli_minimum_change,
            "change_sigma_threshold": change_sigma_threshold,
            "exchange_model_selection": "2*robust_cost+k*log(n_points)",
            "fit_covariance_scale": (
                "absolute_point_errors_when_available_else_residual_variance"
            ),
            "control_initial_populations_fixed_to_n0": True,
            "control_population_weighting": {
                "actual": actual_control.population_weighting,
                "reference": reference_control.population_weighting,
            },
            "diagnostic_components_used_in_fit": False,
            "population_blank_correction": "post_fit_effective_rate_subtraction",
            "segmentwise_population_propagator_used": False,
            "idle_noise_uncertainty_not_propagated": True,
            "control_blank_idle_e_to_g_subtracted_from_t1": control_idle_noise
            is not None,
            "target_blank_transverse_decay_subtracted_from_t2_star": target_idle_noise
            is not None,
            "unknown_idle_excitation_leakage_and_seepage_assumed_zero": True,
            "blank_to_cr_active_duration_ratio": blank_to_active,
            "echo_rate_interpretation": (
                "CR-active-equivalent with internal echoed-ZX90 pi slots "
                "preserved in references"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _error_bar(errors: NDArray[np.float64]) -> dict[str, object]:
    return {
        "type": "data",
        "array": errors,
        "visible": bool(np.any(np.isfinite(errors))),
    }


def _nice_leakage_upper(*arrays: NDArray[np.float64]) -> float:
    finite_parts = [array[np.isfinite(array)] for array in arrays if array.size]
    finite = np.concatenate(finite_parts) if finite_parts else np.array([0.0])
    maximum = max(float(np.max(finite)), 0.0)
    raw = max(_MIN_LEAKAGE_AXIS_UPPER, _LEAKAGE_AXIS_HEADROOM * maximum)
    if raw >= 1.0:
        return 1.0
    exponent = np.floor(np.log10(raw))
    base = 10.0**exponent
    scaled = raw / base
    for nice in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
        if scaled <= nice + 1e-12:
            return min(1.0, float(nice * base))
    return min(1.0, raw)


def _plot_control_population(
    measurements: CrDissipationMeasurements,
    analysis: CrDissipationAnalysis,
    protocol: _CONTROL_STATE_PROTOCOL,
) -> go.Figure:
    rows = 2 if measurements.diagnostic_components_measured else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    times_us = measurements.cr_active_times[protocol] * 1e-3
    for kind in ("reference", "actual"):
        fit = analysis.control_rate_fits[kind]
        opacity = 0.4 if kind == "reference" else 1.0
        for index, state in enumerate(_STATE_NAMES):
            fig.add_trace(
                go.Scatter(
                    x=times_us,
                    y=measurements.populations[protocol][kind][
                        measurements.control_qubit
                    ][:, index],
                    mode="markers",
                    marker={
                        "color": COLORS[index],
                        "symbol": "diamond-open" if kind == "reference" else "circle",
                    },
                    opacity=opacity,
                    error_y=_error_bar(
                        measurements.population_standard_errors[protocol][kind][
                            measurements.control_qubit
                        ][:, index]
                    ),
                    name=f"{kind} P{state}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=fit.curve_times * 1e-3,
                    y=fit.curve_populations[protocol][:, index],
                    mode="lines",
                    line={
                        "color": COLORS[index],
                        "dash": "dot" if kind == "reference" else "solid",
                    },
                    opacity=opacity,
                    name=f"{kind} fit P{state}",
                ),
                row=1,
                col=1,
            )
    if measurements.diagnostic_components_measured:
        key = _diagnostic_key(protocol, "control")
        for component in ("X", "Y"):
            color = COLORS[("X", "Y", "Z").index(component)]
            for kind in ("reference", "actual"):
                fig.add_trace(
                    go.Scatter(
                        x=times_us,
                        y=measurements.pauli_expectations[key][component][kind],
                        mode="markers",
                        marker={
                            "symbol": "diamond-open"
                            if kind == "reference"
                            else "circle",
                            "color": color,
                        },
                        opacity=0.4 if kind == "reference" else 1.0,
                        error_y=_error_bar(
                            measurements.pauli_standard_errors[key][component][kind]
                        ),
                        name=f"{kind} control {component}",
                    ),
                    row=2,
                    col=1,
                )
        fig.update_yaxes(title_text="Control X/Y", range=[-1.1, 1.1], row=2, col=1)
    fig.update_xaxes(title_text="CR-active time (µs)", row=rows, col=1)
    fig.update_layout(
        title=(
            f"{_PROTOCOL_LABELS[protocol]}: control "
            f"{measurements.control_qubit} GEF dissipation"
        ),
        xaxis_title="CR-active time (µs)",
        yaxis={"title": "Population", "range": [0.0, 1.0]},
        template="qubex",
    )
    return fig


def _plot_target_dissipation(
    measurements: CrDissipationMeasurements,
    analysis: CrDissipationAnalysis,
    protocol: _CONTROL_STATE_PROTOCOL,
) -> go.Figure:
    leakage_row = 3 if measurements.diagnostic_components_measured else 2
    fig = make_subplots(
        rows=leakage_row, cols=1, shared_xaxes=True, vertical_spacing=0.06
    )
    times_us = measurements.cr_active_times[protocol] * 1e-3
    target_x = measurements.target_x
    leakage_arrays: list[NDArray[np.float64]] = []
    for kind in ("reference", "actual"):
        opacity = 0.4 if kind == "reference" else 1.0
        symbol = "diamond-open" if kind == "reference" else "circle"
        color = COLORS[2]
        xfit = analysis.target_t1rho_fits[protocol][kind]
        ffit = analysis.target_leakage_fits[protocol][kind]
        fig.add_trace(
            go.Scatter(
                x=times_us,
                y=target_x[protocol][kind],
                mode="markers",
                marker={"symbol": symbol, "color": COLORS[2]},
                opacity=opacity,
                error_y=_error_bar(
                    measurements.target_x_standard_errors[protocol][kind]
                ),
                name=f"{kind} target X",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xfit.curve_times * 1e-3,
                y=xfit.curve_values,
                mode="lines",
                line={
                    "dash": "dot" if kind == "reference" else "solid",
                    "color": COLORS[2],
                },
                opacity=opacity,
                name=f"{kind} X fit",
            ),
            row=1,
            col=1,
        )
        pf = measurements.populations[protocol][kind][measurements.target_qubit][:, 2]
        pf_error = measurements.population_standard_errors[protocol][kind][
            measurements.target_qubit
        ][:, 2]
        leakage_arrays.extend(
            [
                pf,
                pf + np.where(np.isfinite(pf_error), pf_error, 0.0),
                ffit.curve_values,
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=times_us,
                y=pf,
                mode="markers",
                marker={"symbol": symbol, "color": color},
                opacity=opacity,
                error_y=_error_bar(pf_error),
                name=f"{kind} target Pf",
            ),
            row=leakage_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ffit.curve_times * 1e-3,
                y=ffit.curve_values,
                mode="lines",
                line={
                    "dash": "dot" if kind == "reference" else "solid",
                    "color": color,
                },
                opacity=opacity,
                name=f"{kind} Pf fit",
            ),
            row=leakage_row,
            col=1,
        )
    if measurements.diagnostic_components_measured:
        diagnostic_key = _diagnostic_key(protocol, "target")
        for component in ("Y", "Z"):
            color = COLORS[("X", "Y", "Z").index(component)]
            for kind in ("reference", "actual"):
                fig.add_trace(
                    go.Scatter(
                        x=times_us,
                        y=measurements.pauli_expectations[diagnostic_key][component][
                            kind
                        ],
                        mode="markers",
                        marker={
                            "symbol": "diamond-open"
                            if kind == "reference"
                            else "circle",
                            "color": color,
                        },
                        opacity=0.4 if kind == "reference" else 1.0,
                        error_y=_error_bar(
                            measurements.pauli_standard_errors[diagnostic_key][
                                component
                            ][kind]
                        ),
                        name=f"{kind} target {component}",
                    ),
                    row=2,
                    col=1,
                )
        fig.update_yaxes(title_text="Target Y/Z", range=[-1.1, 1.1], row=2, col=1)
    upper = _nice_leakage_upper(*leakage_arrays)
    fig.update_yaxes(title_text="Target X", range=[-1.1, 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Target Pf", range=[0.0, upper], row=leakage_row, col=1)
    fig.update_xaxes(title_text="CR-active time (µs)", row=leakage_row, col=1)
    fig.update_layout(
        title=f"{_PROTOCOL_LABELS[protocol]}: target dissipation", template="qubex"
    )
    return fig


def _plot_echo_dissipation(
    measurements: CrDissipationMeasurements,
    analysis: CrDissipationAnalysis,
    protocol: _ECHO_PROTOCOL,
) -> go.Figure:
    primary = "X" if protocol == _CONTROL_TRANSVERSE_ECHO else "Y"
    label = (
        measurements.control_qubit
        if protocol == _CONTROL_TRANSVERSE_ECHO
        else measurements.target_qubit
    )
    has_leakage = protocol in measurements.populations
    diagnostic_row = 2
    leakage_row = 2 + int(measurements.diagnostic_components_measured)
    n_rows = 1 + int(measurements.diagnostic_components_measured) + int(has_leakage)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    times_us = measurements.total_times[protocol] * 1e-3
    for kind in ("reference", "actual"):
        opacity = 0.4 if kind == "reference" else 1.0
        color = COLORS[0]
        fit = analysis.transverse_echo_fits[protocol][kind]
        fig.add_trace(
            go.Scatter(
                x=times_us,
                y=measurements.pauli_expectations[protocol][primary][kind],
                mode="markers",
                marker={
                    "symbol": "diamond-open" if kind == "reference" else "circle",
                    "color": color,
                },
                opacity=opacity,
                error_y=_error_bar(
                    measurements.pauli_standard_errors[protocol][primary][kind]
                ),
                name=f"{kind} {primary}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fit.curve_times * 1e-3,
                y=fit.curve_values,
                mode="lines",
                line={
                    "dash": "dot" if kind == "reference" else "solid",
                    "color": color,
                },
                opacity=opacity,
                name=f"{kind} {primary} fit",
            ),
            row=1,
            col=1,
        )

    if measurements.diagnostic_components_measured:
        for component in ("X", "Y", "Z"):
            if (
                component == primary
                or component not in measurements.pauli_expectations[protocol]
            ):
                continue
            component_color = COLORS[("X", "Y", "Z").index(component)]
            for kind in ("reference", "actual"):
                fig.add_trace(
                    go.Scatter(
                        x=times_us,
                        y=measurements.pauli_expectations[protocol][component][kind],
                        mode="markers",
                        marker={
                            "symbol": "circle" if kind == "actual" else "diamond-open",
                            "color": component_color,
                        },
                        opacity=0.55 if kind == "reference" else 0.75,
                        error_y=_error_bar(
                            measurements.pauli_standard_errors[protocol][component][
                                kind
                            ]
                        ),
                        name=f"{kind} {component} diagnostic",
                    ),
                    row=diagnostic_row,
                    col=1,
                )
    fig.update_yaxes(title_text=f"Primary {primary}", range=[-1.1, 1.1], row=1, col=1)
    if measurements.diagnostic_components_measured:
        fig.update_yaxes(
            title_text="Orthogonal components",
            range=[-1.1, 1.1],
            row=diagnostic_row,
            col=1,
        )
    if has_leakage:
        leakage_arrays: list[NDArray[np.float64]] = []
        observed_qubit = (
            measurements.control_qubit
            if protocol == _CONTROL_TRANSVERSE_ECHO
            else measurements.target_qubit
        )
        for kind in ("reference", "actual"):
            population = measurements.populations[protocol][kind][observed_qubit]
            errors = measurements.population_standard_errors[protocol][kind][
                observed_qubit
            ][:, 2]
            leakage_arrays.append(population[:, 2])
            fig.add_trace(
                go.Scatter(
                    x=times_us,
                    y=population[:, 2],
                    mode="markers",
                    marker={
                        "symbol": "diamond-open" if kind == "reference" else "circle",
                        "color": COLORS[2],
                    },
                    opacity=0.4 if kind == "reference" else 1.0,
                    error_y=_error_bar(errors),
                    name=f"{kind} {observed_qubit} Pf",
                ),
                row=leakage_row,
                col=1,
            )
        fig.update_yaxes(
            title_text=f"{observed_qubit} Pf",
            range=[0.0, _nice_leakage_upper(*leakage_arrays)],
            row=leakage_row,
            col=1,
        )
    fig.update_xaxes(title_text="Evolution time (µs)", row=n_rows, col=1)
    fig.update_layout(
        title=f"{_PROTOCOL_LABELS[protocol]}: {label} primary {primary} dissipation",
        template="qubex",
    )
    return fig


def plot_cr_dissipation(
    measurements: CrDissipationMeasurements,
    analysis: CrDissipationAnalysis,
) -> dict[str, go.Figure]:
    """
    Build the standard data-and-fit figures for a dissipation characterization result.

    Primary observables include fitted curves.  Optional orthogonal Pauli
    components are shown only as diagnostic points and never receive fit lines.

    Parameters
    ----------
    measurements
        Processed measurements used by `analyze_cr_dissipation`.
    analysis
        Analysis result corresponding to `measurements`.

    Returns
    -------
    dict[str, plotly.graph_objects.Figure]
        Named standard figures for the measured protocols.
    """
    figures: dict[str, go.Figure] = {
        f"{_CONTROL_GROUND}_control": _plot_control_population(
            measurements, analysis, _CONTROL_GROUND
        ),
        f"{_CONTROL_GROUND}_target": _plot_target_dissipation(
            measurements, analysis, _CONTROL_GROUND
        ),
        f"{_CONTROL_EXCITED}_control": _plot_control_population(
            measurements, analysis, _CONTROL_EXCITED
        ),
        f"{_CONTROL_EXCITED}_target": _plot_target_dissipation(
            measurements, analysis, _CONTROL_EXCITED
        ),
        _CONTROL_TRANSVERSE_ECHO: _plot_echo_dissipation(
            measurements, analysis, _CONTROL_TRANSVERSE_ECHO
        ),
        _TARGET_ROTATING_FRAME_ECHO: _plot_echo_dissipation(
            measurements, analysis, _TARGET_ROTATING_FRAME_ECHO
        ),
    }
    return figures


# ---------------------------------------------------------------------------
# Hardware acquisition
# ---------------------------------------------------------------------------


def _analytic_population_error(fit: GefPopulationFit) -> NDArray[np.float64]:
    """Return component-wise analytic population SEs, preserving usable entries."""
    error = np.asarray(fit.population_standard_error, dtype=np.float64)
    if error.shape != (3,):
        return np.full(3, np.nan, dtype=np.float64)
    valid = np.isfinite(error) & (error >= 0)
    return np.where(valid, error, np.nan).astype(np.float64)


def _analytic_population_covariance(fit: GefPopulationFit) -> NDArray[np.float64]:
    """
    Return the validated full analytic G/E/F population covariance.

    Invalid or unavailable full covariance is represented by an all-NaN sentinel
    rather than silently replacing it by a diagonal covariance.  This preserves
    provenance through acquisition so the control-rate fitter can accurately
    report whether it used full covariance whitening or a fallback weighting.
    Component-wise standard errors remain stored separately.
    """
    covariance = np.asarray(
        getattr(fit, "population_covariance", np.empty((0, 0))),
        dtype=np.float64,
    )
    if covariance.shape == (3, 3) and np.all(np.isfinite(covariance)):
        with suppress(ValueError, np.linalg.LinAlgError):
            return _validated_population_covariance(covariance)
    return np.full((3, 3), np.nan, dtype=np.float64)


def _analytic_computational_polarization_error(fit: GefPopulationFit) -> float:
    """Propagate full G/E covariance to a conditioned Pauli standard error."""
    population = np.asarray(fit.population, dtype=np.float64)
    if population.shape != (3,) or not np.all(np.isfinite(population[:2])):
        return float("nan")

    covariance = _analytic_population_covariance(fit)
    if not np.all(np.isfinite(covariance)):
        errors = _analytic_population_error(fit)
        covariance = np.diag(np.square(errors))
    covariance_ge = np.asarray(covariance[:2, :2], dtype=np.float64)
    if covariance_ge.shape != (2, 2) or not np.all(np.isfinite(covariance_ge)):
        return float("nan")
    covariance_ge = 0.5 * (covariance_ge + covariance_ge.T)
    eigenvalues = np.linalg.eigvalsh(covariance_ge)
    scale = max(float(np.max(np.abs(eigenvalues))), _EPS)
    negative_tolerance = max(1e-12 * scale, 1e-15 * scale, _EPS)
    if float(np.min(eigenvalues)) < -negative_tolerance:
        return float("nan")

    denominator = population[0] + population[1]
    if denominator <= _EPS:
        return float("nan")
    gradient = np.array(
        [
            2.0 * population[1] / denominator**2,
            -2.0 * population[0] / denominator**2,
        ],
        dtype=np.float64,
    )
    variance = float(gradient @ covariance_ge @ gradient)
    if not np.isfinite(variance):
        return float("nan")
    if variance < -negative_tolerance:
        return float("nan")
    return float(np.sqrt(max(variance, 0.0)))


def _print_dissipation_summary(analysis: CrDissipationAnalysis) -> None:
    """Print a compact human-readable dissipation summary."""
    print("CR pulse dissipation characterization:")
    rate_names = (
        "gamma_control_e_to_g",
        "gamma_control_g_to_e",
        "gamma_control_e_to_f",
        "gamma_control_f_to_e",
        "gamma_target_t1rho",
        "gamma_target_leakage",
        "gamma_target_seepage",
        "gamma_control_transverse",
        "gamma_target_t2rho",
        "gamma_control_phi_cr",
        "gamma_target_phi_rho_cr",
    )
    for name in rate_names:
        rate = analysis.cr_active_rates[name]
        error = analysis.cr_active_rate_standard_errors[name]
        lifetime = analysis.decay_times[name]
        if not np.isfinite(rate):
            print(f"  {name:27s}: unresolved")
            continue
        difference = analysis.rate_differences_from_reference.get(name, float("nan"))
        if rate <= 0:
            difference_text = (
                f", Δref ≈ {difference * 1e3:+.3g} /µs"
                if np.isfinite(difference) and difference != 0.0
                else ""
            )
            print(
                f"  {name:27s}: nominal 0 (no positive rate resolved{difference_text})"
            )
            continue
        rate_us = rate * 1e3
        error_us = error * 1e3 if np.isfinite(error) else float("nan")
        lifetime_us = lifetime * 1e-3
        difference_text = (
            f", Δref ≈ {difference * 1e3:+.3g} /µs" if np.isfinite(difference) else ""
        )
        if np.isfinite(error_us):
            print(
                f"  {name:27s}: {rate_us:.5g} ± {error_us:.2g} /µs  "
                f"(T ≈ {lifetime_us:.4g} µs{difference_text})"
            )
        else:
            print(
                f"  {name:27s}: {rate_us:.5g} /µs  "
                f"(T ≈ {lifetime_us:.4g} µs{difference_text})"
            )

    poor = [
        name
        for name, quality in analysis.quality_flags.items()
        if quality in {"poor", "failed"}
    ]
    if poor:
        print("  Fit-quality warning         : " + ", ".join(poor))

    fidelity = analysis.fidelity
    if fidelity.available:
        if np.isfinite(fidelity.estimated_fidelity_standard_error):
            print(
                "  Rough fidelity estimate     : "
                f"{fidelity.estimated_fidelity:.4%} ± "
                f"{fidelity.estimated_fidelity_standard_error:.3%}"
            )
        else:
            print(f"  Rough fidelity estimate     : {fidelity.estimated_fidelity:.4%}")
        print(f"  Idle coherence estimate     : {fidelity.idle_coherence_limit:.4%}")
        print(f"  Rough average leakage       : {fidelity.average_leakage:.4%}")
        print("  (Diagnostic estimate only; not a calibrated gate fidelity.)")
    else:
        print(f"  Rough fidelity estimate     : unavailable ({fidelity.message})")


def characterize_cr_dissipation(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    n_values: Sequence[int] | None = None,
    measure_diagnostic_components: bool = False,
    zx90_no_echo: PulseSchedule | None = None,
    zx90_echo: PulseSchedule | None = None,
    reference_ix45_amplitude: float | None = None,
    reference_calibration_valid_days: int | None = 30,
    force_reference_calibration: bool = False,
    reference_calibration_n_shots: int = DEFAULT_REFERENCE_CALIBRATION_N_SHOTS,
    reference_calibration_n_points: int = 21,
    reference_calibration_n_rotations: int = 2,
    reference_calibration_r2_threshold: float = 0.5,
    n_shots: int | None = DEFAULT_N_SHOTS,
    calibration_n_shots: int | None = DEFAULT_CALIBRATION_N_SHOTS,
    shot_interval: float | None = None,
    covariance_rcond: float = 1e-12,
    population_minimum_change: float = 0.01,
    leakage_minimum_change: float = 0.003,
    pauli_minimum_change: float = 0.05,
    change_sigma_threshold: float = 3.0,
    robust_loss: str = "soft_l1",
    leakage_warning_threshold: float | None = 0.01,
    estimate_fidelity: bool = True,
    estimate_fidelity_uncertainty: bool = True,
    idle_t1: Mapping[str, float] | None = None,
    idle_t2_echo: Mapping[str, float] | None = None,
    idle_t2_star: Mapping[str, float] | None = None,
    enable_tqdm: bool = True,
    plot: bool = True,
) -> Result:
    """
    Run a fast CR-pulse decoherence and leakage dissipation characterization.

    Parameters
    ----------
    exp
        Qubex experiment used for pulse construction and measurement.
    control_qubit, target_qubit
        Control and target qubit labels.
    n_values
        Strictly increasing nonnegative repetition indices beginning at zero.
        The default is `(0, 1, 2, 3, 5, 8, 13, 21, 34, 55)`.
    measure_diagnostic_components
        If `True`, additionally acquire control X/Y and target Y/Z for the
        two control-state protocols, plus the two orthogonal Pauli components
        of each echo protocol.  These points are plotted but never fitted.
    zx90_no_echo, zx90_echo
        Optional pulse-schedule overrides. `zx90_no_echo` must be a *full*
        un-echoed ZX90-equivalent schedule (not the single ZX45-like primitive
        returned by `exp.pulse.zx90(..., echo=False)`). Both overrides must expose
        Qubex `cr_duration` and `echo` metadata. For the full un-echoed override,
        `cr_duration` is the total CR-active duration of both lobes. Hardware-free
        overrides without waveform metadata use a duration-matched legacy reference.
    reference_ix45_amplitude
        Optional calibrated amplitude for one CR-shaped IX45. When omitted, a
        compatible cached calibration is loaded or the IX45 pair is calibrated
        automatically. This does not overwrite the ordinary hpi calibration.
    reference_calibration_valid_days, force_reference_calibration
        Cache lifetime and explicit recalibration control for CR-shaped IX45.
    reference_calibration_n_shots, reference_calibration_n_points
        Acquisition size of the IX45-pair amplitude calibration.
    reference_calibration_n_rotations
        Number of full rotations amplified during the pair calibration.
    reference_calibration_r2_threshold
        Minimum accepted coefficient of determination for that calibration.
    n_shots, calibration_n_shots
        Shots per dissipation measurement and per GEF calibration configuration.
    shot_interval
        Interval between shots in ns. `None` uses
        `measurement_defaults.yaml` and then the shared Qubex fallback.
    covariance_rcond
        Relative cutoff used by the analytic GEF covariance pseudo-inverse.
    population_minimum_change, leakage_minimum_change, pauli_minimum_change
        Absolute change thresholds used before a nominal zero rate is released
        and fitted.
    change_sigma_threshold
        Required robust-range-to-point-SE ratio for declaring a curve changing.
    robust_loss
        Robust least-squares loss used by changing-curve fits.
    leakage_warning_threshold
        Warn when either qubit's measured actual F population exceeds this
        value. `None` disables the warning and leaves the low-leakage assessment
        unspecified.
    estimate_fidelity
        Whether to calculate the rough diagnostic fidelity estimate.
    estimate_fidelity_uncertainty
        Whether to propagate local rate errors to that estimate.
    idle_t1, idle_t2_echo, idle_t2_star
        Optional qubit-to-lifetime mappings in ns for blank/CR-off correction
        and the rough fidelity estimate. Stored Qubex values are used when
        omitted.
    enable_tqdm
        Whether to display measurement progress.
    plot
        Whether to construct and show Plotly figures and print the human-readable
        summary. If `False`, `Result.figure` is `None` and `Result.figures` is empty.

    Returns
    -------
    Result
        `data["measurements"]` contains reusable processed measurements,
        `data["analysis"]` contains lightweight dissipation fits, and
        `data["raw_data"]` retains calibration and IQ information.

    Notes
    -----
    `control_ground_cr_population`
        Prepare `|0,+>` and apply ``C+ -> blank -> C+ -> blank`` `4n` times.
        Target -Y90 followed by GEF readout gives control G/E/F populations,
        target X polarization, and target F population. The reference replaces
        both CR lobes by positive CR-shaped IX45 pulses.
    `control_excited_cr_population`
        Same measurement with control prepared in `|1>`.
    `control_cr_transverse_echo`
        Four-ZX90 echoed control sequence.  Control X is the primary observable.
    `target_cr_rotating_frame_echo`
        Four-ZX90 target rotating-frame echo repeated `n` times. Target Y is
        the primary observable. The target virtual-Z frame update is applied
        to both the target and CR channels.

    GEF uncertainties use the analytic GLS full covariance when available.
    Component-wise analytic standard errors, and where necessary an empirical
    scale, provide fallback weighting.  All of these condition on the measured
    GEF calibration; calibration finite-shot uncertainty is not propagated.  No
    raw-shot bootstrap is run.  Stable observables are assigned nominal zero
    rates.  The optional fidelity is intentionally a rough local-noise
    diagnostic, not a calibrated gate fidelity or rigorous lower bound.
    """
    for name, value in (
        ("measure_diagnostic_components", measure_diagnostic_components),
        ("force_reference_calibration", force_reference_calibration),
        ("estimate_fidelity", estimate_fidelity),
        ("estimate_fidelity_uncertainty", estimate_fidelity_uncertainty),
        ("enable_tqdm", enable_tqdm),
        ("plot", plot),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")

    robust_loss = _validate_robust_loss(robust_loss)
    if reference_calibration_valid_days is not None and (
        isinstance(reference_calibration_valid_days, bool)
        or not isinstance(reference_calibration_valid_days, Integral)
        or reference_calibration_valid_days < 0
    ):
        raise ValueError(
            "reference_calibration_valid_days must be nonnegative or None."
        )
    for name, value, minimum in (
        ("reference_calibration_n_points", reference_calibration_n_points, 5),
        ("reference_calibration_n_rotations", reference_calibration_n_rotations, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    n_values_resolved = _validate_n_values(n_values)
    shots = _resolve_shot_count(n_shots, default=DEFAULT_N_SHOTS, name="n_shots")
    calibration_shots = _resolve_shot_count(
        calibration_n_shots,
        default=DEFAULT_CALIBRATION_N_SHOTS,
        name="calibration_n_shots",
    )
    configured_interval = resolve_measurement_defaults(
        exp.ctx.experiment_system.measurement_defaults
    ).execution.shot_interval_ns
    interval = _positive_real(
        shot_interval,
        default=configured_interval,
        name="shot_interval",
    )
    covariance_rcond = _nonnegative_real(covariance_rcond, name="covariance_rcond")
    if covariance_rcond >= 1:
        raise ValueError("covariance_rcond must be in [0, 1).")
    population_minimum_change = _nonnegative_real(
        population_minimum_change, name="population_minimum_change"
    )
    leakage_minimum_change = _nonnegative_real(
        leakage_minimum_change, name="leakage_minimum_change"
    )
    pauli_minimum_change = _nonnegative_real(
        pauli_minimum_change, name="pauli_minimum_change"
    )
    change_sigma_threshold = _positive_real(
        change_sigma_threshold,
        default=3.0,
        name="change_sigma_threshold",
    )
    leakage_warning = _optional_probability(
        leakage_warning_threshold,
        name="leakage_warning_threshold",
    )
    reference_r2_threshold = _nonnegative_real(
        reference_calibration_r2_threshold,
        name="reference_calibration_r2_threshold",
    )
    if reference_r2_threshold > 1.0:
        raise ValueError("reference_calibration_r2_threshold must be in [0, 1].")

    control = exp.ctx.resolve_qubit_label(control_qubit)
    target = exp.ctx.resolve_qubit_label(target_qubit)
    if control == target:
        raise ValueError("control_qubit and target_qubit must differ.")
    cr_label = f"{control}-{target}"
    labels = (control, cr_label, target)

    if zx90_echo is None:
        zx90_echo = exp.pulse.zx90(control, target, echo=True)
    echo_timing = _extract_zx90_timing(zx90_echo)
    if not echo_timing.echo:
        raise ValueError("zx90_echo must be echoed.")
    if zx90_no_echo is None:
        zx90_no_echo = _build_un_echoed_zx90(
            exp,
            control,
            target,
            blank_duration=_echo_pi_slot_duration(zx90_echo),
        )
    noecho_timing = _extract_zx90_timing(zx90_no_echo)
    if noecho_timing.echo:
        raise ValueError("zx90_no_echo must be un-echoed.")
    if not np.isclose(
        noecho_timing.cr_active_duration,
        echo_timing.cr_active_duration,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "zx90_no_echo and zx90_echo must contain the same CR-active duration "
            "per ZX90-equivalent schedule call."
        )

    cr_envelope = getattr(zx90_echo, "cr_waveform", None)
    ix45_calibration: CrShapedIx45Calibration | None = None
    ix45: FlatTop | None = None
    if isinstance(cr_envelope, FlatTop):
        ix45_calibration = _resolve_cr_shaped_ix45_calibration(
            exp,
            control,
            target,
            cr_envelope,
            amplitude=reference_ix45_amplitude,
            valid_days=reference_calibration_valid_days,
            force=force_reference_calibration,
            n_shots=_resolve_shot_count(
                reference_calibration_n_shots,
                default=DEFAULT_REFERENCE_CALIBRATION_N_SHOTS,
                name="reference_calibration_n_shots",
            ),
            shot_interval=interval,
            n_points=reference_calibration_n_points,
            n_rotations=reference_calibration_n_rotations,
            r2_threshold=reference_r2_threshold,
            plot=plot,
        )
        ix45 = _make_cr_shaped_ix45(cr_envelope, ix45_calibration.amplitude)
    elif reference_ix45_amplitude is not None:
        raise ValueError(
            "reference_ix45_amplitude requires zx90_echo.cr_waveform metadata."
        )
    else:
        warnings.warn(
            "zx90_echo does not expose a FlatTop `cr_waveform`; falling back to "
            "legacy duration-matched single-qubit references instead of the "
            "specified CR-shaped IX45 references.",
            RuntimeWarning,
            stacklevel=2,
        )

    calibration: dict[str, GefPopulationCalibration] = calibrate_gef_population(
        exp,
        targets=[control, target],
        n_shots=calibration_shots,
        shot_interval=interval,
    )

    all_protocols = (*_CONTROL_STATE_PROTOCOLS, *_ECHO_PROTOCOLS)
    populations_buffer: dict[str, dict[str, dict[str, list[NDArray[np.float64]]]]] = {
        protocol: {kind: {control: [], target: []} for kind in ("actual", "reference")}
        for protocol in all_protocols
    }
    population_error_buffer: dict[
        str, dict[str, dict[str, list[NDArray[np.float64]]]]
    ] = {
        protocol: {kind: {control: [], target: []} for kind in ("actual", "reference")}
        for protocol in all_protocols
    }
    population_covariance_buffer: dict[
        str, dict[str, dict[str, list[NDArray[np.float64]]]]
    ] = {
        protocol: {kind: {control: [], target: []} for kind in ("actual", "reference")}
        for protocol in all_protocols
    }
    target_x_error_buffer: dict[str, dict[str, list[float]]] = {
        protocol: {kind: [] for kind in ("actual", "reference")}
        for protocol in _CONTROL_STATE_PROTOCOLS
    }
    total_times_buffer = {protocol: [] for protocol in all_protocols}
    cr_times_buffer = {protocol: [] for protocol in all_protocols}

    pauli_components: dict[str, tuple[str, ...]] = {
        _CONTROL_TRANSVERSE_ECHO: ("X", "Y", "Z")
        if measure_diagnostic_components
        else ("X",),
        _TARGET_ROTATING_FRAME_ECHO: ("Y", "X", "Z")
        if measure_diagnostic_components
        else ("Y",),
    }
    if measure_diagnostic_components:
        pauli_components.update(
            {
                _diagnostic_key(_CONTROL_GROUND, "control"): ("X", "Y"),
                _diagnostic_key(_CONTROL_GROUND, "target"): ("Y", "Z"),
                _diagnostic_key(_CONTROL_EXCITED, "control"): ("X", "Y"),
                _diagnostic_key(_CONTROL_EXCITED, "target"): ("Y", "Z"),
            }
        )
    pauli_buffer: dict[str, dict[str, dict[str, list[_PauliMeasurement]]]] = {
        key: {
            component: {kind: [] for kind in ("actual", "reference")}
            for component in components
        }
        for key, components in pauli_components.items()
    }

    gef_raw_iq: dict[str, object] = {}
    gef_population_fits: dict[str, object] = {}
    gef_moment_summaries: dict[str, object] = {}
    pauli_raw_iq: dict[str, dict[str, dict[str, list[object]]]] = {
        key: {
            component: {kind: [] for kind in ("actual", "reference")}
            for component in components
        }
        for key, components in pauli_components.items()
    }

    # Build n-independent reference units and echo blocks once.  Repetition of
    # these validated units then preserves actual/reference duration matching.
    if ix45 is not None:
        reference_unit = _build_non_echoed_reference_unit(
            labels,
            target,
            ix45,
            _echo_pi_slot_duration(zx90_echo),
            frequencies=zx90_no_echo.get_frequencies(),
        )
        ground_reference_unit = reference_unit
        excited_reference_unit = reference_unit.copy()
    else:
        # Explicit schedule overrides used by simulations may not expose the
        # calibrated CrossResonance waveforms. Retain a duration-matched
        # fallback so offline/hardware-free callers remain usable.
        ground_reference_unit = _reference_unit(
            labels,
            zx90_no_echo.duration,
            pulse_target=target,
            pulse=exp.pulse.x90(target),
            frequencies=zx90_no_echo.get_frequencies(),
        )
        excited_reference_unit = _reference_unit(
            labels,
            zx90_no_echo.duration,
            pulse_target=target,
            pulse=exp.pulse.x90(target),
            frequencies=zx90_no_echo.get_frequencies(),
        )
    _require_matched_duration(zx90_no_echo, ground_reference_unit, name=_CONTROL_GROUND)
    _require_matched_duration(
        zx90_no_echo, excited_reference_unit, name=_CONTROL_EXCITED
    )

    if ix45 is not None:
        blank_echo_reference_unit = _build_echoed_reference_unit(
            zx90_echo, control, target, None
        )
        target_echo_reference_unit = _build_echoed_reference_unit(
            zx90_echo, control, target, ix45
        )
    else:
        blank_echo_reference_unit = _reference_unit(
            labels,
            zx90_echo.duration,
            frequencies=zx90_echo.get_frequencies(),
        )
        target_echo_reference_unit = _reference_unit(
            labels,
            zx90_echo.duration,
            pulse_target=target,
            pulse=exp.pulse.x90(target),
            frequencies=zx90_echo.get_frequencies(),
        )
    control_echo_actual_block = _control_cr_transverse_echo_block(
        exp, control, target, zx90_echo
    )
    control_echo_reference_block = _control_cr_transverse_echo_block(
        exp, control, target, blank_echo_reference_unit
    )
    target_echo_actual_block = _target_cr_rotating_frame_echo_block(
        exp, control, target, zx90_echo
    )
    target_echo_reference_block = _target_cr_rotating_frame_echo_block(
        exp, control, target, target_echo_reference_unit
    )
    _require_matched_duration(
        control_echo_actual_block,
        control_echo_reference_block,
        name=_CONTROL_TRANSVERSE_ECHO,
    )
    _require_matched_duration(
        target_echo_actual_block,
        target_echo_reference_block,
        name=_TARGET_ROTATING_FRAME_ECHO,
    )

    progress = tqdm(
        n_values_resolved,
        desc=f"CR dissipation {control}-{target}",
        disable=not enable_tqdm,
    )
    for n in progress:
        # Control-state actual and reference evolutions.
        control_state_evolutions = {
            (_CONTROL_GROUND, "actual"): zx90_no_echo.repeated(4 * n),
            (_CONTROL_GROUND, "reference"): ground_reference_unit.repeated(4 * n),
            (_CONTROL_EXCITED, "actual"): zx90_no_echo.repeated(4 * n),
            (_CONTROL_EXCITED, "reference"): excited_reference_unit.repeated(4 * n),
        }
        control_state_sequences = {
            (protocol, kind): _control_state_base_sequence(
                exp,
                control,
                target,
                cast(Literal["0", "1"], control_state),
                control_state_evolutions[(protocol, kind)],
            )
            for protocol, control_state in (
                (_CONTROL_GROUND, "0"),
                (_CONTROL_EXCITED, "1"),
            )
            for kind in ("actual", "reference")
        }

        gef_sequences = {
            f"{protocol}_{kind}": _control_state_gef_sequence(
                exp,
                control_state_sequences[(protocol, kind)],
                target,
            )
            for protocol in _CONTROL_STATE_PROTOCOLS
            for kind in ("actual", "reference")
        }
        gef_result = measure_gef_populations(
            exp,
            targets=[control, target],
            sequences=gef_sequences,
            calibration=calibration,
            n_shots=shots,
            shot_interval=interval,
            covariance_rcond=covariance_rcond,
            n_bootstrap=0,
        )
        for protocol in _CONTROL_STATE_PROTOCOLS:
            for kind in ("actual", "reference"):
                condition = f"{protocol}_{kind}"
                gef_raw_iq[f"n={n}/{condition}"] = gef_result.data["raw_iq"][condition]
                gef_population_fits[f"n={n}/{condition}"] = gef_result.data["fits"][
                    condition
                ]
                gef_moment_summaries[f"n={n}/{condition}"] = gef_result.data[
                    "moment_summaries"
                ][condition]
                for qubit in (control, target):
                    population = np.asarray(
                        gef_result.data["populations"][condition][qubit],
                        dtype=np.float64,
                    )
                    fit: GefPopulationFit = gef_result.data["fits"][condition][qubit]
                    populations_buffer[protocol][kind][qubit].append(population)
                    population_error_buffer[protocol][kind][qubit].append(
                        _analytic_population_error(fit)
                    )
                    population_covariance_buffer[protocol][kind][qubit].append(
                        _analytic_population_covariance(fit)
                    )
                    if qubit == target:
                        target_x_error_buffer[protocol][kind].append(
                            _analytic_computational_polarization_error(fit)
                        )

        for protocol in _CONTROL_STATE_PROTOCOLS:
            evolution = control_state_evolutions[(protocol, "actual")]
            total_times_buffer[protocol].append(float(evolution.duration))
            cr_times_buffer[protocol].append(4.0 * n * noecho_timing.cr_active_duration)

        # Echo-protocol actual and reference evolutions.
        control_echo_evolutions = {
            "actual": control_echo_actual_block.repeated(n),
            "reference": control_echo_reference_block.repeated(n),
        }
        target_echo_evolutions = {
            "actual": target_echo_actual_block.repeated(n),
            "reference": target_echo_reference_block.repeated(n),
        }
        echo_sequences = {
            (_CONTROL_TRANSVERSE_ECHO, kind): _echo_protocol_sequence(
                exp,
                control,
                target,
                _CONTROL_TRANSVERSE_ECHO,
                control_echo_evolutions[kind],
            )
            for kind in ("actual", "reference")
        }
        echo_sequences.update(
            {
                (_TARGET_ROTATING_FRAME_ECHO, kind): _echo_protocol_sequence(
                    exp,
                    control,
                    target,
                    _TARGET_ROTATING_FRAME_ECHO,
                    target_echo_evolutions[kind],
                )
                for kind in ("actual", "reference")
            }
        )
        total_times_buffer[_CONTROL_TRANSVERSE_ECHO].append(
            float(control_echo_evolutions["actual"].duration)
        )
        total_times_buffer[_TARGET_ROTATING_FRAME_ECHO].append(
            float(target_echo_evolutions["actual"].duration)
        )
        # Both echo protocols contain exactly 4n echoed ZX90 schedule calls.
        cr_time = 4.0 * n * echo_timing.cr_active_duration
        cr_times_buffer[_CONTROL_TRANSVERSE_ECHO].append(cr_time)
        cr_times_buffer[_TARGET_ROTATING_FRAME_ECHO].append(cr_time)

        requests: list[tuple[str, str, str, PulseSchedule, str, _BASIS]] = [
            (
                protocol,
                component,
                kind,
                echo_sequences[(protocol, kind)],
                target_qubit_label,
                cast(_BASIS, component),
            )
            for protocol, target_qubit_label in (
                (_CONTROL_TRANSVERSE_ECHO, control),
                (_TARGET_ROTATING_FRAME_ECHO, target),
            )
            for component in pauli_components[protocol]
            for kind in ("actual", "reference")
        ]
        if measure_diagnostic_components:
            requests.extend(
                (
                    _diagnostic_key(cast(_CONTROL_STATE_PROTOCOL, protocol), "control"),
                    component,
                    kind,
                    control_state_sequences[(protocol, kind)],
                    control,
                    cast(_BASIS, component),
                )
                for protocol in _CONTROL_STATE_PROTOCOLS
                for kind in ("actual", "reference")
                for component in ("X", "Y")
            )
            requests.extend(
                (
                    _diagnostic_key(cast(_CONTROL_STATE_PROTOCOL, protocol), "target"),
                    component,
                    kind,
                    control_state_sequences[(protocol, kind)],
                    target,
                    cast(_BASIS, component),
                )
                for protocol in _CONTROL_STATE_PROTOCOLS
                for kind in ("actual", "reference")
                for component in ("Y", "Z")
            )
        if requests:
            pauli_sequences = {
                f"pauli_{index}": _append_pauli_analyzer(exp, seq, qubit, basis)
                for index, (_, _, _, seq, qubit, basis) in enumerate(requests)
            }
            pauli_gef_result = measure_gef_populations(
                exp,
                targets=[control, target],
                sequences=pauli_sequences,
                calibration=calibration,
                n_shots=shots,
                shot_interval=interval,
                covariance_rcond=covariance_rcond,
                n_bootstrap=0,
            )
            for index, (key, component, kind, _, qubit, _) in enumerate(requests):
                condition = f"pauli_{index}"
                population = np.asarray(
                    pauli_gef_result.data["populations"][condition][qubit],
                    dtype=np.float64,
                )
                fit: GefPopulationFit = pauli_gef_result.data["fits"][condition][qubit]
                denominator = float(population[0] + population[1])
                expectation = (
                    float((population[0] - population[1]) / denominator)
                    if denominator > _EPS
                    else float("nan")
                )
                measurement = _PauliMeasurement(
                    expectation=expectation,
                    standard_error=_analytic_computational_polarization_error(fit),
                )
                pauli_buffer[key][component][kind].append(measurement)
                pauli_raw_iq[key][component][kind].append(
                    pauli_gef_result.data["raw_iq"][condition]
                )
                primary = (key == _CONTROL_TRANSVERSE_ECHO and component == "X") or (
                    key == _TARGET_ROTATING_FRAME_ECHO and component == "Y"
                )
                if primary:
                    for measured_qubit in (control, target):
                        marginal = np.asarray(
                            pauli_gef_result.data["populations"][condition][
                                measured_qubit
                            ],
                            dtype=np.float64,
                        )
                        marginal_fit: GefPopulationFit = pauli_gef_result.data["fits"][
                            condition
                        ][measured_qubit]
                        populations_buffer[key][kind][measured_qubit].append(marginal)
                        population_error_buffer[key][kind][measured_qubit].append(
                            _analytic_population_error(marginal_fit)
                        )
                        population_covariance_buffer[key][kind][measured_qubit].append(
                            _analytic_population_covariance(marginal_fit)
                        )

    populations: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    population_errors: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    population_covariances: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    target_x_errors: dict[str, dict[str, NDArray[np.float64]]] = {}
    for protocol in all_protocols:
        populations[protocol] = {}
        population_errors[protocol] = {}
        population_covariances[protocol] = {}
        if protocol in _CONTROL_STATE_PROTOCOLS:
            target_x_errors[protocol] = {}
        for kind in ("actual", "reference"):
            populations[protocol][kind] = {
                qubit: np.stack(populations_buffer[protocol][kind][qubit])
                for qubit in (control, target)
            }
            population_errors[protocol][kind] = {
                qubit: np.stack(population_error_buffer[protocol][kind][qubit])
                for qubit in (control, target)
            }
            population_covariances[protocol][kind] = {
                qubit: np.stack(population_covariance_buffer[protocol][kind][qubit])
                for qubit in (control, target)
            }
            if protocol in _CONTROL_STATE_PROTOCOLS:
                target_x_errors[protocol][kind] = np.asarray(
                    target_x_error_buffer[protocol][kind], dtype=np.float64
                )

    pauli_expectations: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    pauli_errors: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    for key, components in pauli_components.items():
        pauli_expectations[key] = {}
        pauli_errors[key] = {}
        for component in components:
            pauli_expectations[key][component] = {}
            pauli_errors[key][component] = {}
            for kind in ("actual", "reference"):
                values = pauli_buffer[key][component][kind]
                pauli_expectations[key][component][kind] = np.asarray(
                    [value.expectation for value in values], dtype=np.float64
                )
                pauli_errors[key][component][kind] = np.asarray(
                    [value.standard_error for value in values], dtype=np.float64
                )

    measurements = CrDissipationMeasurements(
        control_qubit=control,
        target_qubit=target,
        n_values=n_values_resolved,
        total_times={
            protocol: np.asarray(total_times_buffer[protocol], dtype=np.float64)
            for protocol in all_protocols
        },
        cr_active_times={
            protocol: np.asarray(cr_times_buffer[protocol], dtype=np.float64)
            for protocol in all_protocols
        },
        populations=populations,
        population_standard_errors=population_errors,
        target_x_standard_errors=target_x_errors,
        pauli_expectations=pauli_expectations,
        pauli_standard_errors=pauli_errors,
        diagnostic_components_measured=measure_diagnostic_components,
        population_covariances=population_covariances,
    )

    maximum_target_f = max(
        float(np.max(populations[protocol]["actual"][target][:, 2]))
        for protocol in all_protocols
    )
    maximum_control_f = max(
        float(np.max(populations[protocol]["actual"][control][:, 2]))
        for protocol in all_protocols
    )
    if leakage_warning is not None and maximum_target_f > leakage_warning:
        warnings.warn(
            f"Maximum measured target F population ({maximum_target_f:.3%}) exceeds "
            f"the dissipation threshold ({leakage_warning:.3%}).",
            RuntimeWarning,
            stacklevel=2,
        )
    if leakage_warning is not None and maximum_control_f > leakage_warning:
        warnings.warn(
            f"Maximum measured control F population ({maximum_control_f:.3%}) "
            f"exceeds the dissipation threshold ({leakage_warning:.3%}).",
            RuntimeWarning,
            stacklevel=2,
        )

    control_idle: IdleNoiseParameters | None = None
    target_idle: IdleNoiseParameters | None = None
    idle_input_error: str | None = None
    try:
        control_idle, target_idle = _load_idle_noise(
            exp, control, target, idle_t1, idle_t2_echo, idle_t2_star
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        idle_input_error = str(exc)
        warnings.warn(
            "Idle correction unavailable; reference decay will be used as the "
            f"CR-off fallback: {idle_input_error}",
            RuntimeWarning,
            stacklevel=2,
        )

    analysis = analyze_cr_dissipation(
        measurements,
        population_minimum_change=population_minimum_change,
        leakage_minimum_change=leakage_minimum_change,
        pauli_minimum_change=pauli_minimum_change,
        change_sigma_threshold=change_sigma_threshold,
        robust_loss=robust_loss,
        zx90_echo_timing=echo_timing if idle_input_error is None else None,
        control_idle_noise=control_idle,
        target_idle_noise=target_idle,
        estimate_fidelity=estimate_fidelity,
        estimate_fidelity_uncertainty=estimate_fidelity_uncertainty,
    )
    if estimate_fidelity and idle_input_error is not None:
        warnings.warn(
            "Rough fidelity estimate skipped: " + idle_input_error,
            RuntimeWarning,
            stacklevel=2,
        )

    figures: dict[str, go.Figure] = {}
    if plot:
        figures = plot_cr_dissipation(measurements, analysis)
        _print_dissipation_summary(analysis)
        for figure in figures.values():
            figure.show()

    return Result(
        data={
            "measurements": measurements,
            "analysis": analysis,
            "raw_data": {
                "calibration": calibration,
                "reference_ix45_calibration": ix45_calibration,
                "gef_raw_iq": gef_raw_iq,
                "gef_population_fits": gef_population_fits,
                "gef_moment_summaries": gef_moment_summaries,
                "pauli_raw_iq": pauli_raw_iq,
            },
            "measurement_options": {
                "n_shots": shots,
                "calibration_n_shots": calibration_shots,
                "shot_interval": interval,
                "covariance_rcond": covariance_rcond,
                "measure_diagnostic_components": measure_diagnostic_components,
                "reference_ix45_calibration_cache_key": _REFERENCE_CALIBRATION_NOTE_KEY,
                "population_uncertainty_method": "analytic_GLS_full_covariance_and_component_SE_no_bootstrap",
            },
            "analysis_options": {
                "population_minimum_change": population_minimum_change,
                "leakage_minimum_change": leakage_minimum_change,
                "pauli_minimum_change": pauli_minimum_change,
                "change_sigma_threshold": change_sigma_threshold,
                "robust_loss": robust_loss,
                "estimate_fidelity": estimate_fidelity,
                "estimate_fidelity_uncertainty": estimate_fidelity_uncertainty,
                "idle_input_error": idle_input_error,
                "fidelity_input_error": idle_input_error if estimate_fidelity else None,
                "leakage_warning_threshold": leakage_warning,
                "maximum_target_f_population": maximum_target_f,
                "maximum_control_f_population": maximum_control_f,
                "marginal_conditioning": True,
                "joint_gef_classification_required": False,
                "independent_leakage_assumption": True,
                "approximate_low_leakage": (
                    None
                    if leakage_warning is None
                    else max(maximum_control_f, maximum_target_f) <= leakage_warning
                ),
            },
            "pulse_timing": {
                "zx90_no_echo": noecho_timing,
                "zx90_echo": echo_timing,
            },
        },
        figure=next(iter(figures.values()), None),
        figures=figures,
    )


__all__ = [
    "DEFAULT_N_VALUES",
    "ChangeAssessment",
    "ControlPopulationRateFit",
    "CrDissipationAnalysis",
    "CrDissipationFidelityEstimate",
    "CrDissipationMeasurements",
    "CrShapedIx45Calibration",
    "ExponentialDecayFit",
    "IdleNoiseParameters",
    "PopulationExchangeFit",
    "ZX90Timing",
    "analyze_cr_dissipation",
    "characterize_cr_dissipation",
    "plot_cr_dissipation",
]
