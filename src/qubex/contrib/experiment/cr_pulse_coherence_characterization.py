"""Characterize relaxation, leakage, and coherence under repeated CR pulses."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from typing import Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.linalg import expm
from scipy.optimize import least_squares
from tqdm.auto import tqdm

from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, PulseSchedule, Waveform
from qubex.visualization import COLORS

from .gef_population_estimation import (
    GefPopulationBootstrap,
    GefPopulationCalibration,
    GefPopulationFit,
    IQMomentSummary,
    bootstrap_gef_populations,
    calibrate_gef_population,
    measure_gef_populations,
)

_Protocol = Literal[
    "control_ground",
    "control_excited",
    "control_t2_echo",
    "target_t2rho_echo",
]
_GefProtocol = Literal["control_ground", "control_excited"]
_PauliProtocol = Literal["control_t2_echo", "target_t2rho_echo"]
_PauliBasis = Literal["X", "Y", "Z"]

DEFAULT_N_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55)
_CONTROL_GROUND = "control_ground"
_CONTROL_EXCITED = "control_excited"
_CONTROL_T2_ECHO = "control_t2_echo"
_TARGET_T2RHO_ECHO = "target_t2rho_echo"
_PROTOCOLS: tuple[_Protocol, ...] = (
    _CONTROL_GROUND,
    _CONTROL_EXCITED,
    _CONTROL_T2_ECHO,
    _TARGET_T2RHO_ECHO,
)
_GEF_PROTOCOLS: tuple[_GefProtocol, ...] = (
    _CONTROL_GROUND,
    _CONTROL_EXCITED,
)
_PAULI_PROTOCOLS: tuple[_PauliProtocol, ...] = (
    _CONTROL_T2_ECHO,
    _TARGET_T2RHO_ECHO,
)
_PAULI_COMPONENT_ORDER: dict[_PauliProtocol, tuple[_PauliBasis, ...]] = {
    _CONTROL_T2_ECHO: ("X", "Y", "Z"),
    _TARGET_T2RHO_ECHO: ("Z", "X", "Y"),
}
_PAULI_MARKERS: dict[_PauliBasis, str] = {
    "X": "circle",
    "Y": "square",
    "Z": "triangle-up",
}
_PROTOCOL_LABELS: dict[_Protocol, str] = {
    _CONTROL_GROUND: "Control initialized in |g>",
    _CONTROL_EXCITED: "Control initialized in |e>",
    _CONTROL_T2_ECHO: "Control T2 echo",
    _TARGET_T2RHO_ECHO: "Target T2rho echo",
}
_STATE_NAMES = ("g", "e", "f")
_REFERENCE_OPACITY = 0.38
_RATE_PARAMETER_COUNT = 4
_EXPONENTIAL_PARAMETER_COUNT = 3


@dataclass(frozen=True)
class _PauliMeasurement:
    """Store one single-basis Pauli expectation measurement."""

    expectation: float
    standard_error: float
    normalized_shots: NDArray[np.float64]
    raw_iq: NDArray[np.complex128]


@dataclass(frozen=True)
class ExponentialDecayFit:
    """Store an offset exponential-decay fit and its diagnostics."""

    success: bool
    message: str
    amplitude: float
    offset: float
    tau: float
    amplitude_error: float
    offset_error: float
    tau_error: float
    covariance: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    r_squared: float


@dataclass(frozen=True)
class ThreeLevelRateFit:
    """Store a joint ground/excited adjacent-transition three-level rate fit."""

    success: bool
    message: str
    gamma_ge_down: float
    gamma_ge_up: float
    gamma_ef_down: float
    gamma_ef_up: float
    gamma_ge_down_error: float
    gamma_ge_up_error: float
    gamma_ef_down_error: float
    gamma_ef_up_error: float
    t1_eff: float
    t1_eff_error: float
    covariance: NDArray[np.float64]
    initial_ground: NDArray[np.float64]
    initial_excited: NDArray[np.float64]
    fitted_ground: NDArray[np.float64]
    fitted_excited: NDArray[np.float64]
    r_squared: float


@dataclass(frozen=True)
class _ProtocolSequences:
    """Store selected schedules and timing metadata for one n value."""

    sequences: dict[str, PulseSchedule]
    evolution_durations: dict[str, float]
    cr_pulse_count: int


def _validate_time_series(
    times: ArrayLike,
    values: ArrayLike,
    *,
    name: str,
    minimum_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a finite series sampled on an increasing zero-based time axis."""
    time_array = np.asarray(times, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if time_array.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if value_array.ndim == 0:
        raise ValueError(f"{name} must be at least one-dimensional.")
    if value_array.shape[0] != time_array.size:
        raise ValueError(f"{name} must have the same leading length as times.")
    if time_array.size < minimum_points:
        raise ValueError(f"times must contain at least {minimum_points} points.")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(value_array)):
        raise ValueError(f"times and {name} must contain only finite values.")
    if not np.isclose(time_array[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("times must start at zero.")
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("times must be strictly increasing.")
    return time_array, value_array


def _resolve_standard_errors(
    standard_errors: ArrayLike | None,
    shape: tuple[int, ...],
) -> NDArray[np.float64] | None:
    """Validate uncertainties and floor zero values for stable weighting."""
    if standard_errors is None:
        return None
    errors = np.asarray(standard_errors, dtype=np.float64)
    if errors.shape != shape:
        raise ValueError(f"standard_errors must have shape {shape}.")
    if np.any(np.isfinite(errors) & (errors < 0)):
        raise ValueError("standard_errors must be nonnegative where finite.")
    positive = errors[np.isfinite(errors) & (errors > 0)]
    if positive.size == 0:
        return None
    typical = float(np.median(positive))
    floor = max(typical * 1e-3, np.finfo(float).eps)
    return np.where(
        np.isfinite(errors) & (errors > 0),
        np.maximum(errors, floor),
        typical,
    )


def _estimate_covariance(
    jacobian: NDArray[np.float64],
    cost: float,
    residual_count: int,
    parameter_count: int,
    *,
    absolute_weights: bool,
) -> NDArray[np.float64]:
    """Estimate covariance using absolute weights or fitted residual variance."""
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    if absolute_weights:
        return covariance
    degrees_of_freedom = residual_count - parameter_count
    if degrees_of_freedom > 0:
        covariance *= 2 * cost / degrees_of_freedom
    else:
        covariance.fill(np.nan)
    return covariance


def _r_squared(observed: NDArray[np.float64], fitted: NDArray[np.float64]) -> float:
    """Return an R-squared value, including the constant-data edge case."""
    residual_sum = float(np.sum((observed - fitted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else float("nan")
    return 1 - residual_sum / total_sum


def fit_exponential_decay(
    times: ArrayLike,
    values: ArrayLike,
    standard_errors: ArrayLike | None = None,
) -> ExponentialDecayFit:
    """
    Fit `offset + amplitude * exp(-time / tau)` to a decay series.

    Parameters
    ----------
    times
        Strictly increasing times beginning at zero, in any consistent unit.
    values
        Finite measured values.
    standard_errors
        Optional nonnegative one-standard-error uncertainties used as absolute
        fit weights. Missing or zero entries use the median positive error when
        available; otherwise the fit is unweighted.

    Returns
    -------
    ExponentialDecayFit
        Fitted parameters. `tau` uses the same unit as `times`.
    """
    time_array, value_array = _validate_time_series(
        times,
        values,
        name="values",
        minimum_points=_EXPONENTIAL_PARAMETER_COUNT,
    )
    if value_array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    errors = _resolve_standard_errors(standard_errors, value_array.shape)
    weights = np.ones_like(value_array) if errors is None else 1 / errors
    time_scale = float(time_array[-1])
    scaled_times = time_array / time_scale

    offset_guess = float(value_array[-1])
    amplitude_guess = float(value_array[0] - offset_guess)
    if np.isclose(amplitude_guess, 0.0):
        amplitude_guess = float(np.ptp(value_array))

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        amplitude, offset, scaled_tau = parameters
        prediction = offset + amplitude * np.exp(-scaled_times / scaled_tau)
        return (prediction - value_array) * weights

    try:
        optimization = least_squares(
            residual,
            x0=np.array([amplitude_guess, offset_guess, 0.5]),
            bounds=(
                np.array([-np.inf, -np.inf, np.finfo(float).eps]),
                np.array([np.inf, np.inf, np.inf]),
            ),
            x_scale=1.0,
            max_nfev=20_000,
        )
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        return _failed_exponential_fit(str(exc), value_array.size)

    amplitude, offset, scaled_tau = optimization.x
    tau = float(scaled_tau * time_scale)
    fitted_values = offset + amplitude * np.exp(-time_array / tau)
    scaled_covariance = _estimate_covariance(
        np.asarray(optimization.jac, dtype=np.float64),
        float(optimization.cost),
        value_array.size,
        _EXPONENTIAL_PARAMETER_COUNT,
        absolute_weights=errors is not None,
    )
    transform = np.diag([1.0, 1.0, time_scale])
    covariance = transform @ scaled_covariance @ transform
    errors_by_parameter = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    return ExponentialDecayFit(
        success=bool(optimization.success),
        message=str(optimization.message),
        amplitude=float(amplitude),
        offset=float(offset),
        tau=tau,
        amplitude_error=float(errors_by_parameter[0]),
        offset_error=float(errors_by_parameter[1]),
        tau_error=float(errors_by_parameter[2]),
        covariance=covariance,
        fitted_values=np.asarray(fitted_values, dtype=np.float64),
        r_squared=_r_squared(value_array, fitted_values),
    )


def _failed_exponential_fit(message: str, n_values: int) -> ExponentialDecayFit:
    """Return a structured failed exponential fit."""
    return ExponentialDecayFit(
        success=False,
        message=message,
        amplitude=float("nan"),
        offset=float("nan"),
        tau=float("nan"),
        amplitude_error=float("nan"),
        offset_error=float("nan"),
        tau_error=float("nan"),
        covariance=np.full((_EXPONENTIAL_PARAMETER_COUNT,) * 2, np.nan),
        fitted_values=np.full(n_values, np.nan),
        r_squared=float("nan"),
    )


def _safe_fit_exponential_decay(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    standard_errors: NDArray[np.float64] | None,
) -> ExponentialDecayFit:
    """Fit a decay while preserving measured data when validation fails."""
    try:
        return fit_exponential_decay(times, values, standard_errors)
    except ValueError as exc:
        return _failed_exponential_fit(str(exc), values.size)


def _three_level_rate_matrix(
    rates: ArrayLike,
) -> NDArray[np.float64]:
    """Return the GEF rate matrix with no direct G-to-F transition."""
    gamma_ge_down, gamma_ge_up, gamma_ef_down, gamma_ef_up = np.asarray(
        rates,
        dtype=np.float64,
    )
    return np.array(
        [
            [-gamma_ge_up, gamma_ge_down, 0.0],
            [
                gamma_ge_up,
                -(gamma_ge_down + gamma_ef_up),
                gamma_ef_down,
            ],
            [0.0, gamma_ef_up, -gamma_ef_down],
        ],
        dtype=np.float64,
    )


def _population_trajectory(
    times: NDArray[np.float64],
    initial_population: NDArray[np.float64],
    rates: ArrayLike,
) -> NDArray[np.float64]:
    """Propagate a GEF population under the adjacent-transition rate model."""
    matrix = _three_level_rate_matrix(rates)
    return np.stack([expm(matrix * float(time)) @ initial_population for time in times])


def _validate_population_series(
    populations: NDArray[np.float64],
    *,
    name: str,
) -> None:
    """Validate physical GEF population rows."""
    if populations.ndim != 2 or populations.shape[1] != len(_STATE_NAMES):
        raise ValueError(f"{name} must have shape (n_times, 3).")
    tolerance = 1e-7
    if np.any(populations < -tolerance) or np.any(populations > 1 + tolerance):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    if not np.allclose(np.sum(populations, axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"Rows of {name} must sum to one.")


def fit_three_level_rate_model(
    times: ArrayLike,
    populations_ground: ArrayLike,
    populations_excited: ArrayLike,
    standard_errors_ground: ArrayLike | None = None,
    standard_errors_excited: ArrayLike | None = None,
) -> ThreeLevelRateFit:
    """
    Jointly fit ground/excited initial states to a four-rate adjacent GEF model.

    Parameters
    ----------
    times
        Strictly increasing times beginning at zero, in any consistent unit.
    populations_ground
        Populations initialized near g, ordered as g, e, f, with shape
        `(n_times, 3)`.
    populations_excited
        Populations initialized near e, ordered as g, e, f, with shape
        `(n_times, 3)`.
    standard_errors_ground
        Optional nonnegative one-standard-error uncertainties for
        `populations_ground`, used as absolute fit weights.
    standard_errors_excited
        Optional nonnegative one-standard-error uncertainties for
        `populations_excited`, used as absolute fit weights.

    Returns
    -------
    ThreeLevelRateFit
        Four directed rates and `T1_eff`. Rates are inverse `times` units.
    """
    time_array, population_ground = _validate_time_series(
        times,
        populations_ground,
        name="populations_ground",
        minimum_points=3,
    )
    _, population_excited = _validate_time_series(
        times,
        populations_excited,
        name="populations_excited",
        minimum_points=3,
    )
    _validate_population_series(population_ground, name="populations_ground")
    _validate_population_series(population_excited, name="populations_excited")

    errors_ground = _resolve_standard_errors(
        standard_errors_ground, population_ground.shape
    )
    errors_excited = _resolve_standard_errors(
        standard_errors_excited, population_excited.shape
    )
    weights_ground = (
        np.ones_like(population_ground) if errors_ground is None else 1 / errors_ground
    )
    weights_excited = (
        np.ones_like(population_excited)
        if errors_excited is None
        else 1 / errors_excited
    )
    initial_ground = np.clip(population_ground[0], 0.0, 1.0)
    initial_excited = np.clip(population_excited[0], 0.0, 1.0)
    initial_ground /= np.sum(initial_ground)
    initial_excited /= np.sum(initial_excited)
    time_scale = float(time_array[-1])
    scaled_times = time_array / time_scale

    def trajectories(
        scaled_rates: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            _population_trajectory(scaled_times, initial_ground, scaled_rates),
            _population_trajectory(scaled_times, initial_excited, scaled_rates),
        )

    def residual(scaled_rates: NDArray[np.float64]) -> NDArray[np.float64]:
        fitted_ground, fitted_excited = trajectories(scaled_rates)
        return np.concatenate(
            (
                ((fitted_ground - population_ground) * weights_ground).ravel(),
                ((fitted_excited - population_excited) * weights_excited).ravel(),
            )
        )

    def optimize(initial_scale: float):
        """Run one rate fit from a scalar initial-rate scale."""
        try:
            return least_squares(
                residual,
                x0=np.full(_RATE_PARAMETER_COUNT, initial_scale),
                bounds=(0.0, np.inf),
                x_scale=1.0,
                max_nfev=30_000,
            )
        except (FloatingPointError, RuntimeError, ValueError):
            return None

    optimizations = [
        optimization
        for initial_scale in (0.05, 0.2, 0.7, 2.0)
        if (optimization := optimize(initial_scale)) is not None
    ]
    if not optimizations:
        return _failed_rate_fit(
            "All least-squares attempts failed.",
            initial_ground,
            initial_excited,
            time_array.size,
        )

    optimization = min(optimizations, key=lambda candidate: candidate.cost)
    scaled_rates = np.asarray(optimization.x, dtype=np.float64)
    rates = scaled_rates / time_scale
    fitted_ground, fitted_excited = (
        _population_trajectory(time_array, initial_ground, rates),
        _population_trajectory(time_array, initial_excited, rates),
    )
    residual_count = population_ground.size + population_excited.size
    scaled_covariance = _estimate_covariance(
        np.asarray(optimization.jac, dtype=np.float64),
        float(optimization.cost),
        residual_count,
        _RATE_PARAMETER_COUNT,
        absolute_weights=errors_ground is not None or errors_excited is not None,
    )
    covariance = scaled_covariance / time_scale**2
    rate_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    ge_rate_sum = float(rates[0] + rates[1])
    t1_eff = 1 / ge_rate_sum if ge_rate_sum > 0 else float("inf")
    ge_rate_sum_variance = float(
        covariance[0, 0] + covariance[1, 1] + 2 * covariance[0, 1]
    )
    t1_eff_error = (
        np.sqrt(max(ge_rate_sum_variance, 0.0)) / ge_rate_sum**2
        if ge_rate_sum > 0
        else float("nan")
    )
    observed = np.concatenate((population_ground.ravel(), population_excited.ravel()))
    fitted = np.concatenate((fitted_ground.ravel(), fitted_excited.ravel()))
    return ThreeLevelRateFit(
        success=bool(optimization.success),
        message=str(optimization.message),
        gamma_ge_down=float(rates[0]),
        gamma_ge_up=float(rates[1]),
        gamma_ef_down=float(rates[2]),
        gamma_ef_up=float(rates[3]),
        gamma_ge_down_error=float(rate_errors[0]),
        gamma_ge_up_error=float(rate_errors[1]),
        gamma_ef_down_error=float(rate_errors[2]),
        gamma_ef_up_error=float(rate_errors[3]),
        t1_eff=float(t1_eff),
        t1_eff_error=float(t1_eff_error),
        covariance=covariance,
        initial_ground=initial_ground,
        initial_excited=initial_excited,
        fitted_ground=fitted_ground,
        fitted_excited=fitted_excited,
        r_squared=_r_squared(observed, fitted),
    )


def _failed_rate_fit(
    message: str,
    initial_ground: NDArray[np.float64],
    initial_excited: NDArray[np.float64],
    n_times: int,
) -> ThreeLevelRateFit:
    """Return a structured failed rate-model fit."""
    return ThreeLevelRateFit(
        success=False,
        message=message,
        gamma_ge_down=float("nan"),
        gamma_ge_up=float("nan"),
        gamma_ef_down=float("nan"),
        gamma_ef_up=float("nan"),
        gamma_ge_down_error=float("nan"),
        gamma_ge_up_error=float("nan"),
        gamma_ef_down_error=float("nan"),
        gamma_ef_up_error=float("nan"),
        t1_eff=float("nan"),
        t1_eff_error=float("nan"),
        covariance=np.full((_RATE_PARAMETER_COUNT,) * 2, np.nan),
        initial_ground=initial_ground,
        initial_excited=initial_excited,
        fitted_ground=np.full((n_times, len(_STATE_NAMES)), np.nan),
        fitted_excited=np.full((n_times, len(_STATE_NAMES)), np.nan),
        r_squared=float("nan"),
    )


def _reference_unit(
    labels: Sequence[str],
    duration: float,
    *,
    pulse_target: str | None = None,
    pulse: Waveform | None = None,
) -> PulseSchedule:
    """Build a pulse-plus-blank reference with an exact requested duration."""
    with PulseSchedule(list(labels)) as schedule:
        if pulse_target is not None and pulse is not None:
            schedule.add(pulse_target, pulse)
        else:
            schedule.add(labels[0], Blank(0))
    if schedule.duration > duration and not np.isclose(schedule.duration, duration):
        raise ValueError(
            "The reference single-qubit pulse is longer than its ZX90 schedule "
            f"({schedule.duration} ns > {duration} ns)."
        )
    return schedule.padded(duration, pad_side="right")


def _state_preparation(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    control_state: Literal["0", "1", "+"],
    target_state: Literal["0", "+"],
) -> PulseSchedule:
    """Prepare control and target states in parallel."""
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


def _gef_sequence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    control_state: Literal["0", "1"],
    evolution: PulseSchedule,
) -> PulseSchedule:
    """Build a GEF preparation, evolution, and IY90 sequence."""
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
        schedule.barrier()
        schedule.add(target_qubit, exp.pulse.y90(target_qubit))
    return schedule


def _control_t2_echo_block(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    zx90: PulseSchedule,
) -> PulseSchedule:
    """Build one four-ZX90 block for control T2 echo."""
    with PulseSchedule() as block:
        block.call(zx90, copy=True)
        block.barrier()
        block.add(control_qubit, exp.pulse.x180(control_qubit))
        block.barrier()

        block.call(zx90, copy=True)
        block.barrier()
        block.add(control_qubit, exp.pulse.x180(control_qubit))
        block.add(target_qubit, exp.pulse.x180(target_qubit))
        block.barrier()

        block.call(zx90, copy=True)
        block.barrier()
        block.add(control_qubit, exp.pulse.x180(control_qubit))
        block.barrier()

        block.call(zx90, copy=True)
    return block


def _target_t2rho_echo_block(
    exp: Experiment,
    target_qubit: str,
    zx90: PulseSchedule,
) -> PulseSchedule:
    """Build one two-ZX90 block for target T2rho echo."""
    with PulseSchedule() as block:
        block.call(zx90, copy=True)
        block.barrier()
        block.add(target_qubit, exp.pulse.z180())
        block.barrier()
        block.call(zx90, copy=True)
    return block


def _pauli_sequence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    protocol: _PauliProtocol,
    evolution: PulseSchedule,
) -> PulseSchedule:
    """Build the preparation and evolution for a Pauli-decay protocol."""
    initial_state: Literal["0", "+"] = "+" if protocol == _CONTROL_T2_ECHO else "0"
    preparation = _state_preparation(
        exp,
        control_qubit,
        target_qubit,
        initial_state,
        initial_state,
    )
    with PulseSchedule() as schedule:
        schedule.call(preparation, copy=True)
        schedule.call(evolution, copy=True)
    return schedule


def _build_protocol_sequences(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    n: int,
    zx90_no_echo: PulseSchedule | None,
    zx90_echo: PulseSchedule | None,
    protocols: Sequence[_Protocol] = _PROTOCOLS,
) -> _ProtocolSequences:
    """Build the actual and reference schedules for one sweep point."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a nonnegative integer.")
    n = int(n)
    selected = tuple(protocols)
    if not selected or any(protocol not in _PROTOCOLS for protocol in selected):
        valid_names = ", ".join(_PROTOCOLS)
        raise ValueError(f"protocols must contain only: {valid_names}.")
    if any(protocol in _GEF_PROTOCOLS for protocol in selected):
        if zx90_no_echo is None:
            raise ValueError(
                "zx90_no_echo is required for control_ground and control_excited."
            )
        no_echo_duration = zx90_no_echo.duration
    else:
        no_echo_duration = 0.0
    if any(protocol in _PAULI_PROTOCOLS for protocol in selected):
        if zx90_echo is None:
            raise ValueError(
                "zx90_echo is required for control_t2_echo and target_t2rho_echo."
            )
        echo_duration = zx90_echo.duration
    else:
        echo_duration = 0.0

    cr_label = f"{control_qubit}-{target_qubit}"
    labels = (control_qubit, cr_label, target_qubit)

    evolutions: dict[str, PulseSchedule] = {}
    sequences: dict[str, PulseSchedule] = {}
    if _CONTROL_GROUND in selected:
        no_echo_schedule = cast(PulseSchedule, zx90_no_echo)
        reference_unit = _reference_unit(
            labels,
            no_echo_duration,
            pulse_target=target_qubit,
            pulse=exp.pulse.x90(target_qubit),
        )
        evolutions[_CONTROL_GROUND] = no_echo_schedule.repeated(4 * n)
        evolutions[f"{_CONTROL_GROUND}_reference"] = reference_unit.repeated(4 * n)
        sequences[_CONTROL_GROUND] = _gef_sequence(
            exp, control_qubit, target_qubit, "0", evolutions[_CONTROL_GROUND]
        )
        sequences[f"{_CONTROL_GROUND}_reference"] = _gef_sequence(
            exp,
            control_qubit,
            target_qubit,
            "0",
            evolutions[f"{_CONTROL_GROUND}_reference"],
        )

    if _CONTROL_EXCITED in selected:
        no_echo_schedule = cast(PulseSchedule, zx90_no_echo)
        reference_unit = _reference_unit(
            labels,
            no_echo_duration,
            pulse_target=target_qubit,
            pulse=exp.pulse.x90m(target_qubit),
        )
        evolutions[_CONTROL_EXCITED] = no_echo_schedule.repeated(4 * n)
        evolutions[f"{_CONTROL_EXCITED}_reference"] = reference_unit.repeated(4 * n)
        sequences[_CONTROL_EXCITED] = _gef_sequence(
            exp, control_qubit, target_qubit, "1", evolutions[_CONTROL_EXCITED]
        )
        sequences[f"{_CONTROL_EXCITED}_reference"] = _gef_sequence(
            exp,
            control_qubit,
            target_qubit,
            "1",
            evolutions[f"{_CONTROL_EXCITED}_reference"],
        )

    if _CONTROL_T2_ECHO in selected:
        echo_schedule = cast(PulseSchedule, zx90_echo)
        reference_unit = _reference_unit(labels, echo_duration)
        evolutions[_CONTROL_T2_ECHO] = _control_t2_echo_block(
            exp, control_qubit, target_qubit, echo_schedule
        ).repeated(n)
        evolutions[f"{_CONTROL_T2_ECHO}_reference"] = _control_t2_echo_block(
            exp, control_qubit, target_qubit, reference_unit
        ).repeated(n)
        sequences[_CONTROL_T2_ECHO] = _pauli_sequence(
            exp,
            control_qubit,
            target_qubit,
            _CONTROL_T2_ECHO,
            evolutions[_CONTROL_T2_ECHO],
        )
        sequences[f"{_CONTROL_T2_ECHO}_reference"] = _pauli_sequence(
            exp,
            control_qubit,
            target_qubit,
            _CONTROL_T2_ECHO,
            evolutions[f"{_CONTROL_T2_ECHO}_reference"],
        )

    if _TARGET_T2RHO_ECHO in selected:
        echo_schedule = cast(PulseSchedule, zx90_echo)
        reference_unit = _reference_unit(
            labels,
            echo_duration,
            pulse_target=target_qubit,
            pulse=exp.pulse.x90(target_qubit),
        )
        evolutions[_TARGET_T2RHO_ECHO] = _target_t2rho_echo_block(
            exp, target_qubit, echo_schedule
        ).repeated(2 * n)
        evolutions[f"{_TARGET_T2RHO_ECHO}_reference"] = _target_t2rho_echo_block(
            exp, target_qubit, reference_unit
        ).repeated(2 * n)
        sequences[_TARGET_T2RHO_ECHO] = _pauli_sequence(
            exp,
            control_qubit,
            target_qubit,
            _TARGET_T2RHO_ECHO,
            evolutions[_TARGET_T2RHO_ECHO],
        )
        sequences[f"{_TARGET_T2RHO_ECHO}_reference"] = _pauli_sequence(
            exp,
            control_qubit,
            target_qubit,
            _TARGET_T2RHO_ECHO,
            evolutions[f"{_TARGET_T2RHO_ECHO}_reference"],
        )

    for protocol in selected:
        reference_name = f"{protocol}_reference"
        if not np.isclose(
            evolutions[reference_name].duration,
            evolutions[protocol].duration,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{protocol} reference and actual evolution durations differ."
            )
        if not np.isclose(
            sequences[reference_name].duration,
            sequences[protocol].duration,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{protocol} reference and actual schedules differ in length."
            )
    return _ProtocolSequences(
        sequences=sequences,
        evolution_durations={
            name: evolution.duration for name, evolution in evolutions.items()
        },
        cr_pulse_count=4 * n,
    )


def _measure_pauli_expectation(
    exp: Experiment,
    sequence: PulseSchedule,
    target: str,
    basis: _PauliBasis,
    *,
    n_shots: int,
    shot_interval: float,
) -> _PauliMeasurement:
    """Measure only one requested Pauli basis and estimate its shot error."""
    with PulseSchedule() as measurement_sequence:
        measurement_sequence.call(sequence, copy=True)
        analyzer: Waveform | None = None
        if basis == "X":
            analyzer = exp.pulse.y90m(target)
        elif basis == "Y":
            analyzer = exp.pulse.x90(target)
        if analyzer is not None:
            measurement_sequence.barrier()
            measurement_sequence.add(target, analyzer)
    measurement = exp.measurement_service.measure(
        sequence=measurement_sequence,
        mode="single",
        n_shots=n_shots,
        shot_interval=shot_interval,
        time_integration=True,
        state_classification=False,
        plot=False,
    )
    if target not in measurement.data:
        raise ValueError(f"Pauli measurement did not return `{target}`.")
    iq = np.asarray(measurement.data[target].kerneled, dtype=np.complex128)
    if iq.ndim == 0:
        iq = np.atleast_1d(iq)
    if iq.ndim != 1 or iq.size < 2 or not np.all(np.isfinite(iq)):
        raise ValueError("Pauli measurement must return at least two finite IQ shots.")
    rabi_param = exp.pulse.rabi_params.get(target)
    if rabi_param is None:
        raise ValueError(f"Rabi parameters for {target} are not stored.")
    normalized_shots = np.asarray(rabi_param.normalize(iq), dtype=np.float64)
    if normalized_shots.shape != iq.shape or not np.all(np.isfinite(normalized_shots)):
        raise ValueError(
            "Normalized Pauli shots must be finite and match raw IQ shape."
        )
    expectation = float(np.mean(normalized_shots))
    standard_error = float(np.std(normalized_shots, ddof=1) / np.sqrt(iq.size))
    return _PauliMeasurement(
        expectation=expectation,
        standard_error=standard_error,
        normalized_shots=normalized_shots,
        raw_iq=iq,
    )


def _validate_n_values(n_values: Sequence[int] | None) -> tuple[int, ...]:
    """Return a validated increasing nonnegative n sweep beginning at zero."""
    values = DEFAULT_N_VALUES if n_values is None else tuple(n_values)
    if len(values) < 3:
        raise ValueError("n_values must contain at least three values.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError("n_values must contain only integers.")
    normalized = tuple(int(value) for value in values)
    if normalized[0] != 0:
        raise ValueError("n_values must start at zero.")
    if any(value < 0 for value in normalized):
        raise ValueError("n_values must be nonnegative.")
    if any(right <= left for left, right in pairwise(normalized)):
        raise ValueError("n_values must be unique and strictly increasing.")
    return normalized


def _validate_protocols(
    protocols: Collection[str] | str | None,
) -> tuple[_Protocol, ...]:
    """Return selected protocols in canonical measurement order."""
    if protocols is None:
        return _PROTOCOLS
    requested = (protocols,) if isinstance(protocols, str) else tuple(protocols)
    if not requested:
        raise ValueError("protocols must contain at least one protocol.")
    if len(set(requested)) != len(requested):
        raise ValueError("protocols must not contain duplicates.")
    invalid = [protocol for protocol in requested if protocol not in _PROTOCOLS]
    if invalid:
        valid_names = ", ".join(_PROTOCOLS)
        raise ValueError(f"protocols must contain only: {valid_names}.")
    if (_CONTROL_GROUND in requested) != (_CONTROL_EXCITED in requested):
        raise ValueError(
            "control_ground and control_excited must be selected together for "
            "their joint rate fit."
        )
    return tuple(protocol for protocol in _PROTOCOLS if protocol in requested)


def _resolve_shot_count(value: int | None, *, default: int, name: str) -> int:
    """Resolve an optional shot count of at least two."""
    resolved = default if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, Integral):
        raise TypeError(f"{name} must be an integer of at least two.")
    if resolved < 2:
        raise ValueError(f"{name} must be an integer of at least two.")
    return int(resolved)


def _nonnegative_integer(value: int, *, name: str) -> int:
    """Validate and return a nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _positive_real(value: float | None, *, default: float, name: str) -> float:
    """Resolve an optional positive finite real value."""
    resolved = default if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, Real):
        raise TypeError(f"{name} must be a positive finite real number.")
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be a positive finite real value.")
    return float(resolved)


def _unit_interval_real(
    value: float,
    *,
    name: str,
    include_zero: bool,
) -> float:
    """Validate a finite real in either `(0, 1)` or `[0, 1)`."""
    interval = "in [0, 1)" if include_zero else "strictly between zero and one"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number {interval}.")
    resolved = float(value)
    valid = 0.0 <= resolved < 1.0 if include_zero else 0.0 < resolved < 1.0
    if not np.isfinite(resolved) or not valid:
        raise ValueError(f"{name} must be {interval}.")
    return resolved


def _population_standard_error(
    bootstrap: GefPopulationBootstrap,
) -> NDArray[np.float64]:
    """Return bootstrap population errors or NaNs when unavailable."""
    if bootstrap.unavailable_reason is not None or bootstrap.standard_error is None:
        return np.full(len(_STATE_NAMES), np.nan)
    return np.asarray(bootstrap.standard_error, dtype=np.float64)


def _polarization(
    population: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute `(P_e - P_g) / (P_e + P_g)` row-wise."""
    denominator = population[..., 1] + population[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > np.finfo(float).eps,
            (population[..., 1] - population[..., 0]) / denominator,
            np.nan,
        )


def _polarization_standard_error(
    bootstrap: GefPopulationBootstrap,
) -> float:
    """Propagate a GEF bootstrap distribution through the polarization ratio."""
    if bootstrap.unavailable_reason is not None:
        return float("nan")
    samples = np.asarray(bootstrap.samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != len(_STATE_NAMES):
        return float("nan")
    values = _polarization(samples)
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")


def _finite_errors_or_none(errors: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Return errors only when at least one positive finite value is available."""
    return errors if np.any(np.isfinite(errors) & (errors > 0)) else None


def _condition_name(protocol: str, reference: bool) -> str:
    """Return an internal actual/reference condition name."""
    return f"{protocol}_reference" if reference else protocol


def _pauli_components(
    protocol: _PauliProtocol,
    include_orthogonal: bool,
) -> tuple[_PauliBasis, ...]:
    """Return primary-first Pauli components for a Pauli-decay protocol."""
    components = _PAULI_COMPONENT_ORDER[protocol]
    return components if include_orthogonal else components[:1]


def _bootstrap_name(n: int, condition: str) -> str:
    """Return a globally unique bootstrap sequence name."""
    return f"n={n}/{condition}"


def _collect_gef_arrays(
    point_populations: Mapping[str, Mapping[str, list[NDArray[np.float64]]]],
    point_errors: Mapping[str, Mapping[str, list[NDArray[np.float64]]]],
    control_qubit: str,
    target_qubit: str,
    protocols: Sequence[_GefProtocol],
) -> tuple[
    dict[str, dict[str, dict[str, NDArray[np.float64]]]],
    dict[str, dict[str, dict[str, NDArray[np.float64]]]],
]:
    """Convert list-backed population buffers into result arrays."""
    populations: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    errors: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    for protocol in protocols:
        populations[protocol] = {}
        errors[protocol] = {}
        for kind in ("actual", "reference"):
            condition = _condition_name(protocol, kind == "reference")
            populations[protocol][kind] = {
                control_qubit: np.stack(point_populations[condition][control_qubit]),
                target_qubit: np.stack(point_populations[condition][target_qubit]),
            }
            errors[protocol][kind] = {
                control_qubit: np.stack(point_errors[condition][control_qubit]),
                target_qubit: np.stack(point_errors[condition][target_qubit]),
            }
    return populations, errors


def _fit_all_results(
    times: Mapping[str, NDArray[np.float64]],
    populations: Mapping[str, Mapping[str, Mapping[str, NDArray[np.float64]]]],
    population_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ],
    target_polarizations: Mapping[str, Mapping[str, NDArray[np.float64]]],
    target_polarization_errors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    pauli_expectations: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ],
    pauli_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ],
    control_qubit: str,
    protocols: Sequence[_Protocol],
) -> dict[str, object]:
    """Fit actual and reference data for every selected protocol."""
    fits: dict[str, object] = {}
    if _CONTROL_GROUND in protocols:
        fits["control_rate_model"] = {
            kind: fit_three_level_rate_model(
                times[_CONTROL_GROUND],
                populations[_CONTROL_GROUND][kind][control_qubit],
                populations[_CONTROL_EXCITED][kind][control_qubit],
                _finite_errors_or_none(
                    population_errors[_CONTROL_GROUND][kind][control_qubit]
                ),
                _finite_errors_or_none(
                    population_errors[_CONTROL_EXCITED][kind][control_qubit]
                ),
            )
            for kind in ("actual", "reference")
        }
        fits["target_t1rho"] = {
            protocol: {
                kind: _safe_fit_exponential_decay(
                    times[protocol],
                    target_polarizations[protocol][kind],
                    _finite_errors_or_none(target_polarization_errors[protocol][kind]),
                )
                for kind in ("actual", "reference")
            }
            for protocol in _GEF_PROTOCOLS
        }

    for protocol in _PAULI_PROTOCOLS:
        if protocol not in protocols:
            continue
        basis = _PAULI_COMPONENT_ORDER[protocol][0]
        fits[protocol] = {
            kind: _safe_fit_exponential_decay(
                times[protocol],
                pauli_expectations[protocol][basis][kind],
                _finite_errors_or_none(pauli_errors[protocol][basis][kind]),
            )
            for kind in ("actual", "reference")
        }
    return fits


def _add_top_cr_axis(
    figure: go.Figure,
    times_us: NDArray[np.float64],
    cr_counts: Sequence[int],
    *,
    axis_name: str = "xaxis2",
    overlaying: str = "x",
) -> None:
    """Add a fixed top axis labelled by ZX90 schedule count."""
    maximum = max(float(times_us[-1]), np.finfo(float).eps)
    figure.update_layout(
        xaxis={"title": "Evolution time (µs)", "range": [0.0, maximum]},
    )
    setattr(
        figure.layout,
        axis_name,
        go.layout.XAxis(
            title="ZX90 schedule count",
            overlaying=overlaying,
            side="top",
            range=[0.0, maximum],
            tickmode="array",
            tickvals=times_us,
            ticktext=[str(value) for value in cr_counts],
            showgrid=False,
        ),
    )


def _error_array(errors: NDArray[np.float64]) -> dict[str, object]:
    """Build a Plotly error-array configuration."""
    finite = np.isfinite(errors)
    return {
        "type": "data",
        "array": errors,
        "visible": bool(np.any(finite)),
    }


def _rate_curve(
    fit: ThreeLevelRateFit,
    protocol: _GefProtocol,
    dense_times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a fitted rate model for one initial condition."""
    if not fit.success:
        return np.full((dense_times.size, len(_STATE_NAMES)), np.nan)
    initial = fit.initial_ground if protocol == _CONTROL_GROUND else fit.initial_excited
    rates = (
        fit.gamma_ge_down,
        fit.gamma_ge_up,
        fit.gamma_ef_down,
        fit.gamma_ef_up,
    )
    return _population_trajectory(dense_times, initial, rates)


def _make_control_figure(
    protocol: _GefProtocol,
    times: NDArray[np.float64],
    cr_counts: Sequence[int],
    populations: Mapping[str, Mapping[str, NDArray[np.float64]]],
    errors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    control_qubit: str,
    rate_fits: Mapping[str, ThreeLevelRateFit],
) -> go.Figure:
    """Plot control GEF populations and the joint rate-model curves."""
    figure = go.Figure()
    times_us = times * 1e-3
    dense_times = np.linspace(0.0, float(times[-1]), 500)
    dense_times_us = dense_times * 1e-3
    for kind in ("reference", "actual"):
        reference = kind == "reference"
        fit_curve = _rate_curve(rate_fits[kind], protocol, dense_times)
        for state_index, state in enumerate(_STATE_NAMES):
            color = COLORS[state_index]
            figure.add_trace(
                go.Scatter(
                    x=times_us,
                    y=populations[kind][control_qubit][:, state_index],
                    mode="markers",
                    marker={
                        "color": color,
                        "symbol": "diamond-open" if reference else "circle",
                    },
                    opacity=_REFERENCE_OPACITY if reference else 1.0,
                    error_y=_error_array(errors[kind][control_qubit][:, state_index]),
                    name=f"{kind} P{state}",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=dense_times_us,
                    y=fit_curve[:, state_index],
                    mode="lines",
                    line={"color": color, "dash": "dot" if reference else "solid"},
                    opacity=_REFERENCE_OPACITY if reference else 1.0,
                    name=f"{kind} fit P{state}",
                )
            )
    figure.update_layout(
        title=f"{_PROTOCOL_LABELS[protocol]}: control {control_qubit} GEF populations",
        yaxis={"title": "Population", "range": [0.0, 1.0]},
    )
    _add_top_cr_axis(figure, times_us, cr_counts)
    return figure


def _exponential_curve(
    fit: ExponentialDecayFit,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate an exponential fit on a requested grid."""
    if not fit.success or not np.isfinite(fit.tau) or fit.tau <= 0:
        return np.full(times.shape, np.nan)
    return fit.offset + fit.amplitude * np.exp(-times / fit.tau)


def _make_target_figure(
    protocol: _GefProtocol,
    times: NDArray[np.float64],
    cr_counts: Sequence[int],
    populations: Mapping[str, Mapping[str, NDArray[np.float64]]],
    population_errors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    target_qubit: str,
    polarizations: Mapping[str, NDArray[np.float64]],
    polarization_errors: Mapping[str, NDArray[np.float64]],
    fits: Mapping[str, ExponentialDecayFit],
) -> go.Figure:
    """Plot target polarization decay and measured F-state leakage."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("GE-normalized polarization", "F-state leakage"),
    )
    times_us = times * 1e-3
    dense_times = np.linspace(0.0, float(times[-1]), 500)
    for kind in ("reference", "actual"):
        reference = kind == "reference"
        opacity = _REFERENCE_OPACITY if reference else 1.0
        marker_symbol = "diamond-open" if reference else "circle"
        line_dash = "dot" if reference else "solid"
        figure.add_trace(
            go.Scatter(
                x=times_us,
                y=polarizations[kind],
                mode="markers",
                marker={"color": COLORS[0], "symbol": marker_symbol},
                opacity=opacity,
                error_y=_error_array(polarization_errors[kind]),
                name=f"{kind} polarization",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=dense_times * 1e-3,
                y=_exponential_curve(fits[kind], dense_times),
                mode="lines",
                line={"color": COLORS[0], "dash": line_dash},
                opacity=opacity,
                name=f"{kind} fit",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=times_us,
                y=populations[kind][target_qubit][:, 2],
                mode="markers",
                marker={"color": COLORS[2], "symbol": marker_symbol},
                opacity=opacity,
                error_y=_error_array(population_errors[kind][target_qubit][:, 2]),
                name=f"{kind} Pf",
            ),
            row=2,
            col=1,
        )
    figure.update_layout(title=f"{_PROTOCOL_LABELS[protocol]}: target {target_qubit}")
    figure.update_yaxes(title_text="(Pe-Pg)/(Pe+Pg)", range=[-1.05, 1.05], row=1, col=1)
    figure.update_yaxes(title_text="Pf", range=[0.0, 1.0], row=2, col=1)
    maximum = max(float(times_us[-1]), np.finfo(float).eps)
    figure.update_xaxes(range=[0.0, maximum], row=1, col=1)
    figure.update_xaxes(
        title_text="Evolution time (µs)",
        range=[0.0, maximum],
        row=2,
        col=1,
    )
    figure.update_layout(
        xaxis3={
            "title": "ZX90 schedule count",
            "overlaying": "x",
            "side": "top",
            "range": [0.0, maximum],
            "tickmode": "array",
            "tickvals": times_us,
            "ticktext": [str(value) for value in cr_counts],
            "showgrid": False,
        }
    )
    return figure


def _make_pauli_figure(
    protocol: _PauliProtocol,
    times: NDArray[np.float64],
    cr_counts: Sequence[int],
    expectations: Mapping[str, Mapping[str, NDArray[np.float64]]],
    errors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    fits: Mapping[str, ExponentialDecayFit],
    target: str,
    primary_basis: _PauliBasis,
) -> go.Figure:
    """Plot Pauli data and fit only the protocol's primary component."""
    figure = go.Figure()
    times_us = times * 1e-3
    dense_times = np.linspace(0.0, float(times[-1]), 500)
    for kind in ("reference", "actual"):
        reference = kind == "reference"
        opacity = _REFERENCE_OPACITY if reference else 1.0
        for component_index, (basis, component_values) in enumerate(
            expectations.items()
        ):
            color = COLORS[component_index]
            marker_symbol = _PAULI_MARKERS[cast(_PauliBasis, basis)]
            if reference:
                marker_symbol = f"{marker_symbol}-open"
            figure.add_trace(
                go.Scatter(
                    x=times_us,
                    y=component_values[kind],
                    mode="markers",
                    marker={
                        "color": color,
                        "symbol": marker_symbol,
                    },
                    opacity=opacity,
                    error_y=_error_array(errors[basis][kind]),
                    name=f"{kind} <{basis}> data",
                )
            )
            if basis == primary_basis:
                figure.add_trace(
                    go.Scatter(
                        x=dense_times * 1e-3,
                        y=_exponential_curve(fits[kind], dense_times),
                        mode="lines",
                        line={
                            "color": color,
                            "dash": "dot" if reference else "solid",
                        },
                        opacity=opacity,
                        name=f"{kind} <{basis}> fit",
                    )
                )
    yaxis_title = (
        f"<{primary_basis}>" if len(expectations) == 1 else "Pauli expectation"
    )
    figure.update_layout(
        title=f"{_PROTOCOL_LABELS[protocol]}: {target}",
        yaxis={"title": yaxis_title, "range": [-1.05, 1.05]},
    )
    _add_top_cr_axis(figure, times_us, cr_counts)
    return figure


def _make_figures(
    times: Mapping[str, NDArray[np.float64]],
    cr_counts: Sequence[int],
    populations: Mapping[str, Mapping[str, Mapping[str, NDArray[np.float64]]]],
    population_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ],
    target_polarizations: Mapping[str, Mapping[str, NDArray[np.float64]]],
    target_polarization_errors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    pauli_expectations: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ],
    pauli_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ],
    fits: Mapping[str, object],
    control_qubit: str,
    target_qubit: str,
    protocols: Sequence[_Protocol],
) -> dict[str, go.Figure]:
    """Build fixed-scale figures for the selected protocols."""
    figures: dict[str, go.Figure] = {}
    selected_gef_protocols: tuple[_GefProtocol, ...] = tuple(
        cast(_GefProtocol, protocol)
        for protocol in protocols
        if protocol in _GEF_PROTOCOLS
    )
    if selected_gef_protocols:
        rate_fits = cast(
            Mapping[str, ThreeLevelRateFit],
            fits["control_rate_model"],
        )
        t1rho_fits = cast(
            Mapping[str, Mapping[str, ExponentialDecayFit]],
            fits["target_t1rho"],
        )
        for protocol in selected_gef_protocols:
            figures[f"{protocol}_control_populations"] = _make_control_figure(
                protocol,
                times[protocol],
                cr_counts,
                populations[protocol],
                population_errors[protocol],
                control_qubit,
                rate_fits,
            )
            figures[f"{protocol}_target_polarization"] = _make_target_figure(
                protocol,
                times[protocol],
                cr_counts,
                populations[protocol],
                population_errors[protocol],
                target_qubit,
                target_polarizations[protocol],
                target_polarization_errors[protocol],
                t1rho_fits[protocol],
            )
    if _CONTROL_T2_ECHO in protocols:
        control_t2_fits = cast(
            Mapping[str, ExponentialDecayFit],
            fits[_CONTROL_T2_ECHO],
        )
        figures[_CONTROL_T2_ECHO] = _make_pauli_figure(
            _CONTROL_T2_ECHO,
            times[_CONTROL_T2_ECHO],
            cr_counts,
            pauli_expectations[_CONTROL_T2_ECHO],
            pauli_errors[_CONTROL_T2_ECHO],
            control_t2_fits,
            control_qubit,
            "X",
        )
    if _TARGET_T2RHO_ECHO in protocols:
        target_t2rho_fits = cast(
            Mapping[str, ExponentialDecayFit],
            fits[_TARGET_T2RHO_ECHO],
        )
        figures[_TARGET_T2RHO_ECHO] = _make_pauli_figure(
            _TARGET_T2RHO_ECHO,
            times[_TARGET_T2RHO_ECHO],
            cr_counts,
            pauli_expectations[_TARGET_T2RHO_ECHO],
            pauli_errors[_TARGET_T2RHO_ECHO],
            target_t2rho_fits,
            target_qubit,
            "Z",
        )
    return figures


def _value_with_error(value: float, standard_error: float) -> dict[str, float]:
    """Return one scalar estimate and its one-standard-error uncertainty."""
    return {"value": value, "standard_error": standard_error}


def _summarize_fit_parameters(
    fits: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build convenient rate and decay-time summaries from fit dataclasses."""
    transition_rates: dict[str, object] = {"unit": "1/ns"}
    decay_times: dict[str, object] = {"unit": "ns"}
    if "control_rate_model" in fits:
        rate_fits = cast(
            Mapping[str, ThreeLevelRateFit],
            fits["control_rate_model"],
        )
        t1rho_fits = cast(
            Mapping[str, Mapping[str, ExponentialDecayFit]],
            fits["target_t1rho"],
        )
        transition_rates.update(
            {
                kind: {
                    "gamma_ge_down": _value_with_error(
                        fit.gamma_ge_down,
                        fit.gamma_ge_down_error,
                    ),
                    "gamma_ge_up": _value_with_error(
                        fit.gamma_ge_up,
                        fit.gamma_ge_up_error,
                    ),
                    "gamma_ef_down": _value_with_error(
                        fit.gamma_ef_down,
                        fit.gamma_ef_down_error,
                    ),
                    "gamma_ef_up": _value_with_error(
                        fit.gamma_ef_up,
                        fit.gamma_ef_up_error,
                    ),
                }
                for kind, fit in rate_fits.items()
            }
        )
        decay_times["T1_eff"] = {
            kind: _value_with_error(fit.t1_eff, fit.t1_eff_error)
            for kind, fit in rate_fits.items()
        }
        decay_times["T1rho"] = {
            protocol: {
                kind: _value_with_error(fit.tau, fit.tau_error)
                for kind, fit in protocol_fits.items()
            }
            for protocol, protocol_fits in t1rho_fits.items()
        }

    if _CONTROL_T2_ECHO in fits:
        control_t2_fits = cast(
            Mapping[str, ExponentialDecayFit],
            fits[_CONTROL_T2_ECHO],
        )
        decay_times["T2_echo"] = {
            kind: _value_with_error(fit.tau, fit.tau_error)
            for kind, fit in control_t2_fits.items()
        }
    if _TARGET_T2RHO_ECHO in fits:
        target_t2rho_fits = cast(
            Mapping[str, ExponentialDecayFit],
            fits[_TARGET_T2RHO_ECHO],
        )
        decay_times["T2rho_echo"] = {
            kind: _value_with_error(fit.tau, fit.tau_error)
            for kind, fit in target_t2rho_fits.items()
        }
    return transition_rates, decay_times


def characterize_cr_pulse_coherence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    protocols: Collection[str] | str | None = None,
    measure_orthogonal_components: bool = False,
    n_values: Sequence[int] | None = None,
    zx90_no_echo: PulseSchedule | None = None,
    zx90_echo: PulseSchedule | None = None,
    n_shots: int | None = None,
    calibration_n_shots: int | None = None,
    shot_interval: float | None = None,
    covariance_rcond: float = 1e-12,
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = 0,
    bootstrap_confidence_level: float = 0.95,
    enable_tqdm: bool = True,
    plot: bool = True,
) -> Result:
    """
    Characterize CR-induced GEF transitions and echoed coherence decay.

    Parameters
    ----------
    exp
        Experiment used for pulse construction and hardware measurements.
    control_qubit
        Control qubit label.
    target_qubit
        Target qubit label.
    protocols
        Protocol or protocols to measure. Supported names are `control_ground`,
        `control_excited`, `control_t2_echo`, and `target_t2rho_echo`. Defaults
        to all four. `control_ground` and `control_excited` must be selected
        together because their four-rate model is fitted jointly. Measurements
        always run in the order listed above.
    measure_orthogonal_components
        Whether `control_t2_echo` and `target_t2rho_echo` also measure their two
        orthogonal Pauli components. Defaults to `False`.
    n_values
        Unique increasing nonnegative repetition indices beginning at zero.
        Defaults to `(0, 1, 2, 3, 5, 8, 13, 21, 34, 55)`.
    zx90_no_echo
        Optional un-echoed ZX90 schedule override.
    zx90_echo
        Optional echoed ZX90 schedule override.
    n_shots
        Shots per measurement configuration. Must be at least two. Defaults to
        `DEFAULT_SHOTS`.
    calibration_n_shots
        Shots per GEF calibration configuration. Must be at least two. Defaults
        to `CALIBRATION_SHOTS`.
    shot_interval
        Interval between shots in ns. Defaults to `DEFAULT_INTERVAL`.
    covariance_rcond
        Relative cutoff used by GEF covariance pseudo-inverses.
    n_bootstrap
        Number of joint raw-shot GEF bootstrap resamples. Defaults to 1000.
    bootstrap_seed
        Nonnegative bootstrap seed, or `None` for nondeterministic resampling.
    bootstrap_confidence_level
        Marginal bootstrap confidence level strictly between zero and one.
    enable_tqdm
        Whether to show progress over `n_values`.
    plot
        Whether to display all generated figures.

    Returns
    -------
    Result
        Raw measurements, derived observables, actual/reference fits, timing
        metadata, optional GEF calibration, and figures for selected protocols.

    Notes
    -----
    `control_ground` prepares control `|0>` and target `|+>`, applies an
    un-echoed ZX90 `4n` times, and finishes with target Y90.
    `control_excited` uses control `|1>` and the same target preparation and
    analyzer. Their references replace ZX90 by duration-matched target `+X90`
    and `-X90`, respectively.

    `control_t2_echo` prepares `|+,+>`, repeats
    `(ZX90_echo -> XI180 -> ZX90_echo -> XI180 + IX180 -> ZX90_echo -> XI180
    -> ZX90_echo)` `n` times, and measures control X. `target_t2rho_echo`
    prepares `|0,0>`, repeats `(ZX90_echo -> IZ180 -> ZX90_echo)` `2n` times,
    and measures target Z. Their ZX90 references are a matched blank and
    matched target X90.

    GEF calibration is performed only when `control_ground` and
    `control_excited` are selected. When orthogonal components are requested,
    `control_t2_echo` measures X/Y/Z and `target_t2rho_echo` measures Z/X/Y,
    with the primary component listed first. Orthogonal components are plotted
    as reference information without fitting. The upper plot axis counts ZX90
    schedule calls; one echoed ZX90 internally contains two physical CR lobes
    in the current pulse implementation. Fitted rates use `1/ns`; all returned
    decay times use ns. Convenient scalar summaries are available in
    `result.data["transition_rates"]` and `result.data["decay_times"]`.
    """
    selected_protocols = _validate_protocols(protocols)
    if not isinstance(measure_orthogonal_components, bool):
        raise TypeError("measure_orthogonal_components must be a boolean.")
    selected_gef_protocols: tuple[_GefProtocol, ...] = tuple(
        cast(_GefProtocol, protocol)
        for protocol in selected_protocols
        if protocol in _GEF_PROTOCOLS
    )
    selected_pauli_protocols: tuple[_PauliProtocol, ...] = tuple(
        cast(_PauliProtocol, protocol)
        for protocol in selected_protocols
        if protocol in _PAULI_PROTOCOLS
    )
    normalized_n_values = _validate_n_values(n_values)
    resolved_n_shots = _resolve_shot_count(
        n_shots,
        default=DEFAULT_SHOTS,
        name="n_shots",
    )
    resolved_calibration_n_shots = (
        _resolve_shot_count(
            calibration_n_shots,
            default=CALIBRATION_SHOTS,
            name="calibration_n_shots",
        )
        if selected_gef_protocols
        else None
    )
    resolved_shot_interval = _positive_real(
        shot_interval,
        default=DEFAULT_INTERVAL,
        name="shot_interval",
    )
    resolved_n_bootstrap = _nonnegative_integer(n_bootstrap, name="n_bootstrap")
    resolved_bootstrap_seed = (
        None
        if bootstrap_seed is None
        else _nonnegative_integer(bootstrap_seed, name="bootstrap_seed")
    )
    resolved_bootstrap_confidence_level = _unit_interval_real(
        bootstrap_confidence_level,
        name="bootstrap_confidence_level",
        include_zero=False,
    )
    resolved_covariance_rcond = _unit_interval_real(
        covariance_rcond,
        name="covariance_rcond",
        include_zero=True,
    )

    control = exp.ctx.resolve_qubit_label(control_qubit)
    target = exp.ctx.resolve_qubit_label(target_qubit)
    if control == target:
        raise ValueError("control_qubit and target_qubit must be different.")
    if selected_gef_protocols and zx90_no_echo is None:
        zx90_no_echo = exp.pulse.zx90(control, target, echo=False)
    if selected_pauli_protocols and zx90_echo is None:
        zx90_echo = exp.pulse.zx90(control, target, echo=True)

    calibration: dict[str, GefPopulationCalibration] | None = None
    if selected_gef_protocols:
        calibration = calibrate_gef_population(
            exp,
            targets=[control, target],
            n_shots=resolved_calibration_n_shots,
            shot_interval=resolved_shot_interval,
        )
    gef_conditions = tuple(
        condition
        for protocol in selected_gef_protocols
        for condition in (f"{protocol}_reference", protocol)
    )
    point_populations: dict[str, dict[str, list[NDArray[np.float64]]]] = {
        condition: {control: [], target: []} for condition in gef_conditions
    }
    point_errors: dict[str, dict[str, list[NDArray[np.float64]]]] = {
        condition: {control: [], target: []} for condition in point_populations
    }
    aggregate_raw_iq: dict[
        str,
        dict[str, dict[str, NDArray[np.complex128]]],
    ] = {}
    aggregate_gef_fits: dict[str, dict[str, GefPopulationFit]] = {}
    aggregate_moment_summaries: dict[
        str,
        dict[str, dict[str, IQMomentSummary]],
    ] = {}
    bootstrap_key_order: list[tuple[int, str, str]] = []
    times_buffer: dict[str, list[float]] = {
        protocol: [] for protocol in selected_protocols
    }
    cr_pulse_counts_buffer: list[int] = []
    sequence_durations: dict[str, list[float]] = {
        condition: []
        for protocol in selected_protocols
        for condition in (protocol, f"{protocol}_reference")
    }
    pauli_components = {
        protocol: _pauli_components(protocol, measure_orthogonal_components)
        for protocol in selected_pauli_protocols
    }
    pauli_measurements: dict[
        str,
        dict[_PauliBasis, list[_PauliMeasurement]],
    ] = {
        condition: {basis: [] for basis in pauli_components[protocol]}
        for protocol in selected_pauli_protocols
        for condition in (f"{protocol}_reference", protocol)
    }

    progress = tqdm(
        normalized_n_values,
        desc=f"CR coherence {control}-{target}",
        disable=not enable_tqdm,
    )
    for point_index, n in enumerate(progress):
        point = _build_protocol_sequences(
            exp,
            control,
            target,
            n=n,
            zx90_no_echo=zx90_no_echo,
            zx90_echo=zx90_echo,
            protocols=selected_protocols,
        )
        cr_pulse_counts_buffer.append(point.cr_pulse_count)
        for protocol in selected_protocols:
            times_buffer[protocol].append(point.evolution_durations[protocol])
            for condition in (protocol, f"{protocol}_reference"):
                sequence_durations[condition].append(
                    point.sequences[condition].duration
                )

        if calibration is not None:
            gef_sequences = {
                condition: point.sequences[condition] for condition in gef_conditions
            }
            gef_result = measure_gef_populations(
                exp,
                targets=[control, target],
                sequences=gef_sequences,
                calibration=calibration,
                n_shots=resolved_n_shots,
                shot_interval=resolved_shot_interval,
                covariance_rcond=resolved_covariance_rcond,
                n_bootstrap=0,
            )
            for condition in gef_sequences:
                global_name = _bootstrap_name(n, condition)
                aggregate_raw_iq[global_name] = gef_result.data["raw_iq"][condition]
                aggregate_gef_fits[global_name] = gef_result.data["fits"][condition]
                aggregate_moment_summaries[global_name] = gef_result.data[
                    "moment_summaries"
                ][condition]
                bootstrap_key_order.append((point_index, condition, global_name))
                for qubit in (control, target):
                    point_populations[condition][qubit].append(
                        np.asarray(
                            gef_result.data["populations"][condition][qubit],
                            dtype=np.float64,
                        )
                    )

        for protocol in selected_pauli_protocols:
            measured_qubit = control if protocol == _CONTROL_T2_ECHO else target
            for condition in (f"{protocol}_reference", protocol):
                for basis in pauli_components[protocol]:
                    pauli_measurements[condition][basis].append(
                        _measure_pauli_expectation(
                            exp,
                            point.sequences[condition],
                            measured_qubit,
                            basis,
                            n_shots=resolved_n_shots,
                            shot_interval=resolved_shot_interval,
                        )
                    )

    bootstrap: dict[str, dict[str, GefPopulationBootstrap]] = {}
    bootstrap_lookup: dict[tuple[int, str, str], GefPopulationBootstrap] = {}
    if calibration is not None:
        bootstrap = bootstrap_gef_populations(
            calibration,
            aggregate_raw_iq,
            n_resamples=resolved_n_bootstrap,
            seed=resolved_bootstrap_seed,
            confidence_level=resolved_bootstrap_confidence_level,
            covariance_rcond=resolved_covariance_rcond,
        )
        for point_index, condition, global_name in bootstrap_key_order:
            for qubit in (control, target):
                population_bootstrap = bootstrap[global_name][qubit]
                bootstrap_lookup[(point_index, condition, qubit)] = population_bootstrap
                point_errors[condition][qubit].append(
                    _population_standard_error(population_bootstrap)
                )

        populations, population_errors = _collect_gef_arrays(
            point_populations,
            point_errors,
            control,
            target,
            selected_gef_protocols,
        )
    else:
        populations, population_errors = {}, {}
    times = {
        protocol: np.asarray(values, dtype=np.float64)
        for protocol, values in times_buffer.items()
    }
    cr_pulse_counts = tuple(cr_pulse_counts_buffer)
    target_polarizations: dict[str, dict[str, NDArray[np.float64]]] = {}
    target_polarization_errors: dict[str, dict[str, NDArray[np.float64]]] = {}
    for protocol in selected_gef_protocols:
        target_polarizations[protocol] = {}
        target_polarization_errors[protocol] = {}
        for kind in ("actual", "reference"):
            condition = _condition_name(protocol, kind == "reference")
            target_polarizations[protocol][kind] = _polarization(
                populations[protocol][kind][target]
            )
            target_polarization_errors[protocol][kind] = np.array(
                [
                    _polarization_standard_error(
                        bootstrap_lookup[(point_index, condition, target)]
                    )
                    for point_index in range(len(normalized_n_values))
                ],
                dtype=np.float64,
            )

    pauli_expectations: dict[
        str,
        dict[str, dict[str, NDArray[np.float64]]],
    ] = {}
    pauli_errors: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    pauli_raw_iq: dict[
        str,
        dict[str, dict[str, list[NDArray[np.complex128]]]],
    ] = {}
    pauli_normalized_shots: dict[
        str,
        dict[str, dict[str, list[NDArray[np.float64]]]],
    ] = {}
    for protocol in selected_pauli_protocols:
        pauli_expectations[protocol] = {}
        pauli_errors[protocol] = {}
        pauli_raw_iq[protocol] = {}
        pauli_normalized_shots[protocol] = {}
        for basis in pauli_components[protocol]:
            pauli_expectations[protocol][basis] = {}
            pauli_errors[protocol][basis] = {}
            pauli_raw_iq[protocol][basis] = {}
            pauli_normalized_shots[protocol][basis] = {}
            for kind in ("actual", "reference"):
                condition = _condition_name(protocol, kind == "reference")
                measurements = pauli_measurements[condition][basis]
                pauli_expectations[protocol][basis][kind] = np.asarray(
                    [measurement.expectation for measurement in measurements],
                    dtype=np.float64,
                )
                pauli_errors[protocol][basis][kind] = np.asarray(
                    [measurement.standard_error for measurement in measurements],
                    dtype=np.float64,
                )
                pauli_raw_iq[protocol][basis][kind] = [
                    measurement.raw_iq for measurement in measurements
                ]
                pauli_normalized_shots[protocol][basis][kind] = [
                    measurement.normalized_shots for measurement in measurements
                ]

    fits = _fit_all_results(
        times,
        populations,
        population_errors,
        target_polarizations,
        target_polarization_errors,
        pauli_expectations,
        pauli_errors,
        control,
        selected_protocols,
    )
    transition_rates, decay_times = _summarize_fit_parameters(fits)
    figures = _make_figures(
        times,
        cr_pulse_counts,
        populations,
        population_errors,
        target_polarizations,
        target_polarization_errors,
        pauli_expectations,
        pauli_errors,
        fits,
        control,
        target,
        selected_protocols,
    )
    if plot:
        for figure in figures.values():
            figure.show()

    return Result(
        data={
            "control_qubit": control,
            "target_qubit": target,
            "protocols": selected_protocols,
            "pauli_components": pauli_components,
            "state_order": _STATE_NAMES,
            "n_values": normalized_n_values,
            "cr_pulse_counts": cr_pulse_counts,
            "cr_pulse_count_definition": "ZX90 schedule calls",
            "times": times,
            "time_unit": "ns",
            "sequence_durations": {
                name: np.asarray(values, dtype=np.float64)
                for name, values in sequence_durations.items()
            },
            "populations": populations,
            "population_standard_errors": population_errors,
            "target_polarizations": target_polarizations,
            "target_polarization_standard_errors": target_polarization_errors,
            "pauli_expectations": pauli_expectations,
            "pauli_standard_errors": pauli_errors,
            "fits": fits,
            "transition_rates": transition_rates,
            "decay_times": decay_times,
            "calibration": calibration,
            "gef_bootstrap": bootstrap,
            "gef_population_fits": aggregate_gef_fits,
            "gef_raw_iq": aggregate_raw_iq,
            "gef_moment_summaries": aggregate_moment_summaries,
            "pauli_raw_iq": pauli_raw_iq,
            "pauli_normalized_shots": pauli_normalized_shots,
            "measurement_options": {
                "n_shots": resolved_n_shots,
                "calibration_n_shots": resolved_calibration_n_shots,
                "shot_interval": resolved_shot_interval,
                "covariance_rcond": resolved_covariance_rcond,
                "n_bootstrap": resolved_n_bootstrap,
                "bootstrap_seed": resolved_bootstrap_seed,
                "bootstrap_confidence_level": resolved_bootstrap_confidence_level,
                "measure_orthogonal_components": measure_orthogonal_components,
            },
            "pulse_durations": {
                "zx90_no_echo": (
                    None if zx90_no_echo is None else zx90_no_echo.duration
                ),
                "zx90_echo": None if zx90_echo is None else zx90_echo.duration,
            },
        },
        figure=next(iter(figures.values())),
        figures=figures,
    )


__all__ = [
    "DEFAULT_N_VALUES",
    "ExponentialDecayFit",
    "ThreeLevelRateFit",
    "characterize_cr_pulse_coherence",
    "fit_exponential_decay",
    "fit_three_level_rate_model",
]
