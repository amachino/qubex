"""
Experimental CPMG noise spectroscopy utilities.

This module provides a frequency-swept CPMG workflow for exploratory noise
spectroscopy. The implementation intentionally lives in `contrib` because its
API and frequency-label approximation remain experimental. Only positive
frequencies are accepted.

The public functions follow the same calling style as qubex experiment methods:
`exp` is the first argument, `targets` is optional, measurement options are
keyword-only, and omitted options fall back to qubex defaults. Result payloads
are returned in a generic `Result` object and can also be reloaded from saved
JSON and NumPy matrix files.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.experiment_util import ExperimentUtil
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, PulseSchedule, Waveform

from ._deprecated_options import resolve_shot_options

__all__ = [
    "CPMGResultPayload",
    "cpmg_noise_spectroscopy",
    "plot_cpmg_results",
]

CPMGPattern = Literal["++++", "+--+", "+-+-"]
CPMGSummaryMode = Literal["t2", "gamma2", "both"]
CPMGFitStatus = Literal[
    "ok",
    "quality_warning",
    "unresolved_decay",
    "invalid_shape",
    "insufficient_points",
    "invalid_time_axis",
    "optimizer_failed",
]
AxisType = Literal["linear", "log"]
_DisplayedMode = Literal["t2", "gamma2"]
_DEFAULT_SAVE_DIR = "cpmg_noise_spectroscopy"
_GRID_ATOL_NS = 1e-9
_SCHEMA_VERSION = 2
_DEFAULT_FREQUENCY_RANGE = np.linspace(0.0005, 0.004, 20)
_DEFAULT_TIME_RANGE = np.geomspace(200.0, 200_000.0, 31)
_DEFAULT_MIN_PI_PULSES = 4
_DEFAULT_MAX_PULSE_DUTY_RATIO = 0.25
_DEFAULT_MIN_FIT_POINTS = 6
_LARGE_RELATIVE_ERROR = 0.3
_MIN_DECAY_FRACTION = 0.05
_MIN_ACCEPTABLE_R2 = 0.8

logger = logging.getLogger(__name__)


class CPMGResultPayload(TypedDict):
    """In-memory CPMG result payload used by measurement and plotting helpers."""

    timestamp: str
    targets: list[str]
    n_repeats: int
    schema_version: int
    frequency_range: list[float]
    realized_frequency_matrix: NDArray[np.float64]
    valid_frequency_mask: NDArray[np.bool_]
    half_tau_range: list[float]
    tau_spacing_list_ns: list[float]
    time_range: list[float] | None
    t2_matrix: NDArray[np.float64]
    t2_error_matrix: NDArray[np.float64]
    gamma2_matrix: NDArray[np.float64]
    gamma2_error_matrix: NDArray[np.float64]
    r2_matrix: NDArray[np.float64]
    fit_records: list[dict[str, object]]
    frequency_plans: list[list[dict[str, object]]]
    acquisition_orders: list[list[list[int]]]
    random_seed: int
    random_seed_requested: int | None
    metadata: dict[str, object]
    data: NDArray[np.float64]
    failed_fits: list[dict[str, object]]
    save_dir: str | None


@dataclass(frozen=True)
class _FrequencyPlan:
    """Realized CPMG timing for one requested-frequency point."""

    requested_index: int
    requested_frequency: float
    frequency: float | None
    half_tau: float | None
    center_spacing: float | None
    pulse_duty_ratio: float | None
    status: Literal["valid", "skipped", "duplicate"]
    reason: str | None = None
    representative_index: int | None = None


@dataclass(frozen=True)
class _CPMGFitResult:
    """Numerical result of one CPMG decay fit without visualization state."""

    fit_status: CPMGFitStatus
    fit_message: str
    quality_flags: tuple[str, ...]
    amplitude: float = math.nan
    offset: float = math.nan
    gamma2_per_ns: float = math.nan
    gamma2_error_per_ns: float = math.nan
    t2_ns: float = math.nan
    t2_error_ns: float = math.nan
    covariance: NDArray[np.float64] | None = None
    r2: float = math.nan
    reduced_chi2: float = math.nan
    n_points: int = 0
    observed_contrast: float = math.nan
    time_min_ns: float = math.nan
    time_max_ns: float = math.nan
    relative_gamma_error: float = math.nan
    fit_method_used: str = "unweighted_least_squares"
    uncertainty_source: str = "residual_covariance"


@dataclass(frozen=True)
class _RepeatSummary:
    """Repeat-aggregated decay rates for one quality group."""

    gamma2: NDArray[np.float64]
    errors: NDArray[np.float64]
    counts: NDArray[np.int64]


@dataclass(frozen=True)
class _MeasurementPlan:
    """Target-specific pulse and timing plan for one CPMG run."""

    targets: list[str]
    requested_frequencies: NDArray[np.float64]
    requested_times: NDArray[np.float64]
    sampling_period: float
    pi_pulses: dict[str, Waveform]
    frequency_plans: list[list[_FrequencyPlan]]
    n_pi_values: list[dict[int, NDArray[np.int64]]]


@dataclass(frozen=True)
class _AcquisitionOptions:
    """Validated options used while acquiring and fitting CPMG data."""

    n_repeats: int
    n_shots: int
    shot_interval: float
    min_fit_points: int
    frequency_order: Literal["random", "ascending"]
    random_seed: int | None
    pi_pulse_pattern: CPMGPattern
    save_fit_images: bool
    xaxis_type: AxisType
    enable_tqdm: bool


@dataclass
class _AcquisitionData:
    """Mutable numerical outputs populated during a CPMG run."""

    t2: NDArray[np.float64]
    t2_error: NDArray[np.float64]
    gamma2: NDArray[np.float64]
    gamma2_error: NDArray[np.float64]
    r2: NDArray[np.float64]
    realized_frequencies: NDArray[np.float64]
    valid_frequencies: NDArray[np.bool_]
    fit_records: list[dict[str, object]]
    failed_fits: list[dict[str, object]]
    acquisition_orders: list[list[list[int]]]


@dataclass(frozen=True)
class _PointMeasurement:
    """Measured data and fit result for one target-frequency-repeat point."""

    fit: _CPMGFitResult
    frequency_ghz: float
    record: dict[str, object]
    times_ns: NDArray[np.float64]
    probabilities: NDArray[np.float64]


# Input normalization and pulse planning


def _resolve_deprecated_alias(
    *,
    options: dict[str, Any],
    deprecated_name: str,
    replacement_name: str,
    current_value: Any,
    function_name: str,
) -> Any:
    """Resolve one deprecated keyword alias for this module."""
    if deprecated_name not in options:
        return current_value

    legacy_value = options.pop(deprecated_name)
    if legacy_value is None:
        return current_value
    warnings.warn(
        f"`{deprecated_name}` is deprecated in `{function_name}`; "
        f"use `{replacement_name}`.",
        DeprecationWarning,
        stacklevel=3,
    )
    if current_value is not None and current_value != legacy_value:
        raise ValueError(
            f"`{deprecated_name}` conflicts with `{replacement_name}`. "
            f"Provide only `{replacement_name}`."
        )
    return legacy_value


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    """Return target labels in the same style as qubex experiment methods."""
    if targets is None:
        labels = getattr(getattr(exp, "ctx", object()), "qubit_labels", None)
        if labels is None:
            labels = getattr(exp, "qubit_labels", None)
        if labels is None:
            raise ValueError("targets must be provided when exp has no qubit labels.")
        return list(labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _validate_rabi_params(exp: Experiment, targets: Collection[str]) -> None:
    """Validate calibrated pulses when the experiment exposes pulse service."""
    pulse_service = getattr(exp, "pulse", None)
    validate = getattr(pulse_service, "validate_rabi_params", None)
    if callable(validate):
        validate(list(targets))


def _as_non_empty_1d_array(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Validate and convert a positive one-dimensional numeric range."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array <= 0):
        raise ValueError(f"{name} must contain only positive values.")
    return array


def _require_integer(value: object, *, name: str, minimum: int) -> int:
    """Return an integer option after validating its type and lower bound."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {normalized}.")
    return normalized


def _resolve_sampling_period(exp: Experiment) -> float:
    """Return the measurement sampling period in ns."""
    measurement = getattr(getattr(exp, "ctx", object()), "measurement", None)
    sampling_period = getattr(measurement, "sampling_period", None)
    if sampling_period is not None:
        return float(sampling_period)
    return float(ExperimentUtil.resolve_sampling_period())


def _sampling_ticks(duration_ns: float, *, sampling_period: float, name: str) -> int:
    """Return integer sampling ticks after checking grid alignment."""
    if not np.isfinite(duration_ns):
        raise ValueError(f"{name} must be finite, got {duration_ns}.")
    if not np.isfinite(sampling_period) or sampling_period <= 0:
        raise ValueError(
            f"sampling_period must be finite and positive, got {sampling_period}."
        )

    ticks = round(duration_ns / sampling_period)
    snapped = ticks * sampling_period
    if not np.isclose(duration_ns, snapped, rtol=0.0, atol=_GRID_ATOL_NS):
        raise ValueError(
            f"{name} must be on the {sampling_period} ns sampling grid; "
            f"got {duration_ns} ns."
        )
    return int(ticks)


def _get_hpi_pulse(exp: Experiment, target: str) -> Waveform:
    """Return the half-pi pulse for a target."""
    pulse_service = getattr(exp, "pulse", None)
    get_hpi_pulse = getattr(pulse_service, "get_hpi_pulse", None)
    if callable(get_hpi_pulse):
        return _require_waveform(
            get_hpi_pulse(target),
            source="exp.pulse.get_hpi_pulse()",
            target=target,
        )

    hpi_pulses = getattr(exp, "hpi_pulse", None)
    if isinstance(hpi_pulses, Mapping) and target in hpi_pulses:
        return _require_waveform(
            hpi_pulses[target],
            source="exp.hpi_pulse",
            target=target,
        )

    raise ValueError(f"No half-pi pulse is available for target `{target}`.")


def _require_waveform(value: object, *, source: str, target: str) -> Waveform:
    """Return `value` after checking that it follows the Qubex waveform API."""
    if isinstance(value, Waveform):
        return value
    raise TypeError(
        f"{source} for target `{target}` must return a Waveform, "
        f"got {type(value).__name__}."
    )


def _get_optional_waveform(
    provider: object,
    *,
    target: str,
    source: str,
) -> Waveform | None:
    """Call an available pulse provider and return a validated waveform."""
    if not callable(provider):
        return None
    value = provider(target)
    return _require_waveform(value, source=source, target=target)


def _resolve_pi_pulse(
    exp: Experiment,
    target: str,
    pi_pulses: Mapping[str, Waveform] | None,
) -> Waveform:
    """Return the CPMG pi pulse for `target`."""
    if pi_pulses is not None and target in pi_pulses:
        return _require_waveform(
            pi_pulses[target],
            source="pi_pulses",
            target=target,
        )

    pulse_service = getattr(exp, "pulse", None)
    pi_pulse = _get_optional_waveform(
        getattr(pulse_service, "x180", None),
        target=target,
        source="exp.pulse.x180()",
    )
    if pi_pulse is not None:
        return pi_pulse.shifted(np.pi / 2)

    pi_pulse = _get_optional_waveform(
        getattr(exp, "x180", None),
        target=target,
        source="exp.x180()",
    )
    if pi_pulse is not None:
        return pi_pulse.shifted(np.pi / 2)

    get_pi_pulse = getattr(pulse_service, "get_pi_pulse", None)
    pi_pulse = _get_optional_waveform(
        get_pi_pulse,
        target=target,
        source="exp.pulse.get_pi_pulse()",
    )
    if pi_pulse is not None:
        return pi_pulse.shifted(np.pi / 2)

    return _get_hpi_pulse(exp, target).repeated(2).shifted(np.pi / 2)


def _build_cpmg_sequence(
    exp: Experiment,
    *,
    target: str,
    half_tau: float,
    n_pi: int,
    pattern: CPMGPattern,
    pi_pulses: Mapping[str, Waveform] | None,
    sampling_period: float,
) -> PulseSchedule:
    """Build one CPMG pulse schedule for a target and pi-pulse count."""
    if n_pi < 1:
        raise ValueError(f"n_pi must be at least 1, got {n_pi}.")
    if half_tau <= 0:
        raise ValueError(f"half_tau must be positive, got {half_tau}.")
    _sampling_ticks(half_tau, sampling_period=sampling_period, name="half_tau")

    hpi = _get_hpi_pulse(exp, target)
    pi = _resolve_pi_pulse(exp, target, pi_pulses)

    with PulseSchedule([target]) as schedule:
        schedule.add(target, hpi)
        for index in range(n_pi):
            schedule.add(target, Blank(half_tau, sampling_period=sampling_period))
            if pattern == "++++":
                signed_pi = pi
            elif pattern == "+--+":
                signed_pi = pi if index % 4 in (0, 3) else pi.scaled(-1)
            elif pattern == "+-+-":
                signed_pi = pi if index % 2 == 0 else pi.scaled(-1)
            else:
                raise ValueError(
                    f"Invalid pi_pulse_pattern `{pattern}`. "
                    "Expected one of '++++', '+--+', '+-+-'."
                )
            schedule.add(target, signed_pi)
            schedule.add(target, Blank(half_tau, sampling_period=sampling_period))
        schedule.add(target, hpi.scaled(-1))
    return schedule


def _make_frequency_plan(
    frequency_range: ArrayLike,
    *,
    sampling_period: float,
    pi_duration: float,
    max_pulse_duty_ratio: float,
) -> list[_FrequencyPlan]:
    """Convert requested frequencies to target-specific grid-aligned timings."""
    requested = _as_non_empty_1d_array(
        frequency_range,
        name="frequency_range",
    )
    if not 0 < max_pulse_duty_ratio <= 1:
        raise ValueError("max_pulse_duty_ratio must be in the interval (0, 1].")
    pi_ticks = _sampling_ticks(
        pi_duration,
        sampling_period=sampling_period,
        name="pi pulse duration",
    )
    if pi_ticks <= 0:
        raise ValueError("pi pulse duration must be positive.")

    plan: list[_FrequencyPlan] = []
    representative_by_ticks: dict[int, int] = {}
    for requested_index, frequency in enumerate(requested):
        center_spacing_requested = 1.0 / (2.0 * float(frequency))
        half_tau_requested = (center_spacing_requested - pi_duration) / 2.0
        if half_tau_requested <= 0:
            plan.append(
                _FrequencyPlan(
                    requested_index=requested_index,
                    requested_frequency=float(frequency),
                    frequency=None,
                    half_tau=None,
                    center_spacing=None,
                    pulse_duty_ratio=None,
                    status="skipped",
                    reason="nonpositive_half_tau",
                )
            )
            continue

        half_tau_ticks = round(half_tau_requested / sampling_period)
        if half_tau_ticks <= 0:
            plan.append(
                _FrequencyPlan(
                    requested_index=requested_index,
                    requested_frequency=float(frequency),
                    frequency=None,
                    half_tau=None,
                    center_spacing=None,
                    pulse_duty_ratio=None,
                    status="skipped",
                    reason="nonpositive_quantized_half_tau",
                )
            )
            continue

        center_spacing_ticks = 2 * half_tau_ticks + pi_ticks
        center_spacing = center_spacing_ticks * sampling_period
        realized_frequency = 1.0 / (2.0 * center_spacing)
        duty_ratio = pi_ticks / center_spacing_ticks
        if duty_ratio > max_pulse_duty_ratio + np.finfo(float).eps:
            plan.append(
                _FrequencyPlan(
                    requested_index=requested_index,
                    requested_frequency=float(frequency),
                    frequency=realized_frequency,
                    half_tau=half_tau_ticks * sampling_period,
                    center_spacing=center_spacing,
                    pulse_duty_ratio=duty_ratio,
                    status="skipped",
                    reason="pulse_duty_ratio_exceeded",
                )
            )
            continue

        representative = representative_by_ticks.get(half_tau_ticks)
        if representative is not None:
            plan.append(
                _FrequencyPlan(
                    requested_index=requested_index,
                    requested_frequency=float(frequency),
                    frequency=realized_frequency,
                    half_tau=half_tau_ticks * sampling_period,
                    center_spacing=center_spacing,
                    pulse_duty_ratio=duty_ratio,
                    status="duplicate",
                    reason="duplicate_realized_frequency",
                    representative_index=representative,
                )
            )
            continue

        representative_by_ticks[half_tau_ticks] = requested_index
        plan.append(
            _FrequencyPlan(
                requested_index=requested_index,
                requested_frequency=float(frequency),
                frequency=realized_frequency,
                half_tau=half_tau_ticks * sampling_period,
                center_spacing=center_spacing,
                pulse_duty_ratio=duty_ratio,
                status="valid",
                representative_index=requested_index,
            )
        )
    return plan


def _pattern_period(pattern: CPMGPattern) -> int:
    """Return the pulse-count period required by a CPMG sign pattern."""
    return 4 if pattern == "+--+" else 2 if pattern == "+-+-" else 1


def _n_pi_from_time_range(
    time_range: ArrayLike,
    *,
    center_spacing_ns: float,
    pattern: CPMGPattern,
    min_pi_pulses: int = 1,
) -> NDArray[np.int64]:
    """Round decay times directly to the nearest complete pattern cycles."""
    times = _as_non_empty_1d_array(
        time_range,
        name="time_range",
    )
    if not np.isfinite(center_spacing_ns) or center_spacing_ns <= 0:
        raise ValueError("center_spacing_ns must be finite and positive.")
    min_pi_pulses = _require_integer(
        min_pi_pulses,
        name="min_pi_pulses",
        minimum=1,
    )
    period = _pattern_period(pattern)
    with np.errstate(over="ignore"):
        requested_cycles = times / center_spacing_ns / period
    maximum_cycles = np.iinfo(np.int64).max // period
    if np.any(~np.isfinite(requested_cycles)) or np.any(
        requested_cycles > maximum_cycles
    ):
        raise ValueError("time_range produces pi-pulse counts outside int64 range.")
    counts = np.floor(requested_cycles + 0.5).astype(np.int64) * period
    return np.unique(counts[counts >= min_pi_pulses])


# Decay fitting


def _decay_model_us(
    time_us: NDArray[np.float64],
    amplitude: float,
    gamma2_per_us: float,
    offset: float,
) -> NDArray[np.float64]:
    """Evaluate the CPMG exponential model using microsecond-scaled time."""
    return amplitude * np.exp(-gamma2_per_us * time_us) + offset


def _fit_cpmg_decay(
    times_ns: ArrayLike,
    values: ArrayLike,
    *,
    sem_values: ArrayLike | None = None,
    min_fit_points: int = _DEFAULT_MIN_FIT_POINTS,
) -> _CPMGFitResult:
    """Fit one CPMG decay and classify its numerical quality without plotting."""
    times = np.asarray(times_ns, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or signal.ndim != 1 or times.shape != signal.shape:
        return _CPMGFitResult(
            fit_status="invalid_shape",
            fit_message="times and values must be one-dimensional arrays of equal shape.",
            quality_flags=("invalid_shape",),
        )
    min_fit_points = _require_integer(
        min_fit_points,
        name="min_fit_points",
        minimum=4,
    )

    finite = np.isfinite(times) & np.isfinite(signal)
    sigma: NDArray[np.float64] | None = None
    uncertainty_source = "residual_covariance"
    fit_method = "unweighted_least_squares"
    if sem_values is not None:
        candidate = np.asarray(sem_values, dtype=np.float64)
        retained_sem = candidate[finite] if candidate.shape == signal.shape else None
        if (
            retained_sem is not None
            and np.all(np.isfinite(retained_sem))
            and np.all(retained_sem > 0)
        ):
            sigma = retained_sem
            uncertainty_source = "measurement_sem"
            fit_method = "weighted_least_squares"
        else:
            uncertainty_source = "residual_covariance_invalid_sem"

    x_ns = times[finite]
    y = signal[finite]
    if len(x_ns) < min_fit_points:
        return _CPMGFitResult(
            fit_status="insufficient_points",
            fit_message=f"Only {len(x_ns)} finite points remain; {min_fit_points} are required.",
            quality_flags=("insufficient_points",),
            n_points=len(x_ns),
        )
    order = np.argsort(x_ns)
    x_ns = x_ns[order]
    y = y[order]
    if sigma is not None:
        sigma = sigma[order]
    if np.any(np.diff(x_ns) <= 0):
        return _CPMGFitResult(
            fit_status="invalid_time_axis",
            fit_message="Decay times must be distinct and increasing.",
            quality_flags=("invalid_time_axis",),
            n_points=len(x_ns),
        )

    contrast = float(np.ptp(y))
    scale = max(float(np.max(np.abs(y))), 1.0)
    time_min = float(x_ns[0])
    time_max = float(x_ns[-1])
    if contrast <= np.finfo(float).eps * scale:
        return _CPMGFitResult(
            fit_status="unresolved_decay",
            fit_message="The observed signal has no resolvable decay contrast.",
            quality_flags=("weak_decay",),
            n_points=len(x_ns),
            observed_contrast=contrast,
            time_min_ns=time_min,
            time_max_ns=time_max,
            fit_method_used=fit_method,
            uncertainty_source=uncertainty_source,
        )

    from scipy.optimize import curve_fit

    x_us = x_ns * 1e-3
    span_us = float(x_us[-1] - x_us[0])
    edge_count = max(2, len(y) // 5)
    start = float(np.median(y[:edge_count]))
    end = float(np.median(y[-edge_count:]))
    amplitude_guess = start - end
    if abs(amplitude_guess) < contrast * 0.1:
        amplitude_guess = float(y[0] - y[-1])
    if abs(amplitude_guess) < contrast * 0.1:
        amplitude_guess = math.copysign(contrast, float(y[0] - np.median(y)))
    offset_guess = end
    gamma_guesses = (0.2 / span_us, 1.0 / span_us, 5.0 / span_us)
    best: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None
    last_error = "optimizer failed"
    for gamma_guess in gamma_guesses:
        try:
            popt, pcov = curve_fit(
                _decay_model_us,
                x_us,
                y,
                p0=(amplitude_guess, gamma_guess, offset_guess),
                bounds=((-np.inf, 0.0, -np.inf), (np.inf, np.inf, np.inf)),
                sigma=sigma,
                absolute_sigma=sigma is not None,
                maxfev=20_000,
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            last_error = str(exc)
            continue
        residuals = y - _decay_model_us(x_us, *popt)
        score = float(
            np.sum((residuals / sigma) ** 2)
            if sigma is not None
            else np.sum(residuals**2)
        )
        if best is None or score < best[0]:
            best = score, np.asarray(popt), np.asarray(pcov)
    if best is None:
        return _CPMGFitResult(
            fit_status="optimizer_failed",
            fit_message=last_error,
            quality_flags=("optimizer_failed",),
            n_points=len(x_ns),
            observed_contrast=contrast,
            time_min_ns=time_min,
            time_max_ns=time_max,
            fit_method_used=fit_method,
            uncertainty_source=uncertainty_source,
        )

    _, popt, covariance_us = best
    amplitude, gamma_per_us, offset = (float(value) for value in popt)
    prediction = _decay_model_us(x_us, *popt)
    residuals = y - prediction
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / total_sum if total_sum > 0 else math.nan
    reduced_chi2 = math.nan
    if sigma is not None and len(y) > 3:
        reduced_chi2 = float(np.sum((residuals / sigma) ** 2) / (len(y) - 3))

    covariance = covariance_us.copy()
    covariance[1, :] *= 1e-3
    covariance[:, 1] *= 1e-3
    gamma_per_ns = gamma_per_us * 1e-3
    gamma_error = (
        float(np.sqrt(covariance[1, 1]))
        if covariance.shape == (3, 3)
        and np.isfinite(covariance[1, 1])
        and covariance[1, 1] >= 0
        else math.nan
    )
    relative_error = (
        gamma_error / gamma_per_ns
        if gamma_per_ns > 0 and np.isfinite(gamma_error)
        else math.nan
    )
    fitted_decay_fraction = float(1.0 - np.exp(-gamma_per_ns * (time_max - time_min)))
    flags: list[str] = []
    if fitted_decay_fraction < _MIN_DECAY_FRACTION:
        flags.append("weak_decay")
    if not np.isfinite(gamma_error) or gamma_error >= gamma_per_ns:
        flags.append("unresolved_decay")
    elif relative_error > _LARGE_RELATIVE_ERROR:
        flags.append("large_relative_error")
    if np.isfinite(r2) and r2 < _MIN_ACCEPTABLE_R2:
        flags.append("poor_model_fit")

    unresolved = "weak_decay" in flags or "unresolved_decay" in flags
    if unresolved:
        fit_status: CPMGFitStatus = "unresolved_decay"
        fit_message = "Decay is not resolved over the measured time range."
    elif flags:
        fit_status = "quality_warning"
        fit_message = "Fitting succeeded with quality warnings."
    else:
        fit_status = "ok"
        fit_message = "Fitting successful."
    t2_ns = 1.0 / gamma_per_ns if gamma_per_ns > 0 and not unresolved else math.nan
    t2_error = (
        gamma_error / gamma_per_ns**2
        if np.isfinite(t2_ns) and np.isfinite(gamma_error)
        else math.nan
    )
    return _CPMGFitResult(
        fit_status=fit_status,
        fit_message=fit_message,
        quality_flags=tuple(dict.fromkeys(flags)),
        amplitude=amplitude,
        offset=offset,
        gamma2_per_ns=gamma_per_ns,
        gamma2_error_per_ns=gamma_error,
        t2_ns=t2_ns,
        t2_error_ns=t2_error,
        covariance=covariance,
        r2=r2,
        reduced_chi2=reduced_chi2,
        n_points=len(x_ns),
        observed_contrast=contrast,
        time_min_ns=time_min,
        time_max_ns=time_max,
        relative_gamma_error=relative_error,
        fit_method_used=fit_method,
        uncertainty_source=uncertainty_source,
    )


def _fit_result_record(result: _CPMGFitResult) -> dict[str, object]:
    """Convert a fit result to JSON-compatible metadata."""
    record = asdict(result)
    record["covariance"] = (
        result.covariance.tolist() if result.covariance is not None else None
    )
    return record


# Result serialization


def _json_safe(value: object) -> object:
    """Return a recursively JSON-compatible value with nonfinite floats as null."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_payload(*, payload: CPMGResultPayload, save_dir: Path) -> None:
    """Save result metadata and numeric arrays."""
    save_dir.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, object] = {
        "schema_version": payload.get("schema_version", _SCHEMA_VERSION),
        "timestamp": payload["timestamp"],
        "targets": payload["targets"],
        "n_repeats": payload["n_repeats"],
        "frequency_range": payload["frequency_range"],
        "half_tau_range": payload.get("half_tau_range", []),
        "tau_spacing_list_ns": payload.get("half_tau_range", []),
        "time_range": payload.get("time_range"),
        "failed_fits": payload.get("failed_fits", []),
        "frequency_plans": payload.get("frequency_plans", []),
        "acquisition_orders": payload.get("acquisition_orders", []),
        "random_seed": payload.get("random_seed"),
        "random_seed_requested": payload.get("random_seed_requested"),
        "metadata": payload.get("metadata", {}),
    }
    with (save_dir / "params.json").open("w", encoding="utf-8") as fp:
        json.dump(_json_safe(serializable), fp, indent=2, allow_nan=False)
    for name in (
        "t2_matrix",
        "t2_error_matrix",
        "r2_matrix",
        "gamma2_matrix",
        "gamma2_error_matrix",
        "realized_frequency_matrix",
        "valid_frequency_mask",
    ):
        np.save(save_dir / f"{name}.npy", payload[name])
    with (save_dir / "fit_results.json").open("w", encoding="utf-8") as fp:
        json.dump(
            _json_safe(payload.get("fit_records", [])),
            fp,
            indent=2,
            allow_nan=False,
        )


def _load_optional_matrix(
    save_dir: Path,
    name: str,
    *,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Load a matrix or return an all-NaN matrix for older save directories."""
    path = save_dir / name
    if not path.exists():
        return np.full(shape, np.nan, dtype=np.float64)
    matrix = np.asarray(np.load(path), dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{name} has shape {matrix.shape}; expected {shape}.")
    return matrix


def _load_payload(save_dir: str | Path) -> CPMGResultPayload:
    """Load a result payload saved by `cpmg_noise_spectroscopy`."""
    directory = Path(save_dir)
    with (directory / "params.json").open("r", encoding="utf-8") as fp:
        params = json.load(fp)

    half_tau_range = params.get("half_tau_range", params.get("tau_spacing_list_ns"))
    if half_tau_range is None:
        raise ValueError("params.json is missing `half_tau_range`.")
    schema_version = int(params.get("schema_version", 1))
    if schema_version not in (1, _SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported CPMG save schema version {schema_version}; "
            f"expected 1 or {_SCHEMA_VERSION}."
        )

    frequency_range = [float(value) for value in params["frequency_range"]]
    n_targets = len(params["targets"])
    n_repeats = int(params["n_repeats"])
    n_frequencies = len(frequency_range)
    result_shape = (n_targets, n_repeats, n_frequencies)
    frequency_shape = (n_targets, n_frequencies)
    t2_matrix = np.asarray(np.load(directory / "t2_matrix.npy"), dtype=np.float64)
    if t2_matrix.shape != result_shape:
        raise ValueError(
            f"t2_matrix.npy has shape {t2_matrix.shape}; expected {result_shape}."
        )

    fit_records: list[dict[str, object]] = []
    fit_results_path = directory / "fit_results.json"
    if fit_results_path.exists():
        with fit_results_path.open("r", encoding="utf-8") as fp:
            loaded_records = json.load(fp)
        if not isinstance(loaded_records, list) or not all(
            isinstance(record, dict) for record in loaded_records
        ):
            raise ValueError("fit_results.json must contain a list of objects.")
        fit_records = loaded_records

    payload: CPMGResultPayload = {
        "schema_version": schema_version,
        "timestamp": str(params["timestamp"]),
        "targets": list(params["targets"]),
        "n_repeats": n_repeats,
        "frequency_range": frequency_range,
        "half_tau_range": [
            float(value) if value is not None else math.nan for value in half_tau_range
        ],
        "tau_spacing_list_ns": [
            float(value) if value is not None else math.nan for value in half_tau_range
        ],
        "time_range": (
            [float(value) for value in params["time_range"]]
            if params.get("time_range") is not None
            else None
        ),
        "t2_matrix": t2_matrix,
        "t2_error_matrix": _load_optional_matrix(
            directory, "t2_error_matrix.npy", shape=result_shape
        ),
        "r2_matrix": _load_optional_matrix(
            directory, "r2_matrix.npy", shape=result_shape
        ),
        "gamma2_matrix": _load_optional_matrix(
            directory, "gamma2_matrix.npy", shape=result_shape
        ),
        "gamma2_error_matrix": _load_optional_matrix(
            directory, "gamma2_error_matrix.npy", shape=result_shape
        ),
        "realized_frequency_matrix": _load_optional_matrix(
            directory, "realized_frequency_matrix.npy", shape=frequency_shape
        ),
        "valid_frequency_mask": np.ones(frequency_shape, dtype=np.bool_),
        "fit_records": fit_records,
        "frequency_plans": list(params.get("frequency_plans", [])),
        "acquisition_orders": list(params.get("acquisition_orders", [])),
        "random_seed": int(params.get("random_seed") or 0),
        "random_seed_requested": params.get("random_seed_requested"),
        "metadata": dict(params.get("metadata", {})),
        "data": t2_matrix,
        "failed_fits": list(params.get("failed_fits", [])),
        "save_dir": str(directory),
    }
    valid_mask_path = directory / "valid_frequency_mask.npy"
    if valid_mask_path.exists():
        valid_mask = np.asarray(np.load(valid_mask_path), dtype=np.bool_)
        if valid_mask.shape != frequency_shape:
            raise ValueError(
                f"valid_frequency_mask.npy has shape {valid_mask.shape}; "
                f"expected {frequency_shape}."
            )
        payload["valid_frequency_mask"] = valid_mask
    if schema_version == 1:
        payload["realized_frequency_matrix"] = np.tile(
            np.asarray(frequency_range, dtype=np.float64), (n_targets, 1)
        )
        gamma, gamma_error = _derived_gamma_matrices(payload)
        payload["gamma2_matrix"] = gamma
        payload["gamma2_error_matrix"] = gamma_error
    return payload


def _result_matrix(
    payload: Mapping[str, object],
    name: str,
    *,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Return one validated result matrix, defaulting to all NaNs."""
    value = payload.get(name)
    matrix = (
        np.full(shape, np.nan, dtype=np.float64)
        if value is None
        else np.asarray(value, dtype=np.float64)
    )
    if matrix.shape != shape:
        raise ValueError(f"{name} has shape {matrix.shape}; expected {shape}.")
    return matrix


def _derived_gamma_matrices(
    payload: Mapping[str, object],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Derive decay-rate matrices from T2 values and uncertainties."""
    t2 = np.asarray(payload["t2_matrix"], dtype=np.float64)
    t2_error = np.asarray(payload["t2_error_matrix"], dtype=np.float64)
    finite_t2 = np.isfinite(t2) & (t2 > 0)
    gamma = np.full_like(t2, np.nan)
    gamma[finite_t2] = 1.0 / t2[finite_t2]
    finite_error = finite_t2 & np.isfinite(t2_error) & (t2_error >= 0)
    gamma_error = np.full_like(t2, np.nan)
    gamma_error[finite_error] = t2_error[finite_error] / t2[finite_error] ** 2
    return gamma, gamma_error


def _payload_from_result_or_mapping(
    result: Result | Mapping[str, object],
) -> CPMGResultPayload:
    """Normalize supported in-memory result containers to one payload type."""
    source = result.data if isinstance(result, Result) else result
    payload = dict(source)
    if "t2_matrix" not in payload and "data" in payload:
        payload["t2_matrix"] = payload["data"]
    if "data" not in payload and "t2_matrix" in payload:
        payload["data"] = payload["t2_matrix"]
    if "half_tau_range" not in payload and "tau_spacing_list_ns" in payload:
        payload["half_tau_range"] = payload["tau_spacing_list_ns"]
    if "tau_spacing_list_ns" not in payload and "half_tau_range" in payload:
        payload["tau_spacing_list_ns"] = payload["half_tau_range"]

    required = {
        "timestamp",
        "targets",
        "n_repeats",
        "frequency_range",
        "half_tau_range",
        "tau_spacing_list_ns",
        "t2_matrix",
        "data",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"result is missing CPMG payload key(s): {missing}.")

    targets = list(cast(Collection[str], payload["targets"]))
    frequencies = np.asarray(
        cast(ArrayLike, payload["frequency_range"]), dtype=np.float64
    ).tolist()
    n_repeats = int(cast(int, payload["n_repeats"]))
    result_shape = (len(targets), n_repeats, len(frequencies))
    frequency_shape = (len(targets), len(frequencies))
    payload["targets"] = targets
    payload["frequency_range"] = frequencies
    payload["n_repeats"] = n_repeats
    payload["t2_matrix"] = _result_matrix(payload, "t2_matrix", shape=result_shape)
    payload["t2_error_matrix"] = _result_matrix(
        payload, "t2_error_matrix", shape=result_shape
    )
    payload["r2_matrix"] = _result_matrix(payload, "r2_matrix", shape=result_shape)
    payload.setdefault("schema_version", 1)
    payload.setdefault("time_range", None)
    payload.setdefault("failed_fits", [])
    payload.setdefault("save_dir", None)
    payload.setdefault("fit_records", [])
    payload.setdefault("frequency_plans", [])
    payload.setdefault("acquisition_orders", [])
    payload.setdefault("metadata", {})
    payload.setdefault("random_seed", 0)
    payload.setdefault("random_seed_requested", None)
    payload.setdefault(
        "realized_frequency_matrix",
        np.tile(np.asarray(frequencies, dtype=np.float64), (len(targets), 1)),
    )
    payload.setdefault("valid_frequency_mask", np.ones(frequency_shape, dtype=np.bool_))
    derived_gamma, derived_gamma_error = _derived_gamma_matrices(payload)
    payload.setdefault("gamma2_matrix", derived_gamma)
    payload.setdefault("gamma2_error_matrix", derived_gamma_error)
    payload["gamma2_matrix"] = _result_matrix(
        payload, "gamma2_matrix", shape=result_shape
    )
    payload["gamma2_error_matrix"] = _result_matrix(
        payload, "gamma2_error_matrix", shape=result_shape
    )
    realized = np.asarray(payload["realized_frequency_matrix"], dtype=np.float64)
    if realized.shape != frequency_shape:
        raise ValueError(
            "realized_frequency_matrix has shape "
            f"{realized.shape}; expected {frequency_shape}."
        )
    valid = np.asarray(payload["valid_frequency_mask"], dtype=np.bool_)
    if valid.shape != frequency_shape:
        raise ValueError(
            f"valid_frequency_mask has shape {valid.shape}; expected {frequency_shape}."
        )
    payload["realized_frequency_matrix"] = realized
    payload["valid_frequency_mask"] = valid
    payload["data"] = payload["t2_matrix"]
    return cast(CPMGResultPayload, payload)


# Result visualization


def _resolve_plot_options(
    *,
    show_t1: bool,
    save_image: bool | None,
    deprecated_options: dict[str, Any],
) -> tuple[bool, bool]:
    """Resolve plotting defaults and aliases from the first contrib API."""
    if "show_T1" in deprecated_options:
        legacy_show_t1 = deprecated_options.pop("show_T1")
        if legacy_show_t1 is not None:
            warnings.warn(
                "`show_T1` is deprecated in `plot_cpmg_results`; use `show_t1`.",
                DeprecationWarning,
                stacklevel=3,
            )
            if show_t1 not in (False, legacy_show_t1):
                raise ValueError(
                    "`show_T1` conflicts with `show_t1`. Provide only `show_t1`."
                )
            show_t1 = bool(legacy_show_t1)
    if "save_fig" in deprecated_options:
        legacy_save_image = deprecated_options.pop("save_fig")
        if legacy_save_image is not None:
            warnings.warn(
                "`save_fig` is deprecated in `plot_cpmg_results`; use `save_image`.",
                DeprecationWarning,
                stacklevel=3,
            )
            if save_image is not None and save_image != legacy_save_image:
                raise ValueError(
                    "`save_fig` conflicts with `save_image`. Provide only `save_image`."
                )
            save_image = bool(legacy_save_image)
    if deprecated_options:
        joined = ", ".join(f"`{key}`" for key in sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword argument(s): {joined}")
    return show_t1, False if save_image is None else save_image


def _resolve_plot_source(
    result: Result | Mapping[str, object] | None,
    save_dir: str | Path | None,
) -> tuple[CPMGResultPayload, Path | None]:
    """Load and normalize one plotting payload and its output directory."""
    if result is None:
        if save_dir is None:
            raise ValueError("Either result or save_dir must be provided.")
        return _load_payload(save_dir), Path(save_dir)
    payload = _payload_from_result_or_mapping(result)
    directory = Path(save_dir) if save_dir is not None else None
    if directory is None and payload["save_dir"] is not None:
        directory = Path(payload["save_dir"])
    return payload, directory


def _quality_groups(
    payload: CPMGResultPayload,
    *,
    target: str,
    gamma2: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], list[set[str]]]:
    """Classify repeat fits for one target as reliable or warning-quality."""
    n_frequencies = gamma2.shape[1]
    reliable = np.isfinite(gamma2) & (gamma2 > 0)
    warning = np.zeros_like(reliable, dtype=np.bool_)
    warning_flags: list[set[str]] = [set() for _ in range(n_frequencies)]
    for record in payload["fit_records"]:
        frequency_index = record.get("requested_frequency_index")
        repeat_index = record.get("repeat_index")
        flags = record.get("quality_flags")
        is_warning = bool(flags) or record.get("fit_status") not in (None, "ok")
        if (
            record.get("target") != target
            or not is_warning
            or not isinstance(frequency_index, int)
            or not isinstance(repeat_index, int)
            or not 0 <= frequency_index < n_frequencies
            or not 0 <= repeat_index < gamma2.shape[0]
        ):
            continue
        reliable[repeat_index, frequency_index] = False
        warning[repeat_index, frequency_index] = True
        if isinstance(flags, (list, tuple)):
            warning_flags[frequency_index].update(str(flag) for flag in flags)
        elif flags:
            warning_flags[frequency_index].add(str(flags))
    return reliable, warning, warning_flags


def _summarize_repeats(
    gamma2: NDArray[np.float64],
    gamma2_errors: NDArray[np.float64],
    selected: NDArray[np.bool_],
) -> _RepeatSummary:
    """Aggregate selected repeat fits without mixing quality groups."""
    n_frequencies = gamma2.shape[1]
    averages = np.full(n_frequencies, np.nan, dtype=np.float64)
    errors = np.full(n_frequencies, np.nan, dtype=np.float64)
    counts = np.zeros(n_frequencies, dtype=np.int64)
    for frequency_index in range(n_frequencies):
        included = (
            selected[:, frequency_index]
            & np.isfinite(gamma2[:, frequency_index])
            & (gamma2[:, frequency_index] > 0)
        )
        values = gamma2[included, frequency_index]
        counts[frequency_index] = len(values)
        if len(values) == 0:
            continue
        averages[frequency_index] = float(np.mean(values))
        if len(values) >= 2:
            errors[frequency_index] = float(
                np.std(values, ddof=1) / np.sqrt(len(values))
            )
            continue
        candidates = gamma2_errors[included, frequency_index]
        candidates = candidates[np.isfinite(candidates) & (candidates >= 0)]
        if len(candidates):
            errors[frequency_index] = float(candidates[0])
    return _RepeatSummary(averages, errors, counts)


def _display_values(
    summary: _RepeatSummary,
    mode: _DisplayedMode,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert decay-rate summaries to the requested display units."""
    if mode == "gamma2":
        return summary.gamma2 * 1e3, summary.errors * 1e3
    finite = np.isfinite(summary.gamma2) & (summary.gamma2 > 0)
    values = np.full_like(summary.gamma2, np.nan)
    errors = np.full_like(summary.errors, np.nan)
    values[finite] = 1.0 / summary.gamma2[finite] * 1e-3
    errors[finite] = summary.errors[finite] / summary.gamma2[finite] ** 2 * 1e-3
    return values, errors


def _add_target_traces(
    figure: go.Figure,
    *,
    target: str,
    frequencies_mhz: NDArray[np.float64],
    gamma2: NDArray[np.float64],
    gamma2_errors: NDArray[np.float64],
    payload: CPMGResultPayload,
    mode: CPMGSummaryMode,
) -> None:
    """Add reliable and warning-quality summary traces for one target."""
    reliable, warning, warning_flags = _quality_groups(
        payload, target=target, gamma2=gamma2
    )
    summaries = {
        "reliable": _summarize_repeats(gamma2, gamma2_errors, reliable),
        "warning": _summarize_repeats(gamma2, gamma2_errors, warning),
    }
    displayed_modes: tuple[_DisplayedMode, ...] = (
        ("t2", "gamma2") if mode == "both" else (mode,)
    )
    plot_indices = np.argsort(frequencies_mhz, kind="stable")
    for displayed_mode in displayed_modes:
        for quality, summary in summaries.items():
            finite = np.isfinite(summary.gamma2) & (summary.gamma2 > 0)
            if not np.any(finite):
                continue
            values, errors = _display_values(summary, displayed_mode)
            base_name = (
                f"{target} (T2)"
                if mode == "both" and displayed_mode == "t2"
                else f"{target} (Gamma2)"
                if displayed_mode == "gamma2"
                else target
            )
            value_hover = (
                "T2_CPMG=%{y:.6g} us<br>"
                if displayed_mode == "t2"
                else "Gamma_2_CPMG=%{y:.6g} 1/us<br>"
            )
            yaxis = "y2" if mode == "both" and displayed_mode == "gamma2" else "y"
            if quality == "reliable":
                figure.add_trace(
                    go.Scatter(
                        x=frequencies_mhz[plot_indices],
                        y=values[plot_indices],
                        mode="markers+lines",
                        name=base_name,
                        error_y={
                            "type": "data",
                            "array": errors[plot_indices],
                            "visible": bool(np.any(np.isfinite(errors))),
                        },
                        marker={"symbol": "circle"},
                        customdata=summary.counts[plot_indices],
                        hovertemplate=(
                            "frequency=%{x:.6g} MHz<br>"
                            f"{value_hover}"
                            "reliable repeats=%{customdata}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                        connectgaps=False,
                        yaxis=yaxis,
                    )
                )
                continue
            warning_indices = plot_indices[finite[plot_indices]]
            flag_labels = [
                ", ".join(sorted(warning_flags[index])) or "fit status warning"
                for index in warning_indices
            ]
            customdata = np.empty((len(warning_indices), 2), dtype=object)
            customdata[:, 0] = summary.counts[warning_indices]
            customdata[:, 1] = flag_labels
            figure.add_trace(
                go.Scatter(
                    x=frequencies_mhz[warning_indices],
                    y=values[warning_indices],
                    mode="markers",
                    name=f"{base_name} (quality warning)",
                    error_y={
                        "type": "data",
                        "array": errors[warning_indices],
                        "visible": bool(np.any(np.isfinite(errors[warning_indices]))),
                    },
                    marker={
                        "symbol": "diamond-open",
                        "size": 10,
                        "color": "#d62728",
                        "line": {"width": 2},
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "frequency=%{x:.6g} MHz<br>"
                        f"{value_hover}"
                        "warning repeats=%{customdata[0]}<br>"
                        "quality flags=%{customdata[1]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                    yaxis=yaxis,
                )
            )


def _add_t1_reference(figure: go.Figure, exp: Experiment, target: str) -> None:
    """Add a target's `2*T1` reference when calibration data are available."""
    try:
        t1_values = exp.ctx.system_manager.config_loader.load_param_data("t1")
        if target in t1_values:
            figure.add_hline(
                y=2 * float(t1_values[target]) * 1e-3,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"2*T1 of {target}",
                annotation_position="top left",
            )
    except Exception as exc:
        warnings.warn(
            f"Could not load T1 reference for {target}: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )


def _configure_summary_layout(
    figure: go.Figure,
    *,
    payload: CPMGResultPayload,
    mode: CPMGSummaryMode,
) -> None:
    """Apply shared axes, legend, and title settings to a summary figure."""
    yaxis_title = (
        "T2_CPMG = 1 / mean(Gamma_2) (us)"
        if mode in ("t2", "both")
        else "Mean Gamma_2_CPMG (1/us)"
    )
    figure.update_layout(
        title={
            "text": "CPMG Noise Spectroscopy",
            "subtitle": {
                "text": (
                    f"timestamp: {payload['timestamp']}, "
                    f"repetition: {payload['n_repeats']} times"
                )
            },
        },
        xaxis={"title": "Effective CPMG Frequency (MHz)", "showgrid": True},
        yaxis={"title": yaxis_title, "showgrid": True, "rangemode": "tozero"},
        legend={
            "x": 1.12 if mode == "both" else 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        margin={"r": 200 if mode == "both" else 140},
        xaxis_type="linear",
        yaxis_type="linear",
        width=1000,
        height=600,
    )
    if mode == "both":
        figure.update_layout(
            yaxis2={
                "title": "Mean Gamma_2_CPMG (1/us)",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "rangemode": "tozero",
            }
        )


def plot_cpmg_results(
    exp: Experiment,
    result: Result | Mapping[str, object] | None = None,
    *,
    save_dir: str | Path | None = None,
    mode: CPMGSummaryMode | None = None,
    show_t1: bool = False,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> go.Figure:
    """
    Plot aggregated CPMG noise spectroscopy results.

    Fits without quality flags form the connected summary series. Finite fits
    carrying quality warnings are excluded from that average and shown as
    separate red open diamonds without connecting lines.

    Parameters
    ----------
    exp
        Experiment instance used to load optional T1 reference values.
    result
        In-memory result returned by `cpmg_noise_spectroscopy`, or a mapping
        with the same payload keys.
    save_dir
        Directory containing saved result files, or an output directory when
        `result` is provided.
    mode
        Summary mode: `"t2"`, `"gamma2"`, or `"both"`. Defaults to `"both"`.
    show_t1
        Whether to overlay `2*T1` references. Defaults to `False`.
    plot
        Whether to display the figure. Defaults to `True`.
    save_image
        Whether to save the summary figure. Defaults to `False`.

    Returns
    -------
    plotly.graph_objects.Figure
        Summary figure.

    Raises
    ------
    ValueError
        Raised when the mode, input source, or result payload is invalid.
    """
    show_t1, save_image = _resolve_plot_options(
        show_t1=show_t1,
        save_image=save_image,
        deprecated_options=deprecated_options,
    )
    mode = "both" if mode is None else mode
    plot = True if plot is None else plot
    if mode not in ("t2", "gamma2", "both"):
        raise ValueError("mode must be 't2', 'gamma2', or 'both'.")
    if mode == "gamma2" and show_t1:
        warnings.warn(
            "`show_t1` is ignored when mode='gamma2'.",
            RuntimeWarning,
            stacklevel=2,
        )
        show_t1 = False

    payload, directory = _resolve_plot_source(result, save_dir)
    if not payload["targets"]:
        raise ValueError("result contains no targets.")
    realized = np.asarray(payload["realized_frequency_matrix"], dtype=np.float64)
    gamma2 = np.asarray(payload["gamma2_matrix"], dtype=np.float64)
    gamma2_errors = np.asarray(payload["gamma2_error_matrix"], dtype=np.float64)
    figure = viz.make_figure()
    for target_index, target in enumerate(payload["targets"]):
        _add_target_traces(
            figure,
            target=target,
            frequencies_mhz=realized[target_index] * 1e3,
            gamma2=gamma2[target_index],
            gamma2_errors=gamma2_errors[target_index],
            payload=payload,
            mode=mode,
        )
        if show_t1:
            _add_t1_reference(figure, exp, target)
    _configure_summary_layout(figure, payload=payload, mode=mode)

    if save_image:
        if directory is None:
            directory = Path(_DEFAULT_SAVE_DIR) / payload["timestamp"]
        directory.mkdir(parents=True, exist_ok=True)
        figure.write_image(directory / "cpmg_noise_spectroscopy_summary.png")
    if plot:
        figure.show()
    return figure


def _plot_cpmg_decay(
    *,
    times_ns: NDArray[np.float64],
    values: NDArray[np.float64],
    fit: _CPMGFitResult,
    target: str,
    frequency_ghz: float,
    xaxis_type: AxisType,
) -> go.Figure:
    """Create one CPMG decay figure from numerical data."""
    figure = viz.make_figure()
    figure.add_trace(
        go.Scatter(x=times_ns * 1e-3, y=values, mode="markers", name="Data")
    )
    if np.all(np.isfinite([fit.amplitude, fit.gamma2_per_ns, fit.offset])):
        fine_ns = np.linspace(float(np.min(times_ns)), float(np.max(times_ns)), 500)
        fitted = fit.amplitude * np.exp(-fit.gamma2_per_ns * fine_ns) + fit.offset
        figure.add_trace(
            go.Scatter(x=fine_ns * 1e-3, y=fitted, mode="lines", name="Fit")
        )
    figure.update_layout(
        title=f"CPMG decay: {target}, {frequency_ghz * 1e3:.6g} MHz",
        xaxis_title="Effective decay time (us)",
        yaxis_title="Normalized signal",
        xaxis_type=xaxis_type,
        yaxis_type="linear",
    )
    return figure


def _plot_cpmg_raw_data_heatmap(
    *,
    payload: CPMGResultPayload,
    target: str,
    target_index: int,
) -> go.Figure:
    """Plot pre-fit values at each target's exact realized frequency and time."""
    figure = viz.make_figure()
    figure.update_layout(
        title={
            "text": f"CPMG measured data before fitting: {target}",
            "subtitle": {
                "text": "Exact measured times; repeats at identical points are averaged"
            },
        },
        xaxis={"title": "Effective CPMG Frequency (MHz)"},
        yaxis={"title": "Effective decay time (us)", "type": "log"},
        width=900,
        height=600,
    )
    frequencies = np.asarray(payload["realized_frequency_matrix"], dtype=np.float64)[
        target_index
    ]
    valid_indices = np.flatnonzero(np.isfinite(frequencies) & (frequencies > 0))
    if len(valid_indices) == 0:
        figure.add_annotation(
            text="No valid CPMG measurements",
            showarrow=False,
        )
        return figure
    ordered_indices = valid_indices[np.argsort(frequencies[valid_indices])]
    valid_frequency_indices = {int(index) for index in ordered_indices}
    value_sums: dict[tuple[int, float], float] = {}
    value_counts: dict[tuple[int, float], int] = {}
    for record in payload["fit_records"]:
        frequency_index = record.get("requested_frequency_index")
        if (
            record.get("target") != target
            or not isinstance(frequency_index, int)
            or frequency_index not in valid_frequency_indices
        ):
            continue
        times = np.asarray(record.get("actual_decay_times_ns", []), dtype=np.float64)
        values = np.asarray(record.get("P_values", []), dtype=np.float64)
        if times.ndim != 1 or values.ndim != 1 or times.shape != values.shape:
            continue
        finite = np.isfinite(times) & (times > 0) & np.isfinite(values)
        for time_ns, value in zip(times[finite], values[finite], strict=True):
            coordinate = (frequency_index, float(time_ns))
            value_sums[coordinate] = value_sums.get(coordinate, 0.0) + float(value)
            value_counts[coordinate] = value_counts.get(coordinate, 0) + 1
    if not value_sums:
        figure.add_annotation(
            text="No valid CPMG measurements",
            showarrow=False,
        )
        return figure

    frequency_order = {
        int(frequency_index): order
        for order, frequency_index in enumerate(ordered_indices)
    }
    coordinates = sorted(
        value_sums,
        key=lambda coordinate: (frequency_order[coordinate[0]], coordinate[1]),
    )
    x_values = [frequencies[index] * 1e3 for index, _ in coordinates]
    y_values = [time_ns * 1e-3 for _, time_ns in coordinates]
    mean_values = [
        value_sums[coordinate] / value_counts[coordinate] for coordinate in coordinates
    ]
    repeat_counts = [value_counts[coordinate] for coordinate in coordinates]
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            name="Measured data",
            showlegend=False,
            customdata=np.column_stack((mean_values, repeat_counts)),
            marker={
                "symbol": "square",
                "size": 14,
                "color": mean_values,
                "cmin": 0,
                "cmax": 1,
                "colorscale": "Viridis",
                "colorbar": {"title": "Normalized signal P"},
                "showscale": True,
                "line": {"width": 0},
            },
            hovertemplate=(
                "frequency=%{x:.6g} MHz<br>"
                "actual time=%{y:.6g} us<br>"
                "mean P=%{customdata[0]:.6g}<br>"
                "measurements=%{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    return figure


# Experiment execution


def _resolve_experiment_deprecated_options(
    *,
    plot_mode: CPMGSummaryMode | None,
    save_image: bool | None,
    save_fit_images: bool | None,
    save_data: bool | None,
    deprecated_options: dict[str, Any],
) -> tuple[CPMGSummaryMode | None, bool | None, bool | None, bool | None]:
    """Resolve aliases from the first CPMG contrib implementation."""
    plot_mode = _resolve_deprecated_alias(
        options=deprecated_options,
        deprecated_name="plot_mode",
        replacement_name="summary_mode",
        current_value=plot_mode,
        function_name="cpmg_noise_spectroscopy",
    )
    save_image = _resolve_deprecated_alias(
        options=deprecated_options,
        deprecated_name="save_cpmg_fig",
        replacement_name="save_image",
        current_value=save_image,
        function_name="cpmg_noise_spectroscopy",
    )
    save_fit_images = _resolve_deprecated_alias(
        options=deprecated_options,
        deprecated_name="save_t2_image",
        replacement_name="save_fit_images",
        current_value=save_fit_images,
        function_name="cpmg_noise_spectroscopy",
    )
    save_data = _resolve_deprecated_alias(
        options=deprecated_options,
        deprecated_name="save_cpmg_data",
        replacement_name="save_data",
        current_value=save_data,
        function_name="cpmg_noise_spectroscopy",
    )
    return plot_mode, save_image, save_fit_images, save_data


def _sweep_parameter(exp: Experiment, **kwargs: Any) -> Any:
    """Call the qubex sweep API through the service when available."""
    measurement_service = getattr(exp, "measurement_service", None)
    sweep_parameter = getattr(measurement_service, "sweep_parameter", None)
    if callable(sweep_parameter):
        return sweep_parameter(**kwargs)
    return exp.sweep_parameter(**kwargs)


def _create_measurement_plan(
    exp: Experiment,
    *,
    targets: list[str],
    requested_frequencies: NDArray[np.float64],
    requested_times: NDArray[np.float64],
    pi_pulses: Mapping[str, Waveform] | None,
    min_pi_pulses: int,
    max_pulse_duty_ratio: float,
    min_fit_points: int,
    pi_pulse_pattern: CPMGPattern,
) -> _MeasurementPlan:
    """Create all target-specific frequency and pulse-count plans."""
    _validate_rabi_params(exp, targets)
    sampling_period = _resolve_sampling_period(exp)
    resolved_pi_pulses = {
        target: _resolve_pi_pulse(exp, target, pi_pulses) for target in targets
    }
    frequency_plans: list[list[_FrequencyPlan]] = []
    n_pi_values: list[dict[int, NDArray[np.int64]]] = []
    for target in targets:
        plan = _make_frequency_plan(
            requested_frequencies,
            sampling_period=sampling_period,
            pi_duration=float(resolved_pi_pulses[target].duration),
            max_pulse_duty_ratio=max_pulse_duty_ratio,
        )
        target_counts: dict[int, NDArray[np.int64]] = {}
        for position, point in enumerate(plan):
            if point.status != "valid":
                continue
            if point.center_spacing is None:
                raise RuntimeError("A valid frequency point has no center spacing.")
            counts = _n_pi_from_time_range(
                requested_times,
                center_spacing_ns=point.center_spacing,
                pattern=pi_pulse_pattern,
                min_pi_pulses=min_pi_pulses,
            )
            if len(counts) < min_fit_points:
                plan[position] = replace(
                    point,
                    status="skipped",
                    reason="insufficient_time_points",
                )
                continue
            target_counts[point.requested_index] = counts
        frequency_plans.append(plan)
        n_pi_values.append(target_counts)

    if not any(n_pi_values):
        raise ValueError(
            "No valid target-frequency points remain after timing, duty-ratio, "
            "and pulse-count validation."
        )
    return _MeasurementPlan(
        targets=targets,
        requested_frequencies=requested_frequencies,
        requested_times=requested_times,
        sampling_period=sampling_period,
        pi_pulses=resolved_pi_pulses,
        frequency_plans=frequency_plans,
        n_pi_values=n_pi_values,
    )


def _create_acquisition_data(
    plan: _MeasurementPlan,
    *,
    n_repeats: int,
) -> _AcquisitionData:
    """Allocate result matrices and record valid realized frequencies."""
    shape = (len(plan.targets), n_repeats, len(plan.requested_frequencies))
    frequency_shape = (len(plan.targets), len(plan.requested_frequencies))
    realized_frequencies = np.full(frequency_shape, np.nan, dtype=np.float64)
    valid_frequencies = np.zeros(frequency_shape, dtype=np.bool_)
    for target_index, target_plan in enumerate(plan.frequency_plans):
        for point in target_plan:
            if point.status == "valid" and point.frequency is not None:
                realized_frequencies[target_index, point.requested_index] = (
                    point.frequency
                )
                valid_frequencies[target_index, point.requested_index] = True
            elif point.reason is not None:
                logger.warning(
                    "Skipping CPMG point for %s at %.9g GHz: %s",
                    plan.targets[target_index],
                    point.requested_frequency,
                    point.reason,
                )
    return _AcquisitionData(
        t2=np.full(shape, np.nan, dtype=np.float64),
        t2_error=np.full(shape, np.nan, dtype=np.float64),
        gamma2=np.full(shape, np.nan, dtype=np.float64),
        gamma2_error=np.full(shape, np.nan, dtype=np.float64),
        r2=np.full(shape, np.nan, dtype=np.float64),
        realized_frequencies=realized_frequencies,
        valid_frequencies=valid_frequencies,
        fit_records=[],
        failed_fits=[],
        acquisition_orders=[[[] for _ in range(n_repeats)] for _ in plan.targets],
    )


def _measure_cpmg_point(
    exp: Experiment,
    *,
    plan: _MeasurementPlan,
    options: _AcquisitionOptions,
    target_index: int,
    repeat_index: int,
    frequency_index: int,
) -> _PointMeasurement:
    """Acquire and fit one target-frequency-repeat point."""
    target = plan.targets[target_index]
    point = plan.frequency_plans[target_index][frequency_index]
    n_pi_values = plan.n_pi_values[target_index][frequency_index]
    half_tau = point.half_tau
    center_spacing = point.center_spacing
    frequency = point.frequency
    if half_tau is None or center_spacing is None or frequency is None:
        raise RuntimeError("A valid frequency point has incomplete timing.")

    actual_times = n_pi_values.astype(np.float64) * center_spacing
    acquired_at = datetime.now(timezone.utc).isoformat()
    measurement = _sweep_parameter(
        exp,
        sequence=lambda n_pi: _build_cpmg_sequence(
            exp,
            target=target,
            half_tau=half_tau,
            n_pi=int(n_pi),
            pattern=options.pi_pulse_pattern,
            pi_pulses=plan.pi_pulses,
            sampling_period=plan.sampling_period,
        ),
        sweep_range=n_pi_values,
        n_shots=options.n_shots,
        shot_interval=options.shot_interval,
        plot=False,
        enable_tqdm=False,
        title="CPMG decay",
        xlabel="Number of pi pulses",
        ylabel="Measured value",
        xaxis_type="linear",
    )
    probability = np.asarray(
        0.5 * (1.0 + measurement.data[target].normalized),
        dtype=np.float64,
    )
    fit = _fit_cpmg_decay(
        actual_times,
        probability,
        min_fit_points=options.min_fit_points,
    )
    pulse = plan.pi_pulses[target]
    record = _fit_result_record(fit)
    record.update(
        {
            "target": target,
            "target_index": target_index,
            "repeat_index": repeat_index,
            "requested_frequency_index": frequency_index,
            "f_requested_GHz": point.requested_frequency,
            "f_real_GHz": frequency,
            "pi_duration_ns": float(pulse.duration),
            "pi_pulse_name": pulse.name,
            "pi_pulse_type": type(pulse).__name__,
            "half_tau_ns": half_tau,
            "pi_center_spacing_ns": center_spacing,
            "pulse_duty_ratio": point.pulse_duty_ratio,
            "sampling_period_ns": plan.sampling_period,
            "n_pi_values": n_pi_values.tolist(),
            "actual_decay_times_ns": actual_times.tolist(),
            "P_values": probability.tolist(),
            "n_shots": options.n_shots,
            "shot_interval_ns": options.shot_interval,
            "pi_pulse_pattern": options.pi_pulse_pattern,
            "acquired_at": acquired_at,
        }
    )
    return _PointMeasurement(
        fit=fit,
        frequency_ghz=frequency,
        record=record,
        times_ns=actual_times,
        probabilities=probability,
    )


def _store_point_result(
    data: _AcquisitionData,
    measurement: _PointMeasurement,
    *,
    target_index: int,
    repeat_index: int,
    frequency_index: int,
) -> None:
    """Store one fit in the fixed-index result matrices."""
    fit = measurement.fit
    data.fit_records.append(measurement.record)
    if fit.fit_status in ("ok", "quality_warning"):
        index = (target_index, repeat_index, frequency_index)
        data.t2[index] = fit.t2_ns
        data.t2_error[index] = fit.t2_error_ns
        data.gamma2[index] = fit.gamma2_per_ns
        data.gamma2_error[index] = fit.gamma2_error_per_ns
        data.r2[index] = fit.r2
        return
    data.failed_fits.append(
        {
            "target": measurement.record["target"],
            "repeat_index": repeat_index,
            "requested_frequency_index": frequency_index,
            "frequency": measurement.record["f_real_GHz"],
            "status": fit.fit_status,
            "message": fit.fit_message,
        }
    )


def _save_decay_figure(
    measurement: _PointMeasurement,
    *,
    target: str,
    repeat_index: int,
    output_dir: Path,
    xaxis_type: AxisType,
) -> None:
    """Save one measured decay and its fitted curve."""
    frequency = measurement.frequency_ghz
    figure = _plot_cpmg_decay(
        times_ns=measurement.times_ns,
        values=measurement.probabilities,
        fit=measurement.fit,
        target=target,
        frequency_ghz=frequency,
        xaxis_type=xaxis_type,
    )
    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    figure.write_image(
        target_dir
        / f"{target}_freq_{frequency * 1e3:.6f}_MHz_rep_{repeat_index + 1}.png"
    )


def _run_acquisition(
    exp: Experiment,
    *,
    plan: _MeasurementPlan,
    options: _AcquisitionOptions,
    output_dir: Path,
) -> tuple[_AcquisitionData, int]:
    """Acquire every planned point and return populated result matrices."""
    data = _create_acquisition_data(plan, n_repeats=options.n_repeats)
    seed_used = (
        options.random_seed
        if options.random_seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    rng = np.random.default_rng(seed_used)
    repeat_iterable = tqdm(
        range(options.n_repeats),
        desc="CPMG repetitions",
        disable=not options.enable_tqdm,
    )
    for repeat_index in repeat_iterable:
        for target_index, target in enumerate(plan.targets):
            order = np.flatnonzero(data.valid_frequencies[target_index])
            if options.frequency_order == "random":
                rng.shuffle(order)
            else:
                order = order[
                    np.argsort(data.realized_frequencies[target_index, order])
                ]
            data.acquisition_orders[target_index][repeat_index] = order.tolist()
            for raw_frequency_index in tqdm(
                order,
                total=len(order),
                desc=f"{target} frequencies",
                leave=False,
                disable=not options.enable_tqdm,
            ):
                frequency_index = int(raw_frequency_index)
                measurement = _measure_cpmg_point(
                    exp,
                    plan=plan,
                    options=options,
                    target_index=target_index,
                    repeat_index=repeat_index,
                    frequency_index=frequency_index,
                )
                _store_point_result(
                    data,
                    measurement,
                    target_index=target_index,
                    repeat_index=repeat_index,
                    frequency_index=frequency_index,
                )
                if options.save_fit_images:
                    _save_decay_figure(
                        measurement,
                        target=target,
                        repeat_index=repeat_index,
                        output_dir=output_dir,
                        xaxis_type=options.xaxis_type,
                    )
    return data, seed_used


def _build_result_payload(
    *,
    plan: _MeasurementPlan,
    data: _AcquisitionData,
    options: _AcquisitionOptions,
    timestamp: str,
    seed_used: int,
    save_dir: str | None,
    min_pi_pulses: int,
    max_pulse_duty_ratio: float,
) -> CPMGResultPayload:
    """Assemble the public numerical payload from planning and acquisition data."""
    first_plan_half_taus = [
        float(point.half_tau) if point.half_tau is not None else math.nan
        for point in plan.frequency_plans[0]
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": timestamp,
        "targets": plan.targets,
        "n_repeats": options.n_repeats,
        "frequency_range": plan.requested_frequencies.tolist(),
        "realized_frequency_matrix": data.realized_frequencies,
        "valid_frequency_mask": data.valid_frequencies,
        "half_tau_range": first_plan_half_taus,
        "tau_spacing_list_ns": first_plan_half_taus,
        "time_range": plan.requested_times.tolist(),
        "t2_matrix": data.t2,
        "t2_error_matrix": data.t2_error,
        "gamma2_matrix": data.gamma2,
        "gamma2_error_matrix": data.gamma2_error,
        "r2_matrix": data.r2,
        "fit_records": data.fit_records,
        "frequency_plans": [
            [asdict(point) for point in target_plan]
            for target_plan in plan.frequency_plans
        ],
        "acquisition_orders": data.acquisition_orders,
        "random_seed": seed_used,
        "random_seed_requested": options.random_seed,
        "metadata": {
            "sampling_period_ns": plan.sampling_period,
            "min_pi_pulses": min_pi_pulses,
            "max_pulse_duty_ratio": max_pulse_duty_ratio,
            "min_fit_points": options.min_fit_points,
            "frequency_order": options.frequency_order,
            "pi_pulse_pattern": options.pi_pulse_pattern,
            "n_shots": options.n_shots,
            "shot_interval_ns": options.shot_interval,
            "frequency_definition": "1 / (2 * pi_center_spacing)",
        },
        "data": data.t2,
        "failed_fits": data.failed_fits,
        "save_dir": save_dir,
    }


def _render_result_figures(
    exp: Experiment,
    *,
    payload: CPMGResultPayload,
    output_dir: Path,
    summary_mode: CPMGSummaryMode,
    show_t1: bool,
    plot: bool,
    save_image: bool,
) -> tuple[go.Figure, dict[str, go.Figure]]:
    """Create result figures and optionally display or save them."""
    summary = plot_cpmg_results(
        exp,
        payload,
        save_dir=output_dir,
        mode=summary_mode,
        show_t1=show_t1,
        plot=plot,
        save_image=save_image,
    )
    figures = {"summary": summary}
    for target_index, target in enumerate(payload["targets"]):
        heatmap = _plot_cpmg_raw_data_heatmap(
            payload=payload,
            target=target,
            target_index=target_index,
        )
        figures[f"{target}_raw_data_heatmap"] = heatmap
        if save_image:
            heatmap.write_image(output_dir / f"{target}_raw_data_heatmap.png")
        if plot:
            heatmap.show()
    return summary, figures


def cpmg_noise_spectroscopy(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    frequency_range: ArrayLike | None = None,
    time_range: ArrayLike | None = None,
    min_pi_pulses: int = _DEFAULT_MIN_PI_PULSES,
    max_pulse_duty_ratio: float = _DEFAULT_MAX_PULSE_DUTY_RATIO,
    min_fit_points: int = _DEFAULT_MIN_FIT_POINTS,
    frequency_order: Literal["random", "ascending"] = "random",
    random_seed: int | None = None,
    pi_pulse_pattern: CPMGPattern | None = None,
    pi_pulses: Mapping[str, Waveform] | None = None,
    n_repeats: int | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    save_fit_images: bool | None = None,
    save_data: bool | None = None,
    save_dir: str | Path | None = None,
    summary_mode: CPMGSummaryMode | None = None,
    show_t1: bool = False,
    xaxis_type: AxisType | None = None,
    enable_tqdm: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run target-specific frequency-swept CPMG noise spectroscopy.

    The reported effective frequency is the approximate frequency label based
    on the realized pi-pulse center spacing, `f = 1 / (2 * D)`, where
    `D = 2 * half_tau + pi_duration`. It is not an exact filter-function peak
    and does not model sensitivity during finite pulses or higher harmonics.

    Parameters
    ----------
    exp
        Experiment instance used for pulse generation and measurement.
    targets
        Target qubit labels. Defaults to all labels in the experiment context.
    frequency_range
        Positive requested CPMG frequencies in GHz. Defaults to 20 linearly
        spaced points from 0.0005 to 0.004 GHz. Hardware timing quantization is
        applied per target, and the realized frequencies are recorded in the
        result payload.
    time_range
        Requested target decay times in ns. Each value is converted to the
        nearest valid pi-pulse count independently for every target and
        frequency. For `"+--+"` and `"+-+-"`, counts are rounded to the
        nearest complete pattern cycle. Defaults to 31 geometrically spaced
        points from 200 to 200000 ns.
    min_pi_pulses
        Minimum accepted pi-pulse count. Defaults to 4.
    max_pulse_duty_ratio
        Maximum `pi_duration / D`. Defaults to 0.25.
    min_fit_points
        Minimum number of finite points required for fitting. Defaults to 6.
    frequency_order
        Acquisition order, either `"random"` or `"ascending"`. Ascending order
        uses each target's realized frequencies. Defaults to `"random"`.
    random_seed
        Optional nonnegative seed for reproducible target/repeat acquisition
        orders. When omitted, a generated seed is stored in the result payload.
    pi_pulse_pattern
        Sign pattern: `"++++"`, `"+--+"`, or `"+-+-"`. Pulse counts derived
        from `time_range` are rounded to the nearest complete pattern cycle;
        ties are rounded toward the larger count. Defaults to `"++++"`.
    pi_pulses
        Optional per-target CPMG pi waveforms. Values are used as provided and
        are not phase-shifted. When omitted, an automatically resolved X pi
        pulse is shifted by pi/2 to produce the CPMG pulse.
    n_repeats
        Number of measurements per target and frequency. Defaults to 1.
    n_shots
        Number of shots per sweep point. Defaults to the experiment-wide
        measurement setting.
    shot_interval
        Measurement interval in ns. Defaults to the experiment-wide
        measurement setting.
    plot
        Whether to display the summary figure and pre-fit data heatmaps. The
        figures are generated and returned regardless of this option. Defaults
        to `True`.
    save_image
        Whether to save the summary figure and pre-fit data heatmaps as image
        files. The figures are generated and returned regardless of this
        option. Defaults to `False`.
    save_fit_images
        Whether to render and save individual decay figures. Defaults to
        `False`.
    save_data
        Whether to save schema-v2 numerical result files. This option does not
        control figure generation or image saving. Defaults to `True`.
    save_dir
        Base output directory. Defaults to `"cpmg_noise_spectroscopy"`.
    summary_mode
        Summary mode, `"t2"`, `"gamma2"`, or `"both"`. Defaults to `"both"`.
    show_t1
        Whether to show `2*T1` references in `"t2"` and `"both"` modes.
        Defaults to `False`.
    xaxis_type
        Individual decay-plot x-axis scale. Defaults to `"log"`.
    enable_tqdm
        Whether to show progress bars. Defaults to `True`.

    Returns
    -------
    Result
        Numerical payload, the summary in `figure` and `figures["summary"]`,
        and per-target pre-fit data heatmaps in `figures`. These figures are
        always generated; output options only control display and file writes.

    Raises
    ------
    ValueError
        Raised when an option is invalid or no target-frequency point remains
        after timing and pulse-count constraints are applied.
    TypeError
        Raised when an integer option has a non-integer type.

    Notes
    -----
    This function measures hardware and may write a timestamped result
    directory. The current averaged sweep API does not expose shot-level SEM,
    so fits use unweighted residual covariance unless SEM becomes available.
    A finite fit is labeled `quality_warning` when its relative decay-rate
    error exceeds 0.3 or its R2 is below 0.8. Fits with less than 5 percent
    fitted decay across the measured window, or with decay-rate uncertainty at
    least as large as the rate, are labeled `unresolved_decay` and do not
    produce a finite T2 result.
    """
    summary_mode, save_image, save_fit_images, save_data = (
        _resolve_experiment_deprecated_options(
            plot_mode=summary_mode,
            save_image=save_image,
            save_fit_images=save_fit_images,
            save_data=save_data,
            deprecated_options=deprecated_options,
        )
    )
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="cpmg_noise_spectroscopy",
    )

    target_list = _normalize_targets(exp, targets)
    if not target_list:
        raise ValueError("targets must contain at least one target.")
    if frequency_range is None:
        frequency_range = _DEFAULT_FREQUENCY_RANGE.copy()
    requested_frequencies = _as_non_empty_1d_array(
        frequency_range,
        name="frequency_range",
    )
    if time_range is None:
        time_range = _DEFAULT_TIME_RANGE.copy()
    if pi_pulse_pattern is None:
        pi_pulse_pattern = "++++"
    if n_repeats is None:
        n_repeats = 1
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if save_fit_images is None:
        save_fit_images = False
    if save_data is None:
        save_data = True
    if save_dir is None:
        save_dir = _DEFAULT_SAVE_DIR
    if summary_mode is None:
        summary_mode = "both"
    if xaxis_type is None:
        xaxis_type = "log"
    if enable_tqdm is None:
        enable_tqdm = True

    n_repeats = _require_integer(n_repeats, name="n_repeats", minimum=1)
    n_shots = _require_integer(n_shots, name="n_shots", minimum=1)
    min_pi_pulses = _require_integer(
        min_pi_pulses,
        name="min_pi_pulses",
        minimum=1,
    )
    min_fit_points = _require_integer(
        min_fit_points,
        name="min_fit_points",
        minimum=4,
    )
    if random_seed is not None:
        random_seed = _require_integer(random_seed, name="random_seed", minimum=0)
    if pi_pulse_pattern not in ("++++", "+--+", "+-+-"):
        raise ValueError("pi_pulse_pattern must be one of '++++', '+--+', '+-+-'.")
    if summary_mode not in ("t2", "gamma2", "both"):
        raise ValueError("summary_mode must be 't2', 'gamma2', or 'both'.")
    if xaxis_type not in ("linear", "log"):
        raise ValueError("xaxis_type must be either 'linear' or 'log'.")
    if frequency_order not in ("random", "ascending"):
        raise ValueError("frequency_order must be either 'random' or 'ascending'.")
    requested_times = _as_non_empty_1d_array(time_range, name="time_range")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(save_dir) / timestamp
    if save_data or save_image or save_fit_images:
        output_dir.mkdir(parents=True, exist_ok=True)
    options = _AcquisitionOptions(
        n_repeats=n_repeats,
        n_shots=n_shots,
        shot_interval=shot_interval,
        min_fit_points=min_fit_points,
        frequency_order=frequency_order,
        random_seed=random_seed,
        pi_pulse_pattern=pi_pulse_pattern,
        save_fit_images=save_fit_images,
        xaxis_type=xaxis_type,
        enable_tqdm=enable_tqdm,
    )
    measurement_plan = _create_measurement_plan(
        exp,
        targets=target_list,
        requested_frequencies=requested_frequencies,
        requested_times=requested_times,
        pi_pulses=pi_pulses,
        min_pi_pulses=min_pi_pulses,
        max_pulse_duty_ratio=max_pulse_duty_ratio,
        min_fit_points=min_fit_points,
        pi_pulse_pattern=pi_pulse_pattern,
    )
    acquisition_data, seed_used = _run_acquisition(
        exp,
        plan=measurement_plan,
        options=options,
        output_dir=output_dir,
    )
    save_dir_for_payload = (
        str(output_dir) if (save_data or save_image or save_fit_images) else None
    )
    payload = _build_result_payload(
        plan=measurement_plan,
        data=acquisition_data,
        options=options,
        timestamp=timestamp,
        seed_used=seed_used,
        save_dir=save_dir_for_payload,
        min_pi_pulses=min_pi_pulses,
        max_pulse_duty_ratio=max_pulse_duty_ratio,
    )
    if save_data:
        _save_payload(payload=payload, save_dir=output_dir)
    summary_figure, heatmap_figures = _render_result_figures(
        exp,
        payload=payload,
        output_dir=output_dir,
        summary_mode=summary_mode,
        show_t1=show_t1,
        plot=plot,
        save_image=save_image,
    )
    return Result(
        data=payload,
        figure=summary_figure,
        figures=heatmap_figures,
    )
