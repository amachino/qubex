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
from typing import Any, Literal, TypedDict

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
AxisType = Literal["linear", "log"]

_DEFAULT_SAVE_DIR = "cpmg_noise_spectroscopy"
_GRID_ATOL_NS = 1e-9
_SCHEMA_VERSION = 2
_DEFAULT_FREQUENCY_RANGE = np.linspace(0.0005, 0.004, 20)
_DEFAULT_TIME_RANGE = np.geomspace(200.0, 200_000.0, 31)
_DEFAULT_MIN_PI_PULSES = 4
_DEFAULT_MAX_PI_PULSES = 512
_DEFAULT_MAX_SEQUENCE_TIME_NS = 200_000.0
_DEFAULT_MAX_PULSE_DUTY_RATIO = 0.25
_DEFAULT_MIN_FIT_POINTS = 6
_LARGE_RELATIVE_ERROR = 0.3
_MIN_DECAY_FRACTION = 0.05

logger = logging.getLogger(__name__)


class CPMGResultPayload(TypedDict):
    """In-memory CPMG result payload used by measurement and plotting helpers."""

    timestamp: str
    targets: list[str]
    n_repeats: int
    schema_version: int
    frequency_range: list[float]
    requested_frequency_range: list[float]
    realized_frequency_matrix: NDArray[np.float64]
    valid_frequency_mask: NDArray[np.bool_]
    half_tau_range: list[float]  # First-target compatibility projection.
    tau_spacing_list_ns: list[float]  # Legacy alias of half_tau_range.
    time_range: list[float] | None
    n_pi_range: list[int]
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

    fit_status: str
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


def _default_frequency_range() -> NDArray[np.float64]:
    """Return the default requested CPMG frequency grid in GHz."""
    return _DEFAULT_FREQUENCY_RANGE.copy()


def _default_time_range() -> NDArray[np.float64]:
    """Return the default requested CPMG decay-time grid in ns."""
    return _DEFAULT_TIME_RANGE.copy()


def _pattern_period(pattern: CPMGPattern) -> int:
    """Return the pulse-count period required by a CPMG sign pattern."""
    return 4 if pattern == "+--+" else 2 if pattern == "+-+-" else 1


def _prepare_n_pi_values(
    values: ArrayLike,
    *,
    center_spacing_ns: float,
    pattern: CPMGPattern,
    min_pi_pulses: int,
    max_pi_pulses: int,
    max_sequence_time_ns: float | None,
    warn_on_normalization: bool = True,
) -> NDArray[np.int64]:
    """Validate and filter fixed CPMG pulse counts for one frequency."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("n_pi_range must be a non-empty one-dimensional array.")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_pi_range must contain integers.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("n_pi_range must contain finite integers.")
    if np.any(numeric <= 0):
        raise ValueError("n_pi_range must contain only positive integers.")
    min_pi_pulses = _require_integer(
        min_pi_pulses,
        name="min_pi_pulses",
        minimum=1,
    )
    max_pi_pulses = _require_integer(
        max_pi_pulses,
        name="max_pi_pulses",
        minimum=1,
    )
    if max_pi_pulses < min_pi_pulses:
        raise ValueError("pi-pulse limits must satisfy 1 <= min <= max.")
    if not np.isfinite(center_spacing_ns) or center_spacing_ns <= 0:
        raise ValueError("center_spacing_ns must be positive.")
    if max_sequence_time_ns is not None and (
        not np.isfinite(max_sequence_time_ns) or max_sequence_time_ns <= 0
    ):
        raise ValueError("max_sequence_time_ns must be finite and positive or None.")

    unique = np.unique(numeric.astype(np.int64))
    if warn_on_normalization and (
        len(unique) != len(numeric) or not np.array_equal(unique, numeric)
    ):
        warnings.warn(
            "n_pi_range was sorted and duplicate values were removed.",
            RuntimeWarning,
            stacklevel=2,
        )
    period = _pattern_period(pattern)
    if np.any(unique % period != 0):
        raise ValueError(
            f"n_pi_range values must be multiples of {period} for pattern {pattern!r}."
        )
    mask = (unique >= min_pi_pulses) & (unique <= max_pi_pulses)
    if max_sequence_time_ns is not None:
        mask &= unique.astype(np.float64) * center_spacing_ns <= max_sequence_time_ns
    return unique[mask]


def _n_pi_from_time_range(
    time_range: ArrayLike,
    *,
    center_spacing_ns: float,
    pattern: CPMGPattern,
) -> NDArray[np.int64]:
    """Convert requested decay times to complete CPMG pulse-pattern cycles."""
    times = _as_non_empty_1d_array(
        time_range,
        name="time_range",
    )
    counts = np.rint(times / center_spacing_ns).astype(np.int64)
    period = _pattern_period(pattern)
    return np.unique(counts[(counts > 0) & (counts % period == 0)])


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
    """Fit one CPMG decay numerically without creating a Plotly figure."""
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
    if np.isfinite(r2) and r2 < 0.8:
        flags.append("poor_model_fit")

    unresolved = "weak_decay" in flags or "unresolved_decay" in flags
    fit_status = "unresolved_decay" if unresolved else "ok"
    t2_ns = 1.0 / gamma_per_ns if gamma_per_ns > 0 and not unresolved else math.nan
    t2_error = (
        gamma_error / gamma_per_ns**2
        if np.isfinite(t2_ns) and np.isfinite(gamma_error)
        else math.nan
    )
    return _CPMGFitResult(
        fit_status=fit_status,
        fit_message=(
            "Decay is not resolved over the measured time range."
            if unresolved
            else "Fitting successful."
        ),
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
    covariance = result.covariance
    record["covariance"] = covariance.tolist() if covariance is not None else None
    return record


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


def _save_payload(
    *,
    payload: CPMGResultPayload,
    save_dir: Path,
) -> None:
    """Save result metadata and numeric arrays."""
    save_dir.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, object] = {
        "schema_version": payload.get("schema_version", _SCHEMA_VERSION),
        "timestamp": payload["timestamp"],
        "targets": payload["targets"],
        "n_repeats": payload["n_repeats"],
        "frequency_range": payload["frequency_range"],
        "requested_frequency_range": payload.get(
            "requested_frequency_range", payload["frequency_range"]
        ),
        "half_tau_range": payload.get("half_tau_range", []),
        "tau_spacing_list_ns": payload.get("half_tau_range", []),
        "time_range": payload.get("time_range"),
        "n_pi_range": payload.get("n_pi_range", []),
        "failed_fits": payload.get("failed_fits", []),
        "frequency_plans": payload.get("frequency_plans", []),
        "acquisition_orders": payload.get("acquisition_orders", []),
        "random_seed": payload.get("random_seed"),
        "random_seed_requested": payload.get("random_seed_requested"),
        "metadata": payload.get("metadata", {}),
    }
    with (save_dir / "params.json").open("w", encoding="utf-8") as fp:
        json.dump(_json_safe(serializable), fp, indent=2, allow_nan=False)
    np.save(save_dir / "t2_matrix.npy", payload["t2_matrix"])
    np.save(save_dir / "t2_error_matrix.npy", payload["t2_error_matrix"])
    np.save(save_dir / "r2_matrix.npy", payload["r2_matrix"])
    for name in (
        "gamma2_matrix",
        "gamma2_error_matrix",
        "realized_frequency_matrix",
        "valid_frequency_mask",
    ):
        if name in payload:
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
    if path.exists():
        matrix = np.asarray(np.load(path), dtype=np.float64)
        if matrix.shape != shape:
            raise ValueError(f"{name} has shape {matrix.shape}; expected {shape}.")
        return matrix
    return np.full(shape, np.nan, dtype=np.float64)


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
    n_frequencies = len(frequency_range)
    expected_shape = (n_targets, int(params["n_repeats"]), n_frequencies)
    t2_matrix = np.asarray(np.load(directory / "t2_matrix.npy"), dtype=np.float64)
    if t2_matrix.shape != expected_shape:
        raise ValueError(
            f"t2_matrix.npy has shape {t2_matrix.shape}; expected {expected_shape}."
        )
    frequency_shape = (n_targets, n_frequencies)
    fit_results_path = directory / "fit_results.json"
    fit_records: list[dict[str, object]] = []
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
        "n_repeats": int(params["n_repeats"]),
        "frequency_range": frequency_range,
        "requested_frequency_range": [
            float(value)
            for value in params.get("requested_frequency_range", frequency_range)
        ],
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
        "n_pi_range": [int(value) for value in params.get("n_pi_range", [])],
        "t2_matrix": t2_matrix,
        "t2_error_matrix": _load_optional_matrix(
            directory,
            "t2_error_matrix.npy",
            shape=t2_matrix.shape,
        ),
        "r2_matrix": _load_optional_matrix(
            directory,
            "r2_matrix.npy",
            shape=t2_matrix.shape,
        ),
        "gamma2_matrix": _load_optional_matrix(
            directory,
            "gamma2_matrix.npy",
            shape=t2_matrix.shape,
        ),
        "gamma2_error_matrix": _load_optional_matrix(
            directory,
            "gamma2_error_matrix.npy",
            shape=t2_matrix.shape,
        ),
        "realized_frequency_matrix": _load_optional_matrix(
            directory,
            "realized_frequency_matrix.npy",
            shape=frequency_shape,
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
    if payload["schema_version"] == 1:
        common = np.asarray(frequency_range, dtype=np.float64)
        payload["realized_frequency_matrix"] = np.tile(common, (n_targets, 1))
        finite_t2 = np.isfinite(t2_matrix) & (t2_matrix > 0)
        gamma = np.full_like(t2_matrix, np.nan)
        gamma[finite_t2] = 1.0 / t2_matrix[finite_t2]
        payload["gamma2_matrix"] = gamma
        t2_error = payload["t2_error_matrix"]
        finite_error = finite_t2 & np.isfinite(t2_error) & (t2_error >= 0)
        gamma_error = np.full_like(t2_matrix, np.nan)
        gamma_error[finite_error] = (
            t2_error[finite_error] / t2_matrix[finite_error] ** 2
        )
        payload["gamma2_error_matrix"] = gamma_error
    return payload


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

    required_keys = {
        "timestamp",
        "targets",
        "n_repeats",
        "frequency_range",
        "half_tau_range",
        "tau_spacing_list_ns",
        "t2_matrix",
        "data",
    }
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise ValueError(f"result is missing CPMG payload key(s): {missing}.")

    targets = list(payload["targets"])  # type: ignore[arg-type]
    frequencies = list(payload["frequency_range"])  # type: ignore[arg-type]
    n_repeats = int(payload["n_repeats"])  # type: ignore[arg-type]
    payload["targets"] = targets
    payload["frequency_range"] = frequencies
    payload["n_repeats"] = n_repeats
    n_targets = len(targets)
    n_frequencies = len(frequencies)
    result_shape = (n_targets, n_repeats, n_frequencies)
    frequency_shape = (n_targets, n_frequencies)

    def result_matrix(name: str, *, default: float = math.nan) -> NDArray[np.float64]:
        value = payload.get(name)
        matrix = (
            np.full(result_shape, default, dtype=np.float64)
            if value is None
            else np.asarray(value, dtype=np.float64)
        )
        if matrix.shape != result_shape:
            raise ValueError(
                f"{name} has shape {matrix.shape}; expected {result_shape}."
            )
        return matrix

    t2_matrix = result_matrix("t2_matrix")
    t2_error_matrix = result_matrix("t2_error_matrix")
    r2_matrix = result_matrix("r2_matrix")
    payload.setdefault("failed_fits", [])
    payload.setdefault("save_dir", None)
    payload.setdefault("schema_version", 1)
    payload.setdefault("time_range", None)
    payload.setdefault("requested_frequency_range", payload["frequency_range"])
    if "realized_frequency_matrix" not in payload:
        payload["realized_frequency_matrix"] = np.tile(
            np.asarray(payload["frequency_range"], dtype=np.float64),
            (n_targets, 1),
        )
    payload.setdefault(
        "valid_frequency_mask",
        np.ones((n_targets, n_frequencies), dtype=np.bool_),
    )
    if "gamma2_matrix" not in payload:
        gamma = np.full_like(t2_matrix, np.nan)
        finite = np.isfinite(t2_matrix) & (t2_matrix > 0)
        gamma[finite] = 1.0 / t2_matrix[finite]
        payload["gamma2_matrix"] = gamma
    if "gamma2_error_matrix" not in payload:
        gamma_error = np.full_like(t2_matrix, np.nan)
        finite = (
            np.isfinite(t2_matrix)
            & (t2_matrix > 0)
            & np.isfinite(t2_error_matrix)
            & (t2_error_matrix >= 0)
        )
        gamma_error[finite] = t2_error_matrix[finite] / t2_matrix[finite] ** 2
        payload["gamma2_error_matrix"] = gamma_error
    payload.setdefault("fit_records", [])
    payload.setdefault("frequency_plans", [])
    payload.setdefault("acquisition_orders", [])
    payload.setdefault("metadata", {})
    payload.setdefault("n_pi_range", [])
    payload.setdefault("random_seed", 0)
    payload.setdefault("random_seed_requested", None)
    payload["t2_matrix"] = t2_matrix
    payload["t2_error_matrix"] = t2_error_matrix
    payload["r2_matrix"] = r2_matrix
    payload["gamma2_matrix"] = result_matrix("gamma2_matrix")
    payload["gamma2_error_matrix"] = result_matrix("gamma2_error_matrix")
    realized = np.asarray(payload["realized_frequency_matrix"], dtype=np.float64)
    if realized.shape != frequency_shape:
        raise ValueError(
            "realized_frequency_matrix has shape "
            f"{realized.shape}; expected {frequency_shape}."
        )
    payload["realized_frequency_matrix"] = realized
    valid_mask = np.asarray(payload["valid_frequency_mask"], dtype=np.bool_)
    if valid_mask.shape != frequency_shape:
        raise ValueError(
            f"valid_frequency_mask has shape {valid_mask.shape}; "
            f"expected {frequency_shape}."
        )
    payload["valid_frequency_mask"] = valid_mask
    payload["data"] = t2_matrix
    return payload  # type: ignore[return-value]


def _resolve_plot_deprecated_options(
    *,
    show_t1: bool | None,
    save_image: bool | None,
    deprecated_options: dict[str, Any],
) -> tuple[bool | None, bool | None]:
    """Resolve plotting aliases from the first contrib implementation."""
    show_t1 = _resolve_deprecated_alias(
        options=deprecated_options,
        deprecated_name="show_T1",
        replacement_name="show_t1",
        current_value=show_t1,
        function_name="plot_cpmg_results",
    )
    save_image = _resolve_deprecated_alias(
        options=deprecated_options,
        deprecated_name="save_fig",
        replacement_name="save_image",
        current_value=save_image,
        function_name="plot_cpmg_results",
    )
    if deprecated_options:
        joined = ", ".join(f"`{key}`" for key in sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword argument(s): {joined}")
    return show_t1, save_image


def plot_cpmg_results(
    exp: Experiment,
    result: Result | Mapping[str, object] | None = None,
    *,
    save_dir: str | Path | None = None,
    mode: CPMGSummaryMode | None = None,
    show_t1: bool | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> go.Figure:
    """
    Plot aggregated CPMG noise spectroscopy results.

    Parameters
    ----------
    exp
        Experiment instance used to load optional T1 reference values.
    result
        In-memory result returned by `cpmg_noise_spectroscopy`, or a
        mapping with the same payload keys.
    save_dir
        Directory containing `params.json` and saved matrix files. When
        `result` is also provided, this directory is only used for saving the
        summary image.
    mode
        Summary mode. Use `"t2"` for the inverse of the repeat-averaged
        decay rate in us, `"gamma2"` for that decay rate in `1/us`, or
        `"both"` to display both quantities on separate y-axes. Defaults to
        `"both"`.
    show_t1
        Overlay `2*T1` references when plotting `mode="t2"` or `mode="both"`.
        Defaults to `True` in those modes and `False` in `"gamma2"` mode.
    plot
        Whether to display the figure. Defaults to `True`.
    save_image
        Whether to save the summary figure in `save_dir`. Defaults to `False`.

    Returns
    -------
    plotly.graph_objects.Figure
        Summary figure.

    Raises
    ------
    ValueError
        Raised when the mode, input source, or result payload is invalid.
    """
    show_t1, save_image = _resolve_plot_deprecated_options(
        show_t1=show_t1,
        save_image=save_image,
        deprecated_options=deprecated_options,
    )
    if mode is None:
        mode = "both"
    if show_t1 is None:
        show_t1 = mode in ("t2", "both")
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if mode not in ("t2", "gamma2", "both"):
        raise ValueError("mode must be 't2', 'gamma2', or 'both'.")
    if mode == "gamma2" and show_t1:
        warnings.warn(
            "`show_t1` is ignored when mode='gamma2'.",
            RuntimeWarning,
            stacklevel=2,
        )
        show_t1 = False

    if result is not None:
        payload = _payload_from_result_or_mapping(result)
        directory = Path(save_dir) if save_dir is not None else None
        if directory is None and payload["save_dir"] is not None:
            directory = Path(payload["save_dir"])
    elif save_dir is not None:
        payload = _load_payload(save_dir)
        directory = Path(save_dir)
    else:
        raise ValueError("Either result or save_dir must be provided.")

    targets = payload["targets"]
    if len(targets) == 0:
        raise ValueError("result contains no targets.")

    realized_frequencies = np.asarray(
        payload["realized_frequency_matrix"], dtype=np.float64
    )
    gamma_matrix = np.asarray(payload["gamma2_matrix"], dtype=np.float64)
    gamma_error_matrix = np.asarray(payload["gamma2_error_matrix"], dtype=np.float64)
    yaxis_title = (
        "T2_CPMG = 1 / mean(Gamma_2) (us)"
        if mode in ("t2", "both")
        else "Mean Gamma_2_CPMG (1/us)"
    )
    displayed_modes: tuple[Literal["t2", "gamma2"], ...] = (
        ("t2", "gamma2") if mode == "both" else (mode,)
    )

    figure = viz.make_figure()
    for target_index, target in enumerate(targets):
        frequencies_mhz = realized_frequencies[target_index] * 1e3
        n_frequencies = gamma_matrix.shape[2]
        gamma_average = np.full(n_frequencies, np.nan, dtype=np.float64)
        gamma_summary_error = np.full(n_frequencies, np.nan, dtype=np.float64)
        valid_repeat_counts = np.zeros(n_frequencies, dtype=np.int64)
        for frequency_index in range(n_frequencies):
            values = gamma_matrix[target_index, :, frequency_index]
            finite_values = values[np.isfinite(values) & (values > 0)]
            valid_repeat_counts[frequency_index] = len(finite_values)
            if len(finite_values) == 0:
                continue
            gamma_average[frequency_index] = float(np.mean(finite_values))
            if len(finite_values) >= 2:
                gamma_summary_error[frequency_index] = float(
                    np.std(finite_values, ddof=1) / np.sqrt(len(finite_values))
                )
            else:
                valid_repeats = np.isfinite(values) & (values > 0)
                errors = gamma_error_matrix[
                    target_index, valid_repeats, frequency_index
                ]
                finite_errors = errors[np.isfinite(errors) & (errors >= 0)]
                if len(finite_errors):
                    gamma_summary_error[frequency_index] = float(finite_errors[0])
        finite = np.isfinite(gamma_average) & (gamma_average > 0)
        flagged_indices: set[int] = set()
        for record in payload["fit_records"]:
            index = record.get("requested_frequency_index")
            if (
                record.get("target") == target
                and record.get("quality_flags")
                and isinstance(index, int)
            ):
                flagged_indices.add(index)
        marker_symbols = [
            "diamond" if index in flagged_indices else "circle"
            for index in range(n_frequencies)
        ]

        for displayed_mode in displayed_modes:
            if displayed_mode == "t2":
                y_values = np.full(n_frequencies, np.nan, dtype=np.float64)
                y_errors = np.full(n_frequencies, np.nan, dtype=np.float64)
                y_values[finite] = 1.0 / gamma_average[finite] * 1e-3
                y_errors[finite] = (
                    gamma_summary_error[finite] / gamma_average[finite] ** 2 * 1e-3
                )
                trace_name = f"{target} (T2)" if mode == "both" else target
                hovertemplate = (
                    "frequency=%{x:.6g} MHz<br>"
                    "T2_CPMG=%{y:.6g} us<br>"
                    "valid repeats=%{customdata}<extra>%{fullData.name}</extra>"
                )
            else:
                y_values = gamma_average * 1e3
                y_errors = gamma_summary_error * 1e3
                trace_name = f"{target} (Gamma2)"
                hovertemplate = (
                    "frequency=%{x:.6g} MHz<br>"
                    "Gamma_2_CPMG=%{y:.6g} 1/us<br>"
                    "valid repeats=%{customdata}<extra>%{fullData.name}</extra>"
                )

            figure.add_trace(
                go.Scatter(
                    x=frequencies_mhz,
                    y=y_values,
                    mode="markers+lines",
                    name=trace_name,
                    error_y={
                        "type": "data",
                        "array": y_errors,
                        "visible": bool(np.any(np.isfinite(y_errors))),
                    },
                    marker={"symbol": marker_symbols},
                    customdata=valid_repeat_counts,
                    hovertemplate=hovertemplate,
                    connectgaps=False,
                    yaxis="y2"
                    if mode == "both" and displayed_mode == "gamma2"
                    else "y",
                )
            )

        if show_t1:
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
                    stacklevel=2,
                )

    figure.update_layout(
        title=dict(
            text="CPMG Noise Spectroscopy",
            subtitle=dict(
                text=(
                    f"timestamp: {payload['timestamp']}, "
                    f"repetition: {payload['n_repeats']} times"
                )
            ),
        ),
        xaxis=dict(
            title="Effective CPMG Frequency (MHz)",
            tick0=0,
            showgrid=True,
            rangemode="tozero",
        ),
        yaxis=dict(title=yaxis_title, showgrid=True, rangemode="tozero"),
        legend=dict(x=1.22, xanchor="left", y=1.0, yanchor="top"),
        margin=dict(r=240),
        xaxis_type="linear",
        yaxis_type="linear",
        width=1000,
        height=600,
    )
    if mode == "both":
        figure.update_layout(
            yaxis2=dict(
                title="Mean Gamma_2_CPMG (1/us)",
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
            )
        )

    if save_image:
        if directory is None:
            directory = Path(_DEFAULT_SAVE_DIR) / payload["timestamp"]
        directory.mkdir(parents=True, exist_ok=True)
        figure.write_image(directory / "cpmg_noise_spectroscopy_summary.png")
    if plot:
        figure.show()
    return figure


def _plot_cpmg_raw_data_heatmap(
    *,
    payload: CPMGResultPayload,
    target: str,
    target_index: int,
) -> go.Figure | None:
    """Create a target's pre-fit measured-data heatmap without interpolation."""
    frequencies = np.asarray(payload["realized_frequency_matrix"], dtype=np.float64)[
        target_index
    ]
    valid_indices = np.flatnonzero(np.isfinite(frequencies) & (frequencies > 0))
    if len(valid_indices) == 0:
        return None

    ordered_indices = valid_indices[np.argsort(frequencies[valid_indices])]
    column_by_frequency_index = {
        int(frequency_index): column
        for column, frequency_index in enumerate(ordered_indices)
    }
    samples: list[tuple[int, NDArray[np.float64], NDArray[np.float64]]] = []
    maximum_point_count = 0
    for record in payload["fit_records"]:
        frequency_index = record.get("requested_frequency_index")
        if record.get("target") != target or not isinstance(frequency_index, int):
            continue
        if frequency_index not in column_by_frequency_index:
            continue
        times = np.asarray(record.get("actual_decay_times_ns", []), dtype=np.float64)
        values = np.asarray(record.get("P_values", []), dtype=np.float64)
        if times.ndim != 1 or values.ndim != 1 or times.shape != values.shape:
            continue
        finite = np.isfinite(times) & (times > 0) & np.isfinite(values)
        if not np.any(finite):
            continue
        finite_times = times[finite]
        finite_values = values[finite]
        samples.append((frequency_index, finite_times, finite_values))
        maximum_point_count = max(maximum_point_count, len(finite_times))
    if not samples:
        return None

    all_times = np.concatenate([times for _, times, _ in samples])
    minimum_time = float(np.min(all_times))
    maximum_time = float(np.max(all_times))
    if minimum_time == maximum_time:
        time_centers = np.asarray([minimum_time], dtype=np.float64)
        time_edges = None
    else:
        time_edges = np.geomspace(
            minimum_time,
            maximum_time,
            maximum_point_count + 1,
        )
        time_centers = np.sqrt(time_edges[:-1] * time_edges[1:])

    value_sums = np.zeros((len(time_centers), len(ordered_indices)), dtype=np.float64)
    value_counts = np.zeros_like(value_sums, dtype=np.int64)
    for frequency_index, times, values in samples:
        column = column_by_frequency_index[frequency_index]
        if time_edges is None:
            row_indices = np.zeros(len(times), dtype=np.int64)
        else:
            row_indices = np.searchsorted(time_edges, times, side="right") - 1
            row_indices = np.clip(row_indices, 0, len(time_centers) - 1)
        np.add.at(value_sums[:, column], row_indices, values)
        np.add.at(value_counts[:, column], row_indices, 1)

    heatmap_values = np.full_like(value_sums, np.nan)
    np.divide(
        value_sums,
        value_counts,
        out=heatmap_values,
        where=value_counts > 0,
    )
    figure = viz.make_figure()
    figure.add_trace(
        go.Heatmap(
            x=frequencies[ordered_indices] * 1e3,
            y=time_centers * 1e-3,
            z=heatmap_values,
            colorbar={"title": "Normalized signal P"},
            colorscale="Viridis",
            zsmooth=False,
            hovertemplate=(
                "frequency=%{x:.6g} MHz<br>"
                "time-bin center=%{y:.6g} us<br>"
                "mean P=%{z:.6g}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": f"CPMG measured data before fitting: {target}",
            "subtitle": {"text": "Raw P values averaged within logarithmic time bins"},
        },
        xaxis={
            "title": "Effective CPMG Frequency (MHz)",
            "rangemode": "tozero",
        },
        yaxis={"title": "Effective decay time (us)", "type": "log"},
        width=900,
        height=600,
    )
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
        go.Scatter(
            x=times_ns * 1e-3,
            y=values,
            mode="markers",
            name="Data",
        )
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


def cpmg_noise_spectroscopy(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    frequency_range: ArrayLike | None = None,
    n_pi_range: ArrayLike | None = None,
    time_range: ArrayLike | None = None,
    min_pi_pulses: int = _DEFAULT_MIN_PI_PULSES,
    max_pi_pulses: int = _DEFAULT_MAX_PI_PULSES,
    max_sequence_time_ns: float | None = _DEFAULT_MAX_SEQUENCE_TIME_NS,
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
    show_t1: bool | None = None,
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
    n_pi_range
        Optional fixed positive pi-pulse counts. Do not provide it together
        with `time_range`.
    time_range
        Requested target decay times in ns. Each value is converted to the
        nearest pi-pulse count independently for every target and frequency.
        Defaults to 31 geometrically spaced points from 200 to 200000 ns. Do
        not provide it together with `n_pi_range`.
    min_pi_pulses
        Minimum accepted pi-pulse count. Defaults to 4.
    max_pi_pulses
        Maximum accepted pi-pulse count. Defaults to 512.
    max_sequence_time_ns
        Maximum effective decay time `n_pi * D` in ns. Defaults to 200000.
        Use `None` to disable this limit.
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
        from `time_range` are restricted to complete pattern cycles. Defaults
        to `"++++"`.
    pi_pulses
        Optional per-target pi waveforms.
    n_repeats
        Number of measurements per target and frequency. Defaults to 1.
    n_shots
        Number of shots per sweep point. Defaults to the experiment-wide
        measurement setting.
    shot_interval
        Measurement interval in ns. Defaults to the experiment-wide
        measurement setting.
    plot
        Whether to display the summary figure and pre-fit data heatmaps.
        Defaults to `True`.
    save_image
        Whether to save the summary figure and pre-fit data heatmaps. Defaults
        to `False`.
    save_fit_images
        Whether to render and save individual decay figures. Defaults to
        `False`.
    save_data
        Whether to save schema-v2 numerical results. Defaults to `True`.
    save_dir
        Base output directory. Defaults to `"cpmg_noise_spectroscopy"`.
    summary_mode
        Summary mode, `"t2"`, `"gamma2"`, or `"both"`. Defaults to `"both"`.
    show_t1
        Whether to show `2*T1` references in `"t2"` and `"both"` modes.
        Defaults to `True` in those modes.
    xaxis_type
        Individual decay-plot x-axis scale. Defaults to `"log"`.
    enable_tqdm
        Whether to show progress bars. Defaults to `True`.

    Returns
    -------
    Result
        Numerical payload, an optional summary figure, and optional per-target
        pre-fit data heatmaps in `figures`. Numerical fitting does not create
        Plotly figures.

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
        frequency_range = _default_frequency_range()
    requested_frequencies = _as_non_empty_1d_array(
        frequency_range,
        name="frequency_range",
    )
    if n_pi_range is not None and time_range is not None:
        raise ValueError("n_pi_range and time_range cannot be provided together.")
    if n_pi_range is None and time_range is None:
        time_range = _default_time_range()
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
    max_pi_pulses = _require_integer(
        max_pi_pulses,
        name="max_pi_pulses",
        minimum=1,
    )
    if max_pi_pulses < min_pi_pulses:
        raise ValueError("pi-pulse limits must satisfy 1 <= min <= max.")
    min_fit_points = _require_integer(
        min_fit_points,
        name="min_fit_points",
        minimum=4,
    )
    if max_sequence_time_ns is not None and (
        not np.isfinite(max_sequence_time_ns) or max_sequence_time_ns <= 0
    ):
        raise ValueError("max_sequence_time_ns must be finite and positive or None.")
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
    if n_pi_range is not None:
        n_pi_range = _prepare_n_pi_values(
            n_pi_range,
            center_spacing_ns=1.0,
            pattern=pi_pulse_pattern,
            min_pi_pulses=min_pi_pulses,
            max_pi_pulses=max_pi_pulses,
            max_sequence_time_ns=None,
        )
        if len(n_pi_range) < min_fit_points:
            raise ValueError(
                "n_pi_range contains fewer than min_fit_points values after "
                "applying the pi-pulse limits."
            )

    _validate_rabi_params(exp, target_list)
    sampling_period = _resolve_sampling_period(exp)
    resolved_pi_pulses = {
        target: _resolve_pi_pulse(exp, target, pi_pulses) for target in target_list
    }
    target_plans: list[list[_FrequencyPlan]] = []
    target_n_pi: list[dict[int, NDArray[np.int64]]] = []
    any_valid = False
    for target in target_list:
        pi_duration = float(resolved_pi_pulses[target].duration)
        plan = _make_frequency_plan(
            requested_frequencies,
            sampling_period=sampling_period,
            pi_duration=pi_duration,
            max_pulse_duty_ratio=max_pulse_duty_ratio,
        )
        counts_by_frequency: dict[int, NDArray[np.int64]] = {}
        for position, point in enumerate(plan):
            if point.status != "valid":
                continue
            if point.center_spacing is None:
                raise RuntimeError("A valid frequency point has no center spacing.")
            raw_counts = (
                _n_pi_from_time_range(
                    time_range,
                    center_spacing_ns=point.center_spacing,
                    pattern=pi_pulse_pattern,
                )
                if time_range is not None
                else n_pi_range
            )
            if raw_counts is None:
                raise RuntimeError("No pi-pulse counts were resolved.")
            counts = (
                _prepare_n_pi_values(
                    raw_counts,
                    center_spacing_ns=point.center_spacing,
                    pattern=pi_pulse_pattern,
                    min_pi_pulses=min_pi_pulses,
                    max_pi_pulses=max_pi_pulses,
                    max_sequence_time_ns=max_sequence_time_ns,
                    warn_on_normalization=False,
                )
                if len(raw_counts)
                else np.asarray([], dtype=np.int64)
            )
            if len(counts) < min_fit_points:
                plan[position] = replace(
                    point,
                    status="skipped",
                    reason="insufficient_time_points",
                )
                continue
            counts_by_frequency[point.requested_index] = counts
            any_valid = True
        target_plans.append(plan)
        target_n_pi.append(counts_by_frequency)
    if not any_valid:
        raise ValueError(
            "No valid target-frequency points remain after timing, duty-ratio, "
            "pulse-count, and sequence-time validation."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(save_dir) / timestamp
    if save_data or save_image or save_fit_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    shape = (len(target_list), n_repeats, len(requested_frequencies))
    t2_matrix = np.full(shape, np.nan, dtype=np.float64)
    t2_error_matrix = np.full(shape, np.nan, dtype=np.float64)
    gamma2_matrix = np.full(shape, np.nan, dtype=np.float64)
    gamma2_error_matrix = np.full(shape, np.nan, dtype=np.float64)
    r2_matrix = np.full(shape, np.nan, dtype=np.float64)
    realized_frequency_matrix = np.full(
        (len(target_list), len(requested_frequencies)),
        np.nan,
        dtype=np.float64,
    )
    valid_frequency_mask = np.zeros_like(realized_frequency_matrix, dtype=np.bool_)
    failed_fits: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []
    acquisition_orders: list[list[list[int]]] = [
        [[] for _ in range(n_repeats)] for _ in target_list
    ]
    for target_index, plan in enumerate(target_plans):
        for point in plan:
            if point.status == "valid" and point.frequency is not None:
                realized_frequency_matrix[target_index, point.requested_index] = (
                    point.frequency
                )
                valid_frequency_mask[target_index, point.requested_index] = True
            elif point.reason is not None:
                logger.warning(
                    "Skipping CPMG point for %s at %.9g GHz: %s",
                    target_list[target_index],
                    point.requested_frequency,
                    point.reason,
                )

    seed_used = (
        random_seed
        if random_seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    rng = np.random.default_rng(seed_used)
    repeat_iterable = tqdm(
        range(n_repeats),
        desc="CPMG repetitions",
        disable=not enable_tqdm,
    )
    for repeat_index in repeat_iterable:
        for target_index, target in enumerate(target_list):
            valid_indices = np.flatnonzero(valid_frequency_mask[target_index])
            order = valid_indices.copy()
            if frequency_order == "random":
                rng.shuffle(order)
            else:
                order = order[
                    np.argsort(realized_frequency_matrix[target_index, order])
                ]
            acquisition_orders[target_index][repeat_index] = order.astype(int).tolist()
            frequency_iterable = tqdm(
                order,
                total=len(order),
                desc=f"{target} frequencies",
                leave=False,
                disable=not enable_tqdm,
            )
            for raw_frequency_index in frequency_iterable:
                frequency_index = int(raw_frequency_index)
                point = target_plans[target_index][frequency_index]
                n_pi_values = target_n_pi[target_index][frequency_index]
                if (
                    point.half_tau is None
                    or point.center_spacing is None
                    or point.frequency is None
                ):
                    raise RuntimeError("A valid frequency point has incomplete timing.")
                actual_times = n_pi_values.astype(np.float64) * point.center_spacing
                acquired_at = datetime.now(timezone.utc).isoformat()
                measurement = _sweep_parameter(
                    exp,
                    sequence=lambda n_pi, target=target, half_tau=point.half_tau: (
                        _build_cpmg_sequence(
                            exp,
                            target=target,
                            half_tau=half_tau,
                            n_pi=int(n_pi),
                            pattern=pi_pulse_pattern,
                            pi_pulses=resolved_pi_pulses,
                            sampling_period=sampling_period,
                        )
                    ),
                    sweep_range=n_pi_values,
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                    enable_tqdm=False,
                    title="CPMG decay",
                    xlabel="Number of pi pulses",
                    ylabel="Measured value",
                    xaxis_type="linear",
                )
                sweep_data = measurement.data[target]
                probability = np.asarray(
                    0.5 * (1.0 + sweep_data.normalized),
                    dtype=np.float64,
                )
                fit = _fit_cpmg_decay(
                    actual_times,
                    probability,
                    min_fit_points=min_fit_points,
                )
                record = _fit_result_record(fit)
                record.update(
                    {
                        "target": target,
                        "target_index": target_index,
                        "repeat_index": repeat_index,
                        "requested_frequency_index": frequency_index,
                        "f_requested_GHz": point.requested_frequency,
                        "f_real_GHz": point.frequency,
                        "pi_duration_ns": float(resolved_pi_pulses[target].duration),
                        "pi_pulse_name": resolved_pi_pulses[target].name,
                        "pi_pulse_type": type(resolved_pi_pulses[target]).__name__,
                        "half_tau_ns": point.half_tau,
                        "pi_center_spacing_ns": point.center_spacing,
                        "pulse_duty_ratio": point.pulse_duty_ratio,
                        "sampling_period_ns": sampling_period,
                        "n_pi_values": n_pi_values.tolist(),
                        "actual_decay_times_ns": actual_times.tolist(),
                        "P_values": probability.tolist(),
                        "P_sem_values": None,
                        "n_shots": n_shots,
                        "shot_interval_ns": shot_interval,
                        "pi_pulse_pattern": pi_pulse_pattern,
                        "acquired_at": acquired_at,
                    }
                )
                fit_records.append(record)
                if fit.fit_status == "ok":
                    t2_matrix[target_index, repeat_index, frequency_index] = fit.t2_ns
                    t2_error_matrix[target_index, repeat_index, frequency_index] = (
                        fit.t2_error_ns
                    )
                    gamma2_matrix[target_index, repeat_index, frequency_index] = (
                        fit.gamma2_per_ns
                    )
                    gamma2_error_matrix[target_index, repeat_index, frequency_index] = (
                        fit.gamma2_error_per_ns
                    )
                    r2_matrix[target_index, repeat_index, frequency_index] = fit.r2
                else:
                    failed_fits.append(
                        {
                            "target": target,
                            "repeat_index": repeat_index,
                            "requested_frequency_index": frequency_index,
                            "frequency": point.frequency,
                            "status": fit.fit_status,
                            "message": fit.fit_message,
                        }
                    )
                if save_fit_images:
                    figure = _plot_cpmg_decay(
                        times_ns=actual_times,
                        values=probability,
                        fit=fit,
                        target=target,
                        frequency_ghz=point.frequency,
                        xaxis_type=xaxis_type,
                    )
                    target_dir = output_dir / target
                    target_dir.mkdir(parents=True, exist_ok=True)
                    figure.write_image(
                        target_dir
                        / (
                            f"{target}_freq_{point.frequency * 1e3:.6f}_MHz_"
                            f"rep_{repeat_index + 1}.png"
                        )
                    )

    save_dir_for_payload = (
        str(output_dir) if (save_data or save_image or save_fit_images) else None
    )
    first_plan_half_taus = [
        float(point.half_tau) if point.half_tau is not None else math.nan
        for point in target_plans[0]
    ]
    payload: CPMGResultPayload = {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": timestamp,
        "targets": target_list,
        "n_repeats": n_repeats,
        "frequency_range": requested_frequencies.tolist(),
        "requested_frequency_range": requested_frequencies.tolist(),
        "realized_frequency_matrix": realized_frequency_matrix,
        "valid_frequency_mask": valid_frequency_mask,
        "half_tau_range": first_plan_half_taus,
        "tau_spacing_list_ns": first_plan_half_taus,
        "time_range": (
            np.asarray(time_range, dtype=np.float64).tolist()
            if time_range is not None
            else None
        ),
        "n_pi_range": (
            np.asarray(n_pi_range, dtype=np.int64).tolist()
            if n_pi_range is not None
            else []
        ),
        "t2_matrix": t2_matrix,
        "t2_error_matrix": t2_error_matrix,
        "gamma2_matrix": gamma2_matrix,
        "gamma2_error_matrix": gamma2_error_matrix,
        "r2_matrix": r2_matrix,
        "fit_records": fit_records,
        "frequency_plans": [[asdict(point) for point in plan] for plan in target_plans],
        "acquisition_orders": acquisition_orders,
        "random_seed": seed_used,
        "random_seed_requested": random_seed,
        "metadata": {
            "sampling_period_ns": sampling_period,
            "min_pi_pulses": min_pi_pulses,
            "max_pi_pulses": max_pi_pulses,
            "max_sequence_time_ns": max_sequence_time_ns,
            "max_pulse_duty_ratio": max_pulse_duty_ratio,
            "min_fit_points": min_fit_points,
            "frequency_order": frequency_order,
            "pi_pulse_pattern": pi_pulse_pattern,
            "n_shots": n_shots,
            "shot_interval_ns": shot_interval,
            "frequency_definition": "1 / (2 * pi_center_spacing)",
            "compatibility_timing_target": target_list[0],
            "half_tau_range_semantics": (
                "first-target compatibility projection; use frequency_plans "
                "for target-specific timing"
            ),
            "tau_spacing_list_ns_semantics": "legacy alias of half_tau_range",
        },
        "data": t2_matrix,
        "failed_fits": failed_fits,
        "save_dir": save_dir_for_payload,
    }
    if save_data:
        _save_payload(payload=payload, save_dir=output_dir)

    summary_figure = None
    heatmap_figures: dict[str, go.Figure] = {}
    if plot or save_image:
        summary_figure = plot_cpmg_results(
            exp,
            payload,
            save_dir=output_dir,
            mode=summary_mode,
            show_t1=show_t1,
            plot=plot,
            save_image=save_image,
        )
        for target_index, target in enumerate(target_list):
            heatmap = _plot_cpmg_raw_data_heatmap(
                payload=payload,
                target=target,
                target_index=target_index,
            )
            if heatmap is None:
                continue
            heatmap_figures[f"{target}_raw_data_heatmap"] = heatmap
            if save_image:
                heatmap.write_image(output_dir / f"{target}_raw_data_heatmap.png")
            if plot:
                heatmap.show()
    return Result(
        data=payload,
        figure=summary_figure,
        figures=heatmap_figures or None,
    )
