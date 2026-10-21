"""
Tilted-axis spin-lock relaxation and relaxation-spectroscopy measurements.

Durations are represented in ns and frequencies in GHz throughout the returned
data. The two public functions measure either one drive condition or a sequence
of Rabi frequencies; both use the same seven-expectation global-fit model.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from tqdm import tqdm

import qubex.visualization as viz
from qubex.core.async_bridge import get_shared_async_bridge
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.models.result import Result
from qubex.pulse import FlatTop, PulseSchedule

__all__ = ["spin_lock_relaxation", "spin_lock_relaxation_spectroscopy"]

DEFAULT_DURATION_RANGE = np.geomspace(100.0, 200e3, 21)
DEFAULT_DURATION_POINTS = 18
DEFAULT_SPIN_LOCK_FREQUENCY_RANGE = (
    0.12 * np.tan(np.pi / 3 * np.linspace(0, 1, 15)) / np.sqrt(3)
)[1:]
DEFAULT_RAMP_DURATION = 16.0
MIN_DURATION = 100.0
POINT_UNCERTAINTY_SAMPLES = 500
ANALYSIS_EXCEPTIONS = (
    ValueError,
    RuntimeError,
    np.linalg.LinAlgError,
    FloatingPointError,
)
EXPECTATION_LABELS = ("X+", "Z+", "X-", "Z-", "Xi", "Yi", "Zi")
POINT_DISPLAY_NAMES = {
    "a": "Longitudinal retention a(t)",
    "b": "Transverse coherence b(t)",
    "p": "Longitudinal polarization offset p(t)",
}
POINT_SUBPLOT_TITLES = {
    "a": "a(t)<br><sup>Longitudinal retention (T1ρ)</sup>",
    "b": "b(t)<br><sup>Transverse coherence (T2ρ)</sup>",
    "p": "p(t)<br><sup>Longitudinal polarization offset</sup>",
}
MEASUREMENT_SETTINGS: tuple[tuple[str, Literal["X", "Y", "Z"]], ...] = (
    ("+", "X"),
    ("+", "Z"),
    ("-", "X"),
    ("-", "Z"),
    ("+i", "X"),
    ("+i", "Y"),
    ("+i", "Z"),
)


@dataclass(frozen=True)
class _MeasurementData:
    """Processed data from one Rabi frequency."""

    mean_iq: np.ndarray
    expectations: np.ndarray
    expectation_errors: np.ndarray
    measurement_order: np.ndarray


# Input resolution


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _resolve_measurement_options(
    exp: Experiment,
    targets: Collection[str] | str | None,
    *,
    ramp_duration: float,
    n_shots: int | None,
    shot_interval: float | None,
) -> tuple[list[str], float, int, float]:
    target_list = _normalize_targets(exp, targets)
    if not target_list:
        raise ValueError("targets must contain at least one target.")

    resolved_ramp = float(ramp_duration)
    if not np.isfinite(resolved_ramp) or resolved_ramp < 0:
        raise ValueError("ramp_duration must be finite and nonnegative.")

    if n_shots is not None and (
        isinstance(n_shots, bool) or not isinstance(n_shots, (int, np.integer))
    ):
        raise ValueError("n_shots must be an integer of at least 2.")
    resolved_shots = int(DEFAULT_SHOTS if n_shots is None else n_shots)
    if resolved_shots < 2:
        raise ValueError("n_shots must be at least 2 to estimate shot uncertainty.")

    resolved_interval = float(
        DEFAULT_INTERVAL if shot_interval is None else shot_interval
    )
    if not np.isfinite(resolved_interval) or resolved_interval < 0:
        raise ValueError("shot_interval must be finite and nonnegative.")
    return target_list, resolved_ramp, resolved_shots, resolved_interval


def _mapping_value(
    value: float | Mapping[str, float] | None,
    target: str,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if target not in value:
            raise ValueError(f"`{name}` is missing target `{target}`.")
        return float(value[target])
    return float(value)


def _coherence_value(
    values: Mapping[str, Any], exp: Experiment, target: str
) -> float | None:
    value = values.get(target)
    if value is None:
        try:
            value = values.get(exp.ctx.resolve_qubit_label(target))
        except (AttributeError, KeyError, ValueError):
            value = None
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > 0 else None


def _load_coherence_value(
    exp: Experiment,
    target: str,
    parameter: Literal["t1", "t2_echo"],
) -> float | None:
    """Load one usable coherence time without coupling the two YAML files."""
    try:
        values = exp.ctx.system_manager.config_loader.load_param_data(parameter)
    except (AttributeError, KeyError, RuntimeError):
        return None
    if not isinstance(values, Mapping):
        return None
    return _coherence_value(values, exp, target)


def _default_duration_range(
    exp: Experiment,
    target: str,
    *,
    t1: float | None,
    t2_echo: float | None,
    n_points: int | None,
) -> np.ndarray:
    if n_points is not None and (
        isinstance(n_points, bool)
        or not isinstance(n_points, (int, np.integer))
        or n_points < 3
    ):
        raise ValueError("n_duration_points must be an integer of at least 3.")

    supplied_coherence_time = t1 is not None or t2_echo is not None
    if not supplied_coherence_time:
        t1 = _load_coherence_value(exp, target, "t1")
        t2_echo = _load_coherence_value(exp, target, "t2_echo")

    coherence_times = []
    for name, value in (("t1", t1), ("t2_echo", t2_echo)):
        if value is None:
            continue
        resolved = float(value)
        if not np.isfinite(resolved) or resolved <= 0:
            raise ValueError(f"{name} for `{target}` must be positive and finite.")
        coherence_times.append(resolved)
    if not coherence_times:
        if n_points is None:
            return DEFAULT_DURATION_RANGE.copy()
        return np.geomspace(
            DEFAULT_DURATION_RANGE[0],
            DEFAULT_DURATION_RANGE[-1],
            n_points,
        )

    start = max(MIN_DURATION, 0.025 * min(coherence_times))
    stop = 10.0 * max(coherence_times)
    if stop <= start:
        raise ValueError(
            f"Coherence time for `{target}` is too short to build a "
            f"default duration range above {MIN_DURATION:g} ns."
        )
    return np.geomspace(
        start, stop, DEFAULT_DURATION_POINTS if n_points is None else n_points
    )


def _resolve_duration_range(
    exp: Experiment,
    target: str,
    duration_range: ArrayLike | None,
    *,
    t1: float | None = None,
    t2_echo: float | None = None,
    n_points: int | None = None,
) -> np.ndarray:
    durations = np.asarray(
        _default_duration_range(
            exp,
            target,
            t1=t1,
            t2_echo=t2_echo,
            n_points=n_points,
        )
        if duration_range is None
        else duration_range,
        dtype=np.float64,
    )
    if durations.ndim != 1 or durations.size < 3:
        raise ValueError("duration_range must be a 1D array with at least 3 points.")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("duration_range must contain positive finite values.")

    sampling_period = float(exp.ctx.measurement.sampling_period)
    if not np.isfinite(sampling_period) or sampling_period <= 0:
        raise ValueError("measurement sampling_period must be positive and finite.")
    durations = exp.ctx.util.discretize_time_range(
        durations,
        sampling_period=sampling_period,
    )
    if durations is None:
        raise ValueError("duration_range could not be discretized.")
    durations = np.asarray(durations, dtype=np.float64)
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError(
            "duration_range must remain positive and finite on the sampling grid."
        )
    durations = np.unique(durations)
    if durations.size < 3:
        raise ValueError("duration_range must leave at least 3 unique points.")
    return durations


def _resolve_target_durations(
    exp: Experiment,
    target: str,
    *,
    duration_range: ArrayLike | None,
    t1: float | Mapping[str, float] | None,
    t2_echo: float | Mapping[str, float] | None,
    n_duration_points: int | None,
) -> np.ndarray:
    return _resolve_duration_range(
        exp,
        target,
        duration_range,
        t1=_mapping_value(t1, target, name="t1") if duration_range is None else None,
        t2_echo=_mapping_value(t2_echo, target, name="t2_echo")
        if duration_range is None
        else None,
        n_points=n_duration_points,
    )


def _resolve_spin_lock_frequency_range(
    spin_lock_frequency_range: ArrayLike | None,
) -> np.ndarray:
    frequencies = np.array(
        DEFAULT_SPIN_LOCK_FREQUENCY_RANGE
        if spin_lock_frequency_range is None
        else spin_lock_frequency_range,
        dtype=np.float64,
        copy=True,
    )
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("spin_lock_frequency_range must be a non-empty 1D array.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError(
            "spin_lock_frequency_range must contain positive finite values."
        )
    return frequencies


def _resolve_min_initial_amplitude(
    value: float | None,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be nonnegative and finite, or None.")
    return resolved


def _resolve_drive(
    exp: Experiment,
    *,
    target: str,
    drive_amplitude: float | Mapping[str, float] | None,
    rabi_frequency: float | Mapping[str, float] | None,
) -> tuple[float, float]:
    amplitude = _mapping_value(
        drive_amplitude,
        target,
        name="drive_amplitude",
    )
    if amplitude is None:
        requested_rabi = _mapping_value(
            rabi_frequency,
            target,
            name="rabi_frequency",
        )
        if requested_rabi is None:
            amplitude = float(exp.ctx.params.get_control_amplitude(target))
        else:
            amplitude = float(exp.pulse.calc_control_amplitude(target, requested_rabi))
    if not np.isfinite(amplitude) or abs(amplitude) > 1:
        raise ValueError(
            f"Drive amplitude for `{target}` must be finite with magnitude at most 1."
        )

    estimated_rabi = float(exp.pulse.calc_rabi_rate(target, amplitude))
    if not np.isfinite(estimated_rabi) or estimated_rabi <= 0:
        raise ValueError(f"Estimated Rabi frequency for `{target}` must be positive.")
    return amplitude, estimated_rabi


def _resolve_frequency_sweep(
    exp: Experiment,
    *,
    target: str,
    requested_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    resolved = [
        _resolve_drive(
            exp,
            target=target,
            drive_amplitude=None,
            rabi_frequency=float(frequency),
        )
        for frequency in requested_frequencies
    ]
    amplitudes, rabi_frequencies = zip(*resolved, strict=True)
    return np.asarray(amplitudes), np.asarray(rabi_frequencies)


# Relaxation model and fitting


def _design_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray(
        [
            [c, c**2, s**2, 0.0],
            [s, c * s, -c * s, 0.0],
            [c, -(c**2), -(s**2), 0.0],
            [s, -c * s, c * s, 0.0],
            [c, 0.0, 0.0, -s],
            [0.0, 0.0, 1.0, 0.0],
            [s, 0.0, 0.0, c],
        ],
        dtype=np.float64,
    )


def _predict_expectations(
    durations: ArrayLike,
    *,
    theta: float,
    t1_rho: float,
    t2_rho: float,
    p_inf: float,
    a0: float,
    b0: float,
    p0: float,
    phases: ArrayLike,
) -> np.ndarray:
    times = np.asarray(durations, dtype=np.float64)
    phase_values = np.asarray(phases, dtype=np.float64)
    a = a0 * np.exp(-times / t1_rho)
    b = b0 * np.exp(-times / t2_rho)
    p = p_inf + (p0 - p_inf) * np.exp(-times / t1_rho)
    parameters = np.column_stack(
        (p, a, b * np.cos(phase_values), b * np.sin(phase_values))
    )
    return parameters @ _design_matrix(theta).T


def _profile_initial_values(observations: np.ndarray) -> tuple[float, np.ndarray]:
    best_theta = 0.0
    best_parameters = np.empty((observations.shape[0], 4))
    best_cost = np.inf
    for theta in np.linspace(-1.4, 1.4, 281):
        matrix = _design_matrix(float(theta))
        parameters = np.linalg.lstsq(matrix, observations.T, rcond=None)[0].T
        residual = observations - parameters @ matrix.T
        cost = float(np.sum(residual**2))
        if cost < best_cost:
            best_theta = float(theta)
            best_parameters = parameters
            best_cost = cost
    return best_theta, best_parameters


def _estimate_decay_time(times: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values) & (values > 0.05)
    if np.count_nonzero(mask) >= 3:
        slope = np.polyfit(times[mask], np.log(values[mask]), 1)[0]
        if slope < 0:
            return float(-1.0 / slope)
    return float(np.sqrt(times[0] * times[-1]))


def _fit_global_model(
    durations: ArrayLike,
    observations: ArrayLike,
    *,
    sigma: ArrayLike,
) -> dict[str, Any]:
    from scipy.optimize import least_squares

    times = np.asarray(durations, dtype=np.float64)
    measured = np.asarray(observations, dtype=np.float64)
    uncertainties = np.asarray(sigma, dtype=np.float64)
    if times.ndim != 1 or times.size < 3:
        raise ValueError("durations must be a 1D array with at least 3 points.")
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("durations must contain positive finite values.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("durations must be strictly increasing.")
    if measured.shape != (times.size, len(EXPECTATION_LABELS)):
        raise ValueError("observations must have shape (n_durations, 7).")
    if not np.all(np.isfinite(measured)):
        raise ValueError("observations must contain finite values.")
    try:
        uncertainties = np.broadcast_to(uncertainties, measured.shape)
    except ValueError as exc:
        raise ValueError("sigma must be scalar or match observations.") from exc
    if not np.all(np.isfinite(uncertainties)) or np.any(uncertainties <= 0):
        raise ValueError("sigma must contain positive finite values.")

    theta0, point0 = _profile_initial_values(measured)
    p_values, a_values, k_values, q_values = point0.T
    b_values = np.hypot(k_values, q_values)
    t1_0 = _estimate_decay_time(times, np.abs(a_values))
    t2_0 = _estimate_decay_time(times, b_values)
    x0 = np.concatenate(
        (
            [
                theta0,
                np.log(t1_0),
                np.log(t2_0),
                np.clip(p_values[-1], -1.0, 1.0),
                np.clip(a_values[0] * np.exp(times[0] / t1_0), 0.01, 1.5),
                np.clip(b_values[0] * np.exp(times[0] / t2_0), 0.01, 1.5),
                np.clip(p_values[0], -1.0, 1.0),
            ],
            np.unwrap(np.arctan2(q_values, k_values)),
        )
    )

    lower = np.concatenate(
        ([-1.5, -30.0, -30.0, -1.0, 0.0, 0.0, -1.0], np.full(times.size, -np.inf))
    )
    upper = np.concatenate(
        ([1.5, 40.0, 40.0, 1.0, 1.5, 1.5, 1.0], np.full(times.size, np.inf))
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        predicted = _predict_expectations(
            times,
            theta=vector[0],
            t1_rho=np.exp(vector[1]),
            t2_rho=np.exp(vector[2]),
            p_inf=vector[3],
            a0=vector[4],
            b0=vector[5],
            p0=vector[6],
            phases=vector[7:],
        )
        return ((predicted - measured) / uncertainties).ravel()

    fit = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        max_nfev=20_000,
    )
    if not fit.success:
        raise RuntimeError(f"Spin-lock relaxation fit failed: {fit.message}")

    names = ("theta", "t1_rho", "t2_rho", "p_inf", "a0", "b0", "p0")
    dof = max(1, measured.size - fit.x.size)
    reduced_chi_squared = float(2.0 * fit.cost / dof)
    information_inverse = np.linalg.pinv(fit.jac.T @ fit.jac)
    leverage = np.sum((fit.jac @ information_inverse) * fit.jac, axis=1)
    adjusted_residuals = fit.fun / np.maximum(1.0 - leverage, 1e-6)
    robust_covariance_internal = (
        information_inverse
        @ (fit.jac.T @ ((adjusted_residuals**2)[:, None] * fit.jac))
        @ information_inverse
    )
    t1_rho = float(np.exp(fit.x[1]))
    t2_rho = float(np.exp(fit.x[2]))
    transform = np.ones(fit.x.size)
    transform[1] = t1_rho
    transform[2] = t2_rho
    covariance = information_inverse * np.outer(transform, transform)
    robust_covariance = robust_covariance_internal * np.outer(transform, transform)
    statistical_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    robust_errors = np.sqrt(np.maximum(0.0, np.diag(robust_covariance)))
    errors = np.maximum(statistical_errors, robust_errors)
    parameters = {
        "theta": float(fit.x[0]),
        "t1_rho": t1_rho,
        "t2_rho": t2_rho,
        **{name: float(fit.x[index]) for index, name in enumerate(names[3:], start=3)},
    }
    parameter_errors = {name: float(errors[index]) for index, name in enumerate(names)}
    parameter_statistical_errors = {
        name: float(statistical_errors[index]) for index, name in enumerate(names)
    }
    parameter_robust_errors = {
        name: float(robust_errors[index]) for index, name in enumerate(names)
    }
    predicted = _predict_expectations(
        times,
        theta=fit.x[0],
        t1_rho=t1_rho,
        t2_rho=t2_rho,
        p_inf=fit.x[3],
        a0=fit.x[4],
        b0=fit.x[5],
        p0=fit.x[6],
        phases=fit.x[7:],
    )
    return {
        "success": bool(fit.success),
        "message": fit.message,
        "parameters": parameters,
        "errors": parameter_errors,
        "statistical_errors": parameter_statistical_errors,
        "robust_errors": parameter_robust_errors,
        "phases": fit.x[7:].copy(),
        "phase_errors": errors[7:].copy(),
        "covariance": covariance,
        "robust_covariance": robust_covariance,
        "parameter_order": [*names, *(f"phi_{index}" for index in range(times.size))],
        "predicted_expectations": predicted,
        "residuals": measured - predicted,
        "reduced_chi_squared": reduced_chi_squared,
    }


def _point_estimates(
    observations: np.ndarray,
    *,
    theta: float,
    theta_error: float,
    sigma: ArrayLike,
) -> dict[str, np.ndarray]:
    uncertainties = np.broadcast_to(
        np.asarray(sigma, dtype=np.float64), observations.shape
    )
    parameters = np.empty((observations.shape[0], 4), dtype=np.float64)
    for index, (values, errors) in enumerate(
        zip(observations, uncertainties, strict=True)
    ):
        weighted_matrix = _design_matrix(theta) / errors[:, None]
        parameters[index] = np.linalg.lstsq(
            weighted_matrix,
            values / errors,
            rcond=None,
        )[0]
    p, a, k, q = parameters.T
    b = np.hypot(k, q)

    rng = np.random.default_rng(0)
    theta_samples = rng.normal(theta, theta_error, POINT_UNCERTAINTY_SAMPLES)
    samples = {
        name: np.empty((POINT_UNCERTAINTY_SAMPLES, observations.shape[0]))
        for name in ("a", "b", "p")
    }
    for sample_index, theta_sample in enumerate(theta_samples):
        matrix = _design_matrix(float(theta_sample))
        sampled_observations = rng.normal(observations, uncertainties)
        for time_index, (values, errors) in enumerate(
            zip(sampled_observations, uncertainties, strict=True)
        ):
            weighted_matrix = matrix / errors[:, None]
            point = np.linalg.lstsq(
                weighted_matrix,
                values / errors,
                rcond=None,
            )[0]
            samples["p"][sample_index, time_index] = point[0]
            samples["a"][sample_index, time_index] = point[1]
            samples["b"][sample_index, time_index] = np.hypot(point[2], point[3])

    point_values = {"a": a, "b": b, "p": p}
    uncertainty_results: dict[str, np.ndarray] = {}
    for name, values in point_values.items():
        lower, upper = np.quantile(samples[name], [0.16, 0.84], axis=0)
        uncertainty_results[f"{name}_error"] = np.std(samples[name], axis=0, ddof=1)
        uncertainty_results[f"{name}_error_minus"] = np.maximum(values - lower, 0.0)
        uncertainty_results[f"{name}_error_plus"] = np.maximum(upper - values, 0.0)
    return {
        "a": a,
        "b": b,
        "p": p,
        **uncertainty_results,
        "k": k,
        "q": q,
        "phi": np.arctan2(q, k),
    }


def _model_diagnostics(observations: np.ndarray, theta: float) -> dict[str, Any]:
    x_plus, z_plus, x_minus, z_minus, x_i, _y_i, z_i = observations.T
    c = np.cos(theta)
    s = np.sin(theta)
    c_x = (x_plus + x_minus) / 2
    c_z = (z_plus + z_minus) / 2
    d_x = (x_plus - x_minus) / 2
    d_z = (z_plus - z_minus) / 2
    k = observations[:, 5]
    return {
        "translation_axis_residual": -c_x * s + c_z * c,
        "contraction_axis_residual": -(d_x - k) * s + d_z * c,
        "transverse_plane_residual": (x_i - c_x) * c + (z_i - c_z) * s,
    }


# Pulse construction and acquisition


def _build_sequence(
    exp: Experiment,
    *,
    target: str,
    flat_duration: float,
    drive_amplitude: float,
    ramp_duration: float,
    initial_state: str,
    basis: Literal["X", "Y", "Z"],
) -> PulseSchedule:
    with PulseSchedule([target]) as schedule:
        schedule.add(target, exp.pulse.get_pulse_for_state(target, initial_state))
        schedule.add(
            target,
            FlatTop(
                duration=flat_duration + 2.0 * ramp_duration,
                amplitude=drive_amplitude,
                tau=ramp_duration,
                type="RaisedCosine",
            ),
        )
        if basis == "X":
            schedule.add(target, exp.pulse.y90m(target))
        elif basis == "Y":
            schedule.add(target, exp.pulse.x90(target))
    return schedule


def _measure_target(
    exp: Experiment,
    *,
    target: str,
    durations: np.ndarray,
    drive_amplitude: float,
    ramp_duration: float,
    n_shots: int,
    shot_interval: float,
) -> _MeasurementData:
    """Measure one shuffled duration-setting sweep and restore canonical order."""
    points: list[tuple[float, str, Literal["X", "Y", "Z"]]] = [
        (duration, initial_state, basis)
        for duration in durations
        for initial_state, basis in MEASUREMENT_SETTINGS
    ]
    measurement_order = np.random.default_rng().permutation(len(points))

    def sequence(index: int) -> PulseSchedule:
        duration, initial_state, basis = points[int(measurement_order[int(index)])]
        return _build_sequence(
            exp,
            target=target,
            flat_duration=float(duration),
            drive_amplitude=drive_amplitude,
            ramp_duration=ramp_duration,
            initial_state=initial_state,
            basis=basis,
        )

    rabi_param = exp.get_rabi_param(target)
    if rabi_param is None:
        raise ValueError(f"Rabi parameters are unavailable for `{target}`.")
    exp.ctx.reset_awg_and_capunits(qubits={target})
    bridge = get_shared_async_bridge(key="experiment")
    with exp.ctx.util.no_output():
        sweep = bridge.run(
            lambda: exp.measurement_service.run_sweep_measurement(
                schedule=sequence,
                sweep_values=np.arange(len(points)),
                n_shots=n_shots,
                shot_interval=shot_interval,
                shot_averaging=False,
                time_integration=True,
                state_classification=False,
                final_measurement=True,
                plot=False,
                enable_tqdm=False,
            )
        )

    capture_series = sweep.data.get(target)
    if not capture_series:
        raise ValueError(f"Sweep measurement returned no data for `{target}`.")
    shot_iq = np.asarray(capture_series[-1])
    normalized_shots = np.asarray(rabi_param.normalize(shot_iq), dtype=np.float64)
    expected_shape = (len(points), n_shots)
    if shot_iq.shape != expected_shape:
        raise ValueError(f"Shot IQ for `{target}` must have shape {expected_shape}.")
    if normalized_shots.shape != shot_iq.shape:
        raise ValueError(
            f"Normalized shot data for `{target}` must have shape {shot_iq.shape}."
        )
    canonical_iq = np.empty_like(shot_iq)
    canonical_expectations = np.empty_like(normalized_shots)
    canonical_iq[measurement_order] = shot_iq
    canonical_expectations[measurement_order] = normalized_shots
    shot_shape = (durations.size, len(EXPECTATION_LABELS), n_shots)
    canonical_iq = canonical_iq.reshape(shot_shape)
    canonical_expectations = canonical_expectations.reshape(shot_shape)
    return _MeasurementData(
        mean_iq=np.mean(canonical_iq, axis=-1),
        expectations=np.mean(canonical_expectations, axis=-1),
        expectation_errors=(
            np.std(canonical_expectations, axis=-1, ddof=1) / np.sqrt(n_shots)
        ),
        measurement_order=measurement_order,
    )


# Analysis, visualization, and result assembly


def _make_figure(
    *,
    target: str,
    durations: np.ndarray,
    points: Mapping[str, np.ndarray],
    fit: Mapping[str, Any],
) -> go.Figure:
    figure = viz.make_figure(width=1200, height=360)
    figure.set_subplots(
        rows=1,
        cols=3,
        subplot_titles=tuple(POINT_SUBPLOT_TITLES.values()),
        horizontal_spacing=0.08,
    )
    parameters = fit["parameters"]
    dense_times = np.geomspace(durations[0], durations[-1], 400)
    curves = {
        "a": parameters["a0"] * np.exp(-dense_times / parameters["t1_rho"]),
        "b": parameters["b0"] * np.exp(-dense_times / parameters["t2_rho"]),
        "p": parameters["p_inf"]
        + (parameters["p0"] - parameters["p_inf"])
        * np.exp(-dense_times / parameters["t1_rho"]),
    }
    colors = {"a": "#1f77b4", "b": "#d62728", "p": "#2ca02c"}
    for column, name in enumerate(("a", "b", "p"), start=1):
        figure.add_trace(
            go.Scatter(
                x=durations * 1e-3,
                y=points[name],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": points[f"{name}_error_plus"],
                    "arrayminus": points[f"{name}_error_minus"],
                },
                mode="markers",
                marker={"color": colors[name]},
                name=f"{POINT_DISPLAY_NAMES[name]} estimates",
                hovertemplate=(
                    "Flattop duration: %{x:.5g} μs<br>"
                    f"{POINT_DISPLAY_NAMES[name]}: %{{y:.5g}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=dense_times * 1e-3,
                y=curves[name],
                mode="lines",
                line={"color": colors[name]},
                name=f"{POINT_DISPLAY_NAMES[name]} fit",
                hovertemplate=(
                    "Flattop duration: %{x:.5g} μs<br>"
                    f"{POINT_DISPLAY_NAMES[name]} fit: %{{y:.5g}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text="Flattop duration (μs)", type="log", row=1, col=column
        )
        figure.update_yaxes(title_text=POINT_DISPLAY_NAMES[name], row=1, col=column)

    figure.update_layout(title=f"Tilted-axis spin-lock relaxation: {target}")
    return figure


def _print_fit_summary(
    *,
    target: str,
    fit: Mapping[str, Any],
    rabi_frequency: float,
    shift: float,
    shift_error: float,
    axis_frequency: float,
    axis_frequency_error: float,
) -> None:
    parameters = fit["parameters"]
    errors = fit["errors"]
    print(f"Spin-lock relaxation fit: {target}")
    print(f"  θ = {parameters['theta']:.5g} ± {errors['theta']:.2g} rad")
    print(
        f"  T1ρ = {parameters['t1_rho'] * 1e-3:.5g} ± {errors['t1_rho'] * 1e-3:.2g} μs"
    )
    print(
        f"  T2ρ = {parameters['t2_rho'] * 1e-3:.5g} ± {errors['t2_rho'] * 1e-3:.2g} μs"
    )
    print(f"  p∞ = {parameters['p_inf']:.5g} ± {errors['p_inf']:.2g}")
    print(f"  Rabi frequency = {rabi_frequency * 1e3:.5g} MHz")
    print(f"  shift = {shift * 1e3:.5g} ± {shift_error * 1e3:.2g} MHz")
    print(
        f"  axis rotation frequency = {axis_frequency * 1e3:.5g} ± "
        f"{axis_frequency_error * 1e3:.2g} MHz"
    )


def _print_spectroscopy_diagnostics(
    result: Mapping[str, Any],
    rabi_frequencies: np.ndarray,
) -> None:
    for index in np.flatnonzero(~result["fit_success"]):
        frequency_mhz = rabi_frequencies[index] * 1e3
        print(
            f"  Fit failed at {frequency_mhz:.6g} MHz: {result['fit_messages'][index]}"
        )

    for quantity, messages in (
        ("T1ρ", result["t1_exclusion_messages"]),
        ("T2ρ", result["t2_exclusion_messages"]),
    ):
        for index, message in enumerate(messages):
            if message is None or not result["fit_success"][index]:
                continue
            frequency_mhz = rabi_frequencies[index] * 1e3
            print(
                f"  Omitting {quantity} at {frequency_mhz:.6g} MHz from plot: {message}"
            )


def _output_figure(
    figure: go.Figure,
    *,
    name: str,
    plot: bool,
    save_image: bool,
) -> None:
    if plot:
        figure.show(config=viz.get_config(filename=name))
    if save_image:
        viz.save_figure(figure, name=name)


def _analyze_target(
    *,
    target: str,
    durations: np.ndarray,
    measurement: _MeasurementData,
    drive_amplitude: float,
    rabi_frequency: float,
    ramp_duration: float,
) -> tuple[dict[str, Any], go.Figure]:
    analysis = _analyze_expectations(
        durations=durations,
        expectations=measurement.expectations,
        sigma=measurement.expectation_errors,
        rabi_frequency=rabi_frequency,
    )
    fit = analysis["fit"]
    points = analysis["point_estimates"]
    derived = analysis["derived"]

    figure = _make_figure(
        target=target,
        durations=durations,
        points=points,
        fit=fit,
    )
    _print_fit_summary(
        target=target,
        fit=fit,
        rabi_frequency=rabi_frequency,
        shift=derived["shift"],
        shift_error=derived["shift_error"],
        axis_frequency=derived["axis_rotation_frequency"],
        axis_frequency_error=derived["axis_rotation_frequency_error"],
    )
    data = {
        "duration_range": durations,
        "total_pulse_duration_range": durations + 2.0 * ramp_duration,
        "ramp_duration": ramp_duration,
        "drive_amplitude": drive_amplitude,
        "rabi_frequency": rabi_frequency,
        "expectation_labels": EXPECTATION_LABELS,
        "mean_iq": measurement.mean_iq,
        "expectations": measurement.expectations,
        "expectation_errors": measurement.expectation_errors,
        "measurement_order": measurement.measurement_order,
        **analysis,
        "uncertainty_method": "shot_sem_and_robust_covariance",
    }
    return data, figure


def _analyze_expectations(
    *,
    durations: np.ndarray,
    expectations: np.ndarray,
    sigma: ArrayLike,
    rabi_frequency: float,
) -> dict[str, Any]:
    fit = _fit_global_model(durations, expectations, sigma=sigma)
    theta = fit["parameters"]["theta"]
    theta_error = fit["errors"]["theta"]
    points = _point_estimates(
        expectations,
        theta=theta,
        theta_error=theta_error,
        sigma=sigma,
    )
    # In Qubex's rotating-frame convention, the fitted Z component is
    # f_drive - f_transition = f_R * tan(theta).  The reported shift follows
    # the usual spectroscopy convention, f_transition - f_bare, and the
    # spin-lock carrier is the configured bare transition frequency.
    shift = -rabi_frequency * np.tan(theta)
    shift_error = abs(rabi_frequency / np.cos(theta) ** 2 * theta_error)
    axis_frequency = float(np.hypot(rabi_frequency, shift))
    axis_frequency_error = abs(
        rabi_frequency * np.sin(theta) / np.cos(theta) ** 2 * theta_error
    )
    diagnostics = _model_diagnostics(expectations, theta)
    diagnostics["a_outside_physical_range"] = (points["a"] < 0) | (points["a"] > 1)
    diagnostics["b_outside_physical_range"] = (points["b"] < 0) | (points["b"] > 1)

    return {
        "point_estimates": points,
        "fit": fit,
        "derived": {
            "shift": shift,
            "shift_error": shift_error,
            "axis_rotation_frequency": axis_frequency,
            "axis_rotation_frequency_error": axis_frequency_error,
        },
        "diagnostics": diagnostics,
    }


def _try_analyze_expectations(
    *,
    durations: np.ndarray,
    measurement: _MeasurementData,
    rabi_frequency: float,
) -> dict[str, Any] | Exception:
    """Analyze one frequency without aborting a spectroscopy sweep on failure."""
    try:
        return _analyze_expectations(
            durations=durations,
            expectations=measurement.expectations,
            sigma=measurement.expectation_errors,
            rabi_frequency=rabi_frequency,
        )
    except ANALYSIS_EXCEPTIONS as exc:
        return exc


def _spectroscopy_sort_indices(axis_frequencies: np.ndarray) -> np.ndarray:
    finite_indices = np.flatnonzero(np.isfinite(axis_frequencies))
    return finite_indices[np.argsort(axis_frequencies[finite_indices])]


def _make_parameter_heatmaps(
    *,
    target: str,
    durations: np.ndarray,
    axis_frequencies: np.ndarray,
    point_estimates: Mapping[str, np.ndarray],
) -> go.Figure:
    indices = _spectroscopy_sort_indices(axis_frequencies)
    frequencies_mhz = axis_frequencies[indices] * 1e3
    durations_us = durations * 1e-3
    figure = viz.make_figure(width=1200, height=360)
    figure.set_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=tuple(POINT_SUBPLOT_TITLES.values()),
        horizontal_spacing=0.1,
    )
    for column, name in enumerate(("a", "b", "p"), start=1):
        coloraxis = "coloraxis" if name in ("a", "b") else "coloraxis2"
        figure.add_trace(
            go.Heatmap(
                x=frequencies_mhz,
                y=durations_us,
                z=point_estimates[name][indices].T,
                coloraxis=coloraxis,
                hovertemplate=(
                    "Axis frequency: %{x:.5g} MHz<br>"
                    "Duration: %{y:.5g} μs<br>"
                    f"{POINT_DISPLAY_NAMES[name]}: %{{z:.5g}}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text="Axis rotation frequency (MHz)", row=1, col=column
        )
        figure.update_yaxes(type="log", row=1, col=column)
    figure.update_yaxes(title_text="Flattop duration (μs)", row=1, col=1)
    figure.update_layout(
        title=f"Spin-lock relaxation components: {target}",
        coloraxis={
            "colorscale": "Viridis",
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": "Retention /<br>coherence", "x": 0.64},
        },
        coloraxis2={
            "colorscale": "RdBu",
            "cmin": -1,
            "cmax": 1,
            "cmid": 0,
            "colorbar": {"title": "Polarization<br>offset", "x": 1.02},
        },
    )
    return figure


def _make_spectroscopy_summary_figure(
    *,
    target: str,
    axis_frequencies: np.ndarray,
    t1_rho: np.ndarray,
    t1_rho_error: np.ndarray,
    t2_rho: np.ndarray,
    t2_rho_error: np.ndarray,
    shift: np.ndarray,
    shift_error: np.ndarray,
    t1_plot_mask: np.ndarray,
    t2_plot_mask: np.ndarray,
) -> go.Figure:
    indices = _spectroscopy_sort_indices(axis_frequencies)
    frequencies_mhz = axis_frequencies[indices] * 1e3
    series = (
        ("T1ρ", t1_rho * 1e-3, t1_rho_error * 1e-3, "T1ρ (μs)", t1_plot_mask),
        ("T2ρ", t2_rho * 1e-3, t2_rho_error * 1e-3, "T2ρ (μs)", t2_plot_mask),
        (
            "shift",
            shift * 1e3,
            shift_error * 1e3,
            "Qubit frequency shift (MHz)",
            np.ones(axis_frequencies.shape, dtype=bool),
        ),
    )
    figure = viz.make_figure(width=1200, height=360)
    figure.set_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.08,
    )
    for column, (name, values, errors, ylabel, plot_mask) in enumerate(series, start=1):
        plotted_values = values[indices].copy()
        plotted_errors = errors[indices].copy()
        plotted_values[~plot_mask[indices]] = np.nan
        plotted_errors[~plot_mask[indices]] = np.nan
        figure.add_trace(
            go.Scatter(
                x=frequencies_mhz,
                y=plotted_values,
                error_y={"type": "data", "array": plotted_errors},
                mode="markers+lines",
                name=name,
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text="Axis rotation frequency (MHz)", row=1, col=column
        )
        yaxis_options: dict[str, Any] = {"title_text": ylabel}
        if column < 3:
            yaxis_options.update(rangemode="tozero", minallowed=0)
        figure.update_yaxes(row=1, col=column, **yaxis_options)
    figure.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        xref="x3 domain",
        yref="y3",
        line={"color": "gray", "dash": "dot"},
    )
    figure.update_layout(title=f"Spin-lock relaxation spectroscopy: {target}")
    return figure


def _initial_amplitude_plot_quality(
    *,
    fit_success: np.ndarray,
    fit_messages: list[str],
    amplitudes: np.ndarray,
    amplitude_name: str,
    minimum: float | None,
) -> tuple[np.ndarray, list[str | None]]:
    plot_mask = fit_success.copy()
    messages: list[str | None] = [
        None if success else f"Fit failed: {fit_messages[index]}"
        for index, success in enumerate(fit_success)
    ]
    for index in np.flatnonzero(fit_success):
        amplitude = amplitudes[index]
        if not np.isfinite(amplitude):
            plot_mask[index] = False
            messages[index] = f"{amplitude_name} is not finite."
        elif minimum is not None and amplitude < minimum:
            plot_mask[index] = False
            messages[index] = (
                f"{amplitude_name}={amplitude:.6g} is below "
                f"min_{amplitude_name}={minimum:.6g}."
            )
    return plot_mask, messages


def _assemble_spectroscopy_result(
    *,
    target: str,
    durations: np.ndarray,
    measurements: list[_MeasurementData],
    requested_frequencies: np.ndarray,
    drive_amplitudes: np.ndarray,
    rabi_frequencies: np.ndarray,
    ramp_duration: float,
    frequency_analyses: list[dict[str, Any] | Exception],
    min_a0: float | None = 0.5,
    min_b0: float | None = 0.5,
) -> tuple[dict[str, Any], dict[str, go.Figure]]:
    n_frequencies = rabi_frequencies.size
    n_durations = durations.size
    if len(measurements) != n_frequencies:
        raise ValueError("measurements must match rabi_frequencies.")
    point_estimates = {
        name: np.full((n_frequencies, n_durations), np.nan)
        for name in (
            "a",
            "a_error",
            "a_error_minus",
            "a_error_plus",
            "b",
            "b_error",
            "b_error_minus",
            "b_error_plus",
            "p",
            "p_error",
            "p_error_minus",
            "p_error_plus",
            "k",
            "q",
            "phi",
        )
    }
    values = {
        name: np.full(n_frequencies, np.nan)
        for name in (
            "theta",
            "theta_error",
            "t1_rho",
            "t1_rho_error",
            "t2_rho",
            "t2_rho_error",
            "p_inf",
            "p_inf_error",
            "a0",
            "a0_error",
            "b0",
            "b0_error",
            "shift",
            "shift_error",
            "axis_rotation_frequency",
            "axis_rotation_frequency_error",
        )
    }
    fit_results: list[dict[str, Any] | None] = []
    diagnostics: list[dict[str, Any] | None] = []
    fit_success = np.zeros(n_frequencies, dtype=bool)
    fit_messages: list[str] = []

    if len(frequency_analyses) != n_frequencies:
        raise ValueError("frequency_analyses must match rabi_frequencies.")

    for index, analysis in enumerate(frequency_analyses):
        if isinstance(analysis, Exception):
            fit_results.append(None)
            diagnostics.append(None)
            fit_messages.append(str(analysis))
            continue

        fit = analysis["fit"]
        derived = analysis["derived"]
        for name, estimates in analysis["point_estimates"].items():
            point_estimates[name][index] = estimates
        for name in ("theta", "t1_rho", "t2_rho", "p_inf", "a0", "b0"):
            values[name][index] = fit["parameters"][name]
            values[f"{name}_error"][index] = fit["errors"][name]
        for name in (
            "shift",
            "shift_error",
            "axis_rotation_frequency",
            "axis_rotation_frequency_error",
        ):
            values[name][index] = derived[name]
        fit_results.append(fit)
        diagnostics.append(analysis["diagnostics"])
        fit_success[index] = True
        fit_messages.append(str(fit["message"]))

    t1_plot_mask, t1_exclusion_messages = _initial_amplitude_plot_quality(
        fit_success=fit_success,
        fit_messages=fit_messages,
        amplitudes=values["a0"],
        amplitude_name="a0",
        minimum=min_a0,
    )
    t2_plot_mask, t2_exclusion_messages = _initial_amplitude_plot_quality(
        fit_success=fit_success,
        fit_messages=fit_messages,
        amplitudes=values["b0"],
        amplitude_name="b0",
        minimum=min_b0,
    )
    figures = {
        f"{target}_parameter_heatmaps": _make_parameter_heatmaps(
            target=target,
            durations=durations,
            axis_frequencies=values["axis_rotation_frequency"],
            point_estimates=point_estimates,
        ),
        f"{target}_relaxation_spectroscopy": _make_spectroscopy_summary_figure(
            target=target,
            axis_frequencies=values["axis_rotation_frequency"],
            t1_rho=values["t1_rho"],
            t1_rho_error=values["t1_rho_error"],
            t2_rho=values["t2_rho"],
            t2_rho_error=values["t2_rho_error"],
            shift=values["shift"],
            shift_error=values["shift_error"],
            t1_plot_mask=t1_plot_mask,
            t2_plot_mask=t2_plot_mask,
        ),
    }
    data = {
        "duration_range": durations,
        "total_pulse_duration_range": durations + 2.0 * ramp_duration,
        "ramp_duration": ramp_duration,
        "requested_rabi_frequencies": requested_frequencies,
        "drive_amplitudes": drive_amplitudes,
        "rabi_frequencies": rabi_frequencies,
        "expectation_labels": EXPECTATION_LABELS,
        "mean_iq": np.stack([item.mean_iq for item in measurements]),
        "expectations": np.stack([item.expectations for item in measurements]),
        "expectation_errors": np.stack(
            [item.expectation_errors for item in measurements]
        ),
        "measurement_order": np.stack(
            [item.measurement_order for item in measurements]
        ),
        "point_estimates": point_estimates,
        **values,
        "fit_results": fit_results,
        "fit_success": fit_success,
        "fit_messages": fit_messages,
        "t1_plot_mask": t1_plot_mask,
        "t2_plot_mask": t2_plot_mask,
        "t1_exclusion_messages": t1_exclusion_messages,
        "t2_exclusion_messages": t2_exclusion_messages,
        "min_a0": min_a0,
        "min_b0": min_b0,
        "diagnostics": diagnostics,
        "uncertainty_method": "shot_sem_and_robust_covariance",
    }
    return data, figures


# Public API


def spin_lock_relaxation(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    drive_amplitude: float | Mapping[str, float] | None = None,
    rabi_frequency: float | Mapping[str, float] | None = None,
    duration_range: ArrayLike | None = None,
    t1: float | Mapping[str, float] | None = None,
    t2_echo: float | Mapping[str, float] | None = None,
    n_duration_points: int | None = None,
    ramp_duration: float = DEFAULT_RAMP_DURATION,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = False,
) -> Result:
    """
    Measure tilted-axis spin-lock T1-rho and T2-rho relaxation.

    Parameters
    ----------
    exp
        Experiment used for pulse generation and measurement.
    targets
        Target qubits. Targets are measured independently in the given order.
    drive_amplitude
        Spin-lock command amplitude, either common to all targets or mapped by
        target. This takes precedence over `rabi_frequency`. When both drive
        arguments are omitted, the stored control amplitude is used.
    rabi_frequency
        Requested Rabi frequency in GHz, either common to all targets or mapped
        by target. It is converted to command amplitude only when
        `drive_amplitude` is omitted. The reported Rabi frequency is always
        recalculated from the amplitude actually used.
    duration_range
        Flat-top durations in ns. One explicit range is reused for every target.
        Otherwise, each target gets 18 logarithmic points from
        `max(100 ns, 0.025 * min(T1, T2_echo))` to
        `10 * max(T1, T2_echo)`. A single available coherence time is used for
        both limits; if neither is available, 21 points from 100 ns to 200 μs
        are used.
    t1
        T1 in ns used only to generate an automatic duration range. May be a
        common value or a target mapping. If either `t1` or `t2_echo` is given,
        YAML coherence times are not read.
    t2_echo
        T2 echo in ns used only to generate an automatic duration range. May be
        a common value or a target mapping.
    n_duration_points
        Number of logarithmic points in an automatic duration range. Must be at
        least 3. Defaults to 18 when a coherence time is available, or to 21
        for the fixed 100 ns--200 μs fallback. Ignored when `duration_range` is
        explicitly supplied.
    ramp_duration
        Duration of each raised-cosine ramp in ns. Defaults to 16 ns.
    n_shots
        Number of shots per expectation value. Must be at least 2 and defaults
        to `DEFAULT_SHOTS`.
    shot_interval
        Interval between shots in ns. Defaults to `DEFAULT_INTERVAL`.
    plot
        Whether to display the three-panel relaxation figure. Defaults to
        `True`.
    save_image
        Whether to save each relaxation figure. Defaults to `False`.

    Returns
    -------
    Result
        The `data["results"][target]` mapping contains the measured seven
        expectations and their SEM, mean IQ, pointwise `a`, `b`, `p`, `k`, `q`,
        and `phi`, the global fit, diagnostics, and derived frequencies.
        Durations are in ns and frequencies are in GHz. One three-panel figure
        is stored as `figures["{target}_relaxation"]`.
        Here `a` is longitudinal retention along the effective axis, `b` is
        transverse coherence magnitude, and `p` is the signed longitudinal
        polarization offset.

    Raises
    ------
    ValueError
        If an input, calibration value, duration range, or measured expectation
        value is invalid.
    RuntimeError
        If the global nonlinear fit does not converge.

    Notes
    -----
    This function performs hardware measurements sequentially for each target
    and prints one fit summary per target. Figures are displayed only when
    `plot=True` and saved only when `save_image=True`.

    Each requested duration is the flat-top duration; the total locking pulse
    lasts `duration + 2 * ramp_duration`. The fit uses all seven expectation
    values directly with one independent rotation phase per duration. Its
    decay model is `a0 exp(-t/T1rho)`, `b0 exp(-t/T2rho)`, and
    `p_inf + (p0 - p_inf) exp(-t/T1rho)`.

    The `duration x seven settings` points are measured in a new random order
    on every call to reduce correlation with time drift. Data are restored to
    canonical duration-major order before analysis. `measurement_order[j]` is
    the canonical flattened point index acquired at position `j`; its duration
    and setting indices are respectively `index // 7` and `index % 7`.

    The signed `shift` follows the spectroscopy convention
    `f_dressed - f_bare = -f_R tan(theta)`: a positive value means that the
    drive raises the qubit transition frequency. This interpretation assumes
    that the spin-lock carrier is the calibrated bare transition frequency;
    residual carrier detuning is indistinguishable from a drive-induced shift.
    Rabi frequency uncertainty is not included in the reported shift or axis
    rotation frequency uncertainty. Uncertainty in the stored IQ-normalization
    parameters is also not propagated. Each expectation-value uncertainty is
    the standard error calculated from the normalized single-shot IQ values.
    Global-fit errors use the larger of the local Jacobian error and the HC3
    robust error. Pointwise `a`, `b`, and `p` error bars are the 16th--84th
    percentile range from 500 Gaussian draws of the expectation values and
    fitted `theta`. This lightweight propagation is not a full bootstrap.
    """
    target_list, ramp_duration, n_shots, shot_interval = _resolve_measurement_options(
        exp,
        targets,
        ramp_duration=ramp_duration,
        n_shots=n_shots,
        shot_interval=shot_interval,
    )

    exp.pulse.validate_rabi_params(target_list)
    results: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for index, target in enumerate(target_list, start=1):
        durations = _resolve_target_durations(
            exp,
            target,
            duration_range=duration_range,
            t1=t1,
            t2_echo=t2_echo,
            n_duration_points=n_duration_points,
        )
        amplitude, estimated_rabi = _resolve_drive(
            exp,
            target=target,
            drive_amplitude=drive_amplitude,
            rabi_frequency=rabi_frequency,
        )
        print(
            f"({index}/{len(target_list)}) Measuring spin-lock relaxation for {target}."
        )
        measurement = _measure_target(
            exp,
            target=target,
            durations=durations,
            drive_amplitude=amplitude,
            ramp_duration=ramp_duration,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        target_result, figure = _analyze_target(
            target=target,
            durations=durations,
            measurement=measurement,
            drive_amplitude=amplitude,
            rabi_frequency=estimated_rabi,
            ramp_duration=ramp_duration,
        )
        results[target] = target_result
        figure_name = f"spin_lock_relaxation_{target}"
        figures[f"{target}_relaxation"] = figure
        _output_figure(
            figure,
            name=figure_name,
            plot=plot,
            save_image=save_image,
        )

    return Result(
        data={
            "targets": target_list,
            "results": results,
        },
        figure=figures.get(f"{target_list[0]}_relaxation"),
        figures=figures,
    )


def spin_lock_relaxation_spectroscopy(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    spin_lock_frequency_range: ArrayLike | None = None,
    duration_range: ArrayLike | None = None,
    t1: float | Mapping[str, float] | None = None,
    t2_echo: float | Mapping[str, float] | None = None,
    n_duration_points: int | None = None,
    ramp_duration: float = DEFAULT_RAMP_DURATION,
    min_a0: float | None = 0.5,
    min_b0: float | None = 0.5,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    enable_tqdm: bool = True,
    plot: bool = True,
    save_image: bool = False,
) -> Result:
    """
    Measure tilted-axis spin-lock relaxation over a Rabi-frequency sweep.

    Parameters
    ----------
    exp
        Experiment used for pulse generation and measurement.
    targets
        Target qubits. Targets are measured independently in the given order.
    spin_lock_frequency_range
        Requested positive Rabi frequencies in GHz. Defaults to 14 points from
        approximately 5.2 MHz to 120 MHz, with denser sampling at low
        frequencies. Zero is excluded because it cannot determine `shift` from
        `-f_R tan(theta)`.
    duration_range
        Flat-top durations in ns. One explicit range is reused for every target.
        When omitted, the same per-target T1/T2-echo defaults as
        `spin_lock_relaxation()` are used.
    t1
        T1 in ns used only to generate an automatic duration range. May be a
        common value or a target mapping. If either `t1` or `t2_echo` is given,
        YAML coherence times are not read.
    t2_echo
        T2 echo in ns used only to generate an automatic duration range. May be
        a common value or a target mapping.
    n_duration_points
        Number of logarithmic points in an automatic duration range. Must be at
        least 3. Defaults to 18 when a coherence time is available, or to 21
        for the fixed 100 ns--200 μs fallback. Ignored when `duration_range` is
        explicitly supplied.
    ramp_duration
        Duration of each raised-cosine ramp in ns. Defaults to 16 ns.
    min_a0
        Minimum fitted `a(0)` shown in the T1-rho summary plot. Smaller values
        remain in the returned data but are omitted from that plot. Defaults to
        0.5; use `None` to disable this filter.
    min_b0
        Minimum fitted `b(0)` shown in the T2-rho summary plot. Smaller values
        remain in the returned data but are omitted from that plot. Defaults to
        0.5; use `None` to disable this filter.
    n_shots
        Number of shots per expectation value. Must be at least 2 and defaults
        to `DEFAULT_SHOTS`.
    shot_interval
        Interval between shots in ns. Defaults to `DEFAULT_INTERVAL`.
    enable_tqdm
        Whether to show measurement progress by spin-lock frequency. Defaults
        to `True`.
    plot
        Whether to display the parameter heatmaps and spectroscopy summary.
    save_image
        Whether to save both figures for each target.

    Returns
    -------
    Result
        The `data["results"][target]` mapping contains frequency and duration
        axes, mean IQ, seven expectations and their SEM, `a`, `b`, and `p` maps,
        per-frequency fits, and summary arrays for `t1_rho`, `t2_rho`, `shift`,
        and axis rotation frequency. Failed fits are represented by NaN and
        described by `fit_success` and `fit_messages`. Durations are in ns and
        frequencies are in GHz.

        In the maps, `a` is longitudinal retention along the effective axis,
        `b` is transverse coherence magnitude, and `p` is the signed
        longitudinal polarization offset.

        Two figures are stored per target as
        `"{target}_parameter_heatmaps"` and
        `"{target}_relaxation_spectroscopy"`. Both use axis rotation frequency
        on the horizontal axis.

    Raises
    ------
    ValueError
        If an input, calibration value, duration range, or requested frequency
        is invalid.

    Notes
    -----
    Each frequency is converted to drive amplitude using the stored Rabi
    calibration and then converted back to the reported Rabi frequency. No
    chevron recalibration is performed. The seven-expectation global fit is
    independent at each frequency, while all durations at one frequency share
    one fitted `theta`.

    Each frequency uses one hardware sweep containing `n_durations * 7`
    measurement points. Frequencies and multiple targets are measured
    sequentially. A failed fit does not stop analysis of the remaining
    frequencies. Frequencies with failed fits remain in the returned measured
    data but are omitted from figures because their axis rotation frequency is
    not defined.

    Within each frequency, the `duration x seven settings` points are measured
    in an independently randomized order and restored to canonical order before
    analysis. Frequency order itself is unchanged. Row `i` of
    `measurement_order` records the canonical flattened indices in acquisition
    order for frequency `i`.

    The signed `shift` follows the spectroscopy convention
    `f_dressed - f_bare = -f_R tan(theta)`: a positive value means that the
    drive raises the qubit transition frequency. This interpretation assumes
    that the spin-lock carrier is the calibrated bare transition frequency;
    residual carrier detuning is indistinguishable from a drive-induced shift.
    Rabi-frequency calibration uncertainty is not included in the reported
    shift or axis rotation frequency uncertainty. Uncertainty in the stored
    IQ-normalization parameters is also not propagated. Each expectation-value
    uncertainty is the standard error calculated from the normalized
    single-shot IQ values. Global-fit errors use the larger of the local
    Jacobian error and the HC3 robust error. Pointwise `a`, `b`, and `p` error
    bars use 500 Gaussian draws and include the fitted `theta` uncertainty. No
    full bootstrap is performed.
    """
    target_list, ramp_duration, n_shots, shot_interval = _resolve_measurement_options(
        exp,
        targets,
        ramp_duration=ramp_duration,
        n_shots=n_shots,
        shot_interval=shot_interval,
    )

    requested_frequencies = _resolve_spin_lock_frequency_range(
        spin_lock_frequency_range
    )
    resolved_min_a0 = _resolve_min_initial_amplitude(min_a0, name="min_a0")
    resolved_min_b0 = _resolve_min_initial_amplitude(min_b0, name="min_b0")
    exp.pulse.validate_rabi_params(target_list)
    results: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for target_index, target in enumerate(target_list, start=1):
        durations = _resolve_target_durations(
            exp,
            target,
            duration_range=duration_range,
            t1=t1,
            t2_echo=t2_echo,
            n_duration_points=n_duration_points,
        )
        drive_amplitudes, rabi_frequencies = _resolve_frequency_sweep(
            exp,
            target=target,
            requested_frequencies=requested_frequencies,
        )
        points_per_frequency = durations.size * len(EXPECTATION_LABELS)
        print(
            f"({target_index}/{len(target_list)}) Measuring spin-lock relaxation "
            f"spectroscopy for {target} "
            f"({points_per_frequency} points per frequency)."
        )
        measurements: list[_MeasurementData] = []
        frequency_analyses: list[dict[str, Any] | Exception] = []
        frequency_progress = tqdm(
            zip(drive_amplitudes, rabi_frequencies, strict=True),
            total=rabi_frequencies.size,
            desc=f"Spin-lock spectroscopy: {target}",
            unit="freq",
            disable=not enable_tqdm,
        )
        for amplitude, rabi_frequency in frequency_progress:
            frequency_progress.set_postfix_str(f"{rabi_frequency * 1e3:.3f} MHz")
            measurement = _measure_target(
                exp,
                target=target,
                durations=durations,
                drive_amplitude=float(amplitude),
                ramp_duration=ramp_duration,
                n_shots=n_shots,
                shot_interval=shot_interval,
            )
            measurements.append(measurement)
            frequency_analyses.append(
                _try_analyze_expectations(
                    durations=durations,
                    measurement=measurement,
                    rabi_frequency=float(rabi_frequency),
                )
            )
        target_result, target_figures = _assemble_spectroscopy_result(
            target=target,
            durations=durations,
            measurements=measurements,
            requested_frequencies=requested_frequencies,
            drive_amplitudes=drive_amplitudes,
            rabi_frequencies=rabi_frequencies,
            ramp_duration=ramp_duration,
            min_a0=resolved_min_a0,
            min_b0=resolved_min_b0,
            frequency_analyses=frequency_analyses,
        )
        results[target] = target_result
        figures.update(target_figures)
        _print_spectroscopy_diagnostics(target_result, rabi_frequencies)
        for figure_name, figure in target_figures.items():
            _output_figure(
                figure,
                name=figure_name,
                plot=plot,
                save_image=save_image,
            )

    primary_figure = figures.get(f"{target_list[0]}_relaxation_spectroscopy")
    return Result(
        data={
            "targets": target_list,
            "spin_lock_frequency_range": requested_frequencies,
            "results": results,
        },
        figure=primary_figure,
        figures=figures,
    )
