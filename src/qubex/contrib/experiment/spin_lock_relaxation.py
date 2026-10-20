"""Tilted-axis spin-lock relaxation measurement."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.models.result import Result
from qubex.pulse import FlatTop, PulseSchedule

__all__ = ["spin_lock_relaxation"]

DEFAULT_DURATION_RANGE = np.geomspace(100.0, 200e3, 31)
DEFAULT_RAMP_DURATION = 16.0
MIN_DURATION = 100.0
EXPECTATION_LABELS = ("X+", "Z+", "X-", "Z-", "Xi", "Yi", "Zi")
MEASUREMENT_SETTINGS: tuple[tuple[str, Literal["X", "Y", "Z"]], ...] = (
    ("+", "X"),
    ("+", "Z"),
    ("-", "X"),
    ("-", "Z"),
    ("+i", "X"),
    ("+i", "Y"),
    ("+i", "Z"),
)


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


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


def _default_duration_range(exp: Experiment, target: str) -> np.ndarray:
    try:
        loader = exp.ctx.system_manager.config_loader
        t1_values = loader.load_param_data("t1")
        t2_values = loader.load_param_data("t2_echo")
    except (AttributeError, KeyError, RuntimeError):
        return DEFAULT_DURATION_RANGE.copy()

    coherence_times = [
        value
        for value in (
            _coherence_value(t1_values, exp, target),
            _coherence_value(t2_values, exp, target),
        )
        if value is not None
    ]
    if not coherence_times:
        return DEFAULT_DURATION_RANGE.copy()

    start = max(MIN_DURATION, 0.025 * min(coherence_times))
    stop = 20.0 * max(coherence_times)
    if stop <= start:
        raise ValueError(
            f"Stored coherence time for `{target}` is too short to build a "
            f"default duration range above {MIN_DURATION:g} ns."
        )
    return np.geomspace(start, stop, 21)


def _resolve_duration_range(
    exp: Experiment,
    target: str,
    duration_range: ArrayLike | None,
) -> np.ndarray:
    durations = np.asarray(
        _default_duration_range(exp, target)
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
    sigma: float,
) -> dict[str, Any]:
    from scipy.optimize import least_squares

    times = np.asarray(durations, dtype=np.float64)
    measured = np.asarray(observations, dtype=np.float64)
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
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")

    theta0, point0 = _profile_initial_values(measured)
    p_values, a_values, k_values, q_values = point0.T
    b_values = np.hypot(k_values, q_values)
    t1_0 = _estimate_decay_time(times, np.abs(a_values))
    t2_0 = _estimate_decay_time(times, b_values)
    x0 = np.concatenate(
        (
            [
                theta0,
                t1_0,
                t2_0,
                np.clip(p_values[-1], -1.0, 1.0),
                np.clip(a_values[0] * np.exp(times[0] / t1_0), 0.01, 1.5),
                np.clip(b_values[0] * np.exp(times[0] / t2_0), 0.01, 1.5),
                np.clip(p_values[0], -1.0, 1.0),
            ],
            np.unwrap(np.arctan2(q_values, k_values)),
        )
    )

    lower = np.concatenate(
        ([-1.5, 1e-9, 1e-9, -1.0, 0.0, 0.0, -1.0], np.full(times.size, -np.inf))
    )
    upper = np.concatenate(
        ([1.5, np.inf, np.inf, 1.0, 1.5, 1.5, 1.0], np.full(times.size, np.inf))
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        predicted = _predict_expectations(
            times,
            theta=vector[0],
            t1_rho=vector[1],
            t2_rho=vector[2],
            p_inf=vector[3],
            a0=vector[4],
            b0=vector[5],
            p0=vector[6],
            phases=vector[7:],
        )
        return ((predicted - measured) / sigma).ravel()

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
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    parameters = {name: float(fit.x[index]) for index, name in enumerate(names)}
    parameter_errors = {name: float(errors[index]) for index, name in enumerate(names)}
    predicted = _predict_expectations(
        times,
        theta=fit.x[0],
        t1_rho=fit.x[1],
        t2_rho=fit.x[2],
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
        "phases": fit.x[7:].copy(),
        "phase_errors": errors[7:].copy(),
        "covariance": covariance,
        "parameter_order": [*names, *(f"phi_{index}" for index in range(times.size))],
        "predicted_expectations": predicted,
        "residuals": measured - predicted,
        "reduced_chi_squared": reduced_chi_squared,
    }


def _point_estimates(
    observations: np.ndarray,
    *,
    theta: float,
    sigma: float,
) -> dict[str, np.ndarray]:
    matrix = _design_matrix(theta)
    parameters = np.linalg.lstsq(matrix, observations.T, rcond=None)[0].T
    covariance = sigma**2 * np.linalg.pinv(matrix.T @ matrix)
    p, a, k, q = parameters.T
    b = np.hypot(k, q)
    b_error = np.empty_like(b)
    for index, (k_value, q_value, b_value) in enumerate(zip(k, q, b, strict=True)):
        if b_value > np.finfo(float).eps:
            gradient = np.asarray([k_value, q_value]) / b_value
            b_error[index] = np.sqrt(gradient @ covariance[2:4, 2:4] @ gradient)
        else:
            b_error[index] = np.sqrt(np.trace(covariance[2:4, 2:4]))
    return {
        "a": a,
        "a_error": np.full(a.shape, np.sqrt(covariance[1, 1])),
        "b": b,
        "b_error": b_error,
        "p": p,
        "p_error": np.full(p.shape, np.sqrt(covariance[0, 0])),
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
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, str, Literal["X", "Y", "Z"]]] = [
        (duration, initial_state, basis)
        for duration in durations
        for initial_state, basis in MEASUREMENT_SETTINGS
    ]

    def sequence(index: int) -> PulseSchedule:
        duration, initial_state, basis = points[int(index)]
        return _build_sequence(
            exp,
            target=target,
            flat_duration=float(duration),
            drive_amplitude=drive_amplitude,
            ramp_duration=ramp_duration,
            initial_state=initial_state,
            basis=basis,
        )

    with exp.ctx.util.no_output():
        sweep = exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=np.arange(len(points)),
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            title=f"Spin-lock relaxation: {target}",
            xlabel="Sweep point",
            ylabel="Pauli expectation",
        )
    target_data = sweep.data[target]
    return (
        np.asarray(target_data.data).reshape(durations.size, len(EXPECTATION_LABELS)),
        np.asarray(target_data.normalized, dtype=np.float64).reshape(
            durations.size, len(EXPECTATION_LABELS)
        ),
    )


def _normalization_noise(exp: Experiment, target: str) -> float:
    parameter = exp.pulse.rabi_params[target]
    amplitude = float(parameter.amplitude)
    raw_noise = float(parameter.noise)
    if not np.isfinite(amplitude) or amplitude == 0 or not np.isfinite(raw_noise):
        raise ValueError(f"Rabi normalization noise for `{target}` is not available.")
    noise = abs(raw_noise / amplitude)
    if noise == 0:
        raise ValueError(f"Rabi normalization noise for `{target}` must be positive.")
    return noise


def _make_figure(
    *,
    target: str,
    durations: np.ndarray,
    points: Mapping[str, np.ndarray],
    fit: Mapping[str, Any],
) -> go.Figure:
    figure = viz.make_figure(width=1200, height=480)
    figure.set_subplots(
        rows=1,
        cols=3,
        subplot_titles=("a(t)", "b(t)", "p(t)"),
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
                error_y={"type": "data", "array": points[f"{name}_error"]},
                mode="markers",
                marker={"color": colors[name]},
                name=f"{name} estimates",
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
                name=f"{name} fit",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text="Flattop duration (μs)", type="log", row=1, col=column
        )
        figure.update_yaxes(title_text=name, row=1, col=column)

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


def _analyze_target(
    exp: Experiment,
    *,
    target: str,
    durations: np.ndarray,
    raw_data: np.ndarray,
    expectations: np.ndarray,
    drive_amplitude: float,
    rabi_frequency: float,
    ramp_duration: float,
) -> tuple[dict[str, Any], go.Figure]:
    sigma = _normalization_noise(exp, target)
    fit = _fit_global_model(durations, expectations, sigma=sigma)
    theta = fit["parameters"]["theta"]
    theta_error = fit["errors"]["theta"]
    points = _point_estimates(expectations, theta=theta, sigma=sigma)
    shift = rabi_frequency * np.tan(theta)
    shift_error = abs(rabi_frequency / np.cos(theta) ** 2 * theta_error)
    axis_frequency = float(np.hypot(rabi_frequency, shift))
    axis_frequency_error = abs(
        rabi_frequency * np.sin(theta) / np.cos(theta) ** 2 * theta_error
    )
    diagnostics = _model_diagnostics(expectations, theta)
    diagnostics["a_outside_physical_range"] = (points["a"] < 0) | (points["a"] > 1)
    diagnostics["b_outside_physical_range"] = (points["b"] < 0) | (points["b"] > 1)

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
        shift=shift,
        shift_error=shift_error,
        axis_frequency=axis_frequency,
        axis_frequency_error=axis_frequency_error,
    )
    data = {
        "duration_range": durations,
        "total_pulse_duration_range": durations + 2.0 * ramp_duration,
        "ramp_duration": ramp_duration,
        "drive_amplitude": drive_amplitude,
        "rabi_frequency": rabi_frequency,
        "expectation_labels": EXPECTATION_LABELS,
        "raw_data": raw_data,
        "expectations": expectations,
        "expectation_errors": np.full(expectations.shape, sigma),
        "point_estimates": points,
        "fit": fit,
        "derived": {
            "shift": shift,
            "shift_error": shift_error,
            "axis_rotation_frequency": axis_frequency,
            "axis_rotation_frequency_error": axis_frequency_error,
        },
        "diagnostics": diagnostics,
        "uncertainty_method": "rabi_normalization_noise_and_fit_covariance",
    }
    return data, figure


def spin_lock_relaxation(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    drive_amplitude: float | Mapping[str, float] | None = None,
    rabi_frequency: float | Mapping[str, float] | None = None,
    duration_range: ArrayLike | None = None,
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
        Otherwise, each target gets 21 logarithmic points from
        `max(100 ns, 0.025 * min(T1, T2_echo))` to
        `20 * max(T1, T2_echo)`. A single available coherence time is used for
        both limits; if neither is available, 31 points from 100 ns to 200 μs
        are used.
    ramp_duration
        Duration of each raised-cosine ramp in ns. Defaults to 16 ns.
    n_shots
        Number of shots per expectation value. Defaults to `DEFAULT_SHOTS`.
    shot_interval
        Interval between shots in seconds. Defaults to `DEFAULT_INTERVAL`.
    plot
        Whether to display the three-panel relaxation figure. Defaults to
        `True`.
    save_image
        Whether to save each relaxation figure. Defaults to `False`.

    Returns
    -------
    Result
        The `data["results"][target]` mapping contains the measured seven
        expectations, pointwise `a`, `b`, `p`, `k`, `q`, and `phi` estimates,
        global-fit parameters and covariance, model diagnostics, and derived
        frequencies. Durations are in ns and frequencies are in GHz. One
        three-panel figure is stored as `figures["{target}_relaxation"]`.

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

    The signed `shift` is the fitted effective Z component, `f_R tan(theta)`.
    It equals an AC Stark shift only when no other detuning contributes. Rabi
    frequency uncertainty is not included in the reported shift or axis
    rotation frequency uncertainty. Expectation-value uncertainty is the
    stored Rabi normalization noise divided by its fitted amplitude. Pointwise
    error bars condition on the fitted `theta`; global parameter errors use the
    local Jacobian covariance of the seven-expectation fit.
    """
    target_list = _normalize_targets(exp, targets)
    if not target_list:
        raise ValueError("targets must contain at least one target.")
    if not np.isfinite(ramp_duration) or ramp_duration < 0:
        raise ValueError("ramp_duration must be finite and nonnegative.")
    if n_shots is not None and (
        isinstance(n_shots, bool) or not isinstance(n_shots, (int, np.integer))
    ):
        raise ValueError("n_shots must be a positive integer.")
    resolved_n_shots = int(DEFAULT_SHOTS if n_shots is None else n_shots)
    if resolved_n_shots <= 0:
        raise ValueError("n_shots must be a positive integer.")

    resolved_shot_interval = float(
        DEFAULT_INTERVAL if shot_interval is None else shot_interval
    )
    if not np.isfinite(resolved_shot_interval) or resolved_shot_interval < 0:
        raise ValueError("shot_interval must be finite and nonnegative.")

    exp.pulse.validate_rabi_params(target_list)
    results: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for index, target in enumerate(target_list, start=1):
        durations = _resolve_duration_range(exp, target, duration_range)
        amplitude, estimated_rabi = _resolve_drive(
            exp,
            target=target,
            drive_amplitude=drive_amplitude,
            rabi_frequency=rabi_frequency,
        )
        print(
            f"({index}/{len(target_list)}) Measuring spin-lock relaxation for {target}."
        )
        raw_data, expectations = _measure_target(
            exp,
            target=target,
            durations=durations,
            drive_amplitude=amplitude,
            ramp_duration=float(ramp_duration),
            n_shots=resolved_n_shots,
            shot_interval=resolved_shot_interval,
        )
        target_result, figure = _analyze_target(
            exp,
            target=target,
            durations=durations,
            raw_data=raw_data,
            expectations=expectations,
            drive_amplitude=amplitude,
            rabi_frequency=estimated_rabi,
            ramp_duration=float(ramp_duration),
        )
        results[target] = target_result
        figures[f"{target}_relaxation"] = figure
        if plot:
            figure.show(
                config=viz.get_config(filename=f"spin_lock_relaxation_{target}")
            )
        if save_image:
            viz.save_figure(figure, name=f"spin_lock_relaxation_{target}")

    return Result(
        data={
            "targets": target_list,
            "results": results,
        },
        figures=figures,
    )
