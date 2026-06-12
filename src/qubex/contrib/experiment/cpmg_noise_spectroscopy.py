"""
Experimental CPMG noise spectroscopy utilities.

This module provides a frequency-swept CPMG workflow for exploratory noise
spectroscopy. The implementation intentionally lives in ``contrib`` because the
API and the physical interpretation of some edge cases, especially the
``f=0`` limit, remain experimental.

The public functions follow the same calling style as qubex experiment methods:
``exp`` is the first argument, ``targets`` is optional, measurement options are
keyword-only, and omitted options fall back to qubex defaults. Result payloads
are returned in a generic ``Result`` object and can also be reloaded from the
saved ``params.json`` and ``*.npy`` matrix files.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.experiment_util import ExperimentUtil
from qubex.experiment.models.experiment_result import T2Data
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, PulseSchedule, Waveform

from ._deprecated_options import resolve_shot_options

__all__ = [
    "CPMGResultPayload",
    "cpmg_noise_spectroscopy",
    "plot_cpmg_results",
]

CPMGPattern = Literal["++++", "+--+", "+-+-"]
CPMGSummaryMode = Literal["t2", "gamma2"]
AxisType = Literal["linear", "log"]

_DEFAULT_SAVE_DIR = "cpmg_noise_spectroscopy"
_GRID_ATOL_NS = 1e-9


class CPMGResultPayload(TypedDict):
    """Serializable CPMG spectroscopy result payload."""

    timestamp: str
    targets: list[str]
    n_repeats: int
    frequency_range: list[float]
    half_tau_range: list[float]
    tau_spacing_list_ns: list[float]
    time_range: list[float]
    t2_matrix: NDArray[np.float64]
    t2_error_matrix: NDArray[np.float64]
    r2_matrix: NDArray[np.float64]
    data: NDArray[np.float64]
    failed_fits: list[dict[str, object]]
    save_dir: str | None


@dataclass(frozen=True)
class _FrequencyPlan:
    """Realized CPMG timing for one effective-frequency point."""

    frequency: float
    half_tau: float


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
    allow_zero: bool,
) -> NDArray[np.float64]:
    """Validate and convert a one-dimensional numeric range."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if allow_zero:
        invalid = array < 0
        message = f"{name} must contain only non-negative values."
    else:
        invalid = array <= 0
        message = f"{name} must contain only positive values."
    if np.any(invalid):
        raise ValueError(message)
    return array


def _resolve_sampling_period(exp: Experiment) -> float:
    """Return the measurement sampling period in ns."""
    measurement = getattr(getattr(exp, "ctx", object()), "measurement", None)
    sampling_period = getattr(measurement, "sampling_period", None)
    if sampling_period is not None:
        return float(sampling_period)
    return float(ExperimentUtil.resolve_sampling_period())


def _discretize_time_range(
    exp: Experiment,
    time_range: NDArray[np.float64],
    *,
    sampling_period: float,
) -> NDArray[np.float64]:
    """Discretize a time range with the experiment utility when available."""
    util = getattr(getattr(exp, "ctx", object()), "util", None)
    discretize = getattr(util, "discretize_time_range", None)
    if callable(discretize):
        return np.asarray(
            discretize(time_range=time_range, sampling_period=sampling_period),
            dtype=np.float64,
        )
    return ExperimentUtil.discretize_time_range(
        time_range=time_range,
        sampling_period=sampling_period,
    )


def _sampling_ticks(duration_ns: float, *, sampling_period: float, name: str) -> int:
    """Return integer sampling ticks after checking grid alignment."""
    if not np.isfinite(duration_ns):
        raise ValueError(f"{name} must be finite, got {duration_ns}.")
    if sampling_period <= 0:
        raise ValueError(f"sampling_period must be positive, got {sampling_period}.")

    ticks = round(duration_ns / sampling_period)
    snapped = ticks * sampling_period
    if not np.isclose(duration_ns, snapped, rtol=0.0, atol=_GRID_ATOL_NS):
        raise ValueError(
            f"{name} must be on the {sampling_period} ns sampling grid; "
            f"got {duration_ns} ns."
        )
    return int(ticks)


def _snap_to_sampling_grid(duration_ns: float, *, sampling_period: float) -> float:
    """Round a duration to the nearest hardware sampling grid point."""
    if sampling_period <= 0:
        raise ValueError(f"sampling_period must be positive, got {sampling_period}.")
    return round(duration_ns / sampling_period) * sampling_period


def _floor_to_sampling_grid(
    values_ns: NDArray[np.float64],
    *,
    sampling_period: float,
) -> NDArray[np.float64]:
    """Floor durations to the hardware sampling grid."""
    if sampling_period <= 0:
        raise ValueError(f"sampling_period must be positive, got {sampling_period}.")
    return np.floor(values_ns / sampling_period) * sampling_period


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
    """Return ``value`` after checking that it follows the Qubex waveform API."""
    if isinstance(value, Waveform):
        return value
    raise TypeError(
        f"{source} for target `{target}` must return a Waveform, "
        f"got {type(value).__name__}."
    )


def _try_get_waveform(
    provider: object,
    *,
    target: str,
    source: str,
) -> Waveform | None:
    """Call a dynamic pulse provider and return a validated waveform."""
    if not callable(provider):
        return None
    try:
        value = provider(target)
    except Exception:
        return None
    return _require_waveform(value, source=source, target=target)


def _resolve_pi_pulse(
    exp: Experiment,
    target: str,
    pi_pulses: Mapping[str, Waveform] | None,
) -> Waveform:
    """Return the CPMG pi pulse for ``target``."""
    if pi_pulses is not None and target in pi_pulses:
        return _require_waveform(
            pi_pulses[target],
            source="pi_pulses",
            target=target,
        )

    pulse_service = getattr(exp, "pulse", None)
    pi_pulse = _try_get_waveform(
        getattr(pulse_service, "x180", None),
        target=target,
        source="exp.pulse.x180()",
    )
    if pi_pulse is not None:
        return pi_pulse.shifted(np.pi / 2)

    pi_pulse = _try_get_waveform(
        getattr(exp, "x180", None),
        target=target,
        source="exp.x180()",
    )
    if pi_pulse is not None:
        return pi_pulse.shifted(np.pi / 2)

    get_pi_pulse = getattr(pulse_service, "get_pi_pulse", None)
    pi_pulse = _try_get_waveform(
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
            schedule.add(target, Blank(half_tau))
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
            schedule.add(target, Blank(half_tau))
        schedule.add(target, hpi.scaled(-1))
    return schedule


def _build_no_pi_sequence(
    exp: Experiment,
    *,
    target: str,
    wait_ns: float,
) -> PulseSchedule:
    """Build the zero-frequency reference sequence without CPMG pi pulses."""
    if wait_ns < 0:
        raise ValueError(f"wait_ns must be non-negative, got {wait_ns}.")

    hpi = _get_hpi_pulse(exp, target)
    with PulseSchedule([target]) as schedule:
        schedule.add(target, hpi)
        if wait_ns > 0:
            schedule.add(target, Blank(wait_ns))
        schedule.add(target, hpi.scaled(-1))
    return schedule


def _make_frequency_plan(
    frequency_range: ArrayLike,
    *,
    sampling_period: float,
) -> list[_FrequencyPlan]:
    """Convert requested effective frequencies in GHz to realized timings."""
    requested = _as_non_empty_1d_array(
        frequency_range,
        name="frequency_range",
        allow_zero=True,
    )

    has_zero = bool(np.any(requested == 0))
    half_tau_ticks: list[int] = []
    for frequency in requested[requested > 0]:
        half_tau = 1.0 / (4.0 * float(frequency))
        half_tau = max(
            sampling_period,
            _snap_to_sampling_grid(half_tau, sampling_period=sampling_period),
        )
        ticks = _sampling_ticks(
            half_tau,
            sampling_period=sampling_period,
            name="half_tau",
        )
        half_tau_ticks.append(ticks)

    unique_ticks = np.sort(np.unique(np.asarray(half_tau_ticks, dtype=int)))[::-1]
    plan = [
        _FrequencyPlan(
            frequency=float(0.25 / (ticks * sampling_period)),
            half_tau=float(ticks * sampling_period),
        )
        for ticks in unique_ticks
    ]
    if has_zero:
        plan.insert(0, _FrequencyPlan(frequency=0.0, half_tau=0.0))
    if len(plan) == 0:
        raise ValueError("frequency_range must include at least one valid value.")
    return plan


def _default_frequency_range() -> NDArray[np.float64]:
    """Return a conservative default effective CPMG frequency grid in GHz."""
    return np.concatenate(([0.0], np.linspace(0.001, 0.025, 25)))


def _default_time_range() -> NDArray[np.float64]:
    """Return a default CPMG decay-time grid in ns."""
    return np.logspace(np.log10(300.0), np.log10(200_000.0), 51)


def _save_payload(
    *,
    payload: CPMGResultPayload,
    save_dir: Path,
) -> None:
    """Save result metadata and numeric arrays."""
    save_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        "timestamp": payload["timestamp"],
        "targets": payload["targets"],
        "n_repeats": payload["n_repeats"],
        "frequency_range": payload["frequency_range"],
        "half_tau_range": payload["half_tau_range"],
        "tau_spacing_list_ns": payload["half_tau_range"],
        "time_range": payload["time_range"],
        "failed_fits": payload["failed_fits"],
    }
    with (save_dir / "params.json").open("w", encoding="utf-8") as fp:
        json.dump(serializable, fp, indent=2)
    np.save(save_dir / "t2_matrix.npy", payload["t2_matrix"])
    np.save(save_dir / "t2_error_matrix.npy", payload["t2_error_matrix"])
    np.save(save_dir / "r2_matrix.npy", payload["r2_matrix"])


def _load_optional_matrix(
    save_dir: Path,
    name: str,
    *,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Load a matrix or return an all-NaN matrix for older save directories."""
    path = save_dir / name
    if path.exists():
        return np.asarray(np.load(path), dtype=np.float64)
    return np.full(shape, np.nan, dtype=np.float64)


def _load_payload(save_dir: str | Path) -> CPMGResultPayload:
    """Load a result payload saved by ``cpmg_noise_spectroscopy``."""
    directory = Path(save_dir)
    with (directory / "params.json").open("r", encoding="utf-8") as fp:
        params = json.load(fp)

    t2_matrix = np.asarray(np.load(directory / "t2_matrix.npy"), dtype=np.float64)
    half_tau_range = params.get("half_tau_range", params.get("tau_spacing_list_ns"))
    if half_tau_range is None:
        raise ValueError("params.json is missing `half_tau_range`.")

    return {
        "timestamp": str(params["timestamp"]),
        "targets": list(params["targets"]),
        "n_repeats": int(params["n_repeats"]),
        "frequency_range": [float(value) for value in params["frequency_range"]],
        "half_tau_range": [float(value) for value in half_tau_range],
        "tau_spacing_list_ns": [float(value) for value in half_tau_range],
        "time_range": [float(value) for value in params["time_range"]],
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
        "data": t2_matrix,
        "failed_fits": list(params.get("failed_fits", [])),
        "save_dir": str(directory),
    }


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
        "time_range",
        "t2_matrix",
        "data",
    }
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise ValueError(f"result is missing CPMG payload key(s): {missing}.")

    t2_matrix = np.asarray(payload["t2_matrix"], dtype=np.float64)
    payload.setdefault(
        "t2_error_matrix",
        np.full(t2_matrix.shape, np.nan, dtype=np.float64),
    )
    payload.setdefault(
        "r2_matrix",
        np.full(t2_matrix.shape, np.nan, dtype=np.float64),
    )
    payload.setdefault("failed_fits", [])
    payload.setdefault("save_dir", None)
    payload["t2_matrix"] = t2_matrix
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
        In-memory result returned by :func:`cpmg_noise_spectroscopy`, or a
        mapping with the same payload keys.
    save_dir
        Directory containing ``params.json`` and saved matrix files. When
        ``result`` is also provided, this directory is only used for saving the
        summary image.
    mode
        Summary mode. Use ``"t2"`` for T2_CPMG in us or ``"gamma2"`` for
        ``1 / T2_CPMG`` in ``1/us``.
    show_t1
        Overlay ``2*T1`` references when plotting ``mode="t2"``.
    plot
        Whether to display the figure.
    save_image
        Whether to save the summary figure in ``save_dir``.

    Returns
    -------
    plotly.graph_objects.Figure
        Summary figure.
    """
    show_t1, save_image = _resolve_plot_deprecated_options(
        show_t1=show_t1,
        save_image=save_image,
        deprecated_options=deprecated_options,
    )
    if mode is None:
        mode = "t2"
    if show_t1 is None:
        show_t1 = mode == "t2"
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if mode not in ("t2", "gamma2"):
        raise ValueError("mode must be either 't2' or 'gamma2'.")
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

    frequencies_mhz = np.asarray(payload["frequency_range"], dtype=np.float64) * 1e3
    t2_matrix = np.asarray(payload["t2_matrix"], dtype=np.float64)
    yaxis_title = (
        "Average T2_CPMG (us)" if mode == "t2" else "Average Gamma_2_CPMG (1/us)"
    )

    figure = viz.make_figure()
    for target_index, target in enumerate(targets):
        t2_average_ns = np.nanmean(t2_matrix[target_index], axis=0)
        finite = np.isfinite(t2_average_ns) & (t2_average_ns > 0)

        if mode == "t2":
            y_values = t2_average_ns * 1e-3
            trace_name = target
            if show_t1:
                try:
                    t1_values = exp.ctx.system_manager.config_loader.load_param_data(
                        "t1"
                    )
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
        else:
            y_values = np.full_like(t2_average_ns, np.nan, dtype=np.float64)
            y_values[finite] = 1.0 / (t2_average_ns[finite] * 1e-3)
            trace_name = f"{target} (1/T2)"

        figure.add_trace(
            go.Scatter(
                x=frequencies_mhz,
                y=y_values,
                mode="markers+lines",
                name=trace_name,
            )
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
        ),
        yaxis=dict(title=yaxis_title, showgrid=True),
        xaxis_type="linear",
        yaxis_type="linear",
        width=800,
        height=600,
    )

    if save_image:
        if directory is None:
            directory = Path(_DEFAULT_SAVE_DIR) / payload["timestamp"]
        directory.mkdir(parents=True, exist_ok=True)
        figure.write_image(directory / "cpmg_noise_spectroscopy_summary.png")
    if plot:
        figure.show()
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
    time_range: ArrayLike | None = None,
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
    Run a frequency-swept CPMG noise spectroscopy experiment.

    The function sweeps effective CPMG frequency and fits the decay versus total
    sequence time at each frequency. Requested frequencies are converted to
    realized half-tau values on the hardware sampling grid, so nearby requested
    frequencies may collapse to the same realized point.

    Parameters
    ----------
    exp
        Experiment instance used for pulse generation, measurement, and fitting.
    targets
        Target qubit labels. When omitted, all qubits in ``exp.ctx.qubit_labels``
        are measured.
    frequency_range
        Effective CPMG frequencies in GHz. ``0`` is allowed and is measured as a
        no-pi reference sequence.
    time_range
        Target total sequence durations in ns.
    pi_pulse_pattern
        Sign pattern for CPMG pi pulses. Supported values are ``"++++"``,
        ``"+--+"``, and ``"+-+-"``.
    pi_pulses
        Optional per-target pi pulses. When omitted, calibrated pi pulses are
        used, falling back to two half-pi pulses when necessary.
    n_repeats
        Number of repeated measurements per target and frequency.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in ns.
    plot
        Whether to display the summary figure.
    save_image
        Whether to save the summary figure.
    save_fit_images
        Whether to save each per-frequency decay-fit figure.
    save_data
        Whether to save ``params.json`` and result matrices under ``save_dir``.
    save_dir
        Base directory for saved outputs. A timestamped subdirectory is created.
    summary_mode
        Summary plot mode: ``"t2"`` or ``"gamma2"``.
    show_t1
        Overlay ``2*T1`` references on the summary T2 plot.
    xaxis_type
        X-axis scale used for decay fitting.
    enable_tqdm
        Whether to show progress bars.
    **deprecated_options
        Deprecated aliases from the first contrib implementation are accepted:
        ``shots``, ``interval``, ``plot_mode``, ``save_t2_image``,
        ``save_cpmg_data``, and ``save_cpmg_fig``.

    Returns
    -------
    Result
        Result containing the saved payload in ``data``, the summary figure in
        ``figure``, and per-target/per-frequency fit figures in ``figures``.
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
    if len(target_list) == 0:
        raise ValueError("targets must contain at least one target.")

    if frequency_range is None:
        frequency_range = _default_frequency_range()
    if time_range is None:
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
        summary_mode = "t2"
    if xaxis_type is None:
        xaxis_type = "log"
    if enable_tqdm is None:
        enable_tqdm = True

    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}.")
    if n_shots < 1:
        raise ValueError(f"n_shots must be at least 1, got {n_shots}.")
    if pi_pulse_pattern not in ("++++", "+--+", "+-+-"):
        raise ValueError("pi_pulse_pattern must be one of '++++', '+--+', '+-+-'.")
    if summary_mode not in ("t2", "gamma2"):
        raise ValueError("summary_mode must be either 't2' or 'gamma2'.")
    if xaxis_type not in ("linear", "log"):
        raise ValueError("xaxis_type must be either 'linear' or 'log'.")

    _validate_rabi_params(exp, target_list)
    sampling_period = _resolve_sampling_period(exp)

    requested_time_range = _as_non_empty_1d_array(
        time_range,
        name="time_range",
        allow_zero=True,
    )
    realized_time_range = _discretize_time_range(
        exp,
        requested_time_range,
        sampling_period=sampling_period,
    )
    realized_time_range = _as_non_empty_1d_array(
        realized_time_range,
        name="time_range",
        allow_zero=True,
    )

    frequency_plan = _make_frequency_plan(
        frequency_range,
        sampling_period=sampling_period,
    )
    realized_frequency_range = np.asarray(
        [point.frequency for point in frequency_plan],
        dtype=np.float64,
    )
    half_tau_range = np.asarray(
        [point.half_tau for point in frequency_plan],
        dtype=np.float64,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(save_dir) / timestamp
    if save_data or save_image or save_fit_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    shape = (len(target_list), n_repeats, len(frequency_plan))
    t2_matrix = np.full(shape, np.nan, dtype=np.float64)
    t2_error_matrix = np.full(shape, np.nan, dtype=np.float64)
    r2_matrix = np.full(shape, np.nan, dtype=np.float64)
    failed_fits: list[dict[str, object]] = []
    fit_figures: dict[str, go.Figure] = {}

    repeat_iterable = tqdm(
        range(n_repeats),
        desc="CPMG repetitions",
        disable=not enable_tqdm,
    )
    for repeat_index in repeat_iterable:
        target_iterable = tqdm(
            target_list,
            desc="targets",
            leave=False,
            disable=not enable_tqdm,
        )
        for target_index, target in enumerate(target_iterable):
            pi_duration = float(_resolve_pi_pulse(exp, target, pi_pulses).duration)
            _sampling_ticks(
                pi_duration,
                sampling_period=sampling_period,
                name=f"{target} pi pulse duration",
            )

            frequency_iterable = tqdm(
                enumerate(frequency_plan),
                total=len(frequency_plan),
                desc="frequencies",
                leave=False,
                disable=not enable_tqdm,
            )
            for frequency_index, point in frequency_iterable:
                if point.half_tau == 0:
                    sweep_range = _floor_to_sampling_grid(
                        realized_time_range,
                        sampling_period=sampling_period,
                    )
                    measurement = _sweep_parameter(
                        exp,
                        sequence=lambda wait_ns, target=target: _build_no_pi_sequence(
                            exp,
                            target=target,
                            wait_ns=float(wait_ns),
                        ),
                        sweep_range=sweep_range,
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        plot=False,
                        enable_tqdm=False,
                        title="CPMG zero-frequency reference",
                        xlabel="Time (ns)",
                        ylabel="Measured value",
                        xaxis_type=xaxis_type,
                    )
                    actual_time_range = np.asarray(sweep_range, dtype=np.float64)
                else:
                    unit_time = 2.0 * point.half_tau + pi_duration
                    n_pi_range = np.asarray(
                        sorted(
                            {
                                max(1, round(float(total_time) / unit_time))
                                for total_time in realized_time_range
                            }
                        ),
                        dtype=int,
                    )
                    actual_time_range = n_pi_range.astype(np.float64) * unit_time
                    measurement = _sweep_parameter(
                        exp,
                        sequence=lambda n_pi, target=target, half_tau=point.half_tau: (
                            _build_cpmg_sequence(
                                exp,
                                target=target,
                                half_tau=half_tau,
                                n_pi=int(n_pi),
                                pattern=pi_pulse_pattern,
                                pi_pulses=pi_pulses,
                                sampling_period=sampling_period,
                            )
                        ),
                        sweep_range=n_pi_range,
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
                try:
                    fit_result = fitting.fit_exp_decay(
                        target=target,
                        x=actual_time_range,
                        y=0.5 * (1 + sweep_data.normalized),
                        plot=False,
                        title="CPMG T2 decay",
                        xlabel="Time (us)",
                        ylabel="Normalized signal",
                        xaxis_type=xaxis_type,
                        yaxis_type="linear",
                    )
                except Exception as exc:
                    failed_fits.append(
                        {
                            "target": target,
                            "repeat_index": repeat_index,
                            "frequency": point.frequency,
                            "message": str(exc),
                        }
                    )
                    continue

                sweep_data_for_result = copy.deepcopy(sweep_data)
                sweep_data_for_result.sweep_range = actual_time_range
                if fit_result.status is FitStatus.SUCCESS:
                    t2_data = T2Data.new(
                        sweep_data_for_result,
                        t2=fit_result["tau"],
                        t2_err=fit_result["tau_err"],
                        r2=fit_result["r2"],
                    )
                    t2_matrix[target_index, repeat_index, frequency_index] = float(
                        t2_data.t2
                    )
                    t2_error_matrix[target_index, repeat_index, frequency_index] = (
                        float(t2_data.t2_err)
                    )
                    r2_matrix[target_index, repeat_index, frequency_index] = float(
                        t2_data.r2
                    )
                else:
                    failed_fits.append(
                        {
                            "target": target,
                            "repeat_index": repeat_index,
                            "frequency": point.frequency,
                            "message": fit_result.message,
                        }
                    )

                fit_figure = fit_result.figure
                if fit_figure is None:
                    continue
                figure_key = (
                    f"{target}_freq_{point.frequency * 1e3:.6f}_MHz_"
                    f"rep_{repeat_index + 1}"
                )
                fit_figures[figure_key] = fit_figure
                if save_fit_images:
                    target_dir = output_dir / target
                    target_dir.mkdir(parents=True, exist_ok=True)
                    fit_figure.write_image(target_dir / f"{figure_key}.png")

    save_dir_for_payload = (
        str(output_dir) if (save_data or save_image or save_fit_images) else None
    )
    payload: CPMGResultPayload = {
        "timestamp": timestamp,
        "targets": target_list,
        "n_repeats": n_repeats,
        "frequency_range": realized_frequency_range.tolist(),
        "half_tau_range": half_tau_range.tolist(),
        "tau_spacing_list_ns": half_tau_range.tolist(),
        "time_range": realized_time_range.tolist(),
        "t2_matrix": t2_matrix,
        "t2_error_matrix": t2_error_matrix,
        "r2_matrix": r2_matrix,
        "data": t2_matrix,
        "failed_fits": failed_fits,
        "save_dir": save_dir_for_payload,
    }

    if save_data:
        _save_payload(payload=payload, save_dir=output_dir)

    summary_figure = plot_cpmg_results(
        exp,
        payload,
        save_dir=output_dir,
        mode=summary_mode,
        show_t1=show_t1,
        plot=plot,
        save_image=save_image,
    )

    return Result(
        data=payload,
        figure=summary_figure,
        figures=fit_figures,
    )
