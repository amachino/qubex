"""
Orchestrate CR dissipation acquisition and user-facing output.

This module owns public-call validation, diagnostic IX45 calibration, hardware
acquisition, conversion to the records in `types`, plotting/reporting, and final
`Result` assembly. Pulse construction and numerical inference remain in
`pulses` and `analysis`, respectively. The stable facade is
`qubex.contrib.experiment.cr_dissipation`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from numbers import Integral, Real
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.interpolate import PchipInterpolator

from qubex import visualization as viz
from qubex.analysis import fitting
from qubex.contrib.experiment._single_shot_batch import measure_single_shot_batch
from qubex.contrib.experiment.gef_population_estimation import (
    bootstrap_gef_populations,
    measure_gef_populations,
)
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.pulse import FlatTop, PulseSchedule, Waveform

from .analysis import (
    BOUNDARY_FRACTION_CHANGE_THRESHOLD,
    CHANGE_SIGMA_THRESHOLD,
    LEAKAGE_MINIMUM_CHANGE,
    MODEL_SELECTION_DELTA_AICC,
    PARAMETER_SIGNIFICANCE_THRESHOLD,
    POPULATION_MINIMUM_CHANGE,
    PSD_RELATIVE_TOLERANCE,
    analyze_cr_dissipation,
    computational_polarization,
    nonnormalized_ge_expectation,
    simplex_boundary_bootstrap_required,
)
from .pulses import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    PROTOCOL_D,
    PROTOCOLS,
    ProtocolSchedules,
    ZX90Descriptor,
    append_analyzer,
    build_protocol_schedules,
    build_reference_schedules,
    resolve_zx90_descriptor,
)
from .types import (
    CrDissipationAnalysis,
    CrDissipationMeasurements,
    CrDissipationPauliData,
    CrDissipationProtocolData,
    CrDissipationProtocolMeasurements,
    CrDissipationRateStatus,
    GefPopulationSeries,
    IdleNoiseParameters,
)

DEFAULT_REPETITION_COUNTS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55)
DEFAULT_N_SHOTS = 2048
DEFAULT_GEF_MEASUREMENT_N_SHOTS = 4096
DEFAULT_GEF_CALIBRATION_N_SHOTS = 8192
_REFERENCE_CALIBRATION_NOTE_KEY = "cr_dissipation_reference_ix45"


# Input validation and stored idle-coherence resolution


@dataclass(frozen=True)
class _ReferenceIx45Calibration:
    """Store a CR-envelope IX45 calibration and cache fingerprint."""

    amplitude: float
    duration: float
    ramptime: float
    beta: float | None
    ramp_type: str
    sampling_period: float
    r_squared: float
    timestamp: str

    @property
    def fingerprint(self) -> tuple[float, float, float | None, str, float]:
        """Return pulse-shape fields used for cache compatibility."""
        return (
            self.duration,
            self.ramptime,
            self.beta,
            self.ramp_type,
            self.sampling_period,
        )


def _validate_repetition_counts(values: Sequence[int] | None) -> tuple[int, ...]:
    """Return the validated strictly increasing protocol repetition grid."""
    resolved = DEFAULT_REPETITION_COUNTS if values is None else tuple(values)
    if len(resolved) < 5:
        raise ValueError("repetition_counts must contain at least five points.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in resolved
    ):
        raise TypeError("repetition_counts must contain integers only.")
    normalized = tuple(int(value) for value in resolved)
    if normalized[0] != 0:
        raise ValueError("repetition_counts must start at zero.")
    if any(value < 0 for value in normalized):
        raise ValueError("repetition_counts must be nonnegative.")
    if any(right <= left for left, right in pairwise(normalized)):
        raise ValueError("repetition_counts must be unique and strictly increasing.")
    return normalized


def _positive_integer(value: int, name: str) -> int:
    """Validate an integer lower bound without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 2:
        raise ValueError(f"{name} must be at least two.")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    """Validate a nonnegative integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _optional_seed(value: int | None) -> int | None:
    """Validate an optional nonnegative bootstrap seed."""
    if value is None:
        return None
    return _nonnegative_integer(value, "gef_bootstrap_seed")


def _positive_finite(value: float, name: str) -> float:
    """Validate and normalize a positive finite real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return normalized


def _load_idle_noise(
    exp: Experiment,
    control: str,
    target: str,
    idle_t1: Mapping[str, float] | None,
    idle_t2_echo: Mapping[str, float] | None,
) -> tuple[IdleNoiseParameters, IdleNoiseParameters]:
    """Resolve fixed control and target idle-noise inputs before acquisition."""

    def load_stored(name: str) -> Mapping[str, float]:
        try:
            values = exp.ctx.system_manager.config_loader.load_param_data(name)
        except (
            AttributeError,
            FileNotFoundError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Stored idle `{name}` values could not be resolved before acquisition."
            ) from exc
        if not isinstance(values, Mapping):
            raise TypeError(f"Stored idle `{name}` data must be a mapping.")
        return values

    t1 = load_stored("t1") if idle_t1 is None else idle_t1
    t2 = load_stored("t2_echo") if idle_t2_echo is None else idle_t2_echo
    try:
        return (
            IdleNoiseParameters(t1[control], t2[control]),
            IdleNoiseParameters(t1[target], t2[target]),
        )
    except KeyError as exc:
        raise ValueError(
            f"Idle coherence data missing for qubit {exc.args[0]}."
        ) from exc


# CR-envelope IX45 reference calibration


def _flat_top_unit_area(pulse: FlatTop) -> float:
    """Return the sampled area of a unit-amplitude matching `FlatTop`."""
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
        raise ValueError("Reference pulse envelope must have positive finite area.")
    return area


def _make_ix45(
    cr_envelope: FlatTop, amplitude: float, *, allow_zero: bool = False
) -> FlatTop:
    """Copy a CR envelope with the requested IX45 amplitude."""
    lower = 0.0 if allow_zero else np.nextafter(0.0, 1.0)
    if not np.isfinite(amplitude) or not lower <= amplitude <= 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"reference_ix45_amplitude must be in {interval}.")
    return FlatTop(
        duration=cr_envelope.duration,
        amplitude=amplitude,
        tau=cr_envelope.tau,
        beta=cr_envelope.beta,
        type=cr_envelope.type,
        sampling_period=cr_envelope.sampling_period,
    )


def _reference_fingerprint(
    pulse: FlatTop,
) -> tuple[float, float, float | None, str, float]:
    """Return pulse-shape fields that determine IX45 cache compatibility."""
    return (
        float(pulse.duration),
        float(pulse.tau),
        None if pulse.beta is None else float(pulse.beta),
        str(pulse.type),
        float(pulse.sampling_period),
    )


def _calibrate_reference_ix45(
    exp: Experiment,
    target: str,
    cr_envelope: FlatTop,
    *,
    n_shots: int,
    shot_interval: float,
    plot: bool,
    enable_tqdm: bool,
) -> _ReferenceIx45Calibration:
    """Calibrate an IX45 amplitude using two repeated CR-shaped pulses."""
    _announce_acquisition(
        "Reference IX45 calibration (CR-shaped IX45 pair amplitude sweep)",
        enable_tqdm=enable_tqdm,
    )
    hpi = exp.pulse.get_hpi_pulse(target)
    if not isinstance(hpi, FlatTop):
        raise TypeError("The calibrated target hpi pulse must be a FlatTop pulse.")
    initial = (
        0.5
        * float(hpi.amplitude)
        * _flat_top_unit_area(hpi)
        / _flat_top_unit_area(cr_envelope)
    )
    amplitudes = np.linspace(max(0.0, 0.75 * initial), min(1.0, 1.25 * initial), 21)
    if amplitudes[0] >= amplitudes[-1]:
        amplitudes = np.linspace(0.0, 1.0, 21)

    def sequence(amplitude: float) -> dict[str, Waveform]:
        return {
            target: _make_ix45(cr_envelope, float(amplitude), allow_zero=True).repeated(
                2
            )
        }

    sweep = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=amplitudes,
        repetitions=8,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
        enable_tqdm=enable_tqdm,
    ).data[target]
    fit = fitting.fit_ampl_calib_data(
        target=target,
        amplitude_range=amplitudes,
        data=np.asarray(sweep.normalized, dtype=np.float64),
        plot=plot,
        title="CR-shaped IX45 pair calibration",
        ylabel="Normalized signal",
    )
    amplitude = float(fit["amplitude"])
    r_squared = float(fit["r2"])
    if (
        not np.isfinite(amplitude)
        or not 0.0 < amplitude <= 1.0
        or not np.isfinite(r_squared)
        or r_squared < 0.5
    ):
        raise RuntimeError(
            "CR-shaped IX45 calibration failed quality validation: "
            f"amplitude={amplitude!r}, r_squared={r_squared!r}."
        )
    duration, ramptime, beta, ramp_type, sampling_period = _reference_fingerprint(
        cr_envelope
    )
    return _ReferenceIx45Calibration(
        amplitude,
        duration,
        ramptime,
        beta,
        ramp_type,
        sampling_period,
        r_squared,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _resolve_reference_ix45(
    exp: Experiment,
    control: str,
    target: str,
    descriptor: ZX90Descriptor,
    *,
    amplitude: float | None,
    force: bool,
    n_shots: int,
    shot_interval: float,
    plot: bool,
    enable_tqdm: bool,
) -> tuple[FlatTop, _ReferenceIx45Calibration, bool]:
    """Resolve an explicit, cached, or newly calibrated reference IX45 pulse."""
    cr_envelope = getattr(descriptor.echoed, "cr_waveform", None)
    if not isinstance(cr_envelope, FlatTop):
        raise ValueError(  # noqa: TRY004 - incompatible overrides are ValueError
            "Reference acquisition requires a FlatTop ZX90 cr_waveform."
        )
    fingerprint = _reference_fingerprint(cr_envelope)
    if amplitude is not None:
        return (
            _make_ix45(cr_envelope, amplitude),
            _ReferenceIx45Calibration(
                amplitude,
                *fingerprint,
                float("nan"),
                "user-supplied",
            ),
            False,
        )
    key = f"{control}-{target}"
    cached: Any = None
    if not force:
        try:
            cached = exp.ctx.calib_note.get_property(
                _REFERENCE_CALIBRATION_NOTE_KEY,
                key,
                None,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            cached = None
    if isinstance(cached, Mapping):
        try:
            candidate = _ReferenceIx45Calibration(**cached)
        except (TypeError, ValueError):
            candidate = None
        if (
            candidate is not None
            and candidate.fingerprint == fingerprint
            and np.isfinite(candidate.amplitude)
            and 0.0 < candidate.amplitude <= 1.0
            and np.isfinite(candidate.r_squared)
            and candidate.r_squared >= 0.5
        ):
            return _make_ix45(cr_envelope, candidate.amplitude), candidate, False
    calibration = _calibrate_reference_ix45(
        exp,
        target,
        cr_envelope,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        enable_tqdm=enable_tqdm,
    )
    exp.ctx.calib_note.put_property(
        _REFERENCE_CALIBRATION_NOTE_KEY,
        key,
        asdict(calibration),
    )
    return _make_ix45(cr_envelope, calibration.amplitude), calibration, True


# Measurement acquisition and conversion to analysis data


def _gef_series(
    result: Result, names: Sequence[str], target: str
) -> GefPopulationSeries:
    """Collect named GEF fits into one ordered population series."""
    fits = [result.data["fits"][name][target] for name in names]  # type: ignore[index]
    bootstraps = [result.data["bootstrap"][name][target] for name in names]  # type: ignore[index]
    diagnostics = tuple(
        {
            "objective": fit.objective,
            "residual": fit.residual,
            "success": fit.success,
            "message": fit.message,
            "design_rank": fit.design_rank,
            "design_condition_number": fit.design_condition_number,
        }
        for fit in fits
    )
    return GefPopulationSeries(
        population=np.asarray([fit.population for fit in fits], dtype=np.float64),
        covariance=np.asarray(
            [fit.population_covariance for fit in fits], dtype=np.float64
        ),
        standard_error=np.asarray(
            [fit.population_standard_error for fit in fits], dtype=np.float64
        ),
        population_unconstrained=np.asarray(
            [fit.population_unconstrained for fit in fits], dtype=np.float64
        ),
        fit_diagnostics=diagnostics,
        bootstrap=tuple(bootstraps),
    )


def _protocol_data(
    result: Result,
    names: Sequence[str],
    control: str,
    target: str,
    protocol: str,
) -> CrDissipationProtocolData:
    """Convert one protocol's GEF result into its primary observables."""
    if protocol in (PROTOCOL_A, PROTOCOL_B):
        control_series = _gef_series(result, names, control)
        target_series = _gef_series(result, names, target)
        target_x, target_x_error = computational_polarization(target_series)
        sign = 1.0 if target_x[0] >= 0.0 else -1.0
        return CrDissipationProtocolData(
            control_gef=control_series,
            target_gef=target_series,
            target_x_comp=sign * target_x,
            target_x_comp_standard_error=target_x_error,
        )
    primary_series = _gef_series(
        result,
        names,
        control if protocol == PROTOCOL_C else target,
    )
    primary, primary_error, _ = nonnormalized_ge_expectation(primary_series)
    return CrDissipationProtocolData(
        control_gef=primary_series if protocol == PROTOCOL_C else None,
        target_gef=primary_series if protocol == PROTOCOL_D else None,
        primary_expectation=primary,
        primary_standard_error=primary_error,
    )


def _select_boundary_bootstrap_sequence_names(
    result: Result,
    names_by_protocol: Mapping[str, Sequence[str]],
    control: str,
    target: str,
) -> dict[str, tuple[str, ...]]:
    """Select A/B raw-IQ series requiring a boundary bootstrap diagnostic."""
    selected: dict[str, list[str]] = {control: [], target: []}
    for protocol in (PROTOCOL_A, PROTOCOL_B):
        data = _protocol_data(
            result,
            names_by_protocol[protocol],
            control,
            target,
            protocol,
        )
        for qubit, series in (
            (control, data.control_gef),
            (target, data.target_gef),
        ):
            if series is not None and simplex_boundary_bootstrap_required(
                series,
                2,
                minimum_change=LEAKAGE_MINIMUM_CHANGE,
            ):
                selected[qubit].extend(names_by_protocol[protocol])
    return {
        qubit: tuple(sequence_names)
        for qubit, sequence_names in selected.items()
        if sequence_names
    }


def _run_selective_ab_bootstrap(
    result: Result,
    names_by_protocol: Mapping[str, Sequence[str]],
    control: str,
    target: str,
    *,
    n_resamples: int,
    seed: int | None,
    confidence_level: float,
    covariance_rcond: float,
) -> dict[str, tuple[str, ...]]:
    """Bootstrap only A/B GEF series whose boundary diagnostics need it."""
    if n_resamples == 0:
        return {}
    selected = _select_boundary_bootstrap_sequence_names(
        result,
        names_by_protocol,
        control,
        target,
    )
    calibration = result.data["calibration"]
    raw_iq = result.data["raw_iq"]
    bootstrap = result.data["bootstrap"]
    for qubit, sequence_names in selected.items():
        selected_bootstrap = bootstrap_gef_populations(
            {qubit: calibration[qubit]},
            {name: raw_iq[name] for name in sequence_names},
            n_resamples=n_resamples,
            seed=seed,
            confidence_level=confidence_level,
            covariance_rcond=covariance_rcond,
        )
        for name, values in selected_bootstrap.items():
            bootstrap[name].update(values)
    return selected


def _named_sequences(
    schedules: Mapping[str, Sequence[PulseSchedule]],
    prefix: str,
    protocols: Sequence[str] = PROTOCOLS,
) -> tuple[dict[str, PulseSchedule], dict[str, tuple[str, ...]]]:
    """Assign stable result names to selected protocol schedules."""
    named: dict[str, PulseSchedule] = {}
    grouped: dict[str, tuple[str, ...]] = {}
    for protocol in protocols:
        names = tuple(
            f"{prefix}:{protocol}:{index}" for index in range(len(schedules[protocol]))
        )
        grouped[protocol] = names
        named.update(zip(names, schedules[protocol], strict=True))
    return named, grouped


def _orthogonal_acquisition(
    exp: Experiment,
    schedules: ProtocolSchedules,
    control: str,
    target: str,
    *,
    n_shots: int,
    shot_interval: float,
    enable_tqdm: bool,
) -> tuple[
    dict[str, CrDissipationPauliData],
    dict[str, dict[str, NDArray[np.complex128]]],
]:
    """Acquire every orthogonal diagnostic with calibrated Pauli readout."""
    requests_by_protocol = {
        PROTOCOL_A: ((control, "X"), (control, "Y"), (target, "Y"), (target, "Z")),
        PROTOCOL_B: ((control, "X"), (control, "Y"), (target, "Y"), (target, "Z")),
        PROTOCOL_C: ((control, "Y"), (control, "Z")),
        PROTOCOL_D: ((target, "X"), (target, "Z")),
    }

    def diagnostic_schedule(
        base: PulseSchedule,
        qubit: str,
        basis: str,
    ) -> PulseSchedule:
        if basis == "X":
            return append_analyzer(base, qubit, exp.pulse.y90m(qubit))
        if basis == "Y":
            return append_analyzer(base, qubit, exp.pulse.x90(qubit))
        return base

    named: dict[str, PulseSchedule] = {}
    keys: list[tuple[str, str, str]] = []
    for protocol, requests in requests_by_protocol.items():
        for qubit, basis in requests:
            for index, base in enumerate(schedules.base[protocol]):
                name = f"orthogonal:{protocol}:{qubit}:{basis}:{index}"
                named[name] = diagnostic_schedule(base, qubit, basis)
                keys.append((protocol, qubit, basis))
    _announce_acquisition(
        "A/B/C/D orthogonal Pauli diagnostics",
        enable_tqdm=enable_tqdm,
    )
    results = measure_single_shot_batch(
        exp,
        tuple(named.values()),
        n_shots=n_shots,
        shot_interval=shot_interval,
        enable_tqdm=enable_tqdm,
    )
    raw_iq: dict[str, dict[str, NDArray[np.complex128]]] = {}
    values_by_component: dict[tuple[str, str, str], list[float]] = {}
    errors_by_component: dict[tuple[str, str, str], list[float]] = {}
    for name, key, result in zip(named, keys, results, strict=True):
        protocol, qubit, basis = key
        if qubit not in result:
            raise ValueError(f"Pauli diagnostic `{name}` did not return `{qubit}`.")
        iq = result[qubit]
        normalized = np.asarray(
            exp.pulse.rabi_params[qubit].normalize(iq),
            dtype=np.float64,
        )
        if normalized.shape != (n_shots,) or not np.all(np.isfinite(normalized)):
            raise ValueError(
                f"Pauli diagnostic `{name}` normalization must return "
                f"{n_shots} finite shots."
            )
        values_by_component.setdefault(key, []).append(float(np.mean(normalized)))
        errors_by_component.setdefault(key, []).append(
            float(np.std(normalized, ddof=1) / np.sqrt(normalized.size))
        )
        raw_iq[name] = {qubit: iq}

    processed: dict[str, CrDissipationPauliData] = {}
    for protocol, requests in requests_by_protocol.items():
        components = {}
        errors = {}
        roles = {}
        for qubit, basis in requests:
            key = (protocol, qubit, basis)
            values = np.asarray(values_by_component[key], dtype=np.float64)
            standard_errors = np.asarray(errors_by_component[key], dtype=np.float64)
            name = f"{qubit}:{basis}"
            components[name] = values
            errors[name] = standard_errors
            roles[name] = "control" if qubit == control else "target"
        processed[protocol] = CrDissipationPauliData(
            expectations=components,
            standard_errors=errors,
            component_roles=roles,
        )
    return processed, raw_iq


def _build_measurements(
    schedules: ProtocolSchedules,
    repetition_counts: tuple[int, ...],
    actual_results: Mapping[str, Result],
    actual_names: Mapping[str, Sequence[str]],
    control: str,
    target: str,
    *,
    reference_results: Mapping[str, Result] | None,
    reference_names: Mapping[str, Sequence[str]] | None,
    orthogonal: Mapping[str, CrDissipationPauliData] | None,
) -> CrDissipationMeasurements:
    """Combine protocol results, timing, references, and Pauli diagnostics."""
    protocol_measurements: dict[str, CrDissipationProtocolMeasurements] = {}
    for protocol in PROTOCOLS:
        actual = _protocol_data(
            actual_results[protocol],
            actual_names[protocol],
            control,
            target,
            protocol,
        )
        reference = (
            None
            if reference_results is None or reference_names is None
            else _protocol_data(
                reference_results[protocol],
                reference_names[protocol],
                control,
                target,
                protocol,
            )
        )
        protocol_measurements[protocol] = CrDissipationProtocolMeasurements(
            schedules.elapsed_time_ns[protocol],
            schedules.cr_active_time_ns[protocol],
            actual,
            reference,
            None if orthogonal is None else orthogonal[protocol],
        )
    counts = np.asarray(repetition_counts, dtype=np.int64)
    return CrDissipationMeasurements(
        counts,
        4 * counts,
        protocol_measurements[PROTOCOL_A],
        protocol_measurements[PROTOCOL_B],
        protocol_measurements[PROTOCOL_C],
        protocol_measurements[PROTOCOL_D],
    )


def _merge_result_branch(
    results: Mapping[str, Result],
    key: str,
) -> dict[str, Any]:
    """Merge one named Result data branch from protocol-specific acquisitions."""
    merged: dict[str, Any] = {}
    seen: set[int] = set()
    for result in results.values():
        if id(result) in seen:
            continue
        seen.add(id(result))
        merged.update(result.data[key])  # type: ignore[arg-type]
    return merged


# Plotting and terminal reporting


_STATE_COLORS = dict(zip(("Pg", "Pe", "Pf"), fitting.COLORS[:3], strict=True))
_PRIMARY_COLOR = fitting.COLORS[0]
_PF_COLOR = _STATE_COLORS["Pf"]


def _smooth_plot_curve(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a shape-preserving dense interpolation for display only."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(valid) < 2:
        return x_values[valid], y_values[valid]
    retained_x = x_values[valid]
    retained_y = y_values[valid]
    dense_x = np.linspace(float(retained_x[0]), float(retained_x[-1]), 400)
    dense_y = PchipInterpolator(retained_x, retained_y)(dense_x)
    return dense_x, np.asarray(dense_y, dtype=np.float64)


def _line_trace(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    name: str,
    color: str,
    dash: str = "solid",
) -> go.Scatter:
    """Build one Qubex-style smooth line trace."""
    dense_x, dense_y = _smooth_plot_curve(x, y)
    return go.Scatter(
        x=dense_x,
        y=dense_y,
        mode="lines",
        name=name,
        line={"color": color, "dash": dash, "width": 2.5},
    )


def _announce_acquisition(description: str, *, enable_tqdm: bool) -> None:
    """Print an acquisition heading immediately before its progress bar."""
    if enable_tqdm:
        print(f"\nMeasuring: {description}", flush=True)


def _pf_axis_range(values: Sequence[NDArray[np.float64] | None]) -> list[float]:
    """Return a zero-anchored Pf range with a span of at least 0.05."""
    finite_values = [
        np.asarray(value, dtype=np.float64)[np.isfinite(value)]
        for value in values
        if value is not None
    ]
    finite_values = [value for value in finite_values if value.size]
    if not finite_values:
        return [0.0, 0.05]
    combined = np.concatenate(finite_values)
    lower = min(0.0, float(np.min(combined)))
    upper = max(0.0, float(np.max(combined)))
    if upper - lower < 0.05:
        upper = lower + 0.05
    return [lower, upper]


def _add_zero_line(figure: go.Figure, row: int) -> None:
    """Add a subtle zero baseline to one subplot."""
    # Plotly's type stub accepts axis IDs only, while the runtime also accepts
    # integer subplot coordinates.
    plotly_figure: Any = figure
    plotly_figure.add_hline(
        y=0.0,
        line={"color": "rgba(80, 80, 80, 0.45)", "width": 1},
        row=row,
        col=1,
    )


def _configure_protocol_xaxes(
    figure: go.Figure,
    *,
    rows: int,
    maximum_time_us: float,
    time_us: NDArray[np.float64],
    cr_gate_counts: Sequence[int],
) -> None:
    """Show elapsed time below and CR-gate counts above every subplot."""
    ticktext = [str(value) for value in cr_gate_counts]
    for row in range(1, rows + 1):
        suffix = "" if row == 1 else str(row)
        base_axis = f"x{suffix}"
        y_axis = f"y{suffix}"
        top_layout_key = f"xaxis{rows + row}"
        figure.update_xaxes(
            range=[0.0, maximum_time_us],
            tickvals=time_us,
            showticklabels=True,
            title_text="Elapsed sequence time [us]" if row == rows else None,
            row=row,
            col=1,
        )
        figure.update_layout(
            {
                top_layout_key: {
                    "overlaying": base_axis,
                    "anchor": y_axis,
                    "side": "top",
                    "range": [0.0, maximum_time_us],
                    "tickvals": time_us,
                    "ticktext": ticktext,
                    "showticklabels": True,
                    "showgrid": False,
                    "title": "CR gates (4n)" if row == 1 else None,
                }
            }
        )


def _shift_subplot_titles(figure: go.Figure, count: int) -> None:
    """Move only subplot-title annotations below their upper tick labels."""
    annotations: Any = figure.layout.annotations
    for annotation in annotations[:count]:
        annotation.update(yshift=-16)


def _marker_trace(
    x: NDArray[np.float64],
    y: NDArray[np.float64] | None,
    *,
    name: str,
    color: str,
    error: NDArray[np.float64] | None = None,
    symbol: str = "circle",
) -> go.Scatter:
    """Build one Qubex-style marker trace with optional statistical errors."""
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name=name,
        marker={"color": color, "size": 7, "symbol": symbol},
        error_y=None
        if error is None
        else {"type": "data", "array": error, "visible": True},
    )


def _plot_cr_dissipation(
    measurements: CrDissipationMeasurements,
    analysis: CrDissipationAnalysis,
) -> dict[str, go.Figure]:
    """Build consistently styled protocol figures with display-only smoothing."""
    figures: dict[str, go.Figure] = {}
    maximum_time_us = (
        max(
            float(np.max(getattr(measurements, protocol).elapsed_time_ns))
            for protocol in PROTOCOLS
        )
        / 1000.0
    )
    for protocol in PROTOCOLS:
        measured = getattr(measurements, protocol)
        time_us = np.asarray(measured.elapsed_time_ns / 1000.0, dtype=np.float64)
        has_orthogonal = measured.orthogonal is not None
        if protocol in (PROTOCOL_A, PROTOCOL_B):
            pf_values: list[NDArray[np.float64] | None] = []
            if has_orthogonal:
                titles = [
                    "Control GEF population",
                    "Control orthogonal diagnostics",
                    "Target Xcomp",
                    "Target Pf",
                    "Target orthogonal diagnostics",
                ]
                target_x_row = 3
                target_pf_row = 4
                control_orthogonal_row = 2
                target_orthogonal_row = 5
            else:
                titles = ["Control GEF population", "Target Xcomp", "Target Pf"]
                target_x_row = 2
                target_pf_row = 3
                control_orthogonal_row = None
                target_orthogonal_row = None
            figure = make_subplots(
                rows=len(titles),
                cols=1,
                shared_xaxes=True,
                subplot_titles=titles,
                vertical_spacing=0.06,
            )
            control = measured.actual.control_gef
            fitted = analysis.fits.control_population_ab.fitted_populations.get(
                protocol
            )
            idle = analysis.idle_predictions[protocol].observables
            idle_control = idle.get("control_population")
            if control is not None:
                for index, state in enumerate(("Pg", "Pe", "Pf")):
                    color = _STATE_COLORS[state]
                    figure.add_trace(
                        _marker_trace(
                            time_us,
                            control.population[:, index],
                            error=control.standard_error[:, index],
                            name=f"actual {state}",
                            color=color,
                        ),
                        row=1,
                        col=1,
                    )
                    if fitted is not None:
                        figure.add_trace(
                            _line_trace(
                                time_us,
                                fitted[:, index],
                                name=f"fit {state}",
                                color=color,
                            ),
                            row=1,
                            col=1,
                        )
                    if idle_control is not None:
                        figure.add_trace(
                            _line_trace(
                                time_us,
                                idle_control[:, index],
                                name=f"idle-only {state}",
                                color=color,
                                dash="dash",
                            ),
                            row=1,
                            col=1,
                        )
            if measured.actual.target_x_comp is not None:
                figure.add_trace(
                    _marker_trace(
                        time_us,
                        measured.actual.target_x_comp,
                        error=measured.actual.target_x_comp_standard_error,
                        name="actual Xcomp",
                        color=_PRIMARY_COLOR,
                    ),
                    row=target_x_row,
                    col=1,
                )
            target = measured.actual.target_gef
            if target is not None:
                pf_values.append(target.population[:, 2])
                figure.add_trace(
                    _marker_trace(
                        time_us,
                        target.population[:, 2],
                        error=target.standard_error[:, 2],
                        name="actual target Pf",
                        color=_PF_COLOR,
                    ),
                    row=target_pf_row,
                    col=1,
                )
            t1_fit = (
                analysis.fits.target_a_t1rho
                if protocol == PROTOCOL_A
                else analysis.fits.target_b_t1rho
            )
            leakage_fit = (
                analysis.fits.target_a_leakage
                if protocol == PROTOCOL_A
                else analysis.fits.target_b_leakage
            )
            for row, values, name, color in (
                (
                    target_x_row,
                    t1_fit.fitted_values,
                    "T1rho fit",
                    _PRIMARY_COLOR,
                ),
                (
                    target_pf_row,
                    leakage_fit.fitted_values,
                    "leak/seep fit",
                    _PF_COLOR,
                ),
                (
                    target_x_row,
                    idle["target_x_comp"],
                    "idle-only Xcomp",
                    _PRIMARY_COLOR,
                ),
                (
                    target_pf_row,
                    idle["target_pf"],
                    "idle-only Pf",
                    _PF_COLOR,
                ),
            ):
                if row == target_pf_row:
                    pf_values.append(values)
                figure.add_trace(
                    _line_trace(
                        time_us,
                        values,
                        name=name,
                        color=color,
                        dash="dash" if name.startswith("idle-only") else "solid",
                    ),
                    row=row,
                    col=1,
                )
            if measured.reference is not None:
                reference_control = measured.reference.control_gef
                if reference_control is not None:
                    for index, state in enumerate(("Pg", "Pe", "Pf")):
                        figure.add_trace(
                            _marker_trace(
                                time_us,
                                reference_control.population[:, index],
                                name=f"reference {state}",
                                color=_STATE_COLORS[state],
                                symbol="circle-open",
                            ),
                            row=1,
                            col=1,
                        )
                if measured.reference.target_x_comp is not None:
                    figure.add_trace(
                        _marker_trace(
                            time_us,
                            measured.reference.target_x_comp,
                            name="reference Xcomp",
                            color=_PRIMARY_COLOR,
                            symbol="circle-open",
                        ),
                        row=target_x_row,
                        col=1,
                    )
                reference_target = measured.reference.target_gef
                if reference_target is not None:
                    pf_values.append(reference_target.population[:, 2])
                    figure.add_trace(
                        _marker_trace(
                            time_us,
                            reference_target.population[:, 2],
                            name="reference target Pf",
                            color=_PF_COLOR,
                            symbol="circle-open",
                        ),
                        row=target_pf_row,
                        col=1,
                    )
            figure.update_yaxes(range=[0.0, 1.0], row=1, col=1)
            figure.update_yaxes(range=[-1.0, 1.0], row=target_x_row, col=1)
            figure.update_yaxes(
                range=_pf_axis_range(pf_values), row=target_pf_row, col=1
            )
            _add_zero_line(figure, 1)
            _add_zero_line(figure, target_x_row)
            _add_zero_line(figure, target_pf_row)
            if has_orthogonal and measured.orthogonal is not None:
                if control_orthogonal_row is None or target_orthogonal_row is None:
                    raise ValueError("Missing A/B orthogonal plot rows.")
                for index, (name, component) in enumerate(
                    measured.orthogonal.expectations.items()
                ):
                    role = measured.orthogonal.component_roles.get(name)
                    if role == "control":
                        row = control_orthogonal_row
                    elif role == "target":
                        row = target_orthogonal_row
                    else:
                        raise ValueError(
                            f"Orthogonal component `{name}` has no valid qubit role."
                        )
                    if row is None:
                        raise ValueError(
                            f"Orthogonal component `{name}` has no assigned plot row."
                        )
                    figure.add_trace(
                        _marker_trace(
                            time_us,
                            component,
                            error=measured.orthogonal.standard_errors.get(name),
                            name=name,
                            color=fitting.COLORS[(index + 3) % len(fitting.COLORS)],
                        ),
                        row=row,
                        col=1,
                    )
                figure.update_yaxes(
                    range=[-1.0, 1.0], row=control_orthogonal_row, col=1
                )
                figure.update_yaxes(range=[-1.0, 1.0], row=target_orthogonal_row, col=1)
                _add_zero_line(figure, control_orthogonal_row)
                _add_zero_line(figure, target_orthogonal_row)
        else:
            titles = ["Control Xge" if protocol == PROTOCOL_C else "Target Yge"]
            if has_orthogonal:
                titles.append("Orthogonal diagnostics")
            figure = make_subplots(
                rows=len(titles),
                cols=1,
                shared_xaxes=True,
                subplot_titles=titles,
                vertical_spacing=0.06,
            )
            values = measured.actual.primary_expectation
            errors = measured.actual.primary_standard_error
            if values is not None:
                figure.add_trace(
                    _marker_trace(
                        time_us,
                        values,
                        error=errors,
                        name="actual",
                        color=_PRIMARY_COLOR,
                    ),
                    row=1,
                    col=1,
                )
                fit = (
                    analysis.fits.control_pure_dephasing_c
                    if protocol == PROTOCOL_C
                    else analysis.fits.target_rotating_frame_pure_dephasing_d
                )
                figure.add_trace(
                    _line_trace(
                        time_us,
                        fit.fitted_values,
                        name="physical fit",
                        color=_PRIMARY_COLOR,
                    ),
                    row=1,
                    col=1,
                )
            idle_values = analysis.idle_predictions[protocol].observables.get(
                "primary_expectation"
            )
            if idle_values is not None:
                figure.add_trace(
                    _line_trace(
                        time_us,
                        idle_values,
                        name="idle-only prediction",
                        color=_PRIMARY_COLOR,
                        dash="dash",
                    ),
                    row=1,
                    col=1,
                )
            if measured.reference is not None:
                figure.add_trace(
                    _marker_trace(
                        time_us,
                        measured.reference.primary_expectation,
                        name="reference (diagnostic)",
                        color=_PRIMARY_COLOR,
                        symbol="circle-open",
                    ),
                    row=1,
                    col=1,
                )
            figure.update_yaxes(range=[-1.0, 1.0], row=1, col=1)
            _add_zero_line(figure, 1)
            if has_orthogonal and measured.orthogonal is not None:
                for index, (name, component) in enumerate(
                    measured.orthogonal.expectations.items()
                ):
                    figure.add_trace(
                        _marker_trace(
                            time_us,
                            component,
                            error=measured.orthogonal.standard_errors.get(name),
                            name=name,
                            color=fitting.COLORS[(index + 3) % len(fitting.COLORS)],
                        ),
                        row=2,
                        col=1,
                    )
                figure.update_yaxes(range=[-1.0, 1.0], row=2, col=1)
                _add_zero_line(figure, 2)
            if protocol == PROTOCOL_C and measured.reference is not None:
                figure.add_annotation(
                    text=(
                        "Reference is diagnostic only and is not expected to coincide "
                        "with the actual idle-only prediction."
                    ),
                    xref="paper",
                    yref="paper",
                    x=0.0,
                    y=-0.14,
                    showarrow=False,
                )
        figure.update_layout(
            template=viz.DEFAULT_TEMPLATE,
            title={"text": protocol, "x": 0.5, "xanchor": "center", "y": 0.995},
            height=245 * len(titles) + 80,
            margin={"t": 95, "r": 210, "b": 75, "l": 75},
            legend={"x": 1.02, "xanchor": "left", "y": 1.0, "yanchor": "top"},
        )
        _shift_subplot_titles(figure, len(titles))
        _configure_protocol_xaxes(
            figure,
            rows=len(titles),
            maximum_time_us=maximum_time_us,
            time_us=time_us,
            cr_gate_counts=tuple(int(value) for value in measurements.cr_gate_counts),
        )
        figures[protocol] = figure
    return figures


def _print_summary(
    analysis: CrDissipationAnalysis,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
) -> None:
    """Print rate and fidelity summaries for an interactive measurement."""
    print("CR dissipation characterization")
    print(
        "Idle inputs [us]: "
        f"control T1={control_idle.t1_ns / 1000:.3g}, "
        f"T2_echo={control_idle.t2_echo_ns / 1000:.3g}; "
        f"target T1={target_idle.t1_ns / 1000:.3g}, "
        f"T2_echo={target_idle.t2_echo_ns / 1000:.3g}"
    )
    headings = (
        "Indicator",
        "CR-active rate [/us]",
        "Equivalent time [us]",
        "Idle-only equivalent [us]",
        "Status",
    )
    rows: list[tuple[str, str, str, str, str]] = []
    for name, estimate in analysis.rates.items():
        failed = estimate.status == CrDissipationRateStatus.FIT_FAILED
        rate = "—" if failed else f"{estimate.rate_per_ns * 1000.0:.3g}"
        if not failed and estimate.rate_standard_error_per_ns is not None:
            rate += f" ± {estimate.rate_standard_error_per_ns * 1000.0:.3g}"
        lifetime = (
            "—"
            if failed or not np.isfinite(estimate.lifetime_ns)
            else f"{estimate.lifetime_ns / 1000.0:.3g}"
        )
        if estimate.lifetime_ns == float("inf"):
            lifetime = "inf"
        if not failed and estimate.lifetime_interval_1sigma_ns is not None:
            low, high = estimate.lifetime_interval_1sigma_ns
            high_text = "inf" if np.isinf(high) else f"{high / 1000.0:.3g}"
            lifetime += f" [{low / 1000.0:.3g}, {high_text}]"
        idle_lifetime = estimate.idle_equivalent_lifetime_ns
        idle = (
            "—"
            if np.isnan(idle_lifetime)
            else "inf"
            if np.isinf(idle_lifetime)
            else f"{idle_lifetime / 1000.0:.3g}"
        )
        rows.append((name, rate, lifetime, idle, estimate.status.value))
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    ]
    print(
        "  ".join(
            value.ljust(width) for value, width in zip(headings, widths, strict=True)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                value.ljust(width) for value, width in zip(row, widths, strict=True)
            )
        )
    print(
        "± values are conditional statistical 1σ uncertainties with supplied "
        "idle T1/T2_echo and fitted GEF calibration held fixed; selected cross-fit "
        "and cross-protocol covariances are neglected."
    )
    for label, estimate in (
        ("Idle coherence limited", analysis.fidelity.idle_coherence_limited),
        ("CR-on coherence limited", analysis.fidelity.cr_on_coherence_limited),
        ("CR-on dissipative limited", analysis.fidelity.cr_on_dissipative_limited),
    ):
        if not estimate.available:
            print(f"  {label}: unavailable ({estimate.message})")
        elif estimate.standard_error is None:
            print(f"  {label}: {estimate.value:.8f}")
        else:
            print(f"  {label}: {estimate.value:.8f} ± {estimate.standard_error:.2g}")
    if analysis.warnings:
        print("Warnings:")
        for item in analysis.warnings:
            print(f"  [{item.code}] {item.message}")


# Public experiment orchestration


def characterize_cr_dissipation(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    repetition_counts: Sequence[int] | None = None,
    n_shots: int = DEFAULT_N_SHOTS,
    gef_measurement_n_shots: int = DEFAULT_GEF_MEASUREMENT_N_SHOTS,
    shot_interval: float | None = None,
    measure_reference: bool = False,
    measure_orthogonal_components: bool = False,
    zx90_echo: PulseSchedule | None = None,
    idle_t1: Mapping[str, float] | None = None,
    idle_t2_echo: Mapping[str, float] | None = None,
    gef_calibration_n_shots: int = DEFAULT_GEF_CALIBRATION_N_SHOTS,
    gef_bootstrap_n_resamples: int = 1000,
    gef_bootstrap_seed: int | None = 0,
    gef_bootstrap_confidence_level: float = 0.95,
    gef_covariance_rcond: float = 1e-12,
    reference_ix45_amplitude: float | None = None,
    force_reference_ix45_calibration: bool = False,
    enable_tqdm: bool = True,
    plot: bool = True,
) -> Result:
    """
    Characterize CR-active dissipation and compute three fidelity limits.

    All public time inputs and raw timings are in ns; rates are in 1/ns. C/D
    use the non-normalized qutrit observable `Pg-Pe`. Reference and orthogonal
    acquisitions are diagnostic-only and never enter primary inference.

    Parameters
    ----------
    exp : Experiment
        Experiment used for pulse construction, acquisition, and calibration
        storage.
    control_qubit, target_qubit : str
        Distinct qubit labels defining the calibrated CR pair.
    repetition_counts : Sequence[int] | None, optional
        Strictly increasing nonnegative repetition counts starting at zero.
        At least five points are required. Defaults to a Fibonacci-like grid.
    n_shots : int, optional
        Shots per non-GEF measurement configuration, including all orthogonal
        Pauli diagnostics and IX45 reference calibration. Must be at least two to
        estimate a shot-to-shot standard error. Defaults to 2048.
    gef_measurement_n_shots : int, optional
        Shots per GEF population measurement configuration. Used by A/B/C/D
        primary and GEF reference acquisitions. Must be at least two and defaults
        to 4096.
    shot_interval : float | None, optional
        Interval between shots in ns. When omitted, use the configured
        measurement default.
    measure_reference : bool, optional
        Whether to acquire duration-matched diagnostic reference sequences.
        Reference data never enter primary fits or fidelity estimates.
    measure_orthogonal_components : bool, optional
        Whether to acquire diagnostic components orthogonal to each primary
        observable using Pauli readout. These data never enter primary inference.
    zx90_echo : PulseSchedule | None, optional
        Optional echoed calibrated ZX90 schedule. When omitted, resolve it from
        the pulse service.
    idle_t1, idle_t2_echo : Mapping[str, float] | None, optional
        Optional mappings of qubit labels to fixed idle coherence times in ns.
        When omitted, load the saved `t1` and `t2_echo` parameters.
    gef_calibration_n_shots : int, optional
        Shots per GEF calibration configuration. Must be at least two for the
        IQ moment covariance and defaults to 8192.
    gef_bootstrap_n_resamples : int, optional
        Maximum number of raw-shot bootstrap resamples for A/B primary GEF
        boundary diagnostics. Bootstrap runs only for series whose simplex
        boundary ambiguity is unresolved by unconstrained populations. C/D
        primary and all reference data use analytic covariance only. Zero
        disables bootstrap uncertainty everywhere.
    gef_bootstrap_seed : int | None, optional
        Nonnegative bootstrap seed, or `None` for nondeterministic sampling.
    gef_bootstrap_confidence_level : float, optional
        Marginal bootstrap confidence level strictly between zero and one.
    gef_covariance_rcond : float, optional
        Relative cutoff in `[0, 1)` for GEF and downstream covariance
        pseudo-inverses.
    reference_ix45_amplitude : float | None, optional
        Optional pre-calibrated positive CR-envelope IX45 amplitude no greater
        than one. Used only when `measure_reference=True`.
    force_reference_ix45_calibration : bool, optional
        Whether to ignore a compatible cached IX45 calibration and recalibrate.
        Cannot be combined with an explicit `reference_ix45_amplitude`.
    enable_tqdm : bool, optional
        Whether acquisition services display progress bars.
    plot : bool, optional
        Whether to build and display protocol figures and print the user-facing
        summary.

    Returns
    -------
    Result
        Measurements, physical fits, rates, fidelity limits, raw GEF/Pauli IQ,
        resolved timing, and reproducibility metadata.

    Notes
    -----
    This function performs hardware acquisition. When reference calibration is
    requested, it may also update the experiment calibration-note store.
    """
    for name, value in (
        ("measure_reference", measure_reference),
        ("measure_orthogonal_components", measure_orthogonal_components),
        ("force_reference_ix45_calibration", force_reference_ix45_calibration),
        ("enable_tqdm", enable_tqdm),
        ("plot", plot),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")
    if reference_ix45_amplitude is not None:
        if isinstance(reference_ix45_amplitude, bool) or not isinstance(
            reference_ix45_amplitude, Real
        ):
            raise TypeError("reference_ix45_amplitude must be a real number.")
        reference_ix45_amplitude = float(reference_ix45_amplitude)
        if (
            not np.isfinite(reference_ix45_amplitude)
            or not 0.0 < reference_ix45_amplitude <= 1.0
        ):
            raise ValueError("reference_ix45_amplitude must be in (0, 1].")
    if reference_ix45_amplitude is not None and force_reference_ix45_calibration:
        raise ValueError(
            "reference_ix45_amplitude and force_reference_ix45_calibration=True "
            "cannot be specified together."
        )
    counts = _validate_repetition_counts(repetition_counts)
    shots = _positive_integer(n_shots, "n_shots")
    gef_shots = _positive_integer(
        gef_measurement_n_shots,
        "gef_measurement_n_shots",
    )
    calibration_shots = _positive_integer(
        gef_calibration_n_shots,
        "gef_calibration_n_shots",
    )
    bootstrap_resamples = _nonnegative_integer(
        gef_bootstrap_n_resamples,
        "gef_bootstrap_n_resamples",
    )
    bootstrap_seed = _optional_seed(gef_bootstrap_seed)
    if isinstance(gef_bootstrap_confidence_level, bool) or not isinstance(
        gef_bootstrap_confidence_level, Real
    ):
        raise TypeError("gef_bootstrap_confidence_level must be a real number.")
    confidence = float(gef_bootstrap_confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("gef_bootstrap_confidence_level must be in (0, 1).")
    if isinstance(gef_covariance_rcond, bool) or not isinstance(
        gef_covariance_rcond, Real
    ):
        raise TypeError("gef_covariance_rcond must be a real number.")
    covariance_rcond = float(gef_covariance_rcond)
    if not np.isfinite(covariance_rcond) or not 0.0 <= covariance_rcond < 1.0:
        raise ValueError("gef_covariance_rcond must be in [0, 1).")
    configured_interval = resolve_measurement_defaults(
        exp.ctx.experiment_system.measurement_defaults
    ).execution.shot_interval_ns
    interval = (
        _positive_finite(configured_interval, "configured shot_interval")
        if shot_interval is None
        else _positive_finite(shot_interval, "shot_interval")
    )
    control = exp.ctx.resolve_qubit_label(control_qubit)
    target = exp.ctx.resolve_qubit_label(target_qubit)
    if control == target:
        raise ValueError("control_qubit and target_qubit must differ.")

    # Resolve every required input before the first hardware acquisition.
    control_idle, target_idle = _load_idle_noise(
        exp,
        control,
        target,
        idle_t1,
        idle_t2_echo,
    )
    descriptor = resolve_zx90_descriptor(exp, control, target, zx90_echo)
    schedules = build_protocol_schedules(exp, control, target, descriptor, counts)

    reference_calibration: _ReferenceIx45Calibration | None = None
    reference_calibrated = False
    reference_schedules: dict[str, tuple[PulseSchedule, ...]] | None = None
    if measure_reference:
        ix45, reference_calibration, reference_calibrated = _resolve_reference_ix45(
            exp,
            control,
            target,
            descriptor,
            amplitude=reference_ix45_amplitude,
            force=force_reference_ix45_calibration,
            n_shots=shots,
            shot_interval=interval,
            plot=plot,
            enable_tqdm=enable_tqdm,
        )
        reference_schedules = build_reference_schedules(
            exp,
            control,
            target,
            descriptor,
            counts,
            ix45,
        )

    actual_results: dict[str, Result] = {}
    actual_names: dict[str, tuple[str, ...]] = {}
    ab_sequences, ab_names = _named_sequences(
        schedules.actual,
        "actual",
        (PROTOCOL_A, PROTOCOL_B),
    )
    ab_result = measure_gef_populations(
        exp,
        targets=[control, target],
        sequences=ab_sequences,
        n_shots=gef_shots,
        calibration_n_shots=calibration_shots,
        shot_interval=interval,
        covariance_rcond=covariance_rcond,
        n_bootstrap=0,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence_level=confidence,
        enable_tqdm=enable_tqdm,
        progress_label="A/B primary",
    )
    calibration = ab_result.data["calibration"]
    for protocol in (PROTOCOL_A, PROTOCOL_B):
        actual_results[protocol] = ab_result
        actual_names[protocol] = ab_names[protocol]
    selected_ab_bootstrap = _run_selective_ab_bootstrap(
        ab_result,
        ab_names,
        control,
        target,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=confidence,
        covariance_rcond=covariance_rcond,
    )

    def selected_bootstrap_resamples(protocol: str, qubit: str) -> int:
        """Return the actual bootstrap count for one A/B protocol and qubit."""
        selected_names = set(selected_ab_bootstrap.get(qubit, ()))
        return (
            bootstrap_resamples
            if any(name in selected_names for name in ab_names[protocol])
            else 0
        )

    for protocol, measured_target in (
        (PROTOCOL_C, control),
        (PROTOCOL_D, target),
    ):
        sequences, names = _named_sequences(
            schedules.actual,
            "actual",
            (protocol,),
        )
        result = measure_gef_populations(
            exp,
            targets=[measured_target],
            sequences=sequences,
            calibration={measured_target: calibration[measured_target]},  # type: ignore[index]
            n_shots=gef_shots,
            shot_interval=interval,
            covariance_rcond=covariance_rcond,
            n_bootstrap=0,
            bootstrap_seed=bootstrap_seed,
            bootstrap_confidence_level=confidence,
            enable_tqdm=enable_tqdm,
            progress_label=f"{protocol} primary ({measured_target})",
        )
        actual_results[protocol] = result
        actual_names[protocol] = names[protocol]

    reference_results: dict[str, Result] | None = None
    reference_names: dict[str, tuple[str, ...]] | None = None
    if reference_schedules is not None:
        reference_results = {}
        reference_names = {}
        ab_reference, ab_reference_names = _named_sequences(
            reference_schedules,
            "reference",
            (PROTOCOL_A, PROTOCOL_B),
        )
        ab_reference_result = measure_gef_populations(
            exp,
            targets=[control, target],
            sequences=ab_reference,
            calibration=calibration,  # type: ignore[arg-type]
            n_shots=gef_shots,
            shot_interval=interval,
            covariance_rcond=covariance_rcond,
            n_bootstrap=0,
            bootstrap_seed=bootstrap_seed,
            bootstrap_confidence_level=confidence,
            enable_tqdm=enable_tqdm,
            progress_label="A/B diagnostic reference",
        )
        for protocol in (PROTOCOL_A, PROTOCOL_B):
            reference_results[protocol] = ab_reference_result
            reference_names[protocol] = ab_reference_names[protocol]
        for protocol, measured_target in (
            (PROTOCOL_C, control),
            (PROTOCOL_D, target),
        ):
            sequences, names = _named_sequences(
                reference_schedules,
                "reference",
                (protocol,),
            )
            result = measure_gef_populations(
                exp,
                targets=[measured_target],
                sequences=sequences,
                calibration={measured_target: calibration[measured_target]},  # type: ignore[index]
                n_shots=gef_shots,
                shot_interval=interval,
                covariance_rcond=covariance_rcond,
                n_bootstrap=0,
                bootstrap_seed=bootstrap_seed,
                bootstrap_confidence_level=confidence,
                enable_tqdm=enable_tqdm,
                progress_label=f"{protocol} diagnostic reference ({measured_target})",
            )
            reference_results[protocol] = result
            reference_names[protocol] = names[protocol]

    orthogonal: dict[str, CrDissipationPauliData] | None = None
    pauli_raw_iq: dict[str, Any] | None = None
    if measure_orthogonal_components:
        orthogonal, pauli_raw_iq = _orthogonal_acquisition(
            exp,
            schedules,
            control,
            target,
            n_shots=shots,
            shot_interval=interval,
            enable_tqdm=enable_tqdm,
        )
    measurements = _build_measurements(
        schedules,
        counts,
        actual_results,
        actual_names,
        control,
        target,
        reference_results=reference_results,
        reference_names=reference_names,
        orthogonal=orthogonal,
    )
    analysis = analyze_cr_dissipation(
        measurements,
        descriptor,
        schedules,
        control_idle,
        target_idle,
        covariance_rcond=covariance_rcond,
        initial_warnings=descriptor.warnings,
    )
    figures = _plot_cr_dissipation(measurements, analysis) if plot else {}
    if plot:
        _print_summary(analysis, control_idle, target_idle)
        for figure in figures.values():
            figure.show()

    pulse_timing = {
        "zx90_echo": {
            "duration_ns": descriptor.total_duration_ns,
            "cr_active_duration_ns": descriptor.cr_active_duration_ns,
            "cr_lobe_duration_ns": descriptor.cr_lobe_duration_ns,
            "echo_slot_duration_ns": descriptor.echo_slot_duration_ns,
        },
        "zx90_full_un_echoed": {
            "duration_ns": float(descriptor.full_un_echoed.duration),
            "cr_active_duration_ns": descriptor.cr_active_duration_ns,
        },
        "protocols": schedules.timing,
        "reference_ix45": (
            None
            if reference_calibration is None
            else {
                "amplitude": reference_calibration.amplitude,
                "duration_ns": reference_calibration.duration,
                "ramptime_ns": reference_calibration.ramptime,
                "beta": reference_calibration.beta,
                "ramp_type": reference_calibration.ramp_type,
                "sampling_period_ns": reference_calibration.sampling_period,
                "r_squared": reference_calibration.r_squared,
                "timestamp": reference_calibration.timestamp,
                "calibrated": reference_calibrated,
            }
        ),
        "semantic_rotary_available": descriptor.rotary_integrated_angle_rad is not None,
    }
    measurement_options = {
        "repetition_counts": counts,
        "n_shots": shots,
        "gef_measurement_n_shots": gef_shots,
        "shot_interval_ns": interval,
        "measure_reference": measure_reference,
        "measure_orthogonal_components": measure_orthogonal_components,
        "gef_calibration_n_shots": calibration_shots,
        "gef_bootstrap_n_resamples": bootstrap_resamples,
        "gef_bootstrap_seed": bootstrap_seed,
        "gef_bootstrap_confidence_level": confidence,
        "ab_bootstrap_strategy": "simplex_boundary_selective",
        "ab_bootstrap_selected_sequence_names": selected_ab_bootstrap,
        "gef_covariance_rcond": covariance_rcond,
        "primary_readout": {
            PROTOCOL_A: "gef_control_and_target",
            PROTOCOL_B: "gef_control_and_target",
            PROTOCOL_C: "gef_control_only",
            PROTOCOL_D: "gef_target_only",
        },
        "primary_bootstrap": {
            PROTOCOL_A: {
                control: selected_bootstrap_resamples(PROTOCOL_A, control),
                target: selected_bootstrap_resamples(PROTOCOL_A, target),
            },
            PROTOCOL_B: {
                control: selected_bootstrap_resamples(PROTOCOL_B, control),
                target: selected_bootstrap_resamples(PROTOCOL_B, target),
            },
            PROTOCOL_C: {control: 0},
            PROTOCOL_D: {target: 0},
        },
        "orthogonal_readout": "pauli",
        "reference_readout": "gef",
        "reference_ix45_amplitude": reference_ix45_amplitude,
        "force_reference_ix45_calibration": force_reference_ix45_calibration,
        "enable_tqdm": enable_tqdm,
    }
    analysis_options = {
        "population_minimum_change": POPULATION_MINIMUM_CHANGE,
        "leakage_minimum_change": LEAKAGE_MINIMUM_CHANGE,
        "change_sigma_threshold": CHANGE_SIGMA_THRESHOLD,
        "model_selection_delta_aicc": MODEL_SELECTION_DELTA_AICC,
        "parameter_significance_threshold": PARAMETER_SIGNIFICANCE_THRESHOLD,
        "boundary_fraction_change_threshold": BOUNDARY_FRACTION_CHANGE_THRESHOLD,
        "preliminary_optimizer_loss": "soft_l1",
        "final_optimizer_loss": "linear",
        "model_selection_statistic": "gaussian_gls_aicc",
        "rate_uncertainty_method": "local_gls_covariance",
        "c_d_fit_method": "two_qutrit_physical_forward_fit",
        "fidelity_uncertainty_method": "scaled_unscented_transform",
        "fidelity_uncertainty_is_conditional_on_idle_coherence_and_gef_calibration": True,
        "cross_protocol_covariance_approximation": "block_diagonal",
        "sigma_point_covariance_psd_relative_tolerance": PSD_RELATIVE_TOLERANCE,
        "cr_sign_dependent_dissipation": False,
        "semantic_hamiltonian_model": "intended_rotation_with_actual_timing",
    }
    actual_raw_iq = _merge_result_branch(actual_results, "raw_iq")
    population_fits = _merge_result_branch(actual_results, "fits")
    moment_summaries = _merge_result_branch(actual_results, "moment_summaries")
    bootstrap = _merge_result_branch(actual_results, "bootstrap")
    reference_raw_iq = (
        None
        if reference_results is None
        else _merge_result_branch(reference_results, "raw_iq")
    )
    return Result(
        data={
            "measurements": measurements,
            "analysis": analysis,
            "raw_data": {
                "gef": {
                    "calibration": calibration,
                    "raw_iq": actual_raw_iq,
                    "population_fits": population_fits,
                    "moment_summaries": moment_summaries,
                    "bootstrap": bootstrap,
                    "reference_raw_iq": reference_raw_iq,
                },
                "pauli_raw_iq": pauli_raw_iq,
            },
            "measurement_options": measurement_options,
            "analysis_options": analysis_options,
            "pulse_timing": pulse_timing,
        },
        figure=figures.get(PROTOCOL_A),
        figures=figures,
    )
