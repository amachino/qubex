"""
Characterize decoherence and leakage during calibrated CR pulses.

The only public API in this module is :func:`characterize_cr_dissipation`.
Acquisition uses the calibrated waveforms unchanged; the physical forward and
fidelity models use intended rotations with the actual segment timing.
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

from qubex.analysis import fitting
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.pulse import FlatTop, PulseSchedule, Waveform

from ._cr_dissipation_analysis import (
    BOUNDARY_FRACTION_CHANGE_THRESHOLD,
    CHANGE_SIGMA_THRESHOLD,
    LEAKAGE_MINIMUM_CHANGE,
    MODEL_SELECTION_DELTA_AICC,
    PARAMETER_SIGNIFICANCE_THRESHOLD,
    POPULATION_MINIMUM_CHANGE,
    PSD_RELATIVE_TOLERANCE,
    analyze_cr_dissipation as _run_analysis,
    computational_polarization,
    nonnormalized_ge_expectation,
)
from ._cr_dissipation_pulses import (
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
from ._cr_dissipation_types import (
    CrDissipationAnalysis,
    CrDissipationMeasurements,
    CrDissipationProtocolData,
    CrDissipationProtocolMeasurements,
    CrDissipationRateStatus,
    CrDissipationWarning,
    GefPopulationSeries,
    IdleNoiseParameters,
)
from .gef_population_estimation import measure_gef_populations

__all__ = ["characterize_cr_dissipation"]

DEFAULT_REPETITION_COUNTS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55)
DEFAULT_N_SHOTS = 4096
DEFAULT_GEF_CALIBRATION_N_SHOTS = 8192
_REFERENCE_CALIBRATION_NOTE_KEY = "cr_dissipation_reference_ix45"


@dataclass(frozen=True)
class _ReferenceIx45Calibration:
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
        return (
            self.duration,
            self.ramptime,
            self.beta,
            self.ramp_type,
            self.sampling_period,
        )


def _validate_repetition_counts(values: Sequence[int] | None) -> tuple[int, ...]:
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


def _positive_integer(value: int, name: str, *, minimum: int = 2) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        if minimum == 2:
            raise ValueError(f"{name} must be at least two.")
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _optional_seed(value: int | None) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, "gef_bootstrap_seed")


def _positive_finite(value: float, name: str) -> float:
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
    try:
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
        return (
            IdleNoiseParameters(t1[control], t2[control]),
            IdleNoiseParameters(t1[target], t2[target]),
        )
    except KeyError as exc:
        raise ValueError(
            f"Idle coherence data missing for qubit {exc.args[0]}."
        ) from exc
    except (
        AttributeError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if idle_t1 is None or idle_t2_echo is None:
            raise ValueError(
                "Stored idle T1/T2_echo values could not be resolved before acquisition."
            ) from exc
        raise


def _flat_top_unit_area(pulse: FlatTop) -> float:
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
    if not np.isfinite(amplitude) or not 0.0 < amplitude <= 1.0 or r_squared < 0.5:
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


def _gef_series(
    result: Result, names: Sequence[str], target: str
) -> GefPopulationSeries:
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
    control_series = _gef_series(result, names, control)
    target_series = _gef_series(result, names, target)
    if protocol in (PROTOCOL_A, PROTOCOL_B):
        target_x, target_x_error = computational_polarization(target_series)
        sign = 1.0 if target_x[0] >= 0.0 else -1.0
        return CrDissipationProtocolData(
            control_gef=control_series,
            target_gef=target_series,
            target_x_comp=sign * target_x,
            target_x_comp_standard_error=target_x_error,
        )
    primary_series = control_series if protocol == PROTOCOL_C else target_series
    primary, primary_error, _ = nonnormalized_ge_expectation(primary_series)
    return CrDissipationProtocolData(
        control_gef=control_series,
        target_gef=target_series,
        primary_expectation=primary,
        primary_standard_error=primary_error,
    )


def _named_sequences(
    schedules: Mapping[str, Sequence[PulseSchedule]],
    prefix: str,
) -> tuple[dict[str, PulseSchedule], dict[str, tuple[str, ...]]]:
    named: dict[str, PulseSchedule] = {}
    grouped: dict[str, tuple[str, ...]] = {}
    for protocol in PROTOCOLS:
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
    calibration: Mapping[str, Any],
    *,
    n_shots: int,
    shot_interval: float,
    covariance_rcond: float,
    enable_tqdm: bool,
) -> tuple[dict[str, CrDissipationProtocolData], dict[str, Any]]:
    requests = {
        PROTOCOL_A: ((control, "X"), (control, "Y"), (target, "Y"), (target, "Z")),
        PROTOCOL_B: ((control, "X"), (control, "Y"), (target, "Y"), (target, "Z")),
        PROTOCOL_C: ((control, "Y"), (control, "Z")),
        PROTOCOL_D: ((target, "X"), (target, "Z")),
    }
    named: dict[str, PulseSchedule] = {}
    for protocol in PROTOCOLS:
        for qubit, basis in requests[protocol]:
            for index, base in enumerate(schedules.base[protocol]):
                sequence = base
                if basis == "X":
                    sequence = append_analyzer(base, qubit, exp.pulse.y90m(qubit))
                elif basis == "Y":
                    sequence = append_analyzer(base, qubit, exp.pulse.x90(qubit))
                named[f"orthogonal:{protocol}:{qubit}:{basis}:{index}"] = sequence
    result = measure_gef_populations(
        exp,
        targets=[control, target],
        sequences=named,
        calibration=calibration,
        n_shots=n_shots,
        shot_interval=shot_interval,
        covariance_rcond=covariance_rcond,
        n_bootstrap=0,
        enable_tqdm=enable_tqdm,
    )
    processed: dict[str, CrDissipationProtocolData] = {}
    for protocol in PROTOCOLS:
        components: dict[str, NDArray[np.float64]] = {}
        errors: dict[str, NDArray[np.float64]] = {}
        for qubit, basis in requests[protocol]:
            names = tuple(
                f"orthogonal:{protocol}:{qubit}:{basis}:{index}"
                for index in range(len(schedules.base[protocol]))
            )
            series = _gef_series(result, names, qubit)
            values, standard_errors = computational_polarization(series)
            sign = 1.0 if values[0] >= 0.0 else -1.0
            key = f"{qubit}:{basis}"
            components[key] = sign * values
            errors[key] = standard_errors
        processed[protocol] = CrDissipationProtocolData(
            components=components,
            component_standard_errors=errors,
        )
    return processed, result.data["raw_iq"]  # type: ignore[return-value]


def _build_measurements(
    schedules: ProtocolSchedules,
    repetition_counts: tuple[int, ...],
    actual_result: Result,
    actual_names: Mapping[str, Sequence[str]],
    control: str,
    target: str,
    *,
    reference_result: Result | None,
    reference_names: Mapping[str, Sequence[str]] | None,
    orthogonal: Mapping[str, CrDissipationProtocolData] | None,
) -> CrDissipationMeasurements:
    protocol_measurements: dict[str, CrDissipationProtocolMeasurements] = {}
    for protocol in PROTOCOLS:
        actual = _protocol_data(
            actual_result, actual_names[protocol], control, target, protocol
        )
        reference = (
            None
            if reference_result is None or reference_names is None
            else _protocol_data(
                reference_result,
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


def _plot_cr_dissipation(
    measurements: CrDissipationMeasurements,
    analysis: CrDissipationAnalysis,
) -> dict[str, go.Figure]:
    """Build all main, idle-prediction, reference, and diagnostic panels."""
    figures: dict[str, go.Figure] = {}
    for protocol in PROTOCOLS:
        measured = getattr(measurements, protocol)
        time_us = measured.elapsed_time_ns / 1000.0
        if protocol in (PROTOCOL_A, PROTOCOL_B):
            has_orthogonal = measured.orthogonal is not None
            titles = ["Control GEF population", "Target Xcomp", "Target Pf"]
            if has_orthogonal:
                titles.append("Orthogonal diagnostics")
            figure = make_subplots(
                rows=len(titles),
                cols=1,
                shared_xaxes=True,
                subplot_titles=titles,
            )
            control = measured.actual.control_gef
            if control is not None:
                for index, state in enumerate(("Pg", "Pe", "Pf")):
                    figure.add_trace(
                        go.Scatter(
                            x=time_us,
                            y=control.population[:, index],
                            error_y={
                                "array": control.standard_error[:, index],
                                "visible": True,
                            },
                            mode="markers",
                            name=f"actual {state}",
                        ),
                        row=1,
                        col=1,
                    )
                fitted = analysis.fits.control_population_ab.fitted_populations.get(
                    protocol
                )
                if fitted is not None:
                    for index, state in enumerate(("Pg", "Pe", "Pf")):
                        figure.add_trace(
                            go.Scatter(
                                x=time_us,
                                y=fitted[:, index],
                                mode="lines",
                                name=f"fit {state}",
                            ),
                            row=1,
                            col=1,
                        )
            if measured.actual.target_x_comp is not None:
                figure.add_trace(
                    go.Scatter(
                        x=time_us,
                        y=measured.actual.target_x_comp,
                        error_y={
                            "array": measured.actual.target_x_comp_standard_error,
                            "visible": True,
                        },
                        mode="markers",
                        name="actual Xcomp",
                    ),
                    row=2,
                    col=1,
                )
            target = measured.actual.target_gef
            if target is not None:
                figure.add_trace(
                    go.Scatter(
                        x=time_us,
                        y=target.population[:, 2],
                        error_y={
                            "array": target.standard_error[:, 2],
                            "visible": True,
                        },
                        mode="markers",
                        name="actual target Pf",
                    ),
                    row=3,
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
            figure.add_trace(
                go.Scatter(
                    x=time_us,
                    y=t1_fit.fitted_values,
                    mode="lines",
                    name="T1rho fit",
                ),
                row=2,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=time_us,
                    y=leakage_fit.fitted_values,
                    mode="lines",
                    name="leak/seep fit",
                ),
                row=3,
                col=1,
            )
            idle = analysis.idle_predictions[protocol].observables
            idle_control = idle.get("control_population")
            if idle_control is not None:
                for index, state in enumerate(("Pg", "Pe", "Pf")):
                    figure.add_trace(
                        go.Scatter(
                            x=time_us,
                            y=idle_control[:, index],
                            mode="lines",
                            line={"dash": "dash"},
                            name=f"idle-only {state}",
                        ),
                        row=1,
                        col=1,
                    )
            for row, key in ((2, "target_x_comp"), (3, "target_pf")):
                figure.add_trace(
                    go.Scatter(
                        x=time_us,
                        y=idle[key],
                        mode="lines",
                        line={"dash": "dash"},
                        name=f"idle-only {key}",
                    ),
                    row=row,
                    col=1,
                )
            if measured.reference is not None:
                reference_control = measured.reference.control_gef
                if reference_control is not None:
                    for index, state in enumerate(("Pg", "Pe", "Pf")):
                        figure.add_trace(
                            go.Scatter(
                                x=time_us,
                                y=reference_control.population[:, index],
                                mode="markers",
                                marker={"symbol": "circle-open"},
                                name=f"reference {state}",
                            ),
                            row=1,
                            col=1,
                        )
                if measured.reference.target_x_comp is not None:
                    figure.add_trace(
                        go.Scatter(
                            x=time_us,
                            y=measured.reference.target_x_comp,
                            mode="markers",
                            marker={"symbol": "circle-open"},
                            name="reference Xcomp",
                        ),
                        row=2,
                        col=1,
                    )
                reference_target = measured.reference.target_gef
                if reference_target is not None:
                    figure.add_trace(
                        go.Scatter(
                            x=time_us,
                            y=reference_target.population[:, 2],
                            mode="markers",
                            marker={"symbol": "circle-open"},
                            name="reference target Pf",
                        ),
                        row=3,
                        col=1,
                    )
            if measured.orthogonal is not None:
                for name, component in measured.orthogonal.components.items():
                    figure.add_trace(
                        go.Scatter(
                            x=time_us,
                            y=component,
                            mode="markers",
                            name=name,
                        ),
                        row=4,
                        col=1,
                    )
        else:
            has_orthogonal = measured.orthogonal is not None
            titles = ["Control Xge" if protocol == PROTOCOL_C else "Target Yge"]
            if has_orthogonal:
                titles.append("Orthogonal diagnostics")
            figure = make_subplots(
                rows=len(titles),
                cols=1,
                shared_xaxes=True,
                subplot_titles=titles,
            )
            values = measured.actual.primary_expectation
            errors = measured.actual.primary_standard_error
            if values is not None:
                figure.add_trace(
                    go.Scatter(
                        x=time_us,
                        y=values,
                        error_y={"array": errors, "visible": True},
                        mode="markers",
                        name="actual",
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
                    go.Scatter(
                        x=time_us,
                        y=fit.fitted_values,
                        mode="lines",
                        name="physical fit",
                    ),
                    row=1,
                    col=1,
                )
            idle_values = analysis.idle_predictions[protocol].observables.get(
                "primary_expectation"
            )
            if idle_values is not None:
                figure.add_trace(
                    go.Scatter(
                        x=time_us,
                        y=idle_values,
                        mode="lines",
                        line={"dash": "dash"},
                        name="idle-only prediction",
                    ),
                    row=1,
                    col=1,
                )
            if measured.reference is not None:
                figure.add_trace(
                    go.Scatter(
                        x=time_us,
                        y=measured.reference.primary_expectation,
                        mode="markers",
                        marker={"symbol": "circle-open"},
                        name="reference (diagnostic)",
                    ),
                    row=1,
                    col=1,
                )
            if measured.orthogonal is not None:
                for name, component in measured.orthogonal.components.items():
                    figure.add_trace(
                        go.Scatter(
                            x=time_us,
                            y=component,
                            mode="markers",
                            name=name,
                        ),
                        row=2,
                        col=1,
                    )
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
            title=protocol,
            height=280 * len(titles),
        )
        figure.update_xaxes(
            title_text="Elapsed sequence time [us]",
            row=len(titles),
            col=1,
        )
        figure.update_xaxes(
            title_text="CR gates (4n)",
            tickvals=time_us,
            ticktext=[str(value) for value in measurements.cr_gate_counts],
            side="top",
            showticklabels=True,
            row=1,
            col=1,
        )
        figures[protocol] = figure
    return figures


def _print_summary(
    analysis: CrDissipationAnalysis,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
) -> None:
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


def characterize_cr_dissipation(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    repetition_counts: Sequence[int] | None = None,
    n_shots: int = 4096,
    shot_interval: float | None = None,
    measure_reference: bool = False,
    measure_orthogonal_components: bool = False,
    zx90_echo: PulseSchedule | None = None,
    idle_t1: Mapping[str, float] | None = None,
    idle_t2_echo: Mapping[str, float] | None = None,
    gef_calibration_n_shots: int = 8192,
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
    use the non-normalized qutrit observable ``Pg-Pe``. Reference and orthogonal
    acquisitions are diagnostic-only and never enter primary inference.

    Parameters
    ----------
    exp
        Experiment used for pulse construction, acquisition, and calibration
        storage.
    control_qubit, target_qubit
        Distinct qubit labels defining the calibrated CR pair.
    repetition_counts
        Strictly increasing nonnegative repetition counts starting at zero.
        At least five points are required. Defaults to a Fibonacci-like grid.
    n_shots
        Shots per measurement configuration. Must be at least two because the
        GEF estimator requires an IQ moment covariance.
    shot_interval
        Interval between shots in ns. When omitted, use the configured
        measurement default.
    measure_reference
        Whether to acquire duration-matched diagnostic reference sequences.
        Reference data never enter primary fits or fidelity estimates.
    measure_orthogonal_components
        Whether to acquire diagnostic components orthogonal to each primary
        observable. These data never enter primary inference.
    zx90_echo
        Optional echoed calibrated ZX90 schedule. When omitted, resolve it from
        the pulse service.
    idle_t1, idle_t2_echo
        Optional mappings of qubit labels to fixed idle coherence times in ns.
        When omitted, load the saved ``t1`` and ``t2_echo`` parameters.
    gef_calibration_n_shots
        Shots per GEF calibration configuration. Must be at least two for the
        same covariance requirement.
    gef_bootstrap_n_resamples
        Nonnegative number of raw-shot GEF bootstrap resamples. Zero disables
        bootstrap uncertainty.
    gef_bootstrap_seed
        Nonnegative bootstrap seed, or ``None`` for nondeterministic sampling.
    gef_bootstrap_confidence_level
        Marginal bootstrap confidence level strictly between zero and one.
    gef_covariance_rcond
        Relative cutoff in ``[0, 1)`` for GEF and downstream covariance
        pseudo-inverses.
    reference_ix45_amplitude
        Optional pre-calibrated positive CR-envelope IX45 amplitude no greater
        than one. Used only when ``measure_reference=True``.
    force_reference_ix45_calibration
        Whether to ignore a compatible cached IX45 calibration and recalibrate.
        Cannot be combined with an explicit ``reference_ix45_amplitude``.
    enable_tqdm
        Whether acquisition services display progress bars.
    plot
        Whether to build protocol figures and print the user-facing summary.

    Returns
    -------
    Result
        Measurements, physical fits, rates, fidelity limits, raw GEF IQ,
        resolved timing, and reproducibility metadata.
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

    actual_sequences, actual_names = _named_sequences(schedules.actual, "actual")
    actual_result = measure_gef_populations(
        exp,
        targets=[control, target],
        sequences=actual_sequences,
        n_shots=shots,
        calibration_n_shots=calibration_shots,
        shot_interval=interval,
        covariance_rcond=covariance_rcond,
        n_bootstrap=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence_level=confidence,
        enable_tqdm=enable_tqdm,
    )
    calibration = actual_result.data["calibration"]

    reference_result: Result | None = None
    reference_names: dict[str, tuple[str, ...]] | None = None
    if reference_schedules is not None:
        named_reference, reference_names = _named_sequences(
            reference_schedules, "reference"
        )
        reference_result = measure_gef_populations(
            exp,
            targets=[control, target],
            sequences=named_reference,
            calibration=calibration,  # type: ignore[arg-type]
            n_shots=shots,
            shot_interval=interval,
            covariance_rcond=covariance_rcond,
            n_bootstrap=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_confidence_level=confidence,
            enable_tqdm=enable_tqdm,
        )

    orthogonal: dict[str, CrDissipationProtocolData] | None = None
    pauli_raw_iq: dict[str, Any] | None = None
    if measure_orthogonal_components:
        orthogonal, pauli_raw_iq = _orthogonal_acquisition(
            exp,
            schedules,
            control,
            target,
            calibration,  # type: ignore[arg-type]
            n_shots=shots,
            shot_interval=interval,
            covariance_rcond=covariance_rcond,
            enable_tqdm=enable_tqdm,
        )
    measurements = _build_measurements(
        schedules,
        counts,
        actual_result,
        actual_names,
        control,
        target,
        reference_result=reference_result,
        reference_names=reference_names,
        orthogonal=orthogonal,
    )
    analysis = _analyze_cr_dissipation(
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
                "calibrated": reference_calibrated,
            }
        ),
        "semantic_rotary_available": descriptor.rotary_integrated_angle_rad is not None,
    }
    measurement_options = {
        "repetition_counts": counts,
        "n_shots": shots,
        "shot_interval_ns": interval,
        "measure_reference": measure_reference,
        "measure_orthogonal_components": measure_orthogonal_components,
        "gef_calibration_n_shots": calibration_shots,
        "gef_bootstrap_n_resamples": bootstrap_resamples,
        "gef_bootstrap_seed": bootstrap_seed,
        "gef_bootstrap_confidence_level": confidence,
        "gef_covariance_rcond": covariance_rcond,
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
    reference_raw_iq = (
        None if reference_result is None else reference_result.data["raw_iq"]
    )
    return Result(
        data={
            "measurements": measurements,
            "analysis": analysis,
            "raw_data": {
                "gef": {
                    "calibration": calibration,
                    "raw_iq": actual_result.data["raw_iq"],
                    "population_fits": actual_result.data["fits"],
                    "moment_summaries": actual_result.data["moment_summaries"],
                    "bootstrap": actual_result.data["bootstrap"],
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


def _analyze_cr_dissipation(
    measurements: CrDissipationMeasurements,
    descriptor: ZX90Descriptor,
    schedules: ProtocolSchedules,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    *,
    covariance_rcond: float,
    initial_warnings: Sequence[CrDissipationWarning] = (),
) -> CrDissipationAnalysis:
    """Private offline core with an intentionally unstable input contract."""
    return _run_analysis(
        measurements,
        descriptor,
        schedules,
        control_idle,
        target_idle,
        covariance_rcond=covariance_rcond,
        initial_warnings=initial_warnings,
    )
