"""Characterize relaxation, leakage, and coherence under repeated CR pulses."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from numbers import Integral, Real
from typing import Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, PulseSchedule, Waveform
from qubex.visualization import COLORS

from .cr_pulse_coherence_fitting import (
    CrOnDephasingFit,
    ExponentialDecayFit,
    TargetLeakageFit,
    TargetT1RhoFit,
    ThreeLevelRateFit,
    fit_cr_on_dephasing,
    fit_exponential_decay,
    fit_target_leakage,
    fit_target_t1rho,
    fit_three_level_rate_model,
    three_level_population_trajectory,
)
from .cr_pulse_fidelity_simulation import (
    CrOnNoise,
    CrPulseFidelitySimulationResult,
    IdleQubitNoise,
    extract_zx90_gate_timing,
    prepare_cr_echo_decay_model,
    simulate_cr_pulse_fidelity,
)
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
EchoFitMethod = Literal["auto", "forward", "exponential"]

DEFAULT_N_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55)
_DEFAULT_N_SHOTS = 4096
_DEFAULT_CALIBRATION_N_SHOTS = 8192
_CONTROL_GROUND = "control_ground"
_CONTROL_EXCITED = "control_excited"
_CONTROL_T2_ECHO = "control_t2_echo"
_TARGET_T2RHO_ECHO = "target_t2rho_echo"
_PHENOMENOLOGICAL_ECHO_FITS = "phenomenological_echo_decay"
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


@dataclass(frozen=True)
class _PauliMeasurement:
    """Store one single-basis Pauli expectation measurement."""

    expectation: float
    standard_error: float
    normalized_shots: NDArray[np.float64]
    raw_iq: NDArray[np.complex128]


@dataclass(frozen=True)
class CrPulseFidelityAnalysis:
    """Store the inferred CR-on dephasing and optional fidelity simulation."""

    success: bool
    message: str
    dephasing_fit: CrOnDephasingFit | None
    simulation: CrPulseFidelitySimulationResult | None


@dataclass(frozen=True)
class _ProtocolSequences:
    """Store selected schedules and timing metadata for one n value."""

    sequences: dict[str, PulseSchedule]
    evolution_durations: dict[str, float]
    cr_pulse_count: int


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
        covariance=np.full((3, 3), np.nan),
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
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return _failed_exponential_fit(str(exc), values.size)


def _safe_fit_target_t1rho(
    times: NDArray[np.float64],
    ground: NDArray[np.float64],
    excited: NDArray[np.float64],
    ground_errors: NDArray[np.float64] | None,
    excited_errors: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
) -> TargetT1RhoFit:
    """Fit target T1rho while preserving measurements on numerical failure."""
    try:
        return fit_target_t1rho(
            times,
            ground,
            excited,
            ground_errors,
            excited_errors,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return TargetT1RhoFit(
            success=False,
            message=str(exc),
            t1rho=float("nan"),
            t1rho_error=float("nan"),
            amplitude_ground=float("nan"),
            amplitude_excited=float("nan"),
            amplitude_ground_error=float("nan"),
            amplitude_excited_error=float("nan"),
            covariance=np.full((3, 3), np.nan),
            fitted_ground=np.full(times.shape, np.nan),
            fitted_excited=np.full(times.shape, np.nan),
            r_squared=float("nan"),
        )


def _safe_fit_target_leakage(
    times: NDArray[np.float64],
    ground: NDArray[np.float64],
    excited: NDArray[np.float64],
    ground_errors: NDArray[np.float64] | None,
    excited_errors: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
) -> TargetLeakageFit:
    """Fit target leakage while preserving measurements on numerical failure."""
    try:
        return fit_target_leakage(
            times,
            ground,
            excited,
            ground_errors,
            excited_errors,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return TargetLeakageFit(
            success=False,
            message=str(exc),
            model=None,
            leakage_rate=float("nan"),
            seepage_rate=float("nan"),
            leakage_rate_error=float("nan"),
            seepage_rate_error=float("nan"),
            covariance=np.full((2, 2), np.nan),
            initial_ground=float(ground[0]),
            initial_excited=float(excited[0]),
            fitted_ground=np.full(times.shape, np.nan),
            fitted_excited=np.full(times.shape, np.nan),
            r_squared=float("nan"),
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
        leakage_model=None,
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


def _safe_fit_three_level_rate_model(
    times: NDArray[np.float64],
    populations_ground: NDArray[np.float64],
    populations_excited: NDArray[np.float64],
    standard_errors_ground: NDArray[np.float64] | None,
    standard_errors_excited: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
) -> ThreeLevelRateFit:
    """Fit the rate model without discarding measurements on numerical failure."""
    try:
        return fit_three_level_rate_model(
            times,
            populations_ground,
            populations_excited,
            standard_errors_ground,
            standard_errors_excited,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        initial_ground = np.asarray(populations_ground[0], dtype=np.float64)
        initial_excited = np.asarray(populations_excited[0], dtype=np.float64)
        return _failed_rate_fit(
            str(exc),
            initial_ground,
            initial_excited,
            times.size,
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


def _build_un_echoed_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
) -> PulseSchedule:
    """
    Build a full ZX90 from two identical un-echoed CR primitives.

    Qubex calibrates one ``echo=False`` CR primitive as one lobe of the
    echoed ZX90 (nominally ZX45). Repeating that schedule twice preserves the
    calibrated ramps, cancellation tone, rotary tone, sign, and phase while
    producing the intended un-echoed ZX90 rotation.
    """
    cr_lobe = exp.pulse.zx90(control_qubit, target_qubit, echo=False)
    timing = extract_zx90_gate_timing(cr_lobe)
    if timing.echo:
        raise ValueError("exp.pulse.zx90(..., echo=False) returned an echoed CR gate.")

    zx90 = cr_lobe.repeated(2)
    # ``repeated`` intentionally returns a plain PulseSchedule. Attach
    # semantic metadata for the full, contiguous CR-active ZX90 so the
    # fidelity model interprets both physical lobes as one measured gate.
    zx90.cr_duration = 2 * timing.cr_lobe_duration  # type: ignore[attr-defined]
    zx90.echo = False  # type: ignore[attr-defined]
    return zx90


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


def _validate_echo_fit_method(method: str) -> EchoFitMethod:
    """Validate the requested primary fit model for actual C/D data."""
    valid_methods: tuple[EchoFitMethod, ...] = (
        "auto",
        "forward",
        "exponential",
    )
    if method not in valid_methods:
        valid_names = ", ".join(valid_methods)
        raise ValueError(f"echo_fit_method must be one of: {valid_names}.")
    return cast(EchoFitMethod, method)


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
    target_qubit: str,
    protocols: Sequence[_Protocol],
    relative_uncertainty_threshold: float,
) -> dict[str, object]:
    """Fit actual and reference data for every selected protocol."""
    fits: dict[str, object] = {}
    if _CONTROL_GROUND in protocols:
        fits["control_rate_model"] = {
            kind: _safe_fit_three_level_rate_model(
                times[_CONTROL_GROUND],
                populations[_CONTROL_GROUND][kind][control_qubit],
                populations[_CONTROL_EXCITED][kind][control_qubit],
                _finite_errors_or_none(
                    population_errors[_CONTROL_GROUND][kind][control_qubit]
                ),
                _finite_errors_or_none(
                    population_errors[_CONTROL_EXCITED][kind][control_qubit]
                ),
                relative_uncertainty_threshold,
            )
            for kind in ("actual", "reference")
        }
        fits["target_t1rho"] = {
            kind: _safe_fit_target_t1rho(
                times[_CONTROL_GROUND],
                target_polarizations[_CONTROL_GROUND][kind],
                target_polarizations[_CONTROL_EXCITED][kind],
                _finite_errors_or_none(
                    target_polarization_errors[_CONTROL_GROUND][kind]
                ),
                _finite_errors_or_none(
                    target_polarization_errors[_CONTROL_EXCITED][kind]
                ),
                relative_uncertainty_threshold,
            )
            for kind in ("actual", "reference")
        }
        fits["target_leakage"] = {
            kind: _safe_fit_target_leakage(
                times[_CONTROL_GROUND],
                populations[_CONTROL_GROUND][kind][target_qubit][:, 2],
                populations[_CONTROL_EXCITED][kind][target_qubit][:, 2],
                _finite_errors_or_none(
                    population_errors[_CONTROL_GROUND][kind][target_qubit][:, 2]
                ),
                _finite_errors_or_none(
                    population_errors[_CONTROL_EXCITED][kind][target_qubit][:, 2]
                ),
                relative_uncertainty_threshold,
            )
            for kind in ("actual", "reference")
        }

    echo_fits = {
        protocol: {
            kind: _safe_fit_exponential_decay(
                times[protocol],
                pauli_expectations[protocol][_PAULI_COMPONENT_ORDER[protocol][0]][kind],
                _finite_errors_or_none(
                    pauli_errors[protocol][_PAULI_COMPONENT_ORDER[protocol][0]][kind]
                ),
            )
            for kind in ("actual", "reference")
        }
        for protocol in _PAULI_PROTOCOLS
        if protocol in protocols
    }
    if echo_fits:
        fits[_PHENOMENOLOGICAL_ECHO_FITS] = echo_fits
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
    return three_level_population_trajectory(dense_times, initial, rates)


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


def _target_t1rho_curve(
    fit: TargetT1RhoFit,
    protocol: _GefProtocol,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one branch of the common target T1rho fit."""
    if not fit.success or not np.isfinite(fit.t1rho) or fit.t1rho <= 0:
        return np.full(times.shape, np.nan)
    amplitude = (
        fit.amplitude_ground if protocol == _CONTROL_GROUND else fit.amplitude_excited
    )
    return amplitude * np.exp(-times / fit.t1rho)


def _target_leakage_curve(
    fit: TargetLeakageFit,
    protocol: _GefProtocol,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one branch of the common target leakage fit."""
    if not fit.success:
        return np.full(times.shape, np.nan)
    initial = fit.initial_ground if protocol == _CONTROL_GROUND else fit.initial_excited
    rate_sum = fit.leakage_rate + fit.seepage_rate
    if rate_sum == 0:
        return np.full(times.shape, initial)
    equilibrium = fit.leakage_rate / rate_sum
    return equilibrium + (initial - equilibrium) * np.exp(-rate_sum * times)


def _make_target_figure(
    protocol: _GefProtocol,
    times: NDArray[np.float64],
    cr_counts: Sequence[int],
    populations: Mapping[str, Mapping[str, NDArray[np.float64]]],
    population_errors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    target_qubit: str,
    polarizations: Mapping[str, NDArray[np.float64]],
    polarization_errors: Mapping[str, NDArray[np.float64]],
    t1rho_fits: Mapping[str, TargetT1RhoFit],
    leakage_fits: Mapping[str, TargetLeakageFit],
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
                y=_target_t1rho_curve(t1rho_fits[kind], protocol, dense_times),
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
        figure.add_trace(
            go.Scatter(
                x=dense_times * 1e-3,
                y=_target_leakage_curve(
                    leakage_fits[kind],
                    protocol,
                    dense_times,
                ),
                mode="lines",
                line={"color": COLORS[2], "dash": line_dash},
                opacity=opacity,
                name=f"{kind} Pf fit",
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
    forward_fit: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
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
                if kind == "actual" and forward_fit is not None:
                    curve_times, curve_values = forward_fit
                else:
                    curve_times = dense_times
                    curve_values = _exponential_curve(fits[kind], dense_times)
                figure.add_trace(
                    go.Scatter(
                        x=curve_times * 1e-3,
                        y=curve_values,
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
            Mapping[str, TargetT1RhoFit],
            fits["target_t1rho"],
        )
        leakage_fits = cast(
            Mapping[str, TargetLeakageFit],
            fits["target_leakage"],
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
                t1rho_fits,
                leakage_fits,
            )
    if _CONTROL_T2_ECHO in protocols:
        echo_fits = cast(
            Mapping[str, Mapping[str, ExponentialDecayFit]],
            fits[_PHENOMENOLOGICAL_ECHO_FITS],
        )
        control_t2_fits = cast(
            Mapping[str, ExponentialDecayFit],
            echo_fits[_CONTROL_T2_ECHO],
        )
        dephasing_fit = cast(
            CrOnDephasingFit | None,
            fits.get("cr_on_dephasing"),
        )
        control_forward_fit = None
        if dephasing_fit is not None and dephasing_fit.success:
            control_forward_fit = (
                dephasing_fit.curve_n_values
                * times[_CONTROL_T2_ECHO][-1]
                / max(dephasing_fit.curve_n_values[-1], 1),
                dephasing_fit.curve_control_x,
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
            control_forward_fit,
        )
    if _TARGET_T2RHO_ECHO in protocols:
        echo_fits = cast(
            Mapping[str, Mapping[str, ExponentialDecayFit]],
            fits[_PHENOMENOLOGICAL_ECHO_FITS],
        )
        target_t2rho_fits = cast(
            Mapping[str, ExponentialDecayFit],
            echo_fits[_TARGET_T2RHO_ECHO],
        )
        dephasing_fit = cast(
            CrOnDephasingFit | None,
            fits.get("cr_on_dephasing"),
        )
        target_forward_fit = None
        if dephasing_fit is not None and dephasing_fit.success:
            target_forward_fit = (
                dephasing_fit.curve_n_values
                * times[_TARGET_T2RHO_ECHO][-1]
                / max(dephasing_fit.curve_n_values[-1], 1),
                dephasing_fit.curve_target_z,
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
            target_forward_fit,
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
            Mapping[str, TargetT1RhoFit],
            fits["target_t1rho"],
        )
        leakage_fits = cast(
            Mapping[str, TargetLeakageFit],
            fits["target_leakage"],
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
            kind: _value_with_error(fit.t1rho, fit.t1rho_error)
            for kind, fit in t1rho_fits.items()
        }
        transition_rates["target_leakage"] = {
            kind: {
                "leakage_rate": _value_with_error(
                    fit.leakage_rate,
                    fit.leakage_rate_error,
                ),
                "seepage_rate": _value_with_error(
                    fit.seepage_rate,
                    fit.seepage_rate_error,
                ),
                "model": fit.model,
            }
            for kind, fit in leakage_fits.items()
        }

    echo_fits = cast(
        Mapping[str, Mapping[str, ExponentialDecayFit]],
        fits.get(_PHENOMENOLOGICAL_ECHO_FITS, {}),
    )
    if _CONTROL_T2_ECHO in echo_fits:
        control_t2_fits = cast(
            Mapping[str, ExponentialDecayFit],
            echo_fits[_CONTROL_T2_ECHO],
        )
        decay_times["T2_echo"] = {
            kind: _value_with_error(fit.tau, fit.tau_error)
            for kind, fit in control_t2_fits.items()
        }
    if _TARGET_T2RHO_ECHO in echo_fits:
        target_t2rho_fits = cast(
            Mapping[str, ExponentialDecayFit],
            echo_fits[_TARGET_T2RHO_ECHO],
        )
        decay_times["T2rho_echo"] = {
            kind: _value_with_error(fit.tau, fit.tau_error)
            for kind, fit in target_t2rho_fits.items()
        }
    if "cr_on_dephasing" in fits:
        dephasing_fit = cast(CrOnDephasingFit, fits["cr_on_dephasing"])
        transition_rates["cr_on_dephasing"] = {
            "gamma_phi_control": _value_with_error(
                dephasing_fit.gamma_phi_control,
                dephasing_fit.gamma_phi_control_error,
            ),
            "gamma_phi_rho_target": _value_with_error(
                dephasing_fit.gamma_phi_rho_target,
                dephasing_fit.gamma_phi_rho_target_error,
            ),
        }
    return transition_rates, decay_times


def _resolve_idle_noise(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    idle_t1: Mapping[str, float] | None,
    idle_t2_echo: Mapping[str, float] | None,
) -> tuple[IdleQubitNoise, IdleQubitNoise]:
    """Load or validate the two qubits' idle coherence inputs."""
    t1_values = (
        exp.ctx.system_manager.config_loader.load_param_data("t1")
        if idle_t1 is None
        else idle_t1
    )
    t2_values = (
        exp.ctx.system_manager.config_loader.load_param_data("t2_echo")
        if idle_t2_echo is None
        else idle_t2_echo
    )
    try:
        control_noise = IdleQubitNoise(
            t1=float(t1_values[control_qubit]),
            t2_echo=float(t2_values[control_qubit]),
        )
        target_noise = IdleQubitNoise(
            t1=float(t1_values[target_qubit]),
            t2_echo=float(t2_values[target_qubit]),
        )
    except KeyError as exc:
        raise ValueError(
            f"Idle coherence data are missing for qubit {exc.args[0]}."
        ) from exc
    return control_noise, target_noise


def _validate_forward_model_gate_pair(
    zx90_no_echo: PulseSchedule | None,
    zx90_echo: PulseSchedule | None,
) -> None:
    """Validate timing metadata and echo modes used by the forward model."""
    if zx90_no_echo is None:
        raise ValueError("zx90_no_echo is required for the forward model.")
    if zx90_echo is None:
        raise ValueError("zx90_echo is required for the forward model.")
    no_echo_timing = extract_zx90_gate_timing(zx90_no_echo)
    if no_echo_timing.echo:
        raise ValueError("zx90_no_echo must be an un-echoed ZX90 gate.")
    echo_timing = extract_zx90_gate_timing(zx90_echo)
    if not echo_timing.echo:
        raise ValueError("zx90_echo must be an echoed ZX90 gate.")


def _validate_declared_echo_mode(
    gate: PulseSchedule,
    *,
    parameter_name: str,
    expected_echo: bool,
) -> None:
    """Reject a gate override whose available echo metadata is contradictory."""
    echo = vars(gate).get("echo")
    if echo is None:
        return
    if not isinstance(echo, bool):
        raise TypeError(f"{parameter_name} echo metadata must be a boolean.")
    if echo != expected_echo:
        expected = "echoed" if expected_echo else "un-echoed"
        raise ValueError(f"{parameter_name} must be an {expected} ZX90 gate.")


def _base_cr_on_noise_from_fits(
    fits: Mapping[str, object],
) -> tuple[CrOnNoise | None, str | None]:
    """Build the measured CR-on dissipation model needed by the forward fit."""
    rate_fit = cast(
        Mapping[str, ThreeLevelRateFit],
        fits["control_rate_model"],
    )["actual"]
    t1rho_fit = cast(
        Mapping[str, TargetT1RhoFit],
        fits["target_t1rho"],
    )["actual"]
    leakage_fit = cast(
        Mapping[str, TargetLeakageFit],
        fits["target_leakage"],
    )["actual"]
    failed = [
        name
        for name, success in (
            ("control rate", rate_fit.success),
            ("target T1rho", t1rho_fit.success),
            ("target leakage", leakage_fit.success),
        )
        if not success
    ]
    if failed:
        return None, f"Required fit failed: {', '.join(failed)}."
    try:
        noise = CrOnNoise(
            gamma_control_g_to_e=rate_fit.gamma_ge_up,
            gamma_control_e_to_g=rate_fit.gamma_ge_down,
            gamma_control_e_to_f=rate_fit.gamma_ef_up,
            gamma_control_f_to_e=rate_fit.gamma_ef_down,
            target_t1rho=t1rho_fit.t1rho,
            target_leakage_rate=leakage_fit.leakage_rate,
            target_seepage_rate=leakage_fit.seepage_rate,
        )
    except (TypeError, ValueError) as exc:
        return None, f"Could not construct the CR-on noise model: {exc}"
    return noise, None


def _fidelity_limits(
    analysis: CrPulseFidelityAnalysis | None,
) -> dict[str, object] | None:
    """Build a concise scalar fidelity summary."""
    if analysis is None:
        return None
    if not analysis.success or analysis.simulation is None:
        return {"success": False, "message": analysis.message}
    simulation = analysis.simulation
    return {
        "success": True,
        "idle_coherence_limited_fidelity": (simulation.idle_coherence_limited_fidelity),
        "cr_on_coherence_limited_fidelity": (
            simulation.cr_on_coherence_limited_fidelity
        ),
        "cr_on_dissipative_limited_fidelity": (
            simulation.cr_on_dissipative_limited_fidelity
        ),
        "average_leakage": simulation.average_leakage,
    }


def _print_fidelity_analysis(analysis: CrPulseFidelityAnalysis) -> None:
    """Print the fidelity limits, or the reason that analysis failed."""
    simulation = analysis.simulation
    if not analysis.success or simulation is None:
        print(f"CR-pulse fidelity simulation failed: {analysis.message}")
        return
    print("CR-pulse fidelity simulation:")
    print(
        "  Idle coherence limit:       "
        f"{simulation.idle_coherence_limited_fidelity:.6%}"
    )
    print(
        "  CR-on coherence limit:      "
        f"{simulation.cr_on_coherence_limited_fidelity:.6%}"
    )
    print(
        "  CR-on dissipative limit:    "
        f"{simulation.cr_on_dissipative_limited_fidelity:.6%}"
    )
    print(f"  Average leakage:             {simulation.average_leakage:.6%}")


def characterize_cr_pulse_coherence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    protocols: Collection[str] | str | None = None,
    measure_orthogonal_components: bool = True,
    n_values: Sequence[int] | None = None,
    zx90_no_echo: PulseSchedule | None = None,
    zx90_echo: PulseSchedule | None = None,
    n_shots: int | None = _DEFAULT_N_SHOTS,
    calibration_n_shots: int | None = _DEFAULT_CALIBRATION_N_SHOTS,
    shot_interval: float | None = None,
    covariance_rcond: float = 1e-12,
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = 0,
    bootstrap_confidence_level: float = 0.95,
    relative_uncertainty_threshold: float = 1.0,
    echo_fit_method: EchoFitMethod = "auto",
    run_fidelity_simulation: bool | None = None,
    idle_t1: Mapping[str, float] | None = None,
    idle_t2_echo: Mapping[str, float] | None = None,
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
        orthogonal Pauli components. Defaults to `True`.
    n_values
        Unique increasing nonnegative repetition indices beginning at zero.
        Defaults to `(0, 1, 2, 3, 5, 8, 13, 21, 34, 55)`.
    zx90_no_echo
        Optional full un-echoed ZX90 schedule override. By default, two
        identical `exp.pulse.zx90(..., echo=False)` CR lobes are joined to
        produce a ZX90-equivalent schedule.
    zx90_echo
        Optional echoed ZX90 schedule override.
    n_shots
        Shots per measurement configuration. Must be at least two. Defaults to
        4096.
    calibration_n_shots
        Shots per GEF calibration configuration. Must be at least two. Defaults
        to 8192.
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
    relative_uncertainty_threshold
        Maximum relative uncertainty retained by the shared target T1rho and
        population-rate fits. Unresolved leakage/seepage rates use
        reduced-model fallbacks. Defaults to one.
    echo_fit_method
        Primary fit for the actual control-X and target-Z echo curves.
        `"auto"` uses their joint physical forward model when all four
        protocols and model inputs are available, otherwise it falls back to
        offset exponentials. `"forward"` requires the joint model, while
        `"exponential"` disables it. References always use offset
        exponentials. Defaults to `"auto"`.
    run_fidelity_simulation
        Whether to calculate dissipative ZX90 fidelity limits from the fitted
        CR-on noise. The default `None` enables it automatically after a
        successful forward fit. `True` requires the forward-fit inputs;
        `False` disables only the final fidelity calculation, not fitting.
    idle_t1
        Optional qubit-to-T1 mapping in ns for the echo forward model and
        fidelity simulation. When omitted, the stored `t1` parameters are
        loaded.
    idle_t2_echo
        Optional qubit-to-T2-echo mapping in ns for the echo forward model and
        fidelity simulation. When omitted, the stored `t2_echo` parameters
        are loaded.
    enable_tqdm
        Whether to show progress over `n_values`.
    plot
        Whether to display all generated figures.

    Returns
    -------
    Result
        Raw measurements, derived observables, actual/reference fits, timing
        metadata, optional GEF calibration and fidelity simulation, and
        figures for selected protocols.

    Notes
    -----
    `control_ground` prepares control `|0>` and target `|+>`, applies an
    un-echoed ZX90 `4n` times, and finishes with target Y90. The default
    un-echoed ZX90 is two consecutive, same-sign `echo=False` CR primitives;
    each primitive is one lobe of Qubex's calibrated echoed ZX90.
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
    `control_excited` are selected. By default, `control_t2_echo` measures
    X/Y/Z and `target_t2rho_echo` measures Z/X/Y, with the primary component
    listed first. Set `measure_orthogonal_components=False` to measure only the
    primary components. Orthogonal components are plotted as reference
    information without fitting. The upper plot axis counts ZX90 schedule
    calls; one echoed ZX90 internally contains two physical CR lobes in the
    current pulse implementation. Fitted rates use `1/ns`; all returned decay
    times use ns. Convenient scalar summaries are available in
    `result.data["transition_rates"]` and `result.data["decay_times"]`.
    The forward fit uses only actual data from all four protocols; references
    remain diagnostics and are never subtracted from CR-on rates. Successful
    fidelity limits are printed after fitting. In automatic mode, unavailable
    model inputs select exponential echo fits without affecting measurements.

    The two control-state target-polarization curves share one zero-asymptote
    T1rho fit, and their F populations share one effective leakage/seepage fit.
    When the forward fit succeeds, the actual control- and target-echo plot
    curves come from the full physical model with affine SPAM nuisance
    parameters, even if final fidelity calculation is disabled. Their
    reference curves retain the diagnostic exponential fit. Actual
    exponential fits are also retained as phenomenological decay summaries
    and as a forward-fit fallback.
    """
    selected_protocols = _validate_protocols(protocols)
    resolved_echo_fit_method = _validate_echo_fit_method(echo_fit_method)
    if not isinstance(measure_orthogonal_components, bool):
        raise TypeError("measure_orthogonal_components must be a boolean.")
    if run_fidelity_simulation is not None and not isinstance(
        run_fidelity_simulation, bool
    ):
        raise TypeError("run_fidelity_simulation must be a boolean or None.")
    if run_fidelity_simulation is True and selected_protocols != _PROTOCOLS:
        raise ValueError("Fidelity simulation requires all four protocols.")
    if run_fidelity_simulation is True and resolved_echo_fit_method == "exponential":
        raise ValueError("Fidelity simulation requires a CR echo-decay forward fit.")
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
        default=_DEFAULT_N_SHOTS,
        name="n_shots",
    )
    resolved_calibration_n_shots = (
        _resolve_shot_count(
            calibration_n_shots,
            default=_DEFAULT_CALIBRATION_N_SHOTS,
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
    resolved_relative_uncertainty_threshold = _positive_real(
        relative_uncertainty_threshold,
        default=1.0,
        name="relative_uncertainty_threshold",
    )

    control = exp.ctx.resolve_qubit_label(control_qubit)
    target = exp.ctx.resolve_qubit_label(target_qubit)
    if control == target:
        raise ValueError("control_qubit and target_qubit must be different.")
    if selected_gef_protocols and zx90_no_echo is None:
        zx90_no_echo = _build_un_echoed_zx90(exp, control, target)
    if selected_pauli_protocols and zx90_echo is None:
        zx90_echo = exp.pulse.zx90(control, target, echo=True)
    if selected_gef_protocols and zx90_no_echo is not None:
        _validate_declared_echo_mode(
            zx90_no_echo,
            parameter_name="zx90_no_echo",
            expected_echo=False,
        )
    if selected_pauli_protocols and zx90_echo is not None:
        _validate_declared_echo_mode(
            zx90_echo,
            parameter_name="zx90_echo",
            expected_echo=True,
        )

    control_idle_noise: IdleQubitNoise | None = None
    target_idle_noise: IdleQubitNoise | None = None
    forward_fit_enabled = False
    forward_fit_skip_reason = (
        "All four protocols are required for the CR echo-decay forward fit."
        if selected_protocols != _PROTOCOLS
        else None
    )
    if resolved_echo_fit_method == "forward" and selected_protocols != _PROTOCOLS:
        raise ValueError(forward_fit_skip_reason)
    if resolved_echo_fit_method != "exponential" and selected_protocols == _PROTOCOLS:
        try:
            _validate_forward_model_gate_pair(zx90_no_echo, zx90_echo)
            control_idle_noise, target_idle_noise = _resolve_idle_noise(
                exp,
                control,
                target,
                idle_t1,
                idle_t2_echo,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            forward_fit_skip_reason = str(exc)
        else:
            forward_fit_enabled = True
            forward_fit_skip_reason = None
    elif resolved_echo_fit_method == "exponential":
        forward_fit_skip_reason = "echo_fit_method='exponential' was requested."
    if resolved_echo_fit_method == "forward" and not forward_fit_enabled:
        raise ValueError(
            forward_fit_skip_reason or "Forward-fit inputs are unavailable."
        )
    if run_fidelity_simulation is True and not forward_fit_enabled:
        raise ValueError(
            forward_fit_skip_reason or "Fidelity simulation inputs are unavailable."
        )

    fidelity_simulation_enabled = (
        run_fidelity_simulation is not False and forward_fit_enabled
    )
    fidelity_simulation_skip_reason = (
        None
        if fidelity_simulation_enabled
        else (
            "run_fidelity_simulation=False was requested."
            if run_fidelity_simulation is False
            else forward_fit_skip_reason
        )
    )

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
        target,
        selected_protocols,
        resolved_relative_uncertainty_threshold,
    )
    base_cr_noise: CrOnNoise | None = None
    fitted_cr_noise: CrOnNoise | None = None
    dephasing_fit: CrOnDephasingFit | None = None
    if forward_fit_enabled:
        base_cr_noise, fit_input_error = _base_cr_on_noise_from_fits(fits)
        if fit_input_error is not None:
            forward_fit_skip_reason = fit_input_error
        else:
            try:
                echo_model = prepare_cr_echo_decay_model(
                    extract_zx90_gate_timing(cast(PulseSchedule, zx90_echo)),
                    cast(IdleQubitNoise, control_idle_noise),
                    cast(IdleQubitNoise, target_idle_noise),
                    cast(CrOnNoise, base_cr_noise),
                    exp.pulse.x180(control).duration,
                    exp.pulse.x180(target).duration,
                )
                dephasing_fit = fit_cr_on_dephasing(
                    echo_model,
                    normalized_n_values,
                    pauli_expectations[_CONTROL_T2_ECHO]["X"]["actual"],
                    pauli_expectations[_TARGET_T2RHO_ECHO]["Z"]["actual"],
                    _finite_errors_or_none(
                        pauli_errors[_CONTROL_T2_ECHO]["X"]["actual"]
                    ),
                    _finite_errors_or_none(
                        pauli_errors[_TARGET_T2RHO_ECHO]["Z"]["actual"]
                    ),
                )
                fits["cr_on_dephasing"] = dephasing_fit
                if dephasing_fit.success:
                    fitted_cr_noise = replace(
                        cast(CrOnNoise, base_cr_noise),
                        gamma_phi_control=dephasing_fit.gamma_phi_control,
                        gamma_phi_rho_target=dephasing_fit.gamma_phi_rho_target,
                    )
                    forward_fit_skip_reason = None
                else:
                    forward_fit_skip_reason = dephasing_fit.message
            except (
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                forward_fit_skip_reason = str(exc)

    primary_echo_fit_model = (
        None
        if not selected_pauli_protocols
        else (
            "forward"
            if dephasing_fit is not None and dephasing_fit.success
            else "offset_exponential"
        )
    )
    fidelity_analysis: CrPulseFidelityAnalysis | None = None
    if fidelity_simulation_enabled:
        if fitted_cr_noise is None:
            fidelity_analysis = CrPulseFidelityAnalysis(
                success=False,
                message=(
                    "CR echo-decay forward fit failed: "
                    f"{forward_fit_skip_reason or 'unknown failure'}"
                ),
                dephasing_fit=dephasing_fit,
                simulation=None,
            )
        else:
            try:
                simulation = simulate_cr_pulse_fidelity(
                    cast(PulseSchedule, zx90_echo),
                    cast(IdleQubitNoise, control_idle_noise),
                    cast(IdleQubitNoise, target_idle_noise),
                    fitted_cr_noise,
                )
            except (
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                fidelity_analysis = CrPulseFidelityAnalysis(
                    success=False,
                    message=f"Fidelity simulation failed: {exc}",
                    dephasing_fit=dephasing_fit,
                    simulation=None,
                )
            else:
                fidelity_analysis = CrPulseFidelityAnalysis(
                    success=True,
                    message="Forward fit and fidelity simulation completed.",
                    dephasing_fit=dephasing_fit,
                    simulation=simulation,
                )
        _print_fidelity_analysis(fidelity_analysis)
        if not fidelity_analysis.success:
            fidelity_simulation_skip_reason = fidelity_analysis.message
    elif run_fidelity_simulation is None and fidelity_simulation_skip_reason:
        print(
            f"CR-pulse fidelity simulation skipped: {fidelity_simulation_skip_reason}"
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
    fit_status = {
        "control_leakage_model": (
            cast(
                Mapping[str, ThreeLevelRateFit],
                fits["control_rate_model"],
            )["actual"].leakage_model
            if "control_rate_model" in fits
            else None
        ),
        "target_leakage_model": (
            cast(
                Mapping[str, TargetLeakageFit],
                fits["target_leakage"],
            )["actual"].model
            if "target_leakage" in fits
            else None
        ),
        "echo_actual_primary_model": primary_echo_fit_model,
        "echo_actual_diagnostic_model": (
            "offset_exponential" if selected_pauli_protocols else None
        ),
        "echo_reference_model": (
            "offset_exponential" if selected_pauli_protocols else None
        ),
        "forward_fit_success": bool(
            dephasing_fit is not None and dephasing_fit.success
        ),
        "forward_fit_skip_reason": forward_fit_skip_reason,
    }
    fidelity_model_metadata = (
        {
            **fidelity_analysis.simulation.model_metadata,
            **fit_status,
        }
        if fidelity_analysis is not None and fidelity_analysis.simulation is not None
        else None
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
            "fidelity_analysis": fidelity_analysis,
            "fidelity_limits": _fidelity_limits(fidelity_analysis),
            "fidelity_model_metadata": fidelity_model_metadata,
            "base_cr_on_noise": base_cr_noise,
            "fitted_cr_on_noise": fitted_cr_noise,
            "fit_status": fit_status,
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
                "relative_uncertainty_threshold": (
                    resolved_relative_uncertainty_threshold
                ),
                "echo_fit_method": resolved_echo_fit_method,
                "forward_fit_enabled": forward_fit_enabled,
                "run_fidelity_simulation": run_fidelity_simulation,
                "fidelity_simulation_enabled": fidelity_simulation_enabled,
                "fidelity_simulation_skip_reason": (fidelity_simulation_skip_reason),
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
    "CrPulseFidelityAnalysis",
    "EchoFitMethod",
    "ExponentialDecayFit",
    "ThreeLevelRateFit",
    "characterize_cr_pulse_coherence",
    "fit_exponential_decay",
    "fit_three_level_rate_model",
]
