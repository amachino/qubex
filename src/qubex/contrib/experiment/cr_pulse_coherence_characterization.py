"""Characterize relaxation, leakage, and coherence under repeated CR pulses."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, PulseSchedule, Waveform

from ._single_shot_batch import measure_single_shot_batch
from .cr_pulse_coherence_analysis import (
    CrPulseCoherenceMeasurements,
    CrPulseFidelityAnalysis,
    analyze_cr_pulse_coherence,
    plot_cr_pulse_coherence,
)
from .cr_pulse_fidelity_simulation import (
    IdleQubitNoise,
    extract_zx90_gate_timing,
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

DEFAULT_N_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55)
_DEFAULT_N_SHOTS = 4096
_DEFAULT_CALIBRATION_N_SHOTS = 8192
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
_STATE_NAMES = ("g", "e", "f")


@dataclass(frozen=True)
class _PauliMeasurement:
    """Store one single-basis Pauli expectation measurement."""

    expectation: float
    standard_error: float
    raw_iq: NDArray[np.complex128]


@dataclass(frozen=True)
class _ProtocolSequences:
    """Store selected schedules and timing metadata for one n value."""

    sequences: dict[str, PulseSchedule]
    evolution_durations: dict[str, float]


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

    Qubex calibrates one `echo=False` CR primitive as one lobe of the
    echoed ZX90 (nominally ZX45). Repeating that schedule twice preserves the
    calibrated ramps, cancellation tone, rotary tone, sign, and phase while
    producing the intended un-echoed ZX90 rotation.
    """
    cr_lobe = exp.pulse.zx90(control_qubit, target_qubit, echo=False)
    timing = extract_zx90_gate_timing(cr_lobe)
    if timing.echo:
        raise ValueError("exp.pulse.zx90(..., echo=False) returned an echoed CR gate.")

    zx90 = cr_lobe.repeated(2)
    # `repeated` intentionally returns a plain PulseSchedule. Attach
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
    schedule.set_frequencies(evolution.get_frequencies())
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
    block.set_frequencies(zx90.get_frequencies())
    return block


def _target_t2rho_echo_block(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    zx90: PulseSchedule,
) -> PulseSchedule:
    """Build one two-ZX90 block for target T2rho echo."""
    cr_label = f"{control_qubit}-{target_qubit}"
    z180 = exp.pulse.z180()

    with PulseSchedule() as block:
        block.call(zx90, copy=True)
        block.barrier()

        block.add(target_qubit, z180)
        block.add(cr_label, z180)

        block.barrier()
        block.call(zx90, copy=True)

    block.set_frequencies(zx90.get_frequencies())
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
    schedule.set_frequencies(evolution.get_frequencies())
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
            exp, control_qubit, target_qubit, echo_schedule
        ).repeated(2 * n)
        evolutions[f"{_TARGET_T2RHO_ECHO}_reference"] = _target_t2rho_echo_block(
            exp, control_qubit, target_qubit, reference_unit
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
    )


def _build_pauli_measurement_sequence(
    exp: Experiment,
    sequence: PulseSchedule,
    target: str,
    basis: _PauliBasis,
) -> PulseSchedule:
    """Append the requested Pauli analyzer to a copied preparation."""
    measurement_sequence = sequence.copy()
    with measurement_sequence:
        analyzer: Waveform | None = None
        if basis == "X":
            analyzer = exp.pulse.y90m(target)
        elif basis == "Y":
            analyzer = exp.pulse.x90(target)
        if analyzer is not None:
            measurement_sequence.barrier()
            measurement_sequence.add(target, analyzer)
    return measurement_sequence


def _measure_pauli_batch(
    exp: Experiment,
    requests: Sequence[tuple[PulseSchedule, str, _PauliBasis]],
    *,
    n_shots: int,
    shot_interval: float,
) -> list[_PauliMeasurement]:
    """Acquire all requested Pauli configurations in one single-shot sweep."""
    schedules = [
        _build_pauli_measurement_sequence(exp, sequence, target, basis)
        for sequence, target, basis in requests
    ]
    results = measure_single_shot_batch(
        exp,
        schedules,
        n_shots=n_shots,
        shot_interval=shot_interval,
    )
    measurements = []
    for (_, target, _), result in zip(requests, results, strict=True):
        if target not in result:
            raise ValueError(f"Pauli measurement did not return `{target}`.")
        measurements.append(_summarize_pauli_iq(exp, result[target], target))
    return measurements


def _summarize_pauli_iq(
    exp: Experiment,
    iq: NDArray[np.complex128],
    target: str,
) -> _PauliMeasurement:
    """Normalize single-shot IQ and compute a Pauli mean and standard error."""
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
    analytic_fit: GefPopulationFit,
) -> tuple[NDArray[np.float64], Literal["bootstrap", "analytic", "unweighted"]]:
    """Choose bootstrap SE, then analytic GLS SE, then an unweighted fallback."""
    if bootstrap.unavailable_reason is None and bootstrap.standard_error is not None:
        bootstrap_error = np.asarray(bootstrap.standard_error, dtype=np.float64)
        if bootstrap_error.shape == (len(_STATE_NAMES),) and np.all(
            np.isfinite(bootstrap_error) & (bootstrap_error >= 0)
        ):
            return bootstrap_error, "bootstrap"
    analytic_error = np.asarray(
        analytic_fit.population_standard_error,
        dtype=np.float64,
    )
    if analytic_error.shape == (len(_STATE_NAMES),) and np.all(
        np.isfinite(analytic_error) & (analytic_error >= 0)
    ):
        return analytic_error, "analytic"
    return np.full(len(_STATE_NAMES), np.nan), "unweighted"


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
    analytic_fit: GefPopulationFit,
) -> tuple[float, Literal["bootstrap", "analytic", "unweighted"]]:
    """Choose bootstrap or analytic uncertainty for the GE polarization."""
    if bootstrap.unavailable_reason is not None:
        samples = np.empty((0, len(_STATE_NAMES)), dtype=np.float64)
    else:
        samples = np.asarray(bootstrap.samples, dtype=np.float64)
    if samples.ndim == 2 and samples.shape[1] == len(_STATE_NAMES):
        values = _polarization(samples)
        finite = values[np.isfinite(values)]
        if finite.size >= 2:
            return float(np.std(finite, ddof=1)), "bootstrap"

    population = np.asarray(analytic_fit.population, dtype=np.float64)
    covariance = np.asarray(analytic_fit.population_covariance, dtype=np.float64)
    if population.shape == (len(_STATE_NAMES),) and covariance.shape == (
        len(_STATE_NAMES),
        len(_STATE_NAMES),
    ):
        denominator = population[0] + population[1]
        if (
            denominator > np.finfo(float).eps
            and np.all(np.isfinite(population))
            and np.all(np.isfinite(covariance))
        ):
            gradient = np.array(
                [
                    -2 * population[1] / denominator**2,
                    2 * population[0] / denominator**2,
                    0.0,
                ],
                dtype=np.float64,
            )
            variance = float(gradient @ covariance @ gradient)
            variance_scale = max(
                float(np.linalg.norm(covariance, ord=2)),
                np.finfo(float).eps,
            )
            if np.isfinite(variance) and variance >= -1e-12 * variance_scale:
                return float(np.sqrt(max(variance, 0.0))), "analytic"
    return float("nan"), "unweighted"


def _optional_probability(value: float | None, *, name: str) -> float | None:
    """Validate an optional probability threshold in [0, 1]."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number in [0, 1] or None.")
    resolved = float(value)
    if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be in [0, 1] or None.")
    return resolved


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


def _forward_fit_skip_reason(
    forward_model_error: str | None,
    fit_status: Mapping[str, object],
) -> str:
    """Return one readable reason for an automatically skipped simulation."""
    if forward_model_error is not None:
        return forward_model_error
    reasons_by_protocol = cast(
        Mapping[str, str | None],
        fit_status["forward_fit_skip_reason"],
    )
    reasons = tuple(
        dict.fromkeys(reason for reason in reasons_by_protocol.values() if reason)
    )
    return "; ".join(reasons) or "required forward fits were not available."


def _print_fidelity_analysis(
    analysis: CrPulseFidelityAnalysis,
    fixed_zero_unresolved_rates: tuple[str, ...],
) -> None:
    """Print simulated fidelity limits or the analysis failure reason."""
    simulation = analysis.simulation
    if not analysis.success:
        print(f"CR-pulse fidelity analysis failed: {analysis.message}")
        return
    if simulation is None:
        print(f"CR-pulse fidelity analysis: {analysis.message}")
        return
    print("CR-pulse fidelity simulation:")
    uncertainty = analysis.linear_uncertainty
    estimates = (
        (
            "Idle coherence limit",
            simulation.idle_coherence_limited_fidelity,
            None
            if uncertainty is None
            else uncertainty.idle_coherence_limited_fidelity_standard_error,
        ),
        (
            "CR-on coherence limit",
            simulation.cr_on_coherence_limited_fidelity,
            None
            if uncertainty is None
            else uncertainty.cr_on_coherence_limited_fidelity_standard_error,
        ),
        (
            "CR-on dissipative limit",
            simulation.cr_on_dissipative_limited_fidelity,
            None
            if uncertainty is None
            else uncertainty.cr_on_dissipative_limited_fidelity_standard_error,
        ),
        (
            "Average leakage",
            simulation.average_leakage,
            None if uncertainty is None else uncertainty.average_leakage_standard_error,
        ),
    )
    for label, estimate, standard_error in estimates:
        if (
            uncertainty is not None
            and uncertainty.success
            and standard_error is not None
        ):
            lower = np.clip(estimate - 1.96 * standard_error, 0.0, 1.0)
            upper = np.clip(estimate + 1.96 * standard_error, 0.0, 1.0)
            print(
                f"  {label + ':':29} {estimate:.6%} ± {standard_error:.6%} "
                f"(1 sigma; approx. 95% [{lower:.6%}, {upper:.6%}])"
            )
        else:
            print(f"  {label + ':':29} {estimate:.6%}")
    if uncertainty is not None:
        if uncertainty.success:
            print(
                "    Local covariance propagation; fit-stage cross covariance ignored."
            )
            print("    Idle T1/T2 uncertainty was not propagated.")
        else:
            print(f"    Fidelity uncertainty unavailable: {uncertainty.message}")
    conservative = analysis.conservative_simulation_95
    if conservative is not None:
        print("  Unresolved-outward-rate 95%-upper sensitivity scenario:")
        print(
            "    CR-on dissipative limit: "
            f"{conservative.cr_on_dissipative_limited_fidelity:.6%}"
        )
        print(f"    Average leakage:          {conservative.average_leakage:.6%}")
    if fixed_zero_unresolved_rates:
        print(
            "  Conditional result: unresolved rates were fixed to zero: "
            + ", ".join(fixed_zero_unresolved_rates)
        )


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
    relative_uncertainty_threshold: float = 0.5,
    target_leakage_warning_threshold: float | None = 0.01,
    run_fidelity_simulation: bool | None = None,
    propagate_fidelity_uncertainty: bool | None = None,
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
        Maximum relative uncertainty retained by target T1rho and population-
        rate fits. Unresolved leakage/seepage rates use reduced-model
        fallbacks. Defaults to 0.5.
    target_leakage_warning_threshold
        Warn when the maximum actual target F-state population over the two
        GEF protocols exceeds this value. Defaults to 0.01; `None` disables
        the warning. This diagnostic does not stop analysis or simulation.
    run_fidelity_simulation
        Whether to calculate dissipative ZX90 fidelity limits from the fitted
        CR-on noise. The default `None` enables it automatically after both
        independent forward fits succeed. `True` requires all fit inputs;
        `False` disables only the final fidelity calculation.
    propagate_fidelity_uncertainty
        Whether to propagate the local block-diagonal fit covariance through
        the fidelity simulator using numerical derivatives. `None` (default)
        enables propagation automatically only when the nominal fidelity
        simulation runs; `True` explicitly requires propagation and raises
        `ValueError` when its inputs are unavailable; `False` disables it.
        A/B/C/D fits are not repeated.
    idle_t1
        Optional qubit-to-T1 mapping in ns for the echo forward fits and
        fidelity simulation. When omitted, the stored `t1` parameters are
        loaded.
    idle_t2_echo
        Optional qubit-to-T2-echo mapping in ns for the echo forward fits and
        fidelity simulation. When omitted, the stored `t2_echo` parameters
        are loaded.
    enable_tqdm
        Whether to show progress over `n_values`.
    plot
        Whether to display all generated figures.

    Returns
    -------
    Result
        `data["measurements"]` is a reusable
        `CrPulseCoherenceMeasurements`; `data["analysis"]` contains all fit
        results and the optional fidelity analysis; `data["raw_data"]`
        contains calibration, IQ, and bootstrap details. Measurement and
        analysis settings are kept separately. `Result.figures` contains the
        explicitly ranged data-and-fit figures for the selected protocols.

    Notes
    -----
    Acquisition uses one calibration sweep when A/B are selected, then at
    most two single-shot sweeps per n: one for all GEF configurations and
    one for all selected Pauli configurations. Progress advances after each
    n completes. The default ten-point, four-protocol run uses 21 sweeps.
    Backend support and `measurement.schedule_packing` settings determine
    hardware execution counts and timeline chunking.

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
    current pulse implementation. Each target F-population panel chooses its
    own explicit `[0, upper]` range from the measurements, upper error bars,
    and fitted curves. Fitted rates use `1/ns`; all returned decay times use
    ns. Scalar summaries are available as
    `result.data["analysis"].transition_rates` and
    `result.data["analysis"].decay_times`.
    Population error bars prefer the final joint bootstrap. If that is
    unavailable, the analytic GLS covariance from each
    `GefPopulationFit` is used; this analytic fallback conditions on the GEF
    calibration and therefore does not include its finite-shot uncertainty.
    The control-X and target-Z actual curves are forward-fitted independently;
    each fit varies only its corresponding CR-on dephasing rate. Both require
    the dissipation model inferred from `control_ground` and
    `control_excited`. References remain exponential diagnostics and are never
    subtracted from CR-on rates. Successful fidelity limits are printed after
    fitting. When requested, their standard errors come from a local
    nine-rate delta method that retains covariance within each fit but ignores
    covariance between the A/B control, A/B target, C, and D fit stages. It
    does not repeat those fits or propagate idle T1/T2 uncertainty.

    Actual A/B target-polarization and F-population curves are fitted jointly
    with a two-qutrit Lindblad model after fixing the control-transition rates
    obtained from the A/B control populations. Target T1rho, leakage, and
    seepage may differ for instantaneous control g/e states; their rates are
    averaged to form the effective single-target model used by C/D and the
    fidelity simulation. The unobserved control-f-conditioned target rate is
    approximated by the arithmetic mean of its g/e values. References retain
    phenomenological fits and are never subtracted from CR-on rates. The
    resulting CR-on model is applied to both CR drive signs because this
    workflow does not identify sign-dependent dissipative rates.
    When an independent forward fit succeeds, that protocol's actual plot
    curve comes from the physical model with affine SPAM nuisance parameters,
    even if final fidelity calculation is disabled. Offset exponential fits
    remain phenomenological summaries for actual and reference curves, but
    are not substitutes for unavailable CR-on dephasing rates.
    """
    selected_protocols = _validate_protocols(protocols)
    for name, value in (
        ("measure_orthogonal_components", measure_orthogonal_components),
        ("enable_tqdm", enable_tqdm),
        ("plot", plot),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")
    if run_fidelity_simulation is not None and not isinstance(
        run_fidelity_simulation, bool
    ):
        raise TypeError("run_fidelity_simulation must be a boolean or None.")
    if propagate_fidelity_uncertainty is not None and not isinstance(
        propagate_fidelity_uncertainty, bool
    ):
        raise TypeError("propagate_fidelity_uncertainty must be a boolean or None.")
    if propagate_fidelity_uncertainty is True and selected_protocols != _PROTOCOLS:
        raise ValueError(
            "Fidelity uncertainty propagation requires all four protocols."
        )
    if propagate_fidelity_uncertainty is True and run_fidelity_simulation is False:
        raise ValueError(
            "propagate_fidelity_uncertainty requires fidelity simulation to be enabled."
        )
    if run_fidelity_simulation is True and selected_protocols != _PROTOCOLS:
        raise ValueError("Fidelity simulation requires all four protocols.")
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
        default=0.5,
        name="relative_uncertainty_threshold",
    )
    resolved_target_leakage_warning_threshold = _optional_probability(
        target_leakage_warning_threshold,
        name="target_leakage_warning_threshold",
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
    forward_model_error: str | None = None
    if selected_gef_protocols == _GEF_PROTOCOLS and selected_pauli_protocols:
        try:
            extract_zx90_gate_timing(cast(PulseSchedule, zx90_echo))
            control_idle_noise, target_idle_noise = _resolve_idle_noise(
                exp,
                control,
                target,
                idle_t1,
                idle_t2_echo,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            forward_model_error = str(exc)
    if (
        run_fidelity_simulation is True or propagate_fidelity_uncertainty is True
    ) and forward_model_error is not None:
        raise ValueError(
            forward_model_error or "Fidelity simulation inputs are unavailable."
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
    pauli_components: dict[_PauliProtocol, tuple[_PauliBasis, ...]] = {
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

    pauli_keys: list[tuple[_PauliProtocol, str, _PauliBasis]] = [
        (protocol, condition, basis)
        for protocol in selected_pauli_protocols
        for condition in (f"{protocol}_reference", protocol)
        for basis in pauli_components[protocol]
    ]

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
        for protocol in selected_protocols:
            times_buffer[protocol].append(point.evolution_durations[protocol])

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

        if pauli_keys:
            requests: list[tuple[PulseSchedule, str, _PauliBasis]] = [
                (
                    point.sequences[condition],
                    control if protocol == _CONTROL_T2_ECHO else target,
                    basis,
                )
                for protocol, condition, basis in pauli_keys
            ]
            results = _measure_pauli_batch(
                exp,
                requests,
                n_shots=resolved_n_shots,
                shot_interval=resolved_shot_interval,
            )
            for (_, condition, basis), measurement in zip(
                pauli_keys, results, strict=True
            ):
                pauli_measurements[condition][basis].append(measurement)

    bootstrap: dict[str, dict[str, GefPopulationBootstrap]] = {}
    bootstrap_lookup: dict[tuple[int, str, str], GefPopulationBootstrap] = {}
    population_standard_error_sources: dict[str, dict[str, str]] = {}
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
            population_standard_error_sources[global_name] = {}
            for qubit in (control, target):
                population_bootstrap = bootstrap[global_name][qubit]
                bootstrap_lookup[(point_index, condition, qubit)] = population_bootstrap
                population_error, error_source = _population_standard_error(
                    population_bootstrap,
                    aggregate_gef_fits[global_name][qubit],
                )
                point_errors[condition][qubit].append(population_error)
                population_standard_error_sources[global_name][qubit] = error_source

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
    target_polarization_errors: dict[str, dict[str, NDArray[np.float64]]] = {}
    target_polarization_standard_error_sources: dict[
        str,
        dict[str, tuple[str, ...]],
    ] = {}
    for protocol in selected_gef_protocols:
        target_polarization_errors[protocol] = {}
        target_polarization_standard_error_sources[protocol] = {}
        for kind in ("actual", "reference"):
            condition = _condition_name(protocol, kind == "reference")
            polarization_errors: list[float] = []
            polarization_error_sources: list[str] = []
            for point_index, n in enumerate(normalized_n_values):
                global_name = _bootstrap_name(n, condition)
                error, source = _polarization_standard_error(
                    bootstrap_lookup[(point_index, condition, target)],
                    aggregate_gef_fits[global_name][target],
                )
                polarization_errors.append(error)
                polarization_error_sources.append(source)
            target_polarization_errors[protocol][kind] = np.asarray(
                polarization_errors,
                dtype=np.float64,
            )
            target_polarization_standard_error_sources[protocol][kind] = tuple(
                polarization_error_sources
            )

    maximum_target_leakage: float | None = None
    target_leakage_warning_triggered = False
    if selected_gef_protocols:
        maximum_target_leakage = max(
            float(np.max(populations[protocol]["actual"][target][:, 2]))
            for protocol in selected_gef_protocols
        )
        target_leakage_warning_triggered = bool(
            resolved_target_leakage_warning_threshold is not None
            and maximum_target_leakage > resolved_target_leakage_warning_threshold
        )
        if target_leakage_warning_triggered:
            warnings.warn(
                "Maximum measured target F-state population "
                f"({maximum_target_leakage:.3%}) exceeds "
                "target_leakage_warning_threshold "
                f"({resolved_target_leakage_warning_threshold:.3%}).",
                RuntimeWarning,
                stacklevel=2,
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
    for protocol in selected_pauli_protocols:
        pauli_expectations[protocol] = {}
        pauli_errors[protocol] = {}
        pauli_raw_iq[protocol] = {}
        for basis in pauli_components[protocol]:
            pauli_expectations[protocol][basis] = {}
            pauli_errors[protocol][basis] = {}
            pauli_raw_iq[protocol][basis] = {}
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

    measurements = CrPulseCoherenceMeasurements(
        control_qubit=control,
        target_qubit=target,
        protocols=selected_protocols,
        pauli_components=cast(Mapping[str, tuple[str, ...]], pauli_components),
        n_values=normalized_n_values,
        times=times,
        populations=populations,
        population_standard_errors=population_errors,
        target_polarization_standard_errors=target_polarization_errors,
        pauli_expectations=pauli_expectations,
        pauli_standard_errors=pauli_errors,
    )
    control_x180_duration: float | None = None
    target_x180_duration: float | None = None
    if _CONTROL_T2_ECHO in selected_pauli_protocols:
        control_x180_duration = exp.pulse.x180(control).duration
        target_x180_duration = exp.pulse.x180(target).duration
    analysis = analyze_cr_pulse_coherence(
        measurements,
        relative_uncertainty_threshold=resolved_relative_uncertainty_threshold,
        zx90_gate=zx90_echo if forward_model_error is None else None,
        control_idle_noise=control_idle_noise,
        target_idle_noise=target_idle_noise,
        control_x180_duration=control_x180_duration,
        target_x180_duration=target_x180_duration,
        run_fidelity_simulation=run_fidelity_simulation,
        propagate_fidelity_uncertainty=propagate_fidelity_uncertainty,
    )
    maximum_control_f = analysis.fit_status.get(
        "maximum_control_f_population_in_target_fit"
    )
    maximum_control_f_value = (
        float(maximum_control_f)
        if isinstance(maximum_control_f, (int, float, np.floating))
        and not isinstance(maximum_control_f, bool)
        else float("nan")
    )
    if (
        np.isfinite(maximum_control_f_value)
        and analysis.fit_status.get(
            "control_f_target_rate_approximation_material",
            False,
        )
        is True
    ):
        warnings.warn(
            "Control F population in the A/B target forward fit reached "
            f"{maximum_control_f_value:.3%}; inferred target rates depend on "
            "the control-F arithmetic-mean approximation.",
            RuntimeWarning,
            stacklevel=2,
        )
    fidelity_analysis = analysis.fidelity_analysis
    if fidelity_analysis is not None and (
        fidelity_analysis.simulation is not None or not fidelity_analysis.success
    ):
        _print_fidelity_analysis(
            fidelity_analysis,
            cast(
                tuple[str, ...],
                analysis.fit_status["fixed_zero_unresolved_rates"],
            ),
        )
    elif run_fidelity_simulation is None and selected_protocols == _PROTOCOLS:
        reason = _forward_fit_skip_reason(
            forward_model_error,
            analysis.fit_status,
        )
        print(f"CR-pulse fidelity simulation skipped: {reason}")

    figures = plot_cr_pulse_coherence(measurements, analysis)
    if plot:
        for figure in figures.values():
            figure.show()

    return Result(
        data={
            "measurements": measurements,
            "analysis": analysis,
            "raw_data": {
                "calibration": calibration,
                "gef_bootstrap": bootstrap,
                "gef_population_fits": aggregate_gef_fits,
                "population_standard_error_sources": (
                    population_standard_error_sources
                ),
                "target_polarization_standard_error_sources": (
                    target_polarization_standard_error_sources
                ),
                "gef_raw_iq": aggregate_raw_iq,
                "gef_moment_summaries": aggregate_moment_summaries,
                "pauli_raw_iq": pauli_raw_iq,
            },
            "measurement_options": {
                "n_shots": resolved_n_shots,
                "calibration_n_shots": resolved_calibration_n_shots,
                "shot_interval": resolved_shot_interval,
                "covariance_rcond": resolved_covariance_rcond,
                "n_bootstrap": resolved_n_bootstrap,
                "bootstrap_seed": resolved_bootstrap_seed,
                "bootstrap_confidence_level": resolved_bootstrap_confidence_level,
                "measure_orthogonal_components": measure_orthogonal_components,
                "population_standard_error_priority": (
                    "bootstrap",
                    "analytic",
                    "unweighted",
                ),
                "analytic_population_standard_error_includes_calibration_uncertainty": False,
            },
            "analysis_options": {
                "relative_uncertainty_threshold": (
                    resolved_relative_uncertainty_threshold
                ),
                "run_fidelity_simulation": run_fidelity_simulation,
                "propagate_fidelity_uncertainty": (propagate_fidelity_uncertainty),
                "propagate_fidelity_uncertainty_mode": (
                    "auto"
                    if propagate_fidelity_uncertainty is None
                    else "explicitly_enabled"
                    if propagate_fidelity_uncertainty
                    else "disabled"
                ),
                "propagate_fidelity_uncertainty_enabled": analysis.fit_status[
                    "fidelity_uncertainty_enabled"
                ],
                "forward_model_error": forward_model_error,
                "target_leakage_warning_threshold": (
                    resolved_target_leakage_warning_threshold
                ),
                "maximum_target_leakage": maximum_target_leakage,
                "target_leakage_warning_triggered": (target_leakage_warning_triggered),
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
    "characterize_cr_pulse_coherence",
]
