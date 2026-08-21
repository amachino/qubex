"""Measurement-crosstalk randomized benchmarking experiments."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from plotly.graph_objects import Figure
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis import FitResult, FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
)
from qubex.experiment.models import Result
from qubex.measurement.models.measure_result import MultipleMeasureResult
from qubex.pulse import Blank, FlatTop, PulseArray, PulseSchedule, VirtualZ, Waveform

MCMRBProtocol = Literal["mcm-rb", "delay-rb", "mcm-rep"]
_AncillaMode = Literal["standard", "randomized"]

_MCM_RB: MCMRBProtocol = "mcm-rb"
_DELAY_RB: MCMRBProtocol = "delay-rb"
_MCM_REP: MCMRBProtocol = "mcm-rep"
_MCM_RB_PROTOCOLS: tuple[MCMRBProtocol, ...] = (_MCM_RB, _DELAY_RB, _MCM_REP)
_STANDARD_ANCILLA: _AncillaMode = "standard"
_RANDOMIZED_ANCILLA: _AncillaMode = "randomized"
_RANDOMIZED_ANCILLA_PROTOCOLS: tuple[MCMRBProtocol, ...] = (_MCM_RB, _DELAY_RB)
_ANCILLA_RANDOMIZATION_SPAWN_KEY = (1,)


@dataclass(frozen=True)
class _SequenceResources:
    """Resolved labels and waveforms shared by all protocol schedules."""

    control: str
    ancilla: str
    readout_label: str
    sampling_period: float
    x90: Waveform
    measurement: Waveform
    control_during_measurement: Waveform
    ancilla_x180: Waveform | None

    @property
    def targets(self) -> tuple[str, str]:
        """Return the control and ancilla labels."""
        return self.control, self.ancilla


@dataclass(frozen=True)
class _CompiledCliffordSequence:
    """Physical waveforms for randomized Cliffords and their inverse."""

    cliffords: tuple[PulseArray, ...]
    inverse: PulseArray


@dataclass(frozen=True)
class _CompiledAncillaSequence:
    """Duration-matched ancilla I/X operations and their parity recovery."""

    operations: tuple[Waveform, ...]
    recovery: Waveform


@dataclass(frozen=True)
class _ReadoutTiming:
    """Active readout interval and ramp length on the sampling grid."""

    active_start_samples: int
    active_length_samples: int
    ramp_length_samples: int


class _TargetAnalysis(TypedDict):
    """Analysis payload for one protocol and terminal-measurement target."""

    trials: NDArray[np.float64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    fit_result: FitResult
    decay_parameter: float | None
    decay_parameter_err: float | None
    error_per_cycle: float | None
    error_per_cycle_err: float | None


_TrialData = dict[MCMRBProtocol, dict[str, NDArray[np.float64]]]
_ProtocolResults = dict[MCMRBProtocol, dict[str, _TargetAnalysis]]


def _validate_protocol(protocol: str) -> MCMRBProtocol:
    """Validate and narrow one protocol name."""
    if protocol not in _MCM_RB_PROTOCOLS:
        raise ValueError(
            f"Invalid `protocol`: {protocol!r}. Expected one of {_MCM_RB_PROTOCOLS}."
        )
    return cast(MCMRBProtocol, protocol)


def _validate_ancilla_mode(ancilla_mode: str) -> _AncillaMode:
    """Validate and narrow the ancilla sequence mode."""
    if ancilla_mode not in (_STANDARD_ANCILLA, _RANDOMIZED_ANCILLA):
        raise ValueError(
            "Invalid `ancilla_mode`: "
            f"{ancilla_mode!r}. Expected 'standard' or 'randomized'."
        )
    return cast(_AncillaMode, ancilla_mode)


def _validate_protocol_for_ancilla_mode(
    protocol: MCMRBProtocol,
    *,
    ancilla_mode: _AncillaMode,
) -> None:
    """Reject protocols incompatible with randomized ancilla recovery."""
    if ancilla_mode == _RANDOMIZED_ANCILLA and protocol == _MCM_REP:
        raise ValueError(
            "`mcm-rep` is not supported when `ancilla_mode='randomized'`; "
            "use `mcm-rb` and `delay-rb` as a matched pair."
        )


def _validate_nonnegative_integer(value: object, *, name: str) -> int:
    """Return a scalar integer after rejecting booleans and negative values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"`{name}` must be a nonnegative integer.")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"`{name}` must be a nonnegative integer.")
    return resolved


def _validate_positive_integer(value: object, *, name: str) -> int:
    """Return a scalar integer after requiring a value greater than zero."""
    resolved = _validate_nonnegative_integer(value, name=name)
    if resolved == 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return resolved


def _validate_positive_real(value: object, *, name: str) -> float:
    """Return a positive finite scalar real after rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"`{name}` must be positive and finite.")
    return resolved


def _resolve_qubit_pair(
    exp: Experiment,
    control: str,
    ancilla: str,
) -> tuple[str, str]:
    """Resolve and validate distinct control and ancilla qubit labels."""
    control_qubit = exp.ctx.resolve_qubit_label(control)
    ancilla_qubit = exp.ctx.resolve_qubit_label(ancilla)
    if control_qubit == ancilla_qubit:
        raise ValueError("`control` and `ancilla` must resolve to different qubits.")
    return control_qubit, ancilla_qubit


def _validate_waveform(
    waveform: Waveform,
    *,
    expected_sampling_period: float,
    name: str,
) -> None:
    """Validate one nonempty waveform against the experiment sampling grid."""
    if not np.isclose(
        waveform.sampling_period,
        expected_sampling_period,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"`{name}` sampling period must match the experiment sampling period: "
            f"{waveform.sampling_period} ns != {expected_sampling_period} ns."
        )
    if waveform.duration <= 0:
        raise ValueError(f"`{name}` must have positive duration.")

    if not isinstance(waveform, PulseArray):
        return
    for nested_waveform in waveform.get_flattened_waveforms(apply_frame_shifts=False):
        if not np.isclose(
            nested_waveform.sampling_period,
            expected_sampling_period,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Every pulse in `{name}` must use the experiment sampling period: "
                f"{nested_waveform.sampling_period} ns != "
                f"{expected_sampling_period} ns."
            )


def _blank(duration: float, sampling_period: float) -> Blank:
    """Build a blank pulse on the active sampling grid."""
    return Blank(duration=duration, sampling_period=sampling_period)


def _infer_readout_timing(measurement: Waveform) -> _ReadoutTiming:
    """Infer the active readout interval and FlatTop ramp in samples."""
    active_indices = np.flatnonzero(np.asarray(measurement.values) != 0.0)
    if active_indices.size == 0:
        raise ValueError(
            "The active readout interval cannot be inferred from an all-zero "
            "measurement waveform."
        )
    active_start_samples = int(active_indices[0])
    active_length_samples = int(active_indices[-1]) - active_start_samples + 1

    if isinstance(measurement, PulseArray):
        flat_tops = [
            element
            for element in measurement.flattened_elements
            if isinstance(element, FlatTop)
        ]
    elif isinstance(measurement, FlatTop):
        flat_tops = [measurement]
    else:
        flat_tops = []
    if len(flat_tops) > 1:
        raise ValueError(
            "Echo timing requires a measurement waveform with at most one "
            "FlatTop pulse."
        )

    ramp_length_samples = 0
    if flat_tops:
        ramp_samples = flat_tops[0].tau / measurement.sampling_period
        ramp_length_samples = round(ramp_samples)
        if not np.isclose(
            ramp_samples,
            ramp_length_samples,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "The readout ramp duration must align with the measurement "
                "sampling grid."
            )
        if 2 * ramp_length_samples > active_length_samples:
            raise ValueError(
                "The active readout duration must be at least twice the ramp duration."
            )

    return _ReadoutTiming(
        active_start_samples=active_start_samples,
        active_length_samples=active_length_samples,
        ramp_length_samples=ramp_length_samples,
    )


def _measurement_control_block(
    *,
    measurement: Waveform,
    sampling_period: float,
    echo_x180: Waveform | None,
) -> Waveform:
    """Build an idle or active-readout-aligned X-X echo block."""
    if echo_x180 is None:
        return _blank(measurement.duration, sampling_period)

    if measurement.length < 2 * echo_x180.length:
        raise ValueError(
            "The measurement duration must be at least twice the echo X180 "
            "pulse duration."
        )
    timing = _infer_readout_timing(measurement)

    # Trim half a ramp from each side of the active interval, then align the
    # X180 centers with the quarter and three-quarter points of that interval.
    # Quarter-sample units keep half-ramp and quarter-interval arithmetic exact.
    first_center_quarters = (
        4 * timing.active_start_samples
        + timing.active_length_samples
        + timing.ramp_length_samples
    )
    second_center_quarters = (
        4 * timing.active_start_samples
        + 3 * timing.active_length_samples
        - timing.ramp_length_samples
    )
    leading_quarters = first_center_quarters - 2 * echo_x180.length
    middle_quarters = (
        second_center_quarters - first_center_quarters - 4 * echo_x180.length
    )
    trailing_quarters = (
        4 * measurement.length - second_center_quarters - 2 * echo_x180.length
    )
    blank_quarters = (leading_quarters, middle_quarters, trailing_quarters)
    if any(samples < 0 for samples in blank_quarters):
        raise ValueError(
            "The measurement timing does not provide enough room for two "
            "nonoverlapping echo X180 pulses at the ramp-trimmed active "
            "readout quarter points."
        )
    if any(samples % 4 != 0 for samples in blank_quarters):
        raise ValueError(
            "The ramp-trimmed active readout quarter points do not align the "
            "echo X180 pulses to the sampling grid."
        )

    leading_duration, middle_duration, trailing_duration = (
        samples // 4 * sampling_period for samples in blank_quarters
    )
    return PulseArray(
        [
            _blank(leading_duration, sampling_period),
            echo_x180,
            _blank(middle_duration, sampling_period),
            echo_x180,
            _blank(trailing_duration, sampling_period),
        ]
    )


def _resolve_sequence_resources(
    exp: Experiment,
    *,
    control: str,
    ancilla: str,
    control_echo: bool,
    ancilla_mode: _AncillaMode,
    x90: Waveform | None,
    measurement_waveform: Waveform | None,
    echo_x180: Waveform | None,
    ancilla_x180: Waveform | None,
) -> _SequenceResources:
    """Resolve and validate labels and waveforms used by every protocol."""
    sampling_period = float(exp.ctx.measurement.sampling_period)
    if not np.isfinite(sampling_period) or sampling_period <= 0:
        raise ValueError("The experiment sampling period must be positive and finite.")

    x90_waveform = x90 if x90 is not None else exp.pulse.x90(control)
    measurement = (
        measurement_waveform
        if measurement_waveform is not None
        else exp.pulse.readout(ancilla)
    )
    _validate_waveform(
        x90_waveform,
        expected_sampling_period=sampling_period,
        name="x90",
    )
    _validate_waveform(
        measurement,
        expected_sampling_period=sampling_period,
        name="measurement_waveform",
    )

    resolved_echo: Waveform | None = None
    if control_echo:
        resolved_echo = echo_x180 if echo_x180 is not None else exp.pulse.x180(control)
        _validate_waveform(
            resolved_echo,
            expected_sampling_period=sampling_period,
            name="echo_x180",
        )

    resolved_ancilla_x180: Waveform | None = None
    if ancilla_mode == _RANDOMIZED_ANCILLA:
        resolved_ancilla_x180 = (
            ancilla_x180 if ancilla_x180 is not None else exp.pulse.x180(ancilla)
        )
        _validate_waveform(
            resolved_ancilla_x180,
            expected_sampling_period=sampling_period,
            name="ancilla_x180",
        )

    return _SequenceResources(
        control=control,
        ancilla=ancilla,
        readout_label=exp.experiment_system.resolve_read_label(ancilla),
        sampling_period=sampling_period,
        x90=x90_waveform,
        measurement=measurement,
        control_during_measurement=_measurement_control_block(
            measurement=measurement,
            sampling_period=sampling_period,
            echo_x180=resolved_echo,
        ),
        ancilla_x180=resolved_ancilla_x180,
    )


def _clifford_waveform(gates: list[str], x90: Waveform) -> PulseArray:
    """Translate one Clifford decomposition to physical and virtual pulses."""
    elements: list[Waveform | VirtualZ] = []
    for gate in gates:
        if gate == "X90":
            elements.append(x90)
        elif gate == "Z90":
            elements.append(VirtualZ(np.pi / 2))
        else:
            raise ValueError(f"Unsupported 1Q Clifford gate `{gate}`.")
    return PulseArray(elements)


def _generate_clifford_sequence(
    exp: Experiment,
    *,
    n_cliffords: int,
    seed: int | None,
    x90: Waveform,
) -> _CompiledCliffordSequence:
    """Generate and compile one randomized Clifford sequence and its inverse."""
    cliffords, inverse = (
        exp.benchmarking_service.clifford_generator.create_rb_sequences(
            n=n_cliffords,
            type="1Q",
            seed=seed,
        )
    )
    return _CompiledCliffordSequence(
        cliffords=tuple(_clifford_waveform(clifford, x90) for clifford in cliffords),
        inverse=_clifford_waveform(inverse, x90),
    )


def _compile_ancilla_sequence(
    resources: _SequenceResources,
    *,
    n_cliffords: int,
    seed: int | None,
) -> _CompiledAncillaSequence | None:
    """Compile reproducible ancilla I/X choices when randomized mode is active."""
    x180 = resources.ancilla_x180
    if x180 is None:
        return None

    seed_sequence = np.random.SeedSequence(
        seed,
        spawn_key=_ANCILLA_RANDOMIZATION_SPAWN_KEY,
    )
    values = np.random.default_rng(seed_sequence).integers(
        0,
        2,
        size=n_cliffords,
        dtype=np.int8,
    )
    identity = _blank(x180.duration, resources.sampling_period)
    operations = tuple(x180 if value else identity for value in values)
    recovery = x180 if np.count_nonzero(values) % 2 else identity
    return _CompiledAncillaSequence(operations=operations, recovery=recovery)


def _build_protocol_schedule(
    resources: _SequenceResources,
    clifford_sequence: _CompiledCliffordSequence,
    *,
    protocol: MCMRBProtocol,
    ancilla_sequence: _CompiledAncillaSequence | None = None,
) -> PulseSchedule:
    """Build one protocol schedule from shared resources and Cliffords."""
    schedule_labels = [resources.control, resources.readout_label]
    if ancilla_sequence is not None:
        schedule_labels.insert(1, resources.ancilla)
        if len(ancilla_sequence.operations) != len(clifford_sequence.cliffords):
            raise RuntimeError(
                "The ancilla sequence length must match the Clifford sequence."
            )

    with PulseSchedule(schedule_labels) as schedule:
        for cycle_index, clifford in enumerate(clifford_sequence.cliffords):
            control_operation: Waveform
            if protocol == _MCM_REP:
                control_operation = _blank(
                    clifford.duration,
                    resources.sampling_period,
                )
            else:
                control_operation = clifford
            schedule.add(resources.control, control_operation)
            schedule.barrier()

            if ancilla_sequence is not None:
                schedule.add(
                    resources.ancilla,
                    ancilla_sequence.operations[cycle_index],
                )
                schedule.barrier()

            measurement_operation: Waveform
            if protocol == _DELAY_RB:
                measurement_operation = _blank(
                    resources.measurement.duration,
                    resources.sampling_period,
                )
            else:
                measurement_operation = resources.measurement
            schedule.add(resources.readout_label, measurement_operation)
            schedule.add(
                resources.control,
                resources.control_during_measurement,
            )
            schedule.barrier()

        if ancilla_sequence is not None:
            schedule.add(resources.ancilla, ancilla_sequence.recovery)
            schedule.barrier()

        final_control_operation: Waveform
        if protocol == _MCM_REP:
            final_control_operation = _blank(
                clifford_sequence.inverse.duration,
                resources.sampling_period,
            )
        else:
            final_control_operation = clifford_sequence.inverse
        schedule.add(resources.control, final_control_operation)

    return schedule


def mcm_rb_sequence(
    exp: Experiment,
    control: str,
    ancilla: str,
    *,
    protocol: MCMRBProtocol,
    n_cliffords: int,
    seed: int | None = None,
    control_echo: bool = False,
    ancilla_mode: Literal["standard", "randomized"] = "standard",
    x90: Waveform | None = None,
    measurement_waveform: Waveform | None = None,
    echo_x180: Waveform | None = None,
    ancilla_x180: Waveform | None = None,
) -> PulseSchedule:
    """
    Build one MCM-RB, delay-RB, or MCM-repetition pulse schedule.

    Parameters
    ----------
    exp
        Experiment instance that provides pulse and Clifford services.
    control
        Qubit receiving the randomized Clifford sequence.
    ancilla
        Qubit measured during every randomized-benchmarking cycle.
    protocol
        Sequence variant: `"mcm-rb"`, `"delay-rb"`, or `"mcm-rep"`.
    n_cliffords
        Number of randomized cycles. Must be nonnegative.
    seed
        Nonnegative seed used to generate the one-qubit Clifford sequence.
    control_echo
        Whether to apply an X-X echo to the control during every measurement
        or reference-delay window. The X180 centers are placed at the quarter
        and three-quarter points of the active readout interval after trimming
        half a ramp duration from each end.
    ancilla_mode
        Ancilla sequence mode. `"standard"` preserves the original protocol.
        `"randomized"` inserts a seeded I/X180 before every measurement or
        reference delay and applies a final parity recovery. Randomized mode
        supports only `"mcm-rb"` and `"delay-rb"`.
    x90
        Optional control X90 waveform override.
    measurement_waveform
        Optional ancilla readout waveform override. With `control_echo=True`,
        the active interval is inferred from its nonzero samples. At most one
        FlatTop pulse may be present; other pulse shapes are treated as having
        zero ramp duration.
    echo_x180
        Optional control X180 waveform override used by the X-X echo. Ignored
        when `control_echo=False`.
    ancilla_x180
        Optional ancilla X180 waveform override used for randomized I/X180
        operations and parity recovery. Ignored in standard mode.

    Returns
    -------
    PulseSchedule
        Schedule before the terminal measurements are appended by execution.

    Raises
    ------
    TypeError
        Raised when a Clifford count or seed has an invalid type.
    ValueError
        Raised for invalid targets, protocol or ancilla mode names, lengths,
        waveform timing, or echo timing.

    Notes
    -----
    In standard mode, `mcm-rep` replaces every randomized Clifford and the
    final inverse with duration-matched delays. All selected protocols have
    equal total duration for the same Clifford sequence and seed.

    In randomized mode, the seed determines both the Clifford sequence and a
    separate ancilla I/X180 random stream. The final ancilla recovery is X180
    exactly when the programmed sequence has odd parity. One static I/X180
    sequence is encoded in the returned schedule.
    """
    resolved_protocol = _validate_protocol(protocol)
    resolved_ancilla_mode = _validate_ancilla_mode(ancilla_mode)
    _validate_protocol_for_ancilla_mode(
        resolved_protocol,
        ancilla_mode=resolved_ancilla_mode,
    )
    resolved_n_cliffords = _validate_nonnegative_integer(
        n_cliffords,
        name="n_cliffords",
    )
    resolved_seed = (
        None if seed is None else _validate_nonnegative_integer(seed, name="seed")
    )
    control_qubit, ancilla_qubit = _resolve_qubit_pair(exp, control, ancilla)
    resources = _resolve_sequence_resources(
        exp,
        control=control_qubit,
        ancilla=ancilla_qubit,
        control_echo=control_echo,
        ancilla_mode=resolved_ancilla_mode,
        x90=x90,
        measurement_waveform=measurement_waveform,
        echo_x180=echo_x180,
        ancilla_x180=ancilla_x180,
    )
    clifford_sequence = _generate_clifford_sequence(
        exp,
        n_cliffords=resolved_n_cliffords,
        seed=resolved_seed,
        x90=resources.x90,
    )
    ancilla_sequence = _compile_ancilla_sequence(
        resources,
        n_cliffords=resolved_n_cliffords,
        seed=resolved_seed,
    )
    return _build_protocol_schedule(
        resources,
        clifford_sequence,
        protocol=resolved_protocol,
        ancilla_sequence=ancilla_sequence,
    )


def _integer_array(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    """Validate and return a one-dimensional nonnegative integer array."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"`{name}` must be a nonempty one-dimensional array.")

    resolved: list[int] = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"`{name}` must contain integers, not booleans.")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"`{name}` must contain integers.")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"`{name}` must contain nonnegative integers.")
        if integer > np.iinfo(np.int64).max:
            raise ValueError(f"`{name}` values are too large.")
        resolved.append(integer)
    return np.asarray(resolved, dtype=np.int64)


def _resolve_n_cliffords_range(
    n_cliffords_range: ArrayLike | None,
) -> NDArray[np.int64]:
    """Return and validate the randomized-benchmarking sweep range."""
    if n_cliffords_range is None:
        n_cliffords = np.insert(
            np.ceil(np.geomspace(1, 150, 14)).astype(np.int64),
            0,
            0,
        )
    else:
        n_cliffords = _integer_array(
            n_cliffords_range,
            name="n_cliffords_range",
        )
    if len(n_cliffords) < 3:
        raise ValueError("`n_cliffords_range` must contain at least three values.")
    if np.any(np.diff(n_cliffords) <= 0):
        raise ValueError("`n_cliffords_range` must be strictly increasing.")
    return n_cliffords


def _resolve_protocols(
    protocols: Collection[MCMRBProtocol] | MCMRBProtocol | None,
    *,
    ancilla_mode: _AncillaMode,
) -> tuple[MCMRBProtocol, ...]:
    """Return unique protocol names in caller-specified order."""
    if protocols is None:
        if ancilla_mode == _RANDOMIZED_ANCILLA:
            return _RANDOMIZED_ANCILLA_PROTOCOLS
        return _MCM_RB_PROTOCOLS
    candidates = [protocols] if isinstance(protocols, str) else list(protocols)
    if not candidates:
        raise ValueError("`protocols` must not be empty.")
    resolved = cast(
        tuple[MCMRBProtocol, ...],
        tuple(_validate_protocol(protocol) for protocol in candidates),
    )
    if len(resolved) != len(set(resolved)):
        raise ValueError("`protocols` must not contain duplicate values.")
    for protocol in resolved:
        _validate_protocol_for_ancilla_mode(
            protocol,
            ancilla_mode=ancilla_mode,
        )
    return resolved


def _resolve_seeds(
    seeds: ArrayLike | None,
    *,
    n_trials: int,
) -> NDArray[np.int64]:
    """Return one validated random seed for every trial."""
    if seeds is None:
        return (
            np.random.default_rng()
            .integers(
                0,
                2**32,
                size=n_trials,
                dtype=np.uint32,
            )
            .astype(np.int64)
        )
    resolved = _integer_array(seeds, name="seeds")
    if len(resolved) != n_trials:
        raise ValueError("The number of seeds must be equal to the number of trials.")
    return resolved


def _extract_terminal_iq(
    measure_result: MultipleMeasureResult,
    exp: Experiment,
    *,
    target: str,
) -> complex:
    """Return the final capture for a qubit from a multi-capture result."""
    readout_target = exp.experiment_system.resolve_read_label(target)
    if readout_target in measure_result.data:
        data_target = readout_target
    elif target in measure_result.data:
        data_target = target
    else:
        raise KeyError(
            f"Neither `{readout_target}` nor `{target}` exists in measurement "
            f"result targets {list(measure_result.data)}."
        )
    captures = measure_result.data[data_target]
    if not captures:
        raise ValueError(f"Measurement result for `{data_target}` has no captures.")
    return complex(np.mean(np.asarray(captures[-1].kerneled)))


def _ground_state_probability(exp: Experiment, target: str, iq: complex) -> float:
    """Normalize a terminal IQ value into ground-state probability."""
    normalized = exp.pulse.rabi_params[target].normalize(
        np.asarray([iq], dtype=np.complex128)
    )
    return float(np.real(np.mean(np.asarray(normalized)))) / 2.0 + 0.5


def _acquire_trials(
    exp: Experiment,
    *,
    resources: _SequenceResources,
    protocols: tuple[MCMRBProtocol, ...],
    n_cliffords_range: NDArray[np.int64],
    seeds: NDArray[np.int64],
    n_shots: int,
    shot_interval: float,
    time_integration: bool,
    enable_tqdm: bool,
) -> _TrialData:
    """Execute all schedules and collect terminal ground-state probabilities."""
    trials: _TrialData = {
        protocol: {
            target: np.empty(
                (len(n_cliffords_range), len(seeds)),
                dtype=np.float64,
            )
            for target in resources.targets
        }
        for protocol in protocols
    }
    exp.ctx.reset_awg_and_capunits(qubits=set(resources.targets))
    progress = tqdm(
        total=len(n_cliffords_range) * len(seeds) * len(protocols),
        desc="Running MCM randomized benchmarking",
        disable=not enable_tqdm,
    )
    try:
        for n_index, n_cliffords in enumerate(n_cliffords_range):
            for trial_index, seed in enumerate(seeds):
                clifford_sequence = _generate_clifford_sequence(
                    exp,
                    n_cliffords=int(n_cliffords),
                    seed=int(seed),
                    x90=resources.x90,
                )
                ancilla_sequence = _compile_ancilla_sequence(
                    resources,
                    n_cliffords=int(n_cliffords),
                    seed=int(seed),
                )
                for protocol in protocols:
                    sequence = _build_protocol_schedule(
                        resources,
                        clifford_sequence,
                        protocol=protocol,
                        ancilla_sequence=ancilla_sequence,
                    )
                    measure_result = exp.measurement_service.execute(
                        sequence,
                        mode="avg",
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        time_integration=time_integration,
                        state_classification=False,
                        final_measurement=True,
                        reset_awg_and_capunits=False,
                        plot=False,
                    )
                    for target in resources.targets:
                        iq = _extract_terminal_iq(
                            measure_result,
                            exp,
                            target=target,
                        )
                        trials[protocol][target][n_index, trial_index] = (
                            _ground_state_probability(exp, target, iq)
                        )
                    progress.update()
    finally:
        progress.close()
    return trials


def _fit_protocol_target(
    *,
    protocol: MCMRBProtocol,
    target: str,
    n_cliffords: NDArray[np.int64],
    trials: NDArray[np.float64],
    plot: bool,
    xaxis_type: Literal["linear", "log"],
) -> _TargetAnalysis:
    """Aggregate and fit one target's terminal survival probabilities."""
    mean = np.mean(trials, axis=1)
    std = np.std(trials, axis=1)
    fit_result = fitting.fit_rb(
        target=target,
        x=n_cliffords,
        y=mean,
        error_y=std if trials.shape[1] > 1 else None,
        bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        title=f"{protocol} randomized benchmarking",
        xlabel="Number of Clifford/measurement cycles",
        ylabel="Ground-state probability",
        xaxis_type=xaxis_type,
        yaxis_type="linear",
        plot=plot,
    )
    decay_parameter: float | None = None
    decay_parameter_err: float | None = None
    error_per_cycle: float | None = None
    error_per_cycle_err: float | None = None
    if fit_result.status is FitStatus.SUCCESS:
        decay_parameter = float(fit_result["p"])
        decay_parameter_err = float(fit_result["p_err"])
        error_per_cycle = 0.5 * (1.0 - decay_parameter)
        error_per_cycle_err = 0.5 * decay_parameter_err

    return {
        "trials": trials,
        "mean": mean,
        "std": std,
        "fit_result": fit_result,
        "decay_parameter": decay_parameter,
        "decay_parameter_err": decay_parameter_err,
        "error_per_cycle": error_per_cycle,
        "error_per_cycle_err": error_per_cycle_err,
    }


def _analyze_trials(
    trials: _TrialData,
    *,
    targets: tuple[str, str],
    n_cliffords: NDArray[np.int64],
    plot: bool,
    save_image: bool,
    xaxis_type: Literal["linear", "log"],
) -> tuple[_ProtocolResults, dict[str, Figure]]:
    """Fit every protocol/target pair and collect available figures."""
    protocol_results: _ProtocolResults = {}
    figures: dict[str, Figure] = {}
    for protocol, protocol_trials in trials.items():
        target_results: dict[str, _TargetAnalysis] = {}
        for target in targets:
            target_result = _fit_protocol_target(
                protocol=protocol,
                target=target,
                n_cliffords=n_cliffords,
                trials=protocol_trials[target],
                plot=plot,
                xaxis_type=xaxis_type,
            )
            target_results[target] = target_result
            fit_result = target_result["fit_result"]
            if fit_result.status is not FitStatus.SUCCESS or fit_result.figure is None:
                continue
            figure_key = f"{protocol}:{target}"
            figures[figure_key] = fit_result.figure
            if save_image:
                viz.save_figure(
                    fit_result.figure,
                    name=f"mcm_randomized_benchmarking_{protocol}_{target}",
                )
        protocol_results[protocol] = target_results
    return protocol_results, figures


def _measurement_induced_error(
    protocol_results: _ProtocolResults,
    *,
    target: str,
) -> dict[str, float] | None:
    """Calculate one target's MCM-induced error relative to delay-RB."""
    if _MCM_RB not in protocol_results or _DELAY_RB not in protocol_results:
        return None
    mcm_result = protocol_results[_MCM_RB][target]
    delay_result = protocol_results[_DELAY_RB][target]
    p_mcm = mcm_result["decay_parameter"]
    p_mcm_err = mcm_result["decay_parameter_err"]
    p_delay = delay_result["decay_parameter"]
    p_delay_err = delay_result["decay_parameter_err"]
    if (
        p_mcm is None
        or p_mcm_err is None
        or p_delay is None
        or p_delay_err is None
        or p_delay == 0.0
    ):
        return None

    value = 0.5 * (1.0 - p_mcm / p_delay)
    error = 0.5 * np.sqrt(
        (p_mcm_err / p_delay) ** 2 + (p_mcm * p_delay_err / p_delay**2) ** 2
    )
    return {"value": float(value), "error": float(error)}


def mcm_randomized_benchmarking(
    exp: Experiment,
    control: str,
    ancilla: str,
    *,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    protocols: Collection[MCMRBProtocol] | MCMRBProtocol | None = None,
    control_echo: bool = False,
    ancilla_mode: Literal["standard", "randomized"] = "standard",
    x90: Waveform | None = None,
    measurement_waveform: Waveform | None = None,
    echo_x180: Waveform | None = None,
    ancilla_x180: Waveform | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    time_integration: bool = True,
    xaxis_type: Literal["linear", "log"] = "linear",
    plot: bool | None = None,
    save_image: bool | None = None,
    enable_tqdm: bool = False,
) -> Result:
    """
    Run selected measurement-crosstalk randomized-benchmarking protocols.

    Parameters
    ----------
    exp
        Experiment instance used to generate and execute the schedules.
    control
        Qubit receiving randomized Cliffords.
    ancilla
        Qubit measured during every randomized-benchmarking cycle.
    n_cliffords_range
        Strictly increasing nonnegative cycle counts. Defaults to 0 followed
        by 14 geometrically spaced values from 1 through 150.
    n_trials
        Number of random Clifford trials per cycle count. Defaults to 30.
    seeds
        One nonnegative integer seed for every trial. Defaults to generated
        seeds. Values must have integer types and fit in signed 64 bits.
    protocols
        Protocols to run. Defaults to all of `"mcm-rb"`, `"delay-rb"`, and
        `"mcm-rep"` in standard mode and to the matched `"mcm-rb"` and
        `"delay-rb"` pair in randomized mode. The same generated Clifford and
        ancilla I/X180 sequences are shared across protocols for each sequence
        length and seed.
    control_echo
        Whether to insert an X-X echo on the control during every measurement
        or duration-matched delay window. The X180 centers are placed at the
        quarter and three-quarter points of the active readout interval after
        trimming half a ramp duration from each end.
    ancilla_mode
        Ancilla sequence mode. `"standard"` preserves the original suite.
        `"randomized"` inserts a seeded I/X180 before every measurement or
        reference delay, followed by an X180 for odd parity or a
        duration-matched blank for even parity. Randomized mode does not
        support `"mcm-rep"`.
    x90
        Optional control X90 waveform override.
    measurement_waveform
        Optional ancilla readout waveform override. With `control_echo=True`,
        the active interval is inferred from its nonzero samples. At most one
        FlatTop pulse may be present; other pulse shapes are treated as having
        zero ramp duration.
    echo_x180
        Optional control X180 waveform override for echoed measurement blocks.
        Ignored when `control_echo=False`.
    ancilla_x180
        Optional ancilla X180 waveform override used by randomized mode.
        Ignored in standard mode.
    n_shots
        Number of shots per schedule. Defaults to the experiment default.
    shot_interval
        Positive finite interval between shots in ns. Defaults to the
        experiment default.
    time_integration
        Whether to integrate each capture over time.
    xaxis_type
        X-axis scale used by fit figures.
    plot
        Whether to display fit figures. Defaults to `True`.
    save_image
        Whether to save successful fit figures. Defaults to `False`.
    enable_tqdm
        Whether to show schedule-execution progress.

    Returns
    -------
    Result
        Raw terminal probabilities, per-protocol statistics and fits, the
        MCM-induced control error, optional randomized-ancilla population
        error, metadata, and named fit figures. Protocol results are stored
        under `result.data["protocols"][protocol][target]`.

    Raises
    ------
    TypeError
        Raised when a sweep value, seed, trial count, or shot count has an
        invalid type.
    ValueError
        Raised when targets, sweep values, seeds, protocols, ancilla mode, or
        pulse timing are invalid.

    Notes
    -----
    Intermediate MCM outcomes are intentionally discarded. Only the terminal
    capture for the control and ancilla is normalized and fitted. When both
    MCM-RB and delay-RB succeed, the reported induced error is
    `(1 - p_mcm / p_delay) / 2`.

    Control-qubit MCM-RB and delay-RB have the standard
    randomized-benchmarking exponential form under the usual assumptions.
    MCM-repetition is fitted to the same form for comparison, but its decay is
    not generally guaranteed to be exponential because the ancilla is not
    Clifford twirled.

    Randomized mode benchmarks ancilla computational-basis population
    preservation, not general single-qubit or state-assignment fidelity. Its
    induced ancilla error estimate assumes the randomized I/X180 and
    measurement channels produce compatible exponential decays. The
    uncertainty of an induced-error estimate treats the two fitted decay
    parameters as independent even though matched random sequences can
    correlate them.

    The ancilla I/X180 choices are generated once per sequence length and
    trial seed, then reused for every shot of that schedule. Randomization is
    averaged across trials rather than varied shot by shot.
    """
    resolved_ancilla_mode = _validate_ancilla_mode(ancilla_mode)
    resolved_n_cliffords = _resolve_n_cliffords_range(n_cliffords_range)
    resolved_protocols = _resolve_protocols(
        protocols,
        ancilla_mode=resolved_ancilla_mode,
    )
    resolved_n_trials = _validate_positive_integer(
        DEFAULT_RB_N_TRIALS if n_trials is None else n_trials,
        name="n_trials",
    )
    resolved_seeds = _resolve_seeds(seeds, n_trials=resolved_n_trials)
    resolved_n_shots = _validate_positive_integer(
        DEFAULT_SHOTS if n_shots is None else n_shots,
        name="n_shots",
    )
    resolved_shot_interval = _validate_positive_real(
        DEFAULT_INTERVAL if shot_interval is None else shot_interval,
        name="shot_interval",
    )
    resolved_plot = True if plot is None else plot
    resolved_save_image = False if save_image is None else save_image
    if xaxis_type not in ("linear", "log"):
        raise ValueError("`xaxis_type` must be either `linear` or `log`.")

    control_qubit, ancilla_qubit = _resolve_qubit_pair(exp, control, ancilla)
    exp.pulse.validate_rabi_params([control_qubit, ancilla_qubit])
    resources = _resolve_sequence_resources(
        exp,
        control=control_qubit,
        ancilla=ancilla_qubit,
        control_echo=control_echo,
        ancilla_mode=resolved_ancilla_mode,
        x90=x90,
        measurement_waveform=measurement_waveform,
        echo_x180=echo_x180,
        ancilla_x180=ancilla_x180,
    )
    trials = _acquire_trials(
        exp,
        resources=resources,
        protocols=resolved_protocols,
        n_cliffords_range=resolved_n_cliffords,
        seeds=resolved_seeds,
        n_shots=resolved_n_shots,
        shot_interval=resolved_shot_interval,
        time_integration=time_integration,
        enable_tqdm=enable_tqdm,
    )
    protocol_results, figures = _analyze_trials(
        trials,
        targets=resources.targets,
        n_cliffords=resolved_n_cliffords,
        plot=resolved_plot,
        save_image=resolved_save_image,
        xaxis_type=xaxis_type,
    )

    return Result(
        data={
            "n_cliffords": resolved_n_cliffords,
            "seeds": resolved_seeds,
            "protocols": protocol_results,
            "measurement_induced_control_error": _measurement_induced_error(
                protocol_results,
                target=control_qubit,
            ),
            "measurement_induced_ancilla_population_error": (
                _measurement_induced_error(
                    protocol_results,
                    target=ancilla_qubit,
                )
                if resolved_ancilla_mode == _RANDOMIZED_ANCILLA
                else None
            ),
            "metadata": {
                "control": control_qubit,
                "ancilla": ancilla_qubit,
                "protocols": resolved_protocols,
                "control_echo": control_echo,
                "ancilla_mode": resolved_ancilla_mode,
                "n_trials": resolved_n_trials,
                "n_shots": resolved_n_shots,
                "shot_interval": resolved_shot_interval,
                "time_integration": time_integration,
                "measurement_duration": resources.measurement.duration,
                "ancilla_x180_duration": (
                    resources.ancilla_x180.duration
                    if resources.ancilla_x180 is not None
                    else None
                ),
            },
        },
        figures=figures or None,
    )


__all__ = [
    "MCMRBProtocol",
    "mcm_randomized_benchmarking",
    "mcm_rb_sequence",
]
