"""Single- and multi-qubit MCM benchmarking and repetition experiments."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
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
from qubex.pulse import (
    Blank,
    FlatTop,
    PulseArray,
    PulseSchedule,
    VirtualZ,
    Waveform,
    set_sampling_period,
)

MCMRBProtocol = Literal["mcm-rb", "delay-rb", "mcm-rep", "delay-rep"]
_AncillaMode = Literal["standard", "randomized"]

_MCM_RB: MCMRBProtocol = "mcm-rb"
_DELAY_RB: MCMRBProtocol = "delay-rb"
_MCM_REP: MCMRBProtocol = "mcm-rep"
_DELAY_REP: MCMRBProtocol = "delay-rep"
# The suffix selects the control operation (Clifford or matched delay), while
# the prefix selects the ancilla operation (readout or matched delay).
_SUPPORTED_PROTOCOLS: tuple[MCMRBProtocol, ...] = (
    _MCM_RB,
    _DELAY_RB,
    _MCM_REP,
    _DELAY_REP,
)
_STANDARD_DEFAULT_PROTOCOLS: tuple[MCMRBProtocol, ...] = (
    _MCM_RB,
    _DELAY_RB,
    _MCM_REP,
)
_RANDOMIZED_DEFAULT_PROTOCOLS: tuple[MCMRBProtocol, ...] = (_MCM_RB, _DELAY_RB)
_CONTROL_DELAY_PROTOCOLS: tuple[MCMRBProtocol, ...] = (_MCM_REP, _DELAY_REP)
_READOUT_DELAY_PROTOCOLS: tuple[MCMRBProtocol, ...] = (_DELAY_RB, _DELAY_REP)
_STANDARD_ANCILLA: _AncillaMode = "standard"
_RANDOMIZED_ANCILLA: _AncillaMode = "randomized"
_CONTROL_RANDOMIZATION_SPAWN_KEY = (0,)
_ANCILLA_RANDOMIZATION_SPAWN_KEY = (1,)

_WaveformOverride = Waveform | Mapping[str, Waveform] | None


@dataclass(frozen=True)
class _SequenceResources:
    """Resolved labels and waveforms shared by all protocol schedules."""

    controls: tuple[str, ...]
    ancillas: tuple[str, ...]
    readout_labels: Mapping[str, str]
    sampling_period: float
    x90s: Mapping[str, Waveform]
    measurements: Mapping[str, Waveform]
    controls_during_measurement: Mapping[str, Waveform]
    ancilla_x180s: Mapping[str, Waveform]

    @property
    def targets(self) -> tuple[str, ...]:
        """Return all control and ancilla labels in role order."""
        return (*self.controls, *self.ancillas)


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

    @property
    def ramp_trimmed_interval_half_samples(self) -> tuple[int, int]:
        """Return interval bounds in half-sample units."""
        return (
            2 * self.active_start_samples + self.ramp_length_samples,
            2 * (self.active_start_samples + self.active_length_samples)
            - self.ramp_length_samples,
        )


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


class _ErrorEstimate(TypedDict):
    """Value and propagated uncertainty for one induced-error estimate."""

    value: float
    error: float


_TrialData = dict[MCMRBProtocol, dict[str, NDArray[np.float64]]]
_ProtocolResults = dict[MCMRBProtocol, dict[str, _TargetAnalysis]]
_CompiledCliffordSequences = Mapping[str, _CompiledCliffordSequence]
_CompiledAncillaSequences = Mapping[str, _CompiledAncillaSequence]
_TargetErrorPayload = _ErrorEstimate | dict[str, _ErrorEstimate | None] | None


def _validate_protocol(protocol: str) -> MCMRBProtocol:
    """Validate and narrow one protocol name."""
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"Invalid `protocol`: {protocol!r}. Expected one of {_SUPPORTED_PROTOCOLS}."
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


def _validate_xaxis_type(xaxis_type: str) -> Literal["linear", "log"]:
    """Validate and narrow the fit-figure x-axis scale."""
    if xaxis_type not in ("linear", "log"):
        raise ValueError("`xaxis_type` must be either `linear` or `log`.")
    return cast(Literal["linear", "log"], xaxis_type)


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


def _find_duplicates(values: Collection[str]) -> tuple[str, ...]:
    """Return duplicate labels in first repeated-occurrence order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _resolve_target_role(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    name: str,
) -> tuple[str, ...]:
    """Resolve one nonempty, duplicate-free target role."""
    labels = [targets] if isinstance(targets, str) else list(targets)
    if not labels:
        raise ValueError(f"`{name}` must contain at least one qubit.")
    if any(not isinstance(label, str) for label in labels):
        raise TypeError(f"`{name}` must contain only qubit-label strings.")

    resolved = tuple(exp.ctx.resolve_qubit_label(label) for label in labels)
    duplicates = _find_duplicates(resolved)
    if duplicates:
        raise ValueError(
            f"`{name}` contains duplicate resolved qubit(s): {duplicates}."
        )
    return resolved


def _resolve_target_groups(
    exp: Experiment,
    control: Collection[str] | str,
    ancilla: Collection[str] | str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve and validate disjoint control and ancilla target groups."""
    controls = _resolve_target_role(exp, control, name="control")
    ancillas = _resolve_target_role(exp, ancilla, name="ancilla")
    ancilla_set = set(ancillas)
    overlap = tuple(target for target in controls if target in ancilla_set)
    if overlap:
        raise ValueError(
            "`control` and `ancilla` must resolve to disjoint qubit groups; "
            f"overlap: {overlap}."
        )

    all_targets = (*controls, *ancillas)
    readout_labels = tuple(
        exp.experiment_system.resolve_read_label(target) for target in all_targets
    )
    duplicate_readouts = _find_duplicates(readout_labels)
    if duplicate_readouts:
        raise ValueError(
            "Selected qubits must resolve to distinct readout labels; "
            f"duplicates: {duplicate_readouts}."
        )
    collisions = set(all_targets) & set(readout_labels)
    if collisions:
        raise ValueError(
            "Control, ancilla, and readout labels must be distinct; "
            f"collisions: {tuple(sorted(collisions))}."
        )
    return controls, ancillas


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


def _resolve_waveforms(
    exp: Experiment,
    *,
    targets: tuple[str, ...],
    override: _WaveformOverride,
    default_factory: Callable[[str], Waveform],
    sampling_period: float,
    name: str,
) -> dict[str, Waveform]:
    """Resolve a scalar or target-keyed waveform override for one role."""
    overrides = _normalize_waveform_overrides(
        exp,
        targets=targets,
        override=override,
        name=name,
    )
    resolved: dict[str, Waveform] = {}
    for target in targets:
        waveform = overrides.get(target)
        if waveform is None:
            waveform = default_factory(target)
        _validate_waveform(
            waveform,
            expected_sampling_period=sampling_period,
            name=f"{name}[{target}]",
        )
        resolved[target] = waveform
    return resolved


def _normalize_waveform_overrides(
    exp: Experiment,
    *,
    targets: tuple[str, ...],
    override: _WaveformOverride,
    name: str,
) -> dict[str, Waveform]:
    """Normalize waveform overrides to resolved target labels."""
    if override is None:
        return {}
    if isinstance(override, Waveform):
        if len(targets) != 1:
            raise ValueError(
                f"`{name}` must be a target-keyed mapping when multiple "
                "qubits are selected."
            )
        return {targets[0]: override}
    if not isinstance(override, Mapping):
        raise TypeError(f"`{name}` must be a waveform or target-keyed mapping.")

    overrides: dict[str, Waveform] = {}
    for label, waveform in override.items():
        if not isinstance(label, str):
            raise TypeError(f"`{name}` mapping keys must be qubit-label strings.")
        if not isinstance(waveform, Waveform):
            raise TypeError(f"`{name}` mapping values must be waveforms.")
        target = exp.ctx.resolve_qubit_label(label)
        if target not in targets:
            raise ValueError(
                f"`{name}` contains an override for unselected target {target!r}."
            )
        if target in overrides:
            raise ValueError(
                f"`{name}` contains duplicate overrides for target {target!r}."
            )
        overrides[target] = waveform
    return overrides


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
            "Readout timing inference requires a waveform with at most one "
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
    slot_duration: float,
    sampling_period: float,
    echo_x180: Waveform | None,
) -> Waveform:
    """Build an idle or active-readout-aligned X-X echo block."""
    if echo_x180 is None:
        return _blank(slot_duration, sampling_period)

    slot_length = round(slot_duration / sampling_period)
    if slot_length < 2 * echo_x180.length:
        raise ValueError(
            "The measurement slot duration must be at least twice the echo "
            "X180 pulse duration."
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
    trailing_quarters = 4 * slot_length - second_center_quarters - 2 * echo_x180.length
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


def _validate_matching_readout_intervals(
    measurements: Mapping[str, Waveform],
) -> None:
    """Require equal ramp-trimmed active intervals for multiple readouts."""
    if len(measurements) == 1:
        return
    intervals_half_samples = {
        target: _infer_readout_timing(measurement).ramp_trimmed_interval_half_samples
        for target, measurement in measurements.items()
    }
    if len(set(intervals_half_samples.values())) != 1:
        sampling_period = next(iter(measurements.values())).sampling_period
        intervals_ns = {
            target: tuple(
                half_samples * sampling_period / 2.0 for half_samples in interval
            )
            for target, interval in intervals_half_samples.items()
        }
        raise ValueError(
            "All ancilla ramp-trimmed active intervals must match; "
            f"got {intervals_ns} ns."
        )


def _resolve_sequence_resources(
    exp: Experiment,
    *,
    controls: tuple[str, ...],
    ancillas: tuple[str, ...],
    control_echo: bool,
    ancilla_mode: _AncillaMode,
    x90: _WaveformOverride,
    measurement_waveform: _WaveformOverride,
    echo_x180: _WaveformOverride,
    ancilla_x180: _WaveformOverride,
) -> _SequenceResources:
    """Resolve and validate labels and waveforms used by every protocol."""
    sampling_period = float(exp.ctx.measurement.sampling_period)
    if not np.isfinite(sampling_period) or sampling_period <= 0:
        raise ValueError("The experiment sampling period must be positive and finite.")
    set_sampling_period(sampling_period)

    x90s = _resolve_waveforms(
        exp,
        targets=controls,
        override=x90,
        default_factory=exp.pulse.x90,
        sampling_period=sampling_period,
        name="x90",
    )
    measurements = _resolve_waveforms(
        exp,
        targets=ancillas,
        override=measurement_waveform,
        default_factory=exp.pulse.readout,
        sampling_period=sampling_period,
        name="measurement_waveform",
    )
    _validate_matching_readout_intervals(measurements)
    reference_measurement = next(iter(measurements.values()))
    measurement_slot_duration = max(
        measurement.duration for measurement in measurements.values()
    )

    echo_x180s: dict[str, Waveform] = {}
    if control_echo:
        echo_x180s = _resolve_waveforms(
            exp,
            targets=controls,
            override=echo_x180,
            default_factory=exp.pulse.x180,
            sampling_period=sampling_period,
            name="echo_x180",
        )

    ancilla_x180s: dict[str, Waveform] = {}
    if ancilla_mode == _RANDOMIZED_ANCILLA:
        ancilla_x180s = _resolve_waveforms(
            exp,
            targets=ancillas,
            override=ancilla_x180,
            default_factory=exp.pulse.x180,
            sampling_period=sampling_period,
            name="ancilla_x180",
        )

    return _SequenceResources(
        controls=controls,
        ancillas=ancillas,
        readout_labels={
            target: exp.experiment_system.resolve_read_label(target)
            for target in ancillas
        },
        sampling_period=sampling_period,
        x90s=x90s,
        measurements=measurements,
        controls_during_measurement={
            target: _measurement_control_block(
                measurement=reference_measurement,
                slot_duration=measurement_slot_duration,
                sampling_period=sampling_period,
                echo_x180=echo_x180s.get(target),
            )
            for target in controls
        },
        ancilla_x180s=ancilla_x180s,
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


def _derive_seed(seed: int | None, *, spawn_key: tuple[int, ...]) -> int:
    """Derive one uint32 child seed from a trial seed and stream key."""
    seed_sequence = np.random.SeedSequence(seed, spawn_key=spawn_key)
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _compile_clifford_sequences(
    exp: Experiment,
    resources: _SequenceResources,
    *,
    n_cliffords: int,
    seed: int | None,
) -> dict[str, _CompiledCliffordSequence]:
    """Compile an independent randomized Clifford stream for every control."""
    compiled: dict[str, _CompiledCliffordSequence] = {}
    for index, control in enumerate(resources.controls):
        control_seed = (
            seed
            if len(resources.controls) == 1
            else _derive_seed(
                seed,
                spawn_key=(*_CONTROL_RANDOMIZATION_SPAWN_KEY, index),
            )
        )
        compiled[control] = _generate_clifford_sequence(
            exp,
            n_cliffords=n_cliffords,
            seed=control_seed,
            x90=resources.x90s[control],
        )
    return compiled


def _compile_ancilla_sequences(
    resources: _SequenceResources,
    *,
    n_cliffords: int,
    seed: int | None,
) -> dict[str, _CompiledAncillaSequence]:
    """Compile an independent reproducible I/X stream for every ancilla."""
    if not resources.ancilla_x180s:
        return {}

    compiled: dict[str, _CompiledAncillaSequence] = {}
    for index, ancilla in enumerate(resources.ancillas):
        x180 = resources.ancilla_x180s[ancilla]
        spawn_key = (
            _ANCILLA_RANDOMIZATION_SPAWN_KEY
            if len(resources.ancillas) == 1
            else (*_ANCILLA_RANDOMIZATION_SPAWN_KEY, index)
        )
        values = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=spawn_key)
        ).integers(
            0,
            2,
            size=n_cliffords,
            dtype=np.int8,
        )
        identity = _blank(x180.duration, resources.sampling_period)
        operations = tuple(x180 if value else identity for value in values)
        recovery = x180 if np.count_nonzero(values) % 2 else identity
        compiled[ancilla] = _CompiledAncillaSequence(
            operations=operations,
            recovery=recovery,
        )
    return compiled


def _duration_matched_operation(
    operation: Waveform,
    *,
    replace_with_blank: bool,
    sampling_period: float,
) -> Waveform:
    """Return a waveform or a duration-matched blank."""
    if not replace_with_blank:
        return operation
    return _blank(operation.duration, sampling_period)


def _add_parallel_layer(
    schedule: PulseSchedule,
    operations: Mapping[str, Waveform],
) -> None:
    """Add simultaneous channel operations and synchronize their endpoints."""
    for label, operation in operations.items():
        schedule.add(label, operation)
    schedule.barrier()


def _validate_compiled_sequence_lengths(
    resources: _SequenceResources,
    clifford_sequences: _CompiledCliffordSequences,
    ancilla_sequences: _CompiledAncillaSequences,
) -> int:
    """Return the common cycle count after validating compiled sequences."""
    n_cycles = len(clifford_sequences[resources.controls[0]].cliffords)
    if any(
        len(sequence.cliffords) != n_cycles for sequence in clifford_sequences.values()
    ):
        raise RuntimeError("All control Clifford sequences must have equal length.")
    if any(
        len(sequence.operations) != n_cycles for sequence in ancilla_sequences.values()
    ):
        raise RuntimeError(
            "Every ancilla sequence length must match the Clifford sequence."
        )
    return n_cycles


def _build_protocol_schedule(
    resources: _SequenceResources,
    clifford_sequences: _CompiledCliffordSequences,
    *,
    protocol: MCMRBProtocol,
    ancilla_sequences: _CompiledAncillaSequences,
) -> PulseSchedule:
    """Build one protocol schedule from shared resources and Cliffords."""
    schedule_labels = [*resources.controls]
    if ancilla_sequences:
        schedule_labels.extend(resources.ancillas)
    schedule_labels.extend(
        resources.readout_labels[ancilla] for ancilla in resources.ancillas
    )

    n_cycles = _validate_compiled_sequence_lengths(
        resources,
        clifford_sequences,
        ancilla_sequences,
    )
    replace_control_with_blank = protocol in _CONTROL_DELAY_PROTOCOLS
    replace_readout_with_blank = protocol in _READOUT_DELAY_PROTOCOLS

    with PulseSchedule(schedule_labels) as schedule:
        for cycle_index in range(n_cycles):
            _add_parallel_layer(
                schedule,
                {
                    control: _duration_matched_operation(
                        clifford_sequences[control].cliffords[cycle_index],
                        replace_with_blank=replace_control_with_blank,
                        sampling_period=resources.sampling_period,
                    )
                    for control in resources.controls
                },
            )

            if ancilla_sequences:
                _add_parallel_layer(
                    schedule,
                    {
                        ancilla: ancilla_sequences[ancilla].operations[cycle_index]
                        for ancilla in resources.ancillas
                    },
                )

            measurement_layer = {
                resources.readout_labels[ancilla]: _duration_matched_operation(
                    resources.measurements[ancilla],
                    replace_with_blank=replace_readout_with_blank,
                    sampling_period=resources.sampling_period,
                )
                for ancilla in resources.ancillas
            }
            measurement_layer.update(resources.controls_during_measurement)
            _add_parallel_layer(schedule, measurement_layer)

        if ancilla_sequences:
            _add_parallel_layer(
                schedule,
                {
                    ancilla: ancilla_sequences[ancilla].recovery
                    for ancilla in resources.ancillas
                },
            )

        _add_parallel_layer(
            schedule,
            {
                control: _duration_matched_operation(
                    clifford_sequences[control].inverse,
                    replace_with_blank=replace_control_with_blank,
                    sampling_period=resources.sampling_period,
                )
                for control in resources.controls
            },
        )

    return schedule


def mcm_rb_sequence(
    exp: Experiment,
    control: Collection[str] | str,
    ancilla: Collection[str] | str,
    *,
    protocol: MCMRBProtocol,
    n_cliffords: int,
    seed: int | None = None,
    control_echo: bool = False,
    ancilla_mode: Literal["standard", "randomized"] = "standard",
    x90: Waveform | Mapping[str, Waveform] | None = None,
    measurement_waveform: Waveform | Mapping[str, Waveform] | None = None,
    echo_x180: Waveform | Mapping[str, Waveform] | None = None,
    ancilla_x180: Waveform | Mapping[str, Waveform] | None = None,
) -> PulseSchedule:
    """
    Build one MCM-RB, delay-RB, MCM-repetition, or delay-repetition schedule.

    Parameters
    ----------
    exp
        Experiment instance that provides pulse and Clifford services.
    control
        One control/spectator label or an ordered collection of labels. RB
        protocols apply independently randomized Cliffords and inverses to
        the controls; repetition protocols replace them with duration-matched
        delays.
    ancilla
        One ancilla/readout-target label or an ordered collection of labels.
        MCM protocols read all ancillas simultaneously; delay protocols
        replace every readout with a duration-matched delay.
    protocol
        Sequence variant: `"mcm-rb"`, `"delay-rb"`, `"mcm-rep"`, or
        `"delay-rep"`. Repetition protocols replace the control Cliffords with
        duration-matched delays. Delay protocols replace the intermediate
        readout pulses with duration-matched delays.
    n_cliffords
        Number of protocol cycles. Must be nonnegative.
    seed
        Nonnegative root seed. Multiple controls receive independent derived
        Clifford streams, and multiple randomized ancillas receive independent
        derived I/X180 streams. Repetition protocols use the generated
        Clifford durations only.
    control_echo
        Whether to apply an X-X echo to every control during every measurement
        or reference-delay window. The X180 centers are placed at the quarter
        and three-quarter points of the common ramp-trimmed active readout
        interval. Repetition protocols therefore contain control pulses during
        these windows when this option is `True`.
    ancilla_mode
        Ancilla sequence mode. `"standard"` preserves the original protocol.
        `"randomized"` inserts a seeded I/X180 before every measurement or
        reference delay and applies a final parity recovery.
    x90
        Optional control X90 override. Pass a waveform for one control or a
        target-keyed mapping for multiple controls. Omitted mapping entries use
        calibrated defaults.
    measurement_waveform
        Optional ancilla readout override. Pass a waveform for one ancilla or a
        target-keyed mapping for multiple ancillas. Omitted mapping entries use
        calibrated defaults. Multiple ancillas must have identical
        ramp-trimmed active intervals, although their total slot durations may
        differ. Active intervals are inferred from nonzero samples; at most one
        FlatTop pulse may be present in each waveform, and other shapes are
        treated as having zero ramp duration.
    echo_x180
        Optional control X180 override used by the X-X echo. Pass a waveform
        for one control or a target-keyed mapping for multiple controls.
        Omitted mapping entries use calibrated defaults. Ignored when
        `control_echo=False`.
    ancilla_x180
        Optional ancilla X180 override used for randomized I/X180 operations
        and parity recovery. Pass a waveform for one ancilla or a target-keyed
        mapping for multiple ancillas. Omitted mapping entries use calibrated
        defaults. Ignored in standard mode.

    Returns
    -------
    PulseSchedule
        Schedule before the terminal measurements are appended by execution.

    Raises
    ------
    TypeError
        Raised when a target, Clifford count, seed, or waveform override has
        an invalid type.
    ValueError
        Raised for invalid or overlapping target groups, protocol or ancilla
        mode names, waveform mappings, mismatched ancilla active intervals, or
        echo timing.

    Notes
    -----
    The `mcm-rep` and `delay-rep` protocols replace every randomized Clifford
    and the final inverse with duration-matched delays. All four protocols have
    equal total duration when built with the same cycle count, seed, ancilla
    mode, and echo options.

    In randomized mode, the final recovery on each ancilla is X180 exactly
    when that ancilla's programmed sequence has odd parity. One static I/X180
    sequence per ancilla is encoded in the returned schedule. Independent
    random streams are assigned in resolved input order.

    When multiple ancillas are selected, every control is exposed to their
    simultaneous combined readout. The schedule does not attribute a measured
    control error to an individual ancilla.

    Construction synchronizes the global pulse-library sampling period with
    `exp.ctx.measurement.sampling_period`.
    """
    resolved_protocol = _validate_protocol(protocol)
    resolved_ancilla_mode = _validate_ancilla_mode(ancilla_mode)
    resolved_n_cliffords = _validate_nonnegative_integer(
        n_cliffords,
        name="n_cliffords",
    )
    resolved_seed = (
        None if seed is None else _validate_nonnegative_integer(seed, name="seed")
    )
    control_qubits, ancilla_qubits = _resolve_target_groups(exp, control, ancilla)
    resources = _resolve_sequence_resources(
        exp,
        controls=control_qubits,
        ancillas=ancilla_qubits,
        control_echo=control_echo,
        ancilla_mode=resolved_ancilla_mode,
        x90=x90,
        measurement_waveform=measurement_waveform,
        echo_x180=echo_x180,
        ancilla_x180=ancilla_x180,
    )
    clifford_sequences = _compile_clifford_sequences(
        exp,
        resources,
        n_cliffords=resolved_n_cliffords,
        seed=resolved_seed,
    )
    ancilla_sequences = _compile_ancilla_sequences(
        resources,
        n_cliffords=resolved_n_cliffords,
        seed=resolved_seed,
    )
    return _build_protocol_schedule(
        resources,
        clifford_sequences,
        protocol=resolved_protocol,
        ancilla_sequences=ancilla_sequences,
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
    """Return unique protocol names in input iteration order."""
    if protocols is None:
        if ancilla_mode == _RANDOMIZED_ANCILLA:
            return _RANDOMIZED_DEFAULT_PROTOCOLS
        return _STANDARD_DEFAULT_PROTOCOLS
    candidates = [protocols] if isinstance(protocols, str) else list(protocols)
    if not candidates:
        raise ValueError("`protocols` must not be empty.")
    resolved: tuple[MCMRBProtocol, ...] = tuple(
        _validate_protocol(protocol) for protocol in candidates
    )
    if len(resolved) != len(set(resolved)):
        raise ValueError("`protocols` must not contain duplicate values.")
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
                clifford_sequences = _compile_clifford_sequences(
                    exp,
                    resources,
                    n_cliffords=int(n_cliffords),
                    seed=int(seed),
                )
                ancilla_sequences = _compile_ancilla_sequences(
                    resources,
                    n_cliffords=int(n_cliffords),
                    seed=int(seed),
                )
                for protocol in protocols:
                    sequence = _build_protocol_schedule(
                        resources,
                        clifford_sequences,
                        protocol=protocol,
                        ancilla_sequences=ancilla_sequences,
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
        title=f"{protocol} measurement-crosstalk benchmarking",
        xlabel="Number of protocol cycles",
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
    targets: tuple[str, ...],
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
    measurement_protocol: MCMRBProtocol,
    reference_protocol: MCMRBProtocol,
) -> _ErrorEstimate | None:
    """Calculate one target's induced error from a matched protocol pair."""
    measurement_result = protocol_results[measurement_protocol][target]
    reference_result = protocol_results[reference_protocol][target]
    p_measurement = measurement_result["decay_parameter"]
    p_measurement_err = measurement_result["decay_parameter_err"]
    p_reference = reference_result["decay_parameter"]
    p_reference_err = reference_result["decay_parameter_err"]
    if (
        p_measurement is None
        or p_measurement_err is None
        or p_reference is None
        or p_reference_err is None
        or p_reference == 0.0
    ):
        return None

    value = 0.5 * (1.0 - p_measurement / p_reference)
    error = 0.5 * np.sqrt(
        (p_measurement_err / p_reference) ** 2
        + (p_measurement * p_reference_err / p_reference**2) ** 2
    )
    return {"value": float(value), "error": float(error)}


def _measurement_induced_errors(
    protocol_results: _ProtocolResults,
    *,
    targets: tuple[str, ...],
    measurement_protocol: MCMRBProtocol,
    reference_protocol: MCMRBProtocol,
) -> _TargetErrorPayload:
    """Return one estimate directly or multiple estimates keyed by target."""
    if (
        measurement_protocol not in protocol_results
        or reference_protocol not in protocol_results
    ):
        return None
    estimates = {
        target: _measurement_induced_error(
            protocol_results,
            target=target,
            measurement_protocol=measurement_protocol,
            reference_protocol=reference_protocol,
        )
        for target in targets
    }
    if len(targets) == 1:
        return estimates[targets[0]]
    return estimates


def _waveform_duration_payload(
    waveforms: Mapping[str, Waveform],
    *,
    targets: tuple[str, ...],
) -> float | dict[str, float] | None:
    """Return scalar or target-keyed waveform durations for result metadata."""
    if not waveforms:
        return None
    durations = {target: waveforms[target].duration for target in targets}
    if len(targets) == 1:
        return durations[targets[0]]
    return durations


def mcm_randomized_benchmarking(
    exp: Experiment,
    control: Collection[str] | str,
    ancilla: Collection[str] | str,
    *,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    protocols: Collection[MCMRBProtocol] | MCMRBProtocol | None = None,
    control_echo: bool = False,
    ancilla_mode: Literal["standard", "randomized"] = "standard",
    x90: Waveform | Mapping[str, Waveform] | None = None,
    measurement_waveform: Waveform | Mapping[str, Waveform] | None = None,
    echo_x180: Waveform | Mapping[str, Waveform] | None = None,
    ancilla_x180: Waveform | Mapping[str, Waveform] | None = None,
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
        One control/spectator label or an ordered collection of labels. RB
        protocols apply independently randomized Cliffords to all controls;
        repetition protocols apply duration-matched delays instead.
    ancilla
        One ancilla/readout-target label or an ordered collection of labels.
        MCM protocols read all ancillas simultaneously; delay protocols
        replace every readout with a duration-matched delay.
    n_cliffords_range
        Strictly increasing nonnegative cycle counts. Defaults to 0 followed
        by 14 geometrically spaced values from 1 through 150.
    n_trials
        Number of seed trials per cycle count. Defaults to 30.
    seeds
        One nonnegative integer seed for every trial. Defaults to generated
        seeds. Values must have integer types and fit in signed 64 bits.
    protocols
        Protocols to run. Defaults to all of `"mcm-rb"`, `"delay-rb"`, and
        `"mcm-rep"` in standard mode and to the matched `"mcm-rb"` and
        `"delay-rb"` pair in randomized mode. `"mcm-rep"` and `"delay-rep"`
        are opt-in in randomized mode; `"delay-rep"` is opt-in in both modes.
        The same per-target Clifford and ancilla I/X180 sequences are shared
        across protocols for each sequence length and root seed.
    control_echo
        Whether to insert an X-X echo on every control during each measurement
        or duration-matched delay window. The X180 centers are placed at the
        quarter and three-quarter points of the common ramp-trimmed active
        readout interval. Repetition protocols omit randomized Cliffords but
        are not strictly idle when echo is enabled.
    ancilla_mode
        Ancilla sequence mode. `"standard"` preserves the original suite.
        `"randomized"` inserts a seeded I/X180 before every measurement or
        reference delay, followed by an X180 for odd parity or a
        duration-matched blank for even parity.
    x90
        Optional control X90 override. Use a waveform for one control or a
        target-keyed mapping for multiple controls. Omitted mapping entries use
        calibrated defaults.
    measurement_waveform
        Optional ancilla readout override. Use a waveform for one ancilla or a
        target-keyed mapping for multiple ancillas. Multiple ancillas must have
        identical ramp-trimmed active intervals, although their total slot
        durations may differ. Omitted mapping entries use calibrated defaults.
        Active intervals are inferred from nonzero samples; at most one FlatTop
        pulse may be present in each waveform, and other shapes have zero
        inferred ramp duration.
    echo_x180
        Optional control X180 override for echoed measurement blocks. Use a
        waveform for one control or a target-keyed mapping for multiple
        controls. Omitted mapping entries use calibrated defaults. Ignored when
        `control_echo=False`.
    ancilla_x180
        Optional ancilla X180 override used by randomized mode. Use a waveform
        for one ancilla or a target-keyed mapping for multiple ancillas. Omitted
        mapping entries use calibrated defaults. Ignored in standard mode.
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
        optional MCM-induced control error, randomized-ancilla population
        errors with Clifford-driven or delayed control, metadata, and named
        fit figures.
        Protocol results are stored under
        `result.data["protocols"][protocol][target]`. The Clifford-driven and
        control-delay ancilla estimates are stored as
        `measurement_induced_ancilla_population_error` and
        `measurement_induced_ancilla_population_error_with_idle_control`.
        Each induced-error field contains one estimate for a single target, a
        target-keyed mapping when that role contains multiple targets, or
        `None` when its matched protocol pair was not run.

    Raises
    ------
    TypeError
        Raised when a target, sweep value, seed, count, timing value, or
        waveform override has an invalid type.
    ValueError
        Raised when target groups, sweep values, seeds, protocols, ancilla
        mode, waveform mappings, active readout intervals, or pulse timing are
        invalid.

    Notes
    -----
    Intermediate MCM outcomes are intentionally discarded. Only terminal
    captures are normalized and fitted independently for each control and
    ancilla. A matched measurement/reference pair reports
    `(1 - p_measurement / p_reference) / 2`.
    MCM-RB and delay-RB form the Clifford-driven pair. In randomized mode,
    MCM-repetition and delay-repetition form the control-delay pair.
    MCM-repetition without delay-repetition does not produce a ratio-based
    induced-error estimate. With `control_echo=True`, the control-delay pair
    still contains the requested X-X echoes.

    Control-qubit MCM-RB and delay-RB have the standard
    randomized-benchmarking exponential form under the usual assumptions.
    Every protocol/target pair is fitted to the same exponential form for
    comparison. The repetition-protocol fit is not a standard RB fit: the
    control is not Clifford twirled, and standard-mode ancilla population is
    not I/X180 twirled. Its `error_per_cycle` is therefore a decay metric rather
    than a general average gate error.

    Randomized mode benchmarks ancilla computational-basis population
    preservation, not general single-qubit or state-assignment fidelity. Its
    induced ancilla error estimate assumes the randomized I/X180 and
    measurement channels produce compatible exponential decays. The
    uncertainty of an induced-error estimate treats the two fitted decay
    parameters as independent even though matched random sequences can
    correlate them.

    Independent control and ancilla random streams are assigned in resolved
    input order. Ancilla I/X180 choices are generated once per sequence length
    and trial root seed, then reused for every shot of that schedule.
    Randomization is averaged across trials rather than varied shot by shot.

    With multiple ancillas, the reported control error characterizes their
    simultaneous combined readout and is not an attribution to any individual
    ancilla. Per-target fits are marginal analyses of averaged terminal IQ;
    they do not quantify correlated multi-qubit errors.

    The experiment synchronizes the global pulse-library sampling period with
    `exp.ctx.measurement.sampling_period` before constructing schedules.
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
    resolved_xaxis_type = _validate_xaxis_type(xaxis_type)

    control_qubits, ancilla_qubits = _resolve_target_groups(exp, control, ancilla)
    exp.pulse.validate_rabi_params([*control_qubits, *ancilla_qubits])
    resources = _resolve_sequence_resources(
        exp,
        controls=control_qubits,
        ancillas=ancilla_qubits,
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
        xaxis_type=resolved_xaxis_type,
    )

    return Result(
        data={
            "n_cliffords": resolved_n_cliffords,
            "seeds": resolved_seeds,
            "protocols": protocol_results,
            "measurement_induced_control_error": _measurement_induced_errors(
                protocol_results,
                targets=control_qubits,
                measurement_protocol=_MCM_RB,
                reference_protocol=_DELAY_RB,
            ),
            "measurement_induced_ancilla_population_error": (
                _measurement_induced_errors(
                    protocol_results,
                    targets=ancilla_qubits,
                    measurement_protocol=_MCM_RB,
                    reference_protocol=_DELAY_RB,
                )
                if resolved_ancilla_mode == _RANDOMIZED_ANCILLA
                else None
            ),
            "measurement_induced_ancilla_population_error_with_idle_control": (
                _measurement_induced_errors(
                    protocol_results,
                    targets=ancilla_qubits,
                    measurement_protocol=_MCM_REP,
                    reference_protocol=_DELAY_REP,
                )
                if resolved_ancilla_mode == _RANDOMIZED_ANCILLA
                else None
            ),
            "metadata": {
                "control": control_qubits[0] if len(control_qubits) == 1 else None,
                "ancilla": ancilla_qubits[0] if len(ancilla_qubits) == 1 else None,
                "controls": control_qubits,
                "ancillas": ancilla_qubits,
                "protocols": resolved_protocols,
                "control_echo": control_echo,
                "ancilla_mode": resolved_ancilla_mode,
                "n_trials": resolved_n_trials,
                "n_shots": resolved_n_shots,
                "shot_interval": resolved_shot_interval,
                "time_integration": time_integration,
                "measurement_duration": _waveform_duration_payload(
                    resources.measurements,
                    targets=ancilla_qubits,
                ),
                "ancilla_x180_duration": _waveform_duration_payload(
                    resources.ancilla_x180s,
                    targets=ancilla_qubits,
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
