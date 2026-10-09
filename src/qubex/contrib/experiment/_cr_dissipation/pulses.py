"""
Build hardware schedules and matching semantic CR protocol descriptions.

Actual schedules preserve calibrated waveforms and timing. A/B rebuild their
same-sign lobes with the calibrated cancellation tone and zero rotary, while
C/D retain the complete echoed ZX90. Semantic operations mirror that distinction
without interpreting raw waveform samples as an exact Hamiltonian.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Literal

import numpy as np

from qubex.pulse import Blank, PulseSchedule, Waveform

from .simulation import (
    X_CONTROL,
    X_TARGET,
    Y_CONTROL,
    Y_TARGET,
    Z_TARGET,
    ZX,
    SemanticOperation,
    SemanticSegment,
    SemanticUnitary,
    rotation_hamiltonian,
    rotation_unitary,
    simultaneous_rotation_segments,
)

PROTOCOL_A = "control_ground_cr_population"
PROTOCOL_B = "control_excited_cr_population"
PROTOCOL_C = "control_cr_transverse_echo"
PROTOCOL_D = "target_cr_rotating_frame_echo"
PROTOCOLS = (PROTOCOL_A, PROTOCOL_B, PROTOCOL_C, PROTOCOL_D)


# Protocol identifiers and resolved calibrated-gate metadata


@dataclass(frozen=True)
class ZX90Descriptor:
    """
    Store the actual echoed gate and reconstructable semantic metadata.

    The calibrated echoed schedule is preserved for C/D. When pre-combination
    rotary metadata are unavailable, the semantic model omits the rotary term
    under the echoed-gate approximation that its coherent IX rotation cancels.
    """

    echoed: PulseSchedule
    full_un_echoed: PulseSchedule
    cr_lobe_duration_ns: float
    echo_slot_duration_ns: float
    pi_pulse_duration_ns: float
    echo_margin_duration_ns: float
    total_duration_ns: float
    pi_pulse: Waveform
    rotary_integrated_angle_rad: float | None
    rotary_phase_rad: float | None

    @property
    def cr_active_duration_ns(self) -> float:
        """Return total CR-active time of one echoed ZX90."""
        return 2.0 * self.cr_lobe_duration_ns


@dataclass(frozen=True)
class ProtocolSchedules:
    """Store actual protocol schedules and the repeated semantic blocks."""

    actual: dict[str, tuple[PulseSchedule, ...]]
    base: dict[str, tuple[PulseSchedule, ...]]
    semantic_blocks: dict[str, tuple[SemanticOperation, ...]]
    elapsed_time_ns: dict[str, np.ndarray]
    cr_active_time_ns: dict[str, np.ndarray]
    timing: dict[str, dict[str, float | tuple[str, ...]]]


def _positive_duration(
    value: Any,
    name: str,
    *,
    source: str = "zx90_echo",
) -> float:
    """Validate one required positive duration from calibrated gate metadata."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(  # noqa: TRY004 - public override contract requires ValueError
            f"{source} must expose numeric `{name}` metadata."
        )
    duration = float(value)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{source} `{name}` must be positive and finite.")
    return duration


def _frequencies(
    schedule: PulseSchedule,
    *,
    source: str = "zx90_echo",
) -> Mapping[str, float | None]:
    """Return validated frequency metadata from a pulse schedule."""
    getter = getattr(schedule, "get_frequencies", None)
    if getter is None:
        raise ValueError(f"{source} must expose frequency metadata.")
    frequencies = getter()
    if not isinstance(frequencies, Mapping):
        raise ValueError(  # noqa: TRY004 - incompatible overrides are ValueError
            f"{source} frequency metadata is invalid."
        )
    return frequencies


def _blank_schedule(
    labels: Sequence[str],
    duration_ns: float,
    frequencies: Mapping[str, float | None],
) -> PulseSchedule:
    """Build a frequency-preserving blank schedule of the requested duration."""
    with PulseSchedule(list(labels)) as blank:
        blank.add(labels[0], Blank(duration=duration_ns))
    blank.set_frequencies(frequencies)
    return blank


def resolve_zx90_descriptor(
    exp: Any,
    control_qubit: str,
    target_qubit: str,
    zx90_echo: PulseSchedule | None,
    zx90_no_echo: PulseSchedule | None = None,
) -> ZX90Descriptor:
    """Resolve an echoed ZX90 and construct a rotary-free A/B unit."""
    echoed = (
        exp.pulse.zx90(control_qubit, target_qubit, echo=True)
        if zx90_echo is None
        else zx90_echo
    )
    if getattr(echoed, "echo", None) is not True:
        raise ValueError("zx90_echo must be an echoed calibrated ZX90 schedule.")
    cr_duration = _positive_duration(
        getattr(echoed, "cr_duration", None), "cr_duration"
    )
    total_duration = _positive_duration(getattr(echoed, "duration", None), "duration")
    pi_pulse = getattr(echoed, "pi_pulse", None)
    cr_waveform = getattr(echoed, "cr_waveform", None)
    if not isinstance(pi_pulse, Waveform):
        raise ValueError(  # noqa: TRY004 - incompatible overrides are ValueError
            "zx90_echo must expose its calibrated `pi_pulse` waveform."
        )
    if not isinstance(cr_waveform, Waveform):
        raise ValueError(  # noqa: TRY004 - incompatible overrides are ValueError
            "zx90_echo must expose its calibrated `cr_waveform`."
        )
    if not np.isclose(cr_waveform.duration, cr_duration):
        raise ValueError("ZX90 CR waveform duration does not match cr_duration.")

    slot_duration = (total_duration - 2.0 * cr_duration) / 2.0
    if slot_duration < pi_pulse.duration - 1e-9:
        raise ValueError(
            "ZX90 echo slot is shorter than the embedded control pi pulse."
        )
    if slot_duration < -1e-9:
        raise ValueError("ZX90 duration is inconsistent with its two CR lobes.")
    slot_duration = max(0.0, slot_duration)
    margin_duration = max(0.0, slot_duration - pi_pulse.duration)

    cr_label = f"{control_qubit}-{target_qubit}"
    labels = (control_qubit, cr_label, target_qubit)
    frequencies = _frequencies(echoed)
    if zx90_no_echo is None:
        rotary_free = exp.pulse.zx90(
            control_qubit,
            target_qubit,
            rotary_amplitude=0.0,
            echo=True,
        )
        if getattr(rotary_free, "echo", None) is not True:
            raise ValueError("The rotary-free A/B calibration must be an echoed ZX90.")
        rotary_free_duration = _positive_duration(
            getattr(rotary_free, "duration", None),
            "rotary-free duration",
        )
        rotary_free_cr = getattr(rotary_free, "cr_waveform", None)
        cancel_waveform = getattr(rotary_free, "cancel_waveform", None)
        if not isinstance(rotary_free_cr, Waveform) or not isinstance(
            cancel_waveform, Waveform
        ):
            raise ValueError(
                "A/B rotary removal requires calibrated CR and cancel waveforms."
            )
        if not np.isclose(rotary_free_duration, total_duration) or not np.isclose(
            float(rotary_free_cr.duration), cr_duration
        ):
            raise ValueError(
                "The rotary-free A/B ZX90 timing does not match the supplied ZX90."
            )
        if not np.allclose(rotary_free_cr.values, cr_waveform.values):
            raise ValueError(
                "Removing A/B rotary unexpectedly changed the calibrated CR waveform."
            )
        if not np.isclose(cancel_waveform.duration, cr_duration):
            raise ValueError("A/B cancel waveform duration does not match cr_duration.")
        with PulseSchedule(list(labels)) as lobe:
            lobe.add(cr_label, cr_waveform)
            lobe.add(target_qubit, cancel_waveform)
        lobe.set_frequencies(frequencies)
        blank = _blank_schedule(labels, slot_duration, frequencies)
        with PulseSchedule(list(labels)) as full_un_echoed:
            full_un_echoed.call(lobe, copy=True)
            full_un_echoed.call(blank, copy=True)
            full_un_echoed.call(lobe, copy=True)
            full_un_echoed.call(blank, copy=True)
        full_un_echoed.set_frequencies(frequencies)
        full_un_echoed.cr_duration = 2.0 * cr_duration  # type: ignore[attr-defined]
        full_un_echoed.echo = False  # type: ignore[attr-defined]
    else:
        if not isinstance(zx90_no_echo, PulseSchedule):
            raise TypeError("zx90_no_echo must be a PulseSchedule or None.")
        if not zx90_no_echo.is_valid():
            raise ValueError("zx90_no_echo must be a valid pulse schedule.")
        if getattr(zx90_no_echo, "echo", None) is not False:
            raise ValueError("zx90_no_echo must be explicitly marked un-echoed.")
        no_echo_cr_duration = _positive_duration(
            getattr(zx90_no_echo, "cr_duration", None),
            "cr_duration",
            source="zx90_no_echo",
        )
        no_echo_duration = _positive_duration(
            getattr(zx90_no_echo, "duration", None),
            "duration",
            source="zx90_no_echo",
        )
        if not np.isclose(no_echo_cr_duration, 2.0 * cr_duration):
            raise ValueError(
                "zx90_no_echo must contain the same total CR-active duration as "
                "zx90_echo. A single-lobe echo=False primitive is not sufficient."
            )
        if not np.isclose(no_echo_duration, total_duration):
            raise ValueError(
                "zx90_no_echo and zx90_echo must have the same total duration."
            )
        _frequencies(zx90_no_echo, source="zx90_no_echo")
        full_un_echoed = zx90_no_echo

    rotary_angle = getattr(echoed, "rotary_integrated_angle_rad", None)
    rotary_phase = getattr(echoed, "rotary_phase_rad", None)
    if rotary_angle is None or rotary_phase is None:
        rotary_angle = None
        rotary_phase = None
    else:
        rotary_angle = float(rotary_angle)
        rotary_phase = float(rotary_phase)
        if not np.isfinite(rotary_angle) or not np.isfinite(rotary_phase):
            raise ValueError("Rotary semantic metadata must be finite when provided.")

    return ZX90Descriptor(
        echoed=echoed,
        full_un_echoed=full_un_echoed,
        cr_lobe_duration_ns=cr_duration,
        echo_slot_duration_ns=slot_duration,
        pi_pulse_duration_ns=float(pi_pulse.duration),
        echo_margin_duration_ns=margin_duration,
        total_duration_ns=total_duration,
        pi_pulse=pi_pulse,
        rotary_integrated_angle_rad=rotary_angle,
        rotary_phase_rad=rotary_phase,
    )


# Semantic operations used by the forward model


def semantic_cr_lobe(
    descriptor: ZX90Descriptor,
    *,
    sign: Literal[-1, 1],
    include_rotary: bool = True,
) -> SemanticSegment:
    """Build one intended ZX45 lobe with optional semantic rotary metadata."""
    duration = descriptor.cr_lobe_duration_ns
    hamiltonian = rotation_hamiltonian(ZX, sign * np.pi / 4.0, duration)
    if include_rotary and descriptor.rotary_integrated_angle_rad is not None:
        if descriptor.rotary_phase_rad is None:
            raise ValueError("Rotary angle metadata requires rotary phase metadata.")
        axis = (
            np.cos(descriptor.rotary_phase_rad) * X_TARGET
            + np.sin(descriptor.rotary_phase_rad) * Y_TARGET
        )
        hamiltonian += rotation_hamiltonian(
            axis,
            sign * descriptor.rotary_integrated_angle_rad,
            duration,
        )
    return SemanticSegment(duration, hamiltonian, True, f"ZX45({sign:+d})")


def _idle_segment(duration_ns: float, label: str) -> tuple[SemanticSegment, ...]:
    """Return one zero-Hamiltonian semantic segment when duration is positive."""
    if duration_ns <= 0.0:
        return ()
    return (
        SemanticSegment(
            duration_ns,
            np.zeros((9, 9), dtype=np.complex128),
            False,
            label,
        ),
    )


def semantic_zx90(descriptor: ZX90Descriptor) -> tuple[SemanticOperation, ...]:
    """Build one echoed ZX90 with actual lobe, pi, and margin durations."""
    half_margin = descriptor.echo_margin_duration_ns / 2.0
    pi = SemanticSegment(
        descriptor.pi_pulse_duration_ns,
        rotation_hamiltonian(
            X_CONTROL,
            np.pi,
            descriptor.pi_pulse_duration_ns,
        ),
        False,
        "XI180(internal)",
    )
    slot = (
        *_idle_segment(half_margin, "echo-margin"),
        pi,
        *_idle_segment(half_margin, "echo-margin"),
    )
    return (
        semantic_cr_lobe(descriptor, sign=1),
        *slot,
        semantic_cr_lobe(descriptor, sign=-1),
        *slot,
    )


def semantic_un_echoed_zx90(
    descriptor: ZX90Descriptor,
) -> tuple[SemanticOperation, ...]:
    """Build the A/B same-sign rotary-free unit with duration-matched blanks."""
    blank = _idle_segment(descriptor.echo_slot_duration_ns, "echo-slot-blank")
    return (
        semantic_cr_lobe(descriptor, sign=1, include_rotary=False),
        *blank,
        semantic_cr_lobe(descriptor, sign=1, include_rotary=False),
        *blank,
    )


def semantic_protocol_blocks(
    descriptor: ZX90Descriptor,
    *,
    external_durations_ns: Mapping[str, float],
) -> dict[str, tuple[SemanticOperation, ...]]:
    """Build the four repeated semantic protocol blocks."""
    un_echoed = semantic_un_echoed_zx90(descriptor)
    zx90 = semantic_zx90(descriptor)
    xi = float(external_durations_ns["xi180"])
    yi = float(external_durations_ns["yi180"])
    ix = float(external_durations_ns["ix180"])
    iy = float(external_durations_ns["iy180"])
    xi180 = SemanticSegment(
        xi, rotation_hamiltonian(X_CONTROL, np.pi, xi), False, "XI180"
    )
    yi180 = SemanticSegment(
        yi, rotation_hamiltonian(Y_CONTROL, np.pi, yi), False, "YI180"
    )
    iy180 = SemanticSegment(
        iy, rotation_hamiltonian(Y_TARGET, np.pi, iy), False, "IY180"
    )
    simultaneous = simultaneous_rotation_segments(
        (
            (Y_CONTROL, np.pi, yi, "YI180"),
            (X_TARGET, np.pi, ix, "IX180"),
        )
    )
    iz180 = SemanticUnitary(
        rotation_unitary(Z_TARGET, np.pi),
        "IZ180(target+CR-frame)",
    )
    return {
        PROTOCOL_A: un_echoed * 4,
        PROTOCOL_B: un_echoed * 4,
        PROTOCOL_C: (
            *zx90,
            xi180,
            *zx90,
            *simultaneous,
            *zx90,
            xi180,
            *zx90,
            yi180,
        ),
        PROTOCOL_D: (
            *zx90,
            iz180,
            *zx90,
            iy180,
            *zx90,
            iz180,
            *zx90,
            iy180,
        ),
    }


# Hardware schedules for primary and diagnostic acquisitions


def _state_preparation(
    exp: Any,
    control: str,
    target: str,
    control_state: Literal["0", "1", "+"],
    target_state: Literal["0", "+", "+y"],
) -> PulseSchedule:
    """Build one protocol input-state preparation schedule."""
    with PulseSchedule([control, target]) as schedule:
        schedule.add(control, exp.pulse.get_pulse_for_state(control, control_state))
        if target_state == "+y":
            schedule.add(target, exp.pulse.x90m(target))
        else:
            schedule.add(target, exp.pulse.get_pulse_for_state(target, target_state))
        schedule.barrier()
    return schedule


def append_analyzer(
    schedule: PulseSchedule,
    target: str,
    analyzer: Waveform,
) -> PulseSchedule:
    """Append an analyzer while preserving schedule frequency metadata."""
    measured = schedule.copy()
    with measured:
        measured.barrier()
        measured.add(target, analyzer)
    return measured


def _actual_protocol_block(
    exp: Any,
    control: str,
    target: str,
    descriptor: ZX90Descriptor,
    protocol: str,
) -> PulseSchedule:
    """Build one calibrated repeated block for protocol A, B, C, or D."""
    if protocol in (PROTOCOL_A, PROTOCOL_B):
        return descriptor.full_un_echoed.repeated(4)
    if protocol == PROTOCOL_C:
        with PulseSchedule() as block:
            block.call(descriptor.echoed, copy=True)
            block.add(control, exp.pulse.x180(control))
            block.barrier()
            block.call(descriptor.echoed, copy=True)
            block.add(control, exp.pulse.y180(control))
            block.add(target, exp.pulse.x180(target))
            block.barrier()
            block.call(descriptor.echoed, copy=True)
            block.add(control, exp.pulse.x180(control))
            block.barrier()
            block.call(descriptor.echoed, copy=True)
            block.add(control, exp.pulse.y180(control))
        block.set_frequencies(_frequencies(descriptor.echoed))
        return block
    if protocol == PROTOCOL_D:
        cr_label = f"{control}-{target}"
        with PulseSchedule() as block:
            for _ in range(2):
                block.call(descriptor.echoed, copy=True)
                z180 = exp.pulse.z180()
                block.add(target, z180)
                block.add(cr_label, z180)
                block.barrier()
                block.call(descriptor.echoed, copy=True)
                block.add(target, exp.pulse.y180(target))
                block.barrier()
        block.set_frequencies(_frequencies(descriptor.echoed))
        return block
    raise ValueError(f"Unknown protocol {protocol!r}.")


def _protocol_preparations(
    exp: Any,
    control: str,
    target: str,
) -> dict[str, PulseSchedule]:
    """Build the common input-state preparations for all four protocols."""
    return {
        PROTOCOL_A: _state_preparation(exp, control, target, "0", "+"),
        PROTOCOL_B: _state_preparation(exp, control, target, "1", "+"),
        PROTOCOL_C: _state_preparation(exp, control, target, "+", "+"),
        PROTOCOL_D: _state_preparation(exp, control, target, "0", "0"),
    }


def _primary_analyzers(
    exp: Any,
    control: str,
    target: str,
) -> dict[str, tuple[str, Waveform | None]]:
    """Build the common primary analyzers for all four protocols."""
    return {
        PROTOCOL_A: (target, exp.pulse.y90(target)),
        PROTOCOL_B: (target, exp.pulse.y90(target)),
        PROTOCOL_C: (control, exp.pulse.y90(control)),
        PROTOCOL_D: (target, None),
    }


def build_protocol_schedules(
    exp: Any,
    control: str,
    target: str,
    descriptor: ZX90Descriptor,
    repetition_counts: Sequence[int],
) -> ProtocolSchedules:
    """Construct all primary schedules before any acquisition starts."""
    blocks = {
        protocol: _actual_protocol_block(exp, control, target, descriptor, protocol)
        for protocol in PROTOCOLS
    }
    external = {
        "xi180": float(exp.pulse.x180(control).duration),
        "yi180": float(exp.pulse.y180(control).duration),
        "ix180": float(exp.pulse.x180(target).duration),
        "iy180": float(exp.pulse.y180(target).duration),
    }
    semantic = semantic_protocol_blocks(descriptor, external_durations_ns=external)
    preparations = _protocol_preparations(exp, control, target)
    primary_analyzers = _primary_analyzers(exp, control, target)
    actual: dict[str, tuple[PulseSchedule, ...]] = {}
    base_schedules: dict[str, tuple[PulseSchedule, ...]] = {}
    elapsed: dict[str, np.ndarray] = {}
    cr_active: dict[str, np.ndarray] = {}
    count_array = np.asarray(repetition_counts, dtype=np.float64)
    for protocol in PROTOCOLS:
        sequences = []
        bases = []
        for count in repetition_counts:
            with PulseSchedule() as base:
                base.call(preparations[protocol], copy=True)
                base.call(blocks[protocol].repeated(int(count)), copy=True)
            base.set_frequencies(_frequencies(descriptor.echoed))
            bases.append(base)
            analyzer_target, analyzer = primary_analyzers[protocol]
            sequences.append(
                base
                if analyzer is None
                else append_analyzer(base, analyzer_target, analyzer)
            )
        actual[protocol] = tuple(sequences)
        base_schedules[protocol] = tuple(bases)
        elapsed[protocol] = count_array * blocks[protocol].duration
        cr_active[protocol] = count_array * 8.0 * descriptor.cr_lobe_duration_ns

    timing = {
        PROTOCOL_A: {
            "block_duration_ns": float(blocks[PROTOCOL_A].duration),
            "cr_active_duration_ns": 8.0 * descriptor.cr_lobe_duration_ns,
            "cr_lobe_duration_ns": descriptor.cr_lobe_duration_ns,
            "blank_layer_duration_ns": descriptor.echo_slot_duration_ns,
        },
        PROTOCOL_B: {
            "block_duration_ns": float(blocks[PROTOCOL_B].duration),
            "cr_active_duration_ns": 8.0 * descriptor.cr_lobe_duration_ns,
            "cr_lobe_duration_ns": descriptor.cr_lobe_duration_ns,
            "blank_layer_duration_ns": descriptor.echo_slot_duration_ns,
        },
        PROTOCOL_C: {
            "block_duration_ns": float(blocks[PROTOCOL_C].duration),
            "cr_active_duration_ns": 8.0 * descriptor.cr_lobe_duration_ns,
            "zx90_cr_lobe_duration_ns": descriptor.cr_lobe_duration_ns,
            "zx90_control_echo_duration_ns": descriptor.pi_pulse_duration_ns,
            "zx90_echo_margin_duration_ns": descriptor.echo_margin_duration_ns,
            "external_xi180_duration_ns": external["xi180"],
            "external_yi180_duration_ns": external["yi180"],
            "external_ix180_duration_ns": external["ix180"],
        },
        PROTOCOL_D: {
            "block_duration_ns": float(blocks[PROTOCOL_D].duration),
            "cr_active_duration_ns": 8.0 * descriptor.cr_lobe_duration_ns,
            "zx90_cr_lobe_duration_ns": descriptor.cr_lobe_duration_ns,
            "zx90_control_echo_duration_ns": descriptor.pi_pulse_duration_ns,
            "zx90_echo_margin_duration_ns": descriptor.echo_margin_duration_ns,
            "iy180_duration_ns": external["iy180"],
            "virtual_iz180_duration_ns": 0.0,
            "operation_order": (
                "ZX90",
                "IZ180",
                "ZX90",
                "IY180",
                "ZX90",
                "IZ180",
                "ZX90",
                "IY180",
            ),
        },
    }
    return ProtocolSchedules(
        actual,
        base_schedules,
        semantic,
        elapsed,
        cr_active,
        timing,
    )


def _echo_replacement_gate(
    descriptor: ZX90Descriptor,
    control: str,
    target: str,
    replacement_lobe: PulseSchedule,
) -> PulseSchedule:
    """Preserve internal echo slots while replacing both CR-active lobes."""
    labels = (control, f"{control}-{target}", target)
    half_margin = descriptor.echo_margin_duration_ns / 2.0
    frequencies = _frequencies(descriptor.echoed)
    with PulseSchedule(list(labels)) as gate:
        for _ in range(2):
            gate.call(replacement_lobe, copy=True)
            gate.barrier()
            gate.add(control, Blank(half_margin))
            gate.add(control, descriptor.pi_pulse)
            gate.add(control, Blank(half_margin))
            gate.barrier()
    gate.set_frequencies(frequencies)
    if not np.isclose(gate.duration, descriptor.total_duration_ns, atol=1e-9, rtol=0.0):
        raise ValueError("Reference ZX90 does not preserve the echoed gate duration.")
    return gate


def _ix_lobe(
    descriptor: ZX90Descriptor,
    control: str,
    target: str,
    pulse: Waveform | None,
) -> PulseSchedule:
    """Build one un-echoed IX lobe with the calibrated CR envelope."""
    labels = (control, f"{control}-{target}", target)
    with PulseSchedule(list(labels)) as lobe:
        if pulse is None:
            lobe.add(control, Blank(descriptor.cr_lobe_duration_ns))
        else:
            lobe.add(target, pulse)
    lobe.set_frequencies(_frequencies(descriptor.echoed))
    if not np.isclose(
        lobe.duration, descriptor.cr_lobe_duration_ns, atol=1e-9, rtol=0.0
    ):
        raise ValueError("Reference IX45 must match the CR lobe duration.")
    return lobe


def build_reference_schedules(
    exp: Any,
    control: str,
    target: str,
    descriptor: ZX90Descriptor,
    repetition_counts: Sequence[int],
    ix45: Waveform,
) -> dict[str, tuple[PulseSchedule, ...]]:
    """Build diagnostic-only, duration-matched reference schedules."""
    positive_lobe = _ix_lobe(descriptor, control, target, ix45)
    negative_lobe = _ix_lobe(descriptor, control, target, ix45.scaled(-1.0))
    blank_lobe = _ix_lobe(descriptor, control, target, None)
    slot = _blank_schedule(
        (control, f"{control}-{target}", target),
        descriptor.echo_slot_duration_ns,
        _frequencies(descriptor.echoed),
    )

    def un_echoed(lobe: PulseSchedule) -> PulseSchedule:
        with PulseSchedule() as unit:
            unit.call(lobe, copy=True)
            unit.call(slot, copy=True)
            unit.call(lobe, copy=True)
            unit.call(slot, copy=True)
        unit.set_frequencies(_frequencies(descriptor.echoed))
        return unit

    a_unit = un_echoed(positive_lobe).repeated(4)
    b_unit = un_echoed(negative_lobe).repeated(4)
    c_zx = _echo_replacement_gate(descriptor, control, target, blank_lobe)
    d_zx = _echo_replacement_gate(descriptor, control, target, positive_lobe)
    c_block = _actual_protocol_block(
        exp, control, target, replace(descriptor, echoed=c_zx), PROTOCOL_C
    )
    d_block = _actual_protocol_block(
        exp, control, target, replace(descriptor, echoed=d_zx), PROTOCOL_D
    )
    units = {
        PROTOCOL_A: a_unit,
        PROTOCOL_B: b_unit,
        PROTOCOL_C: c_block,
        PROTOCOL_D: d_block,
    }
    preparations = _protocol_preparations(exp, control, target)
    analyzers = _primary_analyzers(exp, control, target)
    result: dict[str, tuple[PulseSchedule, ...]] = {}
    for protocol in PROTOCOLS:
        sequences = []
        for count in repetition_counts:
            with PulseSchedule() as base:
                base.call(preparations[protocol], copy=True)
                base.call(units[protocol].repeated(int(count)), copy=True)
            base.set_frequencies(_frequencies(descriptor.echoed))
            analyzer_target, analyzer = analyzers[protocol]
            sequences.append(
                base
                if analyzer is None
                else append_analyzer(base, analyzer_target, analyzer)
            )
        result[protocol] = tuple(sequences)
    return result
