"""
Explicit pulse construction and logical-frame transport for manual bSWAP runs.

No function connects to hardware or changes a calibration. Frequencies are GHz,
times ns, and output samples command amplitudes. The SQUAD design convention is
delta=omega_transition-omega_drive and I+iQ with H/hbar=(-delta Z+I X+Q Y)/2.
Its CD quadrature is -strength*delta*dI/dt/(delta**2+I**2). Construction uses
angular rates; one fixed measured conversion K=2*pi*r scales BOTH I and Q back
to command units. `design_delta_scale` changes the design, never this K.

Compiler validation establishes the emitted-waveform/phase contract, not an
analog transfer function, multilevel fidelity, or hardware readiness.
Explicit RaisedCosine recipes provide an I-only comparison family; their
carrier is transported by the same compiler without a SQUAD design detuning.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from itertools import product
from typing import Any

import numpy as np
from qxpulse import Blank, FlatTop, PulseSchedule, VirtualZ


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _grid_time(value: Any, name: str, sampling_period_ns: float = 2.0) -> float:
    value = _finite(value, name)
    if value < 0 or not np.isclose(
        value / sampling_period_ns, round(value / sampling_period_ns), atol=1e-9, rtol=0
    ):
        raise ValueError(
            f"{name} must be nonnegative on the {sampling_period_ns:g} ns grid"
        )
    return value


def _wrap(values: Any) -> Any:
    return np.angle(np.exp(1j * np.asarray(values)))


def make_squad_pulse(
    recipe: Any,
    *,
    rabi_ghz_per_amplitude: float,
    transition_frequency_ghz: float,
    sampling_period_ns: float = 2.0,
    max_command: float = 1.0,
) -> FlatTop:
    """
    Materialize a carrier-adaptive SQUAD+CD pulse in command units.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Command amplitude, carrier `frequency_ghz`, and complete `duration_ns`.
        Optional `ramp_ns=16`, `cd_strength=1`, `design_delta_scale=1`,
        and dictionary `window={"type": "hann"}` define the waveform.
    rabi_ghz_per_amplitude : float
        Positive measured cyclic Rabi rate per command amplitude.
    transition_frequency_ghz : float
        GE reference frequency. Signed design delta is the transition frequency
        minus the requested drive, multiplied by 2*pi and design_delta_scale.
    sampling_period_ns : float, optional
        Positive waveform interval; all pulse durations must lie on its grid.
    max_command : float, optional
        Maximum sampled complex command magnitude, default 1.

    Returns
    -------
    FlatTop
        Complete I+iQ envelope, including both ramps, scaled by the same
        measured K=2*pi*r. Design optimization never changes that conversion.

    Raises
    ------
    ValueError
        Invalid units, native-grid duration, zero detuning, or excess headroom.
    TypeError
        The ramp window is not an explicit dictionary.

    Notes
    -----
    Design delta is an empirical shaping parameter, not a new measurement of
    physical detuning or strong-drive gain. This function has no hardware I/O.
    """
    sample = _finite(sampling_period_ns, "sampling_period_ns")
    if sample <= 0:
        raise ValueError("sampling_period_ns must be positive")
    duration = _grid_time(recipe["duration_ns"], "duration_ns", sample)
    ramp = _grid_time(recipe.get("ramp_ns", 16.0), "ramp_ns", sample)
    amplitude = _finite(recipe["amplitude"], "amplitude")
    gain = _finite(rabi_ghz_per_amplitude, "rabi_ghz_per_amplitude")
    design_scale = _finite(recipe.get("design_delta_scale", 1.0), "design_delta_scale")
    strength = _finite(recipe.get("cd_strength", 1.0), "cd_strength")
    ceiling = _finite(max_command, "max_command")
    if ramp <= 0 or duration < 2 * ramp:
        raise ValueError("duration_ns must include two positive ramp_ns intervals")
    if gain <= 0 or design_scale <= 0 or not 0 < amplitude <= ceiling:
        raise ValueError(
            "Require positive gain/design_delta_scale and amplitude within max_command"
        )
    delta = (
        design_scale
        * 2
        * np.pi
        * (
            _finite(transition_frequency_ghz, "transition_frequency_ghz")
            - _finite(recipe["frequency_ghz"], "frequency_ghz")
        )
    )
    if delta == 0:
        raise ValueError("SQUAD design delta cannot be zero")
    window = recipe.get("window", {"type": "hann"})
    if not isinstance(window, dict):
        raise TypeError("window must be an explicit dictionary")
    k = 2 * np.pi * gain
    pulse = FlatTop(
        duration=duration,
        amplitude=k * amplitude,
        tau=ramp,
        type="Squad",
        delta=delta,
        window=deepcopy(window),
        correction_type="CD",
        correction_factor=strength,
        scale=1 / k,
        sampling_period=sample,
    )
    values = pulse.values
    if not np.isfinite(values).all() or np.max(np.abs(values)) > ceiling + 1e-10:
        raise ValueError("Sampled SQUAD I+iQ exceeds the command headroom")
    return pulse


def make_bswap_pulse(
    recipe: Mapping[str, Any],
    *,
    rabi_ghz_per_amplitude: float,
    transition_frequency_ghz: float,
    sampling_period_ns: float = 2.0,
    max_command: float = 1.0,
) -> FlatTop:
    """
    Materialize an explicitly selected bSWAP envelope in command units.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        `pulse_family` is `"Squad"` by default, or explicit `"RaisedCosine"`.
        Both use command `amplitude`, carrier `frequency_ghz`, complete
        `duration_ns`, and per-side `ramp_ns` (default 16 ns). SQUAD delegates
        unchanged to `make_squad_pulse`, including its default CD strength 1.
        RaisedCosine is I-only: omitted or zero `cd_strength` disables CD;
        nonzero corrections, `window`, and SQUAD design parameters are rejected.
    rabi_ghz_per_amplitude : float
        Positive cyclic Rabi conversion in GHz per command amplitude.
    transition_frequency_ghz : float
        Finite GE reference in GHz. Used for SQUAD's signed transition-minus-
        carrier design detuning; RaisedCosine I-only does not depend on it.
    sampling_period_ns : float, optional
        Positive sampling interval; complete and per-side ramp durations must
        lie on this grid. The campaign compiler requires the native 2 ns grid.
    max_command : float, optional
        Maximum finite complex sample magnitude, default 1.

    Returns
    -------
    FlatTop
        Complete envelope constructed with angular amplitude K*A and scaled
        back by 1/K, where K=2*pi*r. For I-only RaisedCosine this reproduces
        the ordinary command-amplitude FlatTop without a detuning correction.

    Raises
    ------
    ValueError
        Unknown family, unsupported shape/correction input, invalid units or
        timing, or sampled command headroom violation.

    Notes
    -----
    No hardware access or calibration mutation occurs. Family selection does
    not establish gate quality; each family needs independent calibration.
    The existing SQUAD function and its default behavior are unchanged.
    """
    family = recipe.get("pulse_family", "Squad")
    if family == "Squad":
        return make_squad_pulse(
            recipe,
            rabi_ghz_per_amplitude=rabi_ghz_per_amplitude,
            transition_frequency_ghz=transition_frequency_ghz,
            sampling_period_ns=sampling_period_ns,
            max_command=max_command,
        )
    if family != "RaisedCosine":
        raise ValueError(f"Unknown bSWAP pulse family: {family!r}")
    unsupported = {
        "window",
        "design_delta_scale",
        "delta",
        "squad_delta_control",
        "correction_type",
        "correction_factor",
        "factor",
        "beta",
        "beta_mode",
        "beta_sum",
        "tukey_rise_end",
        "tukey_fall_start",
    }.intersection(recipe)
    if unsupported:
        raise ValueError(
            "RaisedCosine I-only does not accept shape/correction keys: "
            + ", ".join(sorted(unsupported))
        )
    strength = _finite(recipe.get("cd_strength", 0.0), "cd_strength")
    if strength != 0.0:
        raise ValueError("RaisedCosine currently supports I-only cd_strength=0")
    sample = _finite(sampling_period_ns, "sampling_period_ns")
    if sample <= 0:
        raise ValueError("sampling_period_ns must be positive")
    duration = _grid_time(recipe["duration_ns"], "duration_ns", sample)
    ramp = _grid_time(recipe.get("ramp_ns", 16.0), "ramp_ns", sample)
    amplitude = _finite(recipe["amplitude"], "amplitude")
    gain = _finite(rabi_ghz_per_amplitude, "rabi_ghz_per_amplitude")
    ceiling = _finite(max_command, "max_command")
    _finite(recipe["frequency_ghz"], "frequency_ghz")
    _finite(transition_frequency_ghz, "transition_frequency_ghz")
    if ramp <= 0 or duration < 2 * ramp:
        raise ValueError("duration_ns must include two positive ramp_ns intervals")
    if gain <= 0 or not 0 < amplitude <= ceiling:
        raise ValueError("Require positive gain and amplitude within max_command")
    k = 2 * np.pi * gain
    pulse = FlatTop(
        duration=duration,
        amplitude=k * amplitude,
        tau=ramp,
        type="RaisedCosine",
        correction_type=None,
        scale=1 / k,
        sampling_period=sample,
    )
    values = pulse.values
    if not np.isfinite(values).all() or np.max(np.abs(values)) > ceiling + 1e-10:
        raise ValueError("Sampled RaisedCosine I-only exceeds the command headroom")
    return pulse


def measured_phase_vectors(recipe: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Return measured Pre P and Post Q, not inverse VZ corrections.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Measured `phase_calibration` with pre_active_rad, post_active_rad,
        post_passive_rad, and optional pre_passive_rad in radians.

    Returns
    -------
    tuple[NDArray, NDArray]
        Active/passive measured Pre and Post vectors in radians.

    Raises
    ------
    KeyError
        Required measured phase fields are absent.

    Notes
    -----
    The manual fit uses passive Pre=0 as its explicit gauge. An uncorrected
    calibration probe may provide explicitly zero measured phases. Missing
    calibration is never silently replaced by zero or by inverse corrections.
    """
    cal = recipe["phase_calibration"]
    pre = np.array(
        [
            _finite(cal["pre_active_rad"], "pre_active_rad"),
            _finite(cal.get("pre_passive_rad", 0.0), "pre_passive_rad"),
        ]
    )
    post = np.array(
        [
            _finite(cal["post_active_rad"], "post_active_rad"),
            _finite(cal["post_passive_rad"], "post_passive_rad"),
        ]
    )
    return pre, post


def local_z(angles: Sequence[float] | np.ndarray) -> np.ndarray:
    """Logical matrix utility only; the compiler never emits physical Z pulses."""
    return np.kron(*(np.diag(np.exp(-0.5j * a * np.array([1, -1]))) for a in angles))


def local_xy(qubit: int, phase: float, angle: float = np.pi / 2) -> np.ndarray:
    """Return the logical two-qubit matrix of a local XY rotation in radians."""
    axis = np.array([[0, np.exp(-1j * phase)], [np.exp(1j * phase), 0]])
    op = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * axis
    return np.kron(op, np.eye(2)) if qubit == 0 else np.kron(np.eye(2), op)


def _exchange(angle: float, zz_phase: float = 0.0) -> np.ndarray:
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    op = np.array(
        [[c, 0, 0, 1j * s], [0, 1, 0, 0], [0, 0, 1, 0], [1j * s, 0, 0, c]], complex
    )
    return op @ np.diag(np.exp(-0.5j * zz_phase * np.array([1, -1, -1, 1])))


def _expanded(gates: Sequence[Any]) -> Iterator[Any]:
    for gate in gates:
        if isinstance(gate, str) and gate == "ROOT_PAIR":
            yield "RAW_SQRT_BSWAP"
            yield "RAW_SQRT_BSWAP"
        elif isinstance(gate, str) and gate == "XX90":
            yield from ("RAW_SQRT_BSWAP", "IX180", "RAW_SQRT_BSWAP", "IX180")
        else:
            yield gate


def ideal_circuit_unitary(
    gates: Sequence[Any], *, zz_phases: Mapping[str, float] | None = None
) -> np.ndarray:
    """
    Return the fixed logical circuit target in active/passive order.

    Parameters
    ----------
    gates : Sequence
        BSWAP, RAW_SQRT_BSWAP, ROOT_PAIR, XX90, local rotations, VZ, or IDLE.
    zz_phases : Mapping[str, float], optional
        Frozen diagnostic `bswap`/`sqrt_bswap` phases zeta in radians for
        exp(-i*zeta*ZZ/2). Omission gives the zero-ZZ ideal benchmark target.

    Returns
    -------
    NDArray
        Complex 4x4 target matrix. XX90 has the plus-i convention up to
        irrelevant global phase.

    Notes
    -----
    Diagnostic ZZ values must not be fitted to the benchmark shots. This is
    logical matrix algebra, not fictitious physical-Z waveform simulation.
    """
    zz = {} if zz_phases is None else zz_phases
    result = np.eye(4, dtype=complex)
    for gate in _expanded(gates):
        if isinstance(gate, (tuple, list)):
            if gate[0] == "XY":
                op = local_xy(1, gate[2]) @ local_xy(0, gate[1])
            elif gate[0] == "VZ":
                op = local_z(gate[1:])
            elif gate[0] == "IDLE":
                op = np.eye(4)
            else:
                raise ValueError(f"Unknown logical operation {gate}")
        elif gate in ("BSWAP", "RAW_SQRT_BSWAP"):
            kind = "bswap" if gate == "BSWAP" else "sqrt_bswap"
            op = _exchange(np.pi if kind == "bswap" else np.pi / 2, zz.get(kind, 0.0))
        elif gate in ("XI90", "IX90", "XI180", "IX180"):
            op = local_xy(
                0 if gate.startswith("XI") else 1,
                0,
                np.pi if gate.endswith("180") else np.pi / 2,
            )
        elif gate in ("ZI90", "IZ90"):
            op = local_z([np.pi / 2, 0] if gate == "ZI90" else [0, np.pi / 2])
        else:
            raise ValueError(f"Unknown logical operation {gate}")
        result = op @ result
    return result


def single_qubit_ensemble64() -> tuple[tuple[float, float], ...]:
    """
    Return eight XY axes times eight post-VZ angles.

    Returns
    -------
    tuple[tuple[float, float], ...]
        Sixty-four pairs of XY-axis and post-VZ angles in radians.

    Notes
    -----
    The enumeration does not claim that this ensemble is a Clifford 2-design.
    """
    return tuple(
        (float(a), float(z)) for a, z in product(np.arange(8) * np.pi / 4, repeat=2)
    )


def xeb_circuit(
    depth: int, seed: int, gate_name: str | None, *, terminal_layer: bool = True
) -> dict[str, Any]:
    """
    Generate a frozen reproducible 64-local-ensemble circuit.

    Parameters
    ----------
    depth : int
        Nonnegative number of local-plus-entangler cycles.
    seed : int
        Random seed shared between paired reference and entangling circuits.
    gate_name : str or None
        BSWAP, RAW_SQRT_BSWAP, ROOT_PAIR, XX90, or None for local-only.
    terminal_layer : bool, optional
        Append an independently sampled XY/VZ layer after the last entangler,
        default True. Entangling depth and primitive count remain unchanged.

    Returns
    -------
    dict
        Gates, local pattern indices, zero-ZZ ideal probabilities, seed,
        terminal_layer metadata, and target_refitted_on_benchmark=False.
        local_indices has depth+1 rows when terminal randomization is enabled.

    Notes
    -----
    None omits entangler time; it is not a duration-matched idle reference.
    Full bSWAP maps one equatorial local layer to uniform computational-basis
    probabilities. The default terminal layer makes its phase information
    visible in population measurements. Paired local-only circuits have the
    same sampled local indices, including this terminal layer.
    """
    if isinstance(depth, bool) or int(depth) != depth or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    if gate_name not in (None, "BSWAP", "RAW_SQRT_BSWAP", "ROOT_PAIR", "XX90"):
        raise ValueError("Unsupported XEB gate_name")
    if not isinstance(terminal_layer, bool):
        raise TypeError("terminal_layer must be a boolean")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 64, (int(depth) + int(terminal_layer), 2))
    ensemble = single_qubit_ensemble64()
    gates = []
    for layer, (active, passive) in enumerate(indices):
        (a, za), (p, zp) = ensemble[active], ensemble[passive]
        gates.extend([("XY", a, p), ("VZ", za, zp)])
        if gate_name is not None and layer < depth:
            gates.append(gate_name)
    probabilities = np.abs(ideal_circuit_unitary(gates)[:, 0]) ** 2
    return {
        "depth": int(depth),
        "seed": int(seed),
        "gate_name": gate_name,
        "terminal_layer": terminal_layer,
        "local_indices": indices.tolist(),
        "gates": gates,
        "ideal_probabilities": probabilities.tolist(),
        "target_refitted_on_benchmark": False,
    }


def compile_campaign(
    gates: Sequence[Any],
    *,
    recipes: Mapping[str, Any],
    qubits: Sequence[str],
    drive_label: str,
    cancel_label: str,
    target_frequencies_ghz: Mapping[str, float],
    reference_frequencies_ghz: Mapping[str, float],
    rabi_ghz_per_amplitude: float,
    x90: Mapping[str, Any],
    xpi: Mapping[str, Any],
    global_start_ns: float = 0.0,
    backend_preamble_ns: float = 0.0,
    prepared: Sequence[str] | None = ("0", "0"),
    basis: str | None = "ZZ",
    delay_ns: float = 0.0,
    initial_frame: Sequence[float] = (0.0, 0.0),
    terminal_frame: bool = True,
    sampling_period_ns: float = 2.0,
    max_command: float = 1.0,
    max_frequency_offset_ghz: float = 0.2,
) -> tuple[PulseSchedule, dict[str, Any]]:
    """
    Compile calibrated full/root/root-pair/XX90 with one logical frame.

    Parameters
    ----------
    gates : Sequence
        Named gates or explicit XY/VZ/IDLE tuples.
    recipes : Mapping[str, Any]
        bswap and/or sqrt_bswap records, each including measured
        phase_calibration and gate_start_ns. Optional cancel_amplitude_ratio
        and cancel_phase_rad specify a phase-coherent passive tone.
        `pulse_family` defaults to SQUAD; explicit `"RaisedCosine"` selects
        the I-only family described by `make_bswap_pulse`.
    qubits : Sequence[str]
        Active then passive GE target labels.
    drive_label, cancel_label : str
        Distinct main and passive Stark target labels.
    target_frequencies_ghz : Mapping[str, float]
        Fixed Stark-target reference frequencies used for acquisition.
    reference_frequencies_ghz : Mapping[str, float]
        Delivered active/passive GE reference frequencies.
    rabi_ghz_per_amplitude : float
        Fixed measured cyclic Rabi conversion for the main drive.
    x90, xpi : Mapping[str, Pulse]
        Fresh native-grid production DRAG pulses for both qubits.
    global_start_ns : float, optional
        Absolute start of the returned parent schedule.
    backend_preamble_ns : float, optional
        Extra prefix inserted after compilation by the measurement backend.
        Used only for carrier-offset phase compensation; no blank is inserted
        and the logical placement/calibration time origin is unchanged.
    prepared : Sequence[str] or None, optional
        Input states, default 00. None omits state preparation.
    basis : str or None, optional
        Two-Pauli analysis setting, default ZZ. None omits analysis pulses.
    delay_ns : float, optional
        Native-grid delay before preparation.
    initial_frame : Sequence[float], optional
        Incoming logical active/passive Z phases in radians.
    terminal_frame : bool, optional
        Export terminal VirtualZ metadata once; required for analysis.
    sampling_period_ns : float, optional
        Native waveform grid, currently 2 ns only.
    max_command : float, optional
        Maximum sampled complex command magnitude.
    max_frequency_offset_ghz : float, optional
        Allowed recipe-minus-target waveform offset, default 0.2 GHz.

    Returns
    -------
    tuple[PulseSchedule, dict]
        Schedule and event/phase/headroom report.

    Raises
    ------
    ValueError
        Invalid labels, timing, carrier offset, preparation, or headroom.
    KeyError
        A used recipe lacks measured calibration or a required reference.

    Notes
    -----
    For measured Pre P, Post Q, and incoming frame F:
    phi=k*(absolute_start-calibrated_start)+sum(P-F)/2 and F_after=F-P-Q.
    Here k=pi*(2*f_drive-f_active_ref-f_passive_ref), and local XY axes are
    requested_axis-F[q]. Because Pulse.detuned(df) uses local time, its
    source phase also includes -2*pi*df*(absolute_start+backend_preamble_ns).
    Main and cancel carriers receive the same logical phase, preserving
    relative phase. The QuEL-1 adapter conjugates U/default envelopes before
    positive-exponent backend modulation; do not reverse the detuned sign.

    Acquire with these same fixed target reference frequencies. Passing a
    recipe carrier again to Experiment.measure would double-detune it.
    Preparation is padded before DRAG pulses to the largest calibrated start.
    Terminal VZ affects emitted analysis pulses; no physical instantaneous Z
    is inserted. This function never connects or changes device settings.
    """
    qubits = tuple(qubits)
    if len(qubits) != 2 or len({*qubits, drive_label, cancel_label}) != 4:
        raise ValueError("Need two distinct qubits and distinct drive/cancel labels")
    sample = float(sampling_period_ns)
    if sample != 2.0:
        raise ValueError("This campaign compiler requires the native 2 ns grid")
    origin = _grid_time(global_start_ns, "global_start_ns", sample)
    backend_preamble = _grid_time(backend_preamble_ns, "backend_preamble_ns", sample)
    delay = _grid_time(delay_ns, "delay_ns", sample)
    frame = np.asarray(initial_frame, dtype=float).copy()
    if frame.shape != (2,) or not np.isfinite(frame).all():
        raise ValueError("initial_frame must contain two finite angles")
    if basis is not None and (len(basis) != 2 or any(a not in "XYZ" for a in basis)):
        raise ValueError("basis must be a two-Pauli string or None")
    if basis is not None and not terminal_frame:
        raise ValueError("Analysis requires terminal_frame=True")
    expanded = list(_expanded(gates))
    used = {
        "bswap" if g == "BSWAP" else "sqrt_bswap"
        for g in expanded
        if isinstance(g, str) and g in ("BSWAP", "RAW_SQRT_BSWAP")
    }
    # Even a local-only matched reference uses the supplied calibrated prep origin.
    records = []
    references = {
        q: _finite(reference_frequencies_ghz[q], q + " reference") for q in qubits
    }
    pulses, phases, starts = {}, {}, {}
    for kind in used:
        recipe = recipes[kind]
        if recipe.get("gate_kind", kind) != kind:
            raise ValueError("Recipe gate_kind does not match its key")
        phases[kind] = measured_phase_vectors(recipe)
        starts[kind] = _grid_time(recipe["gate_start_ns"], "gate_start_ns", sample)
        pulses[kind] = make_bswap_pulse(
            recipe,
            rabi_ghz_per_amplitude=rabi_ghz_per_amplitude,
            transition_frequency_ghz=references[qubits[0]],
            sampling_period_ns=sample,
            max_command=max_command,
        )
    prep_window = max(
        (
            _grid_time(r["gate_start_ns"], "gate_start_ns", sample)
            for r in recipes.values()
        ),
        default=0.0,
    )
    for collection in (x90, xpi):
        for q in qubits:
            pulse = collection[q]
            if float(pulse.sampling_period) != sample:
                raise ValueError("Production 1Q pulses must use the native 2 ns grid")
    labels = [*qubits, drive_label, cancel_label]
    schedule = PulseSchedule(labels)

    def local_operation(
        qi: int, axis: float, angle: float, context: str = "circuit"
    ) -> None:
        start = origin + float(schedule.duration)
        source = float(axis - frame[qi])
        pulse = (xpi if np.isclose(angle, np.pi) else x90)[qubits[qi]]
        schedule.add(qubits[qi], pulse.shifted(source))
        records.append(
            {
                "kind": "local",
                "context": context,
                "qubit": qi,
                "start_ns": start,
                "duration_ns": float(pulse.duration),
                "angle_rad": float(angle),
                "source_phase_rad": source,
            }
        )

    with schedule:
        for label in labels:
            schedule.add(label, Blank(delay, sampling_period=sample))
        if prepared is not None:
            if len(prepared) != 2:
                raise ValueError("prepared must contain two state labels")
            for qi, (q, state) in enumerate(zip(qubits, prepared, strict=True)):
                if state == "0":
                    schedule.add(q, Blank(prep_window, sampling_period=sample))
                    continue
                if state == "1":
                    pulse, axis, angle = xpi[q], 0.0, np.pi
                elif state in ("+", "+i", "-", "-i"):
                    pulse = x90[q]
                    axis = {"+": np.pi / 2, "+i": np.pi, "-": -np.pi / 2, "-i": 0.0}[
                        state
                    ]
                    angle = np.pi / 2
                else:
                    raise ValueError(f"Unsupported preparation {state}")
                if pulse.duration > prep_window:
                    raise ValueError(
                        "Production preparation pulse exceeds calibrated gate_start_ns"
                    )
                axis -= float(frame[qi])
                schedule.add(
                    q, Blank(prep_window - pulse.duration, sampling_period=sample)
                )
                schedule.add(q, pulse.shifted(axis))
                records.append(
                    {
                        "kind": "local",
                        "context": "preparation",
                        "qubit": qi,
                        "start_ns": origin + delay + prep_window - pulse.duration,
                        "duration_ns": float(pulse.duration),
                        "angle_rad": angle,
                        "source_phase_rad": axis,
                    }
                )
            schedule.barrier()
        for gate in expanded:
            start = origin + float(schedule.duration)
            if isinstance(gate, (tuple, list)):
                if gate[0] == "XY" and len(gate) == 3:
                    # Capture simultaneous origin before adding either channel.
                    simultaneous_start = start
                    for qi, axis in enumerate(gate[1:]):
                        local_operation(qi, _finite(axis, "XY axis"), np.pi / 2)
                        records[-1]["start_ns"] = simultaneous_start
                    schedule.barrier()
                elif gate[0] == "VZ" and len(gate) == 3:
                    frame += np.array([_finite(a, "logical VZ") for a in gate[1:]])
                elif gate[0] == "IDLE" and len(gate) == 2:
                    duration = _grid_time(gate[1], "idle_ns", sample)
                    for label in labels:
                        schedule.add(label, Blank(duration, sampling_period=sample))
                else:
                    raise ValueError(f"Unknown logical operation {gate}")
            elif gate in ("ZI90", "IZ90"):
                frame[0 if gate == "ZI90" else 1] += np.pi / 2
            elif gate in ("XI90", "IX90", "XI180", "IX180"):
                local_operation(
                    0 if gate.startswith("XI") else 1,
                    0.0,
                    np.pi if gate.endswith("180") else np.pi / 2,
                )
                schedule.barrier()
            elif gate in ("BSWAP", "RAW_SQRT_BSWAP"):
                kind = "bswap" if gate == "BSWAP" else "sqrt_bswap"
                recipe, pulse = recipes[kind], pulses[kind]
                pre, post = phases[kind]
                carrier = float(recipe["frequency_ghz"])
                rate = np.pi * (
                    2 * carrier - references[qubits[0]] - references[qubits[1]]
                )
                logical_phase = (
                    rate * (start - starts[kind]) + float(np.sum(pre - frame)) / 2
                )
                ratio = _finite(
                    recipe.get("cancel_amplitude_ratio", 0.0), "cancel_amplitude_ratio"
                )
                relative = _finite(
                    recipe.get("cancel_phase_rad", 0.0), "cancel_phase_rad"
                )
                if not 0 <= ratio <= 1:
                    raise ValueError("cancel_amplitude_ratio must lie in [0,1]")
                tone_records = {}
                for label, amplitude_scale, phase_delta in (
                    (drive_label, 1.0, 0.0),
                    (cancel_label, ratio, relative),
                ):
                    if amplitude_scale == 0:
                        schedule.add(
                            label, Blank(pulse.duration, sampling_period=sample)
                        )
                        continue
                    reference = _finite(
                        target_frequencies_ghz[label], label + " target reference"
                    )
                    offset = carrier - reference
                    if abs(offset) > max_frequency_offset_ghz:
                        raise ValueError(
                            "Recipe carrier exceeds permitted source frequency offset"
                        )
                    source_phase = (
                        logical_phase
                        + phase_delta
                        - 2 * np.pi * offset * (start + backend_preamble)
                    )
                    emitted = (
                        pulse.scaled(amplitude_scale)
                        .detuned(offset)
                        .shifted(source_phase)
                    )
                    schedule.add(label, emitted)
                    tone_records[label] = {
                        "frequency_offset_ghz": offset,
                        "source_phase_rad": float(source_phase),
                    }
                schedule.barrier()
                before = frame.copy()
                frame -= pre + post
                records.append(
                    {
                        "kind": kind,
                        "start_ns": start,
                        "duration_ns": float(pulse.duration),
                        "frequency_ghz": carrier,
                        "placement_rate_rad_per_ns": float(rate),
                        "logical_drive_phase_rad": float(logical_phase),
                        "pre_measured_rad": pre.tolist(),
                        "post_measured_rad": post.tolist(),
                        "frame_before_rad": before.tolist(),
                        "frame_after_rad": frame.tolist(),
                        "tones": tone_records,
                    }
                )
            else:
                raise ValueError(f"Unknown logical operation {gate}")
            frame = _wrap(frame)
        if terminal_frame:
            for q, angle in zip(qubits, frame, strict=True):
                schedule.add(q, VirtualZ(float(angle)))
        if basis is not None:
            analysis_start = origin + float(schedule.duration)
            analysis_duration = max(float(p.duration) for p in x90.values())
            for qi, (q, axis) in enumerate(zip(qubits, basis, strict=True)):
                if axis == "Z":
                    schedule.add(q, Blank(analysis_duration, sampling_period=sample))
                else:
                    source = -np.pi / 2 if axis == "X" else 0.0
                    pulse = x90[q]
                    schedule.add(q, pulse.shifted(source))
                    schedule.add(
                        q,
                        Blank(
                            analysis_duration - pulse.duration, sampling_period=sample
                        ),
                    )
                    records.append(
                        {
                            "kind": "local",
                            "context": "analysis",
                            "qubit": qi,
                            "start_ns": analysis_start,
                            "duration_ns": float(pulse.duration),
                            "angle_rad": np.pi / 2,
                            "source_phase_rad": float(source - frame[qi]),
                        }
                    )
            schedule.barrier()
        schedule.barrier()
    values = schedule.get_sampled_sequences()
    peaks = {
        label: float(np.max(np.abs(v))) if len(v) else 0.0
        for label, v in values.items()
    }
    if any(not np.isfinite(p) or p > max_command + 1e-10 for p in peaks.values()):
        raise ValueError("Compiled waveform exceeds command headroom")
    return schedule, {
        "events": records,
        "duration_ns": float(schedule.duration),
        "global_start_ns": origin,
        "backend_preamble_ns": backend_preamble,
        "preparation_window_ns": prep_window,
        "final_frame_rad": frame.tolist(),
        "terminal_frame_exported": bool(terminal_frame),
        "peak_command_by_label": peaks,
        "target_frequencies_ghz": dict(target_frequencies_ghz),
    }
