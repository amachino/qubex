"""Frozen native-full-bSWAP Clifford cache and explicit frame-aware compiler."""

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qxpulse import Blank, PulseSchedule, VirtualZ, Waveform

from .tomography import local_z, raw_unitary


def x90_matrix(q: int, phase: float = 0.0) -> NDArray[np.complex128]:
    """Return an X90-axis rotation on qubit 0 or 1; phase is in radians."""
    axis = np.array([[0, np.exp(-1j * phase)], [np.exp(1j * phase), 0]])
    u = (np.eye(2) - 1j * axis) / np.sqrt(2)
    return np.kron(u, np.eye(2)) if q == 0 else np.kron(np.eye(2), u)


def circuit_unitary(gates: Sequence[str], zeta: float = 0.0) -> NDArray[np.complex128]:
    """Matrix of declared logical gates; ZZ is zero for the IRB inversion target."""
    u = np.eye(4, dtype=complex)
    for name in gates:
        if name == "BSWAP":
            op = raw_unitary("bswap", zeta=zeta)
        elif name in ("XI90", "IX90"):
            op = x90_matrix(0 if name == "XI90" else 1)
        elif name in ("ZI90", "IZ90"):
            op = local_z([np.pi / 2, 0] if name == "ZI90" else [0, np.pi / 2])
        else:
            raise ValueError(f"Unsupported gate {name}")
        u = op @ u
    return u


def unitary_key(u: NDArray[np.complex128]) -> tuple[int, ...]:
    """Return a global-phase-invariant rounded key for a nonzero unitary."""
    first = u.ravel()[np.flatnonzero(np.abs(u.ravel()) > 1e-7)[0]]
    v = u * np.exp(-1j * np.angle(first))
    return tuple(
        np.rint(np.r_[v.real.ravel(), v.imag.ravel()] * 1e8).astype(int).tolist()
    )


class NativeBSWAPCache:
    """Load a saved minus-i table; rebuild matrix keys in the explicit plus-i convention."""

    def __init__(self, path: str | Path) -> None:
        rows = json.loads(Path(path).read_text())
        self.sequences = []
        self.unitaries = []
        for row in rows:
            expanded = []
            for name in row["sequence"]:
                expanded.extend(
                    ["BSWAP", "ZI90", "ZI90", "IZ90", "IZ90"]
                    if name == "BSWAP"
                    else [name]
                )
            self.sequences.append(expanded)
            self.unitaries.append(circuit_unitary(expanded))
        self.lookup = {
            unitary_key(u): g
            for u, g in zip(self.unitaries, self.sequences, strict=True)
        }
        if len(rows) != 11520 or len(self.lookup) != 11520:
            raise ValueError(
                "Saved native table must contain 11520 distinct matrix Cliffords"
            )
        if any(unitary_key(u.conj().T) not in self.lookup for u in self.unitaries):
            raise ValueError("Saved Clifford table is not matrix-inverse closed")

    def circuit(
        self, depth: int, seed: int, interleaved: bool = False
    ) -> dict[str, Any]:
        """Sample Clifford rows and append the exact inverse without regenerating the cache."""
        indices = random.Random(int(seed)).choices(
            range(len(self.sequences)), k=int(depth)
        )
        gates = []
        u = np.eye(4, dtype=complex)
        for index in indices:
            gates.extend(self.sequences[index])
            u = self.unitaries[index] @ u
            if interleaved:
                gates.append("BSWAP")
                u = raw_unitary("bswap") @ u
        inverse = self.lookup[unitary_key(u.conj().T)]
        gates.extend(inverse)
        error = float(1 - abs(np.trace(circuit_unitary(gates))) ** 2 / 16)
        if abs(error) > 1e-10:
            raise ValueError("Ideal matrix inverse did not close")
        return dict(
            depth=int(depth),
            seed=int(seed),
            interleaved=bool(interleaved),
            random_indices=indices,
            inverse=inverse,
            gates=gates,
            bswap_count=gates.count("BSWAP"),
            ideal_closure_error=error,
        )


def compile_bswap(
    gates: Sequence[str],
    *,
    qubits: Sequence[str],
    drive_label: str,
    sizzle_label: str,
    gate: Waveform,
    x90: Mapping[str, Waveform],
    xpi: Mapping[str, Waveform],
    post_vz: Sequence[float] | NDArray[np.float64],
    placement_rate: float,
    reference_start: float,
    delay_ns: float = 0.0,
    prepared: Sequence[str] = ("0", "0"),
    basis: str = "ZZ",
) -> tuple[PulseSchedule, dict[str, Any]]:
    """
    Compile a full-bSWAP circuit with one logical-Z owner and a fixed origin.

    Parameters
    ----------
    gates : sequence of str
        Native full-bSWAP and local Clifford names.
    gate : Waveform
        Calibrated full-bSWAP waveform; this compiler assumes zero Pre gauge.
    post_vz : sequence of float
        Two measured inverse Post-VZ corrections in radians.
    placement_rate : float
        Setting-derived drive phase rate in rad/ns, not a fitted return offset.
    reference_start, delay_ns : float
        Calibrated gate-start reference and added pre-preparation delay in ns.

    Returns
    -------
    tuple
        Concrete schedule and emitted phase/headroom diagnostics. No hardware
        access is performed; residual ZZ is not corrected by local VZ.
    """
    pending = np.zeros(2)
    records = []
    labels = [*qubits, drive_label, sizzle_label]
    analysis_ns = max(p.duration for p in x90.values())
    with PulseSchedule(labels) as sequence:
        for label in labels:
            sequence.add(label, Blank(float(delay_ns)))
        for q, symbol in zip(qubits, prepared, strict=True):
            if symbol in ("+", "+i"):
                pulse = x90[q].shifted(np.pi / 2 if symbol == "+" else np.pi)
            elif symbol == "1":
                pulse = xpi[q]
            elif symbol == "0":
                pulse = Blank(reference_start)
            else:
                raise ValueError(symbol)
            if pulse.duration > reference_start:
                raise ValueError(
                    "Preparation is longer than the calibrated gate start time"
                )
            if pulse.duration < reference_start:
                sequence.add(q, Blank(reference_start - pulse.duration))
            sequence.add(q, pulse)
        sequence.barrier()
        for name in gates:
            start = float(sequence.duration)
            if name in ("ZI90", "IZ90"):
                pending[0 if name == "ZI90" else 1] += np.pi / 2
            elif name in ("XI90", "IX90"):
                qi = 0 if name == "XI90" else 1
                sequence.add(qubits[qi], x90[qubits[qi]].shifted(float(-pending[qi])))
                sequence.barrier()
            elif name == "BSWAP":
                phase = placement_rate * (start - reference_start) - pending.sum() / 2
                sequence.add(drive_label, gate.shifted(float(phase)))
                sequence.barrier()
                records.append(
                    dict(
                        start_ns=start,
                        drive_phase_rad=float(phase),
                        pending_before=pending.tolist(),
                    )
                )
                pending += np.asarray(post_vz)
            else:
                raise ValueError(name)
            pending = np.angle(np.exp(1j * pending))
        for qi, (q, axis) in enumerate(zip(qubits, basis, strict=True)):
            sequence.add(q, VirtualZ(float(pending[qi])))
            pulse = (
                Blank(analysis_ns)
                if axis == "Z"
                else x90[q].shifted(-np.pi / 2 if axis == "X" else 0.0)
            )
            sequence.add(q, pulse)
            if pulse.duration < analysis_ns:
                sequence.add(q, Blank(analysis_ns - pulse.duration))
        sequence.barrier()
    peak = max(
        float(np.max(np.abs(v))) if len(v) else 0.0
        for v in sequence.get_sampled_sequences().values()
    )
    if peak > 1 + 1e-10:
        raise ValueError(f"Compiled waveform exceeds command headroom: {peak}")
    return sequence, dict(
        duration_ns=float(sequence.duration),
        peak=peak,
        final_pending=pending.tolist(),
        bswap_events=records,
    )
