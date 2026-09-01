"""Lightweight resonator specifications."""

from __future__ import annotations

from qxcore import Frequency

from ._normalization import normalize_frequency_to_ghz, resolve_decoherence_rates
from .object import Object


class Resonator(Object):
    """
    Represent a truncated harmonic resonator with phenomenological decoherence.

    Parameters
    ----------
    label : str
        Unique object label within a quantum system.
    dimension : int
        Number of retained Fock levels.
    frequency : float | Frequency
        Resonator cyclic frequency. Bare floats are interpreted as GHz.
    relaxation_rate : float | Frequency | None, optional
        Energy-relaxation rate. Bare floats are interpreted as inverse ns. The
        default is `None`.
    dephasing_rate : float | Frequency | None, optional
        Pure-dephasing rate. Bare floats are interpreted as inverse ns. The
        default is `None`.

    Raises
    ------
    ValueError
        If either decoherence rate is negative.

    Notes
    -----
    Unlike `Qubit` and `Transmon`, a resonator accepts rates but not `t1` or
    `t2`. Omitted rates are zero. Compilation uses the truncated harmonic
    Hamiltonian `2 * pi * frequency * N` in rad/ns. Decay rates do not acquire
    the Hamiltonian's `2 * pi` conversion factor.
    """

    def __init__(
        self,
        *,
        label: str,
        dimension: int,
        frequency: float | Frequency,
        relaxation_rate: float | Frequency | None = None,
        dephasing_rate: float | Frequency | None = None,
    ) -> None:
        normalized_frequency = normalize_frequency_to_ghz(frequency)
        resolved_relaxation_rate, resolved_dephasing_rate = resolve_decoherence_rates(
            t1=None,
            t2=None,
            relaxation_rate=relaxation_rate,
            dephasing_rate=dephasing_rate,
        )
        super().__init__(
            label=label,
            dimension=dimension,
            frequency=normalized_frequency,
            anharmonicity=0.0,
            relaxation_rate=resolved_relaxation_rate,
            dephasing_rate=resolved_dephasing_rate,
        )
