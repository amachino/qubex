"""Lightweight qubit specifications."""

from __future__ import annotations

import math

from qxcore import Frequency, Time

from ._normalization import normalize_frequency_to_ghz, resolve_decoherence_rates
from .object import Object


class Qubit(Object):
    """
    Represent a two-level qubit with phenomenological decoherence.

    Parameters
    ----------
    label : str
        Unique object label within a quantum system.
    frequency : float | Frequency
        Qubit 0-1 cyclic transition frequency. Bare floats are interpreted as
        GHz.
    t1 : float | Time | None, optional
        Energy-relaxation time. Bare floats are interpreted as ns. The default
        is `None`.
    t2 : float | Time | None, optional
        Total transverse-coherence time. Bare floats are interpreted as ns.
        The default is `None`.
    relaxation_rate : float | Frequency | None, optional
        Energy-relaxation rate. Bare floats are interpreted as inverse ns. The
        default is `None`.
    dephasing_rate : float | Frequency | None, optional
        Pure-dephasing rate. Bare floats are interpreted as inverse ns. The
        default is `None`.

    Raises
    ------
    ValueError
        If time and rate parameterizations are mixed, a time is nonpositive,
        a rate is negative, or `t2` implies a negative dephasing rate.

    Notes
    -----
    Times and rates are alternative parameterizations and cannot be mixed. The
    physical rates satisfy
    `1 / t2 = relaxation_rate / 2 + dephasing_rate`; omitted rates are zero.
    `frequency` is a cyclic frequency; Hamiltonian construction converts it to
    angular frequency with `2 * pi`. Decay-rate conversion does not introduce
    this factor. The Hilbert-space dimension is fixed to two, and all stored
    physical values are canonical floats.
    """

    def __init__(
        self,
        *,
        label: str,
        frequency: float | Frequency,
        t1: float | Time | None = None,
        t2: float | Time | None = None,
        relaxation_rate: float | Frequency | None = None,
        dephasing_rate: float | Frequency | None = None,
    ) -> None:
        normalized_frequency = normalize_frequency_to_ghz(frequency)
        resolved_relaxation_rate, resolved_dephasing_rate = resolve_decoherence_rates(
            t1=t1,
            t2=t2,
            relaxation_rate=relaxation_rate,
            dephasing_rate=dephasing_rate,
        )

        super().__init__(
            label=label,
            dimension=2,
            frequency=normalized_frequency,
            anharmonicity=math.inf,
            relaxation_rate=resolved_relaxation_rate,
            dephasing_rate=resolved_dephasing_rate,
        )
