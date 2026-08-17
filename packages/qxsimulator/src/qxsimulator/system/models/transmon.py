"""Lightweight quantum-system object specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qxcore import Frequency, Time
from typing_extensions import override

from qxsimulator.system.compiled_object import CompiledObject

from ._normalization import normalize_frequency_to_ghz, resolve_decoherence_rates
from .object import Object


@dataclass(frozen=True, init=False)
class Transmon(Object):
    """
    Represent a truncated multilevel transmon with a selectable local model.

    Parameters
    ----------
    label : str
        Unique object label within a quantum system.
    dimension : int
        Number of retained local levels.
    frequency : float | Frequency
        Lowest cyclic transition frequency `f_01`. Bare floats are interpreted
        as GHz.
    anharmonicity : float | Frequency | None, optional
        Signed cyclic anharmonicity `f_12 - f_01`. Bare floats are interpreted
        as GHz. If omitted, use `-0.05 * frequency`.
    model : {"duffing", "cosine"}, optional
        Local Hamiltonian model. The default is `"duffing"`.
    charge_cutoff : int | None, optional
        Positive charge-basis cutoff for the cosine model, which retains charge
        numbers from `-charge_cutoff` through `+charge_cutoff`. The cosine
        default is 25; the Duffing model does not use this value.
    offset_charge : float, optional
        Dimensionless offset charge `n_g` in Cooper-pair units for the cosine
        model. The default is 0. Integer shifts are equivalent at compilation.
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
        a rate is negative, `t2` implies a negative dephasing rate, or the
        selected local-model parameters are invalid.

    Notes
    -----
    Times and rates are alternative parameterizations and cannot be mixed. The
    physical rates satisfy
    `1 / t2 = relaxation_rate / 2 + dephasing_rate`; omitted rates are zero.
    `frequency` and `anharmonicity` are cyclic frequencies; Hamiltonian
    construction converts them to angular frequencies with `2 * pi`.
    Decay-rate conversion does not introduce this factor. All stored physical
    values are canonical floats.

    The Duffing model uses a truncated oscillator basis. The cosine model fits
    positive charging and Josephson energies to `f_01` and the anharmonicity,
    diagonalizes a finite Cooper-pair charge basis, and retains the lowest
    `dimension` energy eigenstates. Local-model selection does not itself
    change the exchange-coupling approximation or simulation frame.
    """

    model: Literal["duffing", "cosine"]
    charge_cutoff: int | None
    offset_charge: float

    def __init__(
        self,
        *,
        label: str,
        dimension: int,
        frequency: float | Frequency,
        anharmonicity: float | Frequency | None = None,
        model: Literal["duffing", "cosine"] = "duffing",
        charge_cutoff: int | None = None,
        offset_charge: float = 0.0,
        t1: float | Time | None = None,
        t2: float | Time | None = None,
        relaxation_rate: float | Frequency | None = None,
        dephasing_rate: float | Frequency | None = None,
    ) -> None:
        normalized_frequency = normalize_frequency_to_ghz(frequency)
        if anharmonicity is None:
            normalized_anharmonicity = -0.05 * normalized_frequency
        else:
            normalized_anharmonicity = normalize_frequency_to_ghz(anharmonicity)
        if model not in ("duffing", "cosine"):
            raise ValueError(f"Unsupported transmon model: {model}")
        if model == "cosine":
            if dimension < 2:
                raise ValueError("dimension must be at least 2 for a cosine transmon.")
            if normalized_frequency <= 0:
                raise ValueError("frequency must be positive for a cosine transmon.")
            if normalized_anharmonicity >= 0:
                raise ValueError(
                    "anharmonicity must be negative for a cosine transmon."
                )
            if charge_cutoff is None:
                charge_cutoff = 25
            if charge_cutoff < 1:
                raise ValueError("charge_cutoff must be at least 1.")
            if 2 * charge_cutoff + 1 < max(dimension, 3):
                raise ValueError(
                    "The charge basis must contain at least the retained levels and "
                    "three states for spectral fitting."
                )
        resolved_relaxation_rate, resolved_dephasing_rate = resolve_decoherence_rates(
            t1=t1,
            t2=t2,
            relaxation_rate=relaxation_rate,
            dephasing_rate=dephasing_rate,
        )
        super().__init__(
            label=label,
            dimension=dimension,
            frequency=normalized_frequency,
            anharmonicity=normalized_anharmonicity,
            relaxation_rate=resolved_relaxation_rate,
            dephasing_rate=resolved_dephasing_rate,
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "charge_cutoff", charge_cutoff)
        object.__setattr__(self, "offset_charge", float(offset_charge))

    @override
    def compile(self) -> CompiledObject:
        """
        Compile this transmon using its selected local Hamiltonian model.

        Returns
        -------
        CompiledObject
            Local `H / hbar` in rad/ns, interaction and lowering operators, and
            phenomenological collapse operators in the retained basis. A
            cosine model also includes its charge-basis eigensystem.

        Raises
        ------
        ValueError
            If cosine-model parameter fitting fails or the projected 0-1 charge
            matrix element is numerically zero.
        """
        from qxsimulator.system._compilation import (
            compile_cosine_transmon,
            compile_duffing_object,
        )

        if self.model == "cosine":
            return compile_cosine_transmon(self)
        return compile_duffing_object(self)
