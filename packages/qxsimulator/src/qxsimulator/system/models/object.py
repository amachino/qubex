"""Lightweight quantum-system object specifications."""

from __future__ import annotations

from dataclasses import dataclass

from qxsimulator.system.compiled_object import CompiledObject


@dataclass(frozen=True)
class Object:
    """
    Store canonical parameters shared by local quantum-system objects.

    Attributes
    ----------
    label : str
        Unique object label within a quantum system.
    dimension : int
        Truncated Hilbert-space dimension.
    frequency : float
        Fundamental cyclic transition frequency in GHz.
    anharmonicity : float
        Difference between consecutive cyclic transition frequencies in GHz.
    relaxation_rate : float
        Energy-relaxation rate in inverse ns.
    dephasing_rate : float
        Physical pure-dephasing rate in inverse ns.

    Notes
    -----
    This base representation stores canonical floats for simulator use.
    Hamiltonian construction converts cyclic frequencies to angular frequencies
    with `2 * pi`; decay rates remain in inverse ns.
    """

    label: str
    dimension: int
    frequency: float
    anharmonicity: float
    relaxation_rate: float
    dephasing_rate: float

    def compile(self) -> CompiledObject:
        """
        Compile this specification as a truncated Duffing oscillator.

        Returns
        -------
        CompiledObject
            Local `H / hbar` in rad/ns, interaction and lowering operators, and
            phenomenological collapse operators in the retained basis.
        """
        from qxsimulator.system._compilation import compile_duffing_object

        return compile_duffing_object(self)
