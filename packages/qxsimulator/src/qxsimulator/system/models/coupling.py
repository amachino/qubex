"""Lightweight coupling specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from qxcore import Frequency

from ._normalization import normalize_frequency_to_ghz
from .object import Object


@dataclass(frozen=True)
class Coupling:
    """
    Define an adjacent-transition exchange strength for an object pair.

    Parameters
    ----------
    pair : tuple[str, str] | tuple[Object, Object]
        Object labels or object instances defining the coupled pair. Object
        instances are stored by label, while pair orientation is preserved.
    strength : float | Frequency
        Cyclic coupling frequency. Bare floats are interpreted as GHz.

    Raises
    ------
    ValueError
        If `pair` does not contain exactly two elements.

    Notes
    -----
    The derived `label` property joins the normalized labels as `a-b`.
    `QuantumSystem` converts `strength` to rad/ns with `2 * pi` and constructs
    an exchange Hamiltonian from the compiled adjacent lowering and raising
    operators. This coupling rotating-wave approximation is independent of the
    local object model.
    """

    pair: tuple[str, str]
    strength: float

    def __init__(
        self,
        *,
        pair: tuple[str, str] | tuple[Object, Object],
        strength: float | Frequency,
    ) -> None:
        if len(pair) != 2:
            raise ValueError("Coupling pair must have exactly two elements.")
        normalized_pair: Final[tuple[str, str]] = (
            pair[0].label if isinstance(pair[0], Object) else pair[0],
            pair[1].label if isinstance(pair[1], Object) else pair[1],
        )
        object.__setattr__(self, "pair", normalized_pair)
        object.__setattr__(self, "strength", normalize_frequency_to_ghz(strength))

    @property
    def label(self) -> str:
        """Return the stored endpoint labels joined as `a-b`."""
        return f"{self.pair[0]}-{self.pair[1]}"
