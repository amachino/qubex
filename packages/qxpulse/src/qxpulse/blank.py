"""Blank pulse definition."""

from __future__ import annotations

import numpy as np
from typing_extensions import override

from .pulse import Pulse


class Blank(Pulse):
    """
    A class to represent a blank pulse.

    Parameters
    ----------
    duration : float
        Duration of the blank pulse in ns.

    Examples
    --------
    >>> pulse = Blank(duration=100)
    """

    def __init__(
        self,
        duration: float,
        **kwargs,
    ):
        super().__init__(
            duration=duration,
            **kwargs,
        )
        self._finalize_initialization()

    @override
    def _sample_values(self) -> np.ndarray:
        """Return sampled values for the blank pulse."""
        return np.zeros(self.length, dtype=np.complex128)
