from __future__ import annotations

from copy import deepcopy
from typing import Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import deprecated

from .waveform import Waveform


class Pulse(Waveform):
    """
    A class to represent a pulse.

    Parameters
    ----------
    values : ArrayLike
        I/Q values of the pulse.
    scale : float, optional
        Scaling factor of the pulse.
    detuning : float, optional
        Detuning of the pulse in GHz.
    phase : float, optional
        Phase shift of the pulse in rad.
    """

    def __init__(
        self,
        values: npt.ArrayLike,
        *,
        scale: float | None = None,
        detuning: float | None = None,
        phase: float | None = None,
        **kwargs,
    ):
        super().__init__(
            scale=scale,
            detuning=detuning,
            phase=phase,
            **kwargs,
        )
        self._values = np.array(values, dtype=np.complex128)

    def __repr__(self) -> str:
        return f"{self.name}({self.length})"

    @staticmethod
    def func(
        t: npt.ArrayLike,
        **kwargs,
    ) -> npt.NDArray[np.complex128]:
        """
        Envelope function of the pulse.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        kwargs : dict
            Additional parameters for the pulse function.

        Returns
        -------
        NDArray
            I/Q values of the pulse.
        """
        raise NotImplementedError(
            "The func method must be implemented in the subclass."
        )

    @property
    def length(self) -> int:
        """Returns the length of the pulse in samples."""
        return len(self._values)

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the I/Q values of the pulse."""
        return (
            self._values
            * self._scale
            * np.exp(-1j * (2 * np.pi * self._detuning * self.times - self._phase))
        )

    def copy(self, reset_cached_duration: bool = False) -> Pulse:
        """Returns a copy of the pulse."""
        pulse = deepcopy(self)
        if reset_cached_duration:
            pulse.reset_cached_duration()
        return pulse

    def padded(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> Pulse:
        """
        Returns a copy of the pulse with zero padding.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        N = self._number_of_samples(total_duration)
        if pad_side == "right":
            values = np.pad(self._values, (0, N - self.length), mode="constant")
        elif pad_side == "left":
            values = np.pad(self._values, (N - self.length, 0), mode="constant")
        else:
            raise ValueError("pad_side must be either 'right' or 'left'.")
        new_pulse = self.copy(reset_cached_duration=True)
        new_pulse._values = values
        return new_pulse

    def scaled(self, scale: float) -> Pulse:
        """Returns a copy of the pulse scaled by the given factor."""
        if scale == 1:
            return self
        new_pulse = self.copy()
        new_pulse._scale *= scale
        return new_pulse

    def detuned(self, detuning: float) -> Pulse:
        """Returns a copy of the pulse detuned by the given frequency."""
        if detuning == 0:
            return self
        new_pulse = self.copy()
        new_pulse._detuning += detuning
        return new_pulse

    def shifted(self, phase: float) -> Pulse:
        """Returns a copy of the pulse shifted by the given phase."""
        if phase == 0:
            return self
        new_pulse = self.copy()
        new_pulse._phase += phase
        return new_pulse

    def repeated(self, n: int) -> Pulse:
        """Returns a copy of the pulse repeated n times."""
        if n == 1:
            return self
        new_pulse = self.copy(reset_cached_duration=True)
        new_pulse._values = np.tile(self._values, n)
        return new_pulse

    @deprecated(
        "The `reversed` method is deprecated, use `inverted` instead.",
    )
    def reversed(self) -> Pulse:
        return self.inverted()

    def inverted(self) -> Pulse:
        """Returns a copy of the pulse with the time inverted."""
        new_pulse = self.copy()
        new_pulse._values = np.flip(-1 * self._values)
        return new_pulse
