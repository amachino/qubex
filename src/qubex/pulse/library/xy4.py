from __future__ import annotations

from typing import Final

from ..blank import Blank
from ..pulse_array import PulseArray
from ..waveform import Waveform


class XY4(PulseArray):
    """
    A class representing the XY4 dynamical decoupling pulse sequence.

    Parameters
    ----------
    tau : float
        The half of the inter-pulse spacing in nanoseconds.
    pi_x : Waveform
        The pi pulse waveform for X rotation.
    pi_y : Waveform
        The pi pulse waveform for Y rotation.
    n : int, optional
        The number of XY4 cycles (each cycle is 4 pulses: X, Y, X, Y).
    **kwargs
        Additional keyword arguments passed to the PulseArray constructor.

    Attributes
    ----------
    tau : int
        The half of the inter-pulse spacing in nanoseconds.
    pi_x : Waveform
        The pi pulse waveform for X rotation.
    pi_y : Waveform
        The pi pulse waveform for Y rotation.
    n : int
        The number of XY4 cycles.

    Raises
    ------
    ValueError
        If `tau` is not a multiple of the sampling period.
    """

    def __init__(
        self,
        *,
        tau: float,
        pi_x: Waveform,
        pi_y: Waveform,
        n: int = 1,
        **kwargs,
    ):
        self.tau: Final = tau
        self.pi_x: Final = pi_x
        self.pi_y: Final = pi_y
        self.n: Final = n

        if tau % self.SAMPLING_PERIOD != 0:
            raise ValueError(
                f"Tau must be a multiple of the sampling period ({self.SAMPLING_PERIOD} ns)."
            )
        if n < 1:
            raise ValueError("The number of XY4 cycles must be greater than 0.")

        waveforms: list[Waveform] = []
        if tau > 0:
            for i in range(n):
                # XY4 cycle: X, Y, X, Y
                waveforms += [
                    Blank(tau),
                    pi_x,
                    Blank(tau),
                    Blank(tau),
                    pi_y,
                    Blank(tau),
                    Blank(tau),
                    pi_x,
                    Blank(tau),
                    Blank(tau),
                    pi_y,
                    Blank(tau),
                ]
        super().__init__(waveforms, **kwargs)
