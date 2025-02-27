from __future__ import annotations

from ..pulse import Blank
from ..pulse_array import PulseArray
from ..waveform import Waveform


class CPMG(PulseArray):
    """
    A class representing the CPMG pulse sequence used in quantum experiments.

    Parameters
    ----------
    tau : float
        The half of the inter-pulse spacing in nanoseconds.
    pi : Waveform
        The pi pulse waveform.
    n : int, optional
        The number of pi pulses in the sequence.
    **kwargs
        Additional keyword arguments passed to the PulseSequence constructor.

    Attributes
    ----------
    tau : int
        The half of the inter-pulse spacing in nanoseconds.
    pi : Waveform
        The pi pulse waveform used in the sequence.
    n : int
        The number of pi pulses in the sequence.

    Raises
    ------
    ValueError
        If `tau` is not a multiple of the sampling period.
    """

    def __init__(
        self,
        tau: float,
        pi: Waveform,
        n: int = 2,
        alternating: bool = False,
        **kwargs,
    ):
        if tau % self.SAMPLING_PERIOD != 0:
            raise ValueError(
                f"Tau must be a multiple of the sampling period ({self.SAMPLING_PERIOD} ns)."
            )
        if n < 1:
            raise ValueError("The number of pi pulses must be greater than 0.")
        waveforms: list[Waveform] = []
        if tau > 0:
            self.tau = tau
            self.pi = pi
            self.n = n
            waveforms = []
            for i in range(n):
                if alternating and i % 2 == 1:
                    waveforms += [Blank(tau), pi.scaled(-1), Blank(tau)]
                else:
                    waveforms += [Blank(tau), pi, Blank(tau)]
        super().__init__(waveforms, **kwargs)
