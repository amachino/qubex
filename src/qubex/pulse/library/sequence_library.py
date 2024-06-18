from __future__ import annotations

from ..pulse import Blank
from ..pulse_sequence import PulseSequence
from ..waveform import Waveform


class CPMG(PulseSequence):
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
            waveforms = [Blank(tau)]
            for _ in range(n - 1):
                waveforms += [pi, Blank(2 * tau)]
            waveforms += [pi, Blank(tau)]
        super().__init__(waveforms, **kwargs)
