from ..pulse import Blank
from ..pulse_sequence import PulseSequence
from ..waveform import Waveform


class CPMG(PulseSequence):
    """
    A class representing the CPMG pulse sequence used in quantum experiments.

    Parameters
    ----------
    tau : int
        The inter-pulse spacing in nanoseconds.
    pi : Waveform
        The pi pulse waveform.
    n : int, optional
        The number of pi pulses in the sequence.
    **kwargs
        Additional keyword arguments passed to the PulseSequence constructor.

    Attributes
    ----------
    tau : int
        The inter-pulse spacing used in the sequence.
    pi : Waveform
        The pi pulse waveform used in the sequence.
    n : int
        The number of pi pulses in the sequence.

    Raises
    ------
    ValueError
        If `tau` is not a multiple of twice the sampling period.

    Notes
    -----
    The CPMG sequence typically consists of a series of pi pulses separated
    by a delay `tau`. It's often used in NMR and quantum computing for
    refocusing and decoherence studies.
    """

    def __init__(
        self,
        tau: int,
        pi: Waveform,
        n: int = 2,
        **kwargs,
    ):
        if tau % (2 * self.SAMPLING_PERIOD) != 0:
            raise ValueError(
                f"Tau must be a multiple of twice the sampling period ({2 * self.SAMPLING_PERIOD} ns)."
            )
        waveforms: list[Waveform] = []
        if tau > 0:
            self.tau = tau
            self.pi = pi
            self.n = n
            waveforms = [Blank(tau // 2)]
            for _ in range(n - 1):
                waveforms += [pi, Blank(tau)]
            waveforms += [pi, Blank(tau // 2)]
        super().__init__(waveforms, **kwargs)
