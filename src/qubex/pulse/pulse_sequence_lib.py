import numpy as np

from .pulse import Pulse
from .pulse_lib import Blank
from .pulse_sequence import PulseSequence
from .waveform import Waveform


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


class TabuchiDD(Pulse):
    """
    Class representing the Tabuchi Dynamical Decoupling pulse sequence.

    Parameters
    ----------
    duration : int
        The total duration of the pulse sequence in nanoseconds.
    beta : float, optional
        Beta parameter influencing the x and y components of the pulse.
    phi : float, optional
        Phi parameter influencing the x component of the pulse.
    **kwargs
        Additional keyword arguments passed to the Pulse constructor.

    Attributes
    ----------
    vx_n_T_over_pi : list[float]
        Coefficients for the x component of the pulse.
    vy_n_T_over_pi : list[float]
        Coefficients for the y component of the pulse.
    t : np.ndarray
        Time points for the pulse sequence.
    T : int
        Total duration of the pulse sequence in nanoseconds.
    vx_n : np.ndarray
        Scaled coefficients for the x component.
    vy_n : np.ndarray
        Scaled coefficients for the y component.

    Notes
    -----
    Y. Tabuchi, M. Negoro, and M. Kitagawa, “Design method of dynamical
    decouplingsequences integrated with optimal control theory,” Phys.
    Rev. A, vol.96, p.022331, Aug. 2017.
    """

    vx_n_T_over_pi = [
        -0.7030256,
        3.3281747,
        11.390077,
        2.9375301,
        -1.8758792,
        1.7478474,
        5.6966577,
        -0.5452435,
        4.0826786,
    ]

    vy_n_T_over_pi = [
        -3.6201768,
        3.8753985,
        -1.2311919,
        -0.2998110,
        3.1170274,
        0.3956137,
        -0.3593987,
        -3.5266063,
        2.4900307,
    ]

    def __init__(
        self,
        duration: int,
        beta=0.0,
        phi=0.0,
        **kwargs,
    ):
        self.t = self._sampling_points(duration)
        self.T = duration  # [ns]
        values = np.array([])  # [MHz]
        if duration != 0:
            self.vx_n = np.array(self.vx_n_T_over_pi) * np.pi / duration
            self.vy_n = np.array(self.vy_n_T_over_pi) * np.pi / duration
            values = self._calc_values(beta, phi)
        super().__init__(values, **kwargs)

    def _calc_values(self, beta: float, phi: float) -> np.ndarray:
        error_x = beta + np.tan(phi * np.pi / 180)
        x = (1 + error_x) * np.array([self._vx(t) for t in self.t])

        error_y = beta
        y = (1 + error_y) * np.array([self._vy(t) for t in self.t])

        return x + 1j * y

    def _vx(self, t) -> float:
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vx_n, 1)
        )

    def _vy(self, t) -> float:
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vy_n, 1)
        )
