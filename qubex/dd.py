import numpy as np

from .pulse import Blank, Pulse, PulseSequence, Waveform


class CPMG(PulseSequence):
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
        length = self._ns_to_samples(duration)
        self.t = np.linspace(0, duration, length)
        self.T = duration  # [ns]
        values = np.array([])  # [MHz]
        if duration != 0:
            self.vx_n = np.array(self.vx_n_T_over_pi) * np.pi / duration
            self.vy_n = np.array(self.vy_n_T_over_pi) * np.pi / duration
            values = self._calc_values(beta, phi)
        super().__init__(values, **kwargs)

    def _calc_values(self, beta: float, phi: float) -> np.ndarray:
        error_x = beta + np.tan(phi * np.pi / 180)
        x = (1 + error_x) * np.array([self.vx(t) for t in self.t])

        error_y = beta
        y = (1 + error_y) * np.array([self.vy(t) for t in self.t])

        values = (x + 1j * y) / np.pi / 2 * 1e3
        return values

    def vx(self, t) -> float:
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vx_n, 1)
        )

    def vy(self, t) -> float:
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vy_n, 1)
        )
