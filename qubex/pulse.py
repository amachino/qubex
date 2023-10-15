from math import e
from typing import Final, Union

import numpy as np
import matplotlib.pyplot as plt


class Waveform:
    SAMPLING_PERIOD: Final[int] = 2  # ns

    """
    A class to represent a waveform.

    Attributes
    ----------
    values : np.ndarray[complex]
        A NumPy array of complex numbers representing the waveform.
    times : np.ndarray[int]
        Time array of the waveform in ns.
    """

    def __init__(
        self,
        values: Union[np.ndarray, list],
        time_offset=0,
        phase_offset=0.0,
    ):
        self._values = np.array(values, dtype=complex)
        self.time_offset = time_offset
        self.phase_offset = phase_offset

    @property
    def values(self):
        return self._values * np.exp(1j * self.phase_offset)

    @property
    def length(self):
        return len(self.values)

    @property
    def duration(self):
        return self.length * self.SAMPLING_PERIOD

    @property
    def times(self):
        return np.arange(self.length) * self.SAMPLING_PERIOD + self.time_offset

    @property
    def real(self):
        return np.real(self.values)

    @property
    def imag(self):
        return np.imag(self.values)

    @property
    def ampl(self):
        return np.abs(self.values)

    @property
    def phase(self):
        return np.angle(self.values)

    def _ns_to_samples(self, duration: int) -> int:
        """Converts a duration in ns to a length in samples."""
        if duration % self.SAMPLING_PERIOD != 0:
            raise ValueError(
                f"Duration must be a multiple of the sampling period ({self.SAMPLING_PERIOD} ns)."
            )
        return duration // self.SAMPLING_PERIOD

    def plot(self, polar=False, title=""):
        """Plots the pulse."""
        if polar:
            self.plot_polar(title)
        else:
            self.plot_xy(title)

    def plot_xy(self, title=""):
        _, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.set_xlabel("Time / ns")
        ax.set_ylabel("Amplitude / a.u.")
        ax.grid()

        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)

        real = np.append(self.real, self.real[-1])
        imag = np.append(self.imag, self.imag[-1])
        ax.step(times, real, label="X", where="post")
        ax.step(times, imag, label="Y", where="post")

        ax.legend()
        plt.show()

    def plot_polar(self, title=""):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
        fig.suptitle(title)
        ax[0].set_ylabel("Amplitude / a.u.")
        ax[1].set_ylabel("Phase / rad")
        ax[1].set_xlabel("Time / ns")
        ax[0].grid()
        ax[1].grid()

        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)

        ampl = np.append(self.ampl, self.ampl[-1])
        phase = np.append(self.phase, self.phase[-1])
        ax[0].step(times, ampl, where="post")
        ax[1].step(times, phase, where="post")

        plt.show()


class Sequence(Waveform):
    def __init__(self, waveforms: list[Waveform]):
        self.waveforms = waveforms
        if len(waveforms) == 0:
            values = np.array([])
        else:
            values = np.concatenate([w.values for w in waveforms])
        super().__init__(values)


class Blank(Waveform):
    def __init__(
        self,
        duration: int,
    ):
        length = self._ns_to_samples(duration)
        real = np.zeros(length, dtype=complex)
        imag = 0
        values = real + 1j * imag
        super().__init__(values)


class Rect(Waveform):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        risetime=0,
        time_offset=0,
        phase_offset=0.0,
    ):
        flattime = duration - 2 * risetime

        if flattime < 0:
            raise ValueError("Duration must be at least twice the rise time.")

        length_rise = self._ns_to_samples(risetime)
        length_flat = self._ns_to_samples(flattime)

        t_rise = np.linspace(0, risetime, length_rise)
        t_flat = np.linspace(0, flattime, length_flat)

        v_rise = 0.5 * amplitude * (1 - np.cos(np.pi * t_rise / risetime))
        v_flat = amplitude * np.ones_like(t_flat)
        v_fall = 0.5 * amplitude * (1 + np.cos(np.pi * t_rise / risetime))

        real = np.concatenate((v_rise, v_flat, v_fall))
        imag = 0
        values = real + 1j * imag
        super().__init__(values, time_offset=time_offset, phase_offset=phase_offset)


class Gauss(Waveform):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
        time_offset=0,
        phase_offset=0.0,
    ):
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")
        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = 0
        values = real + 1j * imag
        super().__init__(values, time_offset=time_offset, phase_offset=phase_offset)


class Drag(Waveform):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        anharmonicity: float,
        time_offset=0,
        phase_offset=0.0,
    ):
        if anharmonicity == 0:
            raise ValueError("Anharmonicity cannot be zero.")

        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
        sigma = duration * 0.5
        offset = -np.exp(-0.5)
        factor = amplitude / (1 + offset)
        real = factor * (np.exp(-((t - sigma) ** 2) / (2 * sigma**2)) + offset)
        imag = (
            (sigma - t)
            / (sigma**2)
            * (factor * (np.exp(-((t - sigma) ** 2) / (2 * sigma**2))))
        )
        values = real - 1j / (np.pi * anharmonicity) * imag
        super().__init__(values, time_offset=time_offset, phase_offset=phase_offset)


class DragGauss(Waveform):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
        anharmonicity: float,
        time_offset=0,
        phase_offset=0.0,
    ):
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")
        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = (mu - t) / (sigma**2) * real
        values = real - 1j / (np.pi * anharmonicity) * imag
        super().__init__(values, time_offset=time_offset, phase_offset=phase_offset)


class DragCos(Waveform):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        anharmonicity: float,
        time_offset=0,
        phase_offset=0.0,
    ):
        if anharmonicity == 0:
            raise ValueError("Anharmonicity cannot be zero.")

        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
        real = amplitude * (1.0 - np.cos(2 * np.pi * t / duration)) * 0.5
        imag = 2 * np.pi / duration * amplitude * np.sin(2 * np.pi * t / duration) * 0.5
        values = real - 1j / (np.pi * anharmonicity) * imag
        super().__init__(values, time_offset=time_offset, phase_offset=phase_offset)


class CPMG(Sequence):
    def __init__(self, tau: int, pi: Waveform, n=1):
        if tau % (2 * self.SAMPLING_PERIOD) != 0:
            raise ValueError(
                f"Tau must be a multiple of twice the sampling period ({2 * self.SAMPLING_PERIOD} ns)."
            )
        self.tau = tau
        self.pi = pi
        self.n = n

        sequence: list[Waveform] = [Blank(tau // 2)]
        for _ in range(n - 1):
            sequence += [pi, Blank(tau)]
        sequence += [pi, Blank(tau // 2)]
        super().__init__(sequence)


class TabuchiDD(Waveform):
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

    def __init__(self, duration: int, scale=1.0, beta=0.0, phi=0.0):
        length = self._ns_to_samples(duration)
        self.t = np.linspace(0, duration, length)
        self.T = duration
        values = np.array([])
        if duration != 0:
            self.vx_n = np.array(self.vx_n_T_over_pi) * np.pi / duration
            self.vy_n = np.array(self.vy_n_T_over_pi) * np.pi / duration
            values = self._calc_values(scale, beta, phi)
        super().__init__(values)

    def _calc_values(self, scale: float, beta: float, phi: float) -> np.ndarray:
        error_x = phi + np.tan(beta * np.pi / 180)
        x = (1 + error_x) * np.array([self.vx(t) for t in self.t])

        error_y = phi
        y = (1 + error_y) * np.array([self.vy(t) for t in self.t])

        values = scale * (x + 1j * y) / np.pi / 2 * 1e2

        return values

    def vx(self, t) -> float:
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vx_n, 1)
        )

    def vy(self, t) -> float:
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vy_n, 1)
        )
