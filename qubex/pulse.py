from abc import ABC, abstractmethod
from typing import Final, Optional, Sequence
from copy import deepcopy

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Waveform(ABC):
    SAMPLING_PERIOD: Final[int] = 2  # ns

    """
    An abstract base class to represent a waveform.

    Attributes
    ----------
    values : npt.NDArray[np.complex128]
        A NumPy array of complex numbers representing the waveform.
    times : npt.NDArray[np.int64]
        Time array of the waveform in ns.
    """

    def __init__(
        self,
        scale: float = 1.0,
        time_offset: int = 0,
        phase_offset: float = 0.0,
    ):
        self.scale = scale
        self.time_offset = time_offset
        self.phase_offset = phase_offset

    @property
    @abstractmethod
    def values(self) -> npt.NDArray[np.complex128]:
        pass

    @property
    def length(self) -> int:
        return len(self.values)

    @property
    def duration(self) -> int:
        return self.length * self.SAMPLING_PERIOD

    @property
    def times(self) -> npt.NDArray[np.int64]:
        return np.arange(self.length) * self.SAMPLING_PERIOD + self.time_offset

    @property
    def real(self) -> npt.NDArray[np.float64]:
        return np.real(self.values)

    @property
    def imag(self) -> npt.NDArray[np.float64]:
        return np.imag(self.values)

    @property
    def ampl(self) -> npt.NDArray[np.float64]:
        return np.abs(self.values)

    @property
    def phase(self) -> npt.NDArray[np.float64]:
        return np.angle(self.values)

    @abstractmethod
    def copy(self) -> "Waveform":
        pass

    @abstractmethod
    def scaled(self, scale: float) -> "Waveform":
        pass

    @abstractmethod
    def shifted(self, phase: float) -> "Waveform":
        pass

    @abstractmethod
    def inverted(self) -> "Waveform":
        pass

    @abstractmethod
    def repeated(self, n: int) -> "Waveform":
        pass

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
        _, ax = plt.subplots(figsize=(6, 2))
        ax.set_title(title)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (arb. units)")
        ax.grid()

        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)

        real = np.append(self.real, self.real[-1])
        imag = np.append(self.imag, self.imag[-1])
        ax.step(times, real, label="X", where="post")
        ax.step(times, imag, label="Y", where="post")

        ax.legend()
        plt.show()

    def plot_polar(self, title=""):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 2))
        fig.suptitle(title)
        ax[0].set_ylabel("Amplitude (arb. units)")
        ax[1].set_ylabel("Phase (rad)")
        ax[1].set_xlabel("Time (ns)")
        ax[0].grid()
        ax[1].grid()

        times = np.append(self.times, self.times[-1] + self.SAMPLING_PERIOD)

        ampl = np.append(self.ampl, self.ampl[-1])
        phase = np.append(self.phase, self.phase[-1])
        ax[0].step(times, ampl, where="post")
        ax[1].step(times, phase, where="post")

        plt.show()


class Pulse(Waveform):
    """
    A class to represent a pulse.

    Attributes
    ----------
    values : npt.NDArray[np.complex128]
        A NumPy array of complex numbers representing the pulse.
    times : npt.NDArray[np.int64]
        Time array of the pulse in ns.
    """

    def __init__(
        self,
        values: npt.ArrayLike,
        scale: float = 1.0,
        time_offset: int = 0,
        phase_offset: float = 0.0,
    ):
        self._values = np.array(values)
        self.scale = scale
        self.time_offset = time_offset
        self.phase_offset = phase_offset

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        return self._values * self.scale * np.exp(1j * self.phase_offset)

    def copy(self) -> "Pulse":
        """Returns a copy of the pulse."""
        return deepcopy(self)

    def scaled(self, scale: float) -> "Pulse":
        """Returns a copy of the pulse scaled by the given factor."""
        new_pulse = deepcopy(self)
        new_pulse.scale *= scale
        return new_pulse

    def shifted(self, phase: float) -> "Pulse":
        """Returns a copy of the pulse shifted by the given phase."""
        new_pulse = deepcopy(self)
        new_pulse.phase_offset += phase
        return new_pulse

    def inverted(self) -> "Pulse":
        """Returns a copy of the pulse inverted."""
        new_pulse = deepcopy(self)
        new_pulse.phase_offset += np.pi
        return new_pulse

    def repeated(self, n: int) -> "Pulse":
        """Returns a copy of the pulse repeated n times."""
        new_pulse = deepcopy(self)
        new_pulse._values = np.tile(new_pulse._values, n)
        return new_pulse


class PulseSequence(Waveform):
    def __init__(
        self,
        waveforms: Optional[Sequence[Waveform]] = None,
        scale: float = 1.0,
        time_offset: int = 0,
        phase_offset: float = 0.0,
    ):
        if waveforms is None:
            waveforms = []
        self.waveforms = waveforms
        self.scale = scale
        self.time_offset = time_offset
        self.phase_offset = phase_offset

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the concatenated values of the pulse sequence."""
        if len(self.waveforms) == 0:
            return np.array([])
        concat_values = np.concatenate([w.values for w in self.waveforms])
        values = concat_values * self.scale * np.exp(1j * self.phase_offset)
        return values

    def copy(self) -> "PulseSequence":
        """Returns a copy of the pulse sequence."""
        return deepcopy(self)

    def scaled(self, scale: float) -> "PulseSequence":
        """Returns a copy of the pulse sequence scaled by the given factor."""
        new_sequence = deepcopy(self)
        new_sequence.scale *= scale
        return new_sequence

    def shifted(self, phase: float) -> "PulseSequence":
        """Returns a copy of the pulse sequence shifted by the given phase."""
        new_sequence = deepcopy(self)
        new_sequence.phase_offset += phase
        return new_sequence

    def inverted(self) -> "PulseSequence":
        """Returns a copy of the pulse sequence inverted."""
        new_sequence = deepcopy(self)
        new_sequence.phase_offset += np.pi
        return new_sequence

    def repeated(self, n: int) -> "PulseSequence":
        """Returns a copy of the pulse sequence repeated n times."""
        new_sequence = deepcopy(self)
        new_sequence.waveforms = list(new_sequence.waveforms) * n
        return new_sequence

    def reversed(self) -> "PulseSequence":
        """Returns a copy of the pulse sequence with the order of the waveforms reversed."""
        new_sequence = deepcopy(self)
        new_sequence.waveforms = list(reversed(new_sequence.waveforms))
        return new_sequence


class Blank(Pulse):
    def __init__(
        self,
        duration: int,
    ):
        length = self._ns_to_samples(duration)
        real = np.zeros(length, dtype=complex)
        imag = 0
        values = real + 1j * imag
        super().__init__(values)


class Rect(Pulse):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        risetime: int = 0,
        **kwargs,
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, risetime)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: int,
        amplitude: float,
        risetime: int,
    ) -> npt.NDArray[np.complex128]:
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

        values = np.concatenate((v_rise, v_flat, v_fall), dtype=np.complex128)

        return values


class Gauss(Pulse):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
        **kwargs,
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, sigma)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
    ) -> npt.NDArray[np.complex128]:
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = 0
        values = real + 1j * imag
        return values


class Drag(Pulse):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        anharmonicity: float,
        **kwargs,
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, anharmonicity)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: int,
        amplitude: float,
        anharmonicity: float,
    ) -> npt.NDArray[np.complex128]:
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
        values = real - 1j / (np.pi * anharmonicity * 1e-9) * imag
        return values


class DragGauss(Pulse):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
        anharmonicity: float,
        **kwargs,
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, sigma, anharmonicity)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
        anharmonicity: float,
    ) -> npt.NDArray[np.complex128]:
        if sigma == 0:
            raise ValueError("Sigma cannot be zero.")

        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
        mu = duration * 0.5
        real = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        imag = (mu - t) / (sigma**2) * real
        values = real - 1j / (np.pi * anharmonicity * 1e-9) * imag
        return values


class DragCos(Pulse):
    def __init__(
        self,
        duration: int,
        amplitude: float,
        anharmonicity: float,
        **kwargs,
    ):
        values = np.array([])
        if duration != 0:
            values = self._calc_values(duration, amplitude, anharmonicity)
        super().__init__(values, **kwargs)

    def _calc_values(
        self,
        duration: int,
        amplitude: float,
        anharmonicity: float,
    ) -> npt.NDArray[np.complex128]:
        if anharmonicity == 0:
            raise ValueError("Anharmonicity cannot be zero.")

        length = self._ns_to_samples(duration)
        t = np.linspace(0, duration, length)
        real = amplitude * (1.0 - np.cos(2 * np.pi * t / duration)) * 0.5
        imag = 2 * np.pi / duration * amplitude * np.sin(2 * np.pi * t / duration) * 0.5
        values = real - 1j / (np.pi * anharmonicity * 1e-9) * imag
        return values


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
