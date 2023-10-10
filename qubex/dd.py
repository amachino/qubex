"""Class for generating Tabuchi pulse."""

import numpy as np
import matplotlib.pyplot as plt

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


class TabuchiPulse:
    r"""Class for generating Tabuchi pulse.

    The Tabuchi pulse is defined as
    \[ v_x(t) = \sum_{n=1}^{N} v_{x,n} \sin\left(\frac{2 \pi n t}{T}\right) \]
    \[ v_y(t) = \sum_{n=1}^{N} v_{y,n} \sin\left(\frac{2 \pi n t}{T}\right) \]
    where \(v_{x,n}\) and \(v_{y,n}\) are pre-defined coefficients.

    Parameters
    ----------
    duration : float, optional
        Duration of the pulse in seconds. Default is 1.
    step : float, optional
        Step of the pulse in seconds. Default is 0.0001.

    Methods
    -------
    duration
        Returns the duration of the pulse.
    step
        Returns the step of the pulse.
    values
        Returns an array of pulse values for t in [0, T].
    plot_pulse
        Plots the pulse.
    """

    def __init__(self, duration=1.0, step=0.0001, error_beta=0.0, error_phi=0.0):
        self.T = duration
        self.dT = step
        self.t = np.arange(0, duration, step)
        self.e_beta = error_beta
        self.e_phi = error_phi
        self.vx_n = np.array(vx_n_T_over_pi) * np.pi / duration
        self.vy_n = np.array(vy_n_T_over_pi) * np.pi / duration

    @property
    def duration(self) -> float:
        """Returns the duration of the pulse."""
        return self.T

    @property
    def step(self) -> float:
        """Returns the step of the pulse."""
        return self.dT

    @property
    def length(self) -> float:
        """Returns the step of the pulse."""
        return len(self.t)

    @property
    def values(self) -> np.ndarray:
        """Returns an array of pulse values for t in [0, T]."""
        return self.v_array()

    def vx_value(self, t) -> float:
        r"""Returns the value of vx(t) at time t.

        \[ v_x(t) = \sum_{n=1}^{N} v_{x,n} \sin\left(\frac{2 \pi n t}{T}\right) \]
        """
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vx_n, 1)
        )

    def vy_value(self, t) -> float:
        r"""Returns the value of vy(t) at time t.

        \[ v_y(t) = \sum_{n=1}^{N} v_{y,n} \sin\left(\frac{2 \pi n t}{T}\right) \]
        """
        return sum(
            v * np.sin(2 * np.pi * n * t / self.T) for n, v in enumerate(self.vy_n, 1)
        )

    def vx_array(self) -> np.ndarray:
        r"""Returns an array of vx(t) for t in [0, T].

        \[ v_x(t) = \sum_{n=1}^{N} v_{x,n} \sin\left(\frac{2 \pi n t}{T}\right) \]
        """
        error = self.e_beta + np.tan(self.e_phi * np.pi / 180)
        return (1 + error) * np.array([self.vx_value(t) for t in self.t])

    def vy_array(self) -> np.ndarray:
        r"""Returns an array of vy(t) for t in [0, T].

        \[ v_y(t) = \sum_{n=1}^{N} v_{y,n} \sin\left(\frac{2 \pi n t}{T}\right) \]
        """
        error = self.e_beta
        return (1 + error) * np.array([self.vy_value(t) for t in self.t])

    def v_array(self) -> np.ndarray:
        r"""Returns a complex array of v(t) for t in [0, T].

        \[ v(t) = v_x(t) + i v_y(t) \]
        """
        return self.vx_array() + 1j * self.vy_array()

    def va_array(self) -> np.ndarray:
        r"""Returns an amplitude array of v(t) for t in [0, T].

        \[ |v(t)| = \sqrt{v_x(t)^2 + v_y(t)^2} \]
        """
        return np.abs(self.v_array())

    def vp_array(self) -> np.ndarray:
        r"""Returns a phase array of v(t) for t in [0, T].

        \[ arg(v(t)) = \arctan\left(\frac{v_y(t)}{v_x(t)}\right) \]
        """
        return np.angle(self.v_array())

    # Plotting methods

    def plot_pulse(self, polar=False):
        """Plots the pulse."""
        if polar:
            self.plot_pulse_polar()
        else:
            self.plot_pulse_xy()

    def plot_pulse_xy(self):
        """Plots vx(t) and vy(t) for t in [0, T]."""
        time = self.t / self.T
        vx_t_T_over_2pi = self.vx_array() * self.T / (2 * np.pi)
        vy_t_T_over_2pi = self.vy_array() * self.T / (2 * np.pi)

        plt.figure(figsize=(12, 5))
        plt.plot(time, vx_t_T_over_2pi, label=r"$v_x(t) T / 2 \pi$")
        plt.plot(time, vy_t_T_over_2pi, label=r"$v_y(t) T / 2 \pi$")
        plt.xlabel(r"Time [$t/T$]")
        plt.ylabel(r"$v_\alpha(t) T / 2 \pi$ [Hz]")
        plt.legend()
        plt.title("Generated DD")
        plt.show()

    def plot_pulse_polar(self):
        """Plots |v(t)| and arg(v(t)) for t in [0, T]."""
        time = self.t / self.T
        va_t = self.va_array() / (2 * np.pi)
        vp_t = self.vp_array()

        plt.figure(figsize=(12, 6))

        # Plotting the amplitude
        plt.subplot(2, 1, 1)
        plt.plot(time, va_t, label=r"$|v(t)| / 2\pi$")
        plt.xlabel(r"Time [$t/T$]")
        plt.ylabel(r"$|v(t)| / 2\pi$ [Hz]")
        plt.legend()
        plt.title("Amplitude of Pulse")

        # Plotting the phase
        plt.subplot(2, 1, 2)
        plt.plot(time, vp_t, label=r"arg$(v(t))$")
        plt.xlabel(r"Time [$t/T$]")
        plt.ylabel(r"arg$(v(t))$ [rad]")
        plt.legend()
        plt.title("Phase of Pulse")

        plt.tight_layout()
        plt.show()

    def plot_fourier_coefficients(self):
        """Plots the Fourier coefficients |vx_n| and |vy_n|."""
        n = np.arange(1, len(self.vx_n) + 1)
        vx_n_over_2pi = np.abs(self.vx_n) / (2 * np.pi)
        vy_n_over_2pi = np.abs(self.vy_n) / (2 * np.pi)
        max_value = max(np.max(vx_n_over_2pi), np.max(vy_n_over_2pi))

        plt.figure(figsize=(15, 6))

        # Plotting the vx_n coefficients
        plt.subplot(1, 2, 1)
        plt.bar(n, vx_n_over_2pi, label=r"$|v_{x,n}|/2\pi$")
        plt.ylim(0, max_value)
        plt.xlabel(r"Fourier Order [$n$]")
        plt.ylabel(r"$|v_{x,n}|/2\pi$ [Hz]")
        plt.title(r"Coefficients of $v_x(t)$")
        plt.legend()

        # Plotting the vy_n coefficients
        plt.subplot(1, 2, 2)
        plt.bar(n, vy_n_over_2pi, label=r"$|v_{y,n}|/2\pi$")
        plt.ylim(0, max_value)
        plt.xlabel(r"Fourier Order [$n$]")
        plt.ylabel(r"$|v_{y,n}|/2\pi$ [Hz]")
        plt.title(r"Coefficients of $v_y(t)$")
        plt.legend()

        plt.tight_layout()
        plt.show()
