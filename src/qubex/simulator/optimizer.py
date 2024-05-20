from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import qutip as qt  # type: ignore
from scipy.optimize import OptimizeResult, minimize  # type: ignore

from .system import System


@dataclass
class OptimizationResult:
    result: OptimizeResult
    times: npt.NDArray[np.float64]
    params: npt.NDArray[np.float64]
    pulse: npt.NDArray[np.complex128]

    def plot(self):
        pulse = np.append(self.pulse, self.pulse[-1])
        pulse = pulse / (2 * np.pi) * 1e3
        plt.step(self.times, pulse.real, where="post", label="I")
        plt.step(self.times, pulse.imag, where="post", label="Q")
        plt.grid(color="gray", linestyle="--", alpha=0.2)
        plt.title("Pulse waveform")
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude (MHz)")
        plt.legend()
        plt.show()


class PulseOptimizer:
    def __init__(
        self,
        system: System,
        target: str,
        segment_count: int,
        segment_width: float,
        max_rabi_rate: float,
        target_unitary: qt.Qobj,
    ):
        self.system = system
        self.target = target
        self.dimension = system.hamiltonian.shape[0]
        self.segment_count = segment_count
        self.segment_width = segment_width
        self.duration = segment_count * segment_width
        self.max_amplitude = 2 * np.pi * max_rabi_rate
        self.target_unitary = target_unitary

    def pwc_unitary(self, value: complex, duration: float) -> qt.Qobj:
        H_sys = self.system.hamiltonian
        a = self.system.lowering_operator(self.target)
        ad = a.dag()
        H_ctrl = 0.5 * (ad * value + a * np.conj(value))
        H = H_sys + H_ctrl
        U = (-1j * H * duration).expm()
        return U

    def objective_function(self, params: npt.NDArray[np.float64]) -> float:
        pulse = self.params_to_pulse(params)
        U = self.system.identity
        for value in pulse:
            U = self.pwc_unitary(value, self.segment_width) * U
        return self.unitary_infidelity(U, self.target_unitary)

    def unitary_infidelity(self, U1: qt.Qobj, U2: qt.Qobj) -> float:
        return 1 - np.abs((U1.dag() * U2).tr() / self.dimension) ** 2

    def random_params(self) -> npt.NDArray[np.float64]:
        return np.random.uniform(
            -self.max_amplitude,
            self.max_amplitude,
            2 * self.segment_count,
        )

    def params_to_pulse(
        self,
        params: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        params = params.reshape((self.segment_count, 2))
        return params[:, 0] + 1j * params[:, 1]

    def optimize(
        self,
        initial_params: npt.NDArray[np.float64],
    ) -> OptimizationResult:
        result = minimize(
            self.objective_function,
            initial_params,
            method="L-BFGS-B",
            bounds=[(-self.max_amplitude, self.max_amplitude)] * 2 * self.segment_count,
        )
        params = result.x
        pulse = self.params_to_pulse(params)
        times = np.linspace(0, self.duration, self.segment_count + 1)
        return OptimizationResult(
            result=result,
            times=times,
            params=params,
            pulse=pulse,
        )
