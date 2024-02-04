from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import qctrlvisualizer as qv  # type: ignore
import qutip as qt  # type: ignore

from .system import StateAlias, System

SAMPLING_PERIOD: float = 2.0  # ns


@dataclass
class Result:
    system: System
    control: Control
    states: list[qt.Qobj]

    def ptrace(self, label: str) -> list[qt.Qobj]:
        index = self.system.index(label)
        return [state.ptrace(index) for state in self.states]

    def draw(
        self,
        label: str,
        frame: Literal["qubit", "drive"] = "qubit",
    ) -> None:
        pstates = self.ptrace(label)

        if frame == "qubit":
            qubit = self.system.transmon(label)
            times = self.control.times
            f_drive = self.control.frequency
            f_qubit = qubit.frequency
            delta = 2 * np.pi * (f_drive - f_qubit)
            dim = qubit.dimension
            a = qt.destroy(dim)
            ad = a.dag()

            def U(t):
                return (-1j * delta * ad * a * t).expm()

            pstates = [U(t) * state * U(t).dag() for t, state in zip(times, pstates)]

        rho = np.array(pstates).squeeze()[:, :2, :2]
        qv.display_bloch_sphere_from_density_matrices(rho)


@dataclass
class Control:
    target: str
    frequency: float
    waveform: npt.NDArray
    sampling_period: float = SAMPLING_PERIOD

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        # duplicate the last value to use as a step function
        arr = np.array(self.waveform, dtype=np.complex128)
        arr = np.append(arr, arr[-1])
        return arr

    @property
    def times(self) -> npt.NDArray[np.float64]:
        length = len(self.values)
        return np.linspace(
            0.0,
            (length - 1) * self.sampling_period,
            length,
        )


class Simulator:
    def __init__(
        self,
        system: System,
    ):
        self.system: Final = system

    def simulate(
        self,
        control: Control,
        initial_state: qt.Qobj | StateAlias | dict[str, StateAlias] = "0",
    ):
        # convert the initial state to a Qobj
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)

        static_hamiltonian = self.system.hamiltonian
        dynamic_hamiltonian: list = []
        collapse_operators: list = []

        for transmon in self.system.transmons:
            a = self.system.lowering_operator(transmon.label)
            ad = a.dag()

            # rotating frame of the control frequency
            static_hamiltonian -= 2 * np.pi * control.frequency * ad * a

            if transmon.label == control.target:
                dynamic_hamiltonian.append([0.5 * a, control.values])
                dynamic_hamiltonian.append([0.5 * ad, np.conj(control.values)])

            decay_operator = np.sqrt(transmon.decay_rate) * a
            dephasing_operator = np.sqrt(transmon.dephasing_rate) * ad * a
            collapse_operators.append(decay_operator)
            collapse_operators.append(dephasing_operator)

        total_hamiltonian = [static_hamiltonian] + dynamic_hamiltonian

        result = qt.mesolve(
            H=total_hamiltonian,
            rho0=initial_state,
            tlist=control.times,
            c_ops=collapse_operators,
        )

        return Result(
            system=self.system,
            control=control,
            states=result.states,
        )
