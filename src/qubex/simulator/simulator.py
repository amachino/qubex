from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import qctrlvisualizer as qv
import qutip as qt

from .system import StateAlias, System

SAMPLING_PERIOD: float = 2.0  # ns
STEPS_PER_SAMPLE: int = 4


@dataclass
class Control:
    target: str
    frequency: float
    waveform: list | npt.NDArray
    sampling_period: float = SAMPLING_PERIOD
    steps_per_sample: int = STEPS_PER_SAMPLE

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        waveform = np.array(self.waveform, dtype=np.complex128)
        return np.repeat(waveform, self.steps_per_sample)

    @property
    def times(self) -> npt.NDArray[np.float64]:
        length = len(self.values)
        return np.linspace(
            0.0,
            length * self.sampling_period / self.steps_per_sample,
            length,
        )

    def plot(self, polar: bool = False) -> None:
        durations = [self.sampling_period * 1e-9] * len(self.waveform)
        values = np.array(self.waveform, dtype=np.complex128) * 1e9
        qv.plot_controls(
            controls={
                self.target: {"durations": durations, "values": values},
            },
            polar=polar,
            figure=plt.figure(),
        )


@dataclass
class Result:
    system: System
    control: Control
    states: list[qt.Qobj]

    def substates(
        self,
        label: str,
        frame: Literal["qubit", "drive"] = "qubit",
    ) -> list[qt.Qobj]:
        index = self.system.index(label)
        substates = [state.ptrace(index) for state in self.states]

        if frame == "qubit":
            # rotate the states to the qubit frame
            times = self.control.times
            qubit = self.system.transmon(label)
            f_drive = self.control.frequency
            f_qubit = qubit.frequency
            delta = 2 * np.pi * (f_drive - f_qubit)
            dim = qubit.dimension
            a = qt.destroy(dim)
            U = lambda t: (-1j * delta * a.dag() * a * t).expm()
            substates = [U(t) * rho * U(t).dag() for t, rho in zip(times, substates)]

        return substates

    def display_bloch_sphere(
        self,
        label: str,
        frame: Literal["qubit", "drive"] = "qubit",
    ) -> None:
        substates = self.substates(label, frame)
        rho = np.array(substates).squeeze()[:, :2, :2]
        print(f"{label} in the {frame} frame")
        qv.display_bloch_sphere_from_density_matrices(rho)

    def show_last_population(
        self,
        label: Optional[str] = None,
    ) -> None:
        states = self.states if label is None else self.substates(label)
        population = states[-1].diag()
        for idx, prob in enumerate(population):
            basis = self.system.basis_labels[idx] if label is None else str(idx)
            print(f"|{basis}⟩: {prob:.3f}")

    def plot_population_dynamics(
        self,
        label: Optional[str] = None,
    ) -> None:
        states = self.states if label is None else self.substates(label)
        populations = defaultdict(list)
        for state in states:
            population = state.diag()
            population = np.clip(population, 0, 1)
            for idx, prob in enumerate(population):
                basis = self.system.basis_labels[idx] if label is None else str(idx)
                populations[rf"$|{basis}\rangle$"].append(prob)

        figure = plt.figure()
        figure.suptitle(f"Population dynamics of {label}")
        qv.plot_population_dynamics(
            self.control.times * 1e-9,
            populations,
            figure=figure,
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
                dynamic_hamiltonian.append([0.5 * ad, control.values])
                dynamic_hamiltonian.append([0.5 * a, np.conj(control.values)])

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
