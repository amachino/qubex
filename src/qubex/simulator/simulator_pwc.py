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


@dataclass
class ControlPWC:
    target: str
    frequency: float
    segment_values: list | npt.NDArray
    segment_width: float
    nsteps_per_segment: int = 5

    @property
    def segment_count(self) -> int:
        return len(self.segment_values)

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        values = np.array(self.segment_values, dtype=np.complex128)
        values = np.repeat(values, self.nsteps_per_segment)
        return values

    @property
    def times(self) -> npt.NDArray[np.float64]:
        duration = self.segment_count * self.segment_width
        nsteps = self.segment_count * self.nsteps_per_segment
        times = np.linspace(0.0, duration, nsteps + 1)
        return times

    @property
    def dt(self) -> float:
        return self.segment_width / self.nsteps_per_segment

    def plot(self, polar: bool = False) -> None:
        durations = [self.segment_width * 1e-9] * self.segment_count
        values = np.array(self.segment_values, dtype=np.complex128) * 1e9
        qv.plot_controls(
            controls={
                self.target: {"durations": durations, "values": values},
            },
            polar=polar,
            figure=plt.figure(),
        )


@dataclass
class SimulationResultPWC:
    system: System
    control: ControlPWC
    unitaries: list[qt.Qobj]
    states: list[qt.Qobj]  # density matrices

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
        rho = np.array([substate.full() for substate in substates])[:, :2, :2]
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
        if label is None:
            figure.suptitle("Population dynamics")
        else:
            figure.suptitle(f"Population dynamics of {label}")
        qv.plot_population_dynamics(
            self.control.times * 1e-9,
            populations,
            figure=figure,
        )


class SimulatorPWC:
    def __init__(
        self,
        system: System,
    ):
        self.system: Final = system

    def calculate_unitaries(
        self,
        static_hamiltonian: qt.Qobj,
        control: ControlPWC,
    ) -> list[qt.Qobj]:
        dt = control.dt
        a = self.system.lowering_operator(control.target)
        ad = a.dag()
        unitaries = [self.system.identity]
        for value in control.values:
            control_hamiltonian = 0.5 * (ad * value + a * np.conj(value))
            H = static_hamiltonian + control_hamiltonian
            U = (-1j * H * dt).expm()
            unitaries.append(U * unitaries[-1])
        return unitaries

    def simulate(
        self,
        control: ControlPWC,
        initial_state: qt.Qobj | StateAlias | dict[str, StateAlias] = "0",
    ):
        # convert the initial state to a Qobj
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)

        static_hamiltonian = self.system.hamiltonian

        for transmon in self.system.transmons:
            # rotating frame of the control frequency
            a = self.system.lowering_operator(transmon.label)
            ad = a.dag()
            static_hamiltonian -= 2 * np.pi * control.frequency * ad * a

        unitaries = self.calculate_unitaries(
            static_hamiltonian=static_hamiltonian,
            control=control,
        )

        states = [qt.ket2dm(U * initial_state) for U in unitaries]

        return SimulationResultPWC(
            system=self.system,
            control=control,
            unitaries=unitaries,
            states=states,
        )
