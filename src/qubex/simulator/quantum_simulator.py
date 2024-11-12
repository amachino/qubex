from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import qctrlvisualizer as qv
import qutip as qt

from ..analysis import plot_bloch_vectors
from ..pulse import Pulse, PulseSchedule
from .quantum_system import QuantumSystem


class Control:
    SAMPLING_PERIOD: float = 0.1

    def __init__(
        self,
        target: str,
        frequency: float,
        waveform: list | npt.NDArray,
    ):
        """
        A control signal for a quantum system.

        Parameters
        ----------
        target : str
            The target object of the control signal.
        frequency : float
            The frequency of the control signal.
        waveform : list | npt.NDArray
            The waveform of the control signal.
        """
        self.target = target
        self.frequency = frequency
        self.waveform = np.asarray(waveform).astype(np.complex128)

    @property
    def length(self) -> int:
        return len(self.waveform)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.length * self.SAMPLING_PERIOD, self.length + 1)

    def plot(self):
        Pulse.SAMPLING_PERIOD = self.SAMPLING_PERIOD
        pulse = Pulse(self.waveform * 1e3)
        pulse.plot_xy(
            title=f"{self.target} : {self.frequency} GHz",
            ylabel="Amplitude (MHz)",
            devide_by_two_pi=True,
        )


@dataclass
class SimulationResult:
    """
    The result of a simulation.

    Attributes
    ----------
    system : QuantumSystem
        The quantum system.
    times : npt.NDArray
        The time points of the simulation.
    controls : list[Control]
        The control signals.
    states : list[qt.Qobj]
        The states of the quantum system at each time point.
    unitaries : list[qt.Qobj]
        The unitaries of the quantum system at each time point.
    """

    system: QuantumSystem
    times: npt.NDArray
    controls: list[Control]
    states: list[qt.Qobj]
    unitaries: list[qt.Qobj]

    def substates(
        self,
        label: str,
    ) -> list[qt.Qobj]:
        """
        Extract the substates of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.

        Returns
        -------
        list[qt.Qobj]
            The substates of the qubit.
        """
        index = self.system.get_index(label)
        substates = [state.ptrace(index) for state in self.states]
        return substates

    def display_bloch_sphere(
        self,
        label: str,
        n_max_points: int = 256,
    ) -> None:
        """
        Display the Bloch sphere of a qubit.

        Parameters
        ----------
        label : str
            The label of the qubit.
        """
        substates = self.substates(label)
        rho = np.array([substate.full() for substate in substates])[:, :2, :2]
        sampled_rho = self._sample_data(rho, n_max_points)
        qv.display_bloch_sphere_from_density_matrices(sampled_rho)

    def show_last_population(
        self,
        label: Optional[str] = None,
    ) -> None:
        """
        Show the population of the last state.

        Parameters
        ----------
        label : Optional[str], optional
            The label of the qubit, by default
        """
        states = self.states if label is None else self.substates(label)
        population = states[-1].diag()
        for idx, prob in enumerate(population):
            basis = self.system.basis_labels[idx] if label is None else str(idx)
            print(f"|{basis}⟩: {prob:.6f}")

    def plot_population_dynamics(
        self,
        label: Optional[str] = None,
        n_max_points: int = 256,
    ) -> None:
        """
        Plot the population dynamics of the states.

        Parameters
        ----------
        label : Optional[str], optional
            The label of the qubit, by default
        """
        states = self.states if label is None else self.substates(label)
        populations = defaultdict(list)
        for state in states:
            population = np.abs(state.diag())
            population[population > 1] = 1.0
            for idx, prob in enumerate(population):
                basis = self.system.basis_labels[idx] if label is None else str(idx)
                populations[rf"$|{basis}\rangle$"].append(prob)

        sampled_times = self._sample_data(self.times, n_max_points)
        sampled_populations = {
            key: self._sample_data(np.asarray(value), n_max_points)
            for key, value in populations.items()
        }

        figure = plt.figure()
        figure.suptitle(f"Population dynamics of {label}")
        qv.plot_population_dynamics(
            sampled_times * 1e-9,
            sampled_populations,
            figure=figure,
        )

    def plot_bloch_vectors(
        self,
        label: str,
        n_max_points: int = 256,
    ) -> None:
        substates = self.substates(label)
        vectors = []
        for substate in substates:
            rho = qt.Qobj(substate.full()[:2, :2])
            x = (rho * qt.sigmax()).tr().real
            y = (rho * qt.sigmay()).tr().real
            z = (rho * qt.sigmaz()).tr().real
            vectors.append([x, y, z])

        all_data = np.asarray(vectors)
        sampled_data = self._sample_data(all_data, n_max_points)
        sampled_times = self._sample_data(self.times, n_max_points)
        plot_bloch_vectors(
            times=sampled_times,
            bloch_vectors=sampled_data,
            title=f"State evolution : {label}",
        )

    @staticmethod
    def _sample_data(
        data: npt.NDArray,
        n_max_points: int,
    ) -> npt.NDArray:
        if len(data) <= n_max_points:
            return data
        indices = np.linspace(0, len(data) - 1, n_max_points).astype(int)
        return data[indices]


class QuantumSimulator:
    def __init__(
        self,
        system: QuantumSystem,
    ):
        """
        A quantum simulator to simulate the dynamics of the quantum system.

        Parameters
        ----------
        system : QuantumSystem
            The quantum system.
        """
        self.system: Final = system

    def _validate_controls(
        self,
        controls: list[Control],
    ):
        if len(controls) == 0:
            raise ValueError("At least one control signal is required.")
        if len(set([control.length for control in controls])) != 1:
            raise ValueError("The waveforms must have the same length.")
        if len(set([control.SAMPLING_PERIOD for control in controls])) != 1:
            raise ValueError("The sampling periods must be the same.")

    def simulate(
        self,
        controls: list[Control],
        initial_state: qt.Qobj,
    ):
        """
        Simulate the dynamics of the quantum system.

        Parameters
        ----------
        controls : list[Control]
            The control signals.

        Returns
        -------
        Result
            The result of the simulation.
        """
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)
        if initial_state.dims[0] != self.system.object_dimensions:
            raise ValueError("The dims of the initial state do not match the system.")

        if isinstance(controls, PulseSchedule):
            controls = self._convert_pulse_schedule_to_controls(controls)

        self._validate_controls(controls)

        length = controls[0].length
        times = controls[0].times
        dt = controls[0].SAMPLING_PERIOD

        if length == 0:
            return SimulationResult(
                system=self.system,
                times=times,
                controls=controls,
                states=[],
                unitaries=[],
            )

        unitaries = [self.system.identity_matrix]
        for idx in range(length):
            t = times[idx]
            H = self.system.get_rotating_hamiltonian(t)
            for control in controls:
                target = control.target
                frame_frequency = self.system.get_object(target).frequency
                a = self.system.get_lowering_operator(target)
                ad = self.system.get_raising_operator(target)
                delta = 2 * np.pi * (control.frequency - frame_frequency)
                Omega = 0.5 * control.waveform[idx]
                gamma = Omega * np.exp(-1j * delta * t)
                H_ctrl = gamma * ad + np.conj(gamma) * a
                H += H_ctrl
            U = (-1j * H * dt).expm() * unitaries[-1]
            unitaries.append(U)

        rho0 = qt.ket2dm(initial_state)
        states = [U * rho0 * U.dag() for U in unitaries]

        return SimulationResult(
            system=self.system,
            times=times,
            controls=controls,
            states=states,
            unitaries=unitaries,
        )

    def mesolve(
        self,
        controls: list[Control] | PulseSchedule,
        initial_state: qt.Qobj,
    ) -> SimulationResult:
        """
        Simulate the dynamics of the quantum system using the `mesolve` function.

        Parameters
        ----------
        controls : list[Control]
            The control signals.
        initial_state : qt.Qobj
            The initial state of the quantum system.

        Returns
        -------
        Result
            The result of the simulation.
        """
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)
        if initial_state.dims[0] != self.system.object_dimensions:
            raise ValueError("The dims of the initial state do not match the system.")

        if isinstance(controls, PulseSchedule):
            controls = self._convert_pulse_schedule_to_controls(controls)

        self._validate_controls(controls)

        length = controls[0].length
        times = controls[0].times

        if length == 0:
            return SimulationResult(
                system=self.system,
                times=times,
                controls=controls,
                states=[],
                unitaries=[],
            )

        static_hamiltonian = self.system.zero_matrix
        dynamic_hamiltonian: list = []
        collapse_operators: list = []

        # Add static terms
        for label in self.system.object_labels:
            static_hamiltonian += self.system.get_rotating_object_hamiltonian(label)

        # Add coupling terms
        for coupling in self.system.couplings:
            ad_0 = self.system.get_raising_operator(coupling.pair[0])
            a_1 = self.system.get_lowering_operator(coupling.pair[1])
            op = ad_0 * a_1
            g = 2 * np.pi * coupling.strength
            Delta = self.system.get_coupling_detuning(coupling.label)
            coeffs = g * np.exp(-1j * Delta * times)
            dynamic_hamiltonian.append([op, coeffs])
            dynamic_hamiltonian.append([op.dag(), np.conj(coeffs)])

        # Add control terms
        for control in controls:
            target = control.target
            object = self.system.get_object(target)
            a = self.system.get_lowering_operator(target)
            ad = self.system.get_raising_operator(target)
            delta = 2 * np.pi * (control.frequency - object.frequency)
            waveform = control.waveform
            Omega = 0.5 * np.concatenate([waveform, [waveform[-1]]])
            gamma = Omega * np.exp(-1j * delta * control.times)
            dynamic_hamiltonian.append([ad, gamma])
            dynamic_hamiltonian.append([a, np.conj(gamma)])

        # Add collapse operators
        for object in self.system.objects:
            a = self.system.get_lowering_operator(object.label)
            N = self.system.get_number_operator(object.label)
            relaxation_operator = np.sqrt(object.relaxation_rate) * a
            dephasing_operator = np.sqrt(object.dephasing_rate) * N
            collapse_operators.append(relaxation_operator)
            collapse_operators.append(dephasing_operator)

        total_hamiltonian = [static_hamiltonian] + dynamic_hamiltonian

        H = qt.QobjEvo(  # type: ignore
            total_hamiltonian,
            tlist=times,
            order=0,  # 0th order for piecewise constant control
        )

        result = qt.mesolve(
            H=H,
            rho0=initial_state,
            tlist=times,
            c_ops=collapse_operators,
        )

        return SimulationResult(
            system=self.system,
            times=times,
            controls=controls,
            states=result.states,
            unitaries=[],
        )

    @staticmethod
    def _convert_pulse_schedule_to_controls(
        pulse_schedule: PulseSchedule,
    ) -> list[Control]:
        waveforms = pulse_schedule.sampled_sequences
        frequencies = {}
        objects = {}
        for label in waveforms:
            if frequency := pulse_schedule.frequencies.get(label):
                frequencies[label] = frequency
            else:
                raise ValueError(f"Frequency for {label} is not provided.")
            if object := pulse_schedule.objects.get(label):
                objects[label] = object
            else:
                raise ValueError(f"Object for {label} is not provided.")
        controls = []
        for label, waveform in waveforms.items():
            controls.append(
                Control(
                    target=objects[label],
                    frequency=frequencies[label],
                    waveform=waveform,
                )
            )
        return controls
