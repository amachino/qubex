from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import qctrlvisualizer as qv
import qutip as qt

from ..analysis import plot_bloch_vectors
from ..pulse import Pulse, PulseSchedule, Waveform
from .quantum_system import QuantumSystem


class Control:
    SAMPLING_PERIOD: float = 0.1

    def __init__(
        self,
        target: str,
        frequency: float,
        waveform: list | npt.NDArray | Waveform,
    ):
        """
        A control signal for a quantum system.

        Parameters
        ----------
        target : str
            The target object of the control signal.
        frequency : float
            The frequency of the control signal.
        waveform : list | npt.NDArray | Waveform
            The waveform of the control signal.
        """
        self.target = target
        self.frequency = frequency
        self.waveform = (
            waveform.values
            if isinstance(waveform, Waveform)
            else np.asarray(waveform).astype(np.complex128)
        )

    @property
    def length(self) -> int:
        return len(self.waveform)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.length * self.SAMPLING_PERIOD, self.length + 1)

    def plot(
        self,
        n_samples: int = 256,
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "linear",
    ) -> None:
        Pulse.SAMPLING_PERIOD = self.SAMPLING_PERIOD
        pulse = Pulse(self.waveform * 1e3)
        pulse.plot_xy(
            n_samples=n_samples,
            devide_by_two_pi=True,
            title=f"{self.target} : {self.frequency} GHz",
            ylabel="Amplitude (MHz)",
            line_shape=line_shape,
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

    def get_substates(
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

    def get_times(
        self,
        *,
        n_samples: int | None = None,
    ) -> npt.NDArray:
        """
        Extract the time points of the simulation.

        Returns
        -------
        npt.NDArray
            The time points of the simulation.
        """
        times = self.times
        if n_samples is not None:
            times = self._downsample(self.times, n_samples)
        return times

    def get_bloch_vectors(
        self,
        label: str,
        *,
        n_samples: int | None = None,
    ) -> npt.NDArray:
        """
        Extract the block vectors of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.

        Returns
        -------
        list[qt.Qobj]
            The substates of the qubit.
        """
        X = qt.sigmax()
        Y = qt.sigmay()
        Z = qt.sigmaz()
        substates = self.get_substates(label)
        buffer = []
        for substate in substates:
            rho = qt.Qobj(substate.full()[:2, :2])
            x = qt.expect(X, rho)
            y = qt.expect(Y, rho)
            z = qt.expect(Z, rho)
            buffer.append([x, y, z])
        vectors = np.array(buffer)
        if n_samples is not None:
            vectors = self._downsample(vectors, n_samples)
        return vectors

    def get_density_matrices(
        self,
        label: str,
        *,
        dim: int = 2,
        n_samples: int | None = None,
    ) -> npt.NDArray:
        """
        Extract the density matrices of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.
        dim : int, optional
            The dimension of the qubit, by default 2

        Returns
        -------
        list[qt.Qobj]
            The density matrices of the qubit.
        """
        substates = self.get_substates(label)
        rho = np.array([substate.full() for substate in substates])[:, :dim, :dim]
        if n_samples is not None:
            rho = self._downsample(rho, n_samples)
        return rho

    def plot_bloch_vectors(
        self,
        label: str,
        *,
        n_samples: int = 256,
    ) -> None:
        vectors = self.get_bloch_vectors(
            label,
            n_samples=n_samples,
        )
        times = self.get_times(
            n_samples=n_samples,
        )
        plot_bloch_vectors(
            times=times,
            bloch_vectors=vectors,
            title=f"State evolution : {label}",
        )

    def display_bloch_sphere(
        self,
        label: str,
        *,
        n_samples: int = 256,
    ) -> None:
        """
        Display the Bloch sphere of a qubit.

        Parameters
        ----------
        label : str
            The label of the qubit.
        """
        rho = self.get_density_matrices(
            label,
            n_samples=n_samples,
        )
        qv.display_bloch_sphere_from_density_matrices(rho)

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
        states = self.states if label is None else self.get_substates(label)
        population = states[-1].diag()
        for idx, prob in enumerate(population):
            basis = self.system.basis_labels[idx] if label is None else str(idx)
            print(f"|{basis}⟩: {prob:.6f}")

    def plot_population_dynamics(
        self,
        label: Optional[str] = None,
        n_samples: int = 256,
    ) -> None:
        """
        Plot the population dynamics of the states.

        Parameters
        ----------
        label : Optional[str], optional
            The label of the qubit, by default
        """
        states = self.states if label is None else self.get_substates(label)
        populations = defaultdict(list)
        for state in states:
            population = np.abs(state.diag())
            population[population > 1] = 1.0
            for idx, prob in enumerate(population):
                basis = self.system.basis_labels[idx] if label is None else str(idx)
                populations[rf"$|{basis}\rangle$"].append(prob)

        sampled_times = self.get_times(n_samples=n_samples)
        sampled_populations = {
            key: self._downsample(np.asarray(value), n_samples)
            for key, value in populations.items()
        }

        figure = plt.figure()
        figure.suptitle(f"Population dynamics of {label}")
        qv.plot_population_dynamics(
            sampled_times * 1e-9,
            sampled_populations,
            figure=figure,
        )

    @staticmethod
    def _downsample(
        data: npt.NDArray,
        n_samples: int,
    ) -> npt.NDArray:
        if len(data) <= n_samples:
            return data
        indices = np.linspace(0, len(data) - 1, n_samples).astype(int)
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
