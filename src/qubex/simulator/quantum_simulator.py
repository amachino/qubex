from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Literal, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import qctrlvisualizer as qv
import qutip as qt
from scipy.interpolate import interp1d

from ..analysis import plot_bloch_vectors
from ..pulse import PulseSchedule, Waveform
from .quantum_system import QuantumSystem


class Control:
    def __init__(
        self,
        target: str,
        frequency: float,
        waveform: list | npt.NDArray | Waveform,
        durations: list | npt.NDArray | None = None,
        interpolation: str = "previous",
    ):
        """
        A control signal for a quantum system.

        Parameters
        ----------
        target : str
            The target object.
        frequency : float
            The control frequency in GHz.
        waveform : list | npt.NDArray | Waveform
            The I/Q values of each segment in rad/ns.
        durations : list | npt.NDArray | None, optional
            The durations of each segment in ns, by default None
        """
        self.target = target
        self.frequency = frequency
        self.waveform = (
            waveform.values
            if isinstance(waveform, Waveform)
            else np.asarray(waveform).astype(np.complex128)
        )
        self.durations = (
            np.asarray(durations).astype(np.float64)
            if durations is not None
            else np.full(len(self.waveform), Waveform.SAMPLING_PERIOD)
        )
        self.interpolation = interpolation

        if len(self.waveform) != len(self.durations):
            raise ValueError("The lengths of rabi_rates and durations do not match.")

    @property
    def n_segments(self) -> int:
        return len(self.waveform)

    @property
    def duration(self) -> float:
        return float(np.sum(self.durations))

    @property
    def times(self) -> npt.NDArray[np.float64]:
        return np.concatenate(([0], np.cumsum(self.durations)))

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        return np.append(self.waveform, self.waveform[-1])

    @property
    def interpolator(self) -> interp1d:
        return interp1d(
            x=self.times,
            y=self.values,
            kind=self.interpolation,
            fill_value="extrapolate",  # type: ignore
        )

    def get_samples(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        return self.interpolator(times)

    def plot(
        self,
        times: npt.NDArray[np.float64] | None = None,
        n_samples: int | None = None,
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
    ) -> None:
        if self.n_segments == 0:
            print("Waveform is empty.")
            return

        if times is None:
            times = self.times
            real = self.values.real / (2 * np.pi * 1e-3)
            imag = self.values.imag / (2 * np.pi * 1e-3)
        else:
            samples = self.get_samples(times)
            real = samples.real / (2 * np.pi * 1e-3)
            imag = samples.imag / (2 * np.pi * 1e-3)

        if n_samples is not None and len(times) > n_samples:
            indices = np.linspace(0, len(times) - 1, n_samples).astype(int)
            times = times[indices]
            real = real[indices]
            imag = imag[indices]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=times,
                y=real,
                mode="lines",
                name="I",
                line_shape=line_shape,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=imag,
                mode="lines",
                name="Q",
                line_shape=line_shape,
            )
        )
        fig.update_layout(
            title="Control signal",
            xaxis_title="Time (ns)",
            yaxis_title="Amplitude (MHz)",
        )
        fig.show()


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
    states : npt.NDArray
        The states of the quantum system at each time point.
    unitaries : npt.NDArray
        The unitaries of the quantum system at each time point.
    """

    system: QuantumSystem
    times: npt.NDArray
    controls: list[Control]
    states: npt.NDArray
    unitaries: npt.NDArray

    def get_substates(
        self,
        label: str,
    ) -> npt.NDArray:
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
        substates = np.array([state.ptrace(index) for state in self.states])
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
        times = downsample(self.times, n_samples)
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
        vectors = np.real(buffer)
        vectors = downsample(vectors, n_samples)
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
        rho = downsample(rho, n_samples)
        return rho

    def plot_bloch_vectors(
        self,
        label: str,
        *,
        n_samples: int | None = None,
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
            mode="lines",
            title=f"State evolution : {label}",
        )

    def display_bloch_sphere(
        self,
        label: str,
        *,
        n_samples: int | None = None,
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
        *,
        n_samples: int | None = None,
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
                populations[f"|{basis}〉"].append(prob)

        sampled_times = self.get_times(n_samples=n_samples)
        sampled_populations = {
            key: downsample(np.asarray(value), n_samples)
            for key, value in populations.items()
        }

        fig = go.Figure()
        for key, value in sampled_populations.items():
            fig.add_trace(
                go.Scatter(
                    x=sampled_times,
                    y=value,
                    mode="lines",
                    name=key,
                )
            )
        fig.update_layout(
            title="Population dynamics"
            if label is None
            else f"Population dynamics : {label}",
            xaxis_title="Time (ns)",
            yaxis_title="Population",
        )
        fig.show()


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
        if len(set([control.n_segments for control in controls])) != 1:
            raise ValueError("The waveforms must have the same length.")

    def simulate(
        self,
        controls: list[Control] | PulseSchedule,
        initial_state: qt.Qobj,
        dt: float = 0.1,
        n_samples: int | None = None,
    ) -> SimulationResult:
        """
        Simulate the dynamics of the quantum system.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            The control signals.
        initial_state : qt.Qobj
            The initial state of the quantum system.
        dt : float, optional
            The time step of the simulation, by default 0.1
        n_samples : int | None, optional
            The number of samples to return, by default None

        Returns
        -------
        SimulationResult
            The result of the simulation.
        """
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)
        if initial_state.dims[0] != self.system.object_dimensions:
            raise ValueError("The dims of the initial state do not match the system.")

        if isinstance(controls, PulseSchedule):
            controls = self._convert_pulse_schedule_to_controls(controls)

        self._validate_controls(controls)

        control = controls[0]
        N = int(control.duration / dt)
        times = np.linspace(0, control.duration, N + 1)

        if control.n_segments == 0:
            return SimulationResult(
                system=self.system,
                times=times,
                controls=controls,
                states=np.array([initial_state]),
                unitaries=np.array([self.system.identity_matrix]),
            )

        control_samples = [control.get_samples(times) for control in controls]

        U_list = [self.system.identity_matrix]
        for idx in range(N):
            t = times[idx]
            H = self.system.get_rotating_hamiltonian(t)
            for control, samples in zip(controls, control_samples):
                target = control.target
                frame_frequency = self.system.get_object(target).frequency
                a = self.system.get_lowering_operator(target)
                ad = self.system.get_raising_operator(target)
                delta = 2 * np.pi * (control.frequency - frame_frequency)
                Omega = 0.5 * samples[idx]  # discrete
                gamma = Omega * np.exp(-1j * delta * t)  # continuous
                H_ctrl = gamma * ad + np.conj(gamma) * a
                H += H_ctrl
            U = (-1j * H * dt).expm() * U_list[-1]
            U_list.append(U)

        rho0 = qt.ket2dm(initial_state)
        states = np.array([U * rho0 * U.dag() for U in U_list])
        unitaries = np.array(U_list)

        if n_samples is not None:
            times = downsample(times, n_samples)
            states = downsample(states, n_samples)
            unitaries = downsample(unitaries, n_samples)

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
        dt: float = 0.1,
        n_samples: int | None = None,
    ) -> SimulationResult:
        """
        Simulate the dynamics of the quantum system using the `mesolve` function.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            The control signals.
        initial_state : qt.Qobj
            The initial state of the quantum system.
        dt : float, optional
            The time step of the simulation, by default 0.1
        n_samples : int | None, optional
            The number of samples to return, by default None

        Returns
        -------
        SimulationResult
            The result of the simulation.
        """
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)
        if initial_state.dims[0] != self.system.object_dimensions:
            raise ValueError("The dims of the initial state do not match the system.")

        if isinstance(controls, PulseSchedule):
            controls = self._convert_pulse_schedule_to_controls(controls)

        self._validate_controls(controls)

        control = controls[0]
        N = int(control.duration / dt)
        times = np.linspace(0, control.duration, N + 1)

        if control.n_segments == 0:
            return SimulationResult(
                system=self.system,
                times=times,
                controls=controls,
                states=np.array([initial_state]),
                unitaries=np.array([self.system.identity_matrix]),
            )

        static_hamiltonian = self.system.zero_matrix
        coupling_hamiltonian: list = []
        control_hamiltonian: list = []
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
            coeffs = g * np.exp(-1j * Delta * times)  # continuous
            coupling_hamiltonian.append([op, coeffs])
            coupling_hamiltonian.append([op.dag(), np.conj(coeffs)])

        # Add control terms
        for control in controls:
            target = control.target
            object = self.system.get_object(target)
            a = self.system.get_lowering_operator(target)
            ad = self.system.get_raising_operator(target)
            delta = 2 * np.pi * (control.frequency - object.frequency)
            samples = control.get_samples(times)
            Omega = 0.5 * samples  # discrete
            gamma = Omega * np.exp(-1j * delta * times)  # continuous
            control_hamiltonian.append([ad, gamma])
            control_hamiltonian.append([a, np.conj(gamma)])

        # Total Hamiltonian
        H = [static_hamiltonian] + coupling_hamiltonian + control_hamiltonian

        # Add collapse operators
        for object in self.system.objects:
            a = self.system.get_lowering_operator(object.label)
            N = self.system.get_number_operator(object.label)
            relaxation_operator = np.sqrt(object.relaxation_rate) * a
            dephasing_operator = np.sqrt(object.dephasing_rate) * N
            collapse_operators.append(relaxation_operator)
            collapse_operators.append(dephasing_operator)

        # Run the simulation
        result = qt.mesolve(
            H=H,
            rho0=initial_state,
            tlist=times,
            c_ops=collapse_operators,
        )

        states = np.array(result.states)

        if n_samples is not None:
            times = downsample(times, n_samples)
            states = downsample(states, n_samples)

        return SimulationResult(
            system=self.system,
            times=times,
            controls=controls,
            states=states,
            unitaries=np.array([]),
        )

    @staticmethod
    def _convert_pulse_schedule_to_controls(
        pulse_schedule: PulseSchedule,
    ) -> list[Control]:
        rabi_rates = pulse_schedule.values
        durations = [Waveform.SAMPLING_PERIOD] * pulse_schedule.length
        frequencies = {}
        targets = {}
        for label in rabi_rates:
            if frequency := pulse_schedule.get_frequency(label):
                frequencies[label] = frequency
            else:
                raise ValueError(f"Frequency for {label} is not provided.")
            if object := pulse_schedule.get_target(label):
                targets[label] = object
            else:
                raise ValueError(f"Object for {label} is not provided.")
        controls = []
        for label, waveform in rabi_rates.items():
            controls.append(
                Control(
                    target=targets[label],
                    frequency=frequencies[label],
                    waveform=waveform,
                    durations=durations,
                )
            )
        return controls


def downsample(
    data: npt.NDArray,
    n_samples: int | None,
) -> npt.NDArray:
    if n_samples is None:
        return data
    if len(data) <= n_samples:
        return data
    indices = np.linspace(0, len(data) - 1, n_samples).astype(int)
    return data[indices]
