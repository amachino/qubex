from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Final, Literal, Optional, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import qctrlvisualizer as qv
import qutip as qt
import qutip.typing as qtt
from scipy.interpolate import interp1d

from ..analysis.visualization import plot_bloch_vectors
from ..pulse import PulseSchedule, Waveform
from .quantum_system import Object, QuantumSystem

TIME_STEP = 0.1  # ns


FrameType: TypeAlias = Literal["qubit", "drive"]
SubspaceType: TypeAlias = Literal["ge", "ef", "gf"]


class Control:
    def __init__(
        self,
        target: Object | str,
        waveform: Waveform | list | npt.NDArray,
        durations: list | npt.NDArray | None = None,
        frequency: float | None = None,
        interpolation: str = "previous",
        final_frame_shift: float = 0.0,
    ):
        """
        A control signal for a quantum system.

        Parameters
        ----------
        target : Object | str
            The target object.
        waveform : Waveform | list | npt.NDArray
            The I/Q values of each segment in rad/ns.
        durations : list | npt.NDArray | None, optional
            The durations of each segment in ns, by default None
        frequency : float | None, optional
            The control frequency in GHz.
        """
        if frequency is None:
            if isinstance(target, Object):
                frequency = target.frequency
            else:
                raise ValueError("Frequency is required for a string target.")

        if isinstance(target, Object):
            target = target.label

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
        self.final_frame_shift = final_frame_shift

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
            bounds_error=False,
            fill_value=(self.values[0], self.values[-1]),  # type: ignore
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
class SimulationModel:
    """
    The simulation model for QuTip solvers.

    Attributes
    ----------
    hamiltonian : qtt.QobjEvoLike
        The Hamiltonian of the quantum system.
    initial_state : qt.Qobj
        The initial state of the quantum system.
    times : npt.NDArray
        The time points of the simulation.
    collapse_operators : qt.Qobj | qt.QobjEvo | list[qtt.QobjEvoLike]
        The collapse operators of the quantum system.
    """

    hamiltonian: qtt.QobjEvoLike
    initial_state: qt.Qobj
    times: npt.NDArray
    collapse_operators: qt.Qobj | qt.QobjEvo | list[qtt.QobjEvoLike]


@dataclass
class SimulationResult:
    """
    The result of a simulation.

    Attributes
    ----------
    system : QuantumSystem
        The quantum system.
    controls : list[Control]
        The control signals.
    times : npt.NDArray
        The time points of the simulation.
    states : npt.NDArray
        The states of the quantum system at each time point.
    unitaries : npt.NDArray
        The unitaries of the quantum system at each time point.
    model : SimulationModel | None
        The simulation model used for QuTip solvers.
    """

    system: QuantumSystem
    controls: list[Control]
    times: npt.NDArray
    states: npt.NDArray
    unitaries: npt.NDArray
    model: SimulationModel | None = None

    @cached_property
    def control_frequencies(self) -> dict[str, float]:
        return {control.target: control.frequency for control in self.controls}

    @property
    def initial_state(self) -> qt.Qobj:
        return self.states[0]

    @property
    def final_state(self) -> qt.Qobj:
        return self.states[-1]

    def _get_subspace_slice(self, subspace: SubspaceType) -> slice:
        subspaces = {
            "ge": slice(0, 2),
            "ef": slice(1, 3),
            "gf": slice(0, 3),
        }
        if subspace not in subspaces:
            return slice(0, None)
        else:
            return subspaces[subspace]

    def get_substates(
        self,
        label: str,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
    ) -> npt.NDArray:
        """
        Extract the substates of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.
        frame : FrameType | None, optional
            The frame of the substates, by default "qubit"
        frame_frequency : float | None, optional
            The frequency of the frame, by default None.
            If specified, this takes precedence over `frame`.

        Returns
        -------
        list[qt.Qobj]
            The substates of the qubit.
        """
        if frame is None:
            frame = "qubit"

        index = self.system.get_index(label)
        substates = np.array([state.ptrace(index) for state in self.states])

        target_frequency = None
        if frame_frequency is not None:
            target_frequency = frame_frequency
        elif frame == "drive":
            target_frequency = self.control_frequencies[label]

        if target_frequency is not None:
            times = self.get_times()
            qubit = self.system.get_object(label)
            f_qubit = qubit.frequency
            delta = 2 * np.pi * (target_frequency - f_qubit)
            dim = qubit.dimension
            N = qt.num(dim)
            U = lambda t: (-1j * delta * N * t).expm()
            substates = np.array(
                [U(t).dag() * rho * U(t) for t, rho in zip(times, substates)]
            )

        return substates

    def get_initial_substate(
        self,
        label: str,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
    ) -> qt.Qobj:
        """
        Extract the initial substate of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.
        frame : FrameType | None, optional
            The frame of the substates, by default "qubit"
        frame_frequency : float | None, optional
            The frequency of the frame, by default None.

        Returns
        -------
        qt.Qobj
            The initial substate of the qubit.
        """
        return self.get_substates(label, frame=frame, frame_frequency=frame_frequency)[
            0
        ]

    def get_final_substate(
        self,
        label: str,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
    ) -> qt.Qobj:
        """
        Extract the final substate of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.
        frame : FrameType | None, optional
            The frame of the substates, by default "qubit"
        frame_frequency : float | None, optional
            The frequency of the frame, by default None.

        Returns
        -------
        qt.Qobj
            The final substate of the qubit.
        """
        return self.get_substates(label, frame=frame, frame_frequency=frame_frequency)[
            -1
        ]

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
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        subspace: SubspaceType = "ge",
    ) -> npt.NDArray:
        """
        Extract the block vectors of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.
        n_samples : int | None, optional
            The number of samples to return, by default None
        frame : FrameType | None, optional
            The frame of the substates, by default None
        frame_frequency : float | None, optional
            The frequency of the frame, by default None.

        Returns
        -------
        list[qt.Qobj]
            The substates of the qubit.
        """
        X = qt.sigmax()
        Y = qt.sigmay()
        Z = qt.sigmaz()
        substates = self.get_substates(
            label, frame=frame, frame_frequency=frame_frequency
        )
        buffer = []
        level = self._get_subspace_slice(subspace)
        for substate in substates:
            rho = qt.Qobj(substate.full()[level, level])
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
        n_samples: int | None = None,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        subspace: SubspaceType = "ge",
    ) -> npt.NDArray:
        """
        Extract the density matrices of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.
        dim : int, optional
            The dimension of the qubit, by default 2
        n_samples : int | None, optional
            The number of samples to return, by default None
        frame : FrameType | None, optional
            The frame of the substates, by default None
        frame_frequency : float | None, optional
            The frequency of the frame, by default None.

        Returns
        -------
        list[qt.Qobj]
            The density matrices of the qubit.
        """
        substates = self.get_substates(
            label, frame=frame, frame_frequency=frame_frequency
        )
        level = self._get_subspace_slice(subspace)
        rho = np.array([substate.full() for substate in substates])[:, level, level]
        rho = downsample(rho, n_samples)
        return rho

    def plot_bloch_vectors(
        self,
        label: str,
        *,
        n_samples: int | None = None,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        subspace: SubspaceType = "ge",
    ) -> None:
        vectors = self.get_bloch_vectors(
            label,
            n_samples=n_samples,
            frame=frame,
            frame_frequency=frame_frequency,
            subspace=subspace,
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
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        subspace: SubspaceType = "ge",
    ) -> None:
        """
        Display the Bloch sphere of a qubit.

        Parameters
        ----------
        label : str
            The label of the qubit.
        n_samples : int | None, optional
            The number of samples to return, by default None
        frame : FrameType | None, optional
            The frame of the substates, by default None
        frame_frequency : float | None, optional
            The frequency of the frame, by default None.
        """
        rho = self.get_density_matrices(
            label,
            n_samples=n_samples,
            frame=frame,
            frame_frequency=frame_frequency,
            subspace=subspace,
        )
        qv.display_bloch_sphere_from_density_matrices(rho)

    def _get_general_substates(
        self,
        labels: Sequence[str],
        *,
        frame_frequencies: dict[str, float] | None = None,
    ) -> npt.NDArray:
        # 1. Extract substates (ptrace)
        # Note: qutip.ptrace always sorts the indices, so the order of subsystems
        # in the result might differ from 'labels'.
        target_indices = [self.system.get_index(label) for label in labels]
        substates = np.array([state.ptrace(target_indices) for state in self.states])

        # 2. Restore the order of subsystems if necessary
        sorted_indices = sorted(target_indices)
        if target_indices != sorted_indices:
            # Calculate permutation to match the requested 'labels' order
            perm_order = [sorted_indices.index(i) for i in target_indices]
            substates = np.array([rho.permute(perm_order) for rho in substates])

        # 3. Apply frame transformation if requested
        if frame_frequencies is not None:
            substates = self._apply_frame_transformation(
                substates, labels, frame_frequencies
            )

        return substates

    def _apply_frame_transformation(
        self,
        substates: npt.NDArray,
        labels: Sequence[str],
        frame_frequencies: dict[str, float],
    ) -> npt.NDArray:
        times = self.get_times()
        dims = [self.system.get_object(label).dimension for label in labels]

        # Construct the effective Hamiltonian for the frame change
        # H_frame = sum( delta_i * n_i )
        H_frame = qt.qzero(dims)
        for i, label in enumerate(labels):
            if label not in frame_frequencies:
                continue

            target_freq = frame_frequencies[label]
            qubit_freq = self.system.get_object(label).frequency
            delta = 2 * np.pi * (target_freq - qubit_freq)

            if delta == 0:
                continue

            # Operator for the i-th subsystem: I x ... x n_i x ... x I
            ops = [qt.qeye(d) for d in dims]
            ops[i] = qt.num(dims[i])
            H_frame += delta * qt.tensor(*ops)

        if H_frame.norm() == 0:
            return substates

        # Apply unitary transformation: rho' = U rho U^dagger
        # U(t) = exp(i * H_frame * t)
        transformed_substates = []
        for t, rho in zip(times, substates):
            U = (1j * H_frame * t).expm()
            transformed_substates.append(rho.transform(U))

        return np.array(transformed_substates)

    def _get_general_bloch_vectors(
        self,
        labels: Sequence[str],
        *,
        basis_set: tuple[Sequence[int], Sequence[int]],
        frame_frequencies: dict[str, float] | None = None,
        n_samples: int | None = None,
    ) -> npt.NDArray:
        dimensions = [self.system.get_object(label).dimension for label in labels]
        ket0 = qt.tensor(
            *[qt.basis(dim, basis) for dim, basis in zip(dimensions, basis_set[0])]
        )
        ket1 = qt.tensor(
            *[qt.basis(dim, basis) for dim, basis in zip(dimensions, basis_set[1])]
        )
        bra0 = ket0.dag()
        bra1 = ket1.dag()

        X: qt.Qobj = ket0 @ bra1 + ket1 @ bra0
        Y: qt.Qobj = -1j * ket0 @ bra1 + 1j * ket1 @ bra0
        Z: qt.Qobj = ket0 @ bra0 - ket1 @ bra1

        buffer = []
        states = self._get_general_substates(
            labels=labels,
            frame_frequencies=frame_frequencies,
        )
        for rho in states:
            x = qt.expect(X, rho)
            y = qt.expect(Y, rho)
            z = qt.expect(Z, rho)
            buffer.append([x, y, z])

        vectors = np.real(buffer)
        vectors = downsample(vectors, n_samples)
        return vectors

    def _plot_general_bloch_vectors(
        self,
        labels: Sequence[str],
        *,
        basis_set: tuple[Sequence[int], Sequence[int]],
        frame_frequencies: dict[str, float] | None = None,
        n_samples: int | None = None,
    ) -> None:
        vectors = self._get_general_bloch_vectors(
            labels,
            basis_set=basis_set,
            frame_frequencies=frame_frequencies,
            n_samples=n_samples,
        )
        times = self.get_times(
            n_samples=n_samples,
        )
        plot_bloch_vectors(
            times=times,
            bloch_vectors=vectors,
            mode="lines",
            title=f"State evolution : {', '.join(labels)}",
        )

    def _display_general_bloch_sphere(
        self,
        labels: Sequence[str],
        *,
        basis_set: tuple[Sequence[int], Sequence[int]],
        frame_frequencies: dict[str, float] | None = None,
        n_samples: int | None = None,
    ) -> None:
        vectors = self._get_general_bloch_vectors(
            labels,
            basis_set=basis_set,
            frame_frequencies=frame_frequencies,
            n_samples=n_samples,
        )
        qv.display_bloch_sphere_from_bloch_vectors(vectors)

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
        population = np.real(states[-1].diag())
        for idx, prob in enumerate(population):
            basis = self.system.basis_labels[idx] if label is None else str(idx)
            print(f"|{basis}⟩: {prob * 100:6.3f}%")

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
        *,
        initial_state: qt.Qobj | dict | None = None,
        dt: float = TIME_STEP,
        n_samples: int | None = None,
    ) -> SimulationResult:
        """
        Simulate the dynamics of the quantum system.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            The control signals.
        initial_state : qt.Qobj | dict | None, optional
            The initial state of the quantum system, by default None
        dt : float, optional
            The time step of the simulation, by default TIME_STEP
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

        if initial_state is None:
            initial_state = self.system.ground_state

        control = controls[0]
        times = self._generate_simulation_times(control.duration, dt)

        if control.n_segments == 0:
            return SimulationResult(
                system=self.system,
                controls=controls,
                times=times,
                states=np.array([initial_state]),
                unitaries=np.array([self.system.identity_matrix]),
            )

        control_samples = [control.get_samples(times) for control in controls]

        U_list = [self.system.identity_matrix]
        for idx, delta_t in enumerate(np.diff(times)):
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
            U = (-1j * H * delta_t).expm() @ U_list[-1]
            U_list.append(U)

        rho0 = qt.ket2dm(initial_state)
        states = np.array([U @ rho0 @ U.dag() for U in U_list])
        unitaries = np.array(U_list)

        if n_samples is not None:
            times = downsample(times, n_samples)
            states = downsample(states, n_samples)
            unitaries = downsample(unitaries, n_samples)

        return SimulationResult(
            system=self.system,
            controls=controls,
            times=times,
            states=states,
            unitaries=unitaries,
        )

    def mesolve(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        initial_state: qt.Qobj | dict | None = None,
        dt: float | None = None,
        n_samples: int | None = None,
    ) -> SimulationResult:
        """
        Simulate the dynamics of the quantum system using the `mesolve` function.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            The control signals.
        initial_state : qt.Qobj | dict | None, optional
            The initial state of the quantum system, by default None
        dt : float, optional
            The time step of the simulation.
        n_samples : int | None, optional
            The number of samples to return, by default None

        Returns
        -------
        SimulationResult
            The result of the simulation.
        """
        if isinstance(controls, PulseSchedule):
            controls = self._convert_pulse_schedule_to_controls(controls)
        self._validate_controls(controls)

        model = self.create_simulation_model(
            controls=controls,
            initial_state=initial_state,
            dt=dt,
        )

        hamiltonian = model.hamiltonian
        initial_state = model.initial_state
        times = model.times
        collapse_operators = model.collapse_operators

        # Run the simulation
        result = qt.mesolve(
            H=hamiltonian,
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
            controls=controls,
            times=times,
            states=states,
            unitaries=np.array([]),
            model=model,
        )

    def propagator(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        dt: float | None = None,
        options: dict | None = None,
    ) -> qt.Qobj:
        if options is None:
            options = {
                "nsteps": 200000,
                "rtol": 1e-7,
                "atol": 1e-9,
                "method": "bdf",
            }

        if isinstance(controls, PulseSchedule):
            controls = self._convert_pulse_schedule_to_controls(controls)
        self._validate_controls(controls)

        params = self.create_simulation_parameters(
            controls=controls,
            dt=dt,
        )

        S: qt.Qobj = qt.propagator(
            H=params["hamiltonian"],
            tlist=params["times"],
            c_ops=params["collapse_operators"],
            t=params["times"][-1],
            options=options,
        )  # type: ignore

        R = self.system.get_rotation_matrix(
            {control.target: -control.final_frame_shift for control in controls},
        )
        SR = qt.to_super(R)

        return SR @ S

    def gate_fidelity(
        self,
        controls: list[Control] | PulseSchedule,
        target_unitary: qt.Qobj,
        *,
        dt: float | None = None,
        options: dict | None = None,
    ) -> float:
        superop = self.propagator(
            controls=controls,
            dt=dt,
            options=options,
        )
        superop_truncated = self.system.truncate_superoperator(superop)
        return qt.average_gate_fidelity(
            superop_truncated,
            target_unitary,
        )

    def create_simulation_parameters(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        dt: float | None = None,
    ) -> dict:
        if dt is None:
            dt = TIME_STEP

        if isinstance(controls, PulseSchedule):
            controls = self._convert_pulse_schedule_to_controls(controls)
        self._validate_controls(controls)

        control = controls[0]
        times = self._generate_simulation_times(control.duration, dt)

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
            op = ad_0 @ a_1
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
        hamiltonian = [static_hamiltonian] + coupling_hamiltonian + control_hamiltonian

        # Add collapse operators
        for object in self.system.objects:
            a = self.system.get_lowering_operator(object.label)
            n_steps = self.system.get_number_operator(object.label)
            relaxation_operator = np.sqrt(object.relaxation_rate) * a
            dephasing_operator = np.sqrt(object.dephasing_rate) * n_steps
            collapse_operators.append(relaxation_operator)
            collapse_operators.append(dephasing_operator)

        return {
            "times": times,
            "hamiltonian": hamiltonian,
            "collapse_operators": collapse_operators,
        }

    def create_simulation_model(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        initial_state: qt.Qobj | dict | None = None,
        dt: float | None = None,
    ) -> SimulationModel:
        if initial_state is None:
            initial_state = self.system.ground_state
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)
        if initial_state.dims[0] != self.system.object_dimensions:
            raise ValueError("The dims of the initial state do not match the system.")

        params = self.create_simulation_parameters(
            controls=controls,
            dt=dt,
        )

        return SimulationModel(
            hamiltonian=params["hamiltonian"],
            initial_state=initial_state,
            times=params["times"],
            collapse_operators=params["collapse_operators"],
        )

    @staticmethod
    def _generate_simulation_times(
        duration: float,
        dt: float,
    ) -> npt.NDArray:
        times = np.arange(0, duration, dt)

        # Handle potential floating point overshoot from arange
        if len(times) > 0 and times[-1] > duration:
            times = times[:-1]

        if len(times) == 0 or not np.isclose(times[-1], duration, atol=1e-12):
            times = np.append(times, duration)
        else:
            times[-1] = duration

        return times

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
                    final_frame_shift=pulse_schedule.get_final_frame_shift(label),
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
