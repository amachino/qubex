from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import qctrlvisualizer as qv
import qutip as qt

from .quantum_system import QuantumSystem

SAMPLING_PERIOD: float = 2.0  # ns
STEP_PER_SAMPLE: int = 4


class Control:
    def __init__(
        self,
        target: str,
        frequency: float,
        waveform: list | npt.NDArray,
        sampling_period: float = SAMPLING_PERIOD,
        steps_per_sample: int = STEP_PER_SAMPLE,
    ):
        """
        A control signal for a single qubit.

        Parameters
        ----------
        target : str
            The label of the qubit to control.
        frequency : float
            The frequency of the control signal.
        waveform : list | npt.NDArray
            The waveform of the control signal.
        sampling_period : float, optional
            The sampling period of the control signal, by default 2.0 ns.
        steps_per_sample : int, optional
            The number of steps per sample, by default 4.
        """
        self.target = target
        self.frequency = frequency
        self.waveform = waveform
        self.sampling_period = sampling_period
        self.steps_per_sample = steps_per_sample

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """
        The piecewise constant values of the control signal.

        Returns
        -------
        npt.NDArray[np.complex128]
            The values of the control signal.

        Notes
        -----
        The values are constant during each sample period and repeat `steps_per_sample` times.

        Examples
        --------
        >>> control = Control(
        ...     target="Q01",
        ...     frequency=5.0e9,
        ...     waveform=[1, 2, 3],
        ...     steps_per_sample=4,
        ... )
        >>> control.values
        array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        """
        return np.repeat(self.waveform, self.steps_per_sample).astype(np.complex128)

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """
        The time points of the control signal.

        Returns
        -------
        npt.NDArray[np.float64]
            The time points of the control signal.
        """
        length = len(self.values)
        return np.linspace(
            0.0,
            length * self.sampling_period / self.steps_per_sample,
            length,
        )

    @property
    def sampling_rate(self) -> float:
        """
        The sampling rate of the control signal.

        Returns
        -------
        float
            The sampling rate of the control signal in GHz.
        """
        return self.steps_per_sample / self.sampling_period

    def plot(
        self,
        polar: bool = False,
    ) -> None:
        """
        Plot the control signal.

        Parameters
        ----------
        polar : bool, optional
            Whether to plot the control signal in polar coordinates, by default False.
        """
        durations = [self.sampling_period * 1e-9] * len(self.waveform)
        values = np.array(self.waveform, dtype=np.complex128) * 1e9
        qv.plot_controls(
            controls={
                self.target: {"durations": durations, "values": values},
            },
            polar=polar,
            figure=plt.figure(),
        )


class MultiControl:
    def __init__(
        self,
        frequencies: dict[str, float],
        waveforms: dict[str, list] | dict[str, npt.NDArray],
        sampling_period: float = SAMPLING_PERIOD,
        steps_per_sample: int = STEP_PER_SAMPLE,
    ):
        """
        Parameters
        ----------
        frequencies : dict[str, float]
            The frequencies of the control signals.
        waveforms : dict[str, list | npt.NDArray]
            The waveforms of the control signals.
        sampling_period : float, optional
            The sampling period of the control signals, by default 2.0 ns.
        steps_per_sample : int, optional
            The number of steps per sample, by default 4.

        Raises
        ------
        ValueError
            If the keys of frequencies and waveforms do not match.
            If the waveforms have different lengths.

        Examples
        --------
        >>> control = MultiControl(
        ...     frequencies={"Q01": 5.0e9, "Q02": 6.0e9},
        ...     waveforms={"Q01": [0.5, 0.5], "Q02": [0.5, -0.5]},
        ... )

        Notes
        -----
        Specify `step_per_sample` to sufficiently resolve the frequency difference of the control pulses.
        """
        if set(frequencies.keys()) != set(waveforms.keys()):
            raise ValueError("The keys of frequencies and waveforms must match.")

        if len(set(len(waveform) for waveform in waveforms.values())) > 1:
            raise ValueError("All waveforms must have the same length.")

        self.frequencies = frequencies
        self.waveforms = waveforms
        self.sampling_period = sampling_period
        self.steps_per_sample = steps_per_sample

    @property
    def frequency(self) -> float:
        """
        The average frequency of the control signals.

        Returns
        -------
        float
            The average frequency of the control signals in GHz.

        Notes
        -----
        This frequency is used to calculate the rotating frame of the control signals.
        """
        return sum(self.frequencies.values()) / len(self.frequencies)

    @property
    def values(self) -> dict[str, npt.NDArray[np.complex128]]:
        """
        The piecewise constant values of the control signals.

        Returns
        -------
        dict[str, npt.NDArray[np.complex128]]
            The values of the control signals.

        Notes
        -----
        The values are constant during each sample period and repeat `steps_per_sample` times.

        Examples
        --------
        >>> control = MultiControl(
        ...     frequencies={"Q01": 5.0e9, "Q02": 6.0e9},
        ...     waveforms={"Q01": [1, 2], "Q02": [3, 4]},
        ...     steps_per_sample=6,
        ... )
        >>> control.values
        {
            'Q01': array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
            'Q02': array([3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]),
        }
        """
        waveforms = {
            target: np.repeat(waveform, self.steps_per_sample).astype(np.complex128)
            for target, waveform in self.waveforms.items()
        }
        return waveforms

    @property
    def length(self) -> int:
        """
        The length of the control signals.

        Returns
        -------
        int
            The length of the control signals.
        """
        return len(next(iter(self.values.values())))

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """
        The time points of the control signals.

        Returns
        -------
        npt.NDArray[np.float64]
            The time points of the control signals.
        """
        return np.linspace(
            0.0,
            self.length * self.sampling_period / self.steps_per_sample,
            self.length,
        )

    def plot(
        self,
        polar: bool = False,
    ) -> None:
        """
        Plot the control signals.

        Parameters
        ----------
        polar : bool, optional
            Whether to plot the control signals in polar coordinates, by default False.
        """
        controls = {}
        for target, waveform in self.waveforms.items():
            durations = [self.sampling_period * 1e-9] * len(waveform)
            values = np.array(waveform, dtype=np.complex128) * 1e9
            controls[target] = {"durations": durations, "values": values}
        qv.plot_controls(
            controls=controls,
            polar=polar,
            figure=plt.figure(),
        )


@dataclass
class SimulationResult:
    """
    The result of a simulation.

    Attributes
    ----------
    system : QuantumSystem
        The quantum system.
    control : Control | MultiControl
        The control signal.
    states : list[qt.Qobj]
        The states of the quantum system at each time point.
    """

    system: QuantumSystem
    control: Control | MultiControl
    states: list[qt.Qobj]

    def substates(
        self,
        label: str,
        frame: Literal["qubit", "drive"] = "qubit",
    ) -> list[qt.Qobj]:
        """
        Extract the substates of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.
        frame : Literal["qubit", "drive"], optional
            The frame of the substates, by default "qubit".

        Returns
        -------
        list[qt.Qobj]
            The substates of the qubit.
        """
        index = self.system.get_index(label)
        substates = [state.ptrace(index) for state in self.states]

        if frame == "qubit":
            # rotate the states to the qubit frame
            times = self.control.times
            qubit = self.system.get_object(label)
            f_drive = self.control.frequency
            f_qubit = qubit.frequency
            delta = 2 * np.pi * (f_qubit - f_drive)
            dim = qubit.dimension
            N = qt.num(dim)
            U = lambda t: (-1j * delta * N * t).expm()
            substates = [U(t).dag() * rho * U(t) for t, rho in zip(times, substates)]

        return substates

    def display_bloch_sphere(
        self,
        label: str,
        frame: Literal["qubit", "drive"] = "qubit",
    ) -> None:
        """
        Display the Bloch sphere of a qubit.

        Parameters
        ----------
        label : str
            The label of the qubit.
        frame : Literal["qubit", "drive"], optional
            The frame of the Bloch sphere, by default "qubit".
        """
        substates = self.substates(label, frame)
        rho = np.array([substate.full() for substate in substates])[:, :2, :2]
        print(f"{label} in the {frame} frame")
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
        states = self.states if label is None else self.substates(label)
        population = states[-1].diag()
        for idx, prob in enumerate(population):
            basis = self.system.basis_labels[idx] if label is None else str(idx)
            print(f"|{basis}⟩: {prob:.6f}")

    def plot_population_dynamics(
        self,
        label: Optional[str] = None,
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

    def simulate(
        self,
        control: Control | MultiControl,
        initial_state: qt.Qobj,
    ) -> SimulationResult:
        """
        Simulate the dynamics of the quantum system.

        Parameters
        ----------
        control : Control | MultiControl
            The control signal.

        Returns
        -------
        Result
            The result of the simulation.
        """
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)

        if initial_state.dims[0] != self.system.object_dimensions:
            raise ValueError("The dims of the initial state do not match the system.")

        if len(control.times) == 0:
            return SimulationResult(
                system=self.system,
                control=control,
                states=[],
            )

        static_hamiltonian = self.system.hamiltonian
        dynamic_hamiltonian: list = []
        collapse_operators: list = []

        for object in self.system.objects:
            label = object.label
            a = self.system.get_lowering_operator(label)
            ad = self.system.get_raising_operator(label)
            N = self.system.get_number_operator(label)

            # rotating frame of the control frequency
            frame_frequency = control.frequency
            static_hamiltonian -= 2 * np.pi * frame_frequency * N

            if isinstance(control, Control) and label == control.target:
                gamma = 0.5 * control.values
                dynamic_hamiltonian.append([ad, gamma])
                dynamic_hamiltonian.append([a, np.conj(gamma)])
            elif isinstance(control, MultiControl) and label in control.frequencies:
                control_frequency = control.frequencies[label]
                delta = 2 * np.pi * (control_frequency - frame_frequency)
                Omega = 0.5 * control.values[label]
                gamma = Omega * np.exp(-1j * delta * control.times)
                dynamic_hamiltonian.append([ad, gamma])
                dynamic_hamiltonian.append([a, np.conj(gamma)])

            relaxation_operator = np.sqrt(object.relaxation_rate) * a
            dephasing_operator = np.sqrt(object.dephasing_rate) * N
            collapse_operators.append(relaxation_operator)
            collapse_operators.append(dephasing_operator)

        total_hamiltonian = [static_hamiltonian] + dynamic_hamiltonian

        H = qt.QobjEvo(  # type: ignore
            total_hamiltonian,
            tlist=control.times,
            order=0,  # 0th order for piecewise constant control
        )

        result = qt.mesolve(
            H=H,
            rho0=initial_state,
            tlist=control.times,
            c_ops=collapse_operators,
        )

        return SimulationResult(
            system=self.system,
            control=control,
            states=result.states,
        )
