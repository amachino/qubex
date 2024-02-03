from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import qctrlvisualizer as qv  # type: ignore
import qutip as qt  # type: ignore

from .system import System, StateAlias


SAMPLING_PERIOD: float = 2.0  # ns


@dataclass
class Result:
    system: System
    times: npt.NDArray[np.float64]
    waveforms: dict[str, npt.NDArray[np.complex128]]
    states: list[qt.Qobj]

    def ptrace(self, label: str) -> list[qt.Qobj]:
        index = self.system.index(label)
        return [state.ptrace(index) for state in self.states]

    def draw(self, label: str):
        rho = np.array(self.ptrace(label)).squeeze()
        rho_qubit_subspace = rho[:, :2, :2]
        qv.display_bloch_sphere_from_density_matrices(rho_qubit_subspace)


class Simulator:
    def __init__(
        self,
        system: System,
        sampling_period: float = SAMPLING_PERIOD,
    ):
        self.system: Final = system
        self.sampling_period: Final = sampling_period

    def simulate(
        self,
        controls: dict[str, list | npt.NDArray],
        initial_state: qt.Qobj | StateAlias | dict[str, StateAlias] = "0",
    ):
        # normalize the controls
        times, waveforms = self._normalize(controls)

        # convert the initial state to a Qobj
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)

        # create the hamiltonian and collapse operators
        hamiltonian: list = [self.system.hamiltonian]
        collapse_operators: list = []

        # add controls to the system
        for label, waveform in waveforms.items():
            transmon = self.system.transmon(label)
            a = self.system.lowering_operator(label)
            ad = a.dag()
            hamiltonian.append([0.5 * a, waveform])
            hamiltonian.append([0.5 * ad, np.conj(waveform)])

        # add noise to the system
        for transmon in self.system.transmons:
            a = self.system.lowering_operator(transmon.label)
            ad = a.dag()
            decay_operator = np.sqrt(transmon.decay_rate) * a
            dephasing_operator = np.sqrt(transmon.dephasing_rate) * ad * a
            collapse_operators.append(decay_operator)
            collapse_operators.append(dephasing_operator)

        result = qt.mesolve(
            H=hamiltonian,
            rho0=initial_state,
            tlist=times,
            c_ops=collapse_operators,
        )

        return Result(
            system=self.system,
            waveforms=waveforms,
            times=result.times,
            states=result.states,
        )

    def _normalize(
        self,
        controls: dict[str, list | npt.NDArray],
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.complex128]]]:
        if len(controls) == 0:
            raise ValueError("At least one control must be provided.")

        if len({len(waveform) for waveform in controls.values()}) != 1:
            raise ValueError("All controls must have the same length.")

        def duplicate_last_value(
            waveform: list | npt.NDArray,
        ) -> npt.NDArray[np.complex128]:
            arr = np.array(waveform, dtype=np.complex128)
            arr = np.append(arr, arr[-1])
            return arr

        # duplicate the last value of each waveform to use as a step function
        waveforms = {
            label: duplicate_last_value(waveform)
            for label, waveform in controls.items()
        }

        # get length of the waveforms
        waveform_length = list(waveforms.values())[0].shape[0]

        # create time array for the step function
        times = np.linspace(
            0.0,
            (waveform_length - 1) * self.sampling_period,
            waveform_length,
        )

        return times, waveforms
