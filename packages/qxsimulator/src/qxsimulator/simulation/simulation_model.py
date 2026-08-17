"""Model passed to QuTiP solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import qutip as qt


@dataclass
class SimulationModel:
    """
    Bundle the Hamiltonian, state, boundary times, and dissipation for QuTiP.

    Attributes
    ----------
    hamiltonian : qt.QobjEvo
        Time-dependent system Hamiltonian in angular-frequency units of
        rad/ns, expressed in the system objects' rotating frames.
    initial_state : qt.Qobj
        Initial ket or density matrix in the full system Hilbert space.
    boundary_times : npt.NDArray[np.float64]
        One-dimensional sorted union of all control segment boundaries in ns,
        with shape `(n_boundary_times,)`. Solvers use these as mandatory
        integration checkpoints independently of their public output times.
    collapse_operators : list[qt.Qobj]
        Full-system collapse operators in inverse-square-root ns. An empty
        list represents closed-system evolution.
    """

    hamiltonian: qt.QobjEvo
    initial_state: qt.Qobj
    boundary_times: npt.NDArray[np.float64]
    collapse_operators: list[qt.Qobj]
