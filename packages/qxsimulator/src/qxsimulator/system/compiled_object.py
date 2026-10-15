"""Compiled local quantum-system models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import qutip as qt

if TYPE_CHECKING:
    from .models.object import Object


@dataclass(frozen=True, slots=True)
class ChargeBasisEigensystem:
    """
    Store a cosine transmon's finite-charge-basis eigensystem.

    Attributes
    ----------
    charging_energy : float
        Charging energy divided by Planck's constant in GHz.
    josephson_energy : float
        Josephson energy divided by Planck's constant in GHz.
    offset_charge : float
        Dimensionless offset charge `n_g` used for diagonalization, reduced to
        `[-0.5, 0.5)` in Cooper-pair units.
    charge_numbers : NDArray[np.float64]
        Cooper-pair charge numbers spanning the finite basis, with shape
        `(charge_basis_dimension,)`.
    hamiltonian : NDArray[np.float64]
        Full charge-basis `H / h` in GHz, with shape
        `(charge_basis_dimension, charge_basis_dimension)`.
    eigenenergies : NDArray[np.float64]
        Lowest retained eigenenergies divided by Planck's constant in GHz,
        before subtracting the ground-state energy, with shape `(dimension,)`.
    eigenvectors : NDArray[np.complex128]
        Phase-aligned retained energy eigenvectors as charge-basis columns,
        with shape `(charge_basis_dimension, dimension)`.
    """

    charging_energy: float
    josephson_energy: float
    offset_charge: float
    charge_numbers: npt.NDArray[np.float64]
    hamiltonian: npt.NDArray[np.float64]
    eigenenergies: npt.NDArray[np.float64]
    eigenvectors: npt.NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class CompiledObject:
    """
    Store local QuTiP operators compiled from an object specification.

    Attributes
    ----------
    source : Object
        Source object specification.
    hamiltonian : qt.Qobj
        Local `H / hbar` in rad/ns in the retained basis.
    interaction_operator : qt.Qobj
        Full dimensionless local operator available for physical interactions.
    lowering_operator : qt.Qobj
        Adjacent energy-lowering part used for exchange coupling and
        phenomenological relaxation.
    collapse_operators : tuple[qt.Qobj, ...]
        Local collapse operators for positive decoherence rates in
        `ns**(-1/2)`, ordered as energy relaxation then pure dephasing.
    """

    source: Object
    hamiltonian: qt.Qobj
    interaction_operator: qt.Qobj
    lowering_operator: qt.Qobj
    collapse_operators: tuple[qt.Qobj, ...]


@dataclass(frozen=True, slots=True)
class CompiledCosineTransmon(CompiledObject):
    """
    Store a compiled cosine transmon and its charge-basis provenance.

    Attributes
    ----------
    charge_basis : ChargeBasisEigensystem
        Energy parameters, finite Hamiltonian, and retained eigenvectors used
        to project the relative-charge operator into the local energy basis.
    """

    charge_basis: ChargeBasisEigensystem
