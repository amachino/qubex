"""
Compile object specifications into basis-consistent local QuTiP models.

Source objects store experimentally convenient cyclic frequencies in GHz and
decoherence rates in inverse ns. Compilation converts Hamiltonians to `H / hbar`
in rad/ns, constructs operators in the retained local basis, and packages the
result for embedding by `QuantumSystem`.

Duffing objects use the truncated harmonic-oscillator basis. Cosine transmons
are first diagonalized in a finite charge basis and then projected into their
lowest energy eigenstates. The collapse operators supplied here are
phenomenological `T1` and pure-dephasing models; microscopic noise channels
would additionally require a bath-coupling operator and spectral density.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import qutip as qt
from scipy.optimize import least_squares

from .compiled_object import (
    ChargeBasisEigensystem,
    CompiledCosineTransmon,
    CompiledObject,
)
from .models.object import Object

if TYPE_CHECKING:
    from .models.transmon import Transmon


def compile_duffing_object(source: Object) -> CompiledObject:
    """
    Compile an object in the truncated Duffing-oscillator basis.

    Parameters
    ----------
    source : Object
        Object specification whose frequency and anharmonicity are in GHz and
        whose decoherence rates are in inverse ns.

    Returns
    -------
    CompiledObject
        Local Hamiltonian, oscillator operators, and phenomenological collapse
        operators in the `source.dimension` Hilbert space.

    Notes
    -----
    The Hamiltonian represents `H / hbar` and is constructed as
    `omega N + alpha N (N - 1) / 2`, with `omega` and `alpha` converted from
    cyclic GHz to angular rad/ns by `2 * pi`. The interaction operator is
    `a + a.dag()`, and the lowering operator is the truncated oscillator
    annihilation operator `a`.
    """
    lowering_operator = qt.destroy(source.dimension)
    raising_operator = lowering_operator.dag()
    number_operator = raising_operator @ lowering_operator
    hamiltonian = 2 * np.pi * source.frequency * number_operator
    if source.dimension >= 3 and source.anharmonicity != 0.0:
        hamiltonian += (
            np.pi
            * source.anharmonicity
            * (
                raising_operator
                @ raising_operator
                @ lowering_operator
                @ lowering_operator
            )
        )
    return CompiledObject(
        source=source,
        hamiltonian=hamiltonian,
        interaction_operator=lowering_operator + raising_operator,
        lowering_operator=lowering_operator,
        collapse_operators=build_collapse_operators(source, lowering_operator),
    )


def build_collapse_operators(
    source: Object,
    lowering_operator: qt.Qobj,
) -> tuple[qt.Qobj, ...]:
    """
    Build phenomenological relaxation and pure-dephasing operators.

    Parameters
    ----------
    source : Object
        Object carrying the physical 0-1 relaxation and pure-dephasing rates
        in inverse ns.
    lowering_operator : qt.Qobj
        Energy-lowering operator in the retained local basis. For a Duffing
        object this is `a`; for a cosine transmon it contains the adjacent
        upper-diagonal elements of the projected, normalized charge operator.

    Returns
    -------
    tuple[qt.Qobj, ...]
        Collapse operators for positive rates, with relaxation before pure
        dephasing when both are present.

    Notes
    -----
    Relaxation is modeled as `sqrt(gamma_1) * L`. For a normalized cosine
    charge operator this fixes the 1-to-0 decay rate to `gamma_1` while higher
    transition rates inherit the corresponding charge matrix elements.

    Pure dephasing is modeled as `sqrt(2 * gamma_phi) * N`, where `N` is the
    energy-level index operator `diag(0, 1, 2, ...)`, not the Cooper-pair charge
    operator used in the cosine Hamiltonian. The factor `sqrt(2)` makes the
    0-1 coherence decay at `gamma_phi`; a general matrix element `rho_mn`
    decays at `gamma_phi * (m - n) ** 2`.

    The dephasing construction is therefore a measured-`T_phi` convention,
    rather than a microscopic consequence of the selected local Hamiltonian.
    Noise-specific dephasing would instead use the energy-basis diagonal part
    of the appropriate parameter derivative of the Hamiltonian.
    """
    collapse_operators = []
    if source.relaxation_rate > 0:
        collapse_operators.append(np.sqrt(source.relaxation_rate) * lowering_operator)
    if source.dephasing_rate > 0:
        # D[n] damps rho_mn at (m - n)^2 / 2, so sqrt(2 * gamma_phi)
        # makes dephasing_rate the physical 0-1 coherence-decay rate.
        collapse_operators.append(
            np.sqrt(2 * source.dephasing_rate) * qt.num(source.dimension)
        )
    return tuple(collapse_operators)


def _canonicalize_offset_charge(offset_charge: float) -> float:
    """
    Map an offset charge to the fundamental interval `[-0.5, 0.5)`.

    Parameters
    ----------
    offset_charge : float
        Dimensionless offset charge `n_g` in Cooper-pair units.

    Returns
    -------
    float
        Equivalent offset charge in `[-0.5, 0.5)`.

    Notes
    -----
    Integer shifts of `n_g` are related by a relabeling of the Cooper-pair
    charge states. Canonicalizing before finite-basis diagonalization preserves
    this exact period-one equivalence without moving the numerical cutoff.
    """
    return (offset_charge + 0.5) % 1.0 - 0.5


def _charge_basis_eigensystem(
    *,
    charging_energy: float,
    josephson_energy: float,
    offset_charge: float,
    charge_cutoff: int,
    dimension: int,
) -> ChargeBasisEigensystem:
    """
    Diagonalize a cosine transmon in a finite Cooper-pair charge basis.

    Parameters
    ----------
    charging_energy : float
        Charging energy `E_C / h` in GHz.
    josephson_energy : float
        Josephson energy `E_J / h` in GHz.
    offset_charge : float
        Dimensionless offset charge `n_g` in Cooper-pair units. Integer shifts
        are reduced to the fundamental interval `[-0.5, 0.5)` internally.
    charge_cutoff : int
        Positive cutoff defining charge states from `-charge_cutoff` through
        `+charge_cutoff`.
    dimension : int
        Number of low-energy eigenstates retained in the returned eigensystem.

    Returns
    -------
    ChargeBasisEigensystem
        Energy parameters, charge numbers, Hamiltonian, retained
        eigenenergies, and phase-aligned retained charge-basis eigenvectors.

    Notes
    -----
    In cyclic-frequency units the diagonalized Hamiltonian is
    `H / h = 4 (E_C / h) (n - n_g) ** 2 - (E_J / h) cos(phi)`. In the charge
    basis, `cos(phi)` connects adjacent charge states, producing off-diagonal
    matrix elements `-(E_J / h) / 2`. The finite cutoff is numerical rather
    than physical; convergence of energies and relevant matrix elements should
    be checked when high levels or small `E_J / E_C` ratios are important.
    """
    canonical_offset_charge = _canonicalize_offset_charge(offset_charge)
    charges = np.arange(-charge_cutoff, charge_cutoff + 1, dtype=np.float64)
    hamiltonian = np.diag(
        4 * charging_energy * (charges - canonical_offset_charge) ** 2
    )
    hopping = np.full(len(charges) - 1, -0.5 * josephson_energy)
    hamiltonian += np.diag(hopping, k=1) + np.diag(hopping, k=-1)
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    order = np.argsort(energies)
    retained_energies = energies[order][:dimension].copy()
    retained_eigenvectors = eigenvectors[:, order][:, :dimension].astype(
        np.complex128,
        copy=True,
    )

    # Each eigenvector has an arbitrary phase. Fix those phases recursively so
    # each resolvable adjacent charge matrix element <level|n|level + 1> is
    # real and positive, matching the usual oscillator convention.
    for level in range(dimension - 1):
        matrix_element = np.vdot(
            retained_eigenvectors[:, level],
            charges * retained_eigenvectors[:, level + 1],
        )
        if abs(matrix_element) > 1e-14:
            retained_eigenvectors[:, level + 1] *= np.exp(
                -1j * np.angle(matrix_element)
            )

    return ChargeBasisEigensystem(
        charging_energy=charging_energy,
        josephson_energy=josephson_energy,
        offset_charge=canonical_offset_charge,
        charge_numbers=charges,
        hamiltonian=hamiltonian,
        eigenenergies=retained_energies,
        eigenvectors=retained_eigenvectors,
    )


def _cosine_spectrum(
    *,
    charging_energy: float,
    josephson_energy: float,
    offset_charge: float,
    charge_cutoff: int,
) -> tuple[float, float]:
    """
    Calculate the lowest transition and signed anharmonicity.

    Parameters
    ----------
    charging_energy : float
        Charging energy `E_C / h` in GHz.
    josephson_energy : float
        Josephson energy `E_J / h` in GHz.
    offset_charge : float
        Dimensionless offset charge `n_g`.
    charge_cutoff : int
        Positive charge-basis cutoff used for diagonalization.

    Returns
    -------
    tuple[float, float]
        The 0-1 transition frequency and signed anharmonicity in GHz.

    Notes
    -----
    If the three lowest eigenenergies are `E_0`, `E_1`, and `E_2`, the returned
    values are `E_1 - E_0` and `E_2 - 2 E_1 + E_0`. The latter is negative for
    the transmon regime used by this model.
    """
    eigensystem = _charge_basis_eigensystem(
        charging_energy=charging_energy,
        josephson_energy=josephson_energy,
        offset_charge=offset_charge,
        charge_cutoff=charge_cutoff,
        dimension=3,
    )
    energies = eigensystem.eigenenergies
    frequency = float(energies[1] - energies[0])
    anharmonicity = float(energies[2] - 2 * energies[1] + energies[0])
    return frequency, anharmonicity


def _fit_cosine_parameters(transmon: Transmon) -> tuple[float, float]:
    """
    Fit positive charging and Josephson energies to measured spectral data.

    Parameters
    ----------
    transmon : Transmon
        Cosine-model specification containing the target 0-1 frequency,
        signed anharmonicity, offset charge, and charge-basis cutoff.

    Returns
    -------
    tuple[float, float]
        Fitted `E_C / h` and `E_J / h`, respectively, in GHz.

    Raises
    ------
    ValueError
        If no charge cutoff is available or the numerical fit does not
        reproduce both target spectral quantities to the required tolerance.

    Notes
    -----
    The usual transmon estimates identify `E_C / h` with the magnitude of the
    negative anharmonicity and approximate the transition frequency by
    `sqrt(8 E_J E_C) / h - E_C / h`. They provide only the initial point.
    Optimization is performed in logarithmic energy variables so that both
    fitted energies remain positive. Each residual evaluation uses the
    finite-charge-basis eigenspectrum at the configured `n_g`, reduced modulo
    one to `[-0.5, 0.5)`. Consequently, the fitted parameters and their
    accuracy remain cutoff dependent.
    """
    charge_cutoff = transmon.charge_cutoff
    if charge_cutoff is None:
        raise ValueError("A cosine transmon requires charge_cutoff.")

    initial_charging_energy = -transmon.anharmonicity
    initial_josephson_energy = (transmon.frequency + initial_charging_energy) ** 2 / (
        8 * initial_charging_energy
    )

    def residuals(log_energies: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return relative spectral residuals for log-space energy parameters."""
        charging_energy, josephson_energy = np.exp(log_energies)
        frequency, anharmonicity = _cosine_spectrum(
            charging_energy=float(charging_energy),
            josephson_energy=float(josephson_energy),
            offset_charge=transmon.offset_charge,
            charge_cutoff=charge_cutoff,
        )
        return np.array(
            [
                (frequency - transmon.frequency) / transmon.frequency,
                (anharmonicity - transmon.anharmonicity) / abs(transmon.anharmonicity),
            ]
        )

    fit = least_squares(
        residuals,
        np.log([initial_charging_energy, initial_josephson_energy]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=200,
    )
    if not fit.success or np.max(np.abs(fit.fun)) > 1e-10:
        raise ValueError(f"Cosine transmon parameter fit failed: {fit.message}")

    charging_energy, josephson_energy = np.exp(fit.x)
    return float(charging_energy), float(josephson_energy)


def compile_cosine_transmon(transmon: Transmon) -> CompiledCosineTransmon:
    """
    Compile a cosine transmon in its retained local energy eigenbasis.

    Parameters
    ----------
    transmon : Transmon
        Cosine-model source specification. `dimension` selects the number of
        low-energy eigenstates retained after charge-basis diagonalization.

    Returns
    -------
    CompiledCosineTransmon
        Ground-referenced Hamiltonian, projected interaction and lowering
        operators, phenomenological collapse operators, and the retained
        charge-basis eigensystem used to construct them.

    Raises
    ------
    ValueError
        If the charge cutoff is unavailable, spectral fitting fails, or the
        projected 0-1 charge matrix element is numerically zero.

    Notes
    -----
    The charge-basis Hamiltonian is diagonalized before truncation, so the
    returned basis consists of local cosine-transmon energy eigenstates rather
    than charge states. The ground energy is removed and the retained energies
    are converted from GHz to `H / hbar` in rad/ns.

    Eigenvector phases are chosen recursively so that adjacent projected charge
    matrix elements are real and positive where numerically resolvable. The
    full projected relative-charge operator `n - n_g` is normalized by the
    magnitude of its 0-1 matrix element. This keeps a coupling strength
    calibrated as the adjacent exchange rate compatible with the Duffing
    convention. Diagonal and nonadjacent matrix elements remain available for
    later full-coupling and counter-rotating modeling choices.

    The lowering operator retains only adjacent energy-level transitions from
    the normalized charge operator. The collapse operators use this adjacent
    energy-lowering component for phenomenological relaxation and the
    independent energy-level index operator for phenomenological pure
    dephasing. A microscopic open-system model would additionally resolve the
    physical noise source and its transition-frequency-dependent spectrum.
    """
    if transmon.charge_cutoff is None:
        raise ValueError("A cosine transmon requires charge_cutoff.")

    charging_energy, josephson_energy = _fit_cosine_parameters(transmon)
    eigensystem = _charge_basis_eigensystem(
        charging_energy=charging_energy,
        josephson_energy=josephson_energy,
        offset_charge=transmon.offset_charge,
        charge_cutoff=transmon.charge_cutoff,
        dimension=transmon.dimension,
    )
    charges = eigensystem.charge_numbers
    eigenenergies = eigensystem.eigenenergies
    eigenvectors = eigensystem.eigenvectors

    # Transform the relative-charge operator n - n_g from the finite charge
    # basis into the retained energy basis: n_energy = V.conj().T @ n_charge @ V.
    relative_charges = charges - eigensystem.offset_charge
    projected_charge = eigenvectors.conj().T @ (
        relative_charges[:, np.newaxis] * eigenvectors
    )
    # Remove the roundoff-level anti-Hermitian residue introduced numerically.
    projected_charge = 0.5 * (projected_charge + projected_charge.conj().T)

    # Normalize <0|n - n_g|1> to one so Coupling.strength keeps its established
    # meaning as the adjacent-level exchange rate in GHz. Subtracting n_g does
    # not change this off-diagonal matrix element.
    charge_01 = abs(projected_charge[0, 1])
    if charge_01 < 1e-14:
        raise ValueError(
            f"The 0-1 charge matrix element for {transmon.label} is numerically zero."
        )
    normalized_charge = projected_charge / charge_01

    # Matrix element (m, m + 1) maps |m + 1> to |m>. Keep only this first upper
    # diagonal so every component lowers the rotating-frame level number by one.
    lowering_matrix = np.zeros_like(normalized_charge)
    levels = np.arange(transmon.dimension - 1)
    lowering_matrix[levels, levels + 1] = normalized_charge[levels, levels + 1]

    # Reference the spectrum to the ground level and convert E / h in GHz to
    # H / hbar in rad/ns for QuTiP time evolution.
    energies = 2 * np.pi * (eigenenergies - eigenenergies[0])
    dims = [[transmon.dimension], [transmon.dimension]]
    lowering_operator = qt.Qobj(lowering_matrix, dims=dims)
    return CompiledCosineTransmon(
        source=transmon,
        hamiltonian=qt.Qobj(np.diag(energies), dims=dims),
        interaction_operator=qt.Qobj(normalized_charge, dims=dims),
        lowering_operator=lowering_operator,
        collapse_operators=build_collapse_operators(transmon, lowering_operator),
        charge_basis=eigensystem,
    )
