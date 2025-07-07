from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def hamiltonian(
    qubit_frequency: float,
    qubit_anharmonicity: float,
    drive_frequency: float,
    drive_amplitude: complex,
    dimension: int,
) -> NDArray[np.complex128]:
    """
    Generate a Hamiltonian for a single qubit with a drive.

    Parameters
    ----------
    qubit_frequency : float
        The frequency of the qubit.
    qubit_anharmonicity : float
        The anharmonicity of the qubit.
    drive_frequency : float
        The frequency of the drive.
    drive_amplitude : complex
        The complex amplitude of the drive.
    dimension : int
        The dimension of the Hilbert space.

    Returns
    -------
    NDArray[np.complex128]
        The Hamiltonian matrix.
    """
    H0 = np.diag(
        [
            ((qubit_frequency - drive_frequency) - 0.5 * qubit_anharmonicity) * i
            + 0.5 * qubit_anharmonicity * i**2
            for i in range(dimension)
        ]
    )
    H1 = np.zeros((dimension, dimension), dtype=np.complex128)

    for i in range(dimension - 1):
        H1[i, i + 1] = np.conj(drive_amplitude) * np.sqrt(i + 1)
        H1[i + 1, i] = drive_amplitude * np.sqrt(i + 1)

    return H0 + H1


def adiabatic_coefficients(
    qubit_frequency: float,
    qubit_anharmonicity: float,
    drive_frequency: float,
    drive_waveform: NDArray[np.complex128],
    sampling_period: float,
    dimension: int,
) -> NDArray[np.float64]:
    """
    Calculate the adiabatic coefficients for a single qubit with a drive.

    Parameters
    ----------
    qubit_frequency : float
        The frequency of the qubit.
    qubit_anharmonicity : float
        The anharmonicity of the qubit.
    drive_frequency : float
        The frequency of the drive.
    drive_waveform : NDArray[np.complex128]
        The complex amplitude of the drive at each time step.
    sampling_period : float
        The time interval between samples in the drive waveform.
    dimension : int
        The dimension of the Hilbert space.

    Returns
    -------
    NDArray[np.float64]
        The adiabatic coefficients for the drive waveform.
    """
    H = np.array(
        [
            hamiltonian(
                dimension=dimension,
                qubit_frequency=qubit_frequency,
                qubit_anharmonicity=qubit_anharmonicity,
                drive_frequency=drive_frequency,
                drive_amplitude=amp,
            )
            for amp in drive_waveform
        ]
    )
    dHdt = np.gradient(H, sampling_period, axis=0, edge_order=2)
    E, V = np.linalg.eigh(H)
    V_dag = np.swapaxes(V.conj(), -1, -2)
    M = V_dag @ dHdt @ V
    dE = E[..., :, None] - E[..., None, :]
    diag_mask = np.eye(dimension, dtype=bool)[None, :, :]
    dE = np.where(diag_mask, np.inf, dE)
    A = np.abs(M) / dE**2
    A_total = np.sum(A, axis=(-2, -1))
    return A_total


def rotate(
    data: ArrayLike,
    angle: float,
) -> NDArray[np.complex128]:
    """
    Rotate complex data points by a specified angle.

    Parameters
    ----------
    data : ArrayLike
        Array of complex data points to be rotated.
    angle : float
        Angle in radians by which to rotate the data points.

    Returns
    -------
    NDArray[np.complex128]
        Rotated complex data points.
    """
    points = np.array(data)
    rotated_points = points * np.exp(1j * angle)
    return rotated_points


def calc_1q_gate_coherence_limit(
    *,
    gate_time: float,
    t1: float,
    t2: float,
) -> dict[str, float]:
    """
    Calculate the coherence limit for a 1-qubit gate.

    Parameters
    ----------
    gate_time : float
        The time duration of the gate.
    t1 : float
        The T1 time of the qubit.
    t2 : float
        The T2 time of the qubit.

    Returns
    -------
    dict[str, float]
        A dictionary containing the error and fidelity of the gate.
    """
    error = 1 / 6 * (3 - 2 * np.exp(-gate_time / t2) - np.exp(-gate_time / t1))
    fidelity = 1 - error
    return {
        "error": error,
        "fidelity": fidelity,
    }


def calc_2q_gate_coherence_limit(
    *,
    gate_time: float,
    t1: tuple[float, float] | float,
    t2: tuple[float, float] | float,
) -> dict[str, float]:
    """
    Calculate the coherence limit for a 2-qubit gate.

    Parameters
    ----------
    gate_time : float
        The time duration of the gate.
    t1 : tuple[float, float]
        The T1 times of the qubits.
    t2 : tuple[float, float]
        The T2 times of the qubits.

    Returns
    -------
    dict[str, float]
        A dictionary containing the error and fidelity of the gate.
    """
    t1 = t1 if isinstance(t1, tuple) else (t1, t1)
    t2 = t2 if isinstance(t2, tuple) else (t2, t2)

    N = 2
    term1 = 15
    term2 = sum(
        2 * np.exp(-gate_time / t2[i]) + np.exp(-gate_time / t1[i]) for i in range(N)
    )
    term3 = np.exp(-gate_time * (1 / t1[0] + 1 / t1[1]))
    term4 = 4 * np.exp(-gate_time * (1 / t2[0] + 1 / t2[1]))
    term5 = 2 * np.exp(-gate_time * (1 / t1[0] + 1 / t2[1]))
    term6 = 2 * np.exp(-gate_time * (1 / t2[0] + 1 / t1[1]))

    error = 1 / 20 * (term1 - term2 - term3 - term4 - term5 - term6)
    fidelity = 1 - error
    return {
        "error": error,
        "fidelity": fidelity,
    }
