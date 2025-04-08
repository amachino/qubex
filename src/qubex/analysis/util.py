from __future__ import annotations

import numpy as np


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
