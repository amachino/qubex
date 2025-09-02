from __future__ import annotations

import warnings
from functools import reduce
from itertools import product

import cvxpy as cp
import numpy as np
import plotly.graph_objs as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from .visualization import save_figure_image


def calculate_expected_values(
    probabilities: dict[str, NDArray],
) -> dict[str, float]:
    n_qubits = len(next(iter(probabilities.keys())))
    dim = 2**n_qubits

    expected_values = {}
    for pauli_tuple in product(["I", "X", "Y", "Z"], repeat=n_qubits):
        pauli_label = "".join(pauli_tuple)

        basis_list = [p if p != "I" else "Z" for p in pauli_tuple]
        basis_label = "".join(basis_list)

        probs = probabilities[basis_label]

        total_exp = 0.0
        for i in range(dim):
            bit_array = [int(b) for b in f"{i:0{n_qubits}b}"]
            # For example, i = 5 (0b101) gives bit_array = [1, 0, 1]
            parity = 0
            for k, bit in enumerate(bit_array):
                if pauli_tuple[k] != "I":
                    # Only consider non-identity Pauli operators
                    parity += bit

            sign = (-1) ** parity
            total_exp += sign * probs[i]

        expected_values[pauli_label] = total_exp

    return expected_values


def create_density_matrix(
    probabilities: dict[str, NDArray],
    mle_fit: bool = True,
) -> NDArray:
    n_qubits = len(next(iter(probabilities.keys())))
    dim = 2**n_qubits
    expected_values = calculate_expected_values(probabilities)

    if mle_fit:
        rho = mle_fit_density_matrix(expected_values)
    else:
        paulis = {
            "I": np.array([[1, 0], [0, 1]], dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }
        rho = np.zeros((dim, dim), dtype=np.complex128)
        for pauli_label, exp_val in expected_values.items():
            # pauli_label: "XYZ" -> paulis["X"] ⊗ paulis["Y"] ⊗ paulis["Z"]
            op = reduce(np.kron, [paulis[p] for p in pauli_label])
            rho += exp_val * op
        rho /= dim
    return rho


def mle_fit_density_matrix(
    expected_values: dict[str, float],
) -> NDArray:
    paulis = {
        "I": np.array([[1, 0], [0, 1]], dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    label = list(expected_values.keys())[0]
    n = len(label)
    dim = 2**n

    A_list: list[NDArray] = []
    b_list: list[float] = []
    for basis, val in expected_values.items():
        op = reduce(np.kron, [paulis[p] for p in basis])
        A_list.append(op.reshape(1, -1).conj())
        b_list.append(val)
    A = np.vstack(A_list)
    b = np.array(b_list)

    rho = cp.Variable((dim, dim), hermitian=True)
    constraints = [rho >> 0, cp.trace(rho) == 1]
    objective = cp.Minimize(cp.sum_squares(A @ cp.vec(rho, order="F") - b))
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem.solve(solver=cp.SCS)

    if rho.value is None:
        raise RuntimeError("CVXPY failed to solve the MLE problem.")

    # Post-process: clip tiny negative eigenvalues
    eigvals, eigvecs = np.linalg.eigh(rho.value)
    eigvals_clipped = np.clip(eigvals, 0, None)
    rho_fixed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.conj().T
    rho_fixed /= np.trace(rho_fixed)

    return rho_fixed


def plot_ghz_state_tomography(
    rho: NDArray,
    qubits: list[str],
    fidelity: float,
    width: int,
    height: int,
    title: str | None = None,
    plot: bool = True,
    save_image: bool = False,
    file_name: str | None = None,
):
    n_qubits = len(qubits)
    dim = 2**n_qubits

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Re", "Im"),
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Heatmap(
            z=rho.real,
            zmin=-0.6,
            zmax=0.6,
            colorscale="RdBu_r",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=rho.imag,
            zmin=-0.6,
            zmax=0.6,
            colorscale="RdBu_r",
        ),
        row=1,
        col=2,
    )

    if n_qubits < 4:
        tickvals = np.arange(dim)
        ticktext = [f"{i:0{n_qubits}b}" for i in tickvals]
    else:
        tickvals = [0, 2**n_qubits - 1]
        ticktext = [f"{i:0{n_qubits}b}" for i in tickvals]
    tick_style = dict(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=0,
    )

    if n_qubits == 2:
        title = f"Bell state tomography: {'-'.join(qubits)}"
    else:
        title = f"GHZ state tomography: {'-'.join(qubits)}"

    fig.update_layout(
        title=dict(
            text=title,
            subtitle=dict(text=f"State fidelity: {fidelity * 100:.3f}%"),
        ),
        width=width,
        height=height,
        margin=dict(l=70, r=70, t=100, b=70),
    )
    fig.update_xaxes(tick_style, row=1, col=1)
    fig.update_yaxes(
        dict(**tick_style, autorange="reversed", scaleanchor="x1"),
        row=1,
        col=1,
    )
    fig.update_xaxes(tick_style, row=1, col=2)
    fig.update_yaxes(
        dict(**tick_style, autorange="reversed", scaleanchor="x2"),
        row=1,
        col=2,
    )

    if plot:
        fig.show()
        if fidelity is not None:
            print(f"State fidelity: {fidelity * 100:.3f}%")
    if save_image:
        if file_name is None:
            file_name = f"ghz_state_tomography_{'-'.join(qubits)}"
        save_figure_image(fig, file_name, width=width, height=height)

    return {
        "density_matrix": rho,
        "fidelity": fidelity,
        "figure": fig,
    }
