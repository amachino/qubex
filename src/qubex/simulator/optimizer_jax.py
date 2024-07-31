from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial
from typing import Final, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go
import qutip as qt
from IPython.display import display
from jax import Array
from jax.scipy.linalg import expm
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .simulator import System


@dataclass
class OptimizationResult:
    times: NDArray[np.float64]
    waveforms: dict[str, NDArray[np.complex128]]
    history: NDArray[np.float64]

    def plot_waveforms(self):
        for target, waveform in self.waveforms.items():
            dt = self.times[1] - self.times[0]
            times = np.append(self.times, self.times[-1] + dt)
            waveform = waveform / (2 * np.pi) * 1e3
            real = np.append(waveform.real, waveform.real[-1])
            imag = np.append(waveform.imag, waveform.imag[-1])

            waveform = waveform / (2 * np.pi) * 1e3
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=real,
                    mode="lines",
                    name="I",
                    line_shape="hv",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=imag,
                    mode="lines",
                    name="Q",
                    line_shape="hv",
                )
            )
            fig.update_layout(
                title=f"Waveform : {target}",
                xaxis_title="Time (ns)",
                yaxis_title="Amplitude (MHz)",
            )
            fig.show()

    def plot_history(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=self.history,
                mode="lines",
            )
        )
        fig.update_layout(
            title="Optimization history",
            xaxis_title="Number of iterations",
            yaxis_title="Loss function",
            yaxis=dict(
                type="log",
            ),
        )
        fig.show()


class PulseOptimizer:

    def __init__(
        self,
        *,
        quantum_system: System,
        target_unitary: ArrayLike | qt.Qobj,
        control_qubits: Sequence[str],
        segment_count: int,
        segment_width: float,
        max_rabi_frequency: float,
    ):
        system_hamiltonian = qt.Qobj(quantum_system.hamiltonian)
        target_unitary = qt.Qobj(target_unitary)

        print("System Hamiltonian")
        display(system_hamiltonian)
        print("Target Unitary")
        display(target_unitary)

        if not system_hamiltonian.isherm:
            raise ValueError("Hamiltonian must be Hermitian.")

        if not target_unitary.isunitary:
            raise ValueError("Target unitary must be unitary.")

        if system_hamiltonian.shape[0] != target_unitary.shape[0]:
            raise ValueError(
                "Hamiltonian and target unitary must have the same dimension."
            )

        self.quantum_system: Final = quantum_system
        self.control_qubits: Final = control_qubits
        self.segment_count: Final = segment_count
        self.segment_duration: Final = segment_width
        self.duration: Final = segment_count * segment_width
        self.max_rabi_rate: Final = 2 * np.pi * max_rabi_frequency
        self.system_hamiltonian: Final = jnp.asarray(system_hamiltonian.full())
        self.dimension: Final = system_hamiltonian.shape[0]
        self.identity: Final = jnp.eye(self.dimension)
        self.target_unitary: Final = jnp.asarray(target_unitary.full())
        self.target_unitary_dagger: Final = self.target_unitary.conj().T
        self.jacobian: Final = jax.jit(jax.grad(self.loss_fn))

    @cache
    def a(self, target) -> Array:
        return jnp.asarray(self.quantum_system.lowering_operator(target).full())

    @cache
    def a_dag(self, target) -> Array:
        return self.a(target).conj().T

    @cache
    def X(self, target) -> Array:
        return 0.5 * (self.a_dag(target) + self.a(target))

    @cache
    def Y(self, target) -> Array:
        return 0.5j * (self.a_dag(target) - self.a(target))

    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params: dict[str, Array]) -> Array:
        dt = self.segment_duration
        U = self.identity
        for index in range(self.segment_count):
            H = self.system_hamiltonian
            for target, iq_array in params.items():
                I, Q = iq_array[index]
                H += I * self.X(target) + Q * self.Y(target)
            U = expm(-1j * H * dt) @ U

        V = self.target_unitary_dagger
        D = self.dimension
        return 1 - jnp.abs((V @ U).trace() / D) ** 2

    def optimize(
        self,
        *,
        learning_rate: float = 1e-3,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> OptimizationResult:
        shape = (self.segment_count, 2)
        params = {
            target: jax.random.uniform(
                key=jax.random.key(index),
                shape=shape,
                minval=-self.max_rabi_rate,
                maxval=self.max_rabi_rate,
            )
            for index, target in enumerate(self.control_qubits)
        }
        solver = optax.adam(learning_rate=learning_rate)
        opt_state = solver.init(params)

        loss_history = []
        for _ in range(max_iterations):
            grad = self.jacobian(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = optax.projections.projection_box(
                params,
                lower={target: -self.max_rabi_rate for target in self.control_qubits},
                upper={target: self.max_rabi_rate for target in self.control_qubits},
            )
            loss = self.loss_fn(params)
            loss_history.append(loss)
            if loss < tolerance:
                break

        result = OptimizationResult(
            times=np.linspace(0, self.duration, self.segment_count + 1),
            waveforms={
                target: np.asarray([iq[0] + 1j * iq[1] for iq in iq_array])  # type: ignore
                for target, iq_array in params.items()  # type: ignore
            },
            history=np.array(loss_history),
        )
        result.plot_history()
        return result
