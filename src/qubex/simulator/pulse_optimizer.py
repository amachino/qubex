from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property, partial
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go
import qutip as qt
from IPython.display import display
from jax import Array
from jax.scipy.linalg import expm
from numpy.typing import NDArray

from .quantum_system import QuantumSystem


@dataclass
class OptimizationResult:
    params: optax.Params
    infidelity: float
    unitary: qt.Qobj
    state: qt.Qobj
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
        quantum_system: QuantumSystem,
        target_unitary: qt.Qobj,
        initial_state: qt.Qobj,
        control_frequencies: dict[str, float],
        segment_count: int,
        segment_width: float,
        max_rabi_frequency: float,
    ):
        system_hamiltonian = quantum_system.hamiltonian

        if not target_unitary.isunitary:
            raise ValueError("Target unitary must be unitary.")

        if system_hamiltonian.shape[0] != target_unitary.shape[0]:
            raise ValueError(
                "Hamiltonian and target unitary must have the same dimension."
            )

        if not isinstance(initial_state, qt.Qobj):
            initial_state = quantum_system.state(initial_state)

        print("System Hamiltonian")
        display(system_hamiltonian)
        print("Target Unitary")
        display(target_unitary)
        print("Initial State")
        display(initial_state)

        self.quantum_system: Final = quantum_system
        self.target_unitary: Final = target_unitary
        self.initial_state: Final = initial_state
        self.control_frequencies: Final = control_frequencies
        self.segment_count: Final = segment_count
        self.segment_duration: Final = segment_width
        self.max_rabi_frequency: Final = max_rabi_frequency
        self.jacobian: Final = jax.jit(jax.grad(self.loss_fn))

    @cached_property
    def system_hamiltonian(self) -> Array:
        return jnp.asarray(self.quantum_system.hamiltonian.full())

    @cached_property
    def rotating_system_hamiltonian(self) -> Array:
        H = self.system_hamiltonian
        for target in self.quantum_system.object_labels:
            N = self.number_operator(target)
            H -= 2 * np.pi * self.frame_frequency * N
        return H

    @cached_property
    def target_unitary_dagger(self) -> Array:
        return jnp.asarray(self.target_unitary.full()).conj().T

    @cached_property
    def target_state(self) -> Array:
        return jnp.asarray(self.target_unitary * self.initial_state)

    @cached_property
    def dimension(self) -> int:
        return self.system_hamiltonian.shape[0]

    @cached_property
    def dimensions(self):
        return self.quantum_system.hamiltonian.dims

    @cached_property
    def identity(self) -> Array:
        return jnp.eye(self.dimension)

    @cached_property
    def duration(self) -> float:
        return self.segment_count * self.segment_duration

    @cached_property
    def control_qubits(self) -> list[str]:
        return list(self.control_frequencies.keys())

    @cached_property
    def frame_frequency(self) -> float:
        return np.mean(list(self.control_frequencies.values())).astype(float)

    @cached_property
    def relative_frequencies(self) -> dict[str, float]:
        return {
            target: frequency - self.frame_frequency
            for target, frequency in self.control_frequencies.items()
        }

    @cached_property
    def max_rabi_rate(self) -> float:
        return 2 * np.pi * self.max_rabi_frequency

    @cached_property
    def lower_bound(self) -> dict[str, float]:
        return {target: -self.max_rabi_rate for target in self.control_frequencies}

    @cached_property
    def upper_bound(self) -> dict[str, float]:
        return {target: self.max_rabi_rate for target in self.control_frequencies}

    @cache
    def lowering_operator(self, target) -> Array:
        a = self.quantum_system.get_lowering_operator(target)
        return jnp.asarray(a.full())

    @cache
    def raising_operator(self, target) -> Array:
        ad = self.quantum_system.get_raising_operator(target)
        return jnp.asarray(ad.full())

    @cache
    def number_operator(self, target) -> Array:
        N = self.quantum_system.get_number_operator(target)
        return jnp.asarray(N.full())

    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params: dict[str, Array]) -> Array:
        U = self.evolve(params)
        return self.unitary_infidelity(U)

    @partial(jax.jit, static_argnums=0)
    def evolve(self, params: dict[str, Array]) -> Array:
        dt = self.segment_duration
        U = self.identity
        for index in range(self.segment_count):
            H = self.rotating_system_hamiltonian
            for target, iq_array in params.items():
                a = self.lowering_operator(target)
                ad = self.raising_operator(target)
                delta = self.relative_frequencies[target]
                I, Q = iq_array[index]
                Omega = I + 1j * Q
                Omega = Omega * np.exp(-1j * delta * dt)
                H += 0.5 * (ad * Omega + a * jnp.conj(Omega))
            U = expm(-1j * H * dt) @ U
        return U

    @partial(jax.jit, static_argnums=0)
    def unitary_infidelity(self, U: Array) -> Array:
        D = self.dimension
        V = self.target_unitary_dagger
        return 1 - jnp.abs((V @ U).trace() / D) ** 2

    @partial(jax.jit, static_argnums=0)
    def state_infidelity(self, psi: Array) -> Array:
        phi = self.target_state
        return 1 - jnp.abs(jnp.vdot(phi, psi)) ** 2

    def random_params(self, key: Array) -> dict[str, Array]:
        return {
            target: jax.random.uniform(
                key=jax.random.split(key)[0],
                shape=(self.segment_count, 2),
                minval=-self.max_rabi_rate,
                maxval=self.max_rabi_rate,
            )
            for target in self.control_frequencies
        }

    def optimize(
        self,
        *,
        learning_rate: float = 1e-3,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        seed: int = 0,
    ) -> OptimizationResult:
        key = jax.random.PRNGKey(seed)
        params = self.random_params(key)

        solver = optax.adam(learning_rate=learning_rate)
        opt_state = solver.init(params)

        loss_history = []
        for _ in range(max_iterations):
            grad = self.jacobian(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = optax.projections.projection_box(
                params,
                lower=self.lower_bound,
                upper=self.upper_bound,
            )
            loss = self.loss_fn(params)
            loss_history.append(loss)
            if loss < tolerance:
                break

        infidelity = float(loss)
        unitary = qt.Qobj(np.asarray(self.evolve(params)), dims=self.dimensions)

        state = unitary * self.initial_state

        result = OptimizationResult(
            params=params,
            infidelity=infidelity,
            unitary=unitary,
            state=state,
            times=np.linspace(0, self.duration, self.segment_count + 1),
            waveforms={
                target: np.asarray([iq[0] + 1j * iq[1] for iq in iq_array])  # type: ignore
                for target, iq_array in params.items()  # type: ignore
            },
            history=np.array(loss_history),
        )
        result.plot_history()
        return result
