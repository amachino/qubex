"""Optimize piecewise-constant control pulses with JAX and Optax."""

from __future__ import annotations

import importlib
import logging
from functools import cached_property
from typing import Any, Final

import numpy as np
import qutip as qt
from typing_extensions import deprecated

from qxsimulator.system import QuantumSystem

from .optimization_result import OptimizationResult

logger = logging.getLogger(__name__)

Array = Any


@deprecated("PulseOptimizer is deprecated and will be removed in a future release.")
class PulseOptimizer:
    """
    Optimize piecewise-constant I/Q controls for a target unitary.

    Parameters
    ----------
    quantum_system : QuantumSystem
        Quantum system whose full Hamiltonian defines the closed-system
        dynamics used by the optimizer.
    target_unitary : qt.Qobj
        Full-system target unitary. Its matrix dimension must match the system
        Hamiltonian.
    initial_state : qt.Qobj
        Initial full-system ket. A value that is not a `qt.Qobj` is resolved by
        `quantum_system.state` before it is stored.
    control_frequencies : dict[str, float]
        Nonempty mapping from control-target labels to cyclic drive frequencies
        in GHz. Each label must identify an object in `quantum_system`.
    segment_count : int
        Positive number of piecewise-constant control segments.
    segment_width : float
        Positive duration of each segment in ns.
    max_rabi_frequency : float
        Nonnegative component-wise bound for the cyclic I and Q amplitudes in
        GHz. The complex envelope can therefore have a larger magnitude.

    Raises
    ------
    ValueError
        If `target_unitary` is not unitary or its matrix dimension differs from
        the system Hamiltonian dimension.

    Notes
    -----
    The system Hamiltonian represents `H / hbar` in rad/ns. Optimized I/Q
    amplitudes are also stored in rad/ns, while constructor frequencies use
    cyclic GHz and segment durations use ns. The shared rotating-frame
    frequency is the arithmetic mean of `control_frequencies`. In method
    docstrings, `D` denotes the full Hilbert-space dimension.

    Construction logs the system Hamiltonian, target unitary, and initial
    state at info level.
    """

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

        jax, jnp, optax, expm = _load_optimizer_dependencies()

        logger.info("System Hamiltonian\n%s", system_hamiltonian)
        logger.info("Target Unitary\n%s", target_unitary)
        logger.info("Initial State\n%s", initial_state)

        self.quantum_system: Final = quantum_system
        self.target_unitary: Final = target_unitary
        self.initial_state: Final = initial_state
        self.control_frequencies: Final = control_frequencies
        self.segment_count: Final = segment_count
        self.segment_duration: Final = segment_width
        self.max_rabi_frequency: Final = max_rabi_frequency
        self._jax: Final = jax
        self._jnp: Final = jnp
        self._optax: Final = optax
        self._expm: Final = expm
        self.jacobian: Final = jax.jit(jax.grad(self.loss_fn))

    @cached_property
    def system_hamiltonian(self) -> Array:
        """
        Return the full-system Hamiltonian as a dense JAX array.

        Returns
        -------
        Array
            Angular-frequency matrix `H / hbar` in rad/ns with shape `(D, D)`.
        """
        return self._jnp.asarray(self.quantum_system.hamiltonian.full())

    @cached_property
    def rotating_system_hamiltonian(self) -> Array:
        """
        Return the Hamiltonian in the shared number-operator rotating frame.

        Returns
        -------
        Array
            Rotating-frame angular-frequency matrix in rad/ns with shape
            `(D, D)`.
        """
        H = self.system_hamiltonian
        for target in self.quantum_system.object_labels:
            N = self.number_operator(target)
            H -= 2 * np.pi * self.frame_frequency * N
        return H

    @cached_property
    def target_unitary_dagger(self) -> Array:
        """
        Return the conjugate transpose of the target unitary.

        Returns
        -------
        Array
            Dense JAX array with shape `(D, D)`.
        """
        return self._jnp.asarray(self.target_unitary.full()).conj().T

    @cached_property
    def target_state(self) -> Array:
        """
        Return the initial state transformed by the target unitary.

        Returns
        -------
        Array
            Dense JAX representation of `target_unitary @ initial_state`.
        """
        return self._jnp.asarray(self.target_unitary @ self.initial_state)

    @cached_property
    def dimension(self) -> int:
        """
        Return the full Hilbert-space dimension.

        Returns
        -------
        int
            Matrix dimension `D` of the system Hamiltonian.
        """
        return self.system_hamiltonian.shape[0]

    @cached_property
    def dimensions(self) -> Any:
        """
        Return the QuTiP tensor-dimension metadata for system operators.

        Returns
        -------
        Any
            Nested dimensions copied from `quantum_system.hamiltonian.dims`.
        """
        return self.quantum_system.hamiltonian.dims

    @cached_property
    def identity(self) -> Array:
        """
        Return the full-system identity operator as a JAX array.

        Returns
        -------
        Array
            Identity matrix with shape `(D, D)`.
        """
        return self._jnp.eye(self.dimension)

    @cached_property
    def duration(self) -> float:
        """
        Return the total control duration.

        Returns
        -------
        float
            Product of segment count and segment duration, in ns.
        """
        return self.segment_count * self.segment_duration

    @cached_property
    def control_qubits(self) -> list[str]:
        """
        Return the control-target labels in mapping insertion order.

        Returns
        -------
        list[str]
            Labels from `control_frequencies`.
        """
        return list(self.control_frequencies.keys())

    @cached_property
    def frame_frequency(self) -> float:
        """
        Return the cyclic frequency of the shared rotating frame.

        Returns
        -------
        float
            Arithmetic mean of the control frequencies, in GHz.
        """
        return np.mean(list(self.control_frequencies.values())).astype(float)

    @cached_property
    def relative_frequencies(self) -> dict[str, float]:
        """
        Return signed control frequencies relative to the shared frame.

        Returns
        -------
        dict[str, float]
            Mapping from each target to `frequency - frame_frequency`, in GHz.
        """
        return {
            target: frequency - self.frame_frequency
            for target, frequency in self.control_frequencies.items()
        }

    @cached_property
    def max_rabi_rate(self) -> float:
        """
        Return the component-wise angular-amplitude bound.

        Returns
        -------
        float
            Value of `2 * pi * max_rabi_frequency` in rad/ns.
        """
        return 2 * np.pi * self.max_rabi_frequency

    @cached_property
    def lower_bound(self) -> dict[str, float]:
        """
        Return the lower component-wise control bounds.

        Returns
        -------
        dict[str, float]
            Per-target lower bounds in rad/ns.
        """
        return dict.fromkeys(self.control_frequencies, -self.max_rabi_rate)

    @cached_property
    def upper_bound(self) -> dict[str, float]:
        """
        Return the upper component-wise control bounds.

        Returns
        -------
        dict[str, float]
            Per-target upper bounds in rad/ns.
        """
        return dict.fromkeys(self.control_frequencies, self.max_rabi_rate)

    def lowering_operator(self, target: str) -> Array:
        """
        Return the compiled lowering operator for a target.

        Parameters
        ----------
        target : str
            Object label in the quantum system.

        Returns
        -------
        Array
            Full-system operator as a dense JAX array with shape `(D, D)`.

        Raises
        ------
        ValueError
            If no system object has the requested label.
        """
        a = self.quantum_system.get_lowering_operator(target)
        return self._jnp.asarray(a.full())

    def raising_operator(self, target: str) -> Array:
        """
        Return the compiled raising operator for a target.

        Parameters
        ----------
        target : str
            Object label in the quantum system.

        Returns
        -------
        Array
            Full-system operator as a dense JAX array with shape `(D, D)`.

        Raises
        ------
        ValueError
            If no system object has the requested label.
        """
        ad = self.quantum_system.get_raising_operator(target)
        return self._jnp.asarray(ad.full())

    def number_operator(self, target: str) -> Array:
        """
        Return the local number operator embedded in the full system.

        Parameters
        ----------
        target : str
            Object label in the quantum system.

        Returns
        -------
        Array
            Full-system operator as a dense JAX array with shape `(D, D)`.

        Raises
        ------
        ValueError
            If no system object has the requested label.
        """
        N = self.quantum_system.get_number_operator(target)
        return self._jnp.asarray(N.full())

    def loss_fn(self, params: dict[str, Array]) -> Array:
        """
        Compute unitary infidelity from the normalized trace overlap.

        Parameters
        ----------
        params : dict[str, Array]
            Per-target I/Q arrays with shape `(segment_count, 2)`, in rad/ns.

        Returns
        -------
        Array
            Scalar dimensionless infidelity for the resulting propagator.
        """
        U = self.evolve(params)
        return self.unitary_infidelity(U)

    def evolve(self, params: dict[str, Array]) -> Array:
        """
        Evolve the closed system under piecewise-constant controls.

        Parameters
        ----------
        params : dict[str, Array]
            Per-target arrays with shape `(segment_count, 2)`. Each row stores
            the I and Q angular amplitudes for one segment, in rad/ns.

        Returns
        -------
        Array
            End-of-pulse propagator with shape `(D, D)`.

        Notes
        -----
        The static term is `rotating_system_hamiltonian`. Each segment consumes
        one I/Q row per target and holds the resulting drive Hamiltonian
        constant for `segment_duration`. Collapse operators are excluded.
        """
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
                H += 0.5 * (ad * Omega + a * self._jnp.conj(Omega))
            U = self._expm(-1j * H * dt) @ U
        return U

    def unitary_infidelity(self, U: Array) -> Array:
        """
        Compute global-phase-insensitive trace-overlap infidelity.

        Parameters
        ----------
        U : Array
            Full-system propagator with shape `(D, D)`.

        Returns
        -------
        Array
            Scalar value
            `1 - abs(trace(target_unitary_dagger @ U) / D) ** 2`. The value is
            dimensionless.

        Notes
        -----
        This quantity is the complement of normalized trace overlap squared; it
        is not average gate infidelity.
        """
        D = self.dimension
        V = self.target_unitary_dagger
        return 1 - self._jnp.abs((V @ U).trace() / D) ** 2

    def state_infidelity(self, psi: Array) -> Array:
        """
        Compute pure-state infidelity relative to the target state.

        Parameters
        ----------
        psi : Array
            Normalized state array compatible with `target_state`.

        Returns
        -------
        Array
            Scalar value `1 - abs(vdot(target_state, psi)) ** 2`. The value is
            dimensionless.
        """
        phi = self.target_state
        return 1 - self._jnp.abs(self._jnp.vdot(phi, psi)) ** 2

    def random_params(self, key: Array) -> dict[str, Array]:
        """
        Generate uniformly distributed initial I/Q parameters.

        Parameters
        ----------
        key : Array
            JAX pseudorandom key controlling reproducible initialization.

        Returns
        -------
        dict[str, Array]
            Per-target arrays with shape `(segment_count, 2)`, in rad/ns. Each
            component lies in `[-max_rabi_rate, max_rabi_rate)`.
        """
        return {
            target: self._jax.random.uniform(
                key=self._jax.random.split(key)[0],
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
        """
        Optimize I/Q segments with Adam against trace-overlap infidelity.

        Parameters
        ----------
        learning_rate : float, optional
            Positive Adam learning rate. The default is `1e-3`.
        max_iterations : int, optional
            Positive maximum number of Adam updates. The default is `1000`.
        tolerance : float, optional
            Nonnegative early-stopping threshold for dimensionless infidelity.
            Optimization stops when the loss is below this value. The default
            is `1e-6`.
        seed : int, optional
            Seed for JAX parameter initialization. The default is `0`.

        Returns
        -------
        OptimizationResult
            Optimized controls, final propagator and state, and the loss
            history recorded after each update.

        Notes
        -----
        Each update is projected onto the component-wise amplitude bounds. The
        method displays the loss history through the configured Plotly renderer
        before returning.
        """
        key = self._jax.random.PRNGKey(seed)
        params = self.random_params(key)

        solver = self._optax.adam(learning_rate=learning_rate)
        opt_state = solver.init(params)

        loss_history = []
        for _ in range(max_iterations):
            grad = self.jacobian(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = self._optax.apply_updates(params, updates)
            params = self._optax.projections.projection_box(
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

        state = unitary @ self.initial_state

        result = OptimizationResult(
            params=params,
            infidelity=infidelity,
            unitary=unitary,
            state=state,
            times=np.linspace(0, self.duration, self.segment_count + 1),
            waveforms={
                target: np.asarray([iq[0] + 1j * iq[1] for iq in iq_array])
                for target, iq_array in params.items()
            },
            history=np.array(loss_history),
        )
        result.plot_history()
        return result


def _load_optimizer_dependencies() -> tuple[Any, Any, Any, Any]:
    """Load dependencies retained only for the deprecated optimizer."""
    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        optax = importlib.import_module("optax")
        linalg = importlib.import_module("jax.scipy.linalg")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PulseOptimizer is deprecated and its JAX and Optax dependencies "
            "are no longer installed with qxsimulator. Install them separately "
            "to continue using PulseOptimizer during the compatibility period."
        ) from error
    return jax, jnp, optax, linalg.expm
