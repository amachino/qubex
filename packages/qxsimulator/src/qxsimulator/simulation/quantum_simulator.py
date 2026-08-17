"""Quantum simulator execution methods and compatibility exports."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import qutip as qt
from qxpulse import PulseSchedule
from typing_extensions import deprecated

from qxsimulator.system import (
    EvaluationSpace,
    QuantumSystem,
    UnitarySpecification,
)

from . import _pulse_schedule_adapter, _time_grid
from .control import Control
from .simulation_model import SimulationModel
from .simulation_result import FrameType, SimulationResult, SubspaceType

_ROTATING_COEFFICIENT = "amplitude * exp(-1j * angular_frequency * t)"
TargetUnitary: TypeAlias = qt.Qobj | UnitarySpecification

__all__ = [
    "Control",
    "FrameType",
    "QuantumSimulator",
    "SimulationModel",
    "SimulationResult",
    "SubspaceType",
    "TargetUnitary",
]


class QuantumSimulator:
    """Evolve a `QuantumSystem` under finite-duration control signals."""

    def __init__(
        self,
        system: QuantumSystem,
    ):
        """
        Initialize a simulator for a quantum system.

        Parameters
        ----------
        system : QuantumSystem
            System whose object models, couplings, rotating frames, and
            decoherence rates define the simulated dynamics.
        """
        self.system: Final = system

    def simulate(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        initial_state: qt.Qobj | dict | None = None,
        dt: float = 0.1,
        n_samples: int | None = None,
        compute_propagators: bool = True,
    ) -> SimulationResult:
        """
        Simulate closed-system dynamics by propagating the unitary operator.

        Piecewise-exponential steps approximate the Schrödinger evolution. When
        propagators are requested, cumulative propagators and density-matrix
        states are both returned. Otherwise only the density-matrix trajectory
        is retained.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        initial_state : qt.Qobj | dict | None, optional
            Full-system ket, density matrix, or labeled state specification
            accepted by `QuantumSystem.state`. If omitted, use the system
            ground state.
        dt : float, optional
            Finite, positive maximum propagation interval in ns. The default
            is 0.1. Control boundaries and requested output times may introduce
            shorter intervals.
        n_samples : int | None, optional
            Number of uniformly spaced trajectory points to return from zero
            through the common control duration. If specified, it must be at
            least 2 so the initial and final points are retained. If omitted,
            return every fixed-step integration point. A zero-duration
            trajectory always contains only its initial point.
        compute_propagators : bool, optional
            Whether to compute and return cumulative Hilbert-space
            propagators. The default is `True`.

        Returns
        -------
        SimulationResult
            The density-matrix state trajectory and, when requested, the
            cumulative propagator trajectory at the returned times.

        Raises
        ------
        ValueError
            If no control is supplied, control durations differ, `dt` is not
            finite and positive, `n_samples` is less than 2, or the initial
            state is not a ket or density matrix with system dimensions.

        Notes
        -----
        Hamiltonians use angular-frequency units of rad/ns and the rotating
        frames defined by the system objects. The integration grid combines a
        uniform grid with every control boundary and requested output time.
        Within each interval, piecewise-constant control amplitudes are selected
        at the left endpoint, while continuously time-dependent carrier and
        coupling terms are evaluated at the midpoint.

        Couplings retain exchange terms only, and controls retain co-rotating
        drive terms only. These coupling and drive rotating-wave
        approximations are independent of each object's local Hamiltonian
        model.

        `Control.frame_shifts` and `Control.final_frame_shift` are logical-frame
        metadata and are not applied as physical evolution to states or
        propagators. A zero-duration control therefore returns only its initial
        physical state.
        """
        controls = _prepare_controls(controls)
        initial_state = _prepare_initial_state(self.system, initial_state)

        integration_times = _time_grid.create_integration_grid(controls, dt)
        output_times, evolution_times, output_indices = _prepare_trajectory_times(
            integration_times,
            n_samples,
        )
        delta_times = np.diff(evolution_times)
        midpoints = evolution_times[:-1] + delta_times / 2
        drive_coefficients: list[npt.NDArray[np.complex128]] = []
        for control in controls:
            frame_frequency = self.system.get_object(control.target).frequency
            detuning = 2 * np.pi * (control.frequency - frame_frequency)
            samples = control.get_samples(evolution_times[:-1])
            drive_coefficients.append(
                0.5 * samples * np.exp(-1j * detuning * midpoints)
            )

        targets = {control.target for control in controls}
        lowering_operators = {
            target: self.system.get_lowering_operator(target) for target in targets
        }
        raising_operators = {
            target: self.system.get_raising_operator(target) for target in targets
        }

        rho0 = qt.ket2dm(initial_state) if initial_state.isket else initial_state
        states = [rho0]
        propagators = [self.system.identity_matrix] if compute_propagators else []
        for idx, (midpoint, delta_time) in enumerate(
            zip(midpoints, delta_times, strict=True)
        ):
            H = self.system.get_rotating_hamiltonian(midpoint)
            for control, coefficients in zip(
                controls,
                drive_coefficients,
                strict=True,
            ):
                gamma = coefficients[idx]
                target = control.target
                H_ctrl = (
                    gamma * raising_operators[target]
                    + np.conj(gamma) * lowering_operators[target]
                )
                H += H_ctrl
            step_propagator = (-1j * H * delta_time).expm()
            if compute_propagators:
                propagator = step_propagator @ propagators[-1]
                propagators.append(propagator)
                states.append(propagator @ rho0 @ propagator.dag())
            else:
                state = step_propagator @ states[-1] @ step_propagator.dag()
                states.append(state)

        states = [states[index] for index in output_indices]
        propagators = (
            [propagators[index] for index in output_indices] if propagators else []
        )

        return SimulationResult(
            system=self.system,
            controls=controls,
            times=output_times,
            states=states,
            propagators=propagators,
        )

    def mesolve(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        initial_state: qt.Qobj | dict | None = None,
        n_samples: int | None = None,
        compute_propagators: bool = False,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Solve open-system dynamics with QuTiP's `mesolve`.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        initial_state : qt.Qobj | dict | None, optional
            Full-system ket, density matrix, or labeled state specification
            accepted by `QuantumSystem.state`. If omitted, use the system
            ground state. Kets are converted to density matrices.
        n_samples : int | None, optional
            Number of uniformly spaced trajectory points to return from zero
            through the common control duration. If specified, it must be at
            least 2. If omitted, return every control-boundary point. A
            zero-duration trajectory always contains only its initial point.
        compute_propagators : bool, optional
            Whether to compute the superoperator propagator trajectory and
            derive density matrices from it. The default is `False`.
        options : dict[str, Any] | None, optional
            Options passed to QuTiP's `mesolve`. `max_step` is measured in ns.
            Qubex derives `max_step` and `nsteps` when they are omitted; other
            settings use QuTiP defaults. The input dictionary is not mutated.
        **kwargs : Any
            Backward-compatibility keywords. The legacy `dt` keyword is
            accepted, warned about, and ignored.

        Returns
        -------
        SimulationResult
            Density-matrix states at the returned times, optionally
            accompanied by cumulative Liouville-space superoperators.

        Raises
        ------
        ValueError
            If no control is supplied, control durations differ, or the
            initial state is not a ket or density matrix with system
            dimensions, or if `n_samples` is less than 2.
        TypeError
            If an unsupported compatibility keyword is passed.

        Notes
        -----
        The model includes collapse operators only for positive relaxation or
        dephasing rates. QuTiP integrates adaptively through the sorted union
        of all control boundaries and requested output times. Control
        boundaries are internal checkpoints and need not appear in the public
        output trajectory. Stored trajectories remain in the physical rotating
        frame; logical frame shifts are applied only by `SimulationResult`
        analysis helpers.

        Coupling and drive rotating-wave approximations follow
        `create_simulation_parameters` and remain independent of the local
        object model.
        """
        _consume_legacy_dt("mesolve", kwargs)
        controls = _prepare_controls(controls)

        model = self.create_simulation_model(
            controls=controls,
            initial_state=initial_state,
        )

        output_times, solver_times, output_indices = _prepare_trajectory_times(
            model.boundary_times,
            n_samples,
        )
        solver_options = _prepare_solver_options(
            controls,
            options,
            solver_times=solver_times,
        )

        density_matrix = (
            qt.ket2dm(model.initial_state)
            if model.initial_state.isket
            else model.initial_state
        )
        if compute_propagators:
            result = qt.mesolve(
                H=model.hamiltonian,
                rho0=qt.to_super(self.system.identity_matrix),
                tlist=solver_times,
                c_ops=model.collapse_operators,
                options=solver_options,
            )
            all_propagators = list(result.states)
            propagators = [all_propagators[index] for index in output_indices]
            vectorized_state = qt.operator_to_vector(density_matrix)
            states = [
                qt.vector_to_operator(propagator @ vectorized_state)
                for propagator in propagators
            ]
        else:
            result = qt.mesolve(
                H=model.hamiltonian,
                rho0=density_matrix,
                tlist=solver_times,
                c_ops=model.collapse_operators,
                options=solver_options,
            )
            all_states = list(result.states)
            states = [all_states[index] for index in output_indices]
            propagators = []

        return SimulationResult(
            system=self.system,
            controls=controls,
            times=output_times,
            states=states,
            propagators=propagators,
            model=model,
        )

    def sesolve(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        initial_state: qt.Qobj | dict | None = None,
        n_samples: int | None = None,
        compute_propagators: bool = False,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Simulate closed-system dynamics using QuTiP's `sesolve`.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        initial_state : qt.Qobj | dict | None, optional
            Full-system ket or labeled state specification accepted by
            `QuantumSystem.state`. If omitted, use the system ground state.
        n_samples : int | None, optional
            Number of uniformly spaced trajectory points to return from zero
            through the common control duration. If specified, it must be at
            least 2. If omitted, return every control-boundary point. A
            zero-duration trajectory always contains only its initial point.
        compute_propagators : bool, optional
            Whether to compute the unitary propagator trajectory and derive
            kets from it. The default is `False`.
        options : dict[str, Any] | None, optional
            Options passed to QuTiP's `sesolve`. `max_step` is measured in ns.
            Qubex derives `max_step` and `nsteps` when they are omitted; other
            settings use QuTiP defaults. The input dictionary is not mutated.
        **kwargs : Any
            Backward-compatibility keywords. The legacy `dt` keyword is
            accepted, warned about, and ignored.

        Returns
        -------
        SimulationResult
            Ket states at the returned times, optionally accompanied by
            cumulative Hilbert-space propagators.

        Raises
        ------
        ValueError
            If no control is supplied, control durations differ, or the
            initial state is not a ket with system dimensions, or if
            `n_samples` is less than 2.
        TypeError
            If an unsupported compatibility keyword is passed.

        Notes
        -----
        Collapse operators are excluded even when the system specifies
        positive decoherence rates. QuTiP integrates adaptively through the
        sorted union of all control boundaries and requested output times.
        Control boundaries are internal checkpoints and need not appear in the
        public output trajectory. Stored trajectories remain in the physical
        rotating frame; logical frame shifts are applied only by
        `SimulationResult` analysis helpers.

        Coupling and drive rotating-wave approximations follow
        `create_simulation_parameters` and remain independent of the local
        object model.
        """
        _consume_legacy_dt("sesolve", kwargs)
        controls = _prepare_controls(controls)

        model = self.create_simulation_model(
            controls=controls,
            initial_state=initial_state,
            include_collapse_operators=False,
        )

        if not model.initial_state.isket:
            raise ValueError("sesolve requires a ket initial_state.")

        output_times, solver_times, output_indices = _prepare_trajectory_times(
            model.boundary_times,
            n_samples,
        )
        solver_options = _prepare_solver_options(
            controls,
            options,
            solver_times=solver_times,
        )
        if compute_propagators:
            result = qt.sesolve(
                H=model.hamiltonian,
                psi0=self.system.identity_matrix,
                tlist=solver_times,
                options=solver_options,
            )
            all_propagators = list(result.states)
            propagators = [all_propagators[index] for index in output_indices]
            states = [propagator @ model.initial_state for propagator in propagators]
        else:
            result = qt.sesolve(
                H=model.hamiltonian,
                psi0=model.initial_state,
                tlist=solver_times,
                options=solver_options,
            )
            all_states = list(result.states)
            states = [all_states[index] for index in output_indices]
            propagators = []

        return SimulationResult(
            system=self.system,
            controls=controls,
            times=output_times,
            states=states,
            propagators=propagators,
            model=model,
        )

    def propagator(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[qt.Qobj]:
        """
        Compute cumulative propagators at all control boundaries.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        options : dict[str, Any] | None, optional
            Options passed to QuTiP's `propagator`. `max_step` is measured in
            ns. Qubex derives `max_step` and `nsteps` when they are omitted;
            other settings use QuTiP defaults. The input dictionary is not
            mutated.
        **kwargs : Any
            Backward-compatibility keywords. The legacy `dt` keyword is
            accepted, warned about, and ignored.

        Returns
        -------
        list[qt.Qobj]
            Cumulative propagators at every control boundary. Elements are
            unitary operators for a closed system, or superoperators when
            collapse operators are present.

        Raises
        ------
        ValueError
            If no control is supplied or control durations differ.
        TypeError
            If an unsupported compatibility keyword is passed.

        Notes
        -----
        The final element represents the complete evolution in the rotating
        frames defined by the system objects. Logical frame shifts stored by
        the controls are not applied to the returned operators.
        """
        _consume_legacy_dt("propagator", kwargs)
        controls = _prepare_controls(controls)

        params = self.create_simulation_parameters(
            controls=controls,
        )
        solver_options = _prepare_solver_options(
            controls,
            options,
            solver_times=params["boundary_times"],
        )

        return cast(
            list[qt.Qobj],
            qt.propagator(
                H=params["hamiltonian"],
                c_ops=params["collapse_operators"],
                t=params["boundary_times"],
                options=solver_options,
            ),
        )

    def process_fidelity(
        self,
        controls: list[Control] | PulseSchedule,
        target_unitary: TargetUnitary,
        *,
        levels: EvaluationSpace = "computational",
        options: dict[str, Any] | None = None,
    ) -> float:
        """
        Compute the process fidelity of a selected-subspace map.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        target_unitary : TargetUnitary
            Target operator in the full physical space or selected evaluation
            space, or a labeled gate mapping accepted by
            `QuantumSystem.unitary`. The resolved operator must be unitary.
        levels : EvaluationSpace, optional
            Evaluation space. `"computational"` selects levels 0 and 1 of
            each object, `"full"` selects every physical level, and a mapping
            overrides the computational levels for named objects. For a
            labeled target, a mapping also selects its embedding levels. The
            default is `"computational"`.
        options : dict[str, Any] | None, optional
            Options passed to QuTiP's `propagator`. `max_step` is measured in
            ns. Qubex derives `max_step` and `nsteps` when they are omitted;
            other settings use QuTiP defaults.

        Returns
        -------
        float
            Normalized Choi overlap of the selected-subspace map with the
            target, nominally in `[0, 1]`.

        Raises
        ------
        TypeError
            If `target_unitary` is neither a `Qobj` nor a labeled gate mapping.
        ValueError
            If the controls, evaluation levels, propagator type, target
            dimensions, or target unitarity are invalid, or if a full-space
            target does not preserve the selected evaluation space.

        Notes
        -----
        Projecting the physical propagator can produce a trace-decreasing map
        when population leaks from the selected space. This method reports the
        process overlap; use `average_gate_fidelity` when leakage should count
        explicitly as failure through the average survival probability.

        The target is interpreted in the same physical rotating frame as the
        propagator; logical frame-shift metadata is not folded into either
        operator.
        """
        propagators = self.propagator(
            controls=controls,
            options=options,
        )
        projected_propagator = _project_propagator(
            self.system,
            propagators[-1],
            levels,
        )
        projected_target = _prepare_target_unitary(
            self.system,
            target_unitary,
            levels,
        )
        return float(
            qt.process_fidelity(
                projected_propagator,
                projected_target,
            )
        )

    def average_gate_fidelity(
        self,
        controls: list[Control] | PulseSchedule,
        target_unitary: TargetUnitary,
        *,
        levels: EvaluationSpace = "computational",
        options: dict[str, Any] | None = None,
    ) -> float:
        """
        Compute average gate fidelity in a selected evaluation space.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        target_unitary : TargetUnitary
            Target operator in the full physical space or selected evaluation
            space, or a labeled gate mapping accepted by
            `QuantumSystem.unitary`. The resolved operator must be unitary.
        levels : EvaluationSpace, optional
            Evaluation space. `"computational"` selects levels 0 and 1 of
            each object, `"full"` selects every physical level, and a mapping
            overrides the computational levels for named objects. For a
            labeled target, a mapping also selects its embedding levels. The
            default is `"computational"`.
        options : dict[str, Any] | None, optional
            Options passed to QuTiP's `propagator`. `max_step` is measured in
            ns. Qubex derives `max_step` and `nsteps` when they are omitted;
            other settings use QuTiP defaults.

        Returns
        -------
        float
            Average gate fidelity of the selected-subspace map, counting
            leakage from that subspace as failure. The nominal range is
            `[0, 1]`.

        Raises
        ------
        TypeError
            If `target_unitary` is neither a `Qobj` nor a labeled gate mapping.
        ValueError
            If the controls, evaluation levels, propagator type, target
            dimensions, or target unitarity are invalid, or if a full-space
            target does not preserve the selected evaluation space.

        Notes
        -----
        Truncating a full-system propagator can produce a trace-decreasing map.
        The returned Haar average therefore includes the map's average survival
        probability instead of assuming trace preservation. For subspace
        dimension `d`, it returns
        `(d * F_process + p_survival) / (d + 1)`.

        The target is interpreted in the same physical rotating frame as the
        propagator; logical frame-shift metadata is not folded into either
        operator.
        """
        propagators = self.propagator(
            controls=controls,
            options=options,
        )
        projected_propagator = _project_propagator(
            self.system,
            propagators[-1],
            levels,
        )
        projected_target = _prepare_target_unitary(
            self.system,
            target_unitary,
            levels,
        )
        return _average_gate_fidelity_from_subspace_map(
            projected_propagator,
            projected_target,
        )

    @deprecated("Use `average_gate_fidelity` instead.")
    def gate_fidelity(
        self,
        controls: list[Control] | PulseSchedule,
        target_unitary: TargetUnitary,
        *,
        levels: EvaluationSpace = "computational",
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> float:
        """
        Return average gate fidelity through the deprecated method name.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        target_unitary : TargetUnitary
            Target accepted by `average_gate_fidelity`.
        levels : EvaluationSpace, optional
            Evaluation space passed to `average_gate_fidelity`. The default is
            `"computational"`.
        options : dict[str, Any] | None, optional
            QuTiP propagator options passed to `average_gate_fidelity`.
        **kwargs : Any
            Backward-compatibility keywords. The legacy `dt` keyword is
            accepted, warned about, and ignored.

        Returns
        -------
        float
            Average gate fidelity returned by `average_gate_fidelity`.

        Raises
        ------
        TypeError
            If an unsupported compatibility keyword is passed or the target
            type is invalid.
        ValueError
            If the controls, evaluation levels, or resolved target unitary are
            invalid.

        Notes
        -----
        Calling this method emits a `DeprecationWarning`; call
        `average_gate_fidelity` directly instead.
        """
        _consume_legacy_dt("gate_fidelity", kwargs)
        return self.average_gate_fidelity(
            controls=controls,
            target_unitary=target_unitary,
            levels=levels,
            options=options,
        )

    def create_simulation_parameters(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        include_collapse_operators: bool = True,
        **kwargs: Any,
    ) -> dict:
        """
        Build rotating-frame parameters for QuTiP evolution solvers.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        include_collapse_operators : bool, optional
            Whether to include collapse operators for positive decoherence
            rates. The default is `True`.
        **kwargs : Any
            Backward-compatibility keywords. The legacy `dt` keyword is
            accepted, warned about, and ignored.

        Returns
        -------
        dict
            Dictionary with `boundary_times`, the sorted union of control
            boundaries in ns; `hamiltonian`, a `qt.QobjEvo` in rad/ns; and
            `collapse_operators`, a list of full-system `qt.Qobj` operators in
            inverse-square-root ns.

        Raises
        ------
        ValueError
            If no control is supplied or control durations differ.
        TypeError
            If an unsupported compatibility keyword is passed.

        Notes
        -----
        The Hamiltonian is expressed in the rotating frame of each system
        object. Coupling terms use an exchange-only rotating-wave
        approximation: they retain terms that raise one object while lowering
        the other and omit terms that raise or lower both objects
        simultaneously. Control terms use the drive rotating-wave
        approximation: they retain co-rotating raising and lowering components
        and omit counter-rotating drive components. These approximations are
        independent of the selected local object model. Control envelopes use
        exact zero-order hold between segment boundaries, while drive and
        coupling phases remain continuous. Logical frame-shift metadata is not
        part of the physical Hamiltonian.
        """
        _consume_legacy_dt("create_simulation_parameters", kwargs)
        controls = _prepare_controls(controls)
        boundary_times = _time_grid.create_control_boundary_times(controls)

        static_hamiltonian = self.system.zero_matrix
        coupling_hamiltonian: list = []
        control_hamiltonian: list = []
        collapse_operators: list = []

        # Add static terms
        for label in self.system.object_labels:
            static_hamiltonian += self.system.get_rotating_object_hamiltonian(label)

        # Add coupling terms
        for coupling in self.system.couplings:
            ad_0 = self.system.get_raising_operator(coupling.pair[0])
            a_1 = self.system.get_lowering_operator(coupling.pair[1])
            op = ad_0 @ a_1
            g = 2 * np.pi * coupling.strength
            delta = self.system.get_coupling_detuning(coupling.label)
            coefficient = qt.coefficient(
                _ROTATING_COEFFICIENT,
                args={
                    "amplitude": g,
                    "angular_frequency": delta,
                },
            )
            coupling_hamiltonian.append([op, coefficient])
            coupling_hamiltonian.append([op.dag(), coefficient.conj()])

        # Add control terms
        for control in controls:
            target = control.target
            object = self.system.get_object(target)
            a = self.system.get_lowering_operator(target)
            ad = self.system.get_raising_operator(target)
            delta = 2 * np.pi * (control.frequency - object.frequency)
            if control.n_segments == 0:
                envelope = qt.coefficient(0.0)
            else:
                envelope = qt.coefficient(
                    0.5 * control.values,
                    tlist=control.times,
                    order=0,
                )
            phase = qt.coefficient(
                _ROTATING_COEFFICIENT,
                args={
                    "amplitude": 1.0,
                    "angular_frequency": delta,
                },
            )
            coefficient = envelope * phase
            control_hamiltonian.append([ad, coefficient])
            control_hamiltonian.append([a, coefficient.conj()])

        # Total Hamiltonian
        hamiltonian = qt.QobjEvo(
            [
                static_hamiltonian,
                *coupling_hamiltonian,
                *control_hamiltonian,
            ]
        )

        # Add collapse operators
        if include_collapse_operators:
            for object in self.system.objects:
                collapse_operators.extend(
                    self.system.get_collapse_operators(object.label)
                )

        return {
            "boundary_times": boundary_times,
            "hamiltonian": hamiltonian,
            "collapse_operators": collapse_operators,
        }

    def create_simulation_model(
        self,
        controls: list[Control] | PulseSchedule,
        *,
        initial_state: qt.Qobj | dict | None = None,
        include_collapse_operators: bool = True,
        **kwargs: Any,
    ) -> SimulationModel:
        """
        Create a complete model for QuTiP evolution solvers.

        Parameters
        ----------
        controls : list[Control] | PulseSchedule
            Controls with a common duration, or a schedule to convert into
            controls.
        initial_state : qt.Qobj | dict | None, optional
            Full-system ket, density matrix, or labeled state specification
            accepted by `QuantumSystem.state`. If omitted, use the system
            ground state.
        include_collapse_operators : bool, optional
            Whether to include collapse operators for positive decoherence
            rates. The default is `True`.
        **kwargs : Any
            Backward-compatibility keywords. The legacy `dt` keyword is
            accepted, warned about, and ignored.

        Returns
        -------
        SimulationModel
            Hamiltonian, initial state, control-boundary time list, and
            optional collapse operators.

        Raises
        ------
        ValueError
            If no control is supplied, control durations differ, or the
            initial state is not a ket or density matrix with system
            dimensions.
        TypeError
            If an unsupported compatibility keyword is passed.
        """
        _consume_legacy_dt("create_simulation_model", kwargs)
        initial_state = _prepare_initial_state(self.system, initial_state)

        params = self.create_simulation_parameters(
            controls=controls,
            include_collapse_operators=include_collapse_operators,
        )

        return SimulationModel(
            hamiltonian=params["hamiltonian"],
            initial_state=initial_state,
            boundary_times=params["boundary_times"],
            collapse_operators=params["collapse_operators"],
        )


def _prepare_trajectory_times(
    checkpoint_times: npt.NDArray[np.float64],
    n_samples: int | None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """
    Separate public output times from mandatory integration checkpoints.

    Parameters
    ----------
    checkpoint_times : npt.NDArray[np.float64]
        Sorted times through which the evolution must be propagated.
    n_samples : int | None
        Number of uniformly spaced public output points. If omitted, every
        checkpoint is returned.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]
        Public output times, the union used for evolution, and the indices that
        select public outputs from the complete evolution trajectory.
    """
    output_times = (
        checkpoint_times
        if n_samples is None
        else _time_grid.create_uniform_output_times(
            float(checkpoint_times[-1]),
            n_samples,
        )
    )
    evolution_times = _time_grid.create_evolution_times(
        checkpoint_times,
        output_times,
    )
    output_indices = _time_grid.find_time_indices(
        evolution_times,
        output_times,
    )
    return output_times, evolution_times, output_indices


def _prepare_initial_state(
    system: QuantumSystem,
    initial_state: qt.Qobj | dict | None,
) -> qt.Qobj:
    """
    Resolve and validate a full-system initial state.

    Parameters
    ----------
    system : QuantumSystem
        System that defines the required tensor-factor dimensions.
    initial_state : qt.Qobj | dict | None
        Ket, density matrix, or labeled state specification. Non-`Qobj` input
        is passed to `QuantumSystem.state`.

    Returns
    -------
    qt.Qobj
        Ket or density matrix with the system dimensions.

    Raises
    ------
    ValueError
        If the resolved state is neither a ket nor a density matrix, or its
        tensor-factor dimensions do not match the system.
    """
    if not isinstance(initial_state, qt.Qobj):
        initial_state = system.state(initial_state)
    if not initial_state.isket and not initial_state.isoper:
        raise ValueError("The initial state must be a ket or density matrix.")
    object_dimensions = system.object_dimensions
    if initial_state.dims[0] != object_dimensions or (
        initial_state.isoper and initial_state.dims[1] != object_dimensions
    ):
        raise ValueError("The dims of the initial state do not match the system.")
    return initial_state


def _project_propagator(
    system: QuantumSystem,
    propagator: qt.Qobj,
    levels: EvaluationSpace,
) -> qt.Qobj:
    """
    Project an operator or superoperator into an evaluation space.

    Parameters
    ----------
    system : QuantumSystem
        Physical system that defines the projection.
    propagator : qt.Qobj
        Full-system Hilbert-space operator or Liouville-space superoperator.
    levels : EvaluationSpace
        Physical levels retained for each object.

    Returns
    -------
    qt.Qobj
        Propagator restricted to the selected tensor-product subspace.

    Raises
    ------
    ValueError
        If the propagator is neither an operator nor a superoperator, or if
        `levels` is invalid for the system.

    Notes
    -----
    The restricted map can be trace-decreasing when the physical propagator
    transfers population outside the selected space.
    """
    if propagator.issuper:
        return system.project_superoperator(propagator, levels)
    if propagator.isoper:
        return system.project_operator(propagator, levels)
    raise ValueError("The propagator must be an operator or superoperator.")


def _prepare_target_unitary(
    system: QuantumSystem,
    target_unitary: TargetUnitary,
    levels: EvaluationSpace,
) -> qt.Qobj:
    """
    Resolve a target unitary in the selected evaluation space.

    Parameters
    ----------
    system : QuantumSystem
        Physical system used for gate embedding and projection.
    target_unitary : TargetUnitary
        Full-system or selected-space operator, or labeled gate mapping.
    levels : EvaluationSpace
        Physical levels used for target embedding and evaluation.

    Returns
    -------
    qt.Qobj
        Unitary operator with the selected-space tensor dimensions.

    Raises
    ------
    TypeError
        If the target is neither a `Qobj` nor a labeled gate mapping.
    ValueError
        If the target is not unitary, has incompatible dimensions, or a
        full-system target does not preserve the selected space.

    Notes
    -----
    A level mapping is also passed to `QuantumSystem.unitary` when embedding a
    labeled target. The literal selectors `"computational"` and `"full"` do
    not override that method's default gate embedding.
    """
    if not isinstance(target_unitary, qt.Qobj):
        if not isinstance(target_unitary, UnitarySpecification):
            raise TypeError("The target unitary must be a Qobj or gate mapping.")
        embedding_levels = levels if isinstance(levels, Mapping) else None
        target_unitary = system.unitary(
            target_unitary,
            levels=embedding_levels,
        )
    if not target_unitary.isoper or not target_unitary.isunitary:
        raise ValueError("The target unitary must be a unitary operator.")

    physical_dimensions = system.object_dimensions
    if target_unitary.dims == [physical_dimensions, physical_dimensions]:
        projected_target = system.project_operator(target_unitary, levels)
        if not projected_target.isunitary:
            raise ValueError(
                "The target unitary must preserve the selected evaluation space."
            )
        return projected_target

    subspace_dimensions = system.get_subspace_dimensions(levels)
    if target_unitary.dims != [subspace_dimensions, subspace_dimensions]:
        raise ValueError(
            "The target unitary dimensions must match either the full system "
            "or the selected evaluation space."
        )
    return target_unitary


def _average_gate_fidelity_from_subspace_map(
    propagator: qt.Qobj,
    target_unitary: qt.Qobj,
) -> float:
    """
    Average a possibly trace-decreasing map against a unitary target.

    Parameters
    ----------
    propagator : qt.Qobj
        Operator or superoperator restricted to the evaluation space.
    target_unitary : qt.Qobj
        Unitary target on the same evaluation space.

    Returns
    -------
    float
        Haar-average gate fidelity including loss from the evaluation space.

    Notes
    -----
    For evaluation-space dimension `d`, the result is
    `(d * F_process + p_survival) / (d + 1)`. For an operator,
    `p_survival = Tr(K.dag() * K) / d`; for a superoperator it is
    `Tr(E(I)) / d`.
    """
    dimension = target_unitary.shape[0]
    process_fidelity = cast(
        float,
        qt.process_fidelity(propagator, target_unitary),
    )
    if propagator.issuper:
        identity = qt.qeye(target_unitary.dims[0])
        evolved_identity = qt.vector_to_operator(
            propagator @ qt.operator_to_vector(identity)
        )
        average_survival_probability = float(np.real(evolved_identity.tr()) / dimension)
    else:
        propagator_matrix = propagator.full()
        average_survival_probability = float(
            np.vdot(propagator_matrix, propagator_matrix).real / dimension
        )
    return float(
        (dimension * process_fidelity + average_survival_probability) / (dimension + 1)
    )


def _prepare_controls(
    controls: list[Control] | PulseSchedule,
) -> list[Control]:
    """
    Convert schedule input and validate a common control duration.

    Parameters
    ----------
    controls : list[Control] | PulseSchedule
        Existing controls or a schedule to convert.

    Returns
    -------
    list[Control]
        Validated controls. An input list is returned without copying.

    Raises
    ------
    ValueError
        If schedule metadata is missing, no control is supplied, or control
        durations differ.
    """
    if isinstance(controls, PulseSchedule):
        controls = _pulse_schedule_adapter.controls_from_pulse_schedule(controls)
    _validate_controls(controls)
    return controls


def _validate_controls(controls: list[Control]) -> None:
    """
    Validate that controls are nonempty and have one common duration.

    Parameters
    ----------
    controls : list[Control]
        Controls to validate.

    Raises
    ------
    ValueError
        If `controls` is empty or durations differ by more than `1e-12` ns.
    """
    if len(controls) == 0:
        raise ValueError("At least one control signal is required.")
    duration = controls[0].duration
    if not all(
        np.isclose(control.duration, duration, rtol=0, atol=1e-12)
        for control in controls[1:]
    ):
        raise ValueError("The controls must have the same duration.")


def _prepare_solver_options(
    controls: list[Control],
    options: dict[str, Any] | None,
    *,
    solver_times: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """
    Copy QuTiP options and derive safe integration-step defaults.

    Parameters
    ----------
    controls : list[Control]
        Controls whose segment durations determine the default `max_step`.
    options : dict[str, Any] | None
        User-supplied QuTiP options. `max_step`, when present, is in ns. The
        dictionary is copied before defaults are added.
    solver_times : npt.NDArray[np.float64]
        Sorted public-output and mandatory-checkpoint times in ns.

    Returns
    -------
    dict[str, Any]
        Independent options dictionary containing user settings and any
        derived `max_step` or `nsteps` values.

    Notes
    -----
    If omitted, `max_step` is half the shortest nonempty control segment. If
    `nsteps` is also omitted and `max_step` is finite and positive, its default
    is the larger of 2500 and twice the number of steps required to span the
    longest solver interval.
    """
    solver_options = {} if options is None else dict(options)
    if "max_step" not in solver_options:
        nonempty_durations = [
            control.durations for control in controls if control.n_segments > 0
        ]
        if nonempty_durations:
            solver_options["max_step"] = float(
                min(np.min(durations) for durations in nonempty_durations) / 2
            )

    max_step = solver_options.get("max_step")
    if "nsteps" not in solver_options and max_step is not None:
        max_step = float(max_step)
        intervals = np.diff(solver_times)
        if max_step > 0 and np.isfinite(max_step) and intervals.size > 0:
            max_solver_interval = float(np.max(intervals))
            required_steps = math.ceil(max_solver_interval / max_step)
            solver_options["nsteps"] = max(2500, 2 * required_steps)
    return solver_options


def _consume_legacy_dt(
    method_name: str,
    kwargs: dict[str, Any],
) -> None:
    """
    Consume the deprecated `dt` keyword and reject other extra keywords.

    Parameters
    ----------
    method_name : str
        Public method name used in the error message.
    kwargs : dict[str, Any]
        Mutable compatibility-keyword dictionary.

    Raises
    ------
    TypeError
        If `kwargs` contains any key other than `dt`.

    Notes
    -----
    The function removes `dt` from `kwargs`, emits a `DeprecationWarning`, and
    ignores its value. This mutation is internal to the receiving public
    method's collected keyword dictionary.
    """
    if "dt" in kwargs:
        kwargs.pop("dt")
        warnings.warn(
            "The 'dt' keyword is deprecated and ignored by QuTiP-based "
            "simulation methods.",
            DeprecationWarning,
            stacklevel=3,
        )
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(
            f"QuantumSimulator.{method_name}() got an unexpected keyword argument "
            f"'{unexpected}'"
        )
