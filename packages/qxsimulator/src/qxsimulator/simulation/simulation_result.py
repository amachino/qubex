"""Results returned by quantum simulations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import qctrlvisualizer as qv
import qutip as qt
from qxvisualizer import plot_bloch_vectors, show_figure
from typing_extensions import deprecated

from qxsimulator.system import QuantumSystem

from . import _sampling
from .control import Control
from .simulation_model import SimulationModel

FrameType: TypeAlias = Literal["qubit", "drive"]
SubspaceType: TypeAlias = Literal["ge", "ef", "gf"]

_VALID_FRAMES: tuple[FrameType, ...] = ("qubit", "drive")
_SUBSPACE_LEVELS: dict[SubspaceType, tuple[int, int]] = {
    "ge": (0, 1),
    "ef": (1, 2),
    "gf": (0, 2),
}


@dataclass(eq=False, repr=False)
class SimulationResult:
    """
    Store a validated state trajectory and optional propagators.

    Attributes
    ----------
    system : QuantumSystem
        Physical system that defines tensor-factor labels and dimensions.
    controls : list[Control]
        Controls used for the evolution and its logical-frame metadata.
    times : npt.NDArray[np.float64]
        One-dimensional, strictly increasing trajectory times in ns. The
        stored array is copied as `float64` and made read-only.
    states : list[qt.Qobj]
        Full-system kets or density matrices at each time point, expressed in
        the simulator's physical rotating frame.
    propagators : list[qt.Qobj]
        Full-system operators or superoperators at each time point. An empty
        list indicates that propagators were not computed.
    model : SimulationModel | None
        Model used by a QuTiP solver, or `None` for results produced without a
        `SimulationModel`.

    Raises
    ------
    ValueError
        If times are not a nonempty, finite, strictly increasing 1-D array;
        state or propagator counts do not align with times; or any state or
        propagator has dimensions incompatible with the system.

    Notes
    -----
    Constructor containers are copied, but their `Control` and `Qobj` elements
    are retained by reference.

    Stored states and propagators remain in the simulator's physical rotating
    frame. Labeled substate, density-matrix, and Bloch-vector helpers apply
    accumulated logical frame shifts by default.
    """

    system: QuantumSystem
    controls: list[Control]
    times: npt.NDArray[np.float64]
    states: list[qt.Qobj]
    propagators: list[qt.Qobj]
    model: SimulationModel | None = None

    def __post_init__(self) -> None:
        """Snapshot and validate trajectory data."""
        self.controls = list(self.controls)
        self.times = np.array(self.times, dtype=np.float64, copy=True)
        self.states = list(self.states)
        self.propagators = list(self.propagators)

        if self.times.ndim != 1:
            raise ValueError("The times must be one-dimensional.")
        if len(self.times) == 0 or len(self.states) == 0:
            raise ValueError("Times and states must contain at least one sample.")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("The times must contain only finite values.")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("The times must be strictly increasing.")
        if len(self.states) != len(self.times):
            raise ValueError("The number of states must match the number of times.")
        if self.propagators and len(self.propagators) != len(self.times):
            raise ValueError(
                "The number of propagators must match the number of times."
            )

        expected_dimensions = self.system.object_dimensions
        if any(
            state.dims[0] != expected_dimensions
            or not (state.isket or state.isoper)
            or (state.isoper and state.dims[1] != expected_dimensions)
            for state in self.states
        ):
            raise ValueError("The state dimensions must match the system.")
        if any(
            not _propagator_matches_dimensions(propagator, expected_dimensions)
            for propagator in self.propagators
        ):
            raise ValueError("The propagator dimensions must match the system.")

        self.times.flags.writeable = False

    def __repr__(self) -> str:
        """Return a compact trajectory summary."""
        return (
            f"{type(self).__name__}(n_controls={len(self.controls)}, "
            f"n_times={len(self.times)}, "
            f"has_propagators={bool(self.propagators)}, "
            f"has_model={self.model is not None})"
        )

    @property
    @deprecated("`control_frequencies` is deprecated; inspect `controls` instead.")
    def control_frequencies(self) -> dict[str, float]:
        """Return each target's last listed control frequency in GHz."""
        return {control.target: control.frequency for control in self.controls}

    @property
    def final_frame_shifts(self) -> dict[str, float]:
        """Return accumulated final control-frame shifts by target in radians."""
        shifts: defaultdict[str, float] = defaultdict(float)
        for control in self.controls:
            shifts[control.target] += float(control.final_frame_shift)
        return dict(shifts)

    def get_frame_shifts(self, label: str) -> npt.NDArray[np.float64]:
        """
        Return accumulated logical frame shifts for a target over time.

        Parameters
        ----------
        label : str
            Label of the system object.

        Returns
        -------
        npt.NDArray[np.float64]
            Accumulated shifts in radians with shape `(n_times,)`.

        Raises
        ------
        ValueError
            If `label` does not identify a system object.

        Notes
        -----
        Frame shifts from multiple controls targeting the same object are
        added. An internal segment boundary uses the shift of the segment
        starting at that boundary, and a control's final shift persists from
        its final boundary onward.
        """
        self.system.get_object(label)
        frame_shifts = np.zeros_like(self.times)
        for control in self.controls:
            if control.target == label:
                frame_shifts += control.get_frame_shifts(self.times)
        return frame_shifts

    def _resolve_drive_frame_frequency(self, label: str) -> float:
        """
        Infer one cyclic drive-frame frequency for a target.

        Parameters
        ----------
        label : str
            Target label whose controls are inspected.

        Returns
        -------
        float
            The unique control frequency in GHz.

        Raises
        ------
        ValueError
            If no control frequency or more than one distinct frequency is
            available for `label`.
        """
        frequencies = {
            control.frequency for control in self.controls if control.target == label
        }
        if not frequencies:
            raise ValueError(
                f"Cannot infer the drive frame for {label!r}: no control frequency "
                "is available; pass `frame_frequency` explicitly."
            )
        if len(frequencies) > 1:
            frequencies_text = ", ".join(
                f"{frequency:g}" for frequency in sorted(frequencies)
            )
            raise ValueError(
                f"Cannot infer the drive frame for {label!r}: multiple control "
                f"frequencies are available ({frequencies_text} GHz); pass "
                "`frame_frequency` explicitly."
            )
        return next(iter(frequencies))

    @property
    def initial_state(self) -> qt.Qobj:
        """Return the initial state in the physical rotating frame."""
        return self.states[0]

    @property
    def final_state(self) -> qt.Qobj:
        """Return the final state in the physical rotating frame."""
        return self.states[-1]

    @property
    @deprecated("Use `propagators` instead.")
    def unitaries(self) -> list[qt.Qobj]:
        """Return `propagators` through the deprecated attribute name."""
        return self.propagators

    def get_substates(
        self,
        label: str,
        *,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        apply_frame_shifts: bool = True,
    ) -> list[qt.Qobj]:
        """
        Extract one system object's reduced-state trajectory.

        Parameters
        ----------
        label : str
            Label of the system object to retain.
        frame : FrameType | None, optional
            Analysis frame. `"qubit"` uses the object's rotating-frame
            frequency, while `"drive"` uses the one distinct control frequency
            available for `label`. If omitted, use `"qubit"`.
        frame_frequency : float | None, optional
            Explicit cyclic analysis-frame frequency in GHz. If specified, it
            takes precedence over `frame`.
        apply_frame_shifts : bool, optional
            Whether to interpret each substate in the accumulated logical
            frame at its trajectory time. The default is `True`.

        Returns
        -------
        list[qt.Qobj]
            Reduced density matrices in trajectory order, one for every
            element of `times`.

        Raises
        ------
        ValueError
            If `label` is unknown, `frame` is invalid, or `frame="drive"` is
            requested without exactly one distinct control frequency for
            `label` and `frame_frequency` is omitted.

        Notes
        -----
        Stored physical-frame states are not modified. Frequency-frame changes
        and accumulated logical frame shifts are coordinate transformations on
        the returned reduced states. Shifts from multiple controls on `label`
        are added at each trajectory time.
        """
        if frame is None:
            frame = "qubit"
        if frame not in _VALID_FRAMES:
            valid_frames = ", ".join(_VALID_FRAMES)
            raise ValueError(
                f"Unknown frame {frame!r}; expected one of: {valid_frames}."
            )

        index = self.system.get_index(label)
        substates = [state.ptrace(index) for state in self.states]

        target_frequency = None
        if frame_frequency is not None:
            target_frequency = frame_frequency
        elif frame == "drive":
            target_frequency = self._resolve_drive_frame_frequency(label)

        if target_frequency is not None:
            times = self.get_times()
            qubit = self.system.get_object(label)
            f_qubit = qubit.frequency
            delta = 2 * np.pi * (target_frequency - f_qubit)
            dim = qubit.dimension
            N = qt.num(dim)
            U = lambda t: (-1j * delta * N * t).expm()
            substates = [
                U(t).dag() * rho * U(t)
                for t, rho in zip(
                    times,
                    substates,
                    strict=True,
                )
            ]

        if apply_frame_shifts:
            frame_shifts = self.get_frame_shifts(label)
            dimension = self.system.get_object(label).dimension
            rotations: dict[float, qt.Qobj] = {}
            transformed_substates = []
            for frame_shift, substate in zip(
                frame_shifts,
                substates,
                strict=True,
            ):
                shift = float(frame_shift)
                if shift == 0.0:
                    transformed_substates.append(substate)
                    continue
                rotation = rotations.get(shift)
                if rotation is None:
                    rotation = qt.Qobj(
                        np.diag(np.exp(-1j * shift * np.arange(dimension)))
                    )
                    rotations[shift] = rotation
                transformed_substates.append(rotation @ substate @ rotation.dag())
            substates = transformed_substates

        return substates

    def get_initial_substate(
        self,
        label: str,
        *,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        apply_frame_shifts: bool = True,
    ) -> qt.Qobj:
        """
        Extract one system object's initial reduced state.

        Parameters
        ----------
        label : str
            Label of the system object to retain.
        frame : FrameType | None, optional
            Analysis frame: the object's rotating frame (`"qubit"`) or its
            uniquely inferable control frame (`"drive"`). If omitted, use
            `"qubit"`.
        frame_frequency : float | None, optional
            Explicit cyclic analysis-frame frequency in GHz. If specified, it
            takes precedence over `frame`.
        apply_frame_shifts : bool, optional
            Whether to interpret the substate in the accumulated logical
            frame. The default is `True`.

        Returns
        -------
        qt.Qobj
            Initial reduced density matrix in the requested coordinates.

        Raises
        ------
        ValueError
            If `label` is unknown, `frame` is invalid, or the drive frame
            cannot be inferred.
        """
        return self.get_substates(
            label,
            frame=frame,
            frame_frequency=frame_frequency,
            apply_frame_shifts=apply_frame_shifts,
        )[0]

    def get_final_substate(
        self,
        label: str,
        *,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        apply_frame_shifts: bool = True,
    ) -> qt.Qobj:
        """
        Extract one system object's final reduced state.

        Parameters
        ----------
        label : str
            Label of the system object to retain.
        frame : FrameType | None, optional
            Analysis frame: the object's rotating frame (`"qubit"`) or its
            uniquely inferable control frame (`"drive"`). If omitted, use
            `"qubit"`.
        frame_frequency : float | None, optional
            Explicit cyclic analysis-frame frequency in GHz. If specified, it
            takes precedence over `frame`.
        apply_frame_shifts : bool, optional
            Whether to interpret the substate in the accumulated logical
            frame. The default is `True`.

        Returns
        -------
        qt.Qobj
            Final reduced density matrix in the requested coordinates.

        Raises
        ------
        ValueError
            If `label` is unknown, `frame` is invalid, or the drive frame
            cannot be inferred.
        """
        return self.get_substates(
            label,
            frame=frame,
            frame_frequency=frame_frequency,
            apply_frame_shifts=apply_frame_shifts,
        )[-1]

    def get_times(
        self,
        *,
        n_samples: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Return simulation times with optional index-based downsampling.

        Parameters
        ----------
        n_samples : int | None, optional
            Non-negative maximum number of times to return. If omitted, return
            every trajectory time.

        Returns
        -------
        npt.NDArray[np.float64]
            One-dimensional `float64` times in ns.

        Raises
        ------
        ValueError
            If `n_samples` is negative.

        Notes
        -----
        When downsampling to at least two points, the initial and final times
        are retained. If no downsampling is needed, the stored read-only array
        is returned directly.
        """
        times = _sampling.downsample(self.times, n_samples)
        return times

    def get_bloch_vectors(
        self,
        label: str,
        *,
        n_samples: int | None = None,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        subspace: SubspaceType = "ge",
        apply_frame_shifts: bool = True,
    ) -> npt.NDArray[np.float64]:
        """
        Compute Bloch coordinates in a selected two-level subspace.

        Parameters
        ----------
        label : str
            Label of the system object to analyze.
        n_samples : int | None, optional
            Non-negative maximum number of trajectory points to return. If
            omitted, return every point.
        frame : FrameType | None, optional
            Analysis frame: the object's rotating frame (`"qubit"`) or its
            uniquely inferable control frame (`"drive"`). If omitted, use
            `"qubit"`.
        frame_frequency : float | None, optional
            Explicit cyclic analysis-frame frequency in GHz. If specified, it
            takes precedence over `frame`.
        subspace : SubspaceType, optional
            Ordered physical levels used as the Bloch basis: `"ge"` selects
            `(0, 1)`, `"ef"` selects `(1, 2)`, and `"gf"` selects `(0, 2)`.
            The default is `"ge"`.
        apply_frame_shifts : bool, optional
            Whether to interpret substates in their accumulated logical
            frames. The default is `True`.

        Returns
        -------
        npt.NDArray[np.float64]
            Pauli-X, Pauli-Y, and Pauli-Z expectation values with shape
            `(n_times, 3)` and dtype `float64`.

        Raises
        ------
        ValueError
            If the label, frame, subspace, or requested frame inference is
            invalid; the object is too small for the subspace; or `n_samples`
            is negative.

        Notes
        -----
        The selected two-level block is not renormalized. Population outside
        the subspace is excluded from the Pauli expectations and can shorten
        the returned vector.
        """
        X = qt.sigmax()
        Y = qt.sigmay()
        Z = qt.sigmaz()
        levels = _get_subspace_levels(
            subspace,
            dimension=self.system.get_object(label).dimension,
        )
        substates = self.get_substates(
            label,
            frame=frame,
            frame_frequency=frame_frequency,
            apply_frame_shifts=apply_frame_shifts,
        )
        buffer = []
        indices = np.ix_(levels, levels)
        for substate in substates:
            rho = qt.Qobj(substate.full()[indices])
            x = qt.expect(X, rho)
            y = qt.expect(Y, rho)
            z = qt.expect(Z, rho)
            buffer.append([x, y, z])
        vectors = np.asarray(buffer, dtype=np.complex128).real
        vectors = _sampling.downsample(vectors, n_samples)
        return vectors

    def get_density_matrices(
        self,
        label: str,
        *,
        n_samples: int | None = None,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        subspace: SubspaceType = "ge",
        apply_frame_shifts: bool = True,
    ) -> npt.NDArray[np.complex128]:
        """
        Extract density-matrix blocks for a selected two-level subspace.

        Parameters
        ----------
        label : str
            Label of the system object to analyze.
        n_samples : int | None, optional
            Non-negative maximum number of trajectory points to return. If
            omitted, return every point.
        frame : FrameType | None, optional
            Analysis frame: the object's rotating frame (`"qubit"`) or its
            uniquely inferable control frame (`"drive"`). If omitted, use
            `"qubit"`.
        frame_frequency : float | None, optional
            Explicit cyclic analysis-frame frequency in GHz. If specified, it
            takes precedence over `frame`.
        subspace : SubspaceType, optional
            Ordered physical levels to extract: `"ge"` selects `(0, 1)`,
            `"ef"` selects `(1, 2)`, and `"gf"` selects `(0, 2)`. The default
            is `"ge"`.
        apply_frame_shifts : bool, optional
            Whether to interpret substates in their accumulated logical
            frames. The default is `True`.

        Returns
        -------
        npt.NDArray[np.complex128]
            Selected density-matrix blocks with shape `(n_times, 2, 2)` and
            dtype `complex128`.

        Raises
        ------
        ValueError
            If the label, frame, subspace, or requested frame inference is
            invalid; the object is too small for the subspace; or `n_samples`
            is negative.

        Notes
        -----
        Blocks are not renormalized after projection. Their trace is the
        population retained in the selected two-level subspace.
        """
        levels = _get_subspace_levels(
            subspace,
            dimension=self.system.get_object(label).dimension,
        )
        substates = self.get_substates(
            label,
            frame=frame,
            frame_frequency=frame_frequency,
            apply_frame_shifts=apply_frame_shifts,
        )
        indices = np.ix_(levels, levels)
        rho = np.array(
            [substate.full()[indices] for substate in substates],
            dtype=np.complex128,
        )
        rho = _sampling.downsample(rho, n_samples)
        return rho

    def plot_bloch_vectors(
        self,
        label: str,
        *,
        n_samples: int | None = None,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        subspace: SubspaceType = "ge",
        apply_frame_shifts: bool = True,
    ) -> None:
        """
        Plot Bloch-coordinate trajectories for a two-level subspace.

        Parameters
        ----------
        label : str
            Label of the system object to analyze.
        n_samples : int | None, optional
            Non-negative maximum number of trajectory points to display. If
            omitted, display every point.
        frame : FrameType | None, optional
            Analysis frame: the object's rotating frame (`"qubit"`) or its
            uniquely inferable control frame (`"drive"`). If omitted, use
            `"qubit"`.
        frame_frequency : float | None, optional
            Explicit cyclic analysis-frame frequency in GHz. If specified, it
            takes precedence over `frame`.
        subspace : SubspaceType, optional
            Two-level basis selector: `"ge"`, `"ef"`, or `"gf"`. The default
            is `"ge"`.
        apply_frame_shifts : bool, optional
            Whether to interpret substates in their accumulated logical
            frames. The default is `True`.

        Raises
        ------
        ValueError
            If any analysis selector is invalid or `n_samples` is negative.

        Notes
        -----
        This method displays an interactive figure through `qxvisualizer` and
        returns no figure object. Bloch coordinates are not renormalized after
        projection into `subspace`.
        """
        vectors = self.get_bloch_vectors(
            label,
            n_samples=n_samples,
            frame=frame,
            frame_frequency=frame_frequency,
            subspace=subspace,
            apply_frame_shifts=apply_frame_shifts,
        )
        times = self.get_times(
            n_samples=n_samples,
        )
        plot_bloch_vectors(
            times=times,
            bloch_vectors=vectors,
            mode="lines",
            title=f"State evolution : {label}",
        )

    def display_bloch_sphere(
        self,
        label: str,
        *,
        n_samples: int | None = None,
        frame: FrameType | None = None,
        frame_frequency: float | None = None,
        subspace: SubspaceType = "ge",
        apply_frame_shifts: bool = True,
    ) -> None:
        """
        Display selected-subspace states on an interactive Bloch sphere.

        Parameters
        ----------
        label : str
            Label of the system object to analyze.
        n_samples : int | None, optional
            Non-negative maximum number of trajectory points to display. If
            omitted, display every point.
        frame : FrameType | None, optional
            Analysis frame: the object's rotating frame (`"qubit"`) or its
            uniquely inferable control frame (`"drive"`). If omitted, use
            `"qubit"`.
        frame_frequency : float | None, optional
            Explicit cyclic analysis-frame frequency in GHz. If specified, it
            takes precedence over `frame`.
        subspace : SubspaceType, optional
            Two-level basis selector: `"ge"`, `"ef"`, or `"gf"`. The default
            is `"ge"`.
        apply_frame_shifts : bool, optional
            Whether to interpret substates in their accumulated logical
            frames. The default is `True`.

        Raises
        ------
        ValueError
            If any analysis selector is invalid or `n_samples` is negative.

        Notes
        -----
        This method displays a qctrl-visualizer widget and returns no widget
        object. Density-matrix blocks are not renormalized after projection
        into `subspace`.
        """
        rho = self.get_density_matrices(
            label,
            n_samples=n_samples,
            frame=frame,
            frame_frequency=frame_frequency,
            subspace=subspace,
            apply_frame_shifts=apply_frame_shifts,
        )
        qv.display_bloch_sphere_from_density_matrices(rho)

    def _get_general_substates(
        self,
        labels: Sequence[str],
        *,
        frame_frequencies: dict[str, float] | None = None,
    ) -> npt.NDArray:
        """
        Extract a reduced trajectory while preserving requested object order.

        Parameters
        ----------
        labels : Sequence[str]
            System-object labels in the desired tensor-factor order.
        frame_frequencies : dict[str, float] | None, optional
            Explicit cyclic analysis-frame frequencies in GHz by label. Labels
            absent from the mapping remain in their object rotating frames.

        Returns
        -------
        npt.NDArray
            Array containing one reduced `Qobj` state per trajectory time.

        Notes
        -----
        QuTiP sorts partial-trace indices. This helper permutes each reduced
        state back to `labels` order before applying any frame transformation.
        Logical frame-shift metadata is not applied by this private helper.
        """
        # 1. Extract substates (ptrace)
        # Note: qutip.ptrace always sorts the indices, so the order of subsystems
        # in the result might differ from 'labels'.
        target_indices = [self.system.get_index(label) for label in labels]
        substates = np.array([state.ptrace(target_indices) for state in self.states])

        # 2. Restore the order of subsystems if necessary
        sorted_indices = sorted(target_indices)
        if target_indices != sorted_indices:
            # Calculate permutation to match the requested 'labels' order
            perm_order = [sorted_indices.index(i) for i in target_indices]
            substates = np.array([rho.permute(perm_order) for rho in substates])

        # 3. Apply frame transformation if requested
        if frame_frequencies is not None:
            substates = self._apply_frame_transformation(
                substates, labels, frame_frequencies
            )

        return substates

    def _apply_frame_transformation(
        self,
        substates: npt.NDArray,
        labels: Sequence[str],
        frame_frequencies: dict[str, float],
    ) -> npt.NDArray:
        """
        Transform a reduced trajectory to explicit rotating frequencies.

        Parameters
        ----------
        substates : npt.NDArray
            Reduced `Qobj` states aligned with `self.times`.
        labels : Sequence[str]
            Tensor-factor labels in the order used by each reduced state.
        frame_frequencies : dict[str, float]
            Target cyclic frame frequencies in GHz by label. Unlisted labels
            are left in their object rotating frames.

        Returns
        -------
        npt.NDArray
            States after the time-dependent coordinate transformation.

        Notes
        -----
        For label `i`, the signed angular detuning is
        `2 * pi * (frame_frequency[i] - object_frequency[i])` in rad/ns. At
        time `t`, the combined transformation applies
        `U(t) = exp(1j * H_frame * t)` as `U(t) * rho * U(t).dag()`.
        """
        times = self.get_times()
        dims = [self.system.get_object(label).dimension for label in labels]

        # Construct the effective Hamiltonian for the frame change
        # H_frame = sum( delta_i * n_i )
        H_frame = qt.qzero(dims)
        for i, label in enumerate(labels):
            if label not in frame_frequencies:
                continue

            target_freq = frame_frequencies[label]
            qubit_freq = self.system.get_object(label).frequency
            delta = 2 * np.pi * (target_freq - qubit_freq)

            if delta == 0:
                continue

            # Operator for the i-th subsystem: I x ... x n_i x ... x I
            ops = [qt.qeye(d) for d in dims]
            ops[i] = qt.num(dims[i])
            H_frame += delta * qt.tensor(*ops)

        if H_frame.norm() == 0:
            return substates

        # Apply unitary transformation: rho' = U rho U^dagger
        # U(t) = exp(i * H_frame * t)
        transformed_substates = []
        for t, rho in zip(times, substates, strict=True):
            U = (1j * H_frame * t).expm()
            transformed_substates.append(rho.transform(U))

        return np.array(transformed_substates)

    def _get_general_bloch_vectors(
        self,
        labels: Sequence[str],
        *,
        basis_set: tuple[Sequence[int], Sequence[int]],
        frame_frequencies: dict[str, float] | None = None,
        n_samples: int | None = None,
    ) -> npt.NDArray:
        """
        Compute Bloch coordinates for two composite basis states.

        Parameters
        ----------
        labels : Sequence[str]
            System-object labels defining the reduced tensor product.
        basis_set : tuple[Sequence[int], Sequence[int]]
            Two composite basis states. Each sequence supplies one physical
            level index per label.
        frame_frequencies : dict[str, float] | None, optional
            Explicit cyclic analysis-frame frequencies in GHz by label.
        n_samples : int | None, optional
            Non-negative maximum number of trajectory points to return.

        Returns
        -------
        npt.NDArray
            Real Bloch coordinates with shape `(n_times, 3)`.

        Notes
        -----
        The two-state subspace is not renormalized, so population in all other
        reduced-system basis states is excluded from the coordinates.
        """
        dimensions = [self.system.get_object(label).dimension for label in labels]
        ket0 = qt.tensor(
            *[
                qt.basis(dim, basis)
                for dim, basis in zip(dimensions, basis_set[0], strict=True)
            ]
        )
        ket1 = qt.tensor(
            *[
                qt.basis(dim, basis)
                for dim, basis in zip(dimensions, basis_set[1], strict=True)
            ]
        )
        bra0 = ket0.dag()
        bra1 = ket1.dag()

        X: qt.Qobj = ket0 @ bra1 + ket1 @ bra0
        Y: qt.Qobj = -1j * ket0 @ bra1 + 1j * ket1 @ bra0
        Z: qt.Qobj = ket0 @ bra0 - ket1 @ bra1

        buffer = []
        states = self._get_general_substates(
            labels=labels,
            frame_frequencies=frame_frequencies,
        )
        for rho in states:
            x = qt.expect(X, rho)
            y = qt.expect(Y, rho)
            z = qt.expect(Z, rho)
            buffer.append([x, y, z])

        vectors = np.real(buffer)
        vectors = _sampling.downsample(vectors, n_samples)
        return vectors

    def _plot_general_bloch_vectors(
        self,
        labels: Sequence[str],
        *,
        basis_set: tuple[Sequence[int], Sequence[int]],
        frame_frequencies: dict[str, float] | None = None,
        n_samples: int | None = None,
    ) -> None:
        """
        Plot Bloch coordinates for two composite basis states.

        Parameters
        ----------
        labels : Sequence[str]
            System-object labels defining the reduced tensor product.
        basis_set : tuple[Sequence[int], Sequence[int]]
            Two composite basis states, with one physical level per label.
        frame_frequencies : dict[str, float] | None, optional
            Explicit cyclic analysis-frame frequencies in GHz by label.
        n_samples : int | None, optional
            Non-negative maximum number of displayed trajectory points.

        Notes
        -----
        This method displays an interactive figure through `qxvisualizer`.
        """
        vectors = self._get_general_bloch_vectors(
            labels,
            basis_set=basis_set,
            frame_frequencies=frame_frequencies,
            n_samples=n_samples,
        )
        times = self.get_times(
            n_samples=n_samples,
        )
        plot_bloch_vectors(
            times=times,
            bloch_vectors=vectors,
            mode="lines",
            title=f"State evolution : {', '.join(labels)}",
        )

    def _display_general_bloch_sphere(
        self,
        labels: Sequence[str],
        *,
        basis_set: tuple[Sequence[int], Sequence[int]],
        frame_frequencies: dict[str, float] | None = None,
        n_samples: int | None = None,
    ) -> None:
        """
        Display a Bloch sphere for two composite basis states.

        Parameters
        ----------
        labels : Sequence[str]
            System-object labels defining the reduced tensor product.
        basis_set : tuple[Sequence[int], Sequence[int]]
            Two composite basis states, with one physical level per label.
        frame_frequencies : dict[str, float] | None, optional
            Explicit cyclic analysis-frame frequencies in GHz by label.
        n_samples : int | None, optional
            Non-negative maximum number of displayed trajectory points.

        Notes
        -----
        This method displays a qctrl-visualizer widget.
        """
        vectors = self._get_general_bloch_vectors(
            labels,
            basis_set=basis_set,
            frame_frequencies=frame_frequencies,
            n_samples=n_samples,
        )
        qv.display_bloch_sphere_from_bloch_vectors(vectors)

    def show_last_population(
        self,
        label: str | None = None,
    ) -> None:
        """
        Print basis-state populations for the final trajectory state.

        Parameters
        ----------
        label : str | None, optional
            System-object label whose reduced-state populations are printed.
            If omitted, print populations in the full-system tensor basis.

        Raises
        ------
        ValueError
            If `label` does not identify a system object.

        Notes
        -----
        One line per basis state is written to standard output as a percentage.
        """
        states = self.states if label is None else self.get_substates(label)
        population = _get_population(states[-1])
        for idx, prob in enumerate(population):
            basis = self.system.basis_labels[idx] if label is None else str(idx)
            print(f"|{basis}⟩: {prob * 100:6.3f}%")

    def plot_population_dynamics(
        self,
        label: str | None = None,
        *,
        n_samples: int | None = None,
    ) -> None:
        """
        Plot basis-state populations over the trajectory.

        Parameters
        ----------
        label : str | None, optional
            System-object label whose reduced-state populations are plotted.
            If omitted, plot the full-system tensor-basis populations.
        n_samples : int | None, optional
            Non-negative maximum number of trajectory points to display. If
            omitted, display every point.

        Raises
        ------
        ValueError
            If `label` is unknown or `n_samples` is negative.

        Notes
        -----
        This method displays an interactive Plotly figure through
        `qxvisualizer` and returns no figure object.
        """
        states = self.states if label is None else self.get_substates(label)
        populations = defaultdict(list)
        for state in states:
            population = _get_population(state)
            population[population > 1] = 1.0
            for idx, prob in enumerate(population):
                basis = self.system.basis_labels[idx] if label is None else str(idx)
                populations[f"|{basis}〉"].append(prob)

        sampled_times = self.get_times(n_samples=n_samples)
        sampled_populations = {
            key: _sampling.downsample(np.asarray(value), n_samples)
            for key, value in populations.items()
        }

        fig = go.Figure()
        for key, value in sampled_populations.items():
            fig.add_trace(
                go.Scatter(
                    x=sampled_times,
                    y=value,
                    mode="lines",
                    name=key,
                )
            )
        fig.update_layout(
            title="Population dynamics"
            if label is None
            else f"Population dynamics : {label}",
            xaxis_title="Time (ns)",
            yaxis_title="Population",
            template="qubex",
        )
        show_figure(fig, filename="population_dynamics")


def _get_subspace_levels(
    subspace: SubspaceType,
    *,
    dimension: int,
) -> tuple[int, int]:
    """
    Resolve a named two-level subspace within an object dimension.

    Parameters
    ----------
    subspace : SubspaceType
        `"ge"`, `"ef"`, or `"gf"`.
    dimension : int
        Number of retained physical levels in the object.

    Returns
    -------
    tuple[int, int]
        Ordered physical-level indices for the requested subspace.

    Raises
    ------
    ValueError
        If the subspace name is unknown or requires a level outside
        `dimension`.
    """
    try:
        levels = _SUBSPACE_LEVELS[subspace]
    except KeyError:
        valid_subspaces = ", ".join(_SUBSPACE_LEVELS)
        raise ValueError(
            f"Unknown subspace {subspace!r}; expected one of: {valid_subspaces}."
        ) from None

    required_dimension = max(levels) + 1
    if dimension < required_dimension:
        raise ValueError(
            f"Subspace {subspace!r} requires dimension at least "
            f"{required_dimension}, got {dimension}."
        )
    return levels


def _propagator_matches_dimensions(
    propagator: qt.Qobj,
    dimensions: list[int],
) -> bool:
    """
    Check whether a propagator acts on the expected tensor dimensions.

    Parameters
    ----------
    propagator : qt.Qobj
        Candidate Hilbert-space operator or Liouville-space superoperator.
    dimensions : list[int]
        Expected local Hilbert-space dimensions in tensor-factor order.

    Returns
    -------
    bool
        `True` when the operator or superoperator dimensions match exactly.
    """
    if propagator.isoper:
        return propagator.dims == [dimensions, dimensions]
    if propagator.issuper:
        operator_dimensions = [dimensions, dimensions]
        return propagator.dims == [operator_dimensions, operator_dimensions]
    return False


def _get_population(state: qt.Qobj) -> npt.NDArray[np.float64]:
    """
    Return basis-state populations for a ket or density matrix.

    Parameters
    ----------
    state : qt.Qobj
        Ket or density matrix.

    Returns
    -------
    npt.NDArray[np.float64]
        Real populations with shape `(dimension,)` in QuTiP basis order.
    """
    if state.isket:
        return np.asarray(np.abs(state.full().ravel()) ** 2, dtype=np.float64)
    return np.asarray(np.real(state.diag()), dtype=np.float64)
