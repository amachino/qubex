"""Tests for simulation-result analysis helpers."""

from __future__ import annotations

import inspect
from typing import cast

import numpy as np
import pytest
import qutip as qt
from numpy.testing import assert_allclose
from qxsimulator import Control, QuantumSystem, SimulationResult, Transmon
from qxsimulator.simulation import FrameType, SubspaceType


def _single_qutrit_result(state: qt.Qobj) -> SimulationResult:
    """Create a single-qutrit simulation result containing one state."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=3, frequency=5.0)])
    return SimulationResult(
        system=system,
        controls=[],
        times=np.array([0.0]),
        states=[state],
        propagators=[],
    )


def _control(
    frequency: float,
    *,
    target: str = "Q0",
    final_frame_shift: float = 0.0,
) -> Control:
    """Create a one-segment control for frame-resolution tests."""
    return Control(
        target=target,
        waveform=np.array([0.0]),
        durations=np.array([1.0]),
        frequency=frequency,
        final_frame_shift=final_frame_shift,
    )


def _single_qubit_result(controls: list[Control]) -> SimulationResult:
    """Create a single-qubit result for frame-resolution tests."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])
    state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    return SimulationResult(
        system=system,
        controls=controls,
        times=np.array([1.0]),
        states=[state],
        propagators=[],
    )


@pytest.mark.parametrize(
    "method_name",
    ["get_substates", "get_initial_substate", "get_final_substate"],
)
def test_substate_frame_arguments_are_keyword_only(method_name: str) -> None:
    """Substate frame options should be keyword-only."""
    signature = inspect.signature(getattr(SimulationResult, method_name))

    for parameter_name in ("frame", "frame_frequency", "apply_frame_shifts"):
        assert (
            signature.parameters[parameter_name].kind is inspect.Parameter.KEYWORD_ONLY
        )


def test_result_snapshots_constructor_inputs_and_protects_times() -> None:
    """A result should not alias constructor containers or mutable time data."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])
    controls = [_control(5.0)]
    times = np.array([0.0])
    states = [qt.basis(2, 0)]
    propagators: list[qt.Qobj] = []

    result = SimulationResult(
        system=system,
        controls=controls,
        times=times,
        states=states,
        propagators=propagators,
    )
    controls.clear()
    times[0] = 1.0
    states.clear()
    propagators.append(qt.qeye(2))

    assert len(result.controls) == 1
    assert_allclose(result.times, [0.0], rtol=0.0, atol=0.0)
    assert len(result.states) == 1
    assert result.propagators == []
    assert not result.times.flags.writeable


def test_result_uses_identity_equality_and_compact_repr() -> None:
    """Result equality should be identity-based and repr should omit trajectories."""
    result = _single_qubit_result([])
    other = _single_qubit_result([])

    assert result == result
    assert result != other
    assert repr(result) == (
        "SimulationResult(n_controls=0, n_times=1, "
        "has_propagators=False, has_model=False)"
    )


def test_result_rejects_empty_state_trajectory() -> None:
    """A result should require at least one time point and state."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])

    with pytest.raises(ValueError, match="at least one"):
        SimulationResult(
            system=system,
            controls=[],
            times=np.array([]),
            states=[],
            propagators=[],
        )


def test_result_rejects_state_count_mismatch() -> None:
    """A result should contain one state for every time point."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])

    with pytest.raises(ValueError, match=r"states.*times"):
        SimulationResult(
            system=system,
            controls=[],
            times=np.array([0.0, 1.0]),
            states=[qt.basis(2, 0)],
            propagators=[],
        )


def test_result_rejects_propagator_count_mismatch() -> None:
    """A nonempty propagator trajectory should align with all time points."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])

    with pytest.raises(ValueError, match=r"propagators.*times"):
        SimulationResult(
            system=system,
            controls=[],
            times=np.array([0.0, 1.0]),
            states=[qt.basis(2, 0), qt.basis(2, 0)],
            propagators=[qt.qeye(2)],
        )


@pytest.mark.parametrize(
    "times",
    [
        np.array([[0.0, 1.0]]),
        np.array([0.0, np.nan]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
    ],
)
def test_result_rejects_invalid_times(times: np.ndarray) -> None:
    """Result times should be one-dimensional, finite, and strictly increasing."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])
    states = [qt.basis(2, 0) for _ in range(times.size)]

    with pytest.raises(ValueError, match="times"):
        SimulationResult(
            system=system,
            controls=[],
            times=times,
            states=states,
            propagators=[],
        )


def test_result_rejects_state_dimensions_outside_system() -> None:
    """Result states should use the quantum system dimensions."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])

    with pytest.raises(ValueError, match="state dimensions"):
        SimulationResult(
            system=system,
            controls=[],
            times=np.array([0.0]),
            states=[qt.basis(3, 0)],
            propagators=[],
        )


def test_result_rejects_propagator_dimensions_outside_system() -> None:
    """Result propagators should act on the quantum system dimensions."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])

    with pytest.raises(ValueError, match="propagator dimensions"):
        SimulationResult(
            system=system,
            controls=[],
            times=np.array([0.0]),
            states=[qt.basis(2, 0)],
            propagators=[qt.qeye(3)],
        )


def test_drive_frame_requires_a_control_frequency() -> None:
    """Drive-frame inference should reject a target with no controls."""
    result = _single_qubit_result([])

    with pytest.raises(ValueError, match=r"no control frequency.*frame_frequency"):
        result.get_substates("Q0", frame="drive")


def test_drive_frame_rejects_multiple_control_frequencies() -> None:
    """Drive-frame inference should reject multiple tones on one target."""
    result = _single_qubit_result([_control(5.1), _control(5.2)])

    with pytest.raises(
        ValueError,
        match=r"multiple control frequencies.*frame_frequency",
    ):
        result.get_substates("Q0", frame="drive")


def test_drive_frame_accepts_repeated_controls_at_one_frequency() -> None:
    """Drive-frame inference should accept repeated controls at one tone."""
    result = _single_qubit_result([_control(5.1), _control(5.1)])

    inferred = result.get_substates("Q0", frame="drive")
    explicit = result.get_substates("Q0", frame_frequency=5.1)

    assert_allclose(inferred[0].full(), explicit[0].full(), rtol=0.0, atol=1e-12)


def test_explicit_frame_frequency_overrides_multiple_control_frequencies() -> None:
    """An explicit frame frequency should bypass ambiguous drive tones."""
    result = _single_qubit_result([_control(5.1), _control(5.2)])

    actual = result.get_substates("Q0", frame="drive", frame_frequency=5.0)
    expected = result.get_substates("Q0", frame="qubit")

    assert_allclose(actual[0].full(), expected[0].full(), rtol=0.0, atol=1e-12)


def test_substates_rejects_unknown_frame() -> None:
    """Substate extraction should reject frame names outside qubit and drive."""
    result = _single_qubit_result([])

    with pytest.raises(ValueError, match="Unknown frame"):
        result.get_substates("Q0", frame=cast(FrameType, "laboratory"))


def test_control_frequencies_is_deprecated_and_not_cached() -> None:
    """The legacy frequency mapping should warn and reflect current controls."""
    result = _single_qubit_result([_control(5.1)])

    with pytest.warns(DeprecationWarning, match="inspect `controls`"):
        assert result.control_frequencies == {"Q0": 5.1}

    result.controls.append(_control(6.0, target="Q1"))
    with pytest.warns(DeprecationWarning, match="inspect `controls`"):
        assert result.control_frequencies == {"Q0": 5.1, "Q1": 6.0}


def test_final_frame_shifts_sum_controls_by_target() -> None:
    """Final frame shifts should accumulate all controls for each target."""
    result = _single_qubit_result(
        [
            _control(5.0, final_frame_shift=np.pi / 2),
            _control(5.1, final_frame_shift=-np.pi / 4),
        ]
    )

    assert result.final_frame_shifts == pytest.approx({"Q0": np.pi / 4})


def test_substates_apply_frame_shifts_at_each_trajectory_time() -> None:
    """Substates should follow accumulated logical coordinates at every time."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])
    state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    control = Control(
        target="Q0",
        waveform=np.zeros(2, dtype=np.complex128),
        durations=np.ones(2),
        frequency=5.0,
        frame_shifts=np.array([0.0, -np.pi / 2]),
        final_frame_shift=-np.pi / 2,
    )
    result = SimulationResult(
        system=system,
        controls=[control],
        times=np.array([0.0, 1.0, 2.0]),
        states=[state, state, state],
        propagators=[],
    )
    rotation = (0.5j * np.pi * qt.num(2)).expm()

    raw = result.get_substates("Q0", apply_frame_shifts=False)
    logical = result.get_substates("Q0")

    assert_allclose(result.get_frame_shifts("Q0"), [0.0, -np.pi / 2, -np.pi / 2])
    assert_allclose(logical[0].full(), raw[0].full(), rtol=0.0, atol=1e-12)
    for raw_state, logical_state in zip(raw[1:], logical[1:], strict=True):
        assert_allclose(
            logical_state.full(),
            (rotation @ raw_state @ rotation.dag()).full(),
            rtol=0.0,
            atol=1e-12,
        )
    assert_allclose(
        result.get_final_substate("Q0").full(),
        logical[-1].full(),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result.get_density_matrices("Q0")[-1],
        logical[-1].full(),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result.get_density_matrices("Q0", apply_frame_shifts=False)[-1],
        raw[-1].full(),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result.get_bloch_vectors("Q0")[-1],
        [0.0, 1.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result.get_bloch_vectors("Q0", apply_frame_shifts=False)[-1],
        [1.0, 0.0, 0.0],
    )


def test_get_substates_returns_qobj_list() -> None:
    """Substate extraction should return an ordinary list of Qobj instances."""
    result = _single_qutrit_result(qt.basis(3, 0))

    substates = result.get_substates("Q0")

    assert isinstance(substates, list)
    assert all(isinstance(substate, qt.Qobj) for substate in substates)


def test_gf_density_matrix_selects_g_and_f_levels() -> None:
    """The gf density matrix should exclude the intermediate e level."""
    state = (qt.basis(3, 0) + 1j * qt.basis(3, 2)).unit()
    result = _single_qutrit_result(state)

    density_matrices = result.get_density_matrices("Q0", subspace="gf")

    assert density_matrices.shape == (1, 2, 2)
    assert density_matrices.dtype == np.complex128
    assert_allclose(
        density_matrices[0],
        np.array([[0.5, -0.5j], [0.5j, 0.5]]),
        rtol=0.0,
        atol=1e-12,
    )


def test_gf_bloch_vector_uses_g_and_f_as_qubit_basis() -> None:
    """The gf Bloch vector should treat g and f as its two basis states."""
    state = (qt.basis(3, 0) + qt.basis(3, 2)).unit()
    result = _single_qutrit_result(state)

    vectors = result.get_bloch_vectors("Q0", subspace="gf")

    assert vectors.dtype == np.float64
    assert_allclose(vectors, [[1.0, 0.0, 0.0]], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("method_name", ["get_density_matrices", "get_bloch_vectors"])
def test_subspace_rejects_unknown_name(method_name: str) -> None:
    """Subspace analysis should reject names outside ge, ef, and gf."""
    result = _single_qutrit_result(qt.basis(3, 0))
    method = getattr(result, method_name)

    with pytest.raises(ValueError, match="Unknown subspace"):
        method("Q0", subspace=cast(SubspaceType, "unknown"))


@pytest.mark.parametrize("subspace", ["ef", "gf"])
def test_subspace_rejects_insufficient_object_dimension(
    subspace: SubspaceType,
) -> None:
    """A subspace should be rejected when its levels exceed the object dimension."""
    system = QuantumSystem(objects=[Transmon(label="Q0", dimension=2, frequency=5.0)])
    result = SimulationResult(
        system=system,
        controls=[],
        times=np.array([0.0]),
        states=[qt.basis(2, 0)],
        propagators=[],
    )

    with pytest.raises(ValueError, match="requires dimension at least 3"):
        result.get_density_matrices("Q0", subspace=subspace)
