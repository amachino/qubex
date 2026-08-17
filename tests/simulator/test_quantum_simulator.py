"""Tests for quantum simulator solver entrypoints."""

from __future__ import annotations

from inspect import signature
from unittest.mock import patch

import numpy as np
import pytest
import qutip as qt
from numpy.testing import assert_allclose
from qxcore import units
from qxpulse import Arbitrary, PulseChannel, PulseSchedule, VirtualZ, Waveform

from qubex.simulator import Control, Coupling, QuantumSimulator, QuantumSystem, Transmon


def _driven_single_qubit() -> tuple[QuantumSystem, Control]:
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    waveform = np.full(40, 2 * np.pi * 0.01, dtype=np.complex128)
    durations = np.full(40, 0.1)
    control = Control(
        target=qubit,
        waveform=waveform,
        durations=durations,
        frequency=qubit.frequency,
    )
    return system, control


def test_controls_may_have_different_segment_counts() -> None:
    """Controls with the same duration may use independent segment grids."""
    system, control = _driven_single_qubit()
    qubit = system.get_object("Q0")
    second_control = Control(
        target=qubit,
        waveform=np.zeros(20, dtype=np.complex128),
        durations=np.full(20, 0.2),
        frequency=qubit.frequency,
    )

    parameters = QuantumSimulator(system).create_simulation_parameters(
        [control, second_control],
    )

    assert parameters["boundary_times"][-1] == pytest.approx(control.duration)


def test_simulation_parameters_combine_all_control_boundaries() -> None:
    """Simulation times should contain every control segment boundary."""
    qubit_0 = Transmon(label="Q0", dimension=2, frequency=5.0)
    qubit_1 = Transmon(label="Q1", dimension=2, frequency=5.2)
    system = QuantumSystem(objects=[qubit_0, qubit_1])
    controls = [
        Control(
            target=qubit_0,
            waveform=np.zeros(2, dtype=np.complex128),
            durations=np.array([0.25, 0.75]),
        ),
        Control(
            target=qubit_1,
            waveform=np.zeros(2, dtype=np.complex128),
            durations=np.array([0.4, 0.6]),
        ),
    ]

    parameters = QuantumSimulator(system).create_simulation_parameters(controls)

    assert_allclose(parameters["boundary_times"], [0.0, 0.25, 0.4, 1.0])


def test_simulation_parameters_preserve_piecewise_constant_controls() -> None:
    """QuTiP coefficients should preserve control jumps between boundaries."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.array([1.0, 3.0], dtype=np.complex128),
        durations=np.array([0.25, 0.75]),
    )

    parameters = QuantumSimulator(system).create_simulation_parameters([control])
    hamiltonian = parameters["hamiltonian"]
    assert isinstance(hamiltonian, qt.QobjEvo)
    static = system.get_rotating_object_hamiltonian(qubit.label)
    drive = 0.5 * (
        system.get_raising_operator(qubit.label)
        + system.get_lowering_operator(qubit.label)
    )

    assert_allclose(
        hamiltonian(0.249).full(),
        (static + drive).full(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(
        hamiltonian(0.25).full(),
        (static + 3 * drive).full(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_simulation_parameters_keep_drive_rotation_continuous() -> None:
    """Control ZOH should not discretize the analytic drive rotation."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.array([1.0], dtype=np.complex128),
        durations=np.array([1.0]),
        frequency=5.2,
    )
    time = 0.37

    parameters = QuantumSimulator(system).create_simulation_parameters([control])
    hamiltonian = parameters["hamiltonian"]
    assert isinstance(hamiltonian, qt.QobjEvo)
    delta = 2 * np.pi * (control.frequency - qubit.frequency)
    raising = system.get_raising_operator(qubit.label)
    lowering = system.get_lowering_operator(qubit.label)
    expected = system.get_rotating_object_hamiltonian(qubit.label) + 0.5 * (
        np.exp(-1j * delta * time) * raising + np.exp(1j * delta * time) * lowering
    )

    assert_allclose(
        hamiltonian(time).full(),
        expected.full(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_simulation_parameters_keep_coupling_rotation_continuous() -> None:
    """Coupling rotation should remain analytic between output times."""
    qubit_0 = Transmon(label="Q0", dimension=2, frequency=5.0)
    qubit_1 = Transmon(label="Q1", dimension=2, frequency=5.7)
    coupling = Coupling(pair=(qubit_0, qubit_1), strength=0.03)
    system = QuantumSystem(objects=[qubit_0, qubit_1], couplings=[coupling])
    control = Control(
        target=qubit_0,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([1.0]),
    )
    time = 0.37

    parameters = QuantumSimulator(system).create_simulation_parameters([control])
    hamiltonian = parameters["hamiltonian"]
    assert isinstance(hamiltonian, qt.QobjEvo)
    static = sum(
        (
            system.get_rotating_object_hamiltonian(label)
            for label in system.object_labels
        ),
        start=system.zero_matrix,
    )
    operator = system.get_raising_operator(qubit_0.label) @ (
        system.get_lowering_operator(qubit_1.label)
    )
    angular_strength = 2 * np.pi * coupling.strength
    detuning = system.get_coupling_detuning(coupling.label)
    expected = static + angular_strength * (
        np.exp(-1j * detuning * time) * operator
        + np.exp(1j * detuning * time) * operator.dag()
    )

    assert_allclose(
        hamiltonian(time).full(),
        expected.full(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_controls_must_have_the_same_duration() -> None:
    """Controls ending at different times should be rejected."""
    system, control = _driven_single_qubit()
    qubit = system.get_object("Q0")
    longer_control = Control(
        target=qubit,
        waveform=np.zeros(control.n_segments, dtype=np.complex128),
        durations=np.full(control.n_segments, 0.2),
        frequency=qubit.frequency,
    )

    with pytest.raises(ValueError, match="same duration"):
        QuantumSimulator(system).create_simulation_parameters(
            [control, longer_control],
        )


def test_control_uses_waveform_sampling_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control should preserve the sampling period stored by a waveform."""
    waveform = Arbitrary([0.01, 0.02], sampling_period=0.5)
    monkeypatch.setattr(Waveform, "SAMPLING_PERIOD", 1.0)

    control = Control(
        target="Q0",
        waveform=waveform,
        frequency=5.0,
    )

    assert_allclose(control.durations, [0.5, 0.5])


def test_control_get_samples_uses_zero_order_hold() -> None:
    """Control samples should use zero-order hold on nonuniform segments."""
    control = Control(
        target="Q0",
        waveform=np.array([1.0, 2.0j, 3.0], dtype=np.complex128),
        durations=np.array([0.2, 0.3, 0.5]),
        frequency=5.0,
    )

    samples = control.get_samples(
        np.array([-0.1, 0.0, 0.199, 0.2, 0.499, 0.5, 1.0, 1.1])
    )

    assert_allclose(samples, [0.0, 1.0, 1.0, 2.0j, 2.0j, 3.0, 3.0, 0.0])


def test_control_get_frame_shifts_uses_segment_and_terminal_boundaries() -> None:
    """Frame shifts should switch at segment starts and persist after the end."""
    control = Control(
        target="Q0",
        waveform=np.zeros(3, dtype=np.complex128),
        durations=np.array([0.2, 0.3, 0.5]),
        frequency=5.0,
        frame_shifts=np.array([0.1, 0.2, 0.3]),
        final_frame_shift=0.4,
    )

    shifts = control.get_frame_shifts(
        np.array([-0.1, 0.0, 0.199, 0.2, 0.499, 0.5, 0.999, 1.0, 1.1])
    )

    assert_allclose(shifts, [0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4])


def test_empty_control_get_samples_returns_zero() -> None:
    """An empty control should evaluate to zero at every requested time."""
    control = Control(
        target="Q0",
        waveform=np.array([], dtype=np.complex128),
        durations=np.array([], dtype=np.float64),
        frequency=5.0,
    )

    samples = control.get_samples(np.array([0.0, 1.0]))

    assert_allclose(samples, [0.0, 0.0])


def test_control_owns_read_only_segment_data() -> None:
    """Control should own read-only waveform, duration, and frame-shift arrays."""
    waveform = np.array([1.0, 2.0], dtype=np.complex128)
    durations = np.array([0.2, 0.3])
    frame_shifts = np.array([0.1, 0.2])

    control = Control(
        target="Q0",
        waveform=waveform,
        durations=durations,
        frequency=5.0,
        frame_shifts=frame_shifts,
    )
    waveform[0] = 9.0
    durations[0] = 9.0
    frame_shifts[0] = 9.0

    assert_allclose(control.waveform, [1.0, 2.0])
    assert_allclose(control.durations, [0.2, 0.3])
    assert_allclose(control.frame_shifts, [0.1, 0.2])
    with pytest.raises(ValueError, match="read-only"):
        control.waveform[0] = 9.0
    with pytest.raises(ValueError, match="read-only"):
        control.durations[0] = 9.0
    with pytest.raises(ValueError, match="read-only"):
        control.frame_shifts[0] = 9.0
    with pytest.raises(ValueError, match="read-only"):
        control.times[0] = 9.0


def test_control_rejects_mismatched_segment_data() -> None:
    """Control should require one duration for each waveform segment."""
    with pytest.raises(ValueError, match="waveform and durations"):
        Control(
            target="Q0",
            waveform=np.array([1.0, 2.0], dtype=np.complex128),
            durations=np.array([0.5]),
            frequency=5.0,
        )

    with pytest.raises(ValueError, match="waveform and frame_shifts"):
        Control(
            target="Q0",
            waveform=np.array([1.0, 2.0], dtype=np.complex128),
            durations=np.array([0.5, 0.5]),
            frequency=5.0,
            frame_shifts=np.array([0.0]),
        )


@pytest.mark.parametrize("frame_shift", [np.nan, np.inf, -np.inf])
def test_control_rejects_nonfinite_frame_shifts(frame_shift: float) -> None:
    """Control frame metadata should contain only finite values."""
    with pytest.raises(ValueError, match="Frame shifts must be finite"):
        Control(
            target="Q0",
            waveform=np.array([1.0], dtype=np.complex128),
            durations=np.array([0.5]),
            frequency=5.0,
            frame_shifts=np.array([frame_shift]),
        )
    with pytest.raises(ValueError, match="Frame shifts must be finite"):
        Control(
            target="Q0",
            waveform=np.array([1.0], dtype=np.complex128),
            durations=np.array([0.5]),
            frequency=5.0,
            final_frame_shift=frame_shift,
        )


@pytest.mark.parametrize("duration", [0.0, -0.1, np.nan, np.inf])
def test_control_rejects_invalid_segment_durations(duration: float) -> None:
    """Control should require every segment duration to be finite and positive."""
    with pytest.raises(ValueError, match="finite and greater than zero"):
        Control(
            target="Q0",
            waveform=np.array([1.0], dtype=np.complex128),
            durations=np.array([duration]),
            frequency=5.0,
        )


def test_control_normalizes_tunits_frequency_to_ghz() -> None:
    """Control should normalize tunits frequency input to a GHz float."""
    control = Control(
        target="Q0",
        waveform=np.array([0.0], dtype=np.complex128),
        durations=np.array([1.0]),
        frequency=5100 * units.MHz,
    )

    assert control.frequency == pytest.approx(5.1, rel=1e-12, abs=0.0)
    assert isinstance(control.frequency, float)


def test_create_simulation_parameters_accepts_pulse_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PulseSchedule metadata should be converted into simulator controls."""
    monkeypatch.setattr(Waveform, "SAMPLING_PERIOD", 0.5)
    system, _ = _driven_single_qubit()
    qubit = system.get_object("Q0")
    channel = PulseChannel(
        label="drive",
        frequency=qubit.frequency,
        target=qubit.label,
    )
    with PulseSchedule([channel]) as schedule:
        schedule.add("drive", Arbitrary([0.01, 0.02]))

    monkeypatch.setattr(Waveform, "SAMPLING_PERIOD", 1.0)
    parameters = QuantumSimulator(system).create_simulation_parameters(schedule)

    assert parameters["boundary_times"][-1] == pytest.approx(schedule.duration)
    assert isinstance(parameters["hamiltonian"], qt.QobjEvo)


def test_pulse_schedule_controls_support_channel_sampling_periods() -> None:
    """PulseSchedule controls should retain each channel sampling period."""
    qubit_0 = Transmon(label="Q0", dimension=2, frequency=5.0)
    qubit_1 = Transmon(label="Q1", dimension=2, frequency=5.2)
    system = QuantumSystem(objects=[qubit_0, qubit_1])

    channel_0 = PulseChannel(
        label="drive-0",
        frequency=0.0,
        target=qubit_0.label,
    )
    channel_1 = PulseChannel(
        label="drive-1",
        frequency=qubit_1.frequency,
        target=qubit_1.label,
    )

    with PulseSchedule([channel_0, channel_1]) as schedule:
        schedule.add(
            "drive-0",
            Arbitrary([0.01, 0.02], sampling_period=0.5),
        )
        schedule.add(
            "drive-1",
            Arbitrary([0.03], sampling_period=1.0),
        )

    result = QuantumSimulator(system).simulate(schedule, dt=0.5)

    assert result.controls[0].frequency == 0.0
    assert_allclose(result.controls[0].durations, [0.5, 0.5])
    assert_allclose(result.controls[1].durations, [1.0])


def test_pulse_schedule_applies_intermediate_frame_shifts_to_waveforms() -> None:
    """Intermediate virtual Z shifts should change later waveform phases."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    channel = PulseChannel(
        label="drive",
        frequency=qubit.frequency,
        target=qubit.label,
    )
    with PulseSchedule([channel]) as schedule:
        schedule.add("drive", Arbitrary([1.0], sampling_period=0.5))
        schedule.add("drive", VirtualZ(np.pi / 2))
        schedule.add("drive", Arbitrary([1.0], sampling_period=0.5))

    result = QuantumSimulator(system).simulate(schedule, dt=0.5)

    assert_allclose(result.controls[0].waveform, [1.0, -1.0j], atol=1e-12)
    assert_allclose(result.controls[0].frame_shifts, [0.0, -np.pi / 2])
    assert result.controls[0].final_frame_shift == pytest.approx(-np.pi / 2)


def test_pulse_schedule_frame_shifts_change_result_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intermediate virtual Z shifts should rotate result coordinates at boundaries."""
    monkeypatch.setattr(Waveform, "SAMPLING_PERIOD", 0.5)
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    channel = PulseChannel(
        label="drive",
        frequency=qubit.frequency,
        target=qubit.label,
    )
    with PulseSchedule([channel]) as schedule:
        schedule.add("drive", Arbitrary([0.0], sampling_period=0.5))
        schedule.add("drive", VirtualZ(np.pi / 2))
        schedule.add("drive", Arbitrary([0.0], sampling_period=0.5))
    initial_state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

    result = QuantumSimulator(system).simulate(
        schedule,
        initial_state=initial_state,
        dt=0.5,
    )
    raw = result.get_substates("Q0", apply_frame_shifts=False)
    logical = result.get_substates("Q0")
    rotation = (0.5j * np.pi * qt.num(2)).expm()

    assert_allclose(result.get_frame_shifts("Q0"), [0.0, -np.pi / 2, -np.pi / 2])
    assert_allclose(logical[0].full(), raw[0].full(), rtol=0.0, atol=1e-12)
    for raw_state, logical_state in zip(raw[1:], logical[1:], strict=True):
        assert_allclose(
            logical_state.full(),
            (rotation @ raw_state @ rotation.dag()).full(),
            rtol=0.0,
            atol=1e-12,
        )


def test_simulate_includes_all_control_boundaries() -> None:
    """Integration grid should split intervals at every control boundary."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    controls = [
        Control(
            target=qubit,
            waveform=np.zeros(2, dtype=np.complex128),
            durations=np.array([0.15, 0.25]),
        ),
        Control(
            target=qubit,
            waveform=np.zeros(2, dtype=np.complex128),
            durations=np.array([0.1, 0.3]),
        ),
    ]

    result = QuantumSimulator(system).simulate(controls, dt=0.2)

    assert_allclose(result.times, [0.0, 0.1, 0.15, 0.2, 0.4], atol=1e-12)
    assert np.max(np.diff(result.times)) <= 0.2 + 1e-12


def test_simulate_evaluates_continuous_terms_at_interval_midpoints() -> None:
    """Continuous Hamiltonian terms should be evaluated at interval midpoints."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    duration = 0.2
    amplitude = 0.3 + 0.1j
    control = Control(
        target=qubit,
        waveform=np.array([amplitude]),
        durations=np.array([duration]),
        frequency=5.75,
    )

    result = QuantumSimulator(system).simulate([control], dt=duration)

    midpoint = duration / 2
    detuning = 2 * np.pi * (control.frequency - qubit.frequency)
    gamma = 0.5 * amplitude * np.exp(-1j * detuning * midpoint)
    expected_hamiltonian = system.get_rotating_hamiltonian(midpoint)
    expected_hamiltonian += gamma * system.get_raising_operator(qubit.label)
    expected_hamiltonian += np.conj(gamma) * system.get_lowering_operator(qubit.label)
    expected_unitary = (-1j * expected_hamiltonian * duration).expm()
    assert_allclose(
        result.propagators[-1].full(),
        expected_unitary.full(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_simulate_propagates_each_constant_control_segment() -> None:
    """Discontinuous controls should propagate as separate constant segments."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    amplitudes = np.array([0.3 + 0.1j, -0.2j])
    durations = np.array([0.15, 0.25])
    control = Control(
        target=qubit,
        waveform=amplitudes,
        durations=durations,
    )

    result = QuantumSimulator(system).simulate([control], dt=control.duration)

    lowering_operator = system.get_lowering_operator(qubit.label)
    raising_operator = system.get_raising_operator(qubit.label)
    segment_unitaries = []
    for amplitude, duration in zip(amplitudes, durations, strict=True):
        gamma = 0.5 * amplitude
        hamiltonian = gamma * raising_operator + np.conj(gamma) * lowering_operator
        segment_unitaries.append((-1j * hamiltonian * duration).expm())
    expected = segment_unitaries[1] @ segment_unitaries[0]
    assert_allclose(
        result.propagators[-1].full(),
        expected.full(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_simulate_accepts_density_matrix_initial_state() -> None:
    """Piecewise-exponential simulation should accept density matrices."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    initial_state = qt.qeye(2) / 2

    result = QuantumSimulator(system).simulate(
        [control],
        initial_state=initial_state,
        dt=0.1,
    )

    assert result.initial_state.isoper
    assert_allclose(result.initial_state.full(), initial_state.full())
    assert_allclose(result.final_state.full(), initial_state.full())


def test_simulate_returns_state_and_propagator_lists_by_default() -> None:
    """Simulate should return Qobj state and propagator trajectories as lists."""
    system, control = _driven_single_qubit()

    result = QuantumSimulator(system).simulate([control])

    assert isinstance(result.states, list)
    assert isinstance(result.propagators, list)
    assert len(result.states) == len(result.times)
    assert len(result.propagators) == len(result.times)
    assert all(isinstance(state, qt.Qobj) for state in result.states)
    assert all(propagator.isoper for propagator in result.propagators)


def test_simulate_can_skip_cumulative_propagators() -> None:
    """Disabling propagators should retain the same simulated state trajectory."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)

    with_propagators = simulator.simulate([control])
    without_propagators = simulator.simulate(
        [control],
        compute_propagators=False,
    )

    assert without_propagators.propagators == []
    assert len(without_propagators.states) == len(with_propagators.states)
    for state, expected in zip(
        without_propagators.states,
        with_propagators.states,
        strict=True,
    ):
        assert_allclose(state.full(), expected.full(), rtol=1e-12, atol=1e-12)


def test_unitaries_is_a_deprecated_alias_for_propagators() -> None:
    """The legacy unitaries attribute should warn and return propagators."""
    system, control = _driven_single_qubit()
    result = QuantumSimulator(system).simulate([control])

    with pytest.warns(DeprecationWarning, match="propagators"):
        unitaries = result.unitaries

    assert unitaries is result.propagators


def test_create_simulation_model_rejects_mismatched_operator_dimensions() -> None:
    """Simulation models should reject operators with incompatible right dims."""
    q0 = Transmon(label="Q0", dimension=2, frequency=5.0)
    q1 = Transmon(label="Q1", dimension=2, frequency=5.2)
    system = QuantumSystem(objects=[q0, q1])
    control = Control(
        target=q0,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    initial_state = qt.Qobj(np.eye(4), dims=[[2, 2], [4]])

    with pytest.raises(ValueError, match="dims of the initial state"):
        QuantumSimulator(system).create_simulation_model(
            [control],
            initial_state=initial_state,
        )


@pytest.mark.parametrize("dt", [0.0, -0.1, np.inf, np.nan])
def test_simulate_rejects_invalid_dt(dt: float) -> None:
    """Piecewise-exponential simulation should require a valid maximum step."""
    system, control = _driven_single_qubit()

    with pytest.raises(ValueError, match="dt must be finite and greater than zero"):
        QuantumSimulator(system).simulate([control], dt=dt)


@pytest.mark.parametrize("n_samples", [-1, 0, 1])
def test_simulate_rejects_sample_counts_that_drop_endpoints(
    n_samples: int,
) -> None:
    """Positive-duration results should retain both temporal endpoints."""
    system, control = _driven_single_qubit()

    with pytest.raises(ValueError, match="n_samples must be at least 2"):
        QuantumSimulator(system).simulate([control], n_samples=n_samples)


def test_simulate_output_sampling_preserves_temporal_endpoints() -> None:
    """Uniform output sampling should preserve the initial and final points."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)

    full_result = simulator.simulate([control])
    sampled_result = simulator.simulate([control], n_samples=2)

    assert_allclose(sampled_result.times, [0.0, control.duration])
    assert_allclose(
        sampled_result.final_state.full(),
        full_result.final_state.full(),
    )
    assert_allclose(
        sampled_result.propagators[-1].full(),
        full_result.propagators[-1].full(),
    )


def test_simulate_returns_uniform_output_times() -> None:
    """Requested simulate outputs should be uniformly spaced in physical time."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.array([0.1], dtype=np.complex128),
        durations=np.array([1.0]),
    )

    result = QuantumSimulator(system).simulate(
        [control],
        dt=0.4,
        n_samples=4,
    )

    assert_allclose(result.times, np.linspace(0.0, 1.0, 4), rtol=0.0, atol=0.0)
    assert len(result.states) == 4
    assert len(result.propagators) == 4


def test_simulate_keeps_final_frame_shift_as_result_metadata() -> None:
    """Simulate should not turn a logical frame shift into physical evolution."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    frame_shift = np.pi / 2
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
        final_frame_shift=frame_shift,
    )
    initial_ket = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

    result = QuantumSimulator(system).simulate(
        [control],
        initial_state=initial_ket,
    )

    initial_state = qt.ket2dm(initial_ket)
    assert_allclose(result.propagators[-1].full(), system.identity_matrix.full())
    assert_allclose(result.final_state.full(), initial_state.full())
    assert result.controls[0].final_frame_shift == pytest.approx(frame_shift)


def test_simulate_does_not_add_a_point_for_zero_duration_frame_shift() -> None:
    """A logical frame shift alone should not add a physical evolution point."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    frame_shift = np.pi / 2
    control = Control(
        target=qubit,
        waveform=np.array([], dtype=np.complex128),
        durations=np.array([], dtype=np.float64),
        final_frame_shift=frame_shift,
    )
    initial_ket = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    initial_state = qt.ket2dm(initial_ket)

    result = QuantumSimulator(system).simulate(
        [control],
        initial_state=initial_ket,
        n_samples=4,
    )

    assert_allclose(result.times, [0.0])
    assert_allclose(result.initial_state.full(), initial_state.full())
    assert_allclose(result.final_state.full(), initial_state.full())
    assert len(result.propagators) == 1


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
def test_qutip_solver_returns_one_point_for_zero_duration(
    solver_name: str,
) -> None:
    """A zero-duration solver trajectory should contain only its initial point."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.array([], dtype=np.complex128),
        durations=np.array([], dtype=np.float64),
    )
    solver = getattr(QuantumSimulator(system), solver_name)

    result = solver([control], n_samples=4)

    assert_allclose(result.times, [0.0])
    assert len(result.states) == 1


def test_simulate_converges_for_rotating_coupling() -> None:
    """Midpoint propagation should converge quadratically for rotating coupling."""
    qubit_0 = Transmon(label="Q0", dimension=2, frequency=5.0)
    qubit_1 = Transmon(label="Q1", dimension=2, frequency=5.7)
    coupling = Coupling(pair=(qubit_0, qubit_1), strength=0.03)
    system = QuantumSystem(objects=[qubit_0, qubit_1], couplings=[coupling])
    duration = 1.0
    control = Control(
        target=qubit_0,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([duration]),
    )
    simulator = QuantumSimulator(system)

    coarse = simulator.simulate([control], dt=0.2).propagators[-1]
    fine = simulator.simulate([control], dt=0.1).propagators[-1]

    detuning = system.get_coupling_detuning(coupling.label)
    number_operator = system.get_number_operator(qubit_0.label)
    rotating_frame = (-1j * detuning * duration * number_operator).expm()
    effective_hamiltonian = (
        system.get_coupling_hamiltonian(coupling.label) - detuning * number_operator
    )
    exact = rotating_frame @ (-1j * effective_hamiltonian * duration).expm()
    coarse_error = np.linalg.norm(coarse.full() - exact.full())
    fine_error = np.linalg.norm(fine.full() - exact.full())

    assert fine_error < coarse_error / 3.5


@pytest.mark.parametrize(
    "method_name",
    [
        "mesolve",
        "sesolve",
        "propagator",
        "gate_fidelity",
        "create_simulation_parameters",
        "create_simulation_model",
    ],
)
def test_qutip_method_signature_hides_compatibility_dt(method_name: str) -> None:
    """The deprecated dt argument should not appear in QuTiP method signatures."""
    parameters = signature(getattr(QuantumSimulator, method_name)).parameters

    assert "dt" not in parameters
    assert "kwargs" in parameters


@pytest.mark.parametrize(
    "method_name",
    ["average_gate_fidelity", "process_fidelity"],
)
def test_new_fidelity_method_signatures_need_no_compatibility_kwargs(
    method_name: str,
) -> None:
    """New fidelity methods should expose only their supported arguments."""
    parameters = signature(getattr(QuantumSimulator, method_name)).parameters

    assert "dt" not in parameters
    assert "kwargs" not in parameters


@pytest.mark.parametrize(
    ("method_name", "expected_default"),
    [("simulate", True), ("mesolve", False), ("sesolve", False)],
)
def test_solver_compute_propagators_defaults(
    method_name: str,
    expected_default: bool,
) -> None:
    """Only simulate should compute propagators by default."""
    parameter = signature(getattr(QuantumSimulator, method_name)).parameters[
        "compute_propagators"
    ]

    assert parameter.default is expected_default


@pytest.mark.parametrize(
    "method_name",
    ["create_simulation_parameters", "create_simulation_model"],
)
def test_qutip_model_builder_warns_and_ignores_compatibility_dt(
    method_name: str,
) -> None:
    """Model builders should warn and ignore the deprecated dt keyword."""
    system, control = _driven_single_qubit()
    method = getattr(QuantumSimulator(system), method_name)

    with pytest.warns(DeprecationWarning, match="dt.*deprecated.*ignored"):
        first = method([control], dt=0.05)
    with pytest.warns(DeprecationWarning, match="dt.*deprecated.*ignored"):
        second = method([control], dt=0.2)

    first_times = (
        first["boundary_times"] if isinstance(first, dict) else first.boundary_times
    )
    second_times = (
        second["boundary_times"] if isinstance(second, dict) else second.boundary_times
    )
    assert_allclose(first_times, second_times, rtol=0.0, atol=0.0)


def test_propagator_warns_and_ignores_compatibility_dt() -> None:
    """Propagator should accept the legacy dt keyword without forwarding it."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    identity_superoperator = qt.to_super(system.identity_matrix)

    with (
        patch(
            "qxsimulator.simulation.quantum_simulator.qt.propagator",
            return_value=[system.identity_matrix, identity_superoperator],
        ) as mocked_propagator,
        pytest.warns(
            DeprecationWarning,
            match="dt.*deprecated.*ignored",
        ),
    ):
        result = simulator.propagator([control], dt=0.05)

    assert result == [system.identity_matrix, identity_superoperator]
    assert mocked_propagator.call_args is not None
    assert "dt" not in mocked_propagator.call_args.kwargs
    assert mocked_propagator.call_args.kwargs["options"]["max_step"] == pytest.approx(
        0.05
    )


def test_propagator_uses_common_solver_option_defaults() -> None:
    """Propagator should integrate through every control boundary."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    identity_superoperator = qt.to_super(system.identity_matrix)

    with patch(
        "qxsimulator.simulation.quantum_simulator.qt.propagator",
        return_value=[system.identity_matrix, identity_superoperator],
    ) as mocked_propagator:
        simulator.propagator([control])

    assert mocked_propagator.call_args is not None
    forwarded = mocked_propagator.call_args.kwargs
    assert forwarded["options"] == {
        "max_step": pytest.approx(0.05),
        "nsteps": 2500,
    }
    assert_allclose(forwarded["t"], control.times, rtol=0.0, atol=1e-12)


def test_propagator_scales_nsteps_with_solver_interval() -> None:
    """Propagator work limit should cover its longest boundary interval."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(3, dtype=np.complex128),
        durations=np.array([20.0, 0.01, 20.0]),
    )
    identity_superoperator = qt.to_super(system.identity_matrix)

    with patch(
        "qxsimulator.simulation.quantum_simulator.qt.propagator",
        return_value=[system.identity_matrix, identity_superoperator],
    ) as mocked_propagator:
        QuantumSimulator(system).propagator([control])

    assert mocked_propagator.call_args is not None
    forwarded_options = mocked_propagator.call_args.kwargs["options"]
    required_steps = int(
        np.ceil(np.max(np.diff(control.times)) / forwarded_options["max_step"])
    )
    assert forwarded_options["nsteps"] == 2 * required_steps


def test_propagator_keeps_final_frame_shift_out_of_raw_evolution() -> None:
    """Propagator should not turn a logical frame shift into physical evolution."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
        final_frame_shift=np.pi / 2,
    )

    result = QuantumSimulator(system).propagator([control])

    assert len(result) == len(control.times)
    assert all(propagator.isunitary for propagator in result)
    assert_allclose(result[-1].full(), system.identity_matrix.full())


def test_propagator_returns_superoperator_for_dissipative_system() -> None:
    """A dissipative propagator should represent the full quantum channel."""
    qubit = Transmon(
        label="Q0",
        dimension=2,
        frequency=5.0,
        relaxation_rate=0.01,
    )
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )

    result = QuantumSimulator(system).propagator([control])

    assert len(result) == len(control.times)
    assert all(propagator.issuper for propagator in result)


def test_average_gate_fidelity_accepts_closed_system_unitary() -> None:
    """Average gate fidelity should compare a closed-system unitary directly."""
    qubit_0 = Transmon(label="Q0", dimension=3, frequency=5.0)
    qubit_1 = Transmon(label="Q1", dimension=3, frequency=5.2)
    system = QuantumSystem(objects=[qubit_0, qubit_1])
    control = Control(
        target=qubit_0,
        waveform=np.full(10, 2 * np.pi * 0.01, dtype=np.complex128),
        durations=np.full(10, 0.1),
    )
    simulator = QuantumSimulator(system)
    target = qt.qeye(2) & qt.qeye(2)
    unitary = simulator.propagator([control])[-1]
    truncated_unitary = system.truncate_operator(unitary)
    dimension = target.shape[0]
    process_fidelity = qt.process_fidelity(truncated_unitary, target)
    assert process_fidelity is not None
    average_survival_probability = (
        truncated_unitary.dag() @ truncated_unitary
    ).tr().real / dimension
    expected = (dimension * process_fidelity + average_survival_probability) / (
        dimension + 1
    )

    fidelity = simulator.average_gate_fidelity(
        [control],
        target,
    )

    assert fidelity == pytest.approx(expected)


@pytest.mark.parametrize("as_superoperator", [False, True])
def test_average_gate_fidelity_counts_subspace_leakage(
    as_superoperator: bool,
) -> None:
    """Average gate fidelity should count probability lost outside the subspace."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    leaky_unitary = qt.Qobj(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    propagator = qt.to_super(leaky_unitary) if as_superoperator else leaky_unitary

    with patch.object(simulator, "propagator", return_value=[propagator]):
        process_fidelity = simulator.process_fidelity([control], qt.qeye(2))
        average_gate_fidelity = simulator.average_gate_fidelity(
            [control],
            qt.qeye(2),
        )

    assert process_fidelity == pytest.approx(1 / 4)
    assert average_gate_fidelity == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "method_name",
    ["process_fidelity", "average_gate_fidelity"],
)
def test_gate_fidelity_methods_accept_full_qudit_space(method_name: str) -> None:
    """Fidelity evaluation should support the full physical Hilbert space."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    target = qt.Qobj(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )

    with patch.object(simulator, "propagator", return_value=[target]):
        fidelity = getattr(simulator, method_name)(
            [control],
            target,
            levels="full",
        )

    assert fidelity == pytest.approx(1.0)


@pytest.mark.parametrize(
    "method_name",
    ["process_fidelity", "average_gate_fidelity"],
)
@pytest.mark.parametrize("as_superoperator", [False, True])
def test_gate_fidelity_methods_accept_selected_object_levels(
    method_name: str,
    as_superoperator: bool,
) -> None:
    """Fidelity evaluation should compare a target on an explicit ef subspace."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    actual = system.unitary({"Q0": "X"}, levels={"Q0": (1, 2)})
    propagator = qt.to_super(actual) if as_superoperator else actual

    with patch.object(simulator, "propagator", return_value=[propagator]):
        fidelity = getattr(simulator, method_name)(
            [control],
            qt.sigmax(),
            levels={"Q0": (1, 2)},
        )

    assert fidelity == pytest.approx(1.0)


def test_gate_fidelity_rejects_target_that_leaves_evaluation_space() -> None:
    """A full target should preserve the selected space before comparison."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    target = system.unitary({"Q0": "X"}, levels={"Q0": (1, 2)})

    with (
        patch.object(
            simulator,
            "propagator",
            return_value=[system.identity_matrix],
        ),
        pytest.raises(ValueError, match="preserve the selected evaluation space"),
    ):
        simulator.average_gate_fidelity([control], target)


def test_gate_fidelity_mapping_keeps_unspecified_objects_computational() -> None:
    """A level mapping should override named objects and keep other objects qubit-like."""
    q0 = Transmon(label="Q0", dimension=3, frequency=5.0)
    q1 = Transmon(label="Q1", dimension=3, frequency=5.2)
    system = QuantumSystem(objects=[q0, q1])
    control = Control(
        target=q0,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    target = qt.tensor(qt.qeye(2), qt.qeye(2))

    with patch.object(
        simulator,
        "propagator",
        return_value=[system.identity_matrix],
    ):
        fidelity = simulator.average_gate_fidelity(
            [control],
            target,
            levels={"Q0": (1, 2)},
        )

    assert fidelity == pytest.approx(1.0)


def test_gate_fidelity_accepts_embedded_system_unitary_target() -> None:
    """A full-system target should be projected onto the evaluation space."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    target = system.unitary({"Q0": "X"})

    with patch.object(simulator, "propagator", return_value=[target]):
        fidelity = simulator.average_gate_fidelity([control], target)

    assert fidelity == pytest.approx(1.0)


@pytest.mark.parametrize(
    "method_name",
    ["process_fidelity", "average_gate_fidelity"],
)
def test_gate_fidelity_methods_accept_unitary_specification(
    method_name: str,
) -> None:
    """Fidelity methods should embed a labeled gate specification."""
    q0 = Transmon(label="Q0", dimension=3, frequency=5.0)
    q1 = Transmon(label="Q1", dimension=3, frequency=5.2)
    system = QuantumSystem(objects=[q0, q1])
    control = Control(
        target=q0,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    target_specification = {"Q0": "X", "Q1": "H"}
    actual = system.unitary(target_specification)

    with patch.object(simulator, "propagator", return_value=[actual]):
        fidelity = getattr(simulator, method_name)(
            [control],
            target_specification,
        )

    assert fidelity == pytest.approx(1.0)


def test_gate_fidelity_embeds_unitary_specification_on_selected_levels() -> None:
    """A level mapping should control target embedding and evaluation."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    levels = {"Q0": (1, 2)}
    actual = system.unitary({"Q0": "X"}, levels=levels)

    with patch.object(simulator, "propagator", return_value=[actual]):
        fidelity = simulator.average_gate_fidelity(
            [control],
            {"Q0": "X"},
            levels=levels,
        )

    assert fidelity == pytest.approx(1.0)


def test_gate_fidelity_embeds_unitary_specification_in_full_space() -> None:
    """A labeled gate should leave unselected physical levels unchanged."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
    )
    simulator = QuantumSimulator(system)
    actual = system.unitary({"Q0": "X"})

    with patch.object(simulator, "propagator", return_value=[actual]):
        fidelity = simulator.process_fidelity(
            [control],
            {"Q0": "X"},
            levels="full",
        )

    assert fidelity == pytest.approx(1.0)


def test_gate_fidelity_rejects_invalid_evaluation_levels() -> None:
    """Evaluation levels should reject unknown objects and invalid indices."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)

    with pytest.raises(ValueError, match="Object Q1 does not exist"):
        simulator.process_fidelity(
            [control],
            qt.qeye(2),
            levels={"Q1": (0, 1)},
        )
    with pytest.raises(ValueError, match=r"outside.*Q0"):
        simulator.process_fidelity(
            [control],
            qt.qeye(2),
            levels={"Q0": (0, 2)},
        )


def test_gate_fidelity_is_deprecated_alias() -> None:
    """Gate fidelity should warn and delegate to average gate fidelity."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)

    with (
        patch.object(
            simulator,
            "average_gate_fidelity",
            return_value=0.99,
        ) as mocked_average_gate_fidelity,
        pytest.warns(
            DeprecationWarning,
            match="average_gate_fidelity",
        ),
    ):
        fidelity = simulator.gate_fidelity(
            [control],
            system.identity_matrix,
        )

    assert fidelity == pytest.approx(0.99)
    mocked_average_gate_fidelity.assert_called_once_with(
        controls=[control],
        target_unitary=system.identity_matrix,
        levels="computational",
        options=None,
    )


def test_gate_fidelity_warns_and_ignores_compatibility_dt() -> None:
    """Deprecated gate fidelity should accept legacy dt without forwarding it."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)

    with (
        patch.object(
            simulator,
            "average_gate_fidelity",
            return_value=1.0,
        ) as mocked_average_gate_fidelity,
        pytest.warns(
            DeprecationWarning,
            match="dt.*deprecated.*ignored",
        ),
    ):
        fidelity = simulator.gate_fidelity(
            [control],
            system.identity_matrix,
            dt=0.05,
        )

    assert fidelity == pytest.approx(1.0)
    mocked_average_gate_fidelity.assert_called_once_with(
        controls=[control],
        target_unitary=system.identity_matrix,
        levels="computational",
        options=None,
    )


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
def test_qutip_solver_warns_and_ignores_compatibility_dt(
    solver_name: str,
) -> None:
    """The deprecated dt keyword should warn without affecting solver results."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    initial_state = system.state({"Q0": "0"})
    solver = getattr(simulator, solver_name)

    with pytest.warns(
        DeprecationWarning,
        match="dt.*deprecated.*ignored",
    ) as first_warnings:
        first = solver([control], initial_state=initial_state, dt=0.05)
    with pytest.warns(DeprecationWarning, match="dt.*deprecated.*ignored"):
        second = solver([control], initial_state=initial_state, dt=0.2)

    assert first_warnings[0].filename == __file__
    assert_allclose(first.times, second.times, rtol=0.0, atol=0.0)
    assert len(first.states) == len(second.states)
    for first_state, second_state in zip(first.states, second.states, strict=True):
        assert_allclose(
            first_state.full(),
            second_state.full(),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
def test_qutip_solver_rejects_unknown_compatibility_keyword(
    solver_name: str,
) -> None:
    """Unknown compatibility keywords should raise TypeError."""
    system, control = _driven_single_qubit()
    solver = getattr(QuantumSimulator(system), solver_name)

    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        solver([control], unknown=True)


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
def test_qutip_solver_forwards_options(solver_name: str) -> None:
    """Solver options should include a safe default maximum step."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    options = {"rtol": 1e-7, "atol": 1e-9}
    solver = getattr(simulator, solver_name)

    with patch(
        f"qxsimulator.simulation.quantum_simulator.qt.{solver_name}",
        wraps=getattr(qt, solver_name),
    ) as mocked_solver:
        solver(
            [control],
            initial_state=system.ground_state,
            options=options,
        )

    assert mocked_solver.call_args is not None
    forwarded_options = mocked_solver.call_args.kwargs["options"]
    assert forwarded_options["rtol"] == options["rtol"]
    assert forwarded_options["atol"] == options["atol"]
    assert forwarded_options["max_step"] == pytest.approx(0.05)
    assert forwarded_options["nsteps"] == 2500
    assert options == {"rtol": 1e-7, "atol": 1e-9}


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
@pytest.mark.parametrize("n_samples", [-1, 0, 1])
def test_qutip_solver_rejects_sample_counts_that_drop_endpoints(
    solver_name: str,
    n_samples: int,
) -> None:
    """Positive-duration solver results should retain both temporal endpoints."""
    system, control = _driven_single_qubit()
    solver = getattr(QuantumSimulator(system), solver_name)

    with pytest.raises(ValueError, match="n_samples must be at least 2"):
        solver([control], n_samples=n_samples)


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
@pytest.mark.parametrize("compute_propagators", [False, True])
def test_qutip_solver_separates_boundary_and_uniform_output_times(
    solver_name: str,
    compute_propagators: bool,
) -> None:
    """Solvers should checkpoint every boundary but return uniform output times."""
    qubit = Transmon(
        label="Q0",
        dimension=2,
        frequency=5.0,
        relaxation_rate=0.01,
    )
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(3, dtype=np.complex128),
        durations=np.array([0.2, 0.3, 0.5]),
    )
    simulator = QuantumSimulator(system)
    solver = getattr(simulator, solver_name)
    output_times = np.linspace(0.0, 1.0, 4)
    solver_times = np.array([0.0, 0.2, 1 / 3, 0.5, 2 / 3, 1.0])

    with patch(
        f"qxsimulator.simulation.quantum_simulator.qt.{solver_name}",
        wraps=getattr(qt, solver_name),
    ) as mocked_solver:
        result = solver(
            [control],
            initial_state=system.ground_state,
            n_samples=4,
            compute_propagators=compute_propagators,
        )

    assert mocked_solver.call_args is not None
    assert_allclose(
        mocked_solver.call_args.kwargs["tlist"],
        solver_times,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(result.times, output_times, rtol=0.0, atol=0.0)
    assert len(result.states) == 4
    assert len(result.propagators) == (4 if compute_propagators else 0)
    assert result.model is not None
    assert_allclose(
        result.model.boundary_times,
        control.times,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
def test_qutip_solver_preserves_explicit_maximum_step(solver_name: str) -> None:
    """An explicit maximum step should override the control-derived default."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    solver = getattr(simulator, solver_name)

    with patch(
        f"qxsimulator.simulation.quantum_simulator.qt.{solver_name}",
        wraps=getattr(qt, solver_name),
    ) as mocked_solver:
        solver(
            [control],
            initial_state=system.ground_state,
            options={"max_step": 0.01},
        )

    assert mocked_solver.call_args is not None
    assert mocked_solver.call_args.kwargs["options"]["max_step"] == pytest.approx(0.01)


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
def test_qutip_solver_scales_nsteps_with_solver_interval(solver_name: str) -> None:
    """Solver work limit should accommodate the derived maximum step."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(3, dtype=np.complex128),
        durations=np.array([20.0, 0.01, 20.0]),
    )
    solver = getattr(QuantumSimulator(system), solver_name)

    with patch(
        f"qxsimulator.simulation.quantum_simulator.qt.{solver_name}",
        wraps=getattr(qt, solver_name),
    ) as mocked_solver:
        solver([control], initial_state=system.ground_state)

    assert mocked_solver.call_args is not None
    forwarded_options = mocked_solver.call_args.kwargs["options"]
    assert forwarded_options["max_step"] == pytest.approx(0.005)
    required_steps = int(
        np.ceil(np.max(np.diff(control.times)) / forwarded_options["max_step"])
    )
    assert forwarded_options["nsteps"] == 2 * required_steps


@pytest.mark.parametrize("solver_name", ["mesolve", "sesolve"])
def test_qutip_solver_preserves_explicit_nsteps(solver_name: str) -> None:
    """An explicit work limit should override the derived default."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    solver = getattr(simulator, solver_name)

    with patch(
        f"qxsimulator.simulation.quantum_simulator.qt.{solver_name}",
        wraps=getattr(qt, solver_name),
    ) as mocked_solver:
        solver(
            [control],
            initial_state=system.ground_state,
            options={"nsteps": 12345},
        )

    assert mocked_solver.call_args is not None
    assert mocked_solver.call_args.kwargs["options"]["nsteps"] == 12345


def test_sesolve_matches_mesolve_for_closed_system() -> None:
    """Given a closed driven system, sesolve should match mesolve dynamics."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    initial_state = system.state({"Q0": "0"})

    mesolve_result = simulator.mesolve(
        [control],
        initial_state=initial_state,
    )
    sesolve_result = simulator.sesolve(
        [control],
        initial_state=initial_state,
    )

    assert mesolve_result.final_state.isoper
    assert sesolve_result.final_state.isket
    assert isinstance(sesolve_result.states, list)
    assert sesolve_result.propagators == []
    assert sesolve_result.model is not None
    assert_allclose(
        qt.ket2dm(sesolve_result.final_state).full(),
        mesolve_result.final_state.full(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_sesolve_can_compute_states_from_propagators() -> None:
    """Requested sesolve propagators should reproduce the direct state solution."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    initial_state = system.state({"Q0": "0"})

    direct_result = simulator.sesolve([control], initial_state=initial_state)
    propagator_result = simulator.sesolve(
        [control],
        initial_state=initial_state,
        compute_propagators=True,
    )

    assert len(propagator_result.propagators) == len(propagator_result.times)
    assert all(propagator.isoper for propagator in propagator_result.propagators)
    for direct_state, propagated_state in zip(
        direct_result.states,
        propagator_result.states,
        strict=True,
    ):
        assert_allclose(
            propagated_state.full(),
            direct_state.full(),
            rtol=1e-6,
            atol=1e-8,
        )


def test_mesolve_can_compute_states_from_superoperator_propagators() -> None:
    """Requested mesolve propagators should reproduce the direct state solution."""
    qubit = Transmon(
        label="Q0",
        dimension=2,
        frequency=5.0,
        relaxation_rate=0.01,
    )
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.full(10, 2 * np.pi * 0.01, dtype=np.complex128),
        durations=np.full(10, 0.1),
        frequency=qubit.frequency,
    )
    simulator = QuantumSimulator(system)
    initial_state = system.state({"Q0": "1"})

    direct_result = simulator.mesolve([control], initial_state=initial_state)
    propagator_result = simulator.mesolve(
        [control],
        initial_state=initial_state,
        compute_propagators=True,
    )

    assert isinstance(direct_result.states, list)
    assert isinstance(propagator_result.states, list)
    assert len(propagator_result.propagators) == len(propagator_result.times)
    assert all(propagator.issuper for propagator in propagator_result.propagators)
    for direct_state, propagated_state in zip(
        direct_result.states,
        propagator_result.states,
        strict=True,
    ):
        assert_allclose(
            propagated_state.full(),
            direct_state.full(),
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("compute_propagators", [False, True])
def test_sesolve_keeps_final_frame_shift_out_of_raw_trajectory(
    compute_propagators: bool,
) -> None:
    """Sesolve should retain raw states and propagators across a frame shift."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    frame_shift = np.pi / 2
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
        final_frame_shift=frame_shift,
    )
    initial_state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

    result = QuantumSimulator(system).sesolve(
        [control],
        initial_state=initial_state,
        compute_propagators=compute_propagators,
    )

    assert_allclose(result.final_state.full(), initial_state.full())
    if compute_propagators:
        assert_allclose(result.propagators[-1].full(), system.identity_matrix.full())
    else:
        assert result.propagators == []


@pytest.mark.parametrize("compute_propagators", [False, True])
def test_mesolve_keeps_final_frame_shift_out_of_raw_trajectory(
    compute_propagators: bool,
) -> None:
    """Mesolve should retain raw states and propagators across a frame shift."""
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    frame_shift = np.pi / 2
    control = Control(
        target=qubit,
        waveform=np.zeros(1, dtype=np.complex128),
        durations=np.array([0.1]),
        final_frame_shift=frame_shift,
    )
    initial_ket = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    initial_state = qt.ket2dm(initial_ket)

    result = QuantumSimulator(system).mesolve(
        [control],
        initial_state=initial_state,
        compute_propagators=compute_propagators,
    )

    assert_allclose(result.final_state.full(), initial_state.full())
    if compute_propagators:
        identity_superoperator = qt.to_super(system.identity_matrix)
        assert_allclose(result.propagators[-1].full(), identity_superoperator.full())
    else:
        assert result.propagators == []


def test_sesolve_population_display_supports_ket_states(capsys) -> None:
    """Given sesolve output, population display should handle ket states."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)

    result = simulator.sesolve(
        [control],
        initial_state=system.state({"Q0": "0"}),
    )

    result.show_last_population()

    captured = capsys.readouterr()
    assert "|0⟩:" in captured.out
    assert "|1⟩:" in captured.out


def test_sesolve_rejects_density_matrix_initial_state() -> None:
    """Given density matrix input, sesolve should require a pure ket state."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    initial_state = qt.ket2dm(system.ground_state)

    with pytest.raises(ValueError, match="requires a ket initial_state"):
        simulator.sesolve(
            [control],
            initial_state=initial_state,
        )


def test_sesolve_builds_model_without_collapse_operators() -> None:
    """Given a dissipative system, sesolve should build a closed-system model."""
    qubit = Transmon(
        label="Q0",
        dimension=2,
        frequency=5.0,
        relaxation_rate=0.01,
    )
    system = QuantumSystem(objects=[qubit])
    waveform = np.full(40, 2 * np.pi * 0.01, dtype=np.complex128)
    durations = np.full(40, 0.1)
    control = Control(
        target=qubit,
        waveform=waveform,
        durations=durations,
        frequency=qubit.frequency,
    )
    simulator = QuantumSimulator(system)

    result = simulator.sesolve(
        [control],
        initial_state=system.state({"Q0": "0"}),
    )

    assert result.final_state.isket
    assert result.model is not None
    assert result.model.collapse_operators == []


def test_create_simulation_parameters_uses_physical_dephasing_rate() -> None:
    """Pure dephasing should use a collapse coefficient of sqrt(2 gamma_phi)."""
    qubit = Transmon(
        label="Q0",
        dimension=2,
        frequency=5.0,
        dephasing_rate=0.01,
    )
    system = QuantumSystem(objects=[qubit])
    control = Control(
        target=qubit,
        waveform=np.zeros(2, dtype=np.complex128),
        durations=np.ones(2),
        frequency=qubit.frequency,
    )

    parameters = QuantumSimulator(system).create_simulation_parameters([control])

    [dephasing_operator] = parameters["collapse_operators"]
    [expected] = system.get_collapse_operators(qubit.label)
    assert_allclose(
        dephasing_operator.full(),
        expected.full(),
        rtol=1e-12,
        atol=0.0,
    )
