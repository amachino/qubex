import numpy as np
import pytest

import qubex as qx
from qubex.pulse import Pulse, PulseSchedule

dt = qx.pulse.get_sampling_period()


def test_empty_init():
    """PulseSchedule should be initialized without any parameters."""
    ps = PulseSchedule()
    assert ps.labels == []
    assert ps.values == {}
    assert ps.length == 0
    assert ps.duration == 0.0


def test_init():
    """PulseSchedule should be initialized with valid parameters."""
    ps = PulseSchedule(["Q00", "Q01"])
    assert ps.labels == ["Q00", "Q01"]
    assert ps.values["Q00"] == pytest.approx([])
    assert ps.values["Q01"] == pytest.approx([])
    assert ps.length == 0
    assert ps.duration == 0.0


def test_add():
    """PulseSchedule should add a pulse to the sequence."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 0, 1j]))
        ps.add("Q01", Pulse([1j, 0, 1]))
    assert ps.labels == ["Q00", "Q01"]
    assert ps.values["Q00"] == pytest.approx([1, 0, 1j])
    assert ps.values["Q01"] == pytest.approx([1j, 0, 1])
    assert ps.length == 3
    assert ps.duration == 3 * dt


def test_label_order():
    """PulseSchedule should maintain the order of labels."""
    with PulseSchedule(["Q01", "Q00"]) as ps:
        ps.add("Q01", Pulse([1]))
        ps.add("Q00", Pulse([1]))
    assert ps.labels == ["Q01", "Q00"]

    with PulseSchedule() as ps:
        ps.add("Q01", Pulse([1]))
        ps.add("Q00", Pulse([1]))
    assert ps.labels == ["Q01", "Q00"]


def test_barrier():
    """PulseSchedule should add a barrier to the sequence."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 0, 1j]))
        ps.barrier()
        ps.add("Q01", Pulse([1j, 0, 1]))
    assert ps.values["Q00"] == pytest.approx([1, 0, 1j, 0, 0, 0])
    assert ps.values["Q01"] == pytest.approx([0, 0, 0, 1j, 0, 1])


def test_specific_barrier():
    """PulseSchedule should add a barrier to specific qubits."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1]))
        ps.barrier(["Q00", "Q01"])
        ps.add("Q01", Pulse([1, 1, 1]))
        ps.add("Q02", Pulse([1, 1, 1]))

    assert ps.values["Q00"] == pytest.approx([1, 0, 0, 0])
    assert ps.values["Q01"] == pytest.approx([0, 1, 1, 1])
    assert ps.values["Q02"] == pytest.approx([1, 1, 1, 0])


def test_call():
    """PulseSchedule should call another PulseSchedule."""
    with PulseSchedule() as ps1:
        ps1.add("Q00", Pulse([1, 1]))

    with PulseSchedule() as ps2:
        ps2.call(ps1)
        ps2.add("Q01", Pulse([1, 1]))

    assert ps2.values["Q00"] == pytest.approx([1, 1])
    assert ps2.values["Q01"] == pytest.approx([1, 1])


def test_copy():
    """PulseSchedule should be copied."""
    with PulseSchedule() as ps1:
        ps1.add("Q00", Pulse([1, 1]))

    ps2 = ps1.copy()
    assert isinstance(ps2, PulseSchedule)
    assert ps1 != ps2
    assert ps2.values["Q00"] == pytest.approx([1, 1])


def test_scaled():
    """PulseSchedule should be scaled by a given parameter."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 2, 3]))
        ps.add("Q01", Pulse([1j, 2j, 3j]))

    scaled = ps.scaled(0.5)
    assert scaled != ps
    assert scaled.values["Q00"] == pytest.approx([0.5, 1, 1.5])
    assert scaled.values["Q01"] == pytest.approx([0.5j, 1j, 1.5j])


def test_detuned():
    """PulseSchedule should be detuned by a given parameter."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 2, 3]))
        ps.add("Q01", Pulse([1j, 2j, 3j]))

    detuned = ps.detuned(0.001)
    assert detuned != ps
    assert detuned.values["Q00"] == pytest.approx(
        [
            1,
            2 * np.exp(1j * 0.001 * 2 * np.pi * dt),
            3 * np.exp(2j * 0.001 * 2 * np.pi * dt),
        ]
    )
    assert detuned.values["Q01"] == pytest.approx(
        [
            1j,
            2j * np.exp(1j * 0.001 * 2 * np.pi * dt),
            3j * np.exp(2j * 0.001 * 2 * np.pi * dt),
        ]
    )


def test_shifted():
    """PulseSchedule should be shifted by a given parameter."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 2, 3]))
        ps.add("Q01", Pulse([1j, 2j, 3j]))

    shifted = ps.shifted(np.pi / 2)
    assert shifted != ps
    assert shifted.values["Q00"] == pytest.approx([1j, 2j, 3j])
    assert shifted.values["Q01"] == pytest.approx([-1, -2, -3])


def test_repeated():
    """PulseSchedule should be repeated a given number of times."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 2, 3]))
        ps.add("Q01", Pulse([1j, 2j, 3j]))

    repeated = ps.repeated(2)
    assert repeated != ps
    assert repeated.values["Q00"] == pytest.approx([1, 2, 3, 1, 2, 3])
    assert repeated.values["Q01"] == pytest.approx([1j, 2j, 3j, 1j, 2j, 3j])

    repeated = ps.repeated(0)
    assert repeated.values["Q00"] == pytest.approx([])
    assert repeated.values["Q01"] == pytest.approx([])


def test_inverted():
    """PulseSchedule should be inverted."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 2, 3]))
        ps.add("Q01", Pulse([1j, 2j, 3j]))

    inverted = ps.inverted()
    assert inverted != ps
    assert inverted.values["Q00"] == pytest.approx([-3, -2, -1])
    assert inverted.values["Q01"] == pytest.approx([-3j, -2j, -1j])


def test_get_sequences():
    """PulseSchedule should return the sequences."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 0, 1j]))
        ps.add("Q01", Pulse([1j, 0, 1]))
    seq = ps.get_sequences()
    assert seq["Q00"] != ps._channels["Q00"].sequence
    assert seq["Q01"] != ps._channels["Q01"].sequence
    assert seq["Q00"].values == pytest.approx([1, 0, 1j])
    assert seq["Q01"].values == pytest.approx([1j, 0, 1])


def test_get_sampled_sequences():
    """PulseSchedule should return the sampled sequences."""
    with PulseSchedule() as ps:
        ps.add("Q00", Pulse([1, 0, 1j]))
        ps.add("Q01", Pulse([1j, 0, 1]))
    seq_start = ps.get_sampled_sequences(duration=5 * dt, align="start")
    assert seq_start["Q00"] == pytest.approx([1, 0, 1j, 0, 0])
    assert seq_start["Q01"] == pytest.approx([1j, 0, 1, 0, 0])
    seq_end = ps.get_sampled_sequences(duration=10, align="end")
    assert seq_end["Q00"] == pytest.approx([0, 0, 1, 0, 1j])
    assert seq_end["Q01"] == pytest.approx([0, 0, 1j, 0, 1])


def test_get_pulse_ranges():
    """PulseSchedule should return the pulse ranges."""
    with PulseSchedule() as ps:
        ps.add("Q01", Pulse([1, 1, 1]))
        ps.barrier()
        ps.add("Q02", Pulse([1, 1, 1]))
        ps.barrier()
        ps.add("RQ01", Pulse([1, 1, 1]))
        ps.add("RQ02", Pulse([1, 1, 1]))
        ps.barrier()
        ps.add("Q01", Pulse([1, 1, 1]))
        ps.barrier()
        ps.add("RQ01", Pulse([1, 1, 1]))
        ps.add("RQ02", Pulse([1, 1, 1]))

    ranges_all = ps.get_pulse_ranges()
    assert ranges_all["Q01"] == pytest.approx([range(0, 3), range(9, 12)])
    assert ranges_all["RQ01"] == pytest.approx([range(6, 9), range(12, 15)])
    assert ranges_all["Q02"] == pytest.approx([range(3, 6)])
    assert ranges_all["RQ02"] == pytest.approx([range(6, 9), range(12, 15)])

    ranges_read = ps.get_pulse_ranges(["RQ01", "RQ02"])
    assert ranges_read["RQ01"] == pytest.approx([range(6, 9), range(12, 15)])
    assert ranges_read["RQ02"] == pytest.approx([range(6, 9), range(12, 15)])


def test_usecase():
    """PulseSchedule should be working in a typical usecase."""
    x90 = qx.Pulse([1, 1]).scaled(0.5)
    x180 = x90.scaled(2)
    y90 = x90.shifted(np.pi / 2)
    z90 = qx.VirtualZ(np.pi / 2)
    z180 = qx.VirtualZ(np.pi)
    h = qx.PulseArray([z180, y90])

    with qx.PulseSchedule() as ps1:
        ps1.add("Q00", h)
        ps1.add("Q00", z90)
        ps1.add("Q00", x180)
        ps1.add("Q00", z90)
        ps1.add("Q00", h)

    seq1 = ps1.get_sampled_sequence("Q00")
    phase1 = ps1.get_final_frame_shift("Q00")
    assert seq1 == pytest.approx(
        [
            -0.5j,
            -0.5j,
            1j,
            1j,
            -0.5j,
            -0.5j,
        ]
    )
    assert np.abs(phase1) == pytest.approx(np.pi)

    with qx.PulseSchedule(["Q00"]) as ps2:
        ps2.call(ps1)
        ps2.call(ps1.inverted())

    seq2 = ps2.get_sampled_sequence("Q00")
    phase2 = ps2.get_final_frame_shift("Q00")
    assert seq2 == pytest.approx(
        [
            -0.5j,
            -0.5j,
            1j,
            1j,
            -0.5j,
            -0.5j,
            0.5j,
            0.5j,
            -1j,
            -1j,
            0.5j,
            0.5j,
        ]
    )
    assert phase2 == pytest.approx(0)
