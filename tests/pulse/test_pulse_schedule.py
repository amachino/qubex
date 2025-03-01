import numpy as np
import pytest

import qubex as qx
from qubex.pulse import Pulse, PulseSchedule

dt = qx.pulse.get_sampling_period()


def test_empty_init():
    """PulseSchedule should raise a TypeError if no parameters are provided."""
    with pytest.raises(TypeError):
        PulseSchedule()  # type: ignore


def test_empty_list():
    """PulseSchedule should be initialized with an empty list."""
    sched = PulseSchedule([])
    assert sched.targets == {}


def test_init():
    """PulseSchedule should be initialized with valid parameters."""
    sched = PulseSchedule(["Q00", "Q01"])
    assert sched.targets == {
        "Q00": {
            "frequency": None,
            "object": None,
        },
        "Q01": {
            "frequency": None,
            "object": None,
        },
    }


def test_add():
    """PulseSchedule should add a pulse to the sequence."""
    with PulseSchedule(["Q00", "Q01"]) as sched:
        sched.add("Q00", Pulse([1, 0, 1j]))
        sched.add("Q01", Pulse([1j, 0, 1]))
    seqs = sched.get_sampled_sequences()
    assert seqs["Q00"] == pytest.approx([1, 0, 1j])
    assert seqs["Q01"] == pytest.approx([1j, 0, 1])


def test_barrier():
    """PulseSchedule should add a barrier to the sequence."""
    with PulseSchedule(["Q00", "Q01"]) as sched:
        sched.add("Q00", Pulse([1, 0, 1j]))
        sched.barrier()
        sched.add("Q01", Pulse([1j, 0, 1]))
    seqs = sched.get_sampled_sequences()
    assert seqs["Q00"] == pytest.approx([1, 0, 1j, 0, 0, 0])
    assert seqs["Q01"] == pytest.approx([0, 0, 0, 1j, 0, 1])


def test_specific_barrier():
    """PulseSchedule should add a barrier to specific qubits."""
    with PulseSchedule(["Q00", "Q01", "Q02", "Q03"]) as sched:
        sched.add("Q00", Pulse([1]))
        sched.barrier(["Q00", "Q01"])
        sched.add("Q01", Pulse([1, 1, 1]))
        sched.add("Q02", Pulse([1, 1, 1]))
    seqs = sched.get_sampled_sequences()
    assert seqs["Q00"] == pytest.approx([1, 0, 0, 0])
    assert seqs["Q01"] == pytest.approx([0, 1, 1, 1])
    assert seqs["Q02"] == pytest.approx([1, 1, 1, 0])


def test_get_sampled_sequences():
    """PulseSchedule should return the sampled sequences."""
    with PulseSchedule(["Q00", "Q01"]) as sched:
        sched.add("Q00", Pulse([1, 0, 1j]))
        sched.add("Q01", Pulse([1j, 0, 1]))
    seq_start = sched.get_sampled_sequences(duration=5 * dt, align="start")
    assert seq_start["Q00"] == pytest.approx([1, 0, 1j, 0, 0])
    assert seq_start["Q01"] == pytest.approx([1j, 0, 1, 0, 0])
    seq_end = sched.get_sampled_sequences(duration=10, align="end")
    assert seq_end["Q00"] == pytest.approx([0, 0, 1, 0, 1j])
    assert seq_end["Q01"] == pytest.approx([0, 0, 1j, 0, 1])


def test_get_pulse_ranges():
    """PulseSchedule should return the pulse ranges."""
    with PulseSchedule(["Q01", "RQ01", "Q02", "RQ02"]) as sched:
        sched.add("Q01", Pulse([1, 1, 1]))
        sched.barrier()
        sched.add("Q02", Pulse([1, 1, 1]))
        sched.barrier()
        sched.add("RQ01", Pulse([1, 1, 1]))
        sched.add("RQ02", Pulse([1, 1, 1]))
        sched.barrier()
        sched.add("Q01", Pulse([1, 1, 1]))
        sched.barrier()
        sched.add("RQ01", Pulse([1, 1, 1]))
        sched.add("RQ02", Pulse([1, 1, 1]))

    ranges_all = sched.get_pulse_ranges()
    assert ranges_all["Q01"] == pytest.approx([range(0, 3), range(9, 12)])
    assert ranges_all["RQ01"] == pytest.approx([range(6, 9), range(12, 15)])
    assert ranges_all["Q02"] == pytest.approx([range(3, 6)])
    assert ranges_all["RQ02"] == pytest.approx([range(6, 9), range(12, 15)])

    ranges_read = sched.get_pulse_ranges(["RQ01", "RQ02"])
    assert ranges_read["RQ01"] == pytest.approx([range(6, 9), range(12, 15)])
    assert ranges_read["RQ02"] == pytest.approx([range(6, 9), range(12, 15)])


def test_usecase():
    """PulseSchedule should be working in a typical usecase."""
    x90 = qx.Pulse([1, 1])
    x180 = x90.scaled(2)
    y90 = x90.shifted(np.pi / 2)
    z90 = qx.VirtualZ(np.pi / 2)
    z180 = qx.VirtualZ(np.pi)
    h = qx.PulseArray([z180, y90])

    with qx.PulseSchedule(["Q00"]) as sched1:
        sched1.add("Q00", h)
        sched1.add("Q00", z90)
        sched1.add("Q00", x180)
        sched1.add("Q00", z90)
        sched1.add("Q00", h)

    sequence1 = sched1.get_sequences()["Q00"]
    waveform1 = sched1.get_sampled_sequences()["Q00"]
    assert waveform1 == pytest.approx(sequence1.values)
    assert waveform1 == pytest.approx(
        [
            -1j,
            -1j,
            2j,
            2j,
            -1j,
            -1j,
        ]
    )
    final_phase = (sequence1.final_frame_shift + np.pi) % (np.pi * 2) - np.pi
    assert np.abs(final_phase) == pytest.approx(np.pi)

    with qx.PulseSchedule(["Q00"]) as sched2:
        sched2.call(sched1)
        sched2.call(sched1.reversed())

    sequence2 = sched2.get_sequences()["Q00"]
    waveform2 = sched2.get_sampled_sequences()["Q00"]
    assert waveform2 == pytest.approx(sequence2.values)
    assert waveform2 == pytest.approx(
        [
            -1j,
            -1j,
            2j,
            2j,
            -1j,
            -1j,
            1j,
            1j,
            -2j,
            -2j,
            1j,
            1j,
        ]
    )
    assert sequence2.final_frame_shift == pytest.approx(0)
