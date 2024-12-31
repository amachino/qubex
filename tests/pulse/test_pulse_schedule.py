import pytest

from qubex.pulse import Pulse, PulseSchedule


def test_empty_init():
    """PulseSchedule should raise a TypeError if no parameters are provided."""
    with pytest.raises(TypeError):
        PulseSchedule()  # type: ignore


def test_empty_list():
    """PulseSchedule should be initialized with an empty list."""
    ps = PulseSchedule([])
    assert ps.targets == {}


def test_init():
    """PulseSchedule should be initialized with valid parameters."""
    ps = PulseSchedule(["Q00", "Q01"])
    assert ps.targets == {
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
    with PulseSchedule(["Q00", "Q01"]) as ps:
        ps.add("Q00", Pulse([1, 0, 1j]))
        ps.add("Q01", Pulse([1j, 0, 1]))
    seq = ps.get_sampled_sequences()
    assert seq["Q00"] == pytest.approx([1, 0, 1j])
    assert seq["Q01"] == pytest.approx([1j, 0, 1])


def test_barrier():
    """PulseSchedule should add a barrier to the sequence."""
    with PulseSchedule(["Q00", "Q01"]) as ps:
        ps.add("Q00", Pulse([1, 0, 1j]))
        ps.barrier()
        ps.add("Q01", Pulse([1j, 0, 1]))
    seq = ps.get_sampled_sequences()
    assert seq["Q00"] == pytest.approx([1, 0, 1j, 0, 0, 0])
    assert seq["Q01"] == pytest.approx([0, 0, 0, 1j, 0, 1])


def test_specific_barrier():
    """PulseSchedule should add a barrier to specific qubits."""
    with PulseSchedule(["Q00", "Q01", "Q02", "Q03"]) as ps:
        ps.add("Q00", Pulse([1]))
        ps.barrier(["Q00", "Q01"])
        ps.add("Q01", Pulse([1, 1, 1]))
        ps.add("Q02", Pulse([1, 1, 1]))
    seq = ps.get_sampled_sequences()
    assert seq["Q00"] == pytest.approx([1, 0, 0, 0])
    assert seq["Q01"] == pytest.approx([0, 1, 1, 1])
    assert seq["Q02"] == pytest.approx([1, 1, 1, 0])


def test_get_sampled_sequences():
    """PulseSchedule should return the sampled sequences."""
    with PulseSchedule(["Q00", "Q01"]) as ps:
        ps.add("Q00", Pulse([1, 0, 1j]))
        ps.add("Q01", Pulse([1j, 0, 1]))
    dt = Pulse.SAMPLING_PERIOD
    seq_start = ps.get_sampled_sequences(duration=5 * dt, align="start")
    assert seq_start["Q00"] == pytest.approx([1, 0, 1j, 0, 0])
    assert seq_start["Q01"] == pytest.approx([1j, 0, 1, 0, 0])
    seq_end = ps.get_sampled_sequences(duration=10, align="end")
    assert seq_end["Q00"] == pytest.approx([0, 0, 1, 0, 1j])
    assert seq_end["Q01"] == pytest.approx([0, 0, 1j, 0, 1])


def test_get_pulse_ranges():
    """PulseSchedule should return the pulse ranges."""
    with PulseSchedule(["Q01", "RQ01", "Q02", "RQ02"]) as ps:
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
