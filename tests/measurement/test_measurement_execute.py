# ruff: noqa: SLF001

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from qubex.measurement import MeasureMode, MultipleMeasureResult
from qubex.measurement.measurement import Measurement
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.pulse import Pulse, PulseSchedule


def test_execute_uses_instrument_executor(monkeypatch) -> None:
    measurement = Measurement.__new__(Measurement)
    measurement._classifiers = {}

    capture_schedule = CaptureSchedule(captures=[])
    measurement._add_readout_pulses = MagicMock()
    measurement._add_pump_pulses = MagicMock()
    measurement._create_capture_schedule = MagicMock(return_value=capture_schedule)

    executor = MagicMock()
    executor.pad_schedule_for_capture = MagicMock()
    expected_result = MultipleMeasureResult(
        mode=MeasureMode.AVG,
        data={},
        config={},
    )
    executor.execute = MagicMock(return_value=expected_result)
    measurement._instrument_executor = executor

    monkeypatch.setattr(
        Measurement,
        "targets",
        property(lambda self: {"RQ00": SimpleNamespace(is_read=True)}),
    )

    schedule = PulseSchedule(["RQ00"])
    result = Measurement.execute(
        measurement,
        schedule=schedule,
        shots=8,
        interval=16,
        add_last_measurement=True,
        add_pump_pulses=True,
    )

    assert result is expected_result
    measurement._add_readout_pulses.assert_called_once()
    measurement._add_pump_pulses.assert_called_once()
    measurement._create_capture_schedule.assert_called_once()
    executor.pad_schedule_for_capture.assert_called_once_with(schedule)
    executor.execute.assert_called_once()


def test_execute_with_concrete_schedule(monkeypatch) -> None:
    measurement = Measurement.__new__(Measurement)
    measurement._classifiers = {}

    monkeypatch.setattr(
        Measurement,
        "targets",
        property(
            lambda self: {
                "Q00": SimpleNamespace(is_read=False, is_pump=False),
                "RQ00": SimpleNamespace(is_read=True, is_pump=False),
            }
        ),
    )
    monkeypatch.setattr(
        Measurement,
        "mux_dict",
        property(lambda self: {"Q00": SimpleNamespace(index=0)}),
    )
    monkeypatch.setattr(
        Measurement,
        "control_params",
        property(lambda self: SimpleNamespace(capture_delay_word={0: 0})),
    )

    schedule = PulseSchedule(["Q00", "RQ00"])
    schedule.add("Q00", Pulse([1, 1, 1, 1]))
    schedule.add("RQ00", Pulse([0.2, 0.2, 0.2, 0.2]))

    executor = MagicMock()
    executor.pad_schedule_for_capture = MagicMock()
    expected_result = MultipleMeasureResult(
        mode=MeasureMode.AVG,
        data={},
        config={},
    )
    executor.execute = MagicMock(return_value=expected_result)
    measurement._instrument_executor = executor

    result = Measurement.execute(
        measurement,
        schedule=schedule,
        shots=4,
        interval=16,
        add_last_measurement=False,
        add_pump_pulses=False,
    )

    assert result is expected_result
    executor.execute.assert_called_once()
    kwargs = executor.execute.call_args.kwargs
    assert kwargs["schedule"] is schedule
    assert kwargs["measure_mode"] == MeasureMode.AVG
    capture_schedule = kwargs["capture_schedule"]
    assert isinstance(capture_schedule, CaptureSchedule)
    assert len(capture_schedule.captures) == 2
