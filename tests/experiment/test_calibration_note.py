import time
from datetime import datetime, timedelta

import pytest

from qubex.experiment.calibration_note import CalibrationNote
from qubex.experiment.experiment_note import ExperimentNote


def test_inheritance():
    """CalibrationNote should inherit from ExperimentNote."""
    assert issubclass(CalibrationNote, ExperimentNote)


def test_empty_init():
    """CalibrationNote should raise a TypeError if no chip_id is provided."""
    with pytest.raises(TypeError):
        CalibrationNote()  # type: ignore


def test_init(tmp_path):
    """CalibrationNote should be initialized with a chip_id."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    assert note.chip_id == chip_id
    assert note.file_path == calibration_dir / f"{chip_id}.json"
    assert note.file_path.exists()
    assert note.rabi_params == {}
    assert note.hpi_params == {}
    assert note.pi_params == {}
    assert note.drag_hpi_params == {}
    assert note.drag_pi_params == {}
    assert note.state_params == {}
    assert note.cr_params == {}


def test_update_rabi_param(tmp_path):
    """CalibrationNote should update the rabi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "frequency": 0.5,
            "phase": 0.5,
            "offset": 0.5,
            "noise": 0.5,
            "angle": 0.5,
        },
    )
    param = note.get_rabi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["frequency"] == 0.5
    assert param["phase"] == 0.5
    assert param["offset"] == 0.5
    assert param["noise"] == 0.5
    assert param["angle"] == 0.5


def test_update_hpi_param(tmp_path):
    """CalibrationNote should update the hpi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_hpi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "tau": 0.5,
        },
    )
    param = note.get_hpi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["duration"] == 0.5
    assert param["tau"] == 0.5


def test_update_pi_param(tmp_path):
    """CalibrationNote should update the pi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_pi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "tau": 0.5,
        },
    )
    param = note.get_pi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["duration"] == 0.5
    assert param["tau"] == 0.5


def test_update_drag_hpi_param(tmp_path):
    """CalibrationNote should update the drag_hpi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_drag_hpi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "beta": 0.5,
        },
    )
    param = note.get_drag_hpi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["duration"] == 0.5
    assert param["beta"] == 0.5


def test_update_drag_pi_param(tmp_path):
    """CalibrationNote should update the drag_pi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_drag_pi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "beta": 0.5,
        },
    )
    param = note.get_drag_pi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["duration"] == 0.5
    assert param["beta"] == 0.5


def test_update_state_param(tmp_path):
    """CalibrationNote should update the state_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_state_param(
        "Q00",
        {
            "target": "Q00",
            "centers": {"0": [0.5, 0.5], "1": [0.5, 0.5]},
        },
    )
    param = note.get_state_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["centers"]["0"] == [0.5, 0.5]
    assert param["centers"]["1"] == [0.5, 0.5]


def test_update_cr_param(tmp_path):
    """CalibrationNote should update the cr_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_cr_param(
        "Q00",
        {
            "target": "Q00",
            "duration": 0.5,
            "ramptime": 0.5,
            "cr_amplitude": 0.5,
            "cr_phase": 0.5,
            "cancel_amplitude": 0.5,
            "cancel_phase": 0.5,
            "cr_cancel_ratio": 0.5,
        },
    )
    param = note.get_cr_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["duration"] == 0.5
    assert param["ramptime"] == 0.5
    assert param["cr_amplitude"] == 0.5
    assert param["cr_phase"] == 0.5
    assert param["cancel_amplitude"] == 0.5
    assert param["cancel_phase"] == 0.5
    assert param["cr_cancel_ratio"] == 0.5


def test_timestamp(tmp_path):
    """CalibrationNote should return the timestamp of the last update."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "frequency": 0.5,
            "phase": 0.5,
            "offset": 0.5,
            "noise": 0.5,
            "angle": 0.5,
        },
    )
    param = note.get_rabi_param("Q00") or {}
    timestamp = param.get("timestamp")
    assert timestamp is not None
    time.sleep(1)
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 1.0,
            "frequency": 1.0,
            "phase": 1.0,
            "offset": 1.0,
            "noise": 1.0,
            "angle": 1.0,
        },
    )
    updated_param = note.get_rabi_param("Q00") or {}
    updated_timestamp = updated_param.get("timestamp")
    assert updated_timestamp is not None
    assert updated_timestamp > timestamp


def test_get_param_with_cutoff_days(tmp_path):
    """CalibrationNote should return None if the param is older than the cutoff_days."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "frequency": 0.5,
            "phase": 0.5,
            "offset": 0.5,
            "noise": 0.5,
            "angle": 0.5,
            "timestamp": datetime.strftime(
                datetime.now() - timedelta(days=2), "%Y-%m-%d %H:%M:%S"
            ),
        },
    )
    assert note.get_rabi_param("Q00", cutoff_days=1) is None
    assert note.get_rabi_param("Q00", cutoff_days=3) is not None
