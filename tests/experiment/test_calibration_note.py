from pathlib import Path

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


def test_init():
    """CalibrationNote should be initialized with a chip_id."""
    chip_id = "CHIP_ID"
    note = CalibrationNote(chip_id=chip_id)
    assert note.chip_id == chip_id
    assert note.file_path == Path(f".calibration/{chip_id}.json")
    assert note.file_path.exists()
    assert note.rabi_params == {}
    assert note.hpi_params == {}
    assert note.pi_params == {}
    assert note.drag_hpi_params == {}
    assert note.drag_pi_params == {}
    assert note.state_params == {}
    assert note.cr_params == {}


def test_update_rabi_param():
    """CalibrationNote should update the rabi_param."""
    note = CalibrationNote(chip_id="CHIP_ID")
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.0,
            "frequency": 0.0,
            "phase": 0.0,
            "offset": 0.0,
            "noise": 0.0,
            "angle": 0.0,
        },
    )
    param = note.get_rabi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.0
    assert param["frequency"] == 0.0
    assert param["phase"] == 0.0
    assert param["offset"] == 0.0
    assert param["noise"] == 0.0
    assert param["angle"] == 0.0


def test_update_hpi_param():
    """CalibrationNote should update the hpi_param."""
    note = CalibrationNote(chip_id="CHIP_ID")
    note.update_hpi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.0,
            "duration": 0.0,
            "tau": 0.0,
        },
    )
    param = note.get_hpi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.0
    assert param["duration"] == 0.0
    assert param["tau"] == 0.0


def test_update_pi_param():
    """CalibrationNote should update the pi_param."""
    note = CalibrationNote(chip_id="CHIP_ID")
    note.update_pi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.0,
            "duration": 0.0,
            "tau": 0.0,
        },
    )
    param = note.get_pi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.0
    assert param["duration"] == 0.0
    assert param["tau"] == 0.0


def test_update_drag_hpi_param():
    """CalibrationNote should update the drag_hpi_param."""
    note = CalibrationNote(chip_id="CHIP_ID")
    note.update_drag_hpi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.0,
            "duration": 0.0,
            "beta": 0.0,
        },
    )
    param = note.get_drag_hpi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.0
    assert param["duration"] == 0.0
    assert param["beta"] == 0.0


def test_update_drag_pi_param():
    """CalibrationNote should update the drag_pi_param."""
    note = CalibrationNote(chip_id="CHIP_ID")
    note.update_drag_pi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.0,
            "duration": 0.0,
            "beta": 0.0,
        },
    )
    param = note.get_drag_pi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.0
    assert param["duration"] == 0.0
    assert param["beta"] == 0.0


def test_update_state_param():
    """CalibrationNote should update the state_param."""
    note = CalibrationNote(chip_id="CHIP_ID")
    note.update_state_param(
        "Q00",
        {
            "target": "Q00",
            "centers": {"0": [0.0, 0.0], "1": [1.0, 1.0]},
        },
    )
    param = note.get_state_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["centers"]["0"] == [0.0, 0.0]
    assert param["centers"]["1"] == [1.0, 1.0]


def test_update_cr_param():
    """CalibrationNote should update the cr_param."""
    note = CalibrationNote(chip_id="CHIP_ID")
    note.update_cr_param(
        "Q00",
        {
            "target": "Q00",
            "duration": 0.0,
            "ramptime": 0.0,
            "cr_amplitude": 0.0,
            "cr_phase": 0.0,
            "cancel_amplitude": 0.0,
            "cancel_phase": 0.0,
            "cr_cancel_ratio": 0.0,
        },
    )
    param = note.get_cr_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["duration"] == 0.0
    assert param["ramptime"] == 0.0
    assert param["cr_amplitude"] == 0.0
    assert param["cr_phase"] == 0.0
    assert param["cancel_amplitude"] == 0.0
    assert param["cancel_phase"] == 0.0
    assert param["cr_cancel_ratio"] == 0.0
