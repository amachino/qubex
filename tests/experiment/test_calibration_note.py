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
