from pathlib import Path

import numpy as np
import pytest

from qubex.experiment.experiment_note import ExperimentNote


def test_empty_init():
    """ExperimentNote should be initialized with a default file path."""
    note = ExperimentNote()
    assert note.file_path == Path(".experiment_note.json")


def test_init(tmp_path):
    """ExperimentNote should initialize with an empty dictionary."""
    file_path = tmp_path / "note.json"
    note = ExperimentNote(file_path=file_path)
    assert note.file_path == file_path
    assert file_path.exists()


def test_put_invalid_values(tmp_path):
    """ExperimentNote should raise a ValueError if the value is not JSON serializable."""
    note = ExperimentNote(file_path=tmp_path / "note.json")
    # complex
    with pytest.raises(ValueError):
        note.put("foo", 1 + 1j)
    # ndarray
    with pytest.raises(ValueError):
        note.put("foo", np.array([1, 2, 3]))


def test_put_valid_values(tmp_path):
    """ExperimentNote should allow JSON serializable values."""
    note = ExperimentNote(file_path=tmp_path / "note.json")
    # empty
    assert note.get("foo") is None
    # int
    note.put("foo", 1)
    assert note.get("foo") == 1
    # float
    note.put("foo", 1.5)
    assert note.get("foo") == 1.5
    # str
    note.put("foo", "bar")
    assert note.get("foo") == "bar"
    # None
    note.put("foo", None)
    assert note.get("foo") is None
    # list
    note.put("foo", [1, 2, 3])
    assert note.get("foo") == [1, 2, 3]
    # dict
    note.put("foo", {"bar": "baz", "qux": "quux"})
    assert note.get("foo") == {"bar": "baz", "qux": "quux"}


def test_update_existing_dict(tmp_path):
    """ExperimentNote should update (not overwrite) existing dictionaries."""
    note = ExperimentNote(file_path=tmp_path / "note.json")
    note.put("foo", {"bar": "baz", "qux": "quux"})
    note.put("foo", {"qux": "quuux"})
    assert note.get("foo") == {"bar": "baz", "qux": "quuux"}


def test_remove_key(tmp_path):
    """ExperimentNote should remove a key from the dictionary."""
    note = ExperimentNote(file_path=tmp_path / "note.json")
    note.put("foo", "bar")
    note.remove("foo")
    assert note.get("foo") is None


def test_clear(tmp_path):
    """ExperimentNote should clear the dictionary."""
    note = ExperimentNote(file_path=tmp_path / "note.json")
    note.put("foo", "bar")
    note.put("baz", "qux")
    note.clear()
    assert note.get("foo") is None
    assert note.get("baz") is None


def test_save(tmp_path):
    """ExperimentNote should save the dictionary to a file."""
    note = ExperimentNote(file_path=tmp_path / "note.json")
    note.put("foo", "bar")
    note.save()
    note.save(tmp_path / "note2.json")

    saved_note = ExperimentNote(file_path=tmp_path / "note.json")
    assert saved_note.get("foo") == "bar"

    saved_note2 = ExperimentNote(file_path=tmp_path / "note2.json")
    assert saved_note2.get("foo") == "bar"


def test_load(tmp_path):
    """ExperimentNote should load the dictionary from a file."""
    note = ExperimentNote(file_path=tmp_path / "note.json")
    note.put("foo", "bar")
    note.save(tmp_path / "note2.json")

    note2 = ExperimentNote(file_path=tmp_path / "note.json")
    assert note2.get("foo") is None
    note2.load(tmp_path / "note2.json")
    assert note2.get("foo") == "bar"


def test_delete_key(tmp_path):
    """ExperimentNote should delete a key from the dictionary."""
    file_path = tmp_path / "note.json"
    note = ExperimentNote(file_path=file_path)
    note.put("foo", "bar")
    note.save()
    assert file_path.exists()
    note.delete()
    assert not file_path.exists()
