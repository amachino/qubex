import json

import numpy as np
import pytest

from qubex.experiment.experiment_note import ExperimentNote


def test_init(tmp_path):
    """ExperimentNote should initialize with an empty dictionary."""
    file_path = tmp_path / "note.json"
    note = ExperimentNote(file_path=file_path)
    assert note.file_path == file_path


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


def test_change_tracking_after_load(tmp_path):
    """ExperimentNote should track only changes made after load."""
    file_path = tmp_path / "note.json"

    # Create initial data
    note1 = ExperimentNote(file_path=file_path)
    note1.put("initial_key", "initial_value")
    note1.put("shared_key", "original_value")
    note1.save()

    # Load and make changes
    note2 = ExperimentNote(file_path=file_path)
    note2.put("new_key", "new_value")
    note2.put("shared_key", "updated_value")

    # Check that only modified keys are tracked
    assert "new_key" in note2._changed_keys
    assert "shared_key" in note2._changed_keys
    assert "initial_key" not in note2._changed_keys


def test_concurrent_modification_simple(tmp_path):
    """ExperimentNote should preserve changes made by other processes during save."""
    file_path = tmp_path / "note.json"

    # Process 1: Create initial data
    note1 = ExperimentNote(file_path=file_path)
    note1.put("key1", "value1")
    note1.put("shared_key", "original")
    note1.save()

    # Process 2: Load the same file
    note2 = ExperimentNote(file_path=file_path)
    note2.put("key2", "value2")  # Add new key
    note2.put("shared_key", "modified_by_process2")  # Modify existing key

    # Simulate another process modifying the file directly
    with open(file_path, "r") as f:
        data = json.load(f)
    data["key3"] = "value3"  # Added by another process
    data["shared_key"] = "modified_by_other_process"  # Modified by another process
    with open(file_path, "w") as f:
        json.dump(data, f)

    # Process 2 saves its changes
    note2.save()

    # Verify the final state
    final_note = ExperimentNote(file_path=file_path)
    assert final_note.get("key1") == "value1"  # Original from process 1
    assert final_note.get("key2") == "value2"  # Added by process 2
    assert final_note.get("key3") == "value3"  # Added by other process (preserved)
    assert (
        final_note.get("shared_key") == "modified_by_process2"
    )  # Process 2's change wins


def test_concurrent_modification_nested_dict(tmp_path):
    """ExperimentNote should handle concurrent modifications of nested dictionaries."""
    file_path = tmp_path / "note.json"

    # Initial state
    note1 = ExperimentNote(file_path=file_path)
    note1.put("config", {"param1": "value1", "param2": "value2"})
    note1.save()

    # Process 2 loads and modifies
    note2 = ExperimentNote(file_path=file_path)
    note2.put("config", {"param2": "modified", "param3": "new"})

    # Another process modifies the file
    with open(file_path, "r") as f:
        data = json.load(f)
    data["config"]["param4"] = "external"
    data["other_key"] = "external_value"
    with open(file_path, "w") as f:
        json.dump(data, f)

    # Process 2 saves
    note2.save()

    # Verify nested merge
    final_note = ExperimentNote(file_path=file_path)
    config = final_note.get("config")
    assert config["param1"] == "value1"  # Original
    assert config["param2"] == "modified"  # Modified by process 2
    assert config["param3"] == "new"  # Added by process 2
    assert config["param4"] == "external"  # Added by external process (preserved)
    assert (
        final_note.get("other_key") == "external_value"
    )  # External addition preserved


def test_remove_key_concurrent(tmp_path):
    """ExperimentNote should handle key removal with concurrent modifications."""
    file_path = tmp_path / "note.json"

    # Initial state
    note1 = ExperimentNote(file_path=file_path)
    note1.put("key1", "value1")
    note1.put("key2", "value2")
    note1.put("key3", "value3")
    note1.save()

    # Process 2 loads and removes a key
    note2 = ExperimentNote(file_path=file_path)
    note2.remove("key2")

    # Another process adds a new key
    with open(file_path, "r") as f:
        data = json.load(f)
    data["key4"] = "external_value"
    with open(file_path, "w") as f:
        json.dump(data, f)

    # Process 2 saves
    note2.save()

    # Verify removal and preservation
    final_note = ExperimentNote(file_path=file_path)
    assert final_note.get("key1") == "value1"  # Preserved
    assert final_note.get("key2") is None  # Removed by process 2
    assert final_note.get("key3") == "value3"  # Preserved
    assert final_note.get("key4") == "external_value"  # Added by external process


def test_clear_concurrent(tmp_path):
    """ExperimentNote should handle clear operation with concurrent modifications."""
    file_path = tmp_path / "note.json"

    # Initial state
    note1 = ExperimentNote(file_path=file_path)
    note1.put("key1", "value1")
    note1.put("key2", "value2")
    note1.save()

    # Process 2 loads and clears
    note2 = ExperimentNote(file_path=file_path)
    note2.clear()
    note2.put("new_key", "new_value")

    # Another process adds a key
    with open(file_path, "r") as f:
        data = json.load(f)
    data["external_key"] = "external_value"
    with open(file_path, "w") as f:
        json.dump(data, f)

    # Process 2 saves
    note2.save()

    # Verify that original keys are removed but external key is preserved
    final_note = ExperimentNote(file_path=file_path)
    assert final_note.get("key1") is None  # Removed by clear
    assert final_note.get("key2") is None  # Removed by clear
    assert final_note.get("new_key") == "new_value"  # Added by process 2
    assert (
        final_note.get("external_key") == "external_value"
    )  # External addition preserved


def test_save_does_not_clear_changed_keys(tmp_path):
    """ExperimentNote should not clear changed keys tracking after save for flexibility."""
    file_path = tmp_path / "note.json"

    note = ExperimentNote(file_path=file_path)
    note.put("key1", "value1")
    note.put("key2", "value2")

    # Before save, changed keys should be tracked
    assert len(note._changed_keys) == 2
    assert "key1" in note._changed_keys
    assert "key2" in note._changed_keys

    # After save, changed keys should still be tracked
    note.save()
    assert len(note._changed_keys) == 2
    assert "key1" in note._changed_keys
    assert "key2" in note._changed_keys

    # This allows saving to multiple files
    note.save(tmp_path / "backup.json")
    backup_note = ExperimentNote(file_path=tmp_path / "backup.json")
    assert backup_note.get("key1") == "value1"
    assert backup_note.get("key2") == "value2"


def test_load_does_not_clear_changed_keys(tmp_path):
    """ExperimentNote should not clear changed keys tracking after load for flexibility."""
    file_path = tmp_path / "note.json"

    # Create initial file
    note1 = ExperimentNote(file_path=file_path)
    note1.put("key1", "value1")
    note1.save()

    # Load and make changes
    note2 = ExperimentNote(file_path=file_path)
    note2.put("key2", "value2")
    assert len(note2._changed_keys) == 1

    # Load again should not clear changed keys
    note2.load()
    assert len(note2._changed_keys) == 1
    assert "key2" in note2._changed_keys


def test_file_corruption_handling(tmp_path):
    """ExperimentNote should handle corrupted files gracefully during save."""
    file_path = tmp_path / "note.json"

    # Create a corrupted JSON file
    with open(file_path, "w") as f:
        f.write("invalid json content {")

    note = ExperimentNote(file_path=file_path)
    note.put("key1", "value1")
    note.save()  # Should not crash

    # Verify the file was overwritten with valid JSON
    reloaded_note = ExperimentNote(file_path=file_path)
    assert reloaded_note.get("key1") == "value1"
