from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FILE_PATH = ".experiment_note.json"


class ExperimentNote:
    """
    A class to represent an experiment note. An experiment note is a key-value pair dictionary where the keys are
    strings and the values are JSON serializable objects. The experiment note is used to store additional information
    about an experiment that is not part of the main experiment data.
    """

    def __init__(self, file_path: str = FILE_PATH):
        """
        Initializes the ExperimentNote with an empty dictionary.
        """
        self._dict: dict[str, Any] = {}
        self._file_path = file_path
        self.load()

    def put(self, key: str, value: Any):
        """
        Puts the key-value pair into the dictionary. Only allows JSON serializable values.

        Parameters
        ----------
        key : str
            The key to put.
        value : Any
            The value to put.

        Raises
        ------
        ValueError
            If the value is not JSON serializable.
        """
        if not self._is_json_serializable(value):
            raise ValueError(f"Value for key '{key}' is not JSON serializable.")

        old_value = self._dict.get(key)

        if isinstance(old_value, dict) and isinstance(value, dict):
            self._update_dict_recursively(old_value, value)
        else:
            self._dict[key] = value

        if old_value is not None:
            print(
                f"Key '{key}' updated: changed from '{old_value}' to '{self._dict[key]}'."
            )
        else:
            print(f"Key '{key}' added with value '{value}'.")

    def get(self, key: str) -> Any:
        """
        Gets the value associated with the key.

        Parameters
        ----------
        key : str
            The key to get.

        Returns
        -------
        Any
            The value associated with the key, or None if the key is not found.
        """
        if key not in self._dict:
            return None
        return self._dict.get(key)

    def remove(self, key: str):
        """
        Removes the key-value pair from the dictionary.

        Parameters
        ----------
        key : str
            The key to remove.
        """
        removed_value = self._dict.pop(key, None)
        if removed_value is not None:
            print(f"Key '{key}' removed, which had value '{removed_value}'.")
        else:
            print(f"Key '{key}' not found, no removal performed.")

    def clear(self) -> None:
        """
        Clears the dictionary.
        """
        self._dict.clear()
        print("All entries have been cleared from the ExperimentNote.")

    def save(self, filename: str | None = None):
        """
        Saves the ExperimentNote to a JSON file.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save to. Defaults to 'experiment_note.json'.
        """
        try:
            filename = filename or self._file_path
            with open(filename, "w") as file:
                json.dump(self._dict, file, indent=4)
            print(f"ExperimentNote saved to '{filename}'.")
        except Exception as e:
            print(f"Failed to save ExperimentNote: {e}")

    def load(self, filename: str | None = None):
        """
        Loads the ExperimentNote from a JSON file.

        Parameters
        ----------
        filename : str, optional
            The name of the file to load from. Defaults to 'experiment_note.json'.
        """

        filename = filename or self._file_path
        file_path = Path(filename)

        if not file_path.exists():
            with open(filename, "w") as file:
                json.dump({}, file)
        try:
            with open(filename, "r") as file:
                self._dict = json.load(file)
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON from '{filename}'. Starting with an empty ExperimentNote."
            )
        except Exception as e:
            print(f"Failed to load ExperimentNote: {e}")

    def delete(self, filename: str | None = None):
        """
        Deletes the JSON file containing the ExperimentNote.

        Parameters
        ----------
        filename : str, optional
            The name of the file to delete. Defaults to 'experiment_note.json'.
        """
        filename = filename or self._file_path
        file_path = Path(filename)

        if file_path.exists():
            file_path.unlink()
            print(f"ExperimentNote file '{filename}' deleted.")
        else:
            print(f"ExperimentNote file '{filename}' not found.")

    def __str__(self) -> str:
        """
        Returns the JSON representation of the ExperimentNote.

        Returns
        -------
        str
            The JSON representation of the ExperimentNote.
        """
        return json.dumps(self._dict)

    def __repr__(self) -> str:
        """
        Returns the JSON representation of the ExperimentNote.

        Returns
        -------
        str
            The JSON representation of the ExperimentNote.
        """
        return json.dumps(self._dict, indent=4)

    def _is_json_serializable(self, value: Any) -> bool:
        """
        Checks if a value is JSON serializable.

        Parameters
        ----------
        value : Any
            The value to check.

        Returns
        -------
        bool
            True if the value is JSON serializable, False otherwise.
        """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    def _update_dict_recursively(self, old_dict: dict, new_dict: dict):
        """
        Recursively updates old_dict with new_dict.

        Parameters
        ----------
        old_dict : dict
            The dictionary to update.
        new_dict : dict
            The dictionary with updated values.
        """
        for key, value in new_dict.items():
            if (
                isinstance(value, dict)
                and key in old_dict
                and isinstance(old_dict[key], dict)
            ):
                self._update_dict_recursively(old_dict[key], value)
            else:
                old_dict[key] = value
