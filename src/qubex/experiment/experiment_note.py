from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FILE_PATH = ".experiment_note.json"


class ExperimentNote:
    """
    A class to represent an experiment note. An experiment note is a key-value pair dictionary where the keys are
    strings and the values are JSON serializable objects. The experiment note is used to store additional information
    about an experiment that is not part of the main experiment data.
    """

    def __init__(self, file_path: Path | str | None = None):
        """
        Initializes the ExperimentNote with an empty dictionary.
        """
        if file_path is None:
            file_path = Path(FILE_PATH)
        else:
            file_path = Path(file_path)

        self._dict: dict[str, Any] = {}
        self._file_path = file_path
        self.load()

    @property
    def file_path(self) -> Path:
        """
        Returns the file path of the ExperimentNote.

        Returns
        -------
        Path
            The file path of the ExperimentNote.
        """
        return self._file_path

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
            if isinstance(value, float) and np.isnan(value):
                value = None
            self._dict[key] = value

        if old_value is not None:
            print(f"'{key}' updated: {value}")
        else:
            print(f"'{key}' added: {value}")

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

    def save(self, file_path: Path | str | None = None):
        """
        Saves the ExperimentNote to a JSON file.

        Parameters
        ----------
        file_path : Path or str, optional
            The path to save the JSON file. Defaults to the path specified in the constructor.
        """
        try:
            file_path = file_path or self._file_path
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as file:
                sorted_dict = self._sort_dict_recursively(self._dict, depth=2)
                json.dump(sorted_dict, file, indent=4)
            print(f"ExperimentNote saved to '{file_path}'.")
        except Exception as e:
            print(f"Failed to save ExperimentNote: {e}")

    def load(self, file_path: Path | str | None = None):
        """
        Loads the ExperimentNote from a JSON file.

        Parameters
        ----------
        file_path : Path or str, optional
            The path to load the JSON file. Defaults to the path specified in the constructor.
        """

        file_path = file_path or self._file_path
        file_path = Path(file_path)

        try:
            with open(file_path, "r") as file:
                self._dict = json.load(file)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON from '{file_path}'. Starting with an empty ExperimentNote."
            )
        except Exception as e:
            print(f"Failed to load ExperimentNote: {e}")

    def delete(self, file_path: Path | str | None = None):
        """
        Deletes the JSON file containing the ExperimentNote.

        Parameters
        ----------
        file_path : Path or str, optional
            The path to delete the JSON file. Defaults to the path specified in the constructor.
        """
        self.clear()

        file_path = file_path or self._file_path
        file_path = Path(file_path)

        if file_path.exists():
            file_path.unlink()
            print(f"ExperimentNote file '{file_path}' deleted.")
        else:
            print(f"ExperimentNote file '{file_path}' not found.")

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
                if isinstance(value, float) and np.isnan(value):
                    value = None
                old_dict[key] = value

    def _sort_dict_recursively(
        self,
        d: dict,
        depth: int | None = None,
        current_depth: int = 0,
    ):
        """
        Recursively sorts a dictionary by key.

        Parameters
        ----------
        d : dict
            The dictionary to sort.
        depth : int, optional
            The depth to sort to. Defaults to None.

        Returns
        -------
        dict
            The sorted dictionary.
        """
        if isinstance(d, dict):
            if depth is not None and current_depth >= depth:
                return d  # Do not sort if depth is reached
            return {
                k: self._sort_dict_recursively(v, depth, current_depth + 1)
                for k, v in sorted(d.items())
            }

        elif isinstance(d, list):
            if depth is not None and current_depth >= depth:
                return d  # Do not sort if depth is reached
            return [self._sort_dict_recursively(v, depth, current_depth + 1) for v in d]

        else:
            return d  # Return value if not a dictionary or list
