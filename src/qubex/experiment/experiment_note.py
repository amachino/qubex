import json
from typing import Any
from rich.console import Console

console = Console()


class ExperimentNote:
    """
    A class to represent an experiment note. An experiment note is a key-value pair dictionary where the keys are
    strings and the values are JSON serializable objects. The experiment note is used to store additional information
    about an experiment that is not part of the main experiment data.
    """

    def __init__(self):
        """
        Initializes the ExperimentNote with an empty dictionary.
        """
        self._dict = {}

    def put(self, key: str, value: Any):
        """
        Puts the key-value pair into the dictionary. Only allows JSON serializable values.

        Args:
            key: The key to put.
            value: The value to put.

        Raises:
            ValueError: If the value is not JSON serializable.
        """
        if not self._is_json_serializable(value):
            raise ValueError(f"Value for key '{key}' is not JSON serializable.")
        old_value = self._dict.get(key)
        self._dict[key] = value
        if old_value is not None:
            console.print(
                f"Key '{key}' updated: changed from '{old_value}' to '{value}'."
            )
        else:
            console.print(f"Key '{key}' added with value '{value}'.")

    def get(self, key: str) -> Any:
        """
        Gets the value associated with the key.

        Args:
            key: The key to get.

        Returns:
            The value associated with the key.
        """
        if key not in self._dict:
            console.print(f"Key '{key}' not found.")
            return None
        return self._dict.get(key)

    def remove(self, key: str):
        """
        Removes the key-value pair from the dictionary.

        Args:
            key: The key to remove.
        """
        removed_value = self._dict.pop(key, None)
        if removed_value is not None:
            console.print(f"Key '{key}' removed, which had value '{removed_value}'.")
        else:
            console.print(f"Key '{key}' not found, no removal performed.")

    def clear(self) -> None:
        """
        Clears the dictionary.
        """
        self._dict.clear()
        console.print("All entries have been cleared from the ExperimentNote.")

    def save(self, filename: str = "experiment_note.json"):
        """
        Saves the ExperimentNote to a JSON file.

        Args:
            filename: The name of the file to save to. Defaults to 'experiment_note.json'.
        """
        try:
            with open(filename, "w") as file:
                json.dump(self._dict, file, indent=4)
            console.print(f"ExperimentNote saved to '{filename}'.")
        except Exception as e:
            console.print(f"Failed to save ExperimentNote: {e}")

    def load(self, filename: str = "experiment_note.json"):
        """
        Loads the ExperimentNote from a JSON file.

        Args:
            filename: The name of the file to load from. Defaults to 'experiment_note.json'.
        """
        try:
            with open(filename, "r") as file:
                self._dict = json.load(file)
            console.print(f"ExperimentNote loaded from '{filename}'.")
        except FileNotFoundError:
            console.print(
                f"File '{filename}' not found. Starting with an empty ExperimentNote."
            )
        except json.JSONDecodeError:
            console.print(
                f"Error decoding JSON from '{filename}'. Starting with an empty ExperimentNote."
            )
        except Exception as e:
            console.print(f"Failed to load ExperimentNote: {e}")

    def __str__(self) -> str:
        """
        Returns the JSON representation of the ExperimentNote.

        Returns:
            The JSON representation of the ExperimentNote.
        """
        return json.dumps(self._dict)

    def __repr__(self) -> str:
        """
        Returns the JSON representation of the ExperimentNote.

        Returns:
            The JSON representation of the ExperimentNote.
        """
        return json.dumps(self._dict, indent=4)

    def _is_json_serializable(self, value: Any) -> bool:
        """
        Checks if a value is JSON serializable.

        Args:
            value: The value to check.

        Returns:
            True if the value is JSON serializable, False otherwise.
        """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
