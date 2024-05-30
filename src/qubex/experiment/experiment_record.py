import datetime
import os
from dataclasses import dataclass
from typing import Any, Final, TypeVar, Generic

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()


DEFAULT_DATA_DIR: Final[str] = "data"

T = TypeVar("T")


@dataclass
class ExperimentRecord(Generic[T]):
    """
    A dataclass to store the results of an experiment.

    Attributes
    ----------
    data : T
        The data to be saved in the record.
    name : str
        The name of the experiment.
    description : str, optional
        A description of the experiment.
    created_at : str
        The date and time when the record was created.

    Methods
    -------
    save(data_path=DATA_PATH)
        Saves the experiment record to a file.
    create(data, name, description="")
        Creates and saves an ExperimentRecord instance.
    load(name, data_path=DATA_PATH)
        Loads an ExperimentRecord instance from a file.
    """

    data: T
    name: str
    description: str = ""
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self, data_path=DEFAULT_DATA_DIR):
        """
        Save the experiment record to a pickle file.

        Parameters
        ----------
        data_path : str, optional
            Path to the directory where the record will be saved.

        Notes
        -----
        The method creates a unique filename for the record based on the
        current date and the experiment's name to avoid overwriting.
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        extension = ".json"
        counter = 1
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(
            data_path,
            f"{current_date}_{self.name}_{counter}{extension}",
        )

        while os.path.exists(file_path):
            file_path = os.path.join(
                data_path,
                f"{current_date}_{self.name}_{counter}{extension}",
            )
            counter += 1

        with open(file_path, "w") as f:
            encoded = jsonpickle.encode(self, unpicklable=True)
            f.write(encoded)  # type: ignore
        print(f"Data saved to {file_path}")

    @staticmethod
    def create(data: Any, name: str, description: str = "") -> "ExperimentRecord":
        """
        Create and save a new experiment record.

        Parameters
        ----------
        data : Any
            Data to be saved in the record.
        name : str
            Name of the experiment.
        description : str, optional
            Description of the experiment.

        Returns
        -------
        ExperimentRecord
            The newly created and saved ExperimentRecord instance.
        """
        record = ExperimentRecord(data=data, name=name, description=description)
        record.save()
        return record

    @staticmethod
    def load(name: str, data_path=DEFAULT_DATA_DIR) -> "ExperimentRecord":
        """
        Load an experiment record from a file.

        Parameters
        ----------
        name : str
            Name of the experiment record to load.
        data_path : str, optional
            Path to the directory where the record is saved.

        Returns
        -------
        ExperimentRecord
            The loaded ExperimentRecord instance.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if not name.endswith(".json"):
            name = name + ".json"
        path = os.path.join(data_path, name)
        with open(path, "r") as f:
            data = jsonpickle.decode(f.read())
            if not isinstance(data, ExperimentRecord):
                raise TypeError(f"Expected ExperimentRecord, got {type(data)}")
        return data
