from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Generic, TypeVar

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()


DEFAULT_DATA_DIR: Final[str] = ".rawdata"

T = TypeVar("T")


@dataclass
class MeasurementRecord(Generic[T]):
    """
    A dataclass to store the results of a measurement.

    Attributes
    ----------
    data : T
        The data to be saved in the record.
    created_at : str
        The date and time when the record was created.
    """

    data: T
    file_name: str = ""
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(
        self,
        data_dir: Path | str | None = None,
    ):
        """
        Save the experiment record to a pickle file.

        Parameters
        ----------
        datadata_dir: Path | str | None
            Path to the directory where the record will be saved.
        """
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        extension = ".json"
        current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"{current_date}{extension}"
        file_path = os.path.join(
            data_dir,
            file_name,
        )
        self.file_name = file_name
        with open(file_path, "w") as f:
            encoded = jsonpickle.encode(self, unpicklable=True)
            f.write(encoded)  # type: ignore

    @staticmethod
    def create(
        data: Any,
        data_dir: Path | str | None = None,
    ) -> MeasurementRecord:
        """
        Create and save a new experiment record.

        Parameters
        ----------
        data : Any
            Data to be saved in the record.
        data_dir : Path | str | None
            Path to the directory where the record will be saved.

        Returns
        -------
        MeasurementRecord
            The newly created and saved MeasurementRecord instance.
        """
        record = MeasurementRecord(data=data)
        record.save(data_dir=data_dir)
        return record

    @staticmethod
    def load(
        name: str,
        data_dir: Path | str | None = None,
    ) -> MeasurementRecord:
        """
        Load an experiment record from a file.

        Parameters
        ----------
        name : str
            Name of the experiment record to load.
        data_dir : Path | str | None
            Path to the directory where the record is saved.

        Returns
        -------
        MeasurementRecord
            The loaded MeasurementRecord instance.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        if not name.endswith(".json"):
            name = name + ".json"
        path = os.path.join(data_dir, name)
        with open(path, "r") as f:
            data = jsonpickle.decode(f.read())
            if not isinstance(data, MeasurementRecord):
                raise TypeError(f"Expected MeasurementRecord, got {type(data)}")
        return data
