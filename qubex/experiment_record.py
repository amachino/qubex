import os
import datetime
import pickle
from dataclasses import dataclass
from typing import Final, Any


DATA_PATH: Final[str] = "./data"


@dataclass
class ExperimentRecord:
    data: Any
    name: str
    description: str = ""
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self, data_path=DATA_PATH):
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        extension = ".pkl"
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

        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Data saved to {file_path}")

    @staticmethod
    def create(data: Any, name: str, description: str = "") -> "ExperimentRecord":
        record = ExperimentRecord(data=data, name=name, description=description)
        record.save()
        return record

    @staticmethod
    def load(name: str, data_path=DATA_PATH) -> "ExperimentRecord":
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        path = os.path.join(data_path, name)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
