import os
import json
from dataclasses import dataclass
from pprint import pprint


@dataclass
class Params:
    port_config: dict[str, dict[str, int]]
    cavity_frequency: dict[str, float]
    transmon_dressed_frequency_ge: dict[str, float]
    transmon_bare_frequency_ge: dict[str, float]
    anharmonicity: dict[str, float]
    readout_amplitude: dict[str, float]
    default_hpi_amplitude: dict[str, float]

    @classmethod
    def load(cls, path: str):
        """Loads the parameters from a JSON file."""
        if not path.endswith(".json"):
            path = path + ".json"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, path)
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return cls(**params)

    def print(self):
        """Prints the parameters in a human-readable format."""
        pprint(self.__dict__)
