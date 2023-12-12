import os
import json
from pydantic import BaseModel


class Port(BaseModel):
    ssb: str
    lo: int
    nco: int
    awg0: int
    awg1: int
    awg2: int
    vatt: int


class Params(BaseModel):
    cavity_frequency: dict[str, float]
    transmon_dressed_frequency_ge: dict[str, float]
    transmon_bare_frequency_ge: dict[str, float]
    anharmonicity: dict[str, float]
    readout_amplitude: dict[str, float]
    default_hpi_amplitude: dict[str, float]


class Configs(BaseModel):
    chip_id: str
    qube_id: str
    mux_number: int
    qubits: list[str]
    control_ports: list[str]
    readout_ports: list[str]
    ports: dict[str, Port]
    params: Params

    @classmethod
    def load(cls, path: str):
        if not path.endswith(".json"):
            path = path + ".json"
        json_path = os.path.abspath(path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
